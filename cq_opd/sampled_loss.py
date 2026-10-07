"""Baseline sampled-k1 policy gradient with chunk-local frozen-head VJPs.

``coefficient`` always means positive-signed, detached clamp(logp - logq,
-10, 10), NOT the advantage. The baseline advantage is its negative. Freeze
both old_logp and this coefficient at the native unperturbed policy before
scoring either FD side. All reductions here are raw sums; normalize once,
externally, by the total selected tokens across the logical batch.
"""
import math
from numbers import Integral

import torch

from .blocks import shifted_valid_mask
from .losses import prediction_chunks, require_finite, working_float


def _same_shape(reference, *others):
    if any(x.shape != reference.shape for x in others):
        raise ValueError("sampled token tensors must have equal shapes")


def _check_chunk_inputs(hidden, selected, chunk_size):
    if (not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool)
            or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    if (hidden.ndim != 3 or selected.ndim != 2
            or tuple(hidden.shape[:2]) != (selected.shape[0], selected.shape[1] + 1)):
        raise ValueError("expected hidden [B,L,D], mask [B,L-1]")
    if hidden.device != selected.device:
        raise ValueError("hidden and selected must share a device")


def _sampled_logp(rows, next_ids, project):
    # At most chunk_size rows of logits exist at once. Do not materialize a
    # full log-softmax or any Teacher full-vocabulary probability distribution.
    logits = working_float(project(rows))
    if logits.ndim != 2 or logits.shape[0] != next_ids.numel():
        raise ValueError("project must return [tokens,vocabulary] logits")
    require_finite(logits, "sampled logits")
    result = logits.gather(-1, next_ids[:, None]).squeeze(-1) - logits.logsumexp(-1)
    require_finite(result, "sampled log probabilities")
    return result


@torch.no_grad()
def token_log_probs(adapter, batch, chunk_size=64):
    """Detached sampled next-token logp [B,L-1], zero outside valid responses.

    EOS is retained when its response/attention masks are true; prompt and
    padding positions are excluded by masks, never by token ID. This is also
    the Teacher scoring path and never builds a Teacher autograd graph.
    """
    valid = shifted_valid_mask(batch["response_mask"], batch["attention_mask"])
    next_ids = batch["input_ids"][:, 1:]
    _same_shape(valid, next_ids)
    hidden = adapter.last_hidden(batch, with_grad=False, use_cache=False).detach()
    _check_chunk_inputs(hidden, valid, chunk_size)
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    result = hidden.new_zeros(valid.shape, dtype=dtype)
    for b, t in prediction_chunks(hidden, valid, chunk_size):
        result[b, t] = _sampled_logp(hidden[b, t], next_ids[b, t], adapter.project)
    require_finite(result, "sampled log probabilities")
    return result.detach()


def sampled_k1_coefficient(current_logp, teacher_logp):
    """Freeze the baseline k1 coefficient at the unperturbed current policy.

    Clamp the *difference* only; sampled student/Teacher logp are never
    lower-clamped at -10 (that baseline clamp belongs to its top-k path).
    """
    _same_shape(current_logp, teacher_logp)
    lp = working_float(current_logp.detach())
    lq = teacher_logp.detach().to(lp)
    require_finite(lp, "current log probabilities")
    require_finite(lq, "teacher log probabilities")
    raw_k1 = lp - lq
    require_finite(raw_k1, "sampled k1")
    return raw_k1.clamp(-10.0, 10.0).detach()


def sampled_pg_tokens(logp, old_logp, coefficient):
    """Unreduced vanilla PPO token loss, with detached advantage=-coefficient.

    Exact baseline constants: log-ratio clamp [-20,20], asymmetric PPO
    bounds [0.8,1.2], dual-clip 3 for negative advantages. Only logp receives
    derivatives. Callers mask/select positions before reduction.
    """
    _same_shape(logp, old_logp, coefficient)
    lp = working_float(logp)
    old = old_logp.detach().to(lp)
    k1 = coefficient.detach().to(lp)
    require_finite(lp, "current log probabilities")
    require_finite(old, "old log probabilities")
    require_finite(k1, "sampled coefficient")
    log_ratio = lp - old
    require_finite(log_ratio, "sampled log ratio")
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    advantages = -k1
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(0.8, 1.2)
    upper = torch.maximum(unclipped, clipped)
    dual = torch.minimum(-advantages * 3.0, upper)
    result = torch.where(advantages < 0, dual, upper)
    require_finite(result, "sampled PG loss")
    return result


def sampled_pg_utility(good_logp, bad_logp, old_logp, coefficient, delta):
    """Signed token FD utility (loss_bad-loss_good)/(2*delta).

    Reuse the SAME unperturbed old_logp/coefficient for both offset scores.
    Do not recompute k1 from good/bad logp: that would change the derivative.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    good = sampled_pg_tokens(good_logp, old_logp, coefficient)
    bad = sampled_pg_tokens(bad_logp, old_logp, coefficient)
    result = (bad - good) / (2 * delta)
    require_finite(result, "sampled FD utility")
    return result


def chunked_sampled_hidden_grad(hidden, next_ids, selected, project,
                                old_logp, coefficient, chunk_size=64):
    """Return (detached raw-SUM hidden VJP, detached loss sum, token count).

    ``hidden`` is [B,L,D]; IDs/mask/old/coefficient are [B,L-1]. The head is
    frozen by the adapter; autograd.grad requests only local hidden leaves,
    never accumulates head or source-hidden .grad. Neither the final position
    nor unselected positions are projected. Empty selection projects nothing.
    """
    _check_chunk_inputs(hidden, selected, chunk_size)
    _same_shape(selected, next_ids, old_logp, coefficient)
    grad = torch.zeros_like(hidden)
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    total = hidden.new_zeros((), dtype=dtype)
    for b, t in prediction_chunks(hidden, selected, chunk_size):
        with torch.enable_grad():
            leaf = hidden[b, t].detach().requires_grad_(True)
            lp = _sampled_logp(leaf, next_ids[b, t], project)
            loss = sampled_pg_tokens(lp, old_logp[b, t], coefficient[b, t]).sum()
            require_finite(loss, "sampled loss sum")
            dh = torch.autograd.grad(loss, leaf, create_graph=False)[0]
        require_finite(dh, "sampled hidden gradient")
        grad[b, t] = dh.detach()
        total += loss.detach()
    require_finite(total, "sampled loss sum")
    return grad.detach(), total.detach(), int(selected.bool().sum())

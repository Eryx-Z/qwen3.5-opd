"""Symmetric offsets always copy a snapshot and restore it, even on failure."""
import math
import torch
from .losses import mixed_utility_from_logits, log_probs, prediction_chunks, require_finite


def score_symmetric_offsets(named_parameters, direction, delta, score_fn):
    """score_fn() performs a fresh, cache-free no-grad forward, returns its score/hidden.

    Returns (good, bad). Caller owns cache isolation and freezes target metadata.
    No optimizer or .grad is touched; never call with a pending main-loss graph.
    """
    named_parameters = list(named_parameters)
    tensors = direction.tensors if hasattr(direction, "tensors") else list(direction)
    if hasattr(direction, "valid") and not direction.valid:
        raise ValueError("invalid direction")
    if not math.isfinite(delta) or delta <= 0 or len(tensors) != len(named_parameters):
        raise ValueError("invalid delta or direction size")
    for (name, p), d in zip(named_parameters, tensors):
        if d.shape != p.shape:
            raise ValueError(f"direction shape mismatch {name}")
    # Aggregate device scalars before the single host decision. Checking each
    # of 372 LoRA tensors separately forced a synchronization per tensor.
    if tensors and not torch.stack([torch.isfinite(d).all() for d in tensors]).all():
        for (name,_),d in zip(named_parameters,tensors):
            require_finite(d,f"direction {name}")  # localize only on failure
    saved = [p.detach().clone() for _, p in named_parameters]
    try:
        outputs = []
        with torch.no_grad():
            for sign in (1, -1):
                for (_, p), original, d in zip(named_parameters, saved, tensors):
                    p.copy_(original + sign * delta * d)
                outputs.append(score_fn())
        return tuple(outputs)
    finally:
        with torch.no_grad():
            for (_, p), original in zip(named_parameters, saved):
                p.copy_(original)


@torch.no_grad()
def chunked_token_utility(good_hidden, bad_hidden, teacher_hidden, alpha, valid, delta, student_project, teacher_project, chunk_size=64):
    utility = torch.zeros_like(alpha)
    for b, t in prediction_chunks(good_hidden, valid, chunk_size):
        lq = log_probs(teacher_project(teacher_hidden[b, t]))
        utility[b, t] = mixed_utility_from_logits(student_project(good_hidden[b, t]), student_project(bad_hidden[b, t]), lq, alpha[b, t], delta)
    return utility


def parameter_norm(named_parameters):
    values = [p.detach().double().square().sum() for _, p in named_parameters]
    if not values:
        raise ValueError("empty parameters")
    norm = torch.stack(values).sum().sqrt()
    require_finite(norm, "parameter norm")
    return float(norm)


def delta_from_rho(named_parameters, rho):
    norm = parameter_norm(named_parameters)
    if norm == 0 or not math.isfinite(rho) or rho <= 0:
        raise ValueError("nonzero parameter norm and positive rho required")
    return rho * norm

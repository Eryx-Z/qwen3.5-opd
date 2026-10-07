"""LOO task gradients are returned as tensors, never written into .grad."""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from .losses import working_float, require_finite, chunked_hard_label_hidden_grad, log_probs


@dataclass
class DirectionResult:
    valid: bool
    tensors: list
    source_step: int
    reason: str = ""
    grad_norm: float = 0.0
    unused_names: tuple = ()


def _initial_direction_checks(named_parameters, direction):
    """Metadata checks on host, tensor checks deferred to one scalar decision."""
    if (not isinstance(direction, DirectionResult) or not direction.valid
            or direction.source_step != 0 or not named_parameters
            or len(named_parameters) != len(direction.tensors)
            or not math.isfinite(direction.grad_norm)):
        return None
    checks, nonzero = [], []
    for (_, parameter), tensor in zip(named_parameters, direction.tensors):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype or tensor.device != parameter.device):
            return None
        checks.extend((torch.isfinite(parameter).all(), torch.isfinite(tensor).all()))
        nonzero.append(tensor.ne(0).any())
    device = checks[0].device
    checks.append(torch.stack([value.to(device) for value in nonzero]).any())
    return checks


def _all_initial_checks(checks):
    device = checks[0].device
    return bool(torch.stack([value.to(device) for value in checks]).all())


def capture_initial_probe(named_parameters, d, stats, key, rng_before, rng_after):
    """Capture an ephemeral age-zero probe; never a checkpoint or stale cache.

    The caller owns the one-shot lifetime and must never attempt reuse after an
    optimizer update or on resume. Frozen state/protocol identity belongs in key.
    Invalid directions or nonfinite master values are refused before capture.
    """
    named_parameters = list(named_parameters)
    with torch.no_grad():
        checks = _initial_direction_checks(named_parameters, d)
        if checks is None:
            raise ValueError("initial probe requires a valid, matching source_step=0 direction")
        if not _all_initial_checks(checks):
            raise ValueError("initial probe requires finite parameters and a finite nonzero direction")
        parameters = [(name, parameter.detach().clone()) for name, parameter in named_parameters]
        direction = DirectionResult(d.valid, [tensor.detach().clone() for tensor in d.tensors],
                                    d.source_step, d.reason, d.grad_norm, deepcopy(d.unused_names))
    return dict(parameters=parameters, direction=direction, stats=deepcopy(stats),
                key=deepcopy(key), rng_before=deepcopy(rng_before), rng_after=deepcopy(rng_after))


def consume_initial_probe(payload, named_parameters, key, rng):
    """Reuse only an exact initial-value/protocol/RNG match, otherwise do nothing.

    Tensor versions alone cannot prove equality: symmetric scoring offsets bump
    versions while restoring values. All value/finite checks share one scalar
    decision. On a hit RNG advances exactly as if the probe had just been drawn.
    The caller must clear the payload after this initial attempt, hit or miss.
    """
    if (not payload or payload.get("key") != key
            or rng.getstate() != payload.get("rng_before")):
        return None
    named_parameters = list(named_parameters)
    saved = payload.get("parameters", [])
    direction = payload.get("direction")
    if len(named_parameters) != len(saved):
        return None
    with torch.no_grad():
        checks = _initial_direction_checks(named_parameters, direction)
        if checks is None:
            return None
        for (name, parameter), (saved_name, snapshot) in zip(named_parameters, saved):
            if (name != saved_name or parameter.shape != snapshot.shape
                    or parameter.dtype != snapshot.dtype or parameter.device != snapshot.device):
                return None
            checks.append(parameter.eq(snapshot).all())
        if not _all_initial_checks(checks):
            return None
    stats = deepcopy(payload["stats"])
    stats["initial_probe_original_generation_time"] = stats["probe_generation_time"]
    stats["initial_probe_original_gradient_time"] = stats["probe_gradient_time"]
    stats["probe_generation_time"] = 0.0
    stats["probe_gradient_time"] = 0.0
    stats["initial_probe_reused"] = True
    rng.setstate(payload["rng_after"])
    return direction, stats


def leave_one_out_advantages(rewards):
    rewards = working_float(torch.as_tensor(rewards).detach())
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must be [M,K], K>=2")
    if not ((rewards == 0) | (rewards == 1)).all():
        raise ValueError("rewards must be binary")
    return rewards - (rewards.sum(-1, keepdim=True) - rewards) / (rewards.shape[1] - 1)


def direction_from_gradients(named_parameters, gradients, source_step=0, epsilon=1e-12, unused_names=()):
    named_parameters, gradients = list(named_parameters), list(gradients)
    if len(named_parameters) != len(gradients) or not named_parameters:
        raise ValueError("nonempty matching parameter/gradient lists required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive epsilon required")
    clean = []
    for (name, p), g in zip(named_parameters, gradients):
        g = torch.zeros_like(p) if g is None else g.detach()
        if g.shape != p.shape:
            raise ValueError(f"gradient shape mismatch: {name}")
        clean.append(g)
    if not torch.stack([torch.isfinite(g).all() for g in clean]).all():
        for (name,_),g in zip(named_parameters,clean):
            require_finite(g,f"probe gradient {name}")
    # Norm over tensor lists; FP64 reference kept and FP32 LoRA norms accumulated in FP64.
    norm = torch.stack([g.double().square().sum() for g in clean]).sum().sqrt()
    require_finite(norm, "probe gradient norm")
    n = float(norm)
    if n == 0:
        return DirectionResult(False, [torch.zeros_like(g) for g in clean], source_step, "no_probe_signal", n, tuple(unused_names))
    # Normalize before casting: a finite FP64 norm can overflow the gradient dtype.
    direction = [(-g.double() / (norm + epsilon)).to(g) for g in clean]
    if not torch.stack([torch.isfinite(g).all() for g in direction]).all():
        raise FloatingPointError('nonfinite probe direction')
    direction_norm = torch.stack([g.double().square().sum() for g in direction]).sum().sqrt()
    require_finite(direction_norm, "probe direction norm")
    if float(direction_norm) == 0:
        raise FloatingPointError("probe direction vanished after normalization and cast")
    return DirectionResult(True, direction, source_step, "", n, tuple(unused_names))


def probe_loss(student_logits, next_ids, valid, advantages):
    logp = log_probs(student_logits).gather(-1, next_ids[..., None]).squeeze(-1)
    return -(logp * valid * advantages.detach().reshape(-1, 1)).sum() / advantages.numel()


def chunked_probe_gradients(hidden, next_ids, valid, project, named_parameters, weight=1.0, chunk_size=64):
    dh, _ = chunked_hard_label_hidden_grad(hidden, next_ids, valid, project, weight, chunk_size)
    params = [p for _, p in named_parameters]
    return list(torch.autograd.grad(hidden, params, grad_outputs=dh, allow_unused=True, create_graph=False))


def compute_probe_direction(named_parameters, advantages, loss_for_response, source_step=0, epsilon=1e-12):
    """Callback(i,k): unweighted negative SUM logprob scalar OR unweighted gradient list.

    Only nonzero-advantage trajectories are evaluated. Weight A/(M*K) is applied
    here, so chunked_probe_gradients callbacks should use weight=1.
    """
    named_parameters = list(named_parameters)
    a = advantages.detach()
    if a.ndim != 2 or not a.numel():
        raise ValueError("advantages must be nonempty [M,K]")
    require_finite(a, "advantages")
    sums = [torch.zeros_like(p) for _, p in named_parameters]
    used = [False] * len(sums)
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            if float(a[i, k]) == 0:
                continue
            value = loss_for_response(i, k)
            grads = torch.autograd.grad(value, [p for _, p in named_parameters], allow_unused=True, create_graph=False) if isinstance(value, torch.Tensor) else list(value)
            if len(grads) != len(sums):
                raise ValueError("callback gradient count mismatch")
            for j, g in enumerate(grads):
                if g is not None:
                    require_finite(g, "probe gradient")
                    used[j] = True
                    sums[j].add_(g.detach() * (a[i, k].to(g) / a.numel()))
    unused = [name for (name, _), yes in zip(named_parameters, used) if not yes]
    return direction_from_gradients(named_parameters, sums, source_step, epsilon, unused)

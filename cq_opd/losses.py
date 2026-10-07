"""Full-vocabulary mixed KL and frozen-head, chunk-local VJPs."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


def working_float(x):
    return x if x.dtype == torch.float64 else x.float()


def log_probs(logits):
    return F.log_softmax(working_float(logits), dim=-1)


def require_finite(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"nonfinite {name}")


def kl_components(student_logits, teacher_log_probs):
    lp = log_probs(student_logits)
    lq = working_float(teacher_log_probs.detach())
    require_finite(lp, "student log probabilities")
    require_finite(lq, "teacher log probabilities")
    q, p = lq.exp(), lp.exp()
    return (q * (lq - lp)).sum(-1), (p * (lp - lq)).sum(-1)


def _mix_kl_components(forward, reverse, alpha):
    a = alpha.detach().to(forward)
    if not torch.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("alpha must be finite in [0,1]")
    return a * forward + (1 - a) * reverse


def mixed_kl(student_logits, teacher_log_probs, alpha):
    forward, reverse = kl_components(student_logits, teacher_log_probs)
    return _mix_kl_components(forward, reverse, alpha)


def soft_ce(student_logits, teacher_log_probs):
    return -(teacher_log_probs.detach().exp() * log_probs(student_logits)).sum(-1)


def mixed_utility_from_logits(student_good, student_bad, teacher_log_probs, alpha, delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    lg, lb = log_probs(student_good), log_probs(student_bad)
    lq = working_float(teacher_log_probs.detach())
    a = alpha.detach().to(lg)
    if not torch.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("alpha must be finite in [0,1]")
    df = (lq.exp() * (lg - lb)).sum(-1)
    dr = (lb.exp() * (lb - lq) - lg.exp() * (lg - lq)).sum(-1)
    result = (a * df + (1 - a) * dr) / (2 * delta)
    require_finite(result, "utility")
    return result


def calibrate_entropy(entropies):
    values = working_float(torch.as_tensor(entropies).detach()).reshape(-1)
    if not values.numel():
        raise ValueError("no valid entropy positions")
    require_finite(values, "entropy")
    quartiles = torch.quantile(values, values.new_tensor([.25, .5, .75]))
    scale = max(float(quartiles[2] - quartiles[0]), .1)
    return {"tau": float(quartiles[1]), "scale": scale, "k": 1 / scale}


def entropy_alpha(teacher_log_probs, calibration):
    lq = working_float(teacher_log_probs.detach())
    require_finite(lq, "teacher log probabilities")
    entropy = -(lq.exp() * lq).sum(-1)
    mode = calibration.get("mode", "teacher_entropy")
    if mode == "constant":
        value = float(calibration["constant_alpha"])
        if not 0 <= value <= 1:
            raise ValueError("constant_alpha outside [0,1]")
        alpha = torch.full_like(entropy, value)
    elif mode == "teacher_entropy":
        if "k" in calibration:
            k = float(calibration["k"])
        else:
            scale = float(calibration["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid entropy scale")
            k = 1 / scale
        tau = float(calibration["tau"])
        if not math.isfinite(k) or k <= 0 or not math.isfinite(tau):
            raise ValueError("invalid entropy calibration")
        alpha = torch.sigmoid(k * (entropy - tau))
    else:
        raise ValueError(f"unknown mixing mode {mode}")
    return entropy.detach(), alpha.detach()


@dataclass
class TargetMeta:
    entropy: torch.Tensor
    alpha: torch.Tensor


def prediction_chunks(hidden, selected, chunk_size):
    if chunk_size <= 0 or selected.ndim != 2 or tuple(hidden.shape[:2]) != (selected.shape[0], selected.shape[1] + 1):
        raise ValueError("expected hidden [B,L,D], mask [B,L-1], positive chunk size")
    indices = selected.bool().nonzero(as_tuple=False)
    for rows in indices.split(chunk_size):
        if rows.numel():
            yield rows[:, 0], rows[:, 1]


@torch.no_grad()
def prepare_entropy_alpha(teacher_hidden, valid_mask, calibration, teacher_project, chunk_size=64):
    dtype = torch.float64 if teacher_hidden.dtype == torch.float64 else torch.float32
    entropy = torch.zeros(valid_mask.shape, dtype=dtype, device=teacher_hidden.device)
    alpha = torch.zeros_like(entropy)
    for b, t in prediction_chunks(teacher_hidden, valid_mask, chunk_size):
        e, a = entropy_alpha(log_probs(teacher_project(teacher_hidden[b, t])), calibration)
        entropy[b, t], alpha[b, t] = e, a
    return TargetMeta(entropy, alpha)


def chunked_mixed_kl_hidden_grad(student_hidden, teacher_hidden, alpha, selected, student_project, teacher_project, chunk_size=64, *, return_components=False):
    """Return raw SUM hidden gradient, loss and count; caller normalizes once.

    If requested, append detached forward/reverse KL sums on the hidden device,
    using the same projected logits as the gradient (no additional head pass).
    """
    grad = torch.zeros_like(student_hidden)
    total = torch.zeros((), dtype=torch.float64 if student_hidden.dtype == torch.float64 else torch.float32, device=student_hidden.device)
    if return_components:
        forward_sum, reverse_sum = total.new_zeros(()), total.new_zeros(())
    for b, t in prediction_chunks(student_hidden, selected, chunk_size):
        with torch.no_grad():
            lq = log_probs(teacher_project(teacher_hidden[b, t])).detach()
        with torch.enable_grad():
            leaf = student_hidden[b, t].detach().requires_grad_(True)
            forward, reverse = kl_components(student_project(leaf), lq)
            loss = _mix_kl_components(forward, reverse, alpha[b, t]).sum()
            dh = torch.autograd.grad(loss, leaf, create_graph=False)[0]
        require_finite(dh, "hidden gradient")
        grad[b, t] = dh.detach()
        total += loss.detach()
        if return_components:
            forward_sum += forward.detach().sum()
            reverse_sum += reverse.detach().sum()
    result = grad, total, int(selected.bool().sum())
    return (*result, forward_sum, reverse_sum) if return_components else result


def chunked_hard_label_hidden_grad(hidden, next_ids, valid, project, weight=1.0, chunk_size=64):
    """Hard-label negative summed log likelihood, no length normalization."""
    grad = torch.zeros_like(hidden)
    total = hidden.new_zeros((), dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32)
    for b, t in prediction_chunks(hidden, valid, chunk_size):
        with torch.enable_grad():
            leaf = hidden[b, t].detach().requires_grad_(True)
            loss = -log_probs(project(leaf)).gather(-1, next_ids[b, t, None]).sum() * torch.as_tensor(weight, device=hidden.device).detach()
            dh = torch.autograd.grad(loss, leaf)[0]
        require_finite(dh, "probe hidden gradient")
        grad[b, t] = dh.detach()
        total += loss.detach()
    return grad, total


def divide_accumulated_gradients_by(parameters, total_selected_tokens):
    if total_selected_tokens <= 0:
        raise ValueError("cannot normalize zero selected tokens")
    with torch.no_grad():
        for p in parameters:
            if p.grad is not None:
                require_finite(p.grad, "accumulated gradient")
                p.grad.div_(total_selected_tokens)

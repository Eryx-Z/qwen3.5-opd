"""Fixed per-response budgets, stable signed ranks; no positive filtering."""
import math
import torch
from .blocks import block_means
from .losses import require_finite


def select_blocks(token_scores, blocks, keep_ratio):
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0,1]")
    scores = block_means(token_scores, blocks)
    require_finite(scores, "block scores")
    selected = torch.zeros_like(token_scores, dtype=torch.bool)
    groups = {}
    for i, block in enumerate(blocks):
        groups.setdefault(block.response_index, []).append(i)
    for indices in groups.values():
        order = torch.argsort(scores[indices], descending=True, stable=True).tolist()
        count = max(1, math.ceil(keep_ratio * len(indices)))
        for j in order[:count]:
            block = blocks[indices[j]]
            selected[block.response_index, list(block.positions)] = True
    return selected


def select_by_method(method, valid, blocks, keep_ratio=.2, token_scores=None, generator=None):
    if method == "full" or keep_ratio == 1:
        return valid.bool().clone()
    if method == "random":
        # Constant score per block ensures each block (including short tails) has equal chance.
        scores = torch.zeros(valid.shape, dtype=torch.float32, device=valid.device)
        draws = torch.rand(len(blocks), generator=generator, device=valid.device)
        for block, draw in zip(blocks, draws):
            scores[block.response_index, list(block.positions)] = draw
    elif method in ("cq", "kl", "teachability"):
        if token_scores is None:
            raise ValueError(f"{method} requires its own token selection metric")
        scores = token_scores
    else:
        raise ValueError(f"unknown selector {method}")
    return select_blocks(scores, blocks, keep_ratio) & valid.bool()


def teachability_scores(student_logits, teacher_log_probs, top_k):
    from .losses import kl_components
    if not 1 <= top_k <= student_logits.shape[-1]:
        raise ValueError("invalid teachability top_k")
    forward, _ = kl_components(student_logits, teacher_log_probs)
    ids = student_logits.detach().topk(top_k, dim=-1).indices
    coverage = teacher_log_probs.detach().exp().gather(-1, ids).sum(-1)
    return forward * coverage

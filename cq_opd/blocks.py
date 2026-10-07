"""All positions refer to shifted prediction space [B,L-1]."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Block:
    response_index: int
    positions: tuple


def shifted_valid_mask(response_mask, attention_mask):
    if response_mask.shape != attention_mask.shape or response_mask.ndim != 2:
        raise ValueError("masks must have equal [B,L] shapes")
    return response_mask[:, 1:].bool() & attention_mask[:, 1:].bool()


def make_blocks(response_mask_shifted, block_len=16):
    if block_len <= 0 or response_mask_shifted.ndim != 2:
        raise ValueError("positive block_len and [B,L-1] mask required")
    blocks = []
    for b, row in enumerate(response_mask_shifted):
        positions = row.nonzero(as_tuple=True)[0].tolist()
        blocks.extend(Block(b, tuple(positions[i:i + block_len])) for i in range(0, len(positions), block_len))
    return blocks


def block_means(token_scores, blocks):
    return torch.stack([token_scores[b.response_index, list(b.positions)].mean() for b in blocks]) if blocks else token_scores.new_empty(0)

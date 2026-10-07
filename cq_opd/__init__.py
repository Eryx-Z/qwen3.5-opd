"""CQ-OPD V1.1 mathematical core; model/cache/optimizer lifecycle belongs to trainer."""
from .blocks import Block, shifted_valid_mask, make_blocks
from .losses import (TargetMeta, mixed_kl, kl_components, mixed_utility_from_logits,
                     calibrate_entropy, entropy_alpha, prepare_entropy_alpha,
                     chunked_mixed_kl_hidden_grad, divide_accumulated_gradients_by)
from .probe import (DirectionResult, leave_one_out_advantages, compute_probe_direction,
                    direction_from_gradients, chunked_probe_gradients)
from .selectors import select_blocks, select_by_method
from .utility import score_symmetric_offsets, chunked_token_utility, delta_from_rho

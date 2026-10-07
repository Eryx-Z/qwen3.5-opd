"""CPU proof of exact baseline sampled-k1 PG, frozen FD and chunked VJPs."""
import math

import pytest
import torch
import torch.nn.functional as F

from cq_opd.sampled_loss import (
    chunked_sampled_hidden_grad, sampled_k1_coefficient,
    sampled_pg_tokens, sampled_pg_utility, token_log_probs,
)

DT = torch.float64


def baseline_tokens(logp, old_logp, coefficient):
    # Formula transcribed independently from read-only baseline core_algos.py
    # compute_policy_loss_vanilla; distillation_losses supplies detached -k1.
    advantages = -coefficient.detach()
    negative_approx_kl = torch.clamp(logp - old_logp.detach(), min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * 3.0
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    return torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)


def inputs():
    rng = torch.Generator().manual_seed(1927)
    hidden = torch.randn((3, 6, 4), generator=rng, dtype=DT, requires_grad=True)
    head = torch.nn.Linear(4, 9, dtype=DT).requires_grad_(False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=rng, dtype=DT))
        head.bias.copy_(torch.randn(head.bias.shape, generator=rng, dtype=DT))
    ids = torch.randint(0, 9, (3, 5), generator=rng)
    selected = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 0],
                             [1, 1, 0, 1, 1]], dtype=torch.bool)
    native = F.log_softmax(head(hidden[:, :-1]), -1).gather(-1, ids[..., None]).squeeze(-1)
    # Exercise all clipping branches, including ratio-clamp extremes.
    shifts = torch.tensor([[-30, -1, 0, 1, 30], [30, 2, -.1, .1, -30],
                           [-.2, 0, .2, 2, -.4]], dtype=DT)
    old = (native.detach() - shifts).requires_grad_(True)
    coef = torch.tensor([[-4, 2, 3, -2, 1], [-3, 5, -6, 7, 0],
                         [4, -5, 0, 10, -10]], dtype=DT, requires_grad=True)
    return hidden, ids, selected, head, old, coef


@pytest.mark.parametrize('chunk_size', [1, 2, 4, 64])
def test_raw_sum_vjp_matches_full_autograd_and_frozen_gradients(chunk_size):
    hidden, ids, selected, head, old, coef = inputs()
    lp = F.log_softmax(head(hidden[:, :-1]), -1).gather(-1, ids[..., None]).squeeze(-1)
    expected = baseline_tokens(lp, old, coef)[selected].sum()
    expected_grad = torch.autograd.grad(expected, hidden)[0]
    with torch.no_grad():
        grad, total, count = chunked_sampled_hidden_grad(
            hidden, ids, selected, head, old, coef, chunk_size)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(total, expected.detach(), rtol=1e-12, atol=1e-12)
    assert count == int(selected.sum()) == 9
    assert not grad.requires_grad and grad.grad_fn is None
    assert not total.requires_grad and total.grad_fn is None
    assert not grad[:, -1].any() and not grad[:, :-1][~selected].any()
    assert all(x.grad is None for x in (hidden, old, coef, *head.parameters()))
    torch.testing.assert_close(total / count, expected.detach() / selected.sum())


def test_exact_ppo_formula_values_gradients_clips_and_no_minus10_logp_clamp():
    # logp=-80 remains legitimate; only the k1 difference is clipped.
    shifts = torch.tensor([-40., -20.1, -1., math.log(.8), 0., math.log(1.2),
                           math.log(3.), 2., 20.1, 40.], dtype=DT)
    logp = torch.full((3, shifts.numel()), -80., dtype=DT, requires_grad=True)
    old = (logp.detach() - shifts).requires_grad_(True)
    coef = torch.tensor([-2., 0., 2.], dtype=DT)[:, None].expand_as(logp).requires_grad_(True)
    actual = sampled_pg_tokens(logp, old, coef)
    expected = baseline_tokens(logp, old, coef)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dg = torch.autograd.grad(actual.sum(), (logp, old, coef), allow_unused=True)
    expected_dg = torch.autograd.grad(expected.sum(), logp)[0]
    torch.testing.assert_close(dg[0], expected_dg, rtol=0, atol=0)
    assert dg[1:] == (None, None)
    assert torch.all(actual[2] <= 6)  # negative-advantage dual-clip cap
    assert actual[0, -1].item() == pytest.approx(-2.4)  # positive-advantage PPO cap
    assert actual[2, 0].item() == pytest.approx(1.6)  # negative-advantage PPO floor
    assert actual[2, 4].item() == pytest.approx(2.)  # raw logp, not clamped to -10


def test_coefficient_clamps_raw_difference_detaches_both_sources():
    current = torch.tensor([-80., -40., -30., -20., -60.], dtype=DT, requires_grad=True)
    teacher = torch.tensor([-20., -39., -32., -80., -60.], dtype=DT, requires_grad=True)
    coef = sampled_k1_coefficient(current, teacher)
    torch.testing.assert_close(coef, torch.tensor([-10., -1., 2., 10., 0.], dtype=DT))
    assert not coef.requires_grad and coef.grad_fn is None
    loss = sampled_pg_tokens(current, current.detach(), coef).sum()
    dg = torch.autograd.grad(loss, (current, teacher), allow_unused=True)
    torch.testing.assert_close(dg[0], coef)
    assert dg[1] is None and current.grad is None and teacher.grad is None


class Adapter:
    def __init__(self, hidden, head):
        self.hidden, self.head = hidden, head
        self.rows, self.calls, self.grad_flags = [], [], []

    def last_hidden(self, batch, with_grad, use_cache=False):
        self.calls.append((with_grad, use_cache))
        return self.hidden

    def project(self, rows):
        self.rows.append(rows.shape[0])
        self.grad_flags.append(torch.is_grad_enabled())
        return self.head(rows)


@pytest.mark.parametrize('chunk_size', [1, 3, 64])
def test_detached_teacher_scoring_masked_padding_prompt_and_genuine_eos(chunk_size):
    hidden, _, _, head, _, _ = inputs()
    # PAD == EOS == 2: keep genuine sampled EOS, drop post-EOS padding.
    batch = {
        'input_ids': torch.tensor([[1, 3, 4, 5, 2, 2], [1, 6, 2, 2, 2, 2],
                                   [1, 3, 4, 5, 6, 7]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.bool),
        # Intentionally true over padding too: attention must exclude it.
        'response_mask': torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 0]], dtype=torch.bool),
    }
    head.requires_grad_(True)  # no scoring graph even if incorrectly unfrozen
    adapter = Adapter(hidden, head)
    actual = token_log_probs(adapter, batch, chunk_size)
    valid = batch['response_mask'][:, 1:] & batch['attention_mask'][:, 1:]
    expected = F.log_softmax(head(hidden[:, :-1]), -1).gather(
        -1, batch['input_ids'][:, 1:, None]).squeeze(-1)
    expected = expected.detach().masked_fill(~valid, 0.)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0, 3] < 0 and actual[1, 1] < 0  # EOS retained
    assert not actual[0, 4:].any() and not actual[1, 2:].any()
    assert adapter.calls == [(False, False)]
    assert sum(adapter.rows) == 3 and max(adapter.rows) <= chunk_size
    assert not any(adapter.grad_flags)
    assert not actual.requires_grad and actual.grad_fn is None
    assert hidden.grad is None and all(p.grad is None for p in head.parameters())


def test_fd_utility_signed_native_coefficient_frozen_across_both_offsets():
    native = torch.tensor([-50., -40.], dtype=DT, requires_grad=True)
    teacher = torch.tensor([-52., -37.], dtype=DT, requires_grad=True)
    old = native.detach().clone().requires_grad_(True)
    coefficient = sampled_k1_coefficient(native, teacher)
    direction = torch.tensor([.7, .4], dtype=DT)
    delta = 1e-5
    good, bad = native.detach() + delta * direction, native.detach() - delta * direction
    utility = sampled_pg_utility(good, bad, old, coefficient, delta)
    expected = (baseline_tokens(bad, old, coefficient)
                - baseline_tokens(good, old, coefficient)) / (2 * delta)
    torch.testing.assert_close(utility, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(utility, -coefficient * direction, rtol=1e-8, atol=1e-8)
    assert utility[0] < 0 and utility[1] > 0  # signed scores, not abs utility
    torch.testing.assert_close(sampled_pg_utility(bad, good, old, coefficient, delta), -utility)
    # Recomputing k1 at the offsets produces a different, incorrect FD objective.
    wrong = (baseline_tokens(bad, old, sampled_k1_coefficient(bad, teacher))
             - baseline_tokens(good, old, sampled_k1_coefficient(good, teacher))) / (2 * delta)
    assert not torch.allclose(utility, wrong)
    assert not utility.requires_grad
    assert all(x.grad is None for x in (native, teacher, old))


@pytest.mark.parametrize('chunk_size', [1, 3, 64])
def test_chunk_and_microbatch_equality_with_one_external_normalization(chunk_size):
    hidden, ids, selected, head, old, coef = inputs()
    expected_grad, expected_sum, expected_count = chunked_sampled_hidden_grad(
        hidden, ids, selected, head, old, coef, 64)
    parts = [chunked_sampled_hidden_grad(hidden[i:i+1], ids[i:i+1], selected[i:i+1],
             head, old[i:i+1], coef[i:i+1], chunk_size) for i in range(3)]
    count = sum(x[2] for x in parts)
    grad = torch.cat([x[0] for x in parts])
    total = sum(x[1] for x in parts)
    assert count == expected_count
    torch.testing.assert_close(total / count, expected_sum / count, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad / count, expected_grad / count, rtol=1e-12, atol=1e-12)


def test_empty_selection_zero_detached_no_projection_or_grads():
    hidden, ids, selected, head, old, coef = inputs()
    selected.zero_()
    def forbidden(rows):
        pytest.fail('empty selection must not project any rows')
    grad, total, count = chunked_sampled_hidden_grad(hidden, ids, selected, forbidden, old, coef, 2)
    assert not grad.any() and total == 0 and count == 0
    assert not grad.requires_grad and not total.requires_grad
    assert total.dtype == DT and total.shape == () and total.device == hidden.device
    assert all(x.grad is None for x in (hidden, old, coef, *head.parameters()))
    adapter = Adapter(hidden, head)
    batch = {'input_ids': torch.ones((3, 6), dtype=torch.long),
             'attention_mask': torch.ones((3, 6), dtype=torch.bool),
             'response_mask': torch.zeros((3, 6), dtype=torch.bool)}
    lp = token_log_probs(adapter, batch, 2)
    assert not lp.any() and torch.isfinite(lp).all() and adapter.rows == []
    assert not lp.requires_grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_low_precision_scores_and_vjp_return_fp32_loss(dtype):
    torch.manual_seed(124)
    hidden = torch.randn(1, 4, 3, dtype=dtype, requires_grad=True)
    head = torch.nn.Linear(3, 5, dtype=dtype).requires_grad_(False)
    ids = torch.tensor([[1, 2, 3]])
    selected = torch.ones((1, 3), dtype=torch.bool)
    lp = F.log_softmax(head(hidden[:, :-1]).float(), -1).gather(-1, ids[..., None]).squeeze(-1)
    old, coef = lp.detach(), torch.ones_like(lp)
    expected = baseline_tokens(lp, old, coef).sum()
    expected_grad = torch.autograd.grad(expected, hidden)[0]
    grad, total, count = chunked_sampled_hidden_grad(hidden, ids, selected, head, old, coef, 2)
    assert total.dtype == torch.float32 and grad.dtype == dtype and count == 3
    torch.testing.assert_close(total, expected.detach(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(grad, expected_grad, rtol=.02, atol=.005)


@pytest.mark.parametrize('source', ['hidden', 'old', 'coefficient', 'logits'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_selected_inputs_fail(source, value):
    hidden, ids, selected, head, old, coef = inputs()
    if source == 'logits':
        with torch.no_grad():
            head.bias[0] = value
    else:
        with torch.no_grad():
            target = {'hidden': hidden, 'old': old, 'coefficient': coef}[source]
            if source == 'hidden':
                target[2, 4, 0] = value  # selected token in final chunk
            else:
                target[2, 4] = value
    with pytest.raises(FloatingPointError, match='nonfinite'):
        chunked_sampled_hidden_grad(hidden, ids, selected, head, old, coef, 2)


def test_nonfinite_hidden_backward_fails_without_accumulating_head_grads():
    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits):
            return logits.clone()
        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, float('nan'))
    hidden, ids, selected, head, old, coef = inputs()
    with pytest.raises(FloatingPointError, match='nonfinite sampled hidden gradient'):
        chunked_sampled_hidden_grad(hidden, ids, selected,
            lambda rows: BadBackward.apply(head(rows)), old, coef, 2)
    assert hidden.grad is None and all(p.grad is None for p in head.parameters())


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_teacher_and_raw_k1_rejected_before_clamp(value):
    native = torch.tensor([-80.], dtype=DT)
    teacher = torch.tensor([value], dtype=DT, requires_grad=True)
    with pytest.raises(FloatingPointError, match='teacher log probabilities'):
        sampled_k1_coefficient(native, teacher)
    adapter = Adapter(torch.ones((1, 2, 1), dtype=DT, requires_grad=True),
                      lambda rows: torch.full((rows.shape[0], 3), value, dtype=DT))
    batch = {'input_ids': torch.ones((1, 2), dtype=torch.long),
             'attention_mask': torch.ones((1, 2), dtype=torch.bool),
             'response_mask': torch.tensor([[False, True]])}
    with pytest.raises(FloatingPointError, match='sampled logits'):
        token_log_probs(adapter, batch, 1)


@pytest.mark.parametrize('delta', [0, -1, float('nan'), float('inf')])
def test_invalid_fd_delta(delta):
    x = torch.tensor([-5.], dtype=DT)
    with pytest.raises(ValueError, match='delta'):
        sampled_pg_utility(x, x, x, torch.ones_like(x), delta)


@pytest.mark.parametrize('chunk_size', [0, -1, True, 1.5])
def test_invalid_chunk_size_and_shapes_rejected(chunk_size):
    hidden, ids, selected, head, old, coef = inputs()
    with pytest.raises(ValueError, match='chunk_size'):
        chunked_sampled_hidden_grad(hidden, ids, selected, head, old, coef, chunk_size)


def test_shape_mismatches_rejected():
    hidden, ids, selected, head, old, coef = inputs()
    with pytest.raises(ValueError, match='equal shapes'):
        sampled_pg_tokens(old, old[:1], coef)
    with pytest.raises(ValueError, match='equal shapes'):
        sampled_k1_coefficient(old, coef[:1])
    with pytest.raises(ValueError, match='equal shapes'):
        chunked_sampled_hidden_grad(hidden, ids[:, :2], selected, head, old, coef, 2)
    with pytest.raises(ValueError, match=r'hidden \[B,L,D\]'):
        chunked_sampled_hidden_grad(hidden[:, :-1], ids, selected, head, old, coef, 2)

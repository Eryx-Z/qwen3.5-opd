"""Same-pass component logging must not alter the raw-sum CQ objective."""
import pytest
import torch
import torch.nn.functional as F

from cq_opd.losses import chunked_mixed_kl_hidden_grad


DT = torch.float64


def inputs():
    rng = torch.Generator().manual_seed(7301)
    student = torch.randn((2, 7, 4), generator=rng, dtype=DT, requires_grad=True)
    teacher = torch.randn((2, 7, 4), generator=rng, dtype=DT, requires_grad=True)
    student_head = torch.nn.Linear(4, 9, dtype=DT).requires_grad_(False)
    teacher_head = torch.nn.Linear(4, 9, dtype=DT).requires_grad_(False)
    with torch.no_grad():
        for head in (student_head, teacher_head):
            head.weight.copy_(torch.randn(head.weight.shape, generator=rng, dtype=DT))
            head.bias.copy_(torch.randn(head.bias.shape, generator=rng, dtype=DT))
    selected = torch.tensor([[1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1]], dtype=torch.bool)
    alpha = torch.tensor([[0., .9, .13, 1., .2, .48], [.8, .91, .07, .3, .62, 1.]],
                         dtype=DT, requires_grad=True)
    return student, teacher, alpha, selected, student_head, teacher_head


def independent_full_loss(student, teacher, alpha, selected, student_head, teacher_head):
    # Intentionally do not call the production KL helpers or chunking code.
    lp = F.log_softmax(student_head(student[:, :-1]), dim=-1)
    lq = F.log_softmax(teacher_head(teacher[:, :-1]).detach(), dim=-1)
    forward = (lq.exp() * (lq - lp)).sum(-1)
    reverse = (lp.exp() * (lp - lq)).sum(-1)
    a = alpha.detach()
    total = (a * forward + (1 - a) * reverse)[selected].sum()
    gradient = torch.autograd.grad(total, student)[0]
    return gradient, total.detach(), forward[selected].sum().detach(), reverse[selected].sum().detach()


@pytest.mark.parametrize('chunk_size', [1, 3, 4, 64])
def test_fp64_varying_alpha_gradient_and_components_match_full_loss(chunk_size):
    values = inputs()
    student, teacher, alpha, selected, student_head, teacher_head = values
    expected_grad, expected_total, expected_forward, expected_reverse = independent_full_loss(*values)
    # This frozen token-wise alpha is deliberately not a block-average mixture.
    mean_a = alpha.detach()[selected].mean()
    assert not torch.isclose(expected_total, mean_a * expected_forward + (1 - mean_a) * expected_reverse)
    with torch.no_grad():
        result = chunked_mixed_kl_hidden_grad(*values, chunk_size=chunk_size, return_components=True)
    assert len(result) == 5
    grad, total, count, forward_sum, reverse_sum = result
    assert count == 8
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
    for actual, expected in ((total, expected_total), (forward_sum, expected_forward),
                             (reverse_sum, expected_reverse)):
        assert actual.shape == () and actual.dtype == DT and actual.device == student.device
        assert not actual.requires_grad and actual.grad_fn is None
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        # Returning sums leaves the caller's one-time token normalization intact.
        torch.testing.assert_close(actual / count, expected / int(selected.sum()), rtol=1e-12, atol=1e-12)
    assert not grad.requires_grad and grad.grad_fn is None
    assert torch.count_nonzero(grad[:, -1]) == 0
    assert torch.count_nonzero(grad[:, :-1][~selected]) == 0
    assert student.grad is None and teacher.grad is None and alpha.grad is None
    assert all(p.grad is None for head in (student_head, teacher_head) for p in head.parameters())


class CountRows:
    def __init__(self, head):
        self.head = head
        self.rows = []

    def __call__(self, hidden):
        self.rows.append(hidden.shape[0])
        return self.head(hidden)


@pytest.mark.parametrize('chunk_size', [1, 3, 64])
def test_component_logging_projects_no_extra_head_rows_and_preserves_tuple3(chunk_size):
    student, teacher, alpha, selected, student_head, teacher_head = inputs()
    student_project, teacher_project = CountRows(student_head), CountRows(teacher_head)
    default = chunked_mixed_kl_hidden_grad(student, teacher, alpha, selected,
                                          student_project, teacher_project, chunk_size)
    assert len(default) == 3
    default_rows = (student_project.rows[:], teacher_project.rows[:])
    student_project.rows.clear()
    teacher_project.rows.clear()
    logged = chunked_mixed_kl_hidden_grad(student, teacher, alpha, selected,
                                         student_project, teacher_project, chunk_size,
                                         return_components=True)
    assert (student_project.rows, teacher_project.rows) == default_rows
    expected_rows = [min(chunk_size, 8 - start) for start in range(0, 8, chunk_size)]
    assert student_project.rows == teacher_project.rows == expected_rows
    assert sum(student_project.rows) == sum(teacher_project.rows) == int(selected.sum())
    torch.testing.assert_close(logged[0], default[0], rtol=0, atol=0)
    torch.testing.assert_close(logged[1], default[1], rtol=0, atol=0)
    assert logged[2] == default[2]


@pytest.mark.parametrize('return_components', [False, True])
@pytest.mark.parametrize('fault', ['student_nan', 'student_inf', 'teacher_nan', 'teacher_inf',
                                   'alpha_nan', 'alpha_inf', 'alpha_low', 'alpha_high'])
def test_bad_selected_input_in_later_chunk_fails_fast(return_components, fault):
    student, teacher, alpha, selected, student_head, teacher_head = inputs()
    source, kind = fault.split('_')
    value = {'nan': float('nan'), 'inf': float('inf'), 'low': -.01, 'high': 1.01}[kind]
    with torch.no_grad():
        if source == 'alpha':
            alpha[1, 5] = value
        else:
            {'student': student, 'teacher': teacher}[source][1, 5, 0] = value
    exception = ValueError if source == 'alpha' else FloatingPointError
    match = 'alpha must be finite' if source == 'alpha' else f'nonfinite {source} log probabilities'
    with pytest.raises(exception, match=match):
        chunked_mixed_kl_hidden_grad(student, teacher, alpha, selected, student_head, teacher_head,
                                     chunk_size=3, return_components=return_components)


@pytest.mark.parametrize('return_components', [False, True])
def test_nonfinite_hidden_gradient_protection_is_retained(return_components):
    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits):
            return logits.clone()

        @staticmethod
        def backward(ctx, grad):
            return torch.full_like(grad, float('nan'))

    student, teacher, alpha, selected, student_head, teacher_head = inputs()
    with pytest.raises(FloatingPointError, match='nonfinite hidden gradient'):
        chunked_mixed_kl_hidden_grad(student, teacher, alpha, selected,
                                     lambda h: BadBackward.apply(student_head(h)), teacher_head,
                                     chunk_size=3, return_components=return_components)


@pytest.mark.parametrize('return_components', [False, True])
def test_empty_selection_returns_detached_zero_sums_without_projections(return_components):
    student, teacher, alpha, selected, student_head, teacher_head = inputs()
    selected.zero_()
    def no_projection(hidden):
        pytest.fail('empty selection must not project')
    result = chunked_mixed_kl_hidden_grad(student, teacher, alpha, selected,
                                         no_projection, no_projection, chunk_size=3,
                                         return_components=return_components)
    assert len(result) == (5 if return_components else 3)
    assert torch.count_nonzero(result[0]) == 0 and result[2] == 0
    for scalar in (result[1], *result[3:]):
        assert scalar.shape == () and scalar.dtype == DT and scalar.device == student.device
        assert scalar.item() == 0 and not scalar.requires_grad and scalar.grad_fn is None

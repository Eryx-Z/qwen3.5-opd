"""Focused CPU regressions for probe direction normalization and cast safety."""
import math

import pytest
import torch

from cq_opd.probe import direction_from_gradients


def test_extreme_finite_float32_gradient_has_unit_finite_direction():
    gradient = torch.tensor([3e38, 3e38], dtype=torch.float32, requires_grad=True)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    result = direction_from_gradients([("p", parameter)], [gradient], source_step=7)

    assert result.valid and result.reason == "" and result.source_step == 7
    assert result.grad_norm > torch.finfo(torch.float32).max
    direction = result.tensors[0]
    assert direction.dtype == gradient.dtype and not direction.requires_grad
    assert torch.isfinite(direction).all() and torch.count_nonzero(direction) == 2
    torch.testing.assert_close(direction, torch.full_like(gradient, -1 / math.sqrt(2)))
    assert float(direction.double().norm()) == pytest.approx(1.0, rel=1e-7)
    assert parameter.grad is None and gradient.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normal_aggregate_direction_preserves_behavior(dtype):
    gradients = [torch.tensor([3.0], dtype=dtype), torch.tensor([-4.0], dtype=dtype)]
    parameters = [("p", torch.nn.Parameter(torch.zeros(1, dtype=dtype))),
                  ("q", torch.nn.Parameter(torch.zeros(1, dtype=dtype))),
                  ("unused", torch.nn.Parameter(torch.zeros(1, dtype=dtype)))]
    result = direction_from_gradients(parameters, gradients + [None], source_step=9,
                                      epsilon=0.5, unused_names=("unused",))

    assert result.valid and result.reason == "" and result.source_step == 9
    assert result.grad_norm == 5.0 and result.unused_names == ("unused",)
    for actual, gradient in zip(result.tensors, gradients + [torch.zeros(1, dtype=dtype)]):
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, -gradient / 5.5)
    assert all(parameter.grad is None for _, parameter in parameters)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_aggregate_direction_remains_invalid(dtype):
    parameters = [("p", torch.nn.Parameter(torch.ones(2, dtype=dtype))),
                  ("unused", torch.nn.Parameter(torch.ones(1, dtype=dtype)))]
    result = direction_from_gradients(parameters, [torch.zeros(2, dtype=dtype), None],
                                      source_step=11, unused_names=("unused",))

    assert not result.valid and result.reason == "no_probe_signal"
    assert result.grad_norm == 0.0 and result.source_step == 11
    assert result.unused_names == ("unused",)
    for actual, (_, parameter) in zip(result.tensors, parameters):
        assert actual.dtype == dtype
        assert torch.equal(actual, torch.zeros_like(parameter))


def test_nonzero_gradient_cannot_return_valid_all_zero_cast_direction():
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    with pytest.raises(FloatingPointError, match="probe direction"):
        direction_from_gradients([("p", parameter)], [torch.tensor([1e-40])], epsilon=1e308)

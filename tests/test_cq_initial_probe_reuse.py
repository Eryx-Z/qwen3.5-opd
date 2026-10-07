"""CPU proof of exact age-zero probe reuse, not stale direction caching."""
from copy import deepcopy
import json
import random

import pytest
import torch

from cq_opd.probe import (DirectionResult, capture_initial_probe, consume_initial_probe,
                          compute_probe_direction, leave_one_out_advantages)


def parameters():
    return [("adapter.a", torch.nn.Parameter(torch.tensor([1., 2.]))),
            ("adapter.b", torch.nn.Parameter(torch.tensor([[3., 4.]], dtype=torch.float64)))]


def protocol_key():
    return {"seed": 77 + 1001, "M": 4, "K": 4, "cap": 8, "dropout": 0,
            "score_protocol": "binary-v1", "probe_ids": [0, 1, 2, 3, 4, 5],
            "probe_hash": "exact-content", "frozen": [("base", 0, "float32", "cpu")],
            "backend": {"tf32": False}}


def draw_probe(params, rng):
    """Sixteen deterministic generation draws and real production direction math."""
    ids = rng.sample(list(range(6)), 4)
    seeds = [[rng.randrange(2**31) for _ in range(4)] for _ in ids]
    rewards = [[0, 1, 0, 1] for _ in ids]
    advantages = leave_one_out_advantages(torch.tensor(rewards))

    def gradients(i, k):
        return [torch.full_like(p, float(k + 1 + j + seeds[i][k] % 3))
                for j, (_, p) in enumerate(params)]

    d = compute_probe_direction(params, advantages, gradients, source_step=0)
    stats = {"probe_generation_time": 12.5, "probe_gradient_time": 2.25,
             "probe_grad_norm": d.grad_norm, "probe_rewards": rewards,
             "probe_trajectories": [{"id": i, "seeds": s} for i, s in zip(ids, seeds)]}
    return d, stats


def captured():
    params, key = parameters(), protocol_key()
    rng = random.Random(key["seed"])
    before = rng.getstate()
    d, stats = draw_probe(params, rng)
    after = rng.getstate()
    payload = capture_initial_probe(params, d, stats, key, before, after)
    return params, key, payload, d, stats, before, after


def assert_miss(payload, params, key, rng):
    before = rng.getstate()
    assert consume_initial_probe(payload, params, key, rng) is None
    assert rng.getstate() == before


def test_exact_first_probe_direction_stats_next_rng_and_zero_incremental_cost():
    params, key, payload, original_d, original_stats, before, after = captured()
    oracle_rng = random.Random(key["seed"])
    oracle_d, oracle_stats = draw_probe(params, oracle_rng)
    train_rng = random.Random(key["seed"])
    d, stats = consume_initial_probe(payload, params, deepcopy(key), train_rng)
    assert train_rng.getstate() == after == oracle_rng.getstate()
    assert train_rng.randrange(2**31) == oracle_rng.randrange(2**31)
    assert d.valid and d.source_step == 0
    assert (d.reason, d.grad_norm, d.unused_names) == (
        oracle_d.reason, oracle_d.grad_norm, oracle_d.unused_names)
    for actual, expected in zip(d.tensors, oracle_d.tensors):
        assert torch.equal(actual, expected)
    expected_stats = dict(oracle_stats, initial_probe_original_generation_time=12.5,
                          initial_probe_original_gradient_time=2.25,
                          probe_generation_time=0., probe_gradient_time=0.,
                          initial_probe_reused=True)
    assert stats == expected_stats
    assert json.loads(json.dumps(stats)) == expected_stats
    assert payload["stats"] == original_stats  # original calibration cost remains intact
    assert payload["rng_before"] == before
    stats["probe_rewards"][0][0] = 99
    assert payload["stats"] == original_stats
    assert original_d is not d


def test_exact_restored_values_can_hit_despite_symmetric_offset_versions():
    params, key, payload, _, _, _, _ = captured()
    versions = [p._version for _, p in params]
    with torch.no_grad():
        for _, p in params:
            p.add_(0.25)
            p.sub_(0.25)
    assert all(p._version > version for (_, p), version in zip(params, versions))
    assert consume_initial_probe(payload, params, key, random.Random(key["seed"])) is not None


def test_exact_value_check_rejects_one_ulp_change_even_with_equal_versions():
    params, key, payload, _, _, _, _ = captured()
    # A different tensor can have the same version and schema yet different values.
    changed = parameters()
    changed[0][1].data[0] = torch.nextafter(torch.tensor(1.), torch.tensor(2.))
    assert changed[0][1]._version == params[0][1]._version
    assert_miss(payload, changed, key, random.Random(key["seed"]))


@pytest.mark.parametrize("change", ["name", "order", "shape", "dtype", "device", "count",
                                    "key", "budget", "rng", "source", "invalid", "zero", "nan", "inf",
                                    "grad_norm", "parameter_nan", "parameter_inf"])
def test_any_identity_or_direction_fault_misses_without_rng_advance(change):
    params, key, payload, _, _, _, _ = captured()
    rng = random.Random(key["seed"])
    if change == "name":
        params[0] = ("other", params[0][1])
    elif change == "order":
        params.reverse()
    elif change == "shape":
        params[0] = (params[0][0], torch.nn.Parameter(torch.ones(1)))
    elif change == "dtype":
        params[0] = (params[0][0], torch.nn.Parameter(params[0][1].double()))
    elif change == "device":
        params[0] = (params[0][0], torch.empty(2, device="meta"))
    elif change == "count":
        params.pop()
    elif change == "key":
        key["backend"]["tf32"] = True
    elif change == "budget":
        key["cap"] = 2048  # A recalibrated response-token budget is a different protocol.
    elif change == "rng":
        rng.randrange(2**31)
    elif change == "source":
        payload["direction"].source_step = 1
    elif change == "invalid":
        payload["direction"].valid = False
    elif change == "zero":
        for tensor in payload["direction"].tensors:
            tensor.zero_()
    elif change in ("nan", "inf"):
        payload["direction"].tensors[-1][0, 0] = float(change)
    elif change == "grad_norm":
        payload["direction"].grad_norm = float("nan")
    else:
        with torch.no_grad():
            params[-1][1][0, 0] = float(change.removeprefix("parameter_"))
    assert_miss(payload, params, key, rng)


@pytest.mark.parametrize("fault", ["invalid", "source", "zero", "nan", "inf", "parameter_nan",
                                  "parameter_inf", "shape", "dtype", "empty", "grad_norm"])
def test_capture_refuses_invalid_nonfinite_or_wrong_source(fault):
    params, key, _, d, stats, before, after = captured()
    if fault == "invalid":
        d.valid = False
    elif fault == "source":
        d.source_step = 7
    elif fault == "zero":
        for tensor in d.tensors:
            tensor.zero_()
    elif fault in ("nan", "inf"):
        d.tensors[0][0] = float(fault)
    elif fault.startswith("parameter_"):
        with torch.no_grad():
            params[-1][1][0, 0] = float(fault.removeprefix("parameter_"))
    elif fault == "shape":
        d.tensors[0] = torch.ones(1)
    elif fault == "dtype":
        d.tensors[0] = d.tensors[0].double()
    elif fault == "empty":
        params = []
    elif fault == "grad_norm":
        d.grad_norm = float("inf")
    with pytest.raises(ValueError, match="initial probe requires"):
        capture_initial_probe(params, d, stats, key, before, after)


def test_snapshots_and_direction_are_graph_free_unaliased_and_metadata_deepcopied():
    params, key, _, d, stats, before, after = captured()
    # Even an externally constructed graph-bearing direction is detached on capture.
    d = DirectionResult(True, [p * 0.5 for _, p in params], 0, grad_norm=1.)
    payload = capture_initial_probe(params, d, stats, key, before, after)
    for (_, original), (_, snapshot), source, saved in zip(
            params, payload["parameters"], d.tensors, payload["direction"].tensors):
        assert snapshot.grad_fn is None and not snapshot.requires_grad
        assert saved.grad_fn is None and not saved.requires_grad
        assert snapshot.data_ptr() != original.data_ptr()
        assert saved.data_ptr() != source.data_ptr()
        assert original.grad is None
    stats["probe_rewards"][0][0] = 99
    key["probe_ids"].append(99)
    d.tensors[0].detach().zero_()
    assert payload["stats"]["probe_rewards"][0][0] == 0
    assert payload["key"]["probe_ids"] == list(range(6))
    assert torch.count_nonzero(payload["direction"].tensors[0]) == 2
    with torch.no_grad():
        params[0][1].add_(1.)
    assert torch.equal(payload["parameters"][0][1], torch.tensor([1., 2.]))


def test_tensor_value_and_finiteness_validation_uses_one_scalar_decision(monkeypatch):
    params, key, payload, _, _, _, _ = captured()
    calls = []
    original_bool = torch.Tensor.__bool__

    def counted_bool(tensor):
        calls.append(tensor.shape)
        return original_bool(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    assert consume_initial_probe(payload, params, key, random.Random(key["seed"])) is not None
    assert calls == [torch.Size([])]


def test_empty_payload_is_a_noop():
    rng = random.Random(42)
    assert_miss({}, parameters(), protocol_key(), rng)
    assert_miss(None, parameters(), protocol_key(), rng)

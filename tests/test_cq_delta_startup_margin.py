"""Tighter startup headroom chooses an interior point; never weakens FD gates."""
from copy import deepcopy

from cq_opd.trainer import choose_startup_delta


def candidate(delta,rank=.99,overlap=1.,sign=.99,passed=True):
    return dict(delta=delta,passed=passed,agreements=[(rank,overlap),(rank,overlap)],
                sign_agreement=[sign,sign],signal_fraction=1.)


def test_fp32_selects_first_robust_not_just_passing_delta():
    values=[candidate(.001,rank=.81,overlap=.75,sign=.82),candidate(.003,overlap=.875),
            candidate(.01),candidate(.03)]
    assert choose_startup_delta(values,precision='fp32')==.01
    assert [c['startup_robust'] for c in values]==[False,False,True,True]
    assert all(c['passed'] for c in values)  # minimum-gate results remain visible


def test_startup_margin_checks_both_offset_comparisons():
    values=[candidate(.01),candidate(.03)]
    values[0]['agreements'][1]=(.94,1.)
    assert choose_startup_delta(values,precision='fp32')==.03
    values=[candidate(.01),candidate(.03)]
    values[0]['sign_agreement'][1]=.89
    assert choose_startup_delta(values,precision='fp32')==.03


def test_no_robust_candidate_refuses_instead_of_taking_weak_or_failed_one():
    values=[candidate(.001,rank=.85),candidate(.01,passed=False)]
    assert choose_startup_delta(values,precision='fp32') is None
    assert not any(c['startup_robust'] for c in values)
    assert choose_startup_delta([],precision='fp32') is None


def test_original_bf16_minimum_rule_unchanged():
    values=[candidate(.001,passed=False),candidate(.003,rank=.81,overlap=.75,sign=.82),candidate(.01)]
    assert choose_startup_delta(deepcopy(values),precision='bf16')==.003
    assert choose_startup_delta(deepcopy(values))==.003

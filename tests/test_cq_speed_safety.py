"""Aggregated finite flags still stop before any optimizer mutation."""
import random
from types import SimpleNamespace

import pytest
import torch

from cq_opd import trainer
from cq_opd.probe import direction_from_gradients
from cq_opd.utility import score_symmetric_offsets
from test_cq_trainer import TinyAdapter, configuration, setup


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_backbone_gradient_refuses_update(monkeypatch,tmp_path,bad):
    splits=setup(monkeypatch);cfg=configuration();cfg.selector='full'
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    param=student.lora_named_parameters()[0][1]
    before=param.detach().clone()
    param.register_hook(lambda grad:torch.full_like(grad,bad))
    with pytest.raises(FloatingPointError,match='nonfinite training gradient'):
        trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
            {'mode':'constant','constant_alpha':.5},{'cq_valid':False,'delta':None},{},random.Random(42))
    assert torch.equal(param,before)
    assert not list(tmp_path.glob('checkpoint-*')) and not (tmp_path/'metrics.jsonl').exists()


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_multi_tensor_finite_checks_reject_bad_last_direction_before_offset(bad):
    named=[(str(i),torch.nn.Parameter(torch.ones(3))) for i in range(5)]
    gradients=[torch.ones_like(p) for _,p in named]
    gradients[-1][1]=bad
    with pytest.raises(FloatingPointError,match='probe gradient 4'):
        direction_from_gradients(named,gradients)
    before=[p.detach().clone() for _,p in named]
    with pytest.raises(FloatingPointError,match='direction 4'):
        score_symmetric_offsets(named,gradients,.01,lambda:pytest.fail('invalid forward reached'))
    assert all(torch.equal(p,a) for (_,p),a in zip(named,before))

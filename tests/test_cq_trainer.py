"""CPU end-to-end snapshot updates and checkpoint resume, using the production loop."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cq_opd import trainer


class TinyAdapter:
    def __init__(self, seed, trainable):
        torch.manual_seed(seed)
        self.model = torch.nn.Module()
        self.model.embed = torch.nn.Embedding(11, 4).double()
        self.model.head = torch.nn.Linear(4, 11).double()
        self.model.adapter = torch.nn.Linear(4, 4, bias=False).double()
        self.model.requires_grad_(False)
        self.model.adapter.requires_grad_(trainable)

    def lora_named_parameters(self):
        return [('adapter.weight', self.model.adapter.weight)]

    def last_hidden(self, batch, with_grad, use_cache=False):
        with torch.set_grad_enabled(with_grad):
            x = self.model.embed(batch['input_ids']).cumsum(1)
            return torch.tanh(x + self.model.adapter(x))

    def project(self, hidden):
        return self.model.head(hidden)


def configuration():
    return SimpleNamespace(selector='cq', keep_ratio=.2, steps=2, batch_size=2,
        max_new_tokens=4, block_len=2, chunk_size=2, lr=1e-3, weight_decay=0.,
        save_every=1, seed=42, teachability_topk=3, delta_questions=2, delta_recheck_every=5, output='unused', mode='train', resume=None, calibration_dir=None)


def setup(monkeypatch):
    def sample(student, tokenizer, row, cap, seed):
        return SimpleNamespace(input_ids=[1,2,3,4,5,6], prompt_len=2,
            response_ids=[3,4,5,6], response_text='#### 1',id=row['id'],truncated=False,seed=seed)
    monkeypatch.setattr(trainer, 'sample', sample)
    def direction(student, tokenizer, rows, cfg, step, rng):
        from cq_opd.probe import direction_from_gradients
        params=student.lora_named_parameters()
        return direction_from_gradients(params,[torch.ones_like(p) for _,p in params],step), {'probe_grad_norm':1.}
    monkeypatch.setattr(trainer, 'direction', direction)
    # Tiny selector smoke is not an FD-stability experiment. Gate math/refusal
    # and call schedules are tested independently in test_cq_gates.py.
    monkeypatch.setattr(trainer,'check_delta_gate',lambda *args,**kwargs:dict(
        report={'candidates':[{'passed':True}]},rollouts=[]))
    return dict(train=[{'id':str(i)} for i in range(3)],probe=[],dev=[])


@pytest.mark.parametrize('selector',['cq','random','kl','teachability','full'])
def test_end_to_end_two_updates(monkeypatch,tmp_path,selector):
    splits=setup(monkeypatch)
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    cfg=configuration();cfg.selector=selector
    initial=student.model.adapter.weight.detach().clone()
    calibration={'mode':'teacher_entropy','tau':2.,'scale':.5,'k':2.}
    trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
        calibration,{'cq_valid':True,'delta':.01},{},random.Random(42))
    metrics=[json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert [m['step'] for m in metrics]==[1,2]
    assert all(m['selected_tokens']>0 and np.isfinite(m['train_mixed_kl']) for m in metrics)
    assert not torch.equal(initial,student.model.adapter.weight)
    assert all(p.grad is None for p in teacher.model.parameters())
    checkpoint=torch.load(tmp_path/'checkpoint-2.pt',weights_only=False)
    assert checkpoint['step']==2 and checkpoint['scheduler']=='constant'


def test_resume_matches_uninterrupted(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    cfg=configuration()
    calibration={'mode':'constant','constant_alpha':.5}
    delta={'cq_valid':True,'delta':.01}
    trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
        calibration,delta,{},random.Random(42))
    expected=student.model.adapter.weight.detach().clone()
    checkpoint=torch.load(tmp_path/'checkpoint-1.pt',weights_only=False)
    fresh=TinyAdapter(1,True)
    other=tmp_path/'resume';other.mkdir()
    cfg.resume='different';cfg.output=str(other)
    trainer.train(fresh,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,other,
        calibration,delta,{},random.Random(7),checkpoint)
    torch.testing.assert_close(fresh.model.adapter.weight,expected,rtol=0,atol=0)

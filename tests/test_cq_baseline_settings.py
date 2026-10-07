"""Baseline common parameters, microbatch normalization and exact initial reuse."""
import argparse
import json
import random
from copy import deepcopy

import pytest
import torch
from transformers import AutoTokenizer

from cq_opd import trainer
from cq_opd.probe import capture_initial_probe, direction_from_gradients
from test_cq_trainer import TinyAdapter, configuration, setup


def test_cli_baseline_defaults_without_loading_gpu(monkeypatch,tmp_path):
    captured={};parse=argparse.ArgumentParser.parse_args
    def recording(self,*a,**k):
        result=parse(self,*a,**k);captured.update(vars(result));return result
    class StopBeforeModels(Exception):pass
    def stop(*a,**k):raise StopBeforeModels
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',recording)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',stop)
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-test')
    monkeypatch.setattr('sys.argv',['trainer','--mode','preflight','--output',str(tmp_path/'run')])
    with pytest.raises(StopBeforeModels):trainer.main()
    assert {k:captured[k] for k in ('batch_size','micro_batch_size','rollout_batch_size','max_new_tokens',
            'lr','weight_decay','seed','steps','save_every')}==dict(batch_size=32,micro_batch_size=4,
            rollout_batch_size=32,max_new_tokens=2048,lr=1e-6,weight_decay=.01,seed=42,steps=300,save_every=50)


def test_microbatch_raw_sums_match_single_trajectory_accumulation(monkeypatch,tmp_path):
    splits=setup(monkeypatch);results=[]
    for micro in (1,4):
        student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
        cfg=configuration();cfg.batch_size=5;cfg.micro_batch_size=micro;cfg.selector='full'
        output=tmp_path/str(micro);output.mkdir()
        trainer.train(student,teacher,type('Tokenizer',(),{'pad_token_id':0})(),splits,cfg,output,
            {'mode':'constant','constant_alpha':.5},{'cq_valid':False},{},random.Random(42))
        rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        results.append((student.model.adapter.weight.detach().clone(),rows))
    torch.testing.assert_close(results[0][0],results[1][0],atol=1e-15,rtol=1e-13)
    for a,b in zip(results[0][1],results[1][1]):
        assert a['selected_tokens']==b['selected_tokens'] and a['available_tokens']==b['available_tokens']
        assert a['train_mixed_kl']==pytest.approx(b['train_mixed_kl'],abs=1e-14)
        assert a['train_trajectories']==b['train_trajectories']


def test_initial_reuse_is_one_shot_and_resume_never_reuses(monkeypatch,tmp_path):
    splits=setup(monkeypatch);calls=[]
    def direction(student,tokenizer,rows,cfg,step,rng):
        calls.append(step);rng.randrange(1000)
        params=student.lora_named_parameters()
        return direction_from_gradients(params,[torch.ones_like(p) for _,p in params],step),dict(
            probe_grad_norm=1.,probe_generation_time=12.,probe_gradient_time=2.)
    monkeypatch.setattr(trainer,'direction',direction)
    cfg=configuration();cfg.probe_questions=4;cfg.probe_answers=4
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    probe_rng=random.Random(cfg.seed+2001);before=probe_rng.getstate()
    d,stats=direction(student,None,[],cfg,0,probe_rng)
    payload=capture_initial_probe(student.lora_named_parameters(),d,stats,
            trainer.initial_probe_key(student,[],cfg),before,probe_rng.getstate())
    calls.clear();cal={'mode':'constant','constant_alpha':.5};delta={'cq_valid':True,'delta':.01}
    trainer.train(student,teacher,type('Tokenizer',(),{'pad_token_id':0})(),splits,cfg,tmp_path,
            cal,delta,{},random.Random(42),initial_probe_cache=payload)
    assert calls==[1] and payload=={}
    rows=[json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert rows[0]['initial_probe_reused'] and rows[0]['probe_generation_time']==0
    assert rows[0]['initial_probe_original_generation_time']==12
    assert not rows[1]['initial_probe_reused']
    checkpoint=trainer.load_checkpoint(tmp_path/'checkpoint-1.pt')
    fresh=TinyAdapter(1,True);output=tmp_path/'resume';output.mkdir()
    bad_payload={'must_not_be_consumed':True}
    calls.clear()
    trainer.train(fresh,teacher,type('Tokenizer',(),{'pad_token_id':0})(),splits,cfg,output,
            cal,delta,{},random.Random(42),resume=checkpoint,initial_probe_cache=bad_payload)
    assert calls==[1] and bad_payload=={'must_not_be_consumed':True}
    torch.testing.assert_close(fresh.model.adapter.weight,student.model.adapter.weight,atol=0,rtol=0)

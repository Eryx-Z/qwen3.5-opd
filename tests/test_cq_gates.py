"""Production safety gates: modes, FD decisions, fixed contexts and refusal."""
import random
from types import SimpleNamespace

import pytest
import torch

from cq_opd import gates, trainer
from cq_opd.probe import direction_from_gradients
from cq_opd.rollout import Rollout, build_batch
from test_cq_adapter import tiny, records
from test_cq_trainer import TinyAdapter, configuration, setup


def test_native_mode_gate_compares_eval_and_scoring(monkeypatch):
    torch.set_num_threads(1)
    student,teacher=tiny(),tiny()
    batch=build_batch(records(),'cpu',2)
    proof=gates.validate_forward_modes(student,teacher,batch)
    assert proof['passed'] and 'mode_hidden_max_abs' in proof['models']['student']
    forward=student.language_model.forward
    def different_eval_kernel(*args,**kwargs):
        result=forward(*args,**kwargs)
        if not student.language_model.training:
            result.last_hidden_state=result.last_hidden_state+10
        return result
    monkeypatch.setattr(student.language_model,'forward',different_eval_kernel)
    with pytest.raises(AssertionError):
        gates.validate_forward_modes(student,teacher,batch)


@pytest.mark.parametrize('mode',['calibrate','train'])
def test_entry_modes_cannot_bypass_failed_mode_gate(monkeypatch,tmp_path,mode):
    from pathlib import Path
    from transformers import AutoTokenizer
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    tokenizer=SimpleNamespace(pad_token_id=0,apply_chat_template=lambda *a,**k:[1,2])
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-test-only')
    monkeypatch.setattr('sys.argv',['trainer','--mode',mode,'--objective','mixed_kl','--output',str(tmp_path/'run'),
        '--entropy-questions','1','--probe-questions','1','--delta-questions','1'])
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tokenizer)
    monkeypatch.setattr(trainer,'load_student',lambda *a,**k:student)
    monkeypatch.setattr(trainer,'load_teacher',lambda *a,**k:teacher)
    monkeypatch.setattr(trainer,'validate_alignment',lambda *a,**k:{})
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'fixture-metadata')
    row={'prompt':[{'role':'user','content':'hello'}],'extra_info':{'id':'a'}}
    monkeypatch.setattr(trainer,'load_splits',lambda *a:dict(train=[row],probe=[row],dev=[row]))
    monkeypatch.setattr(trainer,'generate',lambda *a,**k:Rollout([1,2,3,4],2,[3,4],'','a',False,42))
    build=build_batch
    monkeypatch.setattr(trainer,'build_batch',lambda records,device,pad:build(records,'cpu',pad))
    def fail_mode(*args):
        raise AssertionError('eval/scoring mode mismatch')
    monkeypatch.setattr(trainer,'validate_forward_modes',fail_mode)
    def forbidden(*args,**kwargs):
        raise AssertionError('calibration/update reached before the mode gate')
    monkeypatch.setattr(trainer,'entropy_calibration',forbidden)
    monkeypatch.setattr(trainer,'train',forbidden)
    with pytest.raises(AssertionError,match='eval/scoring mode mismatch'):
        trainer.main()
    assert not (tmp_path/'run'/'forward_mode_gate.json').exists()


def fixed_sample(student,tokenizer,row,cap,seed):
    return Rollout([1,2,3,4,5,6,7,8],2,[3,4,5,6,7,8],'#### 1',row['id'],False,seed)


def stable_score(student,teacher,batch,ht,alpha,d,delta,cfg):
    return torch.arange(alpha.numel(),dtype=alpha.dtype).reshape_as(alpha)


def test_fd_recheck_reuses_fixed_trajectories_and_refuses_unstable(monkeypatch):
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    cfg=configuration()
    calibration={'mode':'teacher_entropy','tau':2.,'scale':.5,'k':2.}
    params=student.lora_named_parameters()
    d=direction_from_gradients(params,[torch.ones_like(p) for _,p in params])
    calls=[]
    def sample(*args):
        calls.append(args[-1]);return fixed_sample(*args)
    monkeypatch.setattr(trainer,'sample',sample)
    monkeypatch.setattr(trainer,'score',stable_score)
    rng_state=random.getstate();torch_state=torch.get_rng_state().clone()
    args=(student,teacher,SimpleNamespace(pad_token_id=0),[{'id':'a'},{'id':'b'}],calibration,cfg,d,
          {'cq_valid':True,'delta':.01})
    state=gates.check_delta_gate(*args,None)
    assert state['report']['candidates'][0]['passed'] and len(calls)==2
    repeated=gates.check_delta_gate(*args,state)
    assert len(calls)==2 and repeated['rollouts']==state['rollouts']
    assert random.getstate()==rng_state and torch.equal(torch.get_rng_state(),torch_state)
    def unstable(*args):
        result=stable_score(*args)
        return -result if args[-2]>.01 else result
    monkeypatch.setattr(trainer,'score',unstable)
    with pytest.raises(RuntimeError,match='update refused'):
        gates.check_delta_gate(*args,state)


def test_fd_recheck_schedule_resume_and_distribution_shift():
    state={'last_step':1,'probe_success_rate':.5,'probe_grad_norm':10.}
    stats={'probe_success_rate':.5,'probe_grad_norm':10.}
    assert not gates.recheck_due(2,0,state,stats,5)
    assert gates.recheck_due(2,2,state,stats,5)  # resume forces a new current-policy check
    assert gates.recheck_due(5,0,state,stats,5)
    assert gates.recheck_due(1,0,state,stats,5)
    assert gates.recheck_due(3,0,state,dict(stats,probe_success_rate=.8),5)
    assert gates.recheck_due(3,0,state,dict(stats,probe_grad_norm=21.),5)


def test_failed_fd_gate_prevents_any_update(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    before=student.model.adapter.weight.detach().clone()
    def fail(*args,**kwargs):
        raise RuntimeError('current-policy finite difference recheck failed; CQ update refused')
    monkeypatch.setattr(trainer,'check_delta_gate',fail)
    with pytest.raises(RuntimeError,match='update refused'):
        trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,configuration(),tmp_path,
            {'mode':'constant','constant_alpha':.5},{'cq_valid':True,'delta':.01},{},random.Random(42))
    assert torch.equal(student.model.adapter.weight,before)
    assert not list(tmp_path.glob('checkpoint-*')) and not (tmp_path/'metrics.jsonl').exists()


def test_training_rechecks_early_periodic_and_after_resume(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    calls=[]
    def gate(*args,**kwargs):
        calls.append(args[6].source_step)
        return dict(rollouts=[{'id':'fixed'}],report={'candidates':[{'passed':True}]})
    monkeypatch.setattr(trainer,'check_delta_gate',gate)
    cfg=configuration();cfg.steps=7
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    cal={'mode':'constant','constant_alpha':.5};delta={'cq_valid':True,'delta':.01}
    trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
        cal,delta,{},random.Random(42))
    assert calls==[0,1,5]
    checkpoint=trainer.load_checkpoint(tmp_path/'checkpoint-2.pt')
    assert checkpoint['delta_recheck_state']['last_step']==1
    calls.clear();mode_checks=[]
    output=tmp_path/'resume';output.mkdir()
    trainer.train(TinyAdapter(1,True),teacher,SimpleNamespace(pad_token_id=0),splits,cfg,output,
        cal,delta,{},random.Random(1),checkpoint,mode_check=lambda:mode_checks.append(True))
    assert mode_checks==[True] and calls==[2,5]


def test_resume_no_signal_keeps_recheck_pending_until_valid_direction(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    calls=[]
    def gate(*args,**kwargs):
        calls.append(args[6].source_step)
        return dict(rollouts=[],report={'candidates':[{'passed':True}]})
    monkeypatch.setattr(trainer,'check_delta_gate',gate)
    cfg=configuration();cfg.steps=9
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    cal={'mode':'constant','constant_alpha':.5};delta={'cq_valid':True,'delta':.01}
    trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
        cal,delta,{},random.Random(42))
    checkpoint=trainer.load_checkpoint(tmp_path/'checkpoint-7.pt')
    assert checkpoint['delta_recheck_state']['last_step']==5
    normal_direction=trainer.direction
    def direction(*args):
        if args[4]==7:
            params=args[0].lora_named_parameters()
            return direction_from_gradients(params,[torch.zeros_like(p) for _,p in params],7),{'probe_grad_norm':0.}
        return normal_direction(*args)
    monkeypatch.setattr(trainer,'direction',direction)
    calls.clear();output=tmp_path/'resumed';output.mkdir()
    trainer.train(TinyAdapter(1,True),teacher,SimpleNamespace(pad_token_id=0),splits,cfg,output,
        cal,delta,{},random.Random(1),checkpoint)
    assert calls==[8]  # not periodic/due by age; pending resume obligation survives step7
    import json
    metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert metrics[0]['resume_recheck_pending'] and metrics[0]['fallback_reason']=='no_probe_signal'
    assert not metrics[1]['resume_recheck_pending'] and metrics[1]['delta_recheck_step']==8

"""CPU regression proof for caller-owned, frozen Teacher FD target caching."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cq_opd import gates, trainer
from cq_opd.blocks import shifted_valid_mask
from cq_opd.probe import direction_from_gradients
from cq_opd.rollout import Rollout
from test_cq_trainer import TinyAdapter, configuration


@pytest.fixture
def audit(monkeypatch):
    torch.set_num_threads(1)
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    # Explicitly freeze ALL Teacher parameters, including the head/embedding.
    teacher.model.requires_grad_(False)
    cfg=configuration();cfg.block_len=1
    calibration=dict(mode='teacher_entropy',tau=2.,scale=.5,k=2.)
    records=[Rollout([1,2,3,4,5,6,7,8],2,[3,4,5,6,7,8],'','a',False,1),
             Rollout([2,1,8,7,6,5,4,3],2,[8,7,6,5,4,3],'','b',False,2)]
    state=dict(rollouts=[vars(r).copy() for r in records])
    calls=dict(teacher_hidden=0,student_hidden=0,teacher_project=0,meta=0,scores=[])
    for adapter,name in [(teacher,'teacher_hidden'),(student,'student_hidden')]:
        original=adapter.last_hidden
        def hidden(*args,_original=original,_name=name,**kwargs):
            calls[_name]+=1
            return _original(*args,**kwargs)
        monkeypatch.setattr(adapter,'last_hidden',hidden)
    project=teacher.project
    def projected(*args,**kwargs):
        calls['teacher_project']+=1
        return project(*args,**kwargs)
    monkeypatch.setattr(teacher,'project',projected)
    target_meta=trainer.target_meta
    def meta(*args,**kwargs):
        calls['meta']+=1
        return target_meta(*args,**kwargs)
    monkeypatch.setattr(trainer,'target_meta',meta)
    score=trainer.score
    def scored(*args,**kwargs):
        result=score(*args,**kwargs)
        calls['scores'].append((args[5],args[6],result.clone()))
        return result
    monkeypatch.setattr(trainer,'score',scored)
    params=student.lora_named_parameters()
    direction=direction_from_gradients(params,[torch.ones_like(p) for _,p in params])
    tokenizer=SimpleNamespace(pad_token_id=0)
    def check(state=state,cache=None,calibration=calibration,direction=direction):
        return gates.check_delta_gate(student,teacher,tokenizer,[],calibration,cfg,direction,
                                      dict(cq_valid=True,delta=.01),state,cache=cache)
    def reset():
        for name in calls:
            calls[name]=[] if name=='scores' else 0
    return SimpleNamespace(student=student,teacher=teacher,cfg=cfg,calibration=calibration,
        state=state,calls=calls,direction=direction,tokenizer=tokenizer,check=check,reset=reset)


def rng_state():
    return random.getstate(),torch.get_rng_state().clone(),np.random.get_state()


def assert_rng_equal(before):
    after=rng_state()
    assert before[0]==after[0]
    assert torch.equal(before[1],after[1])
    assert before[2][0]==after[2][0]
    np.testing.assert_array_equal(before[2][1],after[2][1])
    assert before[2][2:]==after[2][2:]


def test_repeat_matches_uncached_gates_without_teacher_backbone_or_rng(audit):
    a=audit
    before=rng_state()
    baseline=a.check()
    assert baseline['report']['candidates'][0]['passed']
    assert a.calls['teacher_hidden']==a.calls['meta']==1
    a.reset();cache={}
    first=a.check(cache=cache)
    assert first==baseline and a.calls['teacher_hidden']==a.calls['meta']==1
    preparation_project_calls=a.calls['teacher_project']
    a.reset()
    repeated=a.check(state=first,cache=cache)
    assert repeated==baseline
    assert a.calls['teacher_hidden']==a.calls['meta']==0
    assert 0<a.calls['teacher_project']<preparation_project_calls  # logits NOT cached
    assert a.calls['student_hidden']==8  # noise pair plus both sides of 3 variants
    assert [delta for _,delta,_ in a.calls['scores']]==[.005,.01,.02]
    assert all(d is a.direction for d,_,_ in a.calls['scores'])
    assert_rng_equal(before)
    # Neither serialized checkpoint state nor its caller-owned input is polluted.
    assert json.loads(json.dumps(repeated,allow_nan=False))['rollouts']==a.state['rollouts']
    assert set(repeated)=={'rollouts','report'} and set(a.state)=={'rollouts'}
    assert set(cache)=={'fd_teacher_target'}
    entry=cache['fd_teacher_target']
    assert set(entry)=={'teacher','key','target'}
    hidden,meta=entry['target']
    assert hidden.shape==(2,8,4) and meta.alpha.shape==meta.entropy.shape==(2,7)
    assert not hidden.requires_grad and not meta.alpha.requires_grad


def test_student_and_current_direction_are_never_cached(audit):
    a=audit;cache={}
    previous=a.check(cache=cache)
    old_values=[v.clone() for _,_,v in a.calls['scores']]
    with torch.no_grad():
        a.student.model.adapter.weight.add_(.05)
    params=a.student.lora_named_parameters()
    current=direction_from_gradients(params,[-torch.ones_like(p) for _,p in params],source_step=9)
    a.reset()
    report=a.check(state=previous,cache=cache,direction=current)
    values=[v.clone() for _,_,v in a.calls['scores']]
    assert a.calls['teacher_hidden']==a.calls['meta']==0 and a.calls['student_hidden']==8
    assert len(values)==3 and all(d is current for d,_,_ in a.calls['scores'])
    assert any(not torch.equal(old,new) for old,new in zip(old_values,values))
    a.reset()
    fresh=a.check(state=previous,direction=current)
    assert report==fresh and a.calls['teacher_hidden']==a.calls['meta']==1
    for cached,(_,_,uncached) in zip(values,a.calls['scores']):
        torch.testing.assert_close(cached,uncached,rtol=0,atol=0)


@pytest.mark.parametrize('change',['calibration','calibration_metadata','tokens','rollout_metadata',
                                  'response_mask','attention_mask','input_ids','chunk_size',
                                  'teacher_embedding','teacher_head','teacher_adapter',
                                  'teacher_buffer','teacher_dtype','teacher_identity'])
def test_key_invalidates_all_target_bindings(audit,monkeypatch,change):
    a=audit;cache={}
    state=a.check(cache=cache)
    state=copy.deepcopy(state);calibration=copy.deepcopy(a.calibration)
    if change=='calibration':
        calibration['tau']+=.1
    elif change=='calibration_metadata':
        calibration['positions']=[['new-id',3]]
    elif change=='tokens':
        state['rollouts'][0]['input_ids'][3]=9
        state['rollouts'][0]['response_ids'][1]=9
    elif change=='rollout_metadata':
        state['rollouts'][0]['response_text']='different full serialized trajectory'
    elif change in ('response_mask','attention_mask','input_ids'):
        batch_of=trainer.batch_of
        def changed_batch(*args):
            batch=batch_of(*args)
            batch[change]=batch[change].clone()
            batch[change][0,3]=9 if change=='input_ids' else False
            return batch
        monkeypatch.setattr(trainer,'batch_of',changed_batch)
    elif change=='chunk_size':
        a.cfg.chunk_size+=1
    elif change in ('teacher_embedding','teacher_head','teacher_adapter'):
        module=getattr(a.teacher.model,dict(teacher_embedding='embed',teacher_head='head',
                                           teacher_adapter='adapter')[change])
        with torch.no_grad():
            module.weight.add_(.03)
    elif change=='teacher_buffer':
        a.teacher.model.register_buffer('projection_scale',torch.tensor(1.))
    elif change=='teacher_dtype':
        a.teacher.model.float()
        a.student.model.float()  # legacy noise assignment requires matching working dtype
    elif change=='teacher_identity':
        # Same tensor contents and version metadata, but a new Teacher object.
        teacher=copy.copy(a.teacher)
        def new_check(**kwargs):
            return gates.check_delta_gate(a.student,teacher,a.tokenizer,[],calibration,a.cfg,
                a.direction,dict(cq_valid=True,delta=.01),kwargs.get('state',state),
                cache=kwargs.get('cache'))
        a.check=new_check
    a.reset()
    cached=a.check(state=state,cache=cache,calibration=calibration)
    assert a.calls['teacher_hidden']==a.calls['meta']==1
    assert len(cache)==1  # replacement, not an unbounded trajectory history
    a.reset()
    fresh=a.check(state=state,calibration=calibration)
    assert cached==fresh and a.calls['teacher_hidden']==a.calls['meta']==1


def test_buffer_version_changes_invalidate(audit):
    a=audit
    a.teacher.model.register_buffer('projection_scale',torch.tensor(1.))
    cache={};state=a.check(cache=cache)
    a.teacher.model.projection_scale.add_(1.)
    a.reset();a.check(state=state,cache=cache)
    assert a.calls['teacher_hidden']==a.calls['meta']==1


def test_restored_state_with_empty_process_cache_rebuilds(audit):
    a=audit;cache={}
    state=a.check(cache=cache)
    restored=json.loads(json.dumps(state,allow_nan=False))
    a.reset();new_cache={}
    repeated=a.check(state=restored,cache=new_cache)
    assert repeated==state and a.calls['teacher_hidden']==a.calls['meta']==1
    a.reset();a.check(state=repeated,cache=new_cache)
    assert a.calls['teacher_hidden']==a.calls['meta']==0
    a.reset();a.check(state=repeated)  # cache=None retains legacy preparation
    assert a.calls['teacher_hidden']==a.calls['meta']==1


@pytest.mark.parametrize('parameter',['embed','head','adapter'])
def test_cache_refuses_any_unfrozen_teacher_parameter(audit,parameter):
    a=audit;cache={};state=a.check(cache=cache)
    getattr(a.teacher.model,parameter).weight.requires_grad_(True)
    a.reset()
    with pytest.raises(ValueError,match='fully frozen Teacher'):
        a.check(state=state,cache=cache)
    assert a.calls['teacher_hidden']==a.calls['student_hidden']==0
    assert a.check(state=state)['report']['candidates'][0]['passed']  # legacy untouched


def test_measure_fd_accepts_only_valid_detached_frozen_targets(audit):
    a=audit
    batch=trainer.batch_of([Rollout(**r) for r in a.state['rollouts']],a.student,a.tokenizer)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    hidden=a.teacher.last_hidden(batch,with_grad=False)
    meta=trainer.target_meta(a.teacher,hidden,valid,a.calibration,a.cfg)
    args=(a.student,a.teacher,batch,a.calibration,a.cfg,a.direction,[(None,.01)])
    baseline=gates.measure_fd(*args)
    a.reset()
    assert gates.measure_fd(*args,frozen_target=(hidden,meta))==baseline
    assert a.calls['teacher_hidden']==a.calls['meta']==0 and a.calls['student_hidden']==8
    bad_alpha=meta.alpha.clone();bad_alpha[0,2]=float('nan')
    wrong_meta=SimpleNamespace(alpha=meta.alpha[:,:-1],entropy=meta.entropy)
    for target,error in [((hidden[:,:-1],meta),ValueError),
            ((hidden.clone().requires_grad_(),meta),ValueError),
            ((hidden.detach(),wrong_meta),ValueError),
            ((hidden.detach(),SimpleNamespace(alpha=bad_alpha,entropy=meta.entropy)),
             FloatingPointError)]:
        with pytest.raises(error,match='frozen FD'):
            gates.measure_fd(*args,frozen_target=target)


def test_cache_does_not_hide_failed_current_policy_gate(audit,monkeypatch):
    a=audit;cache={};state=a.check(cache=cache)
    real_score=trainer.score
    def unstable(*args,**kwargs):
        values=real_score(*args,**kwargs)
        return -values if args[6]>.01 else values
    monkeypatch.setattr(trainer,'score',unstable)
    a.reset()
    with pytest.raises(RuntimeError,match='update refused'):
        a.check(state=state,cache=cache)
    assert a.calls['teacher_hidden']==a.calls['meta']==0 and a.calls['student_hidden']==8

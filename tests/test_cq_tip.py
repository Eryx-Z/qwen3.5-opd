import copy
import json
import random
from types import SimpleNamespace
import pytest
import torch
from cq_opd import trainer, topk_trainer as topk, replay_window
from cq_opd.blocks import shifted_valid_mask
from cq_opd.rollout import Rollout
from test_cq_topk import fixture
from test_cq_trainer import TinyAdapter, configuration, setup


def test_tip_score_and_selected_mixed_gradient_reference():
    s,t,b,c=fixture();c.keep_ratio=.5
    frozen=topk.target(t,b,{'mode':'constant','constant_alpha':.3},c)
    h=s.last_hidden(b,True)
    scores,mask=topk.tip_selection(s,h,b,frozen,c)
    valid=shifted_valid_mask(b['response_mask'],b['attention_mask'])
    lp=torch.log_softmax(s.project(h[:,:-1]).gather(-1,frozen.ids),-1)
    ent=-(lp.exp()*lp).sum(-1)/torch.log(torch.tensor(3.))
    r=(lp.exp()*(lp-frozen.logq)).sum(-1)
    for row in range(len(mask)):
        e=ent[row,valid[row]].detach();e=e.clamp_max(torch.quantile(e,.98))
        d=r[row,valid[row]].detach()
        norm=lambda x:(x-x.min())/(x.max()-x.min()) if x.max()>x.min() else torch.zeros_like(x)
        e,d=norm(e),norm(d)
        torch.testing.assert_close(scores[row,valid[row]],e+d-e*d)
        assert mask[row].sum()==max(1,int(valid[row].sum()*.5))
    f=(frozen.logq.exp()*(frozen.logq-lp)).sum(-1)
    ref=(.3*f+.7*r)[mask].sum()
    grad,loss,n,_,_=topk.backward_hidden(s,h,b,mask,frozen,c)
    torch.testing.assert_close(grad,torch.autograd.grad(ref,h)[0])
    torch.testing.assert_close(loss,ref)
    assert not scores.requires_grad and not (mask & ~valid).any()


def test_tip_constant_ties_masks_minimum_and_nonfinite(monkeypatch):
    s,t,b,c=fixture();c.keep_ratio=.01
    frozen=topk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    h=s.last_hidden(b,True)
    monkeypatch.setattr(topk,'project_topk',lambda student,rows,ids:torch.zeros_like(ids,dtype=h.dtype))
    # Uniform teacher makes every score equal; earliest valid position wins.
    frozen.logq.fill_(-torch.log(torch.tensor(3.,dtype=h.dtype)))
    frozen.entropy.fill_(torch.log(torch.tensor(3.,dtype=h.dtype)))
    scores,selected=topk.tip_selection(s,h,b,frozen,c)
    valid=shifted_valid_mask(b['response_mask'],b['attention_mask'])
    assert torch.count_nonzero(scores)==0
    for row in range(len(valid)):
        assert selected[row].nonzero().flatten().tolist()==[valid[row].nonzero()[0].item()]
    c.keep_ratio=1
    assert torch.equal(topk.tip_selection(s,h,b,frozen,c)[1],valid)
    monkeypatch.setattr(topk,'project_topk',lambda student,rows,ids:torch.full_like(ids,float('nan'),dtype=h.dtype))
    with pytest.raises(FloatingPointError):topk.tip_selection(s,h,b,frozen,c)


@pytest.mark.parametrize('width',[2,4])
def test_tip_window_fresh_scores_one_backbone_no_cq_exact_resume(monkeypatch,tmp_path,width):
    splits=setup(monkeypatch)
    monkeypatch.setattr(trainer,'sample',lambda s,t,r,c,seed:Rollout([1,2,3,4,5,6],2,[3,4,5,6],'',r['id'],False,seed))
    def forbidden(*a,**k):raise AssertionError('CQ called on TIP path')
    for name in ('direction','check_delta_gate','delta_calibration'):
        monkeypatch.setattr(trainer,name,forbidden)
    monkeypatch.setattr(topk,'score',forbidden)
    cfg=configuration();cfg.objective='adaptive_topk';cfg.selector='tip';cfg.keep_ratio=.5
    cfg.distillation_topk=3;cfg.reuse_window=width;cfg.steps=width+1;cfg.micro_batch_size=1
    cal=dict(mode='teacher_entropy',tau=1.,k=2.);delta=dict(cq_valid=False,delta=None)
    s,t=TinyAdapter(1,True),TinyAdapter(2,False)
    calls={'student':0,'teacher':0,'selection':0};selection_values=[]
    for adapter,key in ((s,'student'),(t,'teacher')):
        original=adapter.last_hidden
        def wrapped(*a,_f=original,_key=key,**k):calls[_key]+=1;return _f(*a,**k)
        monkeypatch.setattr(adapter,'last_hidden',wrapped)
    original=topk.tip_selection
    def select(*a,**k):
        calls['selection']+=1
        value=original(*a,**k);selection_values.append(value[0].clone());return value
    monkeypatch.setattr(topk,'tip_selection',select)
    trainer.train(s,t,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,cal,delta,{},random.Random(1))
    assert calls==dict(student=2*(width+1),teacher=4,selection=2*(width+1))
    assert not torch.equal(selection_values[0],selection_values[2])
    metrics=[json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert all(m['selection_age']==0 and not m['resume_recheck_pending'] and not m['cq_valid'] for m in metrics)
    assert [m['selection_source_step'] for m in metrics]==list(range(width+1))
    assert metrics[1]['teacher_forward_time']==metrics[1]['train_rollout_time']==0
    ck=trainer.load_checkpoint(tmp_path/'checkpoint-1.pt')
    assert all(e['scores'] is None and e['selected'] is None for e in ck['replay_window']['microbatches'])
    out=tmp_path/'resumed';out.mkdir();fresh=TinyAdapter(1,True)
    trainer.train(fresh,t,SimpleNamespace(pad_token_id=0),splits,cfg,out,cal,delta,{},random.Random(9),ck)
    torch.testing.assert_close(s.model.adapter.weight,fresh.model.adapter.weight,rtol=0,atol=0)
    final=trainer.load_checkpoint(out/f'checkpoint-{width+1}.pt')
    full=trainer.load_checkpoint(tmp_path/f'checkpoint-{width+1}.pt')
    assert final['extra_rngs']==full['extra_rngs'] and final['python_rng']==full['python_rng']
    bad=copy.deepcopy(ck['replay_window']);entry=bad['microbatches'][0]
    entry['scores']=torch.zeros_like(entry['target']['alpha']);replay_window.seal(bad)
    with pytest.raises(ValueError,match='stale selection'):
        replay_window.restore(bad,1,cfg,bad['binding'],0)


def test_tip_masks_change_with_student_and_keep_real_eos(monkeypatch):
    s,t,b,c=fixture();c.keep_ratio=.5
    # Keep last real response token (EOS ID 0), while excluding trailing padding.
    b['attention_mask'][0,-2:]=False;b['response_mask'][0,-2:]=False
    b['input_ids'][0,-3]=0
    frozen=topk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    valid=shifted_valid_mask(b['response_mask'],b['attention_mask'])
    assert valid[0,-3] and not valid[0,-2:].any()
    h=s.last_hidden(b,True)
    first=topk.tip_selection(s,h,b,frozen,c)[1]
    # Changing current hidden values reverses the position difficulty profile.
    flipped=h.detach().clone();positions=valid[0].nonzero(as_tuple=True)[0]
    flipped[0,positions]=flipped[0,positions.flip(0)]*30
    second=topk.tip_selection(s,flipped,b,frozen,c)[1]
    assert not torch.equal(first,second)
    c.keep_ratio=1
    assert torch.equal(topk.tip_selection(s,flipped,b,frozen,c)[1],valid)


def test_tip_microbatch_raw_sum_invariance(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    monkeypatch.setattr(trainer,'sample',lambda s,t,r,c,seed:Rollout([1,2,3,4,5,6],2,[3,4,5,6],'',r['id'],False,seed))
    results=[]
    for micro in (1,2):
        cfg=configuration();cfg.objective='adaptive_topk';cfg.selector='tip';cfg.keep_ratio=.5
        cfg.distillation_topk=3;cfg.reuse_window=2;cfg.micro_batch_size=micro
        s,t=TinyAdapter(1,True),TinyAdapter(2,False);out=tmp_path/str(micro);out.mkdir()
        trainer.train(s,t,SimpleNamespace(pad_token_id=0),splits,cfg,out,
            dict(mode='teacher_entropy',tau=1.,k=2.),dict(cq_valid=False,delta=None),{},random.Random(0))
        results.append(s.model.adapter.weight.detach().clone())
    torch.testing.assert_close(*results,atol=1e-12,rtol=1e-12)


def test_tip_cli_defaults_and_legacy(monkeypatch,tmp_path):
    import argparse
    from transformers import AutoTokenizer
    captured=[];parse=argparse.ArgumentParser.parse_args
    def record(self,*a,**k):
        result=parse(self,*a,**k);captured.append(result);return result
    class Stop(Exception):pass
    def stop(*a,**k):raise Stop()
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',record)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',stop)
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-test')
    for objective,selector,ratio,width in [('adaptive_topk','tip',.5,4),('sampled_k1','cq',.2,1)]:
        monkeypatch.setattr('sys.argv',['cq','--objective',objective,'--output',str(tmp_path/objective)])
        with pytest.raises(Stop):trainer.main()
        cfg=captured[-1]
        assert (cfg.selector,cfg.keep_ratio,cfg.reuse_window)==(selector,ratio,width)

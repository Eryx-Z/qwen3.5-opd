"""Narrow sampled-objective dispatch, frozen-policy FD and protocol checks."""
from types import SimpleNamespace
import random
import torch
import pytest
from cq_opd import trainer, sampled_trainer as st, gates
from cq_opd.probe import DirectionResult


class Adapter:
    def __init__(self,weight):
        self.model=torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad(): self.model.weight.fill_(weight)
        self.calls=0
    def last_hidden(self,batch,with_grad,use_cache=False):
        self.calls+=1
        with torch.set_grad_enabled(with_grad):
            return self.model(batch['input_ids'].double().unsqueeze(-1)/10)
    def project(self,h):
        return h*torch.tensor([-.8,.2,.5,1.1],dtype=h.dtype)
    def lora_named_parameters(self):return [('weight',self.model.weight)]


def setup():
    batch=dict(input_ids=torch.tensor([[1,2,3,1,0,2,1,3,2]]),
               response_mask=torch.tensor([[0,1,1,1,1,1,1,1,1]],dtype=torch.bool),
               attention_mask=torch.ones(1,9,dtype=torch.bool))
    cfg=SimpleNamespace(objective='sampled_k1',fd_scoring_dtype='native',chunk_size=2,
                        block_len=1,keep_ratio=.5)
    return Adapter(.3),Adapter(-.6),batch,cfg


def test_targets_frozen_native_and_teacher_once_across_offsets():
    s,t,b,c=setup();f=st.target(s,t,b,c)
    before=s.model.weight.detach().clone()
    d=DirectionResult(True,[torch.ones_like(s.model.weight)],0,grad_norm=1.)
    score=st.score(s,b,f,d,1e-3,c)
    assert t.calls==1 and s.calls==3
    assert torch.equal(s.model.weight,before)
    assert not f.old.requires_grad and not f.coefficient.requires_grad
    assert torch.isfinite(score).all() and score.abs().sum()>0
    assert s.model.weight.grad is None


def test_measure_dispatch_no_teacher_full_distribution(monkeypatch):
    s,t,b,c=setup();d=DirectionResult(True,[torch.ones_like(s.model.weight)],0,grad_norm=1.)
    monkeypatch.setattr(trainer,'target_meta',lambda *a,**k:pytest.fail('entropy path'))
    report=gates.measure_fd(s,t,b,{},c,d,[(.01,.001)])
    assert report['objective']=='sampled_k1' and report['noise_max']==0
    assert t.calls==1
    assert len(report['candidates'])==1


def test_cached_teacher_never_caches_old_policy():
    s,t,b,c=setup();f=st.target(s,t,b,c)
    with torch.no_grad():s.model.weight.add_(.2)
    new=st.target(s,t,b,c,teacher_logp=f.teacher)
    assert t.calls==1 and not torch.equal(new.old,f.old)
    assert torch.equal(new.teacher,f.teacher)
    assert not torch.equal(new.coefficient,f.coefficient)


def test_selected_gradient_is_raw_sum_and_no_teacher_grad():
    s,t,b,c=setup();f=st.target(s,t,b,c)
    valid=b['response_mask'][:,1:];h=s.last_hidden(b,True)
    gh,loss,n=st.backward_hidden(s,h,b,valid,f,c)
    grad=torch.autograd.grad(h,s.model.weight,grad_outputs=gh)[0]
    assert n==8 and torch.isfinite(loss) and grad.abs().sum()>0
    assert t.model.weight.grad is None and s.model.weight.grad is None


def test_protocol_distinguishes_sampled_and_legacy():
    cfg=SimpleNamespace(max_new_tokens=2048,entropy_questions=32,seed=42)
    legacy=trainer.entropy_protocol(cfg)
    cfg.objective='sampled_k1'
    assert trainer.entropy_protocol(cfg)!=legacy
    assert trainer.training_contract(cfg)['objective']=='sampled_k1'
    assert not st.enabled(SimpleNamespace())


@pytest.mark.parametrize('mode',['train','calibrate','preflight'])
def test_default_sampled_mode_gate_failure_prevents_work(monkeypatch,tmp_path,mode):
    from pathlib import Path
    from transformers import AutoTokenizer
    from cq_opd.rollout import Rollout,build_batch
    from test_cq_trainer import TinyAdapter
    s,t=TinyAdapter(1,True),TinyAdapter(2,False)
    tok=SimpleNamespace(pad_token_id=0,apply_chat_template=lambda *a,**k:[1,2])
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-only')
    monkeypatch.setattr('sys.argv',['trainer','--mode',mode,'--output',str(tmp_path/'run'),
        '--entropy-questions','1','--probe-questions','1','--delta-questions','1'])
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(trainer,'load_student',lambda *a,**k:s)
    monkeypatch.setattr(trainer,'load_teacher',lambda *a,**k:t)
    monkeypatch.setattr(trainer,'validate_alignment',lambda *a,**k:{})
    monkeypatch.setattr(Path,'read_bytes',lambda *a:b'fixture')
    row={'prompt':[],'extra_info':{'id':'1'}}
    monkeypatch.setattr(trainer,'load_splits',lambda *a:dict(train=[row],probe=[row],dev=[row]))
    monkeypatch.setattr(trainer,'generate',lambda *a,**k:Rollout([1,2,3],2,[3],'','1',False,42))
    monkeypatch.setattr(trainer,'build_batch',lambda r,d,p:build_batch(r,'cpu',p))
    def refuse(*a,**k):raise RuntimeError('sampled gate refused')
    monkeypatch.setattr(st,'forward_gate',refuse)
    monkeypatch.setattr(trainer,'train',lambda *a,**k:pytest.fail('update reached'))
    monkeypatch.setattr(trainer,'entropy_calibration',lambda *a,**k:pytest.fail('entropy reached'))
    with pytest.raises(RuntimeError,match='sampled gate refused'):trainer.main()


def test_real_loop_sampled_two_updates_resume_exact(monkeypatch,tmp_path):
    import json
    from test_cq_trainer import TinyAdapter,configuration,setup as loop_setup
    splits=loop_setup(monkeypatch);cfg=configuration();cfg.objective='sampled_k1'
    s,t=TinyAdapter(1,True),TinyAdapter(2,False)
    cal={'mode':'sampled_k1'};delta={'cq_valid':True,'delta':.01}
    monkeypatch.setattr(trainer,'target_meta',lambda *a,**k:pytest.fail('entropy computed'))
    trainer.train(s,t,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,cal,delta,{},random.Random(42))
    expected=s.model.adapter.weight.detach().clone()
    report=[json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert all(m['objective']=='sampled_k1' and 'train_mixed_kl' not in m and 'alpha_mean' not in m for m in report)
    resumed=TinyAdapter(1,True);out=tmp_path/'resume';out.mkdir()
    ck=trainer.load_checkpoint(tmp_path/'checkpoint-1.pt')
    trainer.train(resumed,t,SimpleNamespace(pad_token_id=0),splits,cfg,out,cal,delta,{},random.Random(0),ck)
    torch.testing.assert_close(expected,resumed.model.adapter.weight,rtol=0,atol=0)


def test_rollout_diagnostic_refuses_missing_and_stale_but_reports_finite_discrepancy():
    r=SimpleNamespace(sampled_log_probs=[-2.,-3.],response_ids=[1,2],prompt_len=2,policy_sha256='a')
    lp=torch.tensor([[0.,-1.,-3.]])
    proof=st.rollout_diagnostic([r],lp,expected_hash='a')
    assert proof['outside_tolerance_count']==1 and proof['tokens']==2
    with pytest.raises(RuntimeError,match='stale'):st.rollout_diagnostic([r],lp,expected_hash='b')
    r.sampled_log_probs=None
    with pytest.raises(RuntimeError,match='missing'):st.rollout_diagnostic([r],lp)


def test_native_qwen_sampled_forward_gate_fp32_head_restores_flags():
    from test_cq_adapter import tiny, records
    from cq_opd.rollout import build_batch
    s=tiny();t=tiny()
    s.fp32_head_output=True
    flags=[(m,m.training) for m in s.model.modules()]
    report=st.forward_gate(s,t,build_batch(records(),'cpu',2))
    assert report['passed'] and report['scope']=='sampled_tokens'
    assert all(m.training==flag for m,flag in flags)
    assert not s.model.lm_head._forward_hooks


def test_sampled_precision_gate_same_precision_real_grad():
    s,t,b,c=setup()
    report=st.precision_gate(s,t,b,c)
    assert report['passed'] and report['objective']=='sampled_k1'
    assert report['gradient_cosine']==pytest.approx(1.)
    assert report['gradient_norm_relative']==0

"""Adaptive Teacher-set KL projection, frozen targets and precision safety."""
from dataclasses import replace
from types import SimpleNamespace as S
import pytest
import torch
from cq_opd import topk_trainer as tk
from cq_opd.blocks import shifted_valid_mask
from cq_opd.losses import calibrate_entropy
from cq_opd.probe import DirectionResult
from test_cq_topk import fixture


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32,torch.float64])
def test_gathered_projection_matches_explicit_fp32_policy_head(dtype):
    torch.manual_seed(17)
    head=torch.nn.Linear(7,19,bias=True).to(dtype).requires_grad_(False)
    a=S(model=S(lm_head=head))
    x=torch.randn(5,7,dtype=dtype,requires_grad=True)
    ids=torch.tensor([[0,2,5],[1,4,18],[3,9,15],[6,8,10],[11,12,13]])
    work=torch.float64 if dtype==torch.float64 else torch.float32
    actual=tk.project_topk(a,x,ids)
    expected=torch.nn.functional.linear(x.to(work),head.weight.to(work),head.bias.to(work)).gather(-1,ids)
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
    ga=torch.autograd.grad(actual.sum(),x,retain_graph=True)[0]
    ge=torch.autograd.grad(expected.sum(),x)[0]
    torch.testing.assert_close(ga,ge,atol=2e-7,rtol=2e-6)
    assert head.weight.grad is None and head.bias.grad is None


@pytest.mark.parametrize('fault',['alpha','entropy','logq','ids','duplicate','grad','shape','norm'])
def test_invalid_reusable_targets_rejected(fault):
    s,t,b,c=fixture();f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    if fault=='alpha':f.alpha[0,0]=1.1
    elif fault=='entropy':f.entropy[0,0]+=1
    elif fault=='logq':f.logq[0,0,0]=float('nan')
    elif fault=='ids':f.ids[0,0,0]=-1
    elif fault=='duplicate':f.ids[0,0,0]=f.ids[0,0,1]
    elif fault=='grad':f.logq.requires_grad_(True)
    elif fault=='shape':f=replace(f,alpha=f.alpha[:,:-1])
    elif fault=='norm':f.logq+=1
    with pytest.raises((ValueError,FloatingPointError)):tk.values(s,b,f,c)


def test_loss_ignores_logits_outside_teacher_set_and_detaches_teacher_alpha():
    head=torch.nn.Linear(2,5,bias=False).double().requires_grad_(False)
    student=S(model=S(lm_head=head));rows=torch.tensor([[.2,.7]],dtype=torch.float64,requires_grad=True)
    ids=torch.tensor([[0,2,4]])
    q=torch.log_softmax(torch.tensor([[.3,.5,.8]],dtype=torch.float64),-1).requires_grad_(True)
    alpha=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    loss,_,_=tk.components(student,rows,ids,q,alpha)
    grads=torch.autograd.grad(loss.sum(),(rows,q,alpha),allow_unused=True)
    assert grads[0] is not None and grads[1] is None and grads[2] is None
    with torch.no_grad():head.weight[1].fill_(1000);head.weight[3].fill_(-1000)
    new,_,_=tk.components(student,rows,ids,q,alpha)
    torch.testing.assert_close(new,loss,atol=0,rtol=0)


def test_head_bounds_trainable_bias_and_invalid_selection_refused():
    head=torch.nn.Linear(2,4,bias=True).requires_grad_(False);a=S(model=S(lm_head=head))
    x=torch.zeros(1,2)
    with pytest.raises(ValueError,match='vocabulary'):tk.project_topk(a,x,torch.tensor([[4]]))
    head.bias.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):tk.project_topk(a,x,torch.tensor([[0]]))
    s,t,b,c=fixture();b['response_mask'][0,1]=False
    f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    with pytest.raises(ValueError,match='prompt/padding'):
        tk.backward_hidden(s,s.last_hidden(b,True),b,torch.ones_like(b['response_mask'][:,1:]),f,c)


def test_empty_selection_does_not_project(monkeypatch):
    s,t,b,c=fixture();f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    h=s.last_hidden(b,True)
    monkeypatch.setattr(tk,'project_topk',lambda *a:pytest.fail('empty projection'))
    g,l,n,ff,rr=tk.backward_hidden(s,h,b,torch.zeros_like(f.alpha,dtype=torch.bool),f,c)
    assert n==0 and not g.any() and l==ff==rr==0
    assert not g.requires_grad and not l.requires_grad


def test_adaptive_alpha_and_teacher_no_gradient_microbatch_raw_sums():
    s,t,b,c=fixture()
    b={k:v.repeat(3,1) for k,v in b.items()}
    # Include real response EOS id2 and padding at a later position.
    b['attention_mask'][1,-1]=False;b['response_mask'][1,-1]=False
    first=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    valid=shifted_valid_mask(b['response_mask'],b['attention_mask'])
    cal=calibrate_entropy(first.entropy[valid]);f=tk.target(t,b,cal,c)
    assert torch.allclose(f.alpha[valid],torch.sigmoid((f.entropy[valid]-cal['tau'])/cal['scale']))
    h=s.last_hidden(b,True);g,l,n,ff,rr=tk.backward_hidden(s,h,b,valid,f,c)
    total=torch.autograd.grad(h,s.model.weight,grad_outputs=g)[0]
    accum=torch.zeros_like(total);sums=torch.zeros(3,dtype=l.dtype)
    count=0
    for i in range(3):
        mb={k:v[i:i+1] for k,v in b.items()}
        mf=tk.Target(*(getattr(f,k)[i:i+1] for k in ('ids','logq','entropy','alpha')))
        mh=s.last_hidden(mb,True)
        mg,ml,mn,mfwd,mrev=tk.backward_hidden(s,mh,mb,valid[i:i+1],mf,c)
        accum+=torch.autograd.grad(mh,s.model.weight,grad_outputs=mg)[0]
        sums+=torch.stack([ml,mfwd,mrev]);count+=mn
    torch.testing.assert_close(total,accum,atol=1e-14,rtol=1e-12)
    torch.testing.assert_close(torch.stack([l,ff,rr]),sums)
    assert count==n==23
    assert t.model.weight.grad is None and not f.alpha.requires_grad


def test_teacher_target_immutable_through_student_updates_and_signed_fd():
    s,t,b,c=fixture();f=tk.target(t,b,{'mode':'constant','constant_alpha':.6},c)
    before=[getattr(f,k).clone() for k in ('ids','logq','entropy','alpha')]
    d=DirectionResult(True,[torch.ones_like(s.model.weight)],0,grad_norm=1)
    positive=tk.score(s,b,f,d,1e-4,c)
    negative=tk.score(s,b,f,DirectionResult(True,[-d.tensors[0]],0,grad_norm=1),1e-4,c)
    torch.testing.assert_close(positive,-negative)
    with torch.no_grad():s.model.weight.add_(.1)
    assert not torch.equal(positive,tk.score(s,b,f,d,1e-4,c))
    assert all(torch.equal(x,getattr(f,k)) for x,k in zip(before,('ids','logq','entropy','alpha')))
    assert t.calls==1


def test_precision_gate_same_native_dtype_has_exact_gradient_and_no_grad_pollution():
    s,t,b,c=fixture();proof=tk.precision_gate(s,t,b,c)
    assert proof['passed'] and proof['gradient_cosine']==pytest.approx(1)
    assert proof['gradient_norm_relative']==0
    assert proof['minimum_cosine']==.99 and proof['maximum_norm_relative']==.05
    assert s.model.weight.grad is None and t.model.weight.grad is None


@pytest.mark.parametrize('bad',[True,1,0,-2,2.5])
def test_non_integral_or_too_small_k_refused_before_teacher_forward(bad):
    s,t,b,c=fixture();c.distillation_topk=bad
    with pytest.raises(ValueError):tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    assert t.calls==0


def test_precision_gate_does_not_accept_wrong_gradient_direction(monkeypatch):
    s,t,b,c=fixture();real=tk.backward_hidden;calls=[]
    def backward(*args,**kwargs):
        result=real(*args,**kwargs);calls.append(1)
        if len(calls)==2:return (-result[0],*result[1:])
        return result
    monkeypatch.setattr(tk,'backward_hidden',backward)
    with pytest.raises(RuntimeError,match='gradient mismatch'):tk.precision_gate(s,t,b,c)
    assert s.model.weight.grad is None


def test_fd_failure_restores_master_and_frozen_storage(monkeypatch):
    s,t,b,c=fixture();c.fd_scoring_dtype='fp32'
    s.model.register_buffer('frozen_fixture',torch.ones(2,dtype=torch.bfloat16))
    storage=s.model.frozen_fixture;weight=s.model.weight.detach().clone()
    f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    d=DirectionResult(True,[torch.ones_like(weight)],0,grad_norm=1)
    def fail(*args,**kwargs):raise RuntimeError('injected failure')
    monkeypatch.setattr(tk,'values',fail)
    with pytest.raises(RuntimeError,match='injected'):tk.score(s,b,f,d,.001,c)
    assert s.model.frozen_fixture is storage and torch.equal(s.model.weight,weight)


def test_real_qwen_bf16_topk_vjp_eos_padding_no_teacher_graph():
    from test_cq_adapter import tiny,records
    from cq_opd.rollout import build_batch
    torch.manual_seed(10);torch.set_num_threads(1)
    s=tiny();t=tiny();t.model.requires_grad_(False)
    c=S(distillation_topk=8,chunk_size=2)
    b=build_batch(records(),'cpu',2);valid=shifted_valid_mask(b['response_mask'],b['attention_mask'])
    f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    h=s.last_hidden(b,True);g,l,n,ff,rr=tk.backward_hidden(s,h,b,valid,f,c)
    gradients=torch.autograd.grad(h,[p for _,p in s.lora_named_parameters()],grad_outputs=g)
    assert n==3 and all(torch.isfinite(x).all() for x in gradients)
    assert any(x.abs().sum()>0 for x in gradients)
    assert all(p.grad is None for p in t.model.parameters())
    assert not g[~b['attention_mask']].any()

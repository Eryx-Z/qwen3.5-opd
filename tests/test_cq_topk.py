from types import SimpleNamespace as S
import torch
import pytest
from cq_opd import topk_trainer as tk,trainer,gates
from cq_opd.losses import calibrate_entropy,entropy_alpha
from cq_opd.probe import DirectionResult
from test_cq_sampled_trainer import setup


def fixture():
    s,t,b,c=setup();c.objective='adaptive_topk';c.distillation_topk=3
    return s,t,b,c


def test_teacher_set_normalization_entropy_and_frozen_gradients():
    s,t,b,c=fixture();f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    assert torch.allclose(f.logq.exp().sum(-1),torch.ones_like(f.alpha))
    assert not f.logq.requires_grad and not f.alpha.requires_grad
    assert torch.allclose(f.entropy,-(f.logq.exp()*f.logq).sum(-1))
    cal=calibrate_entropy(f.entropy)
    e,a=entropy_alpha(f.logq,cal)
    assert torch.allclose(a,torch.sigmoid((e-cal['tau'])/cal['scale']))
    assert t.model.weight.grad is None


def test_topk_loss_vjp_matches_full_reference_gather():
    s,t,b,c=fixture();f=tk.target(t,b,{'mode':'constant','constant_alpha':.3},c)
    valid=b['response_mask'][:,1:];valid[0,2]=False
    h=s.last_hidden(b,True)
    grad,loss,n,ff,rr=tk.backward_hidden(s,h,b,valid,f,c)
    p=torch.log_softmax(s.project(h[:,:-1]).gather(-1,f.ids),-1)
    q=f.logq
    forward=(q.exp()*(q-p)).sum(-1);reverse=(p.exp()*(p-q)).sum(-1)
    ref=(.3*forward+.7*reverse)[valid].sum()
    expected=torch.autograd.grad(ref,h,retain_graph=True)[0]
    torch.testing.assert_close(grad,expected)
    torch.testing.assert_close(loss,ref)
    assert n==7 and torch.count_nonzero(grad[0,2])==0
    torch.testing.assert_close(ff,forward[valid].sum());torch.testing.assert_close(rr,reverse[valid].sum())
    d=DirectionResult(True,[torch.ones_like(s.model.weight)],0,grad_norm=1.)
    utility=tk.score(s,b,f,d,1e-4,c)
    derivative=torch.autograd.grad(ref,s.model.weight)[0]
    torch.testing.assert_close(utility[valid].sum(),-derivative.squeeze(),atol=1e-7,rtol=1e-5)
    assert s.model.weight.grad is None


def test_direct_frozen_head_rows_match_full_projection_gradient():
    torch.manual_seed(42)
    head=torch.nn.Linear(5,13).double();head.requires_grad_(False)
    s=S(model=S(lm_head=head));x=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
    ids=torch.tensor([[1,2,3],[4,5,6],[0,11,12],[2,6,9]])
    direct=tk.project_topk(s,x,ids);full=head(x).gather(-1,ids)
    torch.testing.assert_close(direct,full)
    a=torch.autograd.grad(direct.square().sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad(full.square().sum(),x)[0]
    torch.testing.assert_close(a,b);assert head.weight.grad is None


def test_fd_dispatch_reuses_teacher_and_restores_student():
    s,t,b,c=fixture();f=tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)
    before=s.model.weight.clone();d=DirectionResult(True,[torch.ones_like(s.model.weight)],0,grad_norm=1.)
    report=gates.measure_fd(s,t,b,{},c,d,[(.01,.001)],frozen_target=f)
    assert report['objective']=='adaptive_topk' and report['noise_max']==0 and t.calls==1
    assert torch.equal(before,s.model.weight)


def test_topk_protocol_binding():
    c=S(objective='adaptive_topk',distillation_topk=64,max_new_tokens=2048,entropy_questions=32,seed=42)
    a=trainer.entropy_protocol(c);c.distillation_topk=32
    assert trainer.entropy_protocol(c)!=a
    assert a['entropy_scope']=='teacher_topk_conditional'


@pytest.mark.parametrize('k',[1,5])
def test_invalid_k_refused(k):
    s,t,b,c=fixture();c.distillation_topk=k
    with pytest.raises(ValueError):tk.target(t,b,{'mode':'constant','constant_alpha':.5},c)

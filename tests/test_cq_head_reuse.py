from types import SimpleNamespace
import pytest
import torch
from cq_opd.model_adapter import QwenAdapter,gradient_projector
from cq_opd.sampled_loss import chunked_sampled_hidden_grad


def adapter():
    a=object.__new__(QwenAdapter)
    a.model=SimpleNamespace(lm_head=torch.nn.Linear(8,17,bias=False).bfloat16())
    a.model.lm_head.requires_grad_(False);a.fp32_head_output=True
    return a


def test_chunked_reuse_exact_loss_and_hidden_gradient(monkeypatch):
    torch.manual_seed(42);a=adapter()
    h=torch.randn(2,10,8,dtype=torch.bfloat16,requires_grad=True)
    ids=torch.randint(17,(2,9));mask=torch.ones(2,9,dtype=torch.bool)
    old=torch.full((2,9),-3.);coef=torch.randn(2,9)
    expected=chunked_sampled_hidden_grad(h,ids,mask,a.project,old,coef,2)
    calls=[];original=torch.Tensor.float
    def tracked(t,*args,**kw):
        if t.shape==a.model.lm_head.weight.shape:calls.append(1)
        return original(t,*args,**kw)
    monkeypatch.setattr(torch.Tensor,'float',tracked)
    with gradient_projector(a) as project:
        actual=chunked_sampled_hidden_grad(h,ids,mask,project,old,coef,2)
    assert len(calls)==1
    assert actual[2]==expected[2]
    for x,y in zip(actual[:2],expected[:2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert a.model.lm_head.weight.grad is None and h.grad is None
    with pytest.raises(RuntimeError,match='outside'):project(h)


def test_projector_exception_and_mutation_refused():
    a=adapter();h=torch.randn(1,8,dtype=torch.bfloat16,requires_grad=True)
    with pytest.raises(RuntimeError,match='injected'):
        with gradient_projector(a) as project:
            project(h);raise RuntimeError('injected')
    with pytest.raises(RuntimeError,match='outside'):project(h)
    with gradient_projector(a) as project:
        with torch.no_grad():a.model.lm_head.weight.add_(1)
        with pytest.raises(RuntimeError,match='changed'):project(h)


def test_generic_adapter_fallback():
    f=lambda x:x*2
    with gradient_projector(SimpleNamespace(project=f)) as project:assert project is f

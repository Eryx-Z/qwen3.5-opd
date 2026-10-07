"""FD precision lifetime optimization preserves report and exception restoration."""
from contextlib import contextmanager
import pytest
import torch
from cq_opd import sampled_trainer as st
from cq_opd.model_adapter import temporary_scoring_precision
from cq_opd.probe import DirectionResult
from test_cq_sampled_trainer import setup


@pytest.mark.parametrize('fail',[False,True])
def test_all_fd_candidates_share_one_promotion_and_restore(monkeypatch,fail):
    s,t,b,c=setup();c.fd_scoring_dtype='fp32'
    s.model.register_buffer('frozen_fixture',torch.ones(2,dtype=torch.bfloat16))
    original=s.model.frozen_fixture;master=s.model.weight.detach().clone()
    d=DirectionResult(True,[torch.ones_like(s.model.weight)],0,grad_norm=1.)
    candidates=[(.01,.001),(.02,.002)]
    # Reference has the old per-score precision scopes with identical targets.
    f=st.target(s,t,b,c)
    expected=st._measure_fd_frozen(s,b,c,d,candidates,f)
    promotions=[];target_dtypes=[]
    real_target=st.target
    def target(*args,**kwargs):
        target_dtypes.append(s.model.frozen_fixture.dtype)
        return real_target(*args,**kwargs)
    @contextmanager
    def tracked(adapter,dtype):
        if adapter.model.frozen_fixture.dtype!=dtype:promotions.append(dtype)
        with temporary_scoring_precision(adapter,dtype):yield
    monkeypatch.setattr(st,'target',target)
    monkeypatch.setattr(st,'temporary_scoring_precision',tracked)
    if fail:
        real_score=st.score
        calls=[]
        def score(*args,**kwargs):
            calls.append(1)
            result=real_score(*args,**kwargs)
            if len(calls)==2:raise RuntimeError('injected score failure')
            return result
        monkeypatch.setattr(st,'score',score)
        with pytest.raises(RuntimeError,match='injected'):st.measure_fd(s,t,b,{},c,d,candidates)
    else:
        actual=st.measure_fd(s,t,b,{},c,d,candidates)
        assert actual==expected
    assert target_dtypes==[torch.bfloat16]
    assert promotions==[torch.float32]
    assert s.model.frozen_fixture is original
    assert torch.equal(s.model.weight,master) and s.model.weight.grad is None

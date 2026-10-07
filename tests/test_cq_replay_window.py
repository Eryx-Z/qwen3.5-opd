from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from cq_opd import replay_window as rw
from cq_opd.topk_trainer import Target
from cq_opd.rollout import Rollout


def fixture():
    cfg=SimpleNamespace(objective='adaptive_topk',reuse_window=4,batch_size=1,micro_batch_size=1,
        distillation_topk=2,selector='full',keep_ratio=1.,block_len=2)
    records=[Rollout([1,2,3],1,[2,3],'','a',False,42)]
    w=rw.create(0,records,cfg,'binding')
    q=torch.log_softmax(torch.ones(1,2,2),-1)
    target=Target(torch.tensor([[[0,1],[0,1]]]),q,torch.ones(1,2),torch.full((1,2),.5))
    rw.append(w,target,torch.ones(1,2),torch.ones(1,2,dtype=torch.bool));rw.seal(w)
    return cfg,w


def test_roundtrip_boundary_binding_and_detached_cpu():
    c,w=fixture();assert rw.restore(w,1,c,'binding',0) is w
    assert rw.restore(w,4,c,'binding',0) is None
    target,_,_=rw.microbatch(w,0,'cpu');assert not target.logq.requires_grad
    assert rw.records(w)[0].response_ids==[2,3]
    with pytest.raises(ValueError,match='binding'):rw.restore(w,1,c,'other',0)
    with pytest.raises(ValueError,match='expiry'):rw.restore(w,5,c,'binding',0)
    assert rw.restore(None,4,c,'binding',0) is None
    with pytest.raises(ValueError,match='missing mid-window'):rw.restore(None,3,c,'binding',0)
    # A boundary snapshot is validated before being discarded, not silently ignored.
    w['microbatches'][0]['scores'].add_(1)
    with pytest.raises(ValueError,match='integrity'):rw.restore(w,4,c,'binding',0)


@pytest.mark.parametrize('mutation',['hash','expiry','mask','nonfinite','ids','shape'])
def test_corruption_and_structural_errors(mutation):
    c,w=fixture()
    if mutation=='expiry':w['expires_step']=99
    elif mutation=='mask':w['microbatches'][0]['selected'][0,0]=False
    elif mutation=='nonfinite':w['microbatches'][0]['target']['alpha'][0,0]=float('nan')
    elif mutation=='ids':w['microbatches'][0]['target']['ids'][0,0]=torch.tensor([1,1])
    elif mutation=='shape':w['microbatches'][0]['target']['alpha']=torch.zeros(1,1)
    else:w['microbatches'][0]['scores'].add_(1)
    if mutation!='hash':rw.seal(w) # even a recomputed checksum cannot bypass semantic validation
    with pytest.raises(ValueError):rw.restore(w,1,c,'binding',0)


@pytest.mark.parametrize('n',[0,-1,True,1.5])
def test_invalid_width(n):
    with pytest.raises(ValueError):rw.size(SimpleNamespace(objective='adaptive_topk',reuse_window=n))


@pytest.mark.parametrize('objective,expected',[('adaptive_topk',4),('mixed_kl',1),('sampled_k1',1)])
def test_cli_resolves_default_without_gpu(monkeypatch,tmp_path,objective,expected):
    from cq_opd import trainer
    class Stop(Exception):pass
    values=[]
    def capture(cfg):values.append(cfg.reuse_window);raise Stop
    monkeypatch.setattr(rw,'size',capture)
    monkeypatch.setattr('sys.argv',['trainer','--output',str(tmp_path/'run'),'--objective',objective])
    with pytest.raises(Stop):trainer.main()
    assert values==[expected]


@pytest.mark.parametrize('objective,n',[('sampled_k1',2),('mixed_kl',4),('adaptive_topk',0)])
def test_cli_refuses_invalid_reuse_before_models(monkeypatch,tmp_path,objective,n):
    from cq_opd import trainer
    monkeypatch.setattr('sys.argv',['trainer','--output',str(tmp_path/'run'),'--objective',objective,
        '--reuse-window',str(n)])
    with pytest.raises(ValueError,match='reuse-window'):trainer.main()
    assert not (tmp_path/'run').exists()


def test_other_objectives_only_width_one():
    assert rw.size(SimpleNamespace())==1
    for objective in ('mixed_kl','sampled_k1'):
        with pytest.raises(ValueError):rw.size(SimpleNamespace(objective=objective,reuse_window=2))
        assert rw.size(SimpleNamespace(objective=objective,reuse_window=1))==1

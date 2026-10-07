import json
import random
from types import SimpleNamespace
import pytest
import torch
from cq_opd import trainer,topk_trainer as topk
from cq_opd.rollout import Rollout
from test_cq_trainer import TinyAdapter,configuration,setup


@pytest.mark.parametrize('width',[2,4])
def test_replay_updates_refresh_and_exact_resume(monkeypatch,tmp_path,width):
    splits=setup(monkeypatch);calls=dict(sample=0,target=0,score=0,probe=0,gate=0,backward=0)
    def sample(student,tok,row,cap,seed):
        calls['sample']+=1
        return Rollout([1,2,3,4,5,6],2,[3,4,5,6],'#### 1',row['id'],False,seed)
    monkeypatch.setattr(trainer,'sample',sample)
    for owner,name,key in [(topk,'target','target'),(topk,'score','score'),(trainer,'direction','probe'),
                           (trainer,'check_delta_gate','gate'),(topk,'backward_hidden','backward')]:
        original=getattr(owner,name)
        def wrap(*a,_f=original,_k=key,**kw):calls[_k]+=1;return _f(*a,**kw)
        monkeypatch.setattr(owner,name,wrap)
    cfg=configuration();cfg.objective='adaptive_topk';cfg.distillation_topk=3
    cfg.reuse_window=width;cfg.steps=width+1;cfg.micro_batch_size=1
    s,t=TinyAdapter(1,True),TinyAdapter(2,False)
    cal=dict(mode='teacher_entropy',tau=1.,k=2.);delta=dict(cq_valid=True,delta=.01)
    trainer.train(s,t,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,cal,delta,{},random.Random(1))
    assert calls==dict(sample=4,target=4,score=4,probe=2,gate=2,backward=2*(width+1))
    metrics=[json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert [m['window_age'] for m in metrics]==list(range(width))+[0]
    assert [m['fresh_generation'] for m in metrics]==[True]+[False]*(width-1)+[True]
    assert metrics[1]['direction_source_step']==0 and metrics[1]['direction_age']==1
    assert metrics[1]['teacher_forward_time']==metrics[1]['scoring_time']==metrics[1]['train_rollout_time']==0
    assert metrics[0]['train_topk_mixed_kl']!=metrics[1]['train_topk_mixed_kl']
    for checkpoint_step in (1,width):
        before=dict(calls);out=tmp_path/f'resume{checkpoint_step}';out.mkdir()
        ck=trainer.load_checkpoint(tmp_path/f'checkpoint-{checkpoint_step}.pt')
        fresh=TinyAdapter(1,True)
        trainer.train(fresh,t,SimpleNamespace(pad_token_id=0),splits,cfg,out,cal,delta,{},random.Random(8),ck)
        torch.testing.assert_close(s.model.adapter.weight,fresh.model.adapter.weight,rtol=0,atol=0)
        assert calls['sample']-before['sample']==2
        assert calls['probe']-before['probe']==1 and calls['gate']-before['gate']==1
        resumed=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
        if checkpoint_step==1:
            assert resumed[0]['resume_fd_recheck_deferred'] and not resumed[-1]['resume_recheck_pending']
        final=trainer.load_checkpoint(out/f'checkpoint-{width+1}.pt')
        original=trainer.load_checkpoint(tmp_path/f'checkpoint-{width+1}.pt')
        assert final['python_rng']==original['python_rng'] and final['extra_rngs']==original['extra_rngs']
        assert final['train_cursor']==original['train_cursor']
        for key,state in final['optimizer']['state'].items():
            for name,value in state.items():
                if isinstance(value,torch.Tensor):torch.testing.assert_close(value,original['optimizer']['state'][key][name],rtol=0,atol=0)
                else:assert value==original['optimizer']['state'][key][name]
    bad=trainer.load_checkpoint(tmp_path/'checkpoint-1.pt');bad.pop('replay_window')
    out=tmp_path/'bad';out.mkdir()
    with pytest.raises(ValueError,match='missing mid-window'):
        trainer.train(TinyAdapter(1,True),t,SimpleNamespace(pad_token_id=0),splits,cfg,out,cal,delta,{},random.Random(0),bad)


def test_replay_microbatch_gradient_normalization(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    monkeypatch.setattr(trainer,'sample',lambda s,t,r,c,seed:Rollout([1,2,3,4,5,6],2,[3,4,5,6],'',r['id'],False,seed))
    results=[]
    for micro in (1,2):
        cfg=configuration();cfg.objective='adaptive_topk';cfg.distillation_topk=3
        cfg.reuse_window=2;cfg.micro_batch_size=micro
        s,t=TinyAdapter(1,True),TinyAdapter(2,False)
        out=tmp_path/str(micro);out.mkdir()
        trainer.train(s,t,SimpleNamespace(pad_token_id=0),splits,cfg,out,
            dict(mode='teacher_entropy',tau=1.,k=2.),dict(cq_valid=True,delta=.01),{},random.Random(0))
        results.append(s.model.adapter.weight.detach().clone())
    torch.testing.assert_close(*results,atol=1e-12,rtol=1e-12)

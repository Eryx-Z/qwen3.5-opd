"""Chunk fusion preserves Teacher entropy, signed utility, selection and VJP."""
from types import SimpleNamespace

import pytest
import torch

from cq_opd import trainer
from cq_opd.blocks import make_blocks, shifted_valid_mask
from cq_opd.losses import chunked_mixed_kl_hidden_grad
from cq_opd.probe import direction_from_gradients
from cq_opd.rollout import Rollout, build_batch
from cq_opd.selectors import select_blocks
from test_cq_trainer import TinyAdapter


@pytest.mark.parametrize('mode',['teacher_entropy','constant'])
def test_fused_entropy_utility_projection_saving_and_gradient_parity(monkeypatch,mode):
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    cfg=SimpleNamespace(chunk_size=2,block_len=2,keep_ratio=.5)
    calibration={'mode':mode,'constant_alpha':.3,'tau':1.5,'k':2.}
    batch=build_batch([Rollout([1,2,3,4,5,6],2,[3,4,5,6],'','a',False,42)],'cpu',0)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    blocks=make_blocks(valid,2)
    teacher_hidden=teacher.last_hidden(batch,with_grad=False)
    params=student.lora_named_parameters()
    direction=direction_from_gradients(params,[torch.ones_like(p) for _,p in params])
    project=teacher.project;rows=[]
    def counted(x):
        rows.append(x.shape[0]);return project(x)
    monkeypatch.setattr(teacher,'project',counted)
    meta=trainer.target_meta(teacher,teacher_hidden,valid,calibration,cfg)
    original=trainer.score(student,teacher,batch,teacher_hidden,meta.alpha,direction,.01,cfg)
    assert sum(rows)==2*int(valid.sum())
    rows.clear()
    fused,fused_meta=trainer.score(student,teacher,batch,teacher_hidden,None,direction,.01,cfg,
                                 calibration=calibration)
    assert sum(rows)==int(valid.sum())
    torch.testing.assert_close(fused,original,rtol=0,atol=0)
    torch.testing.assert_close(fused_meta.alpha,meta.alpha,rtol=0,atol=0)
    torch.testing.assert_close(fused_meta.entropy,meta.entropy,rtol=0,atol=0)
    selected=select_blocks(original,blocks,.5)
    assert torch.equal(select_blocks(fused,blocks,.5),selected)
    h=student.last_hidden(batch,with_grad=True)
    ref=chunked_mixed_kl_hidden_grad(h,teacher_hidden,meta.alpha,selected,student.project,project,2)
    result=chunked_mixed_kl_hidden_grad(h,teacher_hidden,fused_meta.alpha,selected,student.project,project,2)
    torch.testing.assert_close(result[0],ref[0],rtol=0,atol=0)
    torch.testing.assert_close(result[1],ref[1],rtol=0,atol=0)
    g_ref=torch.autograd.grad(h,[p for _,p in params],grad_outputs=ref[0],retain_graph=True)
    g_result=torch.autograd.grad(h,[p for _,p in params],grad_outputs=result[0])
    for a,b in zip(g_ref,g_result):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(p.grad is None for _,p in params)

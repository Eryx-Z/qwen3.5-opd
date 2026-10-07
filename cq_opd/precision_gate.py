"""Startup proof that higher-precision FD loss matches the actual update path."""
import torch
from .blocks import shifted_valid_mask
from .losses import chunked_mixed_kl_hidden_grad
from .model_adapter import temporary_scoring_precision


def validate_fd_update_consistency(student,teacher,batch,cfg):
    from .trainer import target_meta,fd_precision
    from .losses import calibrate_entropy
    dtype=fd_precision(cfg)
    if dtype is None:return dict(passed=True,same_precision=True)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    ht=teacher.last_hidden(batch,False)
    with torch.no_grad():
        rows,t=valid.nonzero(as_tuple=True)
        lq=torch.log_softmax(teacher.project(ht[rows,t]).float(),-1)
        calibration=calibrate_entropy(-(lq.exp()*lq).sum(-1))
    meta=target_meta(teacher,ht,valid,calibration,cfg)
    params=student.lora_named_parameters();results=[]
    for precision in (None,dtype):
        with temporary_scoring_precision(student,precision):
            hidden=student.last_hidden(batch,True)
            gh,loss,count=chunked_mixed_kl_hidden_grad(hidden,ht,meta.alpha,valid,
                student.project,teacher.project,chunk_size=cfg.chunk_size)
            gradients=torch.autograd.grad(hidden,[p for _,p in params],grad_outputs=gh,allow_unused=True)
            results.append((loss.detach().float(),[torch.zeros_like(p) if g is None else g.detach().clone()
                                                   for (_,p),g in zip(params,gradients)]))
            del hidden,gh,gradients
    torch.testing.assert_close(results[0][0],results[1][0],atol=.02,rtol=.02)
    a,b=results[0][1],results[1][1]
    aa=sum(g.double().square().sum() for g in a);bb=sum(g.double().square().sum() for g in b)
    dot=sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    if not torch.isfinite(aa+bb+dot) or aa<=0 or bb<=0:raise RuntimeError('precision consistency has no finite gradient')
    cosine=float(dot/(aa*bb).sqrt());relative=float((aa.sqrt()-bb.sqrt()).abs()/aa.sqrt())
    if cosine<.99 or relative>.05:raise RuntimeError('FD precision/update gradient mismatch')
    if any(p.grad is not None for _,p in params):raise RuntimeError('precision gate polluted training gradients')
    return dict(passed=True,same_precision=False,loss_native=float(results[0][0]),
                loss_fd=float(results[1][0]),gradient_cosine=cosine,gradient_norm_relative=relative,
                minimum_cosine=.99,maximum_norm_relative=.05,scope='short_input_numerical_consistency_not_FD_permission')

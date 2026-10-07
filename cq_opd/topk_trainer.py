"""Teacher-set conditional top-k mixed KL, not an unbiased full-KL estimator."""
from dataclasses import dataclass
import math
from numbers import Integral
import torch
from .blocks import shifted_valid_mask, make_blocks
from .losses import log_probs, entropy_alpha, kl_components, prediction_chunks, require_finite, calibrate_entropy
from .model_adapter import temporary_scoring_precision
from .utility import score_symmetric_offsets


def enabled(cfg):
    return getattr(cfg,'objective','mixed_kl')=='adaptive_topk'


@dataclass
class Target:
    ids: torch.Tensor
    logq: torch.Tensor
    entropy: torch.Tensor
    alpha: torch.Tensor


def validate_target(frozen,batch,cfg):
    """Validate reusable detached Teacher tensors on real response positions."""
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    k=cfg.distillation_topk
    if not isinstance(k,Integral) or isinstance(k,bool) or k<2:
        raise ValueError('distillation topk must be an integer at least 2')
    if not isinstance(frozen,Target):raise ValueError('expected topk Target')
    for name,shape in [('ids',(*valid.shape,k)),('logq',(*valid.shape,k)),
                       ('entropy',valid.shape),('alpha',valid.shape)]:
        x=getattr(frozen,name)
        if (not isinstance(x,torch.Tensor) or tuple(x.shape)!=tuple(shape) or
                x.device!=valid.device or x.requires_grad):
            raise ValueError('invalid frozen topk '+name)
        if name=='ids':
            if x.dtype!=torch.long or (x<0).any():raise ValueError('invalid topk IDs')
        else:
            if not x.is_floating_point():raise ValueError('invalid topk '+name+' dtype')
            require_finite(x,'frozen topk '+name)
    if ((frozen.alpha<0)|(frozen.alpha>1)).any():raise ValueError('invalid topk alpha')
    if valid.any():
        ids=frozen.ids[valid].sort(-1).values
        if (ids[:,1:]==ids[:,:-1]).any():raise ValueError('duplicate topk IDs')
        q=frozen.logq[valid]
        if not torch.allclose(q.logsumexp(-1),torch.zeros_like(q[:,0]),atol=1e-5,rtol=0):
            raise ValueError('topk Teacher probabilities not normalized')
        entropy=-(q.exp()*q).sum(-1)
        if not torch.allclose(entropy,frozen.entropy[valid],atol=1e-5,rtol=1e-5):
            raise ValueError('topk entropy inconsistent with Teacher probabilities')
    return frozen


@torch.no_grad()
def target(teacher,batch,calibration,cfg):
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    k=cfg.distillation_topk
    if not isinstance(k,Integral) or isinstance(k,bool) or k<2:
        raise ValueError('distillation topk must be an integer at least 2')
    h=teacher.last_hidden(batch,with_grad=False)
    ids=torch.zeros((*valid.shape,k),device=h.device,dtype=torch.long)
    dtype=torch.float64 if h.dtype==torch.float64 else torch.float32
    q=torch.zeros((*valid.shape,k),device=h.device,dtype=dtype)
    entropy=torch.zeros(valid.shape,device=h.device,dtype=dtype);alpha=torch.zeros_like(entropy)
    for b,t in prediction_chunks(h,valid,cfg.chunk_size):
        logits=teacher.project(h[b,t]);require_finite(logits,'Teacher topk logits')
        if not 2<=k<=logits.shape[-1]:raise ValueError('distillation topk must be between 2 and vocabulary size')
        values,indices=logits.topk(k,dim=-1)
        q[b,t]=log_probs(values);ids[b,t]=indices
        entropy[b,t],alpha[b,t]=entropy_alpha(q[b,t],calibration)
    return validate_target(Target(ids,q.detach(),entropy.detach(),alpha.detach()),batch,cfg)


def project_topk(student,rows,ids):
    """Gather frozen linear head rows before multiplication; normalize only on K."""
    if rows.ndim!=2 or ids.ndim!=2 or rows.shape[0]!=ids.shape[0] or ids.dtype!=torch.long:
        raise ValueError('expected rows [N,D] and long IDs [N,K]')
    if rows.device!=ids.device or (ids<0).any():raise ValueError('invalid topk ID device/range')
    head=getattr(student.model,'lm_head',None)
    if isinstance(head,torch.nn.Linear):
        if (head.weight.requires_grad or (head.bias is not None and head.bias.requires_grad)):
            raise ValueError('topk requires frozen Student output head')
        if (ids>=head.out_features).any():raise ValueError('topk ID exceeds vocabulary')
        dtype=torch.float64 if rows.dtype==torch.float64 else torch.float32
        w=head.weight[ids].to(dtype)
        out=torch.einsum('nd,nkd->nk',rows.to(dtype),w)
        if head.bias is not None:out=out+head.bias[ids].to(dtype)
        return out
    logits=student.project(rows)
    if (ids>=logits.shape[-1]).any():raise ValueError('topk ID exceeds vocabulary')
    return logits.gather(-1,ids)


def components(student,rows,ids,logq,alpha):
    if alpha.shape!=rows.shape[:1] or logq.shape!=ids.shape:
        raise ValueError('topk component shape mismatch')
    if not torch.isfinite(alpha).all() or ((alpha<0)|(alpha>1)).any():
        raise ValueError('invalid topk alpha')
    f,r=kl_components(project_topk(student,rows,ids),logq)
    loss=alpha.detach()*f+(1-alpha.detach())*r
    require_finite(loss,'topk mixed KL')
    return loss,f,r


@torch.no_grad()
def values(student,batch,frozen,cfg):
    validate_target(frozen,batch,cfg)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    h=student.last_hidden(batch,with_grad=False);out=torch.zeros_like(frozen.alpha)
    for b,t in prediction_chunks(h,valid,cfg.chunk_size):
        out[b,t]=components(student,h[b,t],frozen.ids[b,t],frozen.logq[b,t],frozen.alpha[b,t])[0]
    return out


def score(student,batch,frozen,d,delta,cfg):
    from .trainer import fd_precision
    with temporary_scoring_precision(student,fd_precision(cfg)):
        good,bad=score_symmetric_offsets(student.lora_named_parameters(),d,delta,
            lambda:values(student,batch,frozen,cfg))
    result=(bad-good)/(2*delta)
    require_finite(result,'topk FD utility')
    return result


@torch.no_grad()
def tip_selection(student,h,batch,frozen,cfg):
    """Per-response Soft-OR on Teacher-support probabilities; no backbone pass.

    Selection is detached. Unlike TIP's batch clipping, each response has its
    own 98th-percentile entropy cap/minmax so microbatch grouping is irrelevant.
    """
    validate_target(frozen,batch,cfg)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    if not math.isfinite(cfg.keep_ratio) or not 0<cfg.keep_ratio<=1:
        raise ValueError('TIP keep ratio must be in (0,1]')
    entropy=torch.zeros_like(frozen.alpha); divergence=torch.zeros_like(entropy)
    for b,t in prediction_chunks(h,valid,cfg.chunk_size):
        lp=log_probs(project_topk(student,h[b,t],frozen.ids[b,t]))
        entropy[b,t]=-(lp.exp()*lp).sum(-1)/math.log(cfg.distillation_topk)
        divergence[b,t]=(lp.exp()*(lp-frozen.logq[b,t])).sum(-1)
    require_finite(entropy,'TIP Student entropy');require_finite(divergence,'TIP divergence')
    scores=torch.zeros_like(entropy);selected=torch.zeros_like(valid)
    def normalize(x):
        span=x.max()-x.min()
        return (x-x.min())/span if span>0 else torch.zeros_like(x)
    for row in range(valid.shape[0]):
        positions=valid[row].nonzero(as_tuple=True)[0]
        if not positions.numel():continue
        e=entropy[row,positions];e=e.clamp_max(torch.quantile(e,.98))
        e=normalize(e);d=normalize(divergence[row,positions])
        values=e+d-e*d;scores[row,positions]=values
        count=max(1,math.floor(cfg.keep_ratio*positions.numel()))
        order=torch.argsort(values,descending=True,stable=True)
        selected[row,positions[order[:count]]]=True
    return scores.detach(),selected


def backward_hidden(student,h,batch,selected,frozen,cfg):
    validate_target(frozen,batch,cfg)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    if selected.shape!=valid.shape or selected.dtype!=torch.bool or selected.device!=valid.device:
        raise ValueError('invalid topk selection mask')
    if (selected & ~valid).any():raise ValueError('topk selection includes prompt/padding')
    grad=torch.zeros_like(h);sums=torch.zeros(3,device=h.device,dtype=frozen.logq.dtype)
    for b,t in prediction_chunks(h,selected,cfg.chunk_size):
        with torch.enable_grad():
            leaf=h[b,t].detach().requires_grad_(True)
            loss,f,r=components(student,leaf,frozen.ids[b,t],frozen.logq[b,t],frozen.alpha[b,t])
            dh=torch.autograd.grad(loss.sum(),leaf)[0]
        require_finite(dh,'topk hidden gradient');grad[b,t]=dh.detach()
        sums+=torch.stack([loss.sum(),f.sum(),r.sum()]).detach()
    require_finite(sums,'topk loss sums')
    return grad.detach(),sums[0].detach(),int(selected.sum()),sums[1].detach(),sums[2].detach()


def measure_fd(student,teacher,batch,calibration,cfg,d,candidates,*,frozen_target=None):
    from .trainer import fd_precision,block_values,rank_agreement
    frozen=target(teacher,batch,calibration,cfg) if frozen_target is None else frozen_target
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask']);blocks=make_blocks(valid,cfg.block_len)
    if not blocks:raise ValueError('no finite-difference audit blocks')
    reports=[]
    with temporary_scoring_precision(student,fd_precision(cfg)):
        noise=block_values((values(student,batch,frozen,cfg)-values(student,batch,frozen,cfg)).abs(),blocks)
        for rho,delta in candidates:
            if not math.isfinite(delta) or delta<=0:raise ValueError('invalid audit delta')
            vals=[block_values(score(student,batch,frozen,d,delta*f,cfg),blocks) for f in (.5,1.,2.)]
            sig=vals[1].abs()*2*delta>torch.maximum(noise*10,torch.full_like(noise,1e-7))
            significant=[block for block,keep in zip(blocks,sig) if keep];agreements=[]
            for v in (vals[0],vals[2]):
                corr,_=rank_agreement(vals[1][sig],v[sig],significant,cfg.keep_ratio)
                _,overlap=rank_agreement(vals[1],v,blocks,cfg.keep_ratio);agreements.append((corr,overlap))
            signs=[float((torch.sign(vals[1][sig])==torch.sign(v[sig])).float().mean()) if sig.any() else 0. for v in (vals[0],vals[2])]
            fraction=float(sig.float().mean())
            reports.append(dict(rho=rho,delta=delta,agreements=agreements,signal_fraction=fraction,sign_agreement=signs,
                passed=fraction>=.5 and all(c is not None and c>=.8 and o>=.7 for c,o in agreements) and min(signs)>=.8))
    return dict(candidates=reports,noise_max=float(noise.max()),objective='adaptive_topk')


def precision_gate(student,teacher,batch,cfg):
    from .trainer import fd_precision
    frozen=target(teacher,batch,{'mode':'constant','constant_alpha':.5},cfg)
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask']);params=student.lora_named_parameters();results=[]
    for dtype in (None,fd_precision(cfg)):
        with temporary_scoring_precision(student,dtype):
            h=student.last_hidden(batch,with_grad=True)
            gh,loss,n,_,_=backward_hidden(student,h,batch,valid,frozen,cfg)
            grads=torch.autograd.grad(h,[p for _,p in params],grad_outputs=gh,allow_unused=True)
            results.append((loss,[torch.zeros_like(p) if g is None else g.detach().clone() for (_,p),g in zip(params,grads)]))
    torch.testing.assert_close(results[0][0],results[1][0],atol=.02,rtol=.02)
    a,b=results[0][1],results[1][1];aa=sum(g.double().square().sum() for g in a);bb=sum(g.double().square().sum() for g in b)
    dot=sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    if not torch.isfinite(aa+bb+dot) or min(aa,bb)<=0:raise RuntimeError('topk precision gate has no finite gradient')
    cosine=float(dot/(aa*bb).sqrt());relative=float((aa.sqrt()-bb.sqrt()).abs()/aa.sqrt())
    if cosine<.99 or relative>.05:raise RuntimeError('FD precision/update gradient mismatch')
    if any(p.grad is not None for _,p in params):raise RuntimeError('precision gate polluted gradients')
    return dict(passed=True,objective='adaptive_topk',selected_tokens=n,loss_native=float(results[0][0]),loss_fd=float(results[1][0]),
        gradient_cosine=cosine,gradient_norm_relative=relative,gradient_norm=float(aa.sqrt()),minimum_cosine=.99,maximum_norm_relative=.05,
        scope='short_input_topk_numerical_consistency_not_FD_permission')

"""Bounded CPU-only replay snapshots; no Student graphs or optimizer state."""
from dataclasses import asdict
import hashlib
import json
from numbers import Integral
import torch
from .rollout import Rollout, build_batch
from .blocks import shifted_valid_mask, make_blocks
from .selectors import select_blocks
from .topk_trainer import Target


def size(cfg):
    value=getattr(cfg,'reuse_window',1)
    if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
        raise ValueError('reuse-window must be a positive integer')
    if value>1 and getattr(cfg,'objective','mixed_kl')!='adaptive_topk':
        raise ValueError('reuse-window >1 requires adaptive_topk')
    return int(value)


def binding(cfg,calibration,delta,metadata):
    config={k:v for k,v in vars(cfg).items() if k not in ('output','resume','calibration_dir','mode')}
    return hashlib.sha256(json.dumps(dict(config=config,calibration=calibration,delta=delta,
        metadata=metadata),sort_keys=True).encode()).hexdigest()


def _hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous()
            h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):
                if k!='sha256':h.update(k.encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for x in v:visit(x)
        else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    visit(value);return h.hexdigest()


def create(source,rollouts,cfg,bind):
    return dict(version=1,source_step=source,expires_step=source+size(cfg),binding=bind,
                rollouts=[asdict(r) for r in rollouts],microbatches=[])


def append(window,target,scores=None,selected=None):
    freeze=lambda t:t.detach().cpu().clone()
    window['microbatches'].append(dict(target={k:freeze(v) for k,v in vars(target).items()},
        scores=None if scores is None else freeze(scores),selected=None if selected is None else freeze(selected)))


def seal(window):
    window['sha256']=_hash(window)
    return window


def restore(window,step,cfg,bind,pad):
    """Validate published snapshot even at its boundary before discarding it."""
    width=size(cfg)
    if window is None:
        if step%width:raise ValueError('missing mid-window replay state')
        return None
    if window.get('version')!=1 or window.get('binding')!=bind or window.get('sha256')!=_hash(window):
        raise ValueError('replay window integrity/binding mismatch')
    source=window.get('source_step');expiry=window.get('expires_step')
    if (type(source)!=int or type(expiry)!=int or source<0 or source%width or
        expiry!=source+width or not source<step<=expiry):
        raise ValueError('invalid replay source/expiry')
    records=[Rollout(**r) for r in window['rollouts']]
    if any(not r.response_ids or len(r.response_ids)>getattr(cfg,'max_new_tokens',len(r.response_ids))
           for r in records):raise ValueError('invalid replay response budget')
    micro=getattr(cfg,'micro_batch_size',1)
    if len(records)!=cfg.batch_size or len(window['microbatches'])!=(len(records)+micro-1)//micro:
        raise ValueError('invalid replay batch count')
    for start,entry in zip(range(0,len(records),micro),window['microbatches']):
        batch=build_batch(records[start:start+micro],'cpu',pad)
        valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
        data=entry['target'];shape=(*valid.shape,cfg.distillation_topk)
        if set(data)!={'ids','logq','entropy','alpha'}:raise ValueError('invalid replay target fields')
        for name,tensor in data.items():
            expected=shape if name in ('ids','logq') else valid.shape
            if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or tensor.requires_grad or tuple(tensor.shape)!=tuple(expected):
                raise ValueError('invalid replay target '+name)
            if name=='ids':
                if tensor.dtype!=torch.long or (tensor<0).any():raise ValueError('invalid replay token IDs')
                if (tensor[valid].sort(-1).values.diff(dim=-1)==0).any():raise ValueError('duplicate replay topk IDs')
            elif not tensor.is_floating_point() or not torch.isfinite(tensor).all():raise ValueError('invalid replay target values')
        if ((data['alpha']<0)|(data['alpha']>1)).any() or (data['entropy']<0).any():raise ValueError('invalid replay entropy/alpha')
        if not torch.allclose(data['logq'][valid].logsumexp(-1),torch.zeros_like(data['alpha'][valid]),atol=1e-5,rtol=1e-5):
            raise ValueError('replay Teacher probabilities not normalized')
        scores=entry['scores'];selected=entry['selected']
        if cfg.selector=='tip':
            if scores is not None or selected is not None:
                raise ValueError('TIP replay must not contain stale selection')
            from .topk_trainer import validate_target
            validate_target(Target(**data),batch,cfg)
            continue
        if (not isinstance(scores,torch.Tensor) or scores.device.type!='cpu' or
            scores.shape!=valid.shape or not scores.is_floating_point() or
            not torch.isfinite(scores).all() or scores.requires_grad):
            raise ValueError('invalid replay scores')
        expected=valid if cfg.keep_ratio==1 or cfg.selector=='full' else select_blocks(scores,make_blocks(valid,cfg.block_len),cfg.keep_ratio)
        if (not isinstance(selected,torch.Tensor) or selected.device.type!='cpu' or
            selected.dtype!=torch.bool or selected.shape!=valid.shape or not torch.equal(selected,expected)):
            raise ValueError('replay selected mask mismatch')
    return None if step==expiry else window


def records(window):
    return [Rollout(**r) for r in window['rollouts']]


def microbatch(window,index,device):
    e=window['microbatches'][index]
    return (Target(**{k:v.to(device) for k,v in e['target'].items()}),
            None if e['scores'] is None else e['scores'].to(device),
            None if e['selected'] is None else e['selected'].to(device))

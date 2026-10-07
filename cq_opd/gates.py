"""Production mode parity and current-policy finite-difference safety gates."""
import hashlib
import json
import math
import random

import torch

from .blocks import make_blocks, shifted_valid_mask
from .losses import mixed_kl
from .model_adapter import check_logits_equivalence
from .rollout import Rollout


def validate_forward_modes(student, teacher, batch):
    # The helper compares all projected logits with native conditional logits,
    # then explicitly compares eval hidden states with scoring/train hidden states.
    proofs={name:check_logits_equivalence(adapter,batch) for name,adapter in
            [('student',student),('teacher',teacher)]}
    return dict(passed=True,atol=.02,rtol=.02,models=proofs)


def _teacher_target_signature(teacher):
    parameters=list(teacher.model.named_parameters())
    if any(p.requires_grad for _,p in parameters):
        raise ValueError('FD target caching requires a fully frozen Teacher')
    # Include buffers as well: quantized Teachers may keep projection scales there.
    tensors=parameters+list(teacher.model.named_buffers())
    return (id(teacher),id(teacher.model),tuple(
        (name,id(p),p._version,tuple(p.shape),str(p.dtype),str(p.device))
        for name,p in tensors))


def _fd_target_key(teacher,batch,rollouts,calibration,cfg):
    signature=_teacher_target_signature(teacher)
    binding=hashlib.sha256(json.dumps(
        dict(rollouts=rollouts,calibration=calibration,chunk_size=cfg.chunk_size),
        sort_keys=True,allow_nan=False).encode())
    # Bind exact input/masks, including dtype/device, not merely rollout IDs.
    for name in ('input_ids','attention_mask','response_mask'):
        tensor=batch[name].detach().contiguous()
        binding.update(json.dumps((name,tuple(tensor.shape),str(tensor.dtype),
                                   str(tensor.device))).encode())
        binding.update(tensor.cpu().view(torch.uint8).numpy().tobytes())
    return signature,binding.hexdigest()


def _validate_frozen_target(frozen_target,batch,valid):
    if not isinstance(frozen_target,tuple) or len(frozen_target)!=2:
        raise ValueError('expected frozen FD target (hidden, meta)')
    ht,meta=frozen_target
    if (not isinstance(ht,torch.Tensor) or ht.ndim!=3 or ht.shape[-1]<=0 or
        tuple(ht.shape[:2])!=tuple(batch['input_ids'].shape)):
        raise ValueError('invalid frozen FD Teacher hidden shape')
    for name,tensor,shape in [('hidden',ht,ht.shape),
            ('entropy',getattr(meta,'entropy',None),valid.shape),
            ('alpha',getattr(meta,'alpha',None),valid.shape)]:
        if (not isinstance(tensor,torch.Tensor) or tuple(tensor.shape)!=tuple(shape) or
            tensor.device!=batch['input_ids'].device or not tensor.is_floating_point() or
            tensor.requires_grad):
            raise ValueError('invalid frozen FD target '+name)
        if not torch.isfinite(tensor).all():
            raise FloatingPointError('nonfinite frozen FD target '+name)
    if ((meta.alpha<0)|(meta.alpha>1)).any():
        raise ValueError('frozen FD alpha outside [0,1]')
    return ht,meta


def measure_fd(student,teacher,batch,calibration,cfg,d,candidates,*,frozen_target=None):
    from . import topk_trainer
    if topk_trainer.enabled(cfg):
        return topk_trainer.measure_fd(student,teacher,batch,calibration,cfg,d,candidates,frozen_target=frozen_target)
    from .sampled_trainer import enabled, measure_fd as sampled_measure_fd
    if enabled(cfg):
        return sampled_measure_fd(student,teacher,batch,calibration,cfg,d,candidates,frozen_target=frozen_target)
    from .trainer import fd_precision
    from .model_adapter import temporary_scoring_precision
    with temporary_scoring_precision(student,fd_precision(cfg)):
        return _measure_fd(student,teacher,batch,calibration,cfg,d,candidates,frozen_target=frozen_target)


def _measure_fd(student,teacher,batch,calibration,cfg,d,candidates,*,frozen_target=None):
    """Same frozen Teacher/alpha and fixed trajectories for all delta variants."""
    from .trainer import target_meta, score, valid_rows, block_values, rank_agreement
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
    blocks=make_blocks(valid,cfg.block_len)
    if not blocks:
        raise ValueError('no finite-difference audit blocks')
    if frozen_target is None:
        ht=teacher.last_hidden(batch,with_grad=False)
        meta=target_meta(teacher,ht,valid,calibration,cfg)
    else:
        _teacher_target_signature(teacher)
        ht,meta=_validate_frozen_target(frozen_target,batch,valid)
    with torch.no_grad():
        h1=student.last_hidden(batch,with_grad=False)
        h2=student.last_hidden(batch,with_grad=False)
        noise=torch.zeros_like(meta.alpha)
        for b,t in valid_rows(valid,cfg.chunk_size):
            lq=torch.log_softmax(teacher.project(ht[b,t]).float(),-1)
            noise[b,t]=(mixed_kl(student.project(h1[b,t]),lq,meta.alpha[b,t])-
                        mixed_kl(student.project(h2[b,t]),lq,meta.alpha[b,t])).abs()
    noise_blocks=block_values(noise,blocks)
    del h1,h2
    reports=[]
    for rho,delta in candidates:
        if not math.isfinite(delta) or delta<=0:
            raise ValueError('invalid audit delta')
        values=[block_values(score(student,teacher,batch,ht,meta.alpha,d,delta*factor,cfg),blocks)
                for factor in (.5,1.,2.)]
        above_noise=values[1].abs()*2*delta>torch.maximum(noise_blocks*10,
                                                       torch.full_like(noise_blocks,1e-7))
        significant=[block for block,keep in zip(blocks,above_noise) if keep]
        agreements=[]
        for v in (values[0],values[2]):
            correlation,_=rank_agreement(values[1][above_noise],v[above_noise],significant,cfg.keep_ratio)
            _,overlap=rank_agreement(values[1],v,blocks,cfg.keep_ratio)
            agreements.append((correlation,overlap))
        signs=[float((torch.sign(values[1][above_noise])==torch.sign(v[above_noise])).float().mean())
               if above_noise.any() else 0. for v in (values[0],values[2])]
        passed=(float(above_noise.float().mean())>=.5 and
                all(c is not None and c>=.8 and o>=.7 for c,o in agreements) and min(signs)>=.8)
        reports.append(dict(rho=rho,delta=delta,agreements=agreements,
            signal_fraction=float(above_noise.float().mean()),sign_agreement=signs,passed=passed))
    return dict(candidates=reports,noise_max=float(noise_blocks.max()))


def recheck_due(step,first,state,stats,every):
    if not state or step==first or step<2 or step%every==0 or step-state['last_step']>=every:
        return True
    if abs(stats.get('probe_success_rate',0.)-state['probe_success_rate'])>=.25:
        return True
    a=stats.get('probe_grad_norm',0.);b=state['probe_grad_norm']
    return min(a,b)<=0 or max(a,b)>=2*min(a,b)


def check_delta_gate(student,teacher,tokenizer,dev,calibration,cfg,d,delta,state,*,cache=None):
    """Recheck the ACCEPTED delta only; never silently reselect a new one.

    Audit trajectories are generated once from an independent stream, retained
    verbatim in checkpoints and reused after updates/resume. No train/probe or
    selector RNG stream is consumed here. Optional cache is caller-owned and
    process-local: only one frozen Teacher hidden/meta pair is retained, never
    logits, Student results, directions, or tensors in checkpoint state.
    """
    from .trainer import sample_many, batch_of
    if not d.valid or not delta.get('cq_valid'):
        raise RuntimeError('FD recheck requires valid probe and calibrated delta')
    state=dict(state or {})
    if 'rollouts' not in state:
        rng=random.Random(cfg.seed+5001)
        rollouts=sample_many(student,tokenizer,rng.sample(dev,cfg.delta_questions),cfg,rng)
        state['rollouts']=[{key:getattr(r,key) for key in Rollout.__dataclass_fields__} for r in rollouts]
    rollouts=[Rollout(**r) for r in state['rollouts']]
    batch=batch_of(rollouts,student,tokenizer)
    frozen_target=None
    if cache is not None:
        from .trainer import target_meta
        key=_fd_target_key(teacher,batch,state['rollouts'],calibration,cfg)
        entry=cache.get('fd_teacher_target')
        if entry is not None and entry['teacher'] is teacher and entry['key']==key:
            frozen_target=entry['target']
        else:
            from .sampled_trainer import enabled
            from . import topk_trainer
            if topk_trainer.enabled(cfg):
                frozen_target=topk_trainer.target(teacher,batch,calibration,cfg)
            elif enabled(cfg):
                from .sampled_loss import token_log_probs
                frozen_target=token_log_probs(teacher,batch,cfg.chunk_size)
            else:
                ht=teacher.last_hidden(batch,with_grad=False)
                valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
                frozen_target=(ht,target_meta(teacher,ht,valid,calibration,cfg))
    report=measure_fd(student,teacher,batch,calibration,cfg,d,
                      [(None,delta['delta'])],frozen_target=frozen_target)
    if cache is not None:
        # Keeping the owner alive prevents Python object-ID reuse. Replacing a
        # single entry bounds hidden-state memory even if calibration changes.
        cache['fd_teacher_target']=dict(key=key,teacher=teacher,target=frozen_target)
    if not report['candidates'][0]['passed']:
        raise RuntimeError('current-policy finite difference recheck failed; CQ update refused: '+
                           json.dumps(report,allow_nan=False))
    state['report']=report
    return state

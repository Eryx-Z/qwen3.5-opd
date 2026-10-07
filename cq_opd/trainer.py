"""Snapshot-level CQ training. GPU entry point is always run under the external guard."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .blocks import make_blocks, shifted_valid_mask
from .losses import (calibrate_entropy, chunked_mixed_kl_hidden_grad,
                     mixed_kl, mixed_utility_from_logits, prepare_entropy_alpha, entropy_alpha, TargetMeta)
from .model_adapter import load_student, load_teacher, validate_alignment
from .probe import (chunked_probe_gradients, leave_one_out_advantages,
                    capture_initial_probe, consume_initial_probe)
from .rollout import build_batch, generate, generate_batch
from .selectors import select_blocks
from .utility import score_symmetric_offsets
from .gates import validate_forward_modes, check_delta_gate, recheck_due
from . import sampled_trainer as sampled
from . import topk_trainer as topk
from . import replay_window


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clock(adapter):
    if next(adapter.model.parameters()).is_cuda:
        torch.cuda.synchronize()
    return time.monotonic()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_calibration_job(path,calibration,delta,metadata):
    try:
        job=json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ValueError('calibration job timing missing; regenerate calibration') from exc
    cost=job['total_elapsed_time']
    if (job.get('metadata_hash')!=digest(metadata) or
        job.get('entropy_calibration_hash')!=digest(calibration) or
        job.get('delta_calibration_hash')!=digest(delta) or
        job.get('includes_model_load_and_mode_gate') is not True or
        not math.isfinite(cost) or cost<calibration.get('elapsed_time',0.)+delta.get('elapsed_time',0.)):
        raise ValueError('calibration job timing/binding mismatch')
    return cost


def load_splits(root):
    import pandas as pd
    splits, seen, all_ids = {}, set(), set()
    for name in ('train', 'probe', 'dev', 'test'):
        rows = pd.read_parquet(root / f'{name}.parquet').to_dict('records')
        for row in rows:
            info = row['extra_info']
            if info['id'] in all_ids or info['question_hash'] in seen:
                raise ValueError('dataset ID/content overlap')
            all_ids.add(info['id']); seen.add(info['question_hash'])
        splits[name] = rows
    return splits


def prompt_ids(tokenizer, row):
    # Reference is intentionally never read here.
    ids = tokenizer.apply_chat_template(list(row['prompt']), tokenize=True,
                                       add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(ids,list) or not ids or not all(isinstance(token,int) for token in ids):
        raise TypeError('chat template must return a nonempty list of token IDs')
    if len(ids) > 512:
        raise ValueError('prompt exceeds fixed 512-token budget')
    return ids


def sample(student, tokenizer, row, cap, seed):
    accelerated=getattr(student,'rollout_engine',None)
    if accelerated is not None:
        return accelerated.generate([prompt_ids(tokenizer,row)],cap,seed,[row['extra_info']['id']])[0]
    return generate(student, tokenizer, prompt_ids(tokenizer, row), cap, seed,
                    id=row['extra_info']['id'])


def sample_many(student,tokenizer,rows,cfg,rng):
    """Native batched sampling; batch seed + row provenance, independent RNG stream."""
    accelerated=getattr(student,'rollout_engine',None)
    if accelerated is not None:
        return accelerated.generate([prompt_ids(tokenizer,row) for row in rows],cfg.max_new_tokens,
                    rng.randrange(2**31),[row['extra_info']['id'] for row in rows])
    result=[];size=getattr(cfg,'rollout_batch_size',1)
    for start in range(0,len(rows),size):
        group=rows[start:start+size]
        if size==1:  # legacy API/test compatibility
            result.append(sample(student,tokenizer,group[0],cfg.max_new_tokens,rng.randrange(2**31)))
        else:
            result.extend(generate_batch(student,tokenizer,[prompt_ids(tokenizer,row) for row in group],
                cfg.max_new_tokens,rng.randrange(2**31),ids=[row['extra_info']['id'] for row in group]))
    return result


def trajectory_records(rollouts):
    return [dict(id=r.id,seed=r.seed,sample_index=getattr(r,'sample_index',0),
                 generation_batch_size=getattr(r,'generation_batch_size',1),
                 token_ids_sha256=digest(r.input_ids),response_tokens=len(r.response_ids),truncated=r.truncated,
                 policy_sha256=getattr(r,'policy_sha256',None))
            for r in rollouts]


def batch_of(rollouts, student, tokenizer):
    device = next(student.model.parameters()).device
    return build_batch(rollouts, device, tokenizer.pad_token_id)


def direction(student, tokenizer, rows, cfg, step, rng):
    from baseline.verifier import compute_score
    params = student.lora_named_parameters()
    accumulated = [torch.zeros_like(p, dtype=torch.float32) for _, p in params]
    rewards, trajectories = [], []
    start = time.monotonic()
    questions=rng.sample(rows,cfg.probe_questions)
    trajectories=sample_many(student,tokenizer,[row for row in questions for _ in range(cfg.probe_answers)],cfg,rng)
    for index,row in enumerate(questions):
        group=trajectories[index*cfg.probe_answers:(index+1)*cfg.probe_answers]
        rewards.append([compute_score(row['data_source'],r.response_text,
                        row['reward_model']['ground_truth']) for r in group])
    generation_time = time.monotonic() - start
    diagnostic_started=time.monotonic()
    probe_diagnostics=[]
    if sampled.enabled(cfg) and getattr(student,'rollout_engine',None) is not None:
        from .sampled_loss import token_log_probs
        for offset in range(0,len(trajectories),getattr(cfg,'micro_batch_size',1)):
            group=trajectories[offset:offset+getattr(cfg,'micro_batch_size',1)]
            lp=token_log_probs(student,batch_of(group,student,tokenizer),cfg.chunk_size)
            probe_diagnostics.append(sampled.rollout_diagnostic(group,lp,expected_hash=student.rollout_engine.policy_hash))
    diagnostic_time=time.monotonic()-diagnostic_started
    advantages = leave_one_out_advantages(torch.tensor(rewards, dtype=torch.float32))
    start = time.monotonic()
    unused = set()
    for rollout, advantage in zip(trajectories, advantages.flatten()):
        if advantage == 0:
            continue
        batch = batch_of([rollout], student, tokenizer)
        valid = shifted_valid_mask(batch['response_mask'], batch['attention_mask'])
        hidden = student.last_hidden(batch, with_grad=True)
        from .model_adapter import gradient_projector
        with gradient_projector(student) as project:
            gradients = chunked_probe_gradients(hidden, batch['input_ids'][:, 1:], valid,
                project, params, float(advantage) / advantages.numel(), chunk_size=cfg.chunk_size)
        for (name, _), total, grad in zip(params, accumulated, gradients):
            if grad is None:
                unused.add(name)
            else:
                total.add_(grad)
        del hidden, gradients, batch
    if not torch.stack([torch.isfinite(g).all() for g in accumulated]).all():
        raise FloatingPointError('nonfinite probe gradient')
    from .probe import direction_from_gradients
    result = direction_from_gradients(params, accumulated, source_step=step, unused_names=sorted(unused))
    n = result.grad_norm
    stats = dict(probe_success_rate=float(torch.tensor(rewards).float().mean()),
        informative_question_fraction=float((advantages.abs().sum(-1) > 0).float().mean()),
        probe_grad_norm=n, probe_generation_time=generation_time,
        probe_rewards=rewards,probe_response_lengths=[len(r.response_ids) for r in trajectories],
        probe_truncated_fraction=sum(r.truncated for r in trajectories)/len(trajectories),
        probe_gradient_time=time.monotonic()-start, unused_probe_parameters=sorted(unused),
        probe_trajectories=[dict(id=r.id,seed=r.seed,sample_index=getattr(r,'sample_index',0),
            generation_batch_size=getattr(r,'generation_batch_size',1),token_ids_sha256=digest(r.input_ids))
            for r in trajectories])
    if probe_diagnostics:
        stats['probe_rollout_probability_diagnostic']=sampled.merge_diagnostics(probe_diagnostics)
        stats['probe_probability_diagnostic_time']=diagnostic_time
    return result, stats


def valid_rows(mask, chunk_size):
    rows = mask.nonzero(as_tuple=False)
    for chunk in rows.split(chunk_size):
        yield chunk[:, 0], chunk[:, 1]


def target_meta(teacher, hidden, valid, calibration, cfg):
    return prepare_entropy_alpha(hidden, valid, calibration, teacher.project,
                                 chunk_size=cfg.chunk_size)


def fd_precision(cfg):
    return torch.float32 if getattr(cfg,'fd_scoring_dtype','native')=='fp32' else None


def score(student,teacher,batch,teacher_hidden,alpha,d,delta,cfg,*,calibration=None):
    from .model_adapter import temporary_scoring_precision
    with temporary_scoring_precision(student,fd_precision(cfg)):
        return _score(student,teacher,batch,teacher_hidden,alpha,d,delta,cfg,calibration=calibration)


def _score(student, teacher, batch, teacher_hidden, alpha, d, delta, cfg, *, calibration=None):
    valid = shifted_valid_mask(batch['response_mask'], batch['attention_mask'])
    good, bad = score_symmetric_offsets(student.lora_named_parameters(), d, delta,
        lambda: student.last_hidden(batch, with_grad=False).detach())
    fused_meta=alpha is None
    if fused_meta:
        if calibration is None:
            raise ValueError('fused scoring requires entropy calibration')
        dtype=torch.float64 if teacher_hidden.dtype==torch.float64 else torch.float32
        entropy=torch.zeros(valid.shape,device=teacher_hidden.device,dtype=dtype)
        alpha=torch.zeros_like(entropy)
    utility = torch.zeros_like(alpha)
    with torch.no_grad():
        for b, t in valid_rows(valid, cfg.chunk_size):
            from .losses import log_probs
            lq = log_probs(teacher.project(teacher_hidden[b, t]))
            if fused_meta:
                entropy[b,t],alpha[b,t]=entropy_alpha(lq,calibration)
            utility[b, t] = mixed_utility_from_logits(student.project(good[b, t]),
                student.project(bad[b, t]), lq, alpha[b, t], delta)
    if not torch.isfinite(utility).all():
        raise FloatingPointError('nonfinite utility')
    return (utility,TargetMeta(entropy,alpha)) if fused_meta else utility


def entropy_protocol(cfg):
    return dict(objective=getattr(cfg,'objective','mixed_kl'),distillation_topk=getattr(cfg,'distillation_topk',None),
                entropy_scope='teacher_topk_conditional' if topk.enabled(cfg) else 'full_or_unused',
                max_new_tokens=cfg.max_new_tokens, entropy_questions=cfg.entropy_questions,
                seed=cfg.seed, trajectory_rng_seed=cfg.seed+7001,
                thinking=False,temperature=1.,top_p=1.,top_k=0,
                rollout_batch_size=getattr(cfg,'rollout_batch_size',1),
                rollout_backend=getattr(cfg,'rollout_backend','native'),
                sampling_rng='vllm_per_request_seed' if getattr(cfg,'rollout_backend','native')=='vllm' else 'native_batch_seed_plus_row')


def scoring_protocol(cfg):
    return dict(selection='tip_soft_or_teacher_support_per_response_v1' if getattr(cfg,'selector','cq')=='tip' else getattr(cfg,'selector','cq'),
                selection_refresh='each_update' if getattr(cfg,'selector','cq')=='tip' else 'window_source_policy',
                reuse_window=replay_window.size(cfg),cq_refresh='window_source_policy',objective=getattr(cfg,'objective','mixed_kl'),distillation_topk=getattr(cfg,'distillation_topk',None),
        normalization='both_on_teacher_topk' if topk.enabled(cfg) else 'full',max_new_tokens=cfg.max_new_tokens, block_len=cfg.block_len,
        keep_ratio=cfg.keep_ratio, chunk_size=cfg.chunk_size,
        probe_questions=cfg.probe_questions, probe_answers=cfg.probe_answers,
        seed=cfg.seed, scoring_dtype=f"{getattr(cfg,'student_dtype','bf16')}_base_fp32_lora_fp32_logsoftmax",
        fd_scoring_dtype=getattr(cfg,'fd_scoring_dtype','native'),rollout_backend=getattr(cfg,'rollout_backend','native'),
        sampled_probability_contract='native_recompute_old_logp_no_rollout_IS_diagnostic_long_parity' if sampled.enabled(cfg) else None,
        delta_choice_rule='fp32_interior_margin_v1' if fd_precision(cfg) is not None or getattr(cfg,'student_dtype','bf16')=='fp32' else 'engineering_minimum',
        calibration_probe_rng_seed=cfg.seed+2001,calibration_trajectory_rng_seed=cfg.seed+6001,
        audit_trajectory_rng_seed=cfg.seed+5001,
        rollout_batch_size=getattr(cfg,'rollout_batch_size',1),
        sampling_rng='vllm_per_request_seed' if getattr(cfg,'rollout_backend','native')=='vllm' else 'native_batch_seed_plus_row')


def entropy_calibration(student, teacher, tokenizer, rows, cfg, metadata, rng):
    values, positions = [], []
    started = time.monotonic()
    entropy_rollouts=sample_many(student,tokenizer,rng.sample(rows,cfg.entropy_questions),cfg,rng)
    for rollout in entropy_rollouts:
        batch = batch_of([rollout], student, tokenizer)
        valid = shifted_valid_mask(batch['response_mask'], batch['attention_mask'])
        hidden = teacher.last_hidden(batch, with_grad=False)
        with torch.no_grad():
            for b, t in valid_rows(valid, cfg.chunk_size):
                logits=teacher.project(hidden[b,t]).float()
                if topk.enabled(cfg):logits=logits.topk(cfg.distillation_topk,dim=-1).values
                lq = torch.log_softmax(logits, -1)
                values.extend((-(lq.exp()*lq).sum(-1)).cpu().tolist())
                positions.extend((rollout.id, int(pos)) for pos in t.cpu().tolist())
        del hidden, batch
    if not values:
        raise ValueError('no generated calibration positions')
    chosen = sorted(rng.sample(range(len(values)), min(8192, len(values))))
    calibration = calibrate_entropy(torch.tensor([values[i] for i in chosen]))
    calibration.update(mode='teacher_entropy', metadata_hash=digest(metadata), seed=cfg.seed,
        positions=[positions[i] for i in chosen],trajectories=trajectory_records(entropy_rollouts),
        protocol=entropy_protocol(cfg),
        elapsed_time=time.monotonic()-started)
    return calibration


def block_values(utility, blocks):
    return torch.stack([utility[b.response_index, list(b.positions)].mean() for b in blocks]).double().cpu()


def rank_agreement(a, b, blocks, ratio):
    # Average ranks handle ties; an all-tied array is explicitly inconclusive.
    def ranks(x):
        _,inverse,counts=torch.unique(x.double(),sorted=True,return_inverse=True,return_counts=True)
        end=counts.cumsum(0).double()
        return (end-(counts.double()-1)/2)[inverse]
    correlation = None
    if a.numel() >= 5 and torch.unique(a).numel() >= 3 and torch.unique(b).numel() >= 3:
        aa=ranks(a);bb=ranks(b)
        aa-=aa.mean();bb-=bb.mean()
        correlation=float((aa*bb).sum()/torch.sqrt(aa.square().sum()*bb.square().sum()))
    overlap, count = 0, 0
    responses = sorted({block.response_index for block in blocks})
    for response in responses:
        indices = [i for i, block in enumerate(blocks) if block.response_index == response]
        k = max(1, math.ceil(len(indices)*ratio))
        aa = sorted(indices, key=lambda i: (-float(a[i]), i))[:k]
        bb = sorted(indices, key=lambda i: (-float(b[i]), i))[:k]
        overlap += len(set(aa) & set(bb)); count += k
    return (correlation if correlation is not None and math.isfinite(correlation) else None,
            overlap/count if count else 0.)


def choose_startup_delta(candidates, *, precision='bf16'):
    """Choose a stable interior point, not a just-passing FP32 rounding edge.

    Recheck gates stay at the original .8/.7/.8 thresholds. FP32 startup
    selection requires extra numerical headroom (.95/.9/.9) on BOTH offsets;
    no candidate or selector is substituted when this stricter test fails.
    """
    for candidate in candidates:
        robust=bool(candidate['passed'])
        if precision=='fp32':
            robust=robust and all(c is not None and c>=.95 and o>=.9
                                  for c,o in candidate['agreements'])
            robust=robust and min(candidate['sign_agreement'])>=.9
        candidate['startup_robust']=robust
    return next((c['delta'] for c in candidates if c['startup_robust']),None)


def initial_probe_key(student,probe,cfg):
    # The direction depends on the actual probe content/rewards, frozen Student
    # and compute backend, not only on LoRA values (which are cloned separately).
    data=[dict(id=row['extra_info']['id'],prompt=list(row['prompt']),
               reward=row['reward_model'],source=row['data_source']) for row in probe]
    frozen=[(name,id(p),p._version,str(p.dtype),str(p.device),tuple(p.shape))
            for name,p in student.model.named_parameters() if not p.requires_grad]
    buffers=[(name,id(p),p._version,str(p.dtype),str(p.device),tuple(p.shape))
             for name,p in student.model.named_buffers()]
    return (scoring_protocol(cfg),digest(data),frozen,buffers,
            torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32)


def delta_calibration(student, teacher, tokenizer, dev, probe, calibration, cfg, rng, *, failure_output=None,
                      probe_rng=None,initial_probe_cache=None):
    started=time.monotonic()
    probe_rng=probe_rng if probe_rng is not None else rng
    rng_before=probe_rng.getstate()
    d, probe_stats = direction(student, tokenizer, probe, cfg, 0, probe_rng)
    if not d.valid:
        if failure_output is not None:
            write_json(Path(failure_output),dict(reason=d.reason,probe=probe_stats,
                protocol=scoring_protocol(cfg),metadata_hash=calibration.get('metadata_hash'),
                entropy_calibration_hash=digest(calibration)))
        raise RuntimeError('delta calibration has no probe signal; CQ training refused; inspect probe rewards and truncation')
    if initial_probe_cache is not None:
        initial_probe_cache.update(capture_initial_probe(student.lora_named_parameters(),d,probe_stats,
            initial_probe_key(student,probe,cfg),rng_before,probe_rng.getstate()))
    trajectories=sample_many(student,tokenizer,rng.sample(dev,cfg.delta_questions),cfg,rng)
    batch = batch_of(trajectories, student, tokenizer)
    from .gates import measure_fd
    from .utility import parameter_norm
    norm = parameter_norm(student.lora_named_parameters())
    if norm == 0 or not math.isfinite(norm):
        raise ValueError('invalid initial LoRA norm')
    report=measure_fd(student,teacher,batch,calibration,cfg,d,
        [(rho,rho*norm) for rho in (1e-5,3e-5,1e-4,3e-4,1e-3,3e-3)])
    candidates=report['candidates']
    selected_delta=choose_startup_delta(candidates,precision='fp32' if fd_precision(cfg) is not None else getattr(cfg,'student_dtype','bf16'))
    return dict(delta=selected_delta, cq_valid=selected_delta is not None,
        candidates=candidates, noise_max=report['noise_max'], lora_norm=norm,
        trajectory_ids=[r.id for r in trajectories],
        trajectories=[dict(id=r.id,seed=r.seed,token_ids_sha256=digest(r.input_ids)) for r in trajectories],
        probe=probe_stats,
        entropy_calibration_hash=digest(calibration), protocol=scoring_protocol(cfg),
        elapsed_time=time.monotonic()-started)


def training_contract(cfg):
    return {k:v for k,v in vars(cfg).items() if k not in ('output','resume','calibration_dir','mode')}


def save_checkpoint(path, student, optimizer, step, cfg, calibration, delta, rng, metadata, order, cursor, extra_rngs,
                    elapsed_clock, delta_recheck_state, replay_state=None):
    import uuid
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    if path.exists():
        raise FileExistsError('refusing to overwrite a published checkpoint')
    checkpoint_id=uuid.uuid4().hex
    torch.save(dict(checkpoint_id=checkpoint_id,cumulative_elapsed_time=elapsed_clock(),
        delta_recheck_state=delta_recheck_state,replay_window=replay_state,
        lora={name: p.detach().cpu() for name,p in student.lora_named_parameters()},
        optimizer=optimizer.state_dict(), step=step, config=training_contract(cfg), metadata=metadata,
        entropy_calibration=calibration, delta_calibration=delta, python_rng=rng.getstate(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), train_order=order, train_cursor=cursor,
        extra_rngs={name:r.getstate() for name,r in extra_rngs.items()},
        direction_source_step=replay_state['source_step'] if replay_state else step,
        scheduler='constant'), temporary)
    # The .pt is the commit marker. Prepare both files and publish its matching
    # sidecar FIRST, so interruption cannot expose a .pt without required timing.
    # Checkpoint paths are immutable; never replace an existing published pair.
    timing=path.with_suffix(path.suffix+'.timing.json')
    tmp_timing=timing.with_suffix(timing.suffix+'.tmp')
    write_json(tmp_timing,dict(checkpoint_id=checkpoint_id,cumulative_elapsed_time=elapsed_clock()))
    tmp_timing.replace(timing)
    temporary.replace(path)


def load_checkpoint(path):
    path=Path(path)
    state=torch.load(path,map_location='cpu',weights_only=False)  # trusted-only inputs
    timing=json.loads(path.with_suffix(path.suffix+'.timing.json').read_text())
    elapsed=timing['cumulative_elapsed_time']
    if (timing['checkpoint_id']!=state['checkpoint_id'] or not math.isfinite(elapsed) or
        elapsed<state['cumulative_elapsed_time']):
        raise ValueError('checkpoint timing sidecar mismatch')
    state['cumulative_elapsed_time']=elapsed
    return state


def train(student, teacher, tokenizer, splits, cfg, output, calibration, delta, metadata, rng, resume=None,
          startup_elapsed_time=0., calibration_is_imported=True, mode_check=None, imported_calibration_cost=None,
          initial_probe_cache=None):
    run_started=time.monotonic()
    reuse_width=replay_window.size(cfg)
    window_binding=replay_window.binding(cfg,calibration,delta,metadata)
    window=None
    parameters = student.lora_named_parameters()
    optimizer = torch.optim.AdamW([p for _,p in parameters], lr=cfg.lr,
        weight_decay=cfg.weight_decay,betas=(.9,.999),eps=1e-8,foreach=False)
    # Keep train question order and sampling streams independent of selector/probe overhead.
    rng=random.Random(cfg.seed+1001)
    extra_rngs={name:random.Random(cfg.seed+offset) for name,offset in
                [('probe',2001),('selector',3001),('data',4001)]}
    order = list(range(len(splits['train']))); extra_rngs['data'].shuffle(order); cursor=0; first=0
    recheck_state=None
    audit_target_cache={}  # small frozen Teacher tensors; reconstructed on resume
    resume_recheck_pending=bool(resume) and cfg.selector=='cq' and cfg.keep_ratio<1
    previous_elapsed=0.
    if resume:
        previous_elapsed=resume['cumulative_elapsed_time']
        if not math.isfinite(previous_elapsed) or previous_elapsed<0:
            raise ValueError('invalid saved cumulative time')
        recheck_state=resume['delta_recheck_state']
        if resume['metadata'] != metadata or resume['config'] != training_contract(cfg):
            raise ValueError('checkpoint config/model metadata mismatch')
        with torch.no_grad():
            for name,p in parameters:
                p.copy_(resume['lora'][name].to(p.device))
        optimizer.load_state_dict(resume['optimizer']); first=resume['step']
        if reuse_width>1:
            window=replay_window.restore(resume.get('replay_window'),first,cfg,window_binding,tokenizer.pad_token_id)
        rng.setstate(resume['python_rng']); torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state_all(resume['cuda_rng']); np.random.set_state(resume['numpy_rng'])
        order=resume['train_order']; cursor=resume['train_cursor']
        for name,state in resume['extra_rngs'].items():
            extra_rngs[name].setstate(state)
        if mode_check is not None:
            mode_check()  # actual resumed LoRA weights, not just the initial model
    calibration_cost=calibration.get('elapsed_time',0.)+delta.get('elapsed_time',0.)
    full_calibration_cost=calibration_cost if imported_calibration_cost is None else imported_calibration_cost
    if not math.isfinite(full_calibration_cost) or full_calibration_cost<0:
        raise ValueError('invalid imported calibration job cost')
    imported_cost=full_calibration_cost if calibration_is_imported and not resume else 0.
    def elapsed():
        return previous_elapsed+startup_elapsed_time+imported_cost+time.monotonic()-run_started
    checkpoint_time=0.
    for step in range(first, cfg.steps):
        if next(student.model.parameters()).is_cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic(); timings = dict(train_rollout_time=0., teacher_forward_time=0.,
            scoring_time=0., train_backward_time=0., optimizer_time=0.,fd_recheck_time=0.)
        cq = cfg.selector == 'cq' and cfg.keep_ratio < 1
        if window is not None and step>=window['expires_step']:
            if step!=window['expires_step']:raise ValueError('expired replay window')
            window=None
        fresh=window is None
        source_step=step if fresh else window['source_step']
        reused=None
        if fresh and cq and step==0 and not resume and initial_probe_cache:
            reused=consume_initial_probe(initial_probe_cache,parameters,
                initial_probe_key(student,splits['probe'],cfg),extra_rngs['probe'])
            initial_probe_cache.clear()  # never carry a direction beyond version0
        d,stats=(reused if reused is not None else
                 direction(student,tokenizer,splits['probe'],cfg,step,extra_rngs['probe'])) if cq and fresh else (None,{})
        stats.setdefault('initial_probe_reused',False)
        if fresh and (sampled.enabled(cfg) or topk.enabled(cfg)) and cq and not d.valid:
            raise RuntimeError('sampled CQ update refused: no probe signal')
        if fresh and cq and d.valid and (reuse_width>1 or resume_recheck_pending or
                recheck_due(step,first,recheck_state,stats,getattr(cfg,'delta_recheck_every',5))):
            t=clock(student)
            recheck_state=check_delta_gate(student,teacher,tokenizer,splits['dev'],calibration,cfg,d,delta,recheck_state,
                                             cache=audit_target_cache)
            recheck_state.update(last_step=step,probe_success_rate=stats.get('probe_success_rate',0.),
                                 probe_grad_norm=stats.get('probe_grad_norm',d.grad_norm))
            timings['fd_recheck_time']=clock(student)-t
            resume_recheck_pending=False
        optimizer.zero_grad(set_to_none=True)
        total = 0; available = 0; truncated=0; numerator=0.; fsum=0.; rsum=0.
        entropy_values=[]; alpha_values=[]; selected_alpha=[]; utility_values=[]; selected_util=[]
        rollout_diagnostics=[]
        micro_size=getattr(cfg,'micro_batch_size',1)
        if fresh:
            question_batch=[]
            for _ in range(cfg.batch_size):
                if cursor==len(order):
                    extra_rngs['data'].shuffle(order);cursor=0
                question_batch.append(splits['train'][order[cursor]]);cursor+=1
            t=clock(student);rollout_batch=sample_many(student,tokenizer,question_batch,cfg,rng)
            timings['train_rollout_time']+=clock(student)-t
            if reuse_width>1:window=replay_window.create(step,rollout_batch,cfg,window_binding)
        else:
            rollout_batch=replay_window.records(window)
        truncated=sum(r.truncated for r in rollout_batch)
        for start in range(0,cfg.batch_size,micro_size):
            rollouts=rollout_batch[start:start+micro_size]
            batch=batch_of(rollouts,student,tokenizer)
            valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
            blocks=make_blocks(valid,cfg.block_len)
            t=clock(student)
            if not fresh:
                ht,scores,selected=replay_window.microbatch(window,start//micro_size,batch['input_ids'].device)
                meta=TargetMeta(ht.entropy,ht.alpha)
            elif topk.enabled(cfg):
                ht=topk.target(teacher,batch,calibration,cfg)
                meta=TargetMeta(ht.entropy,ht.alpha)
            elif sampled.enabled(cfg):
                ht=sampled.target(student,teacher,batch,cfg)
                if getattr(student,'rollout_engine',None) is not None:
                    rollout_diagnostics.append(sampled.rollout_diagnostic(rollouts,ht.old,
                        expected_hash=student.rollout_engine.policy_hash))
                meta=TargetMeta(torch.zeros_like(ht.old),torch.zeros_like(ht.old))
            else:
                ht=teacher.last_hidden(batch,with_grad=False)
                meta=None if cq and d.valid else target_meta(teacher,ht,valid,calibration,cfg)
            if fresh:timings['teacher_forward_time']+=clock(student)-t
            hidden=None
            if cfg.selector=='tip':
                t=clock(student)
                hidden=student.last_hidden(batch,with_grad=True)
                timings['train_backward_time']+=clock(student)-t
                t=clock(student)
                scores,selected=topk.tip_selection(student,hidden,batch,ht,cfg)
                timings['scoring_time']+=clock(student)-t
                if fresh and reuse_width>1:replay_window.append(window,ht)
            elif not fresh:
                pass  # frozen source-policy scores/mask, not recomputed or relabeled
            elif cq and d.valid:
                if not delta['cq_valid']:
                    raise RuntimeError('CQ delta not calibrated')
                t=clock(student)
                if topk.enabled(cfg):
                    scores=topk.score(student,batch,ht,d,delta['delta'],cfg)
                elif sampled.enabled(cfg):
                    scores=sampled.score(student,batch,ht,d,delta['delta'],cfg)
                else:
                    scores,meta=score(student,teacher,batch,ht,None,d,delta['delta'],cfg,calibration=calibration)
                timings['scoring_time']+=clock(student)-t
            elif cfg.selector in ('kl','teachability') and cfg.keep_ratio<1:
                with torch.no_grad():
                    h0=student.last_hidden(batch,with_grad=False); scores=torch.zeros_like(meta.alpha)
                    for b,tt in valid_rows(valid,cfg.chunk_size):
                        lq=torch.log_softmax(teacher.project(ht[b,tt]).float(),-1)
                        logits=student.project(h0[b,tt])
                        ff=mixed_kl(logits,lq,torch.ones_like(meta.alpha[b,tt]))
                        if cfg.selector=='teachability':
                            top=logits.topk(min(cfg.teachability_topk,logits.shape[-1]),dim=-1).indices
                            ff=ff*lq.exp().gather(-1,top).sum(-1)
                        scores[b,tt]=ff
                    del h0
            else:
                # Random per BLOCK, not random token averages (short tails must not change sampling).
                scores=torch.zeros_like(meta.alpha)
                for block in blocks:
                    scores[block.response_index,list(block.positions)]=extra_rngs['selector'].random()
            if fresh and cfg.selector!='tip':
                selected=valid if cfg.keep_ratio==1 or cfg.selector=='full' else select_blocks(scores,blocks,cfg.keep_ratio)
                if reuse_width>1:replay_window.append(window,ht,scores,selected)
            available+=int(valid.sum()); total+=int(selected.sum())
            entropy_values.extend(meta.entropy[valid].detach().cpu().tolist())
            alpha_values.extend(meta.alpha[valid].detach().cpu().tolist())
            selected_alpha.extend(meta.alpha[selected].detach().cpu().tolist())
            if cq and (not fresh or d.valid):
                utility_values.extend(block_values(scores,blocks).tolist())
                selected_util.extend(scores[selected].detach().cpu().tolist())
            t=clock(student)
            if hidden is None:hidden=student.last_hidden(batch,with_grad=True)
            if topk.enabled(cfg):
                gh,loss_sum,n,forward_sum,reverse_sum=topk.backward_hidden(student,hidden,batch,selected,ht,cfg)
            elif sampled.enabled(cfg):
                gh,loss_sum,n=sampled.backward_hidden(student,hidden,batch,selected,ht,cfg)
                forward_sum=loss_sum.new_zeros(())
                reverse_sum=(ht.old-ht.teacher)[selected].sum()
            else:
                gh,loss_sum,n,forward_sum,reverse_sum=chunked_mixed_kl_hidden_grad(hidden,ht,meta.alpha,selected,
                    student.project,teacher.project,chunk_size=cfg.chunk_size,return_components=True)
            if n:
                torch.autograd.backward(hidden,gh)
                numerator+=float(loss_sum)
                fsum+=float(forward_sum); rsum+=float(reverse_sum)
            timings['train_backward_time']+=clock(student)-t
            del hidden,gh,ht,meta,batch,scores,selected
        if fresh and window is not None:replay_window.seal(window)
        if total==0:
            raise RuntimeError('empty optimizer step; no scheduler/optimizer advance')
        for _,p in parameters:
            if p.grad is not None:
                p.grad.div_(total)
        finite_gradients=[torch.isfinite(p.grad).all() for _,p in parameters if p.grad is not None]
        if not finite_gradients or not torch.stack(finite_gradients).all():
            raise FloatingPointError('missing/nonfinite training gradient')
        norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True))
        t=clock(student); optimizer.step()
        if not torch.stack([torch.isfinite(p).all() for _,p in parameters]).all():
            raise FloatingPointError('nonfinite LoRA parameter after optimizer step')
        if next(student.model.parameters()).is_cuda:
            torch.cuda.synchronize()
        timings['optimizer_time']=clock(student)-t
        import psutil
        av=torch.tensor(alpha_values)
        metrics=dict(step=step+1, student_version=step+1, direction_source_step=source_step if cq else None,
            direction_age=step-source_step if cq else 0,
            selection_source_step=step if cfg.selector=='tip' else source_step,
            selection_age=0 if cfg.selector=='tip' else step-source_step,
            selection_protocol='tip_soft_or_teacher_support_per_response_v1' if cfg.selector=='tip' else cfg.selector,
            window_source_step=source_step,window_age=step-source_step,
            window_expires_step=source_step+reuse_width,reuse_window=reuse_width,
            window_use_count=step-source_step+1,fresh_generation=fresh,
            resume_fd_recheck_deferred=bool(resume_recheck_pending and not fresh), cq_valid=delta.get('cq_valid',False),
            fallback_reason=d.reason if d and not d.valid else None, delta=delta.get('delta'),
            delta_rank_stability=(recheck_state['report']['candidates'] if recheck_state else None),
            delta_recheck_step=recheck_state.get('last_step') if recheck_state else None,
            resume_recheck_pending=resume_recheck_pending,
            delta_calibration_candidates=delta.get('candidates'), selected_tokens=total,available_tokens=available,
            actual_keep_ratio=total/available,train_mixed_kl=numerator/total,
            train_forward_kl=fsum/total,train_reverse_kl=rsum/total,grad_norm=norm,
            truncated_fraction=truncated/cfg.batch_size,train_trajectories=trajectory_records(rollout_batch),
            teacher_entropy_mean=float(np.mean(entropy_values)),
            alpha_mean=float(av.mean()), alpha_p10=float(av.quantile(.1)),alpha_p90=float(av.quantile(.9)),
            alpha_selected_mean=float(np.mean(selected_alpha)),entropy_calibration_hash=digest(calibration),
            tau=calibration.get('tau'), k=calibration.get('k'),
            utility_mean=float(np.mean(utility_values)) if utility_values else None,
            utility_positive_fraction=float(np.mean(np.array(utility_values)>0)) if utility_values else None,
            selected_utility_mean=float(np.mean(selected_util)) if selected_util else None,
            step_elapsed_time=time.monotonic()-started,
            total_elapsed_time=elapsed(), previous_elapsed_time=previous_elapsed,
            pre_training_elapsed_time=startup_elapsed_time, calibration_elapsed_time=calibration_cost,
            imported_calibration_elapsed_time=imported_cost,checkpoint_elapsed_time=checkpoint_time,
            system_memory_used=psutil.virtual_memory().used,
            torch_peak_allocated=torch.cuda.max_memory_allocated() if next(student.model.parameters()).is_cuda else 0,
            **timings,**stats)
        if topk.enabled(cfg):
            metrics['objective']='adaptive_topk'
            metrics['distillation_topk']=cfg.distillation_topk
            metrics['entropy_scope']='teacher_topk_conditional'
            metrics['rollout_source_policy_hashes']=sorted({r.policy_sha256 for r in rollout_batch
                if getattr(r,'policy_sha256',None)})
            metrics['replay_window_sha256']=window['sha256'] if window is not None else None
            metrics['train_topk_mixed_kl']=metrics.pop('train_mixed_kl')
            metrics['train_topk_forward_kl']=metrics.pop('train_forward_kl')
            metrics['train_topk_reverse_kl']=metrics.pop('train_reverse_kl')
        if sampled.enabled(cfg):
            metrics['objective']='sampled_k1'
            metrics['rollout_probability_diagnostic']=sampled.merge_diagnostics(rollout_diagnostics)
            metrics['train_sampled_pg_loss']=metrics.pop('train_mixed_kl')
            metrics['train_sampled_k1']=metrics.pop('train_reverse_kl')
            for key in ('train_forward_kl','teacher_entropy_mean','alpha_mean','alpha_p10','alpha_p90',
                        'alpha_selected_mean','tau','k'):
                metrics.pop(key,None)
        engine=getattr(student,'rollout_engine',None)
        if engine is not None:
            metrics['rollout_policy_hash']=engine.policy_hash
            metrics['rollout_policy_gate']=engine.policy_checks[-1]
            metrics['rollout_sync_time_cumulative']=engine.sync_seconds
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics,allow_nan=False)+'\n')
        print(json.dumps(metrics,allow_nan=False),flush=True)
        if (step+1)%cfg.save_every==0 or step+1==cfg.steps:
            t=clock(student)
            save_checkpoint(output/f'checkpoint-{step+1}.pt',student,optimizer,step+1,cfg,
                            calibration,delta,rng,metadata,order,cursor,extra_rngs,elapsed,recheck_state,
                            replay_state=window)
            checkpoint_time+=clock(student)-t
    write_json(output/'training_summary.json',dict(completed_steps=cfg.steps,
        total_elapsed_time=elapsed(),previous_elapsed_time=previous_elapsed,
        pre_training_elapsed_time=startup_elapsed_time,imported_calibration_elapsed_time=imported_cost,
        checkpoint_elapsed_time=checkpoint_time,final_checkpoint_included=True))


def guarded_arguments():
    import os
    import sys
    # Existing guard appends a Ray/Hydra transport argument to EVERY command.
    # This native trainer has no Ray. Accept only the guard's exact tagged path.
    expected='+ray_kwargs.ray_init._temp_dir='+os.environ.get('RAY_TMPDIR','')
    return [arg for arg in sys.argv[1:] if not (arg==expected and
            os.environ.get('VERL_OPD_GUARD_RUN') and os.environ.get('RAY_TMPDIR'))]


def main():
    resources=[]
    try:
        return _main(resources)
    finally:
        for resource in reversed(resources):resource.close()


def _main(resources):
    import os
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['preflight','calibrate','train'],default='preflight')
    parser.add_argument('--objective',choices=['adaptive_topk','sampled_k1','mixed_kl'],default='adaptive_topk')
    parser.add_argument('--distillation-topk',type=int,default=64)
    parser.add_argument('--reuse-window',type=int,default=None,
                        help='Optimizer updates per rollout/CQ snapshot; adaptive_topk default4, other objectives1')
    parser.add_argument('--rollout-backend',choices=['vllm','native'],default='vllm')
    parser.add_argument('--fd-scoring-dtype',choices=['fp32','native'],default='fp32')
    parser.add_argument('--student-dtype',choices=['bf16','fp32'],default='bf16',
                        help='FP32 Student for stable CQ differences; Teacher remains native FP8')
    parser.add_argument('--selector',choices=['tip','cq','random','kl','teachability','full'],default=None)
    parser.add_argument('--teachability-topk',type=int,default=64)
    parser.add_argument('--mixing',choices=['teacher_entropy','constant','sampled_k1'],default='teacher_entropy')
    parser.add_argument('--constant-alpha',type=float,default=1.)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--micro-batch-size',type=int,default=4)
    parser.add_argument('--rollout-batch-size',type=int,default=32)
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    parser.add_argument('--probe-questions',type=int,default=4)
    parser.add_argument('--probe-answers',type=int,default=4)
    parser.add_argument('--entropy-questions',type=int,default=32)
    parser.add_argument('--delta-questions',type=int,default=2)
    parser.add_argument('--delta-recheck-every',type=int,default=5)
    parser.add_argument('--keep-ratio',type=float,default=None)
    parser.add_argument('--block-len',type=int,default=16)
    parser.add_argument('--chunk-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-6)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--save-every',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--calibration-dir',type=Path)
    parser.add_argument('--resume',type=Path)
    cfg=parser.parse_args(guarded_arguments())
    if cfg.selector is None:cfg.selector='tip' if topk.enabled(cfg) else 'cq'
    if cfg.keep_ratio is None:cfg.keep_ratio=.5 if cfg.selector=='tip' else .2
    if cfg.selector=='tip' and not topk.enabled(cfg):raise ValueError('TIP requires adaptive_topk')
    if cfg.reuse_window is None:cfg.reuse_window=4 if topk.enabled(cfg) else 1
    replay_window.size(cfg)
    if cfg.distillation_topk<2:raise ValueError('distillation-topk must be at least 2')
    if topk.enabled(cfg):
        if cfg.selector not in ('tip','cq','random','full'):raise ValueError('adaptive topk supports tip/cq/random/full selectors only')
        if cfg.mixing!='teacher_entropy':raise ValueError('adaptive topk requires teacher_entropy mixing')
    if sampled.enabled(cfg):
        if cfg.selector not in ('cq','random','full'):
            raise ValueError('sampled objective supports cq/random/full selectors only')
        cfg.mixing='sampled_k1'
    elif cfg.mixing=='sampled_k1':
        raise ValueError('sampled_k1 mixing requires sampled_k1 objective')
    if not os.environ.get('VERL_OPD_GUARD_RUN'):
        raise RuntimeError('GPU work requires external guard')
    if not 0<cfg.keep_ratio<=1 or not 0<=cfg.constant_alpha<=1:
        raise ValueError('invalid selection/mixing ratio')
    if min(cfg.steps,cfg.batch_size,cfg.micro_batch_size,cfg.rollout_batch_size,cfg.max_new_tokens,cfg.chunk_size,cfg.block_len,cfg.save_every,
           cfg.probe_questions,cfg.entropy_questions,cfg.delta_questions,cfg.teachability_topk,cfg.delta_recheck_every)<1 or cfg.probe_answers<2:
        raise ValueError('invalid positive budget or probe K<2')
    cfg.output.mkdir(parents=True,exist_ok=False)
    output=cfg.output
    # Paths serialized as strings in config/checkpoint for unambiguous comparison.
    cfg.output=str(cfg.output)
    cfg.calibration_dir=str(cfg.calibration_dir) if cfg.calibration_dir else None
    cfg.resume=str(cfg.resume) if cfg.resume else None
    rng=random.Random(cfg.seed); torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    entropy_rng=random.Random(cfg.seed+7001)
    calibration_probe_rng=random.Random(cfg.seed+2001)
    calibration_trajectory_rng=random.Random(cfg.seed+6001)
    root=Path('/home/eryx/qwen3.5-opd')
    startup_started=time.monotonic()
    from transformers import AutoTokenizer
    sp='/home/eryx/models/Qwen3.5-2B'; tp='/home/eryx/models/Qwen3.5-35B-A3B-FP8'
    tokenizer=AutoTokenizer.from_pretrained(sp,local_files_only=True)
    tt=AutoTokenizer.from_pretrained(tp,local_files_only=True)
    student=load_student(sp,root/'baselines/gsm8k_k1/lora_manifest.json',base_dtype=cfg.student_dtype,
                         fp32_head_output=cfg.rollout_backend=='vllm')
    teacher=load_teacher(tp)
    metadata=validate_alignment(student,teacher,tokenizer,tt,thinking=False)
    initial_hash=hashlib.sha256()
    for name,p in student.lora_named_parameters():
        initial_hash.update(name.encode()); initial_hash.update(p.detach().cpu().contiguous().numpy().tobytes())
    metadata['initial_lora_sha256']=initial_hash.hexdigest()
    metadata['algorithm_sha256']={name:hashlib.sha256((root/'cq_opd'/f'{name}.py').read_bytes()).hexdigest()
        for name in ('losses','sampled_loss','sampled_trainer','topk_trainer','replay_window','probe','utility','blocks','selectors','model_adapter','rollout','vllm_rollout','lora_integrity','vllm_integrity_worker','precision_gate','gates','trainer')}
    import transformers
    metadata['software']={'torch':torch.__version__,'transformers':transformers.__version__,
                          'numpy':np.__version__}
    metadata['dataset_sha256']={name:hashlib.sha256((root/'data/gsm8k'/f'{name}.parquet').read_bytes()).hexdigest()
                                for name in ('train','probe','dev','test')}
    metadata['student_config_sha256']=hashlib.sha256((Path(sp)/'config.json').read_bytes()).hexdigest()
    splits=load_splits(root/'data/gsm8k')
    if cfg.probe_questions>len(splits['probe']) or cfg.entropy_questions>len(splits['train']) or cfg.delta_questions>len(splits['dev']):
        raise ValueError('calibration/probe sample budget exceeds split size')
    # Mandatory for EVERY entry mode, before calibration or optimizer updates.
    # A caller cannot bypass the gate by omitting a separate preflight run.
    ids=prompt_ids(tokenizer,splits['train'][0])
    gate_batch=build_batch([generate(student,tokenizer,ids,2,cfg.seed,id='mode_gate')],
                          'cuda',tokenizer.pad_token_id)
    mode_gate=sampled.forward_gate if sampled.enabled(cfg) or topk.enabled(cfg) else validate_forward_modes
    mode_proofs=mode_gate(student,teacher,gate_batch)
    from .precision_gate import validate_fd_update_consistency
    precision_gate=topk.precision_gate if topk.enabled(cfg) else sampled.precision_gate if sampled.enabled(cfg) else validate_fd_update_consistency
    precision_proof=precision_gate(student,teacher,gate_batch,cfg)
    write_json(output/'fd_update_precision_gate.json',precision_proof)
    metadata['fd_update_precision_gate']=precision_proof
    if cfg.rollout_backend=='vllm':
        from .vllm_rollout import BaselineVLLMRollout
        student.rollout_engine=BaselineVLLMRollout(student,tokenizer,max_num_seqs=cfg.rollout_batch_size,
            parity_scope='sampled' if sampled.enabled(cfg) or topk.enabled(cfg) else 'full')
        resources.append(student.rollout_engine)
        student.rollout_engine.sync_policy()
        metadata['rollout_engine']=dict(backend='vllm',dtype='bf16',
            baseline_extension='verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension',
            policy_atol=.02,policy_rtol=.02)
        write_json(output/'vllm_policy_gate.json',student.rollout_engine.policy_checks)
    metadata['forward_mode_gate']=dict(passed=True,version=1,atol=.02,rtol=.02,
                                      input_ids_sha256=digest(gate_batch['input_ids'].cpu().tolist()))
    write_json(output/'forward_mode_gate.json',dict(metadata_hash=digest(metadata),**mode_proofs))
    write_json(output/'model-manifest.json',metadata); write_json(output/'config.json',vars(cfg))
    if cfg.mode=='preflight':
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
        engine=getattr(student,'rollout_engine',None)
        if engine is None:
            audits=generate_batch(student,tokenizer,[ids]*cfg.rollout_batch_size,2,cfg.seed,
                                  ids=[f'batch_audit_{i}' for i in range(cfg.rollout_batch_size)])
        else:
            audits=engine.generate([ids]*cfg.rollout_batch_size,2,cfg.seed,
                                  [f'batch_audit_{i}' for i in range(cfg.rollout_batch_size)])
        if not torch.equal(cpu_rng,torch.get_rng_state()) or any(not torch.equal(a,b)
                for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):
            raise RuntimeError('native batch generation polluted caller RNG')
        batch=batch_of(audits[:cfg.micro_batch_size],student,tokenizer)
        proofs=dict(forward_mode_gate=mode_proofs,native_batch_generation=dict(
            rows=len(audits),micro_batch_size=int(batch['input_ids'].shape[0]),caller_rng_restored=True,
            scope='two_token_adapter_audit_not_2048_training_acceptance'))
        if sampled.enabled(cfg) or topk.enabled(cfg):
            proofs['sampled_mode_gate']=sampled.forward_gate(student,teacher,batch)
            proofs['backward']=precision_gate(student,teacher,batch,cfg)
            proofs['objective']=cfg.objective
            write_json(output/'preflight.json',proofs)
            return
        for name,adapter in [('student',student),('teacher',teacher)]:
            with torch.no_grad():
                hidden=adapter.last_hidden(batch,with_grad=False)
                rows=torch.arange(hidden.shape[0],device=hidden.device)
                last=batch['attention_mask'].sum(-1)-1
                projected=adapter.project(hidden[rows,last]).unsqueeze(1)
                standard=adapter.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                    use_cache=False,logits_to_keep=0).logits[rows,last].unsqueeze(1)
                torch.testing.assert_close(projected,standard,rtol=.02,atol=.02)
            proofs[name]=dict(shape=list(projected.shape),finite=bool(torch.isfinite(projected).all()))
        # Native real-model backward audit, with no optimizer step and no CQ validity claim.
        valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask'])
        ht=teacher.last_hidden(batch,with_grad=False)
        with torch.no_grad():
            log_q=torch.log_softmax(teacher.project(ht[valid.nonzero()[:,0],valid.nonzero()[:,1]]).float(),-1)
            entropies=-(log_q.exp()*log_q).sum(-1)
        local_calibration=calibrate_entropy(entropies)  # short-input adapter audit only
        meta=target_meta(teacher,ht,valid,local_calibration,cfg)
        hidden=student.last_hidden(batch,with_grad=True)
        gh,loss_sum,count=chunked_mixed_kl_hidden_grad(hidden,ht,meta.alpha,valid,
            student.project,teacher.project,chunk_size=cfg.chunk_size)
        names=student.lora_named_parameters()
        gradients=torch.autograd.grad(hidden,[p for _,p in names],grad_outputs=gh,allow_unused=True)
        from .probe import direction_from_gradients
        gradient_direction=direction_from_gradients(names,gradients)
        if not gradient_direction.valid:
            raise RuntimeError('real native mixed KL backward has no signal')
        proofs['backward']=dict(selected_tokens=count,mixed_sum=float(loss_sum),
            gradient_norm=gradient_direction.grad_norm,finite=True,
            nonzero_gradient_tensors=sum(g is not None and bool(torch.count_nonzero(g)) for g in gradients),
            calibration_scope='preflight_short_input_only_not_formal_entropy_mapping')
        if any(p.grad is not None for _,p in names) or any(p.grad is not None for p in teacher.model.parameters()):
            raise RuntimeError('preflight autograd.grad polluted parameter .grad')
        write_json(output/'preflight.json',proofs)
        return
    resume=load_checkpoint(cfg.resume) if cfg.resume else None
    imported_calibration_cost=None
    initial_probe_cache={} if cfg.mode=='train' and not resume and not cfg.calibration_dir else None
    if resume:
        calibration=resume['entropy_calibration']; delta=resume['delta_calibration']
    elif cfg.calibration_dir:
        calibration=json.loads((Path(cfg.calibration_dir)/'entropy_calibration.json').read_text())
        delta=json.loads((Path(cfg.calibration_dir)/'delta_calibration.json').read_text())
        imported_calibration_cost=load_calibration_job(Path(cfg.calibration_dir)/'calibration_job.json',
                                                      calibration,delta,metadata)
    else:
        calibration=(dict(mode='sampled_k1',metadata_hash=digest(metadata),protocol=entropy_protocol(cfg),
                          elapsed_time=0.,coefficient='native_old_minus_teacher_clamp10_detached',
                          ppo_clip=.2,dual_clip=3.,entropy_calibration=False)
            if sampled.enabled(cfg) else entropy_calibration(student,teacher,tokenizer,splits['train'],cfg,metadata,entropy_rng)
            if cfg.mixing in ('teacher_entropy','sampled_k1') else dict(mode='constant',constant_alpha=cfg.constant_alpha,
                                                       metadata_hash=digest(metadata),protocol=entropy_protocol(cfg)))
        write_json(output/'entropy_calibration.json',calibration)
        delta=(delta_calibration(student,teacher,tokenizer,splits['dev'],splits['probe'],calibration,cfg,calibration_trajectory_rng,
                                  probe_rng=calibration_probe_rng,initial_probe_cache=initial_probe_cache,
                                  failure_output=output/'probe_startup_failure.json')
               if cfg.selector=='cq' and cfg.keep_ratio<1 else dict(cq_valid=False,delta=None,performed=False,reason='selector_does_not_use_CQ'))
    if calibration.get('protocol')!=entropy_protocol(cfg):
        raise ValueError('entropy calibration sampling protocol mismatch')
    if calibration['metadata_hash']!=digest(metadata):
        raise ValueError('entropy calibration model/tokenizer mismatch')
    if calibration['mode']!=cfg.mixing or (cfg.mixing=='constant' and calibration['constant_alpha']!=cfg.constant_alpha):
        raise ValueError('entropy calibration mixing mismatch')
    if cfg.selector=='cq' and cfg.keep_ratio<1:
        if delta.get('entropy_calibration_hash')!=digest(calibration):
            raise ValueError('delta calibration entropy mismatch or missing hash')
        if delta.get('protocol')!=scoring_protocol(cfg):
            raise ValueError('delta calibration scoring protocol mismatch')
    write_json(output/'entropy_calibration.json',calibration)
    write_json(output/'delta_calibration.json',delta)
    if not resume:
        job_wall_time=clock(student)-startup_started
        write_json(output/'calibration_job.json',dict(total_elapsed_time=job_wall_time+(imported_calibration_cost or 0.),
            elapsed_wall_time=job_wall_time,inherited_calibration_cost=imported_calibration_cost or 0.,
            includes_model_load_and_mode_gate=True,metadata_hash=digest(metadata),
            entropy_calibration_hash=digest(calibration),delta_calibration_hash=digest(delta)))
    if cfg.selector=='cq' and cfg.keep_ratio<1 and not delta['cq_valid']:
        raise RuntimeError('finite difference calibration inconclusive or insufficient startup margin; CQ training refused')
    def resumed_checks():
        write_json(output/'resume_forward_mode_gate.json',mode_gate(student,teacher,gate_batch))
        write_json(output/'resume_fd_update_precision_gate.json',precision_gate(student,teacher,gate_batch,cfg))
        engine=getattr(student,'rollout_engine',None)
        if engine is not None:
            engine.sync_policy()
            write_json(output/'resume_vllm_policy_gate.json',engine.policy_checks[-1])
    if cfg.mode=='train':
        train(student,teacher,tokenizer,splits,cfg,output,calibration,delta,metadata,rng,resume,
              startup_elapsed_time=clock(student)-startup_started,
              calibration_is_imported=bool(cfg.calibration_dir),imported_calibration_cost=imported_calibration_cost,
              initial_probe_cache=initial_probe_cache,
              mode_check=resumed_checks)


if __name__=='__main__':
    main()

"""Sampled-k1 CQ integration; no entropy mapping or full-distribution KL."""
from dataclasses import dataclass
import math
import torch
from .blocks import shifted_valid_mask, make_blocks
from .sampled_loss import (token_log_probs, sampled_k1_coefficient, sampled_pg_tokens,
                           sampled_pg_utility, chunked_sampled_hidden_grad)
from .model_adapter import temporary_scoring_precision
from .utility import score_symmetric_offsets


def enabled(cfg):
    return getattr(cfg, 'objective', 'mixed_kl') == 'sampled_k1'


@dataclass
class Target:
    old: torch.Tensor
    teacher: torch.Tensor
    coefficient: torch.Tensor


def target(student, teacher, batch, cfg, teacher_logp=None):
    # MUST run before promoting the frozen Student for FD.
    old = token_log_probs(student, batch, cfg.chunk_size)
    q = token_log_probs(teacher, batch, cfg.chunk_size) if teacher_logp is None else teacher_logp
    return Target(old, q, sampled_k1_coefficient(old, q))


def score(student, batch, frozen, direction, delta, cfg):
    from .trainer import fd_precision
    with temporary_scoring_precision(student, fd_precision(cfg)):
        good, bad = score_symmetric_offsets(student.lora_named_parameters(), direction, delta,
            lambda: token_log_probs(student, batch, cfg.chunk_size))
    return sampled_pg_utility(good, bad, frozen.old, frozen.coefficient, delta)


def rollout_diagnostic(rollouts, native_logp, *, expected_hash=None):
    """Diagnostic only: baseline uses native recomputed old logp, no rollout IS."""
    native=[]; rollout=[]
    for row,r in enumerate(rollouts):
        saved=getattr(r,'sampled_log_probs',None)
        if saved is None or len(saved)!=len(r.response_ids):
            raise RuntimeError('missing sampled rollout probabilities')
        if not getattr(r,'policy_sha256',None) or (expected_hash is not None and r.policy_sha256!=expected_hash):
            raise RuntimeError('stale sampled rollout policy hash')
        native.append(native_logp[row,r.prompt_len-1:r.prompt_len-1+len(saved)].detach().float())
        rollout.append(torch.tensor(saved,device=native_logp.device,dtype=torch.float32))
    a=torch.cat(native);b=torch.cat(rollout)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise RuntimeError('nonfinite rollout diagnostic')
    diff=(a-b).abs();outside=diff>(.02+.02*a.abs())
    return dict(tokens=a.numel(),max_abs=float(diff.max()),sum_abs=float(diff.sum()),
        mean_abs=float(diff.mean()),outside_tolerance_count=int(outside.sum()),
        outside_tolerance_fraction=float(outside.float().mean()),atol=.02,rtol=.02,
        scope='diagnostic_only_native_recompute_old_logp_no_rollout_IS')


def merge_diagnostics(reports):
    if not reports:return None
    count=sum(r['tokens'] for r in reports);total=sum(r['sum_abs'] for r in reports)
    outside=sum(r['outside_tolerance_count'] for r in reports)
    return dict(tokens=count,max_abs=max(r['max_abs'] for r in reports),mean_abs=total/count,
        outside_tolerance_count=outside,outside_tolerance_fraction=outside/count,
        atol=.02,rtol=.02,scope='diagnostic_only_native_recompute_old_logp_no_rollout_IS')


def backward_hidden(student, hidden, batch, selected, frozen, cfg):
    from .model_adapter import gradient_projector
    with gradient_projector(student) as project:
        return chunked_sampled_hidden_grad(hidden, batch['input_ids'][:, 1:], selected,
            project, frozen.old, frozen.coefficient, cfg.chunk_size)


def measure_fd(student, teacher, batch, calibration, cfg, direction, candidates, *, frozen_target=None):
    from .trainer import fd_precision, block_values, rank_agreement
    # Only Teacher probabilities may survive policy changes. Native old/coefficient
    # are always freshly computed, even on a Teacher-cache hit.
    frozen = target(student, teacher, batch, cfg, teacher_logp=frozen_target)
    # All candidates share the native frozen target. Promote frozen weights ONCE
    # for this audit, not for each of its 18 +/- evaluations. Nested score()
    # contexts become no-ops and outer restoration still covers every failure.
    with temporary_scoring_precision(student, fd_precision(cfg)):
        return _measure_fd_frozen(student,batch,cfg,direction,candidates,frozen)


def _measure_fd_frozen(student,batch,cfg,direction,candidates,frozen):
    from .trainer import fd_precision, block_values, rank_agreement
    valid = shifted_valid_mask(batch['response_mask'], batch['attention_mask'])
    blocks = make_blocks(valid, cfg.block_len)
    if not blocks: raise ValueError('no finite-difference audit blocks')
    with temporary_scoring_precision(student, fd_precision(cfg)):
        a = token_log_probs(student, batch, cfg.chunk_size)
        b = token_log_probs(student, batch, cfg.chunk_size)
    noise = block_values((sampled_pg_tokens(a, frozen.old, frozen.coefficient) -
                          sampled_pg_tokens(b, frozen.old, frozen.coefficient)).abs(), blocks)
    reports = []
    for rho, delta in candidates:
        if not math.isfinite(delta) or delta <= 0: raise ValueError('invalid audit delta')
        values = [block_values(score(student, batch, frozen, direction, delta*f, cfg), blocks)
                  for f in (.5, 1., 2.)]
        significant_mask = values[1].abs()*2*delta > torch.maximum(noise*10, torch.full_like(noise, 1e-7))
        significant = [block for block, keep in zip(blocks, significant_mask) if keep]
        agreements = []
        for value in (values[0], values[2]):
            correlation, _ = rank_agreement(values[1][significant_mask], value[significant_mask], significant, cfg.keep_ratio)
            _, overlap = rank_agreement(values[1], value, blocks, cfg.keep_ratio)
            agreements.append((correlation, overlap))
        signs = [float((torch.sign(values[1][significant_mask]) == torch.sign(v[significant_mask])).float().mean())
                 if significant_mask.any() else 0. for v in (values[0], values[2])]
        fraction = float(significant_mask.float().mean())
        passed = fraction >= .5 and all(c is not None and c >= .8 and o >= .7 for c,o in agreements) and min(signs) >= .8
        reports.append(dict(rho=rho, delta=delta, agreements=agreements, signal_fraction=fraction,
                            sign_agreement=signs, passed=passed))
    return dict(candidates=reports, noise_max=float(noise.max()), objective='sampled_k1')


def forward_gate(student, teacher, batch):
    """Compare sampled logits to native output, and eval/scoring logp parity."""
    valid = shifted_valid_mask(batch['response_mask'], batch['attention_mask'])
    rows, positions = valid.nonzero(as_tuple=True)
    tokens = batch['input_ids'][rows, positions+1]
    proofs = {}
    for name, adapter in [('student',student), ('teacher',teacher)]:
        flags = [(module,module.training) for module in adapter.model.modules()]
        hook=None
        if getattr(adapter,'fp32_head_output',False):
            # Native conditional wrapper normally rounds BF16 head output. Use
            # the explicitly configured policy head for both parity paths.
            def configured_head(module, inputs, output):
                if inputs[0].dtype==torch.bfloat16:
                    return adapter.project(inputs[0])
                return output
            hook=adapter.model.lm_head.register_forward_hook(configured_head)
        try:
            adapter.model.eval()
            with torch.no_grad():
                logits = adapter.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    use_cache=False, logits_to_keep=0).logits[rows, positions].float()
                native = logits.gather(-1,tokens[:,None]).squeeze(-1)-logits.logsumexp(-1)
                projected = token_log_probs(adapter,batch)[rows,positions]
            torch.testing.assert_close(projected,native,atol=.02,rtol=.02)
            proofs[name] = dict(passed=True,checked_tokens=tokens.numel(),max_abs=float((projected-native).abs().max()))
        finally:
            if hook is not None: hook.remove()
            for module, flag in flags: module.training=flag
    return dict(passed=True,scope='sampled_tokens',atol=.02,rtol=.02,models=proofs)


def precision_gate(student, teacher, batch, cfg):
    from .trainer import fd_precision
    frozen = target(student, teacher, batch, cfg)
    valid = shifted_valid_mask(batch['response_mask'], batch['attention_mask'])
    params = student.lora_named_parameters()
    results = []
    for dtype in (None, fd_precision(cfg)):
        with temporary_scoring_precision(student,dtype):
            hidden = student.last_hidden(batch,with_grad=True)
            gh, loss, count = backward_hidden(student,hidden,batch,valid,frozen,cfg)
            gradients = torch.autograd.grad(hidden,[p for _,p in params],grad_outputs=gh,allow_unused=True)
            results.append((loss.detach(),[torch.zeros_like(p) if g is None else g.detach().clone()
                                          for (_,p),g in zip(params,gradients)]))
            del hidden, gh, gradients
    torch.testing.assert_close(results[0][0],results[1][0],atol=.02,rtol=.02)
    a,b = results[0][1],results[1][1]
    aa = sum(g.double().square().sum() for g in a); bb = sum(g.double().square().sum() for g in b)
    dot = sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    if not torch.isfinite(aa+bb+dot) or aa<=0 or bb<=0: raise RuntimeError('sampled precision gate has no finite gradient')
    cosine=float(dot/(aa*bb).sqrt()); relative=float((aa.sqrt()-bb.sqrt()).abs()/aa.sqrt())
    if cosine<.99 or relative>.05: raise RuntimeError('FD precision/update gradient mismatch')
    if any(p.grad is not None for _,p in params): raise RuntimeError('precision gate polluted gradients')
    return dict(passed=True,objective='sampled_k1',selected_tokens=count,loss_native=float(results[0][0]),
        loss_fd=float(results[1][0]),gradient_cosine=cosine,gradient_norm_relative=relative,
        gradient_norm=float(aa.sqrt()),minimum_cosine=.99,maximum_norm_relative=.05,
        scope='short_input_numerical_consistency_not_FD_permission')

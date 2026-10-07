"""Fixed-context compute benchmark. No generation, calibration, or optimizer step."""
import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from cq_opd import trainer
from cq_opd.blocks import make_blocks, shifted_valid_mask
from cq_opd.gates import check_delta_gate, measure_fd
from cq_opd.losses import chunked_mixed_kl_hidden_grad, kl_components, log_probs
from cq_opd.model_adapter import load_student, load_teacher, require_gpu_guard
from cq_opd.selectors import select_blocks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--context',type=Path,required=True)
    args=parser.parse_args(trainer.guarded_arguments());require_gpu_guard()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path('/home/eryx/qwen3.5-opd')
    old=root/'runs/cq_calibration_20261006T090843_validation'
    cfg=SimpleNamespace(**json.loads((old/'config.json').read_text()))
    calibration=json.loads((old/'entropy_calibration.json').read_text())
    context=torch.load(args.context,map_location='cuda',weights_only=False)  # trusted own evidence only
    torch.manual_seed(cfg.seed)
    student=load_student('/home/eryx/models/Qwen3.5-2B',root/'baselines/gsm8k_k1/lora_manifest.json',base_dtype='fp32')
    teacher=load_teacher('/home/eryx/models/Qwen3.5-35B-A3B-FP8')
    params=student.lora_named_parameters()
    batch={k:v.cuda() for k,v in context['batch'].items()};d=context['direction']
    valid=shifted_valid_mask(batch['response_mask'],batch['attention_mask']);blocks=make_blocks(valid,cfg.block_len)
    ht=teacher.last_hidden(batch,with_grad=False)
    from cq_opd.utility import parameter_norm
    norm=parameter_norm(params)
    calibration_report=measure_fd(student,teacher,batch,calibration,cfg,d,
        [(rho,rho*norm) for rho in (1e-5,3e-5,1e-4,3e-4,1e-3,3e-3)])
    delta=trainer.choose_startup_delta(calibration_report['candidates'],precision='fp32')
    trainer.write_json(args.output/'diagnostic_delta.json',calibration_report)
    if delta is None:
        raise RuntimeError('diagnostic delta has no robust interior candidate')
    project=teacher.project
    projected_rows=[0]
    def counted(rows):
        projected_rows[0]+=rows.shape[0]
        return project(rows)
    teacher.project=counted

    def compute(fused):
        projected_rows[0]=0
        started=trainer.clock(student)
        if fused:
            utility,meta=trainer.score(student,teacher,batch,ht,None,d,delta,cfg,calibration=calibration)
        else:
            meta=trainer.target_meta(teacher,ht,valid,calibration,cfg)
            utility=trainer.score(student,teacher,batch,ht,meta.alpha,d,delta,cfg)
        mask=select_blocks(utility,blocks,cfg.keep_ratio)
        hidden=student.last_hidden(batch,with_grad=True)
        if fused:
            gh,total,count,f,r=chunked_mixed_kl_hidden_grad(hidden,ht,meta.alpha,mask,
                student.project,teacher.project,cfg.chunk_size,return_components=True)
        else:
            gh,total,count=chunked_mixed_kl_hidden_grad(hidden,ht,meta.alpha,mask,
                student.project,teacher.project,cfg.chunk_size)
            f=total.new_zeros(());r=total.new_zeros(())
            with torch.no_grad():
                for b,t in trainer.valid_rows(mask,cfg.chunk_size):
                    ff,rr=kl_components(student.project(hidden[b,t]),log_probs(teacher.project(ht[b,t])))
                    f+=ff.sum();r+=rr.sum()
        gradients=torch.autograd.grad(hidden,[p for _,p in params],grad_outputs=gh,allow_unused=True)
        elapsed=trainer.clock(student)-started
        return dict(elapsed=elapsed,teacher_projected_rows=projected_rows[0],count=count,
                    total=total,forward=f,reverse=r,utility=utility,mask=mask,gradients=gradients)

    before=compute(False);after=compute(True)  # warmup plus exact correctness audit
    assert torch.equal(before['mask'],after['mask'])
    for key in ('total','forward','reverse','utility'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    for a,b in zip(before['gradients'],after['gradients']):
        assert (a is None)==(b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    before_rows=before['teacher_projected_rows'];after_rows=after['teacher_projected_rows']
    count=after['count'];del before,after
    elapsed={False:[],True:[]}
    for _ in range(3):
        for fused in (False,True):
            result=compute(fused);elapsed[fused].append(result['elapsed']);del result
    report=dict(scope='diagnostic_only_fixed_context_compute_no_sampling_or_optimizer',
        alpha_source_old_diagnostic_calibration=str(old),cannot_authorize_training=True,
        gradients_and_selection_bit_equal=True,selected_tokens=count,delta=delta,
        diagnostic_delta_candidates=calibration_report['candidates'],
        before=dict(samples=elapsed[False],median=statistics.median(elapsed[False]),teacher_projected_rows=before_rows),
        after=dict(samples=elapsed[True],median=statistics.median(elapsed[True]),teacher_projected_rows=after_rows),
        speedup=statistics.median(elapsed[False])/statistics.median(elapsed[True]),
        teacher_precision='original_fp8',student_precision='fp32_tf32_off',
        grads_clean=all(p.grad is None for _,p in params))
    # Publish compute proof before a separate FD-cache audit; never lose measured
    # timings merely because a later diagnostic refuses its numerical gate.
    trainer.write_json(args.output/'benchmark.json',report)
    tokenizer=AutoTokenizer.from_pretrained('/home/eryx/models/Qwen3.5-2B',local_files_only=True)
    state={'rollouts':[{key:getattr(r,key) for key in r.__dataclass_fields__} for r in context['rollouts']]}
    cache={};cache_elapsed=[]
    for _ in range(2):
        t=trainer.clock(student)
        state=check_delta_gate(student,teacher,tokenizer,[],calibration,cfg,d,
                               {'cq_valid':True,'delta':delta},state,cache=cache)
        cache_elapsed.append(trainer.clock(student)-t)
    report.update(fd_recheck_cold_seconds=cache_elapsed[0],fd_recheck_cached_seconds=cache_elapsed[1],
                  fd_gate=state['report'])
    trainer.write_json(args.output/'benchmark.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

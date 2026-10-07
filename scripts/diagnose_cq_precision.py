"""No-update precision diagnostic on identical trajectories and probe direction.

This is NOT a calibration artifact and cannot authorize CQ training.
"""
import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from cq_opd.model_adapter import load_student, load_teacher, require_gpu_guard
from cq_opd.trainer import (guarded_arguments, load_splits, direction, sample, batch_of,
                            write_json, clock, digest)
from cq_opd.gates import measure_fd, validate_forward_modes
from cq_opd.utility import parameter_norm


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(guarded_arguments())
    require_gpu_guard()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path('/home/eryx/qwen3.5-opd')
    old=root/'runs/cq_calibration_20261006T090843_validation'
    cfg=SimpleNamespace(**json.loads((old/'config.json').read_text()))
    calibration=json.loads((old/'entropy_calibration.json').read_text())
    torch.manual_seed(cfg.seed)
    # Differential diagnosis must not introduce TF32 rounding in FP32 GEMMs.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained('/home/eryx/models/Qwen3.5-2B',local_files_only=True)
    student=load_student('/home/eryx/models/Qwen3.5-2B',root/'baselines/gsm8k_k1/lora_manifest.json')
    teacher=load_teacher('/home/eryx/models/Qwen3.5-35B-A3B-FP8')
    splits=load_splits(root/'data/gsm8k')
    rng=random.Random(cfg.seed+7001)
    d,stats=direction(student,tokenizer,splits['probe'],cfg,0,rng)
    if not d.valid:
        raise RuntimeError('diagnostic has no probe signal; no retry')
    records=[sample(student,tokenizer,row,cfg.max_new_tokens,rng.randrange(2**31))
             for row in rng.sample(splits['dev'],cfg.delta_questions)]
    batch=batch_of(records,student,tokenizer)
    torch.save(dict(direction=d,rollouts=records,batch={k:v.cpu() for k,v in batch.items()}),
               args.output/'fixed_context.pt')
    norm=parameter_norm(student.lora_named_parameters())
    candidates=[(rho,rho*norm) for rho in (1e-5,3e-5,1e-4,3e-4,1e-3,3e-3)]
    report=dict(scope='diagnostic_only_no_calibration_no_updates',probe=stats,
        trajectory_hash=digest([r.input_ids for r in records]),lora_norm=norm,
        teacher_precision='original_fp8',student_initial_base='original_bf16',
        tf32=False,results={},setup_elapsed=clock(student)-started)
    write_json(args.output/'diagnostic.json',report)
    for dtype in (torch.bfloat16,torch.float32):
        label=str(dtype).split('.')[-1]
        if dtype==torch.float32:
            # Promotion preserves all LoRA masters exactly. Do NOT round them
            # through BF16 when diagnosing the existing baseline.
            student.model.to(dtype=dtype)
        t=clock(student)
        torch.cuda.reset_peak_memory_stats()
        try:
            parity=validate_forward_modes(student,teacher,batch)
            result=measure_fd(student,teacher,batch,calibration,cfg,d,candidates)
            report['results'][label]=dict(mode_gate=parity,fd=result,
                elapsed=clock(student)-t,peak_allocated=torch.cuda.max_memory_allocated())
        except Exception as exc:
            report['results'][label]=dict(error=repr(exc),elapsed=clock(student)-t)
            write_json(args.output/'diagnostic.json',report)
            raise
        write_json(args.output/'diagnostic.json',report)
        print(json.dumps(report['results'][label]),flush=True)
    report['total_elapsed']=clock(student)-started
    report['all_parameter_grads_clean']=all(p.grad is None for p in student.model.parameters())
    write_json(args.output/'diagnostic.json',report)


if __name__=='__main__':
    main()

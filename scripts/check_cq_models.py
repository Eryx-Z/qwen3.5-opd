#!/usr/bin/env python3
"""Guarded real-model alignment/logits/gradient preflight; no training step."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cq_opd.model_adapter import (check_logits_equivalence, fp8_environment_report,
                                 load_student, load_teacher, prepare_fp8_kernel, prompt_ids, require_gpu_guard,
                                 validate_alignment)
from cq_opd.rollout import Rollout, build_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="/home/eryx/models/Qwen3.5-2B")
    parser.add_argument("--teacher", default="/home/eryx/models/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--manifest", default="baselines/gsm8k_k1/lora_manifest.json")
    parser.add_argument("--output", default="runs/cq_model_preflight.json")
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--check-kernel", action="store_true", help="CPU-only pinned kernel resolution")
    args = parser.parse_args()
    report = {"environment": fp8_environment_report()}
    if args.check_kernel:
        report["fp8_kernel"] = prepare_fp8_kernel()
    if args.environment_only:
        print(json.dumps(report, indent=2))
        return
    require_gpu_guard()
    report["fp8_kernel"] = prepare_fp8_kernel()  # fail before either large model allocation
    student = load_student(args.student, args.manifest)
    teacher = load_teacher(args.teacher)
    st = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    tt = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    report["alignment"] = validate_alignment(student, teacher, st, tt)
    ids = prompt_ids(st, "What is 1 + 1? Answer briefly.", thinking=False)
    # Shared IDs generated exactly once, actual suffix rather than an artificial EOS.
    suffix = st.encode("2", add_special_tokens=False)
    record = Rollout(ids + suffix, len(ids), suffix, "2", "preflight", True, 0)
    batch = build_batch([record], student.device, st.pad_token_id)
    report["prompt_ids"] = ids
    report["student_projection"] = check_logits_equivalence(student, batch)
    report["teacher_projection"] = check_logits_equivalence(teacher, batch)
    named = student.lora_named_parameters()
    report["lora_count"] = len(named)
    report["lora_module_count"] = len(named) // 2
    report["lora_numel"] = sum(p.numel() for _, p in named)
    if report["lora_count"] != 372 or report["lora_module_count"] != 186 or report["lora_numel"] != 8409600:
        raise RuntimeError(f"Real manifest LoRA size mismatch: {report['lora_count']}/{report['lora_numel']}")
    h = student.last_hidden(batch, with_grad=True)
    loss = student.project(h[:, -2]).float().log_softmax(-1)[:, suffix[0]].sum()
    gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    report["unused_lora"] = [n for (n, _), g in zip(named, gradients) if g is None]
    if report["unused_lora"] or not all(torch.isfinite(g).all() for g in gradients if g is not None):
        raise RuntimeError("Missing/nonfinite LoRA gradients")
    report["nonzero_gradient_tensors"] = sum(bool(g.abs().sum() > 0) for g in gradients)
    if report["nonzero_gradient_tensors"] == 0:
        raise RuntimeError("No gradient signal")
    report["teacher_frozen"] = all(not p.requires_grad for p in teacher.model.parameters())
    report["lora_fp32"] = all(p.dtype == torch.float32 for _, p in named)
    if not report["teacher_frozen"] or not report["lora_fp32"]:
        raise RuntimeError("Frozen Teacher/FP32 LoRA contract violated")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

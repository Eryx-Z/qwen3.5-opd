# GSM8K sampled reverse-KL baseline

This is the selected first baseline, separate from the full-vocabulary CQ-OPD specification.

- Frozen teacher: local Qwen3.5-35B-A3B-FP8.
- Student: Qwen3.5-2B, language-only LoRA rank 8, alpha 16, dropout 0; actual module list in `lora_manifest.json`. Embedding/head/vision frozen.
- Existing verl FSDP2 + vLLM, one physical GB10, resident teacher and resident student rollout (`free_cache_engine=false`, `enable_sleep_mode=false` for both). LoRA level-1 sleep failed on this unified-memory machine. A narrow resource-pool validation change permits only unmerged LoRA students to remain resident; the original full-weight restrictions are unchanged. LoRA updates still clear rollout caches after weight synchronization.
- Loss: sampled token k1 (`log p - log q`), detached negative k1 as policy-gradient advantage. `vanilla` is verl's PPO-style clipped ratio surrogate, not direct full-vocabulary KL differentiation. One rollout per prompt, one PPO epoch, one minibatch per optimizer step. No task-reward loss, no CQ, no probe direction, no dynamic KL mixture.
- Unlike the old synthetic smoke, k1 and log-prob clamps are disabled explicitly. Nonfinite results are not acceptable.
- Existing FSDP storage dtype FP32 retained for base and LoRA; mixed-precision compute follows existing engine defaults. AdamW lr 1e-6, default weight decay 0.01, foreach false. This is not the proposed BF16-base standalone trainer.
- Sampling: temperature 1, top-p 1, top-k disabled, vLLM neutral generation config. Thinking disabled consistently through the shared prompt template for this bounded baseline smoke.
- Data: official `openai/gsm8k`, main, pinned revision in `data/gsm8k/manifest.json`; train/probe/dev/test = 6000/500/973/1319, seed 42, normalized question SHA256 checks. Test untouched by training/evaluation. Ground-truth answers only in reward metadata, never in prompts.
- Rules-based score is diagnostic only. Requires explicit `####` or boxed final numeric answer; normalizes integers/fractions/decimals. No arbitrary code execution; parser errors score zero, invalid references raise errors.

## Current integration smoke

Completed 20 optimizer steps on 40 genuine training prompts, batch 2; 512 prompt / 256 response token cap. Four dev records are configured, but validation is disabled (`test_freq=-1`); the trainer reports `Final validation metrics: None`. No test-set evaluation. This is a pipeline check, not a quality experiment. 80% of training responses reached the response cap; a longer budget and initial/final dev evaluation are required for a quality experiment.

Accepted run: `/home/eryx/gk-183212`, exit 0, cleanup OK, 564 seconds including initialization/save/cleanup. Minimum available memory 25.28 GiB; maximum measured ACPI temperature 72.2 C. Checkpoint: `runs/gsm8k_k1_smoke/checkpoints/global_step_20/`. All 372 LoRA tensors are finite FP32, all 186 B tensors became nonzero, and all 372 optimizer states have step=20. See `acceptance-proof.json` and `step-metrics.jsonl`.

Tests: `/home/eryx/verl/.venv/bin/python -m pytest -q tests/test_baseline.py` (13 passed). The six existing single-GPU placement/resident lifecycle/guard regression files pass 109 tests, including four new resident-LoRA cases.

`resident-lora.patch` is the incremental change on top of the previously recorded verl working tree, not a replacement for `snapshots/verl-working-tree.patch`. It is already applied on eryx; do not apply it again. Reverse dry-run verification passed. No installed vLLM package files were modified.

Data preparation supports an offline `--source-dir` holding the official pinned train/test parquet and dataset API `info.json`. This was needed because eryx cannot reach huggingface.co directly; downloads were made on the controlling host, then transferred without changing dependencies or model weights.

Always launch via the existing independent guard (20 GiB available memory floor, 75 C ACPI, 600 seconds):

```bash
RUN=$HOME/gk-$(date +%H%M%S)
/home/eryx/verl/.venv/bin/python \
 /home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py \
 --log-dir "$RUN" --timeout 600 --interval 1 --min-mem-gib 20 --max-temp-c 75 -- \
 bash /home/eryx/qwen3.5-opd/scripts/run_gsm8k_k1.sh
```

Use a fresh checkpoint/output directory for subsequent runs to preserve evidence. The guard may stop a slow 20-step smoke before completion; that is not acceptance. Check `outcome.json`, log step metrics, finite gradients, LoRA synchronization/checkpoint artifacts, and cleanup before declaring success. Only after integration acceptance should longer response budgets, paired initial/final dev evaluation and a 200-step pilot be scheduled.

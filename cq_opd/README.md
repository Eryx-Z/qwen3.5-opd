# CQ-OPD V1.1 implementation

Independent Transformers training entry point for `CQ_OPD_DYNAMIC_CODEX_SPEC.md`. The accepted verl sampled-k1 baseline, its checkpoints, and its test scores are unchanged. This is full-vocabulary dynamic mixed KL, not the baseline PPO surrogate.

## Contract

- Teacher: existing `Qwen3.5-35B-A3B-FP8` checkpoint, frozen, no silent BF16 dequantization. Student: existing Qwen3.5-2B with FP32 language-only rank8/alpha16/dropout0 LoRA. `--student-dtype bf16` preserves the original default; explicit `fp32` uses native FP32 computation with TF32 disabled for scoring, generation, probe and updates. Teacher remains FP8 in both cases.
- Same native Student instance generates, computes probe direction, scores temporary offsets, and trains. No second vLLM Student, CUDA Graph, or framework patch.
- Token-level Teacher entropy mixing: high entropy increases forward-KL weight. Train-only entropy calibration is frozen. All selectors use the same mixed objective.
- Probe: 4 questions × 4 independently sampled categorical responses; native batched generation records batch seed + row index, not an individually seeded replay claim. LOO advantages, summed response logprob, denominator16; returned gradients never overwrite `.grad` or step an optimizer.
- CQ: symmetric LoRA offsets with exact exception-safe restoration; signed per-response Top20% of 16-token blocks. No optimizer state perturbation. No fake EOS or context deletion.
- Frozen heads are projected in chunks (64 positions by default); hidden gradients accumulate raw selected-token loss sums, then one backbone VJP. Across microbatches normalize once by total selected tokens.
- Constant alpha endpoints and full/random/KL/Teachability selectors are numerical/ablation controls. No Router, GRPO main loss, CQ alpha adjustment, or top-K approximation of the mixed KL.

## Running

Use `/home/eryx/verl/.venv/bin/python`, not system Python. The wrapper preserves the existing independent guard (5GiB available-memory floor, temperature telemetry only, 1800s timeout) and refuses overlapping guarded jobs. Keep short unique guard log paths. Hardware protection is unchanged; 5GiB does not prevent instantaneous OOM.

```bash
cd /home/eryx/qwen3.5-opd
# CPU correctness gates: no weights or GPU loads
CUDA_VISIBLE_DEVICES='' /home/eryx/verl/.venv/bin/python -m pytest -q \
  tests/test_cq_core.py tests/test_cq_adapter.py tests/test_cq_trainer.py \
  tests/test_cq_protocol.py tests/test_cq_resume.py tests/test_cq_direction_safety.py \
  tests/test_cq_gates.py tests/test_cq_time_accounting.py

# Native FP8 Teacher + FP32 Student real adapter verification only
bash scripts/run_cq_opd.sh --mode preflight --student-dtype fp32 --output runs/cq_preflight_NEW

# Real entropy mapping + finite-difference calibration; no formal parameter updates
CQ_GUARD_TIMEOUT_SECONDS=7200 bash scripts/run_cq_opd.sh --mode calibrate \
  --student-dtype fp32 --max-new-tokens 2048 --output runs/cq_calibration_NEW

# Only after calibration passes; short correctness smoke, not quality evidence
CQ_GUARD_TIMEOUT_SECONDS=7200 bash scripts/run_cq_opd.sh --mode train \
  --student-dtype fp32 --steps 2 --save-every 1 \
  --batch-size 32 --micro-batch-size 4 --rollout-batch-size 16 \
  --max-new-tokens 2048 --lr 1e-6 --weight-decay .01 \
  --calibration-dir runs/cq_calibration_NEW --output runs/cq_smoke_NEW
```

Use actual fresh paths instead of `NEW`. Startup may instead calibrate and train in one `--mode train` process by omitting `--calibration-dir`, avoiding a second model load. The wrapper retains the default1800s bound; explicit `CQ_GUARD_TIMEOUT_SECONDS=7200` permits a bounded calibration+two-step FP32 start without changing memory or serial-job protections. This does not authorize a long run. Calibration uses the same decoding cap as the training smoke. For a new formal protocol recalibrate with that protocol, do not transfer calibration blindly. Real FD calibration must pass repeated-forward noise, delta/2/delta/2delta sign/rank/overlap checks; an inconclusive CQ calibration fails closed. FP32 startup chooses the smallest candidate with extra interior margin (both rank correlations>=.95, Top20% overlaps>=.9, signs>=.9), rather than a candidate barely passing the minimum gate. Subsequent rechecks retain the original >=.8/.7/.8 gates. No margin candidate means refusal, not a weak/random fallback. Random/full can run independently as diagnostics but must never be reported as CQ.

The compute path fuses Teacher entropy/alpha preparation into CQ utility projection, derives F/R logs from the existing chunk VJP, reuses the already-computed mode-gate hidden tensor, aggregates finite flags at unchanged pre-update boundaries, and caches only small frozen Teacher hidden/meta tensors for fixed FD audits. No full-trajectory vocabulary distribution or Student score is cached. Targeted tests prove utility/mask/gradient equivalence and cache invalidation; timing is recorded separately from correctness.

Every entry mode now performs full-sequence native head/logits parity and explicit eval-versus-scoring mode parity before calibration or training. Passed gate protocol/input fingerprints bind calibration metadata. Resumed LoRA weights are checked again after restoration.

FD is rechecked with the current probe direction before the first two updates, at the global `--delta-recheck-every` interval (default5), at the first valid probe after resume (zero-signal fallback does not discharge this check), and when probe success rate changes by >=.25 or gradient norm by >=2×. A dedicated seed5001 offset creates fixed dev trajectories once; checkpoints preserve them exactly. Rechecks use the original accepted delta and frozen entropy map. Failed stability stops before the update; it does not silently widen delta or random-fallback. Recheck compute is included in timing.

Common settings follow the accepted baseline: response2048/prompt512, effective batch32, microbatch4, native rollout concurrency16, AdamW lr1e-6/weight_decay.01/betas(.9,.999)/eps1e-8/foreachFalse, constant LR/no warmup, clip1, seed42, rank8/alpha16/dropout0. Defaults300 steps/save50 mirror baseline, but bounded acceptance explicitly runs only2 steps/save1. See `BASELINE_ALIGNMENT.json`. FP32 Student compute is a necessary numerical difference from the BF16 baseline, not a hidden equivalence claim.

No stale-direction approximation is enabled. In a fresh same-process start only, the calibration's already-computed version0 probe can be reused for update0 if ALL LoRA values, frozen state, data/protocol, backend and before/after probe RNG match exactly. It is consumed once, never saved or used after an update/resume; all subsequent optimizer steps refresh the probe. Calibration and first-update probe use the existing named seed offset2001; entropy and calibration/audit trajectories use separate named streams. Long training and test evaluation are not automatically launched.

## Artifacts and resume

- `model-manifest.json`, `config.json`: tokenizer/config/LoRA/dataset fingerprints and protocol.
- `entropy_calibration.json`: tau/k/scale, sampled train positions and model metadata hash.
- `delta_calibration.json`: noise, candidate deltas, sign/rank/overlap and validity.
- `calibration_job.json`: whole calibration-job cost including model loading and mode checks, bound to model/entropy/delta hashes. Imported cost is charged once, not again after resume.
- `metrics.jsonl`: losses, utility/selection, probe information, alpha, timing, memory.
- `forward_mode_gate.json`: actual all-position native logits and eval/scoring hidden checks; `resume_forward_mode_gate.json` checks resumed weights.
- `checkpoint-N.pt`: native LoRA tensors, optimizer, constant scheduler identity, RNGs, train order/cursor, calibration, fixed FD audit trajectories/state and cumulative time. Not a verl shard or standalone HF/PEFT adapter.
- `checkpoint-N.pt.timing.json`: UUID-bound completion timing. Preserve it alongside the checkpoint for resume; missing/mismatched timing fails closed.
- `training_summary.json`: cumulative time carried across resumes, initial/imported calibration attribution, and final checkpoint write cost. Step logs are not a substitute for this final total.

Resume into a fresh directory with the same training arguments and `--resume path/checkpoint-N.pt`. Model/config/data/code fingerprints must match; calibrations are restored, not refitted. This fix adds a new mode-gate binding and checkpoint timing contract; old calibration artifacts must be regenerated, and old checkpoints without the timing sidecar cannot be used for cost-valid resume. Named train/probe/selection/data random streams keep question order independent of selector overhead. Directions are recomputed after every update/resume. The exact version0-only transient probe payload is not serialized or restored.

`python -m cq_opd.evaluate --checkpoint ... --output ... --split dev` evaluates original base and CQ checkpoint using matched greedy decoding/thinking-off. It must be wrapped in the independent guard too. Test requires explicit `--confirm-final-test`; do not use exposed test outcomes to choose configurations. The earlier accepted 77.18%→78.85% evaluation is a different model/objective/protocol artifact and must not be relabelled CQ.

## Acceptance boundary

CPU tests prove math, chunk VJP, recovery and tiny-model loop behavior, not GB10 integration or algorithmic improvement. A successful FP8 preflight is not real CQ training proof. A short smoke is not long-run stability or accuracy proof. Independent fixed-dev single-update SGD/AdamW audits and the matched multi-seed 2×2 comparison remain research experiment gates before any efficacy claim or larger pilot.

FP8 Transformers may need additional kernel packages/cache unlike vLLM. They are isolated under `.cq-deps`, not installed into or upgraded in the existing verl environment. Kernel version/cache revision and successful native FP8 tensors must be recorded by preflight before use. Current isolated dependency is `kernels==0.12.3`; upstream `kernels-community/finegrained-fp8` v2 revision is `061130fedf845f320c56de4425f7404f6512c87e`. Required environment: `HF_HUB_CACHE=$PROJECT/.cq-kernel-cache`, `PYTHONPATH=$PROJECT/.cq-deps:$PROJECT`, `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`. Cache source hashes are checked before Teacher loading. These ignored local dependencies/cache must be restored separately on a fresh checkout; no network download occurs inside training. Never substitute the sampled token's logprob, Teacher argmax, or BF16 Teacher silently.

# CQ-OPD baseline handoff

## Status and next-session objective

The user accepted the completed sampled-k1 OPD run as the current baseline. Training and full GSM8K test evaluation finished with exit 0 and successful guard cleanup. Next work is a CQ comparison against this baseline, not rerunning or tuning against the exposed test scores. Communicate in Chinese.

## Accepted baseline

- Remote: SSH alias `eryx`; project `/home/eryx/qwen3.5-opd`; framework `/home/eryx/verl`; Python `/home/eryx/verl/.venv/bin/python`.
- Frozen Teacher: `/home/eryx/models/Qwen3.5-35B-A3B-FP8`.
- Student: `/home/eryx/models/Qwen3.5-2B`, fresh base initialization plus language-only LoRA. Rank 8, alpha 16, dropout 0; 186 modules, 8,409,600 trainable parameters. Vision, embeddings and output head are frozen by explicit target selection.
- Objective: sampled reverse-KL k1, policy-gradient distillation. Actual surrogate is PPO-style clipped ratio (`vanilla` implementation), clip 0.2; one PPO epoch and one minibatch per rollout batch. Distillation coefficient 1; clamps null. No CQ, probe gradient, dynamic KL, entropy bonus or task-reward training loss. This is NOT full-vocabulary reverse-KL and does not satisfy the original full CQ spec.
- AdamW lr 1e-6, constant, weight decay 0.01, betas 0.9/0.999, grad clip 1.0, foreach false. FP32 parameter storage with BF16 computation; FSDP2, gradient checkpointing, SDPA, remove-padding false. CPU parameter/optimizer offload disabled.
- 300 optimizer steps; effective batch/minibatch 32; microbatch 4; one response/question. Approximately 1.6 data passes, 9,600 sampled training responses, NOT 300 epochs.
- Prompt cap 512, response cap 2048; thinking disabled. Training temperature 1, top-p 1, top-k -1, seed 42.
- Student rollout max_num_seqs 16, KV cache 4 GiB; Teacher max_num_seqs 8, KV cache 2 GiB; batched-token limits 4096; model context 2561. Eager inference, TP/EP 1, both engines resident.
- Save every 50 steps. Although config requested latest-two retention, all step-50…300 checkpoint directories were observed; do not assume retention worked or delete them without authorization.
- No dev evaluation in this 300-step run. No early-stop mechanism. Formal comparison cannot select the best checkpoint by dev; evaluated checkpoint is fixed final step 300.

## Definitive training artifacts (remote)

Training run: `/home/eryx/gf-20261005_194228/`
- `resolved-config.yaml`: authoritative configuration; prefer it over conversational parameter summaries.
- `smoke.log`, `guard.jsonl`, `outcome.json`: training, telemetry, exit/cleanup evidence.

Output: `/home/eryx/qwen3.5-opd/runs/gsm8k_k1_300_2048_20261005_194228/`
- `checkpoints/global_step_300/actor/`: model shard, optimizer shard, extra state, LoRA metadata, HF tokenizer/config. Model checkpoint is approximately 8.89 GB and includes frozen parameters; not a standalone HF model directory.
- `checkpoints/global_step_300/data.pt`: data state.
- `rollouts/*.jsonl`: generated training answers.

Run started automatically after a one-hour delay at 2026-10-05 20:42:31 UTC. At step 293 the training timer showed 8h12m; total training took roughly 8.5 hours. Stable steps were about 95–104 seconds, mainly generation (~78 seconds versus ~16 seconds update). Last step loss 0.2340347, grad norm 0.5750514. Guard child exit 0, cleanup OK, no remaining processes. Available memory recovered to 118 GiB. These are integration/stability facts, not accuracy evidence.

Launch chain under project `scripts/`:
`delayed_gsm8k_k1_300.sh` → guard → `run_gsm8k_k1_300.sh` → `run_gsm8k_k1_throughput100.sh` → `run_gsm8k_k1_pilot.sh` → `run_gsm8k_k1.sh` → existing verl launcher. Later Hydra overrides win; reading the base smoke defaults alone is misleading.

## Dataset and final evaluation

Pinned official GSM8K main revision: `740312add88f781978c0658806c59bc2815b9866`, seed 42. Train/probe/dev/test = 6000/500/973/1319, question-hash overlap checks passed. References never enter generation prompts. Source parquet was downloaded locally and transferred because remote HF access failed.

Data: `/home/eryx/qwen3.5-opd/data/gsm8k/`.
Preparation, manifests and tests: `scripts/prepare_gsm8k.py`, `baselines/gsm8k_k1/data_manifest.json`, `tests/test_baseline.py` (13 passed).

Full test evaluation: both original Student and final LoRA on all 1,319 test questions, identical greedy decoding, thinking off, seed 42, 2048 max new tokens. Strict `baseline/verifier.py`: final `####` or boxed numeric answer; fraction/decimal normalization; missing/unparseable final answer scores zero. These scores are protocol-specific, not automatically comparable with external GSM8K leaderboards.

| Metric | Original Student | Step 300 LoRA |
|---|---:|---:|
| Correct | 1018/1319 | 1040/1319 |
| Accuracy | 77.1797% | 78.8476% |
| Truncation | 3.1084% (41) | 5.3071% (70) |
| Unparseable | 4.9280% (65) | 5.6861% (75) |
| Mean output tokens | 422.812 | 442.786 |

Paired: 106 wrong→right, 84 right→wrong, net +22 correct = +1.6679 percentage points. No statistical significance claim established. Training loss reduction alone is not quality proof. Test has now been exposed; do not use these outcomes to tune/select future models.

Authoritative eval output:
`/home/eryx/qwen3.5-opd/runs/gsm8k_test_step300_20261006_061222/`
- `summary.json`: complete scores and paired counts.
- `base.jsonl`, `step300.jsonl`: per-question responses, IDs, reference, correctness, parseability, lengths and finish reasons.
- `manifest.json`: test parquet SHA256, model/checkpoint paths and evaluation protocol.
- `adapter/adapter_config.json`, `adapter_model.safetensors`: extracted step-300 adapter; 372 finite tensors, full language-only targets, r8/alpha16. Export checked single-rank DTensor local/global shapes and nonzero B tensors; config/tensor roundtrip passed.

Eval guard/logs: `/home/eryx/ge-20261006_061222/`; outcome child exit 0, cleanup OK.
Evaluator: `scripts/eval_gsm8k_test.py`; wrapper `scripts/run_gsm8k_test_eval.sh`. It uses one vLLM base engine: first base requests without adapter, then requests with exported adapter. Progress persists every 64 questions.

The first eval `/home/eryx/ge-20261006_060058/` failed before scoring. Wrapper omitted training env: FlashInfer sampler attempted compilation without `ninja`; default vLLM cache also had a permission warning. Fixed by reusing VLLM_USE_FLASHINFER_SAMPLER=0, spawn, writable VLLM_CACHE_ROOT, venv PATH and existing offline/thread limits. No dependency upgrades or package patch required. Failed artifacts preserved.

A 4096-token reevaluation was discussed but NOT authorized/launched. If pursued, evaluate BOTH models under identical budget and report as an additional protocol, not a replacement chosen for a favorable score.

## Memory/lifecycle changes and safety

Hardware: one GB10 with ~121 GiB unified CPU/GPU physical memory. Prior exhaustion caused SSH loss. Preserve independent guard and do not overlap jobs.

Resident-LoRA fix in `/home/eryx/verl/verl/trainer/distillation/resource_pool.py`: shared-pool policy permits only unmerged LoRA Student to stay resident; original full-parameter restrictions remain. Supports old/new LoRA rank config. Reason: level-1 vLLM LoRA sleep backs up weights to CPU and failed a free-memory assertion on unified memory. Actual root numeric trigger was not instrumented. Adapter updates still synchronize and clear caches. No vLLM assertion bypass. Incremental patch: project `baselines/gsm8k_k1/resident-lora.patch`, already applied, reverse dry-run checked. Existing unrelated verl changes must be preserved. Read `/home/eryx/verl/AGENTS.md` before framework edits.

User explicitly requested removing temperature cutoff and reducing available-memory cutoff 20→5 GiB. Guard supports `--no-temperature-protection`; temperatures and sensor errors are telemetry-only in this mode. Default guard temperature behavior for other runs remains unchanged. Long training/eval use `--timeout inf`; stop is step count or error/signal/memory floor, not wall time. 5 GiB cannot guarantee protection against instantaneous OOM. Hardware thermal safeguards were not disabled.

Guard source: `/home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py`.
Guard tests: `tests/trainer/test_opd_diagnostic_guard_on_cpu.py` includes thermal-disable/memory-preservation cases; guard/allocation tests 19 passed. Earlier placement/lifecycle/guard tests 109 passed before additional thermal cases.

## Earlier documentation and caveats

Project `baselines/gsm8k_k1/README.md`, `STATUS.md`, `acceptance-proof.json`, `step-metrics.jsonl` document the successful 20-step smoke, not the final 300-step results. `PILOT.md` documents earlier abandoned/stopped pilot attempts. Treat these as historical records; use this handoff plus final artifacts for the accepted baseline.

Original CQ design: local `/home/spike/CQ_OPD_DYNAMIC_CODEX_SPEC.md`. It describes dynamic entropy-weighted forward/reverse KL, probe task direction, 16-token chunks and top-20% selection. No accepted CQ implementation exists yet. Do not silently replace that specification with sampled-k1 or call the baseline CQ.

Local `/tmp/cq-baseline-stage/` contains abandoned invalid `baseline/data.py` and unused full-vocabulary loss code. Do NOT deploy this whole directory. Only intentional selected files were copied remotely. No final Git commit was made; inspect git status/diff before work.

## Suggested skills

- `lean-build`: scope the smallest comparable CQ implementation and reuse existing runtime.
- `investigate-first`: diagnose runtime/performance failures before editing.
- `surgical-patch`: narrow fixes with regression tests.
- `verify-and-stop`: validate checkpoints, evaluation artifacts and completion claims.

## Throughput optimization update — 2026-10-06

User approved **Student CUDA Graph + concurrency32 + agent worker1** as the preferred candidate for subsequent CQ experiments. Five-step real-update/save/cleanup validation passed. Mean compute step (first warmup excluded, checkpoint time removed): **79.07s**, **238.30 output tokens/s**, minimum available memory **16.53GiB**. eager16:98.04s/185.51 tokens/s; eager32:90.22s/230.12 tokens/s. Sampled lengths differ, so these are not strict fixed-workload speedup ratios. worker4 provided no clear token-throughput advantage.

Evidence: `runs/speed_20261006_072110/{comparison.json,checkpoint-audit.json}` and per-case configs/logs/rollouts. All four cases: 372 finite FP32 LoRA tensors, 186 nonzero B tensors, 372 finite optimizer states at step5, matching prompt batches, valid rollout probability diagnostics, exit0/cleanupOK. CPU regression: framework120/project17 passed.

Graph requires explicit default-off `+distillation.allow_student_cuda_graph=True`, resident unmerged LoRA Student, `enforce_eager=False`, capture sizes `[1,2,4,8,16,32]`. Teacher remains eager. Incremental patch `baselines/gsm8k_k1/resident-student-graph.patch` is already applied, including runtime dataclass support. Existing vLLM batches continuously; no stale-policy cross-step rollout pipeline was introduced.

Long-response stress suite `runs/speed_stress_20261006_082214/` is still running. eager32 completed two fixed2048-token updates/save/cleanup; second compute step144.12s, minimum available23.37GiB. graph32 is pending completion. `ignore_eos=True` is artificial stress only, not recommended formal training behavior. Do not promote short-run results to long-run stability proof.

Details: `baselines/gsm8k_k1/SPEED_OPTIMIZATION_RESULTS.md`. Preserve original accepted baseline/test results. For new CQ comparisons, use matched runtime settings for baseline and CQ, and dev-only tuning.

## Recommended next actions

1. Read final resolved training config, eval manifest/summary and per-question artifacts; preserve accepted baseline.
2. Define CQ comparison with matched models, data split, sampled-vs-full-vocabulary objective explicitly agreed, training/response budget and decode protocol. Use dev for tuning and model selection, not test.
3. If first doing analysis, inspect truncated/repeated answers from saved outputs without changing reported test results.
4. Keep guard cleanup and unique fresh output paths. Confirm no active remote jobs before launching new work.

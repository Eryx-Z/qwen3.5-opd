# Sampled-k1 CQ transition — not yet accepted

User approved replacing the original full-vocabulary dynamic mixed-KL objective with the accepted baseline's sampled-k1 policy-gradient objective. This is an algorithm change, not an exact optimization of the original specification. Top-k comparison is not enabled.

## Fixed settings

- Original Qwen3.5-35B-A3B-FP8 Teacher, unchanged.
- Response budget 2048, training batch32, microbatch4, rollout concurrency16.
- Baseline vLLM BF16 generation and verl tensor-LoRA IPC synchronization.
- AdamW lr1e-6, weight decay .01, gradient clip1; bounded acceptance before long training.

## Objective and retained CQ logic

At the current native policy, compute sampled next-token log probabilities and freeze `coefficient = clamp(logp_student - logp_teacher, -10, 10)`. Baseline PPO uses advantage `-coefficient`, clip .2 and dual-clip3. Freeze native old log probabilities as the PPO reference. No rollout importance correction is added (baseline resolved config uses `rollout_is: null`, `bypass_mode: false`).

Keep LOO probe direction, signed per-response block selection, and finite-difference calibration/rechecks. For symmetric offsets, use the SAME frozen coefficient and old log probabilities; recomputing k1 on either offset would differentiate a different objective. Teacher entropy/alpha and exact forward/reverse mixed-KL are not this objective. Old full-KL calibration and checkpoints do not authorize this variant.

Sampled scoring still needs vocabulary normalization and model output projection. It avoids full-distribution KL comparison; it does not make output projection disappear.

## Evidence so far

- Parent inspected sampled loss source and reran 55 focused CPU tests, all passed, in an isolated `/tmp/cq-parent-sampled.*` directory on eryx.
- Read-only independent sampled-math review found no concrete issues; caller integration and GPU behavior were explicitly outside that review.
- Earlier guarded vLLM bridge check `/home/eryx/cq-vl-1006-140512/` passed initial and saved nonzero-LoRA sampled-token synchronization checks. Sixteen simple arithmetic responses with max_tokens2048 generated in 3.2484 seconds; actual lengths2–125, so this is NOT GSM8K throughput or a2048-token stress result. No optimizer update. Guard exit0 and cleanup clean.

## Outstanding acceptance

Integration tests, actual sampled-objective GPU preflight, fresh FD calibration, two baseline-sized training updates, immutable paired checkpoints, and real checkpoint resume. Actual long-response vLLM/native logprob discrepancies must be inspected; one-token synchronization parity is not exact long-trajectory equivalence. No end-to-end speedup or robust training-start claim yet. No300-step training authorized by these checks.

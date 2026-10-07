# CQ implementation validation — 2026-10-06

## Review fixes and new real-model gate (2026-10-06)

Four reviewed P2 issues are fixed: FP64 direction normalization before dtype cast with finite/nonzero checks; mandatory native-logits/eval-scoring mode parity in all entry modes and after restored LoRA; current-policy fixed-trajectory FD rechecks; cumulative time and checkpoint/calibration job cost attribution.

FD rechecks run before the first two updates, periodically (default5 steps), on substantial probe-statistic changes, and at the first valid direction after resume. No-signal fallback does not clear the pending resume check. Independent audit trajectories and their state persist in checkpoints. Stability failure refuses the update without retuning delta or weakening thresholds.

Checkpoint publication prepares both files, publishes UUID-bound timing sidecar first, and commits the immutable `.pt` last. Trusted resume checks the sidecar and carries cumulative time. Full calibration-job loading/mode-gate cost is bound to artifact hashes and imported once; final training summary includes checkpoint writes.

Deployed CPU suite: **79 passed**, 2 third-party deprecation warnings (2.97s). New tests cover extreme finite gradient norm overflow/cast underflow; true tiny-native eval/scoring parity and mismatches; mandatory train/calibrate gate refusal; fixed FD trajectory reuse, stable/unstable decisions and pre-update refusal; initial/periodic/resume scheduling including no-signal-to-valid resume; sidecar mismatch and interruption publication; calibration whole-job binding and cumulative cost across resume. Additional regression against pre-second-review-fix code: 4 failed/1 passed; latest suite passes. Independent read-only re-review found no remaining blocking issue within the repair scope.

Latest guarded real FP8 preflight (155s, child/wrapper exit0):

- `runs/cq_preflight_review_fix_20261006T100242_validation/`.
- `/home/eryx/cq-1006-100242/{outcome.json,smoke.log}`.
- `forward_mode_gate.json`: passed. Student and Teacher full-position native-head logits maximum absolute errors **0.0**; explicit eval-versus-scoring hidden maximum absolute errors **0.0**. Thresholds remain .02/.02.
- Native mixed-KL backward: 2 tokens; sum1.08142972; gradient norm40.19434625; 186 nonzero finite gradient tensors; no optimizer update.
- Teacher remains actual FP8 using unchanged pinned kernel; Student BF16 with FP32 LoRA. Cleanup ok/no remaining processes; memory returned to ~118GiB.

The prior entropy/FD artifacts are preserved but no longer match new code/mode-gate bindings. A new successful FD calibration is still required. No calibration retry, CQ update, or long training was run as part of these fixes.

## Earlier completed checks

CPU command (CUDA hidden so installed FLA does not select GPU-only kernels for CPU tensors):

```bash
CUDA_VISIBLE_DEVICES='' /home/eryx/verl/.venv/bin/python -m pytest -q \
  tests/test_cq_core.py tests/test_cq_adapter.py tests/test_cq_trainer.py \
  tests/test_cq_protocol.py tests/test_cq_resume.py \
  tests/test_baseline.py tests/test_speed_summary.py
```

**60 passed**, 2 third-party deprecation warnings. Tests cover full-vocabulary dynamic KL, reverse probability gradients, constant-alpha endpoints, entropy mapping, stable utility equivalence, finite differences on FP64, signed per-response selection, exact exception restoration, frozen-head chunk VJP, unequal microbatch normalization, native tiny dense/MoE Qwen adapter gradients, EOS/masks, cache/RNG isolation, production probe gradients without `.grad` pollution, tokenizer BatchEncoding regression, exact guard transport argument, global dataset IDs, all selector training paths, resume with variable-length trajectories/real probe direction/zero-signal fallback. These are CPU gates, not a real CQ train claim.

Real-model guarded preflight:

- Output: `runs/cq_preflight_20261006T090346_backward_validation/`.
- Guard: `/home/eryx/cq-1006-090346/{smoke.log,outcome.json}`.
- Actual existing FP8 Teacher loaded; no BF16 dequantization; pinned upstream Triton kernel revision `061130fedf845f320c56de4425f7404f6512c87e`, isolated `kernels==0.12.3`.
- Student BF16 base plus 372 FP32 LoRA tensors/186 modules/8,409,600 trainable parameters.
- Same token IDs, vocabulary/special-token alignment, native hidden/head versus standard logits parity passed for Student and Teacher at rtol/atol .02.
- Outputs finite, `[1,1,248320]`.
- Native dynamic mixed-KL chunk backward: 2 selected response tokens, mixed sum **1.08142972**, LoRA gradient norm **40.19434625**, **186 nonzero gradient tensors** (zero A gradients at B=0 initialization are expected).
- `autograd.grad` did not pollute `.grad`; no optimizer update occurred.
- Child exit0, guard exit0; cleanup `ok=true`, no remaining processes/errors; available memory recovered to ~118GiB.

This was a very short-input adapter audit. Its temporary entropy statistics are NOT the formal frozen train entropy mapping, and it does not establish FD rank stability, training throughput, long-sequence stability, or model quality.

## Failed attempts preserved

- `/home/eryx/cq-1006-085749/`: argparse rejected the generic guard's injected Ray/Hydra transport argument before model load. Fixed only by accepting the guard's exact tagged transport value; unknown arguments still fail. Dedicated CPU regression added.
- `/home/eryx/cq-1006-085932/`: both models loaded, then chat template returned BatchEncoding rather than integer IDs in Transformers5.12. Fixed explicit `return_dict=False` and integer-list assertion; native tokenizer regression added.

Both failed runs cleaned successfully. They are not counted as passed preflight.

## Formal calibration: CQ refused (no training)

One guarded run: `runs/cq_calibration_20261006T090843_validation/`; guard `/home/eryx/cq-1006-090843/`. Total wall time 577s; child exit1 with `finite difference calibration inconclusive; CQ training refused`. Cleanup succeeded with no remaining processes/errors; available memory recovered to ~118GiB.

- Entropy: 32 initial-train questions, 7,687 response positions; tau=0.00253079599, scale=0.347015589, k=2.881714915; 272.42s.
- Probe: success .125, informative question fraction .25, gradient norm 21.57381. This was NOT a no-signal failure.
- FD: all six rho candidates 1e-5/3e-5/1e-4/3e-4/1e-3/3e-3 failed rank, Top20 overlap or sign gates. At the largest rho, delta=.06681204, correlations were .42962/.66349, overlap .375/.5, sign agreement .5625/.6875 for delta/2 and 2delta. No accepted candidate: `delta=null`, `cq_valid=false`.
- Repeat-forward noise was zero and signal fraction1.0, but that does not prove differential rounding error is zero. BF16 numerical resolution is a hypothesis, not an established root cause.
- No optimizer update, CQ smoke, long training, or test evaluation was run. Do not relax the gates or silently substitute random selection to turn this refusal into success.

## Remaining real experiment gates

Entropy calibration completed, but independent finite-difference calibration still must pass before CQ updates. A calibration refusal or zero probe signal must not be renamed CQ success. Then run bounded short real updates/checkpoint/cleanup, independent fixed-dev SGD/AdamW single-update audits, and matched multi-seed 2×2 comparisons before any efficacy claim or larger pilot.

Old sampled-k1 training and test results are preserved, not relabelled as CQ. No dependency upgrade or framework patch was made to the accepted verl environment. Big model shard bindings are index SHA256 plus shard size/mtime, explicitly not full content hashes. Native checkpoint `.pt` files are trusted-only inputs.

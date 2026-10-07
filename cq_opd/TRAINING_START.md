# Training-start optimization and acceptance

## Objective and limits

Start CQ smoothly and robustly on the existing GB10. Keep the original FP8 Teacher, full-vocabulary mixed KL, same native Student instance, fresh per-step probe direction, and fail-closed FD checks. Preserve the sampled-k1 baseline. This is startup/safety evidence, not accuracy or long-run convergence evidence.

## Verified changes

- Explicit native FP32 Student option; FP32 LoRA and TF32 disabled. Default BF16 path remains available, but its six tested FD candidates failed.
- Fuse Teacher entropy/alpha preparation with utility projection, using exactly the same frozen alpha in both offsets and the update.
- Compute detached forward/reverse logs inside the existing chunk VJP; no second head/KL pass for logs.
- Reuse scoring hidden states in the mode gate; combine finite flags without removing pre-update refusal.
- Cache only small frozen Teacher hidden/entropy/alpha tensors for fixed FD audits. Cache keys bind trajectories, masks, calibration, Teacher versions and precision. Student scores and full-trajectory vocabulary distributions are not cached; resume rebuilds the cache.
- FP32 startup requires extra rank/overlap/sign margin (.95/.9/.9); ordinary rechecks retain .8/.7/.8. Select the smallest robust tested candidate; refuse if none exists. This heuristic does not replace updated-policy rechecks.

## Concrete proof

- Deployed CPU gate: **142 passed**, 2 external deprecation warnings (3.70s).
- Native FP32 preflight: `runs/cq_fp32_preflight_20261006T102444_validation/`; child0, cleanup ok. Both models' native-head and eval/scoring errors0; 2-token mixed-KL backward finite, gradient norm40.09269,186 nonzero tensors. No optimizer update.
- Controlled original Student BF16 vs promoted FP32 diagnostic: `runs/cq_precision_diagnostic_20261006T100939_validation/`; BF16 0/6 FD candidates pass, FP32 4/6. Same trajectories, direction and FP8 Teacher. FP32 audit49.29s versus17.02s; peak allocation50.13GB versus44.20GB. This diagnoses the precision blocker, not a global speed improvement.
- Native FP32 fixed-context compute benchmark: `runs/cq_compute_benchmark_20261006T103829_robust_validation/`. Selection, utility, component losses and backbone gradients are **bit-equal** before/after fusion/logging changes. Three interleaved repetitions: median2.15312s→2.02984s (1.061×,5.7% less elapsed). Teacher projected rows1280→640. FD recheck4.63146s cold→4.16523s cached (10.1% less elapsed). All recheck gates pass at delta0.0066812. These are compute-phase timings, NOT end-to-end training speed.
- Earlier smallest-delta benchmark refusal is preserved: `runs/cq_compute_benchmark_20261006T102742_validation/`, guard `/home/eryx/cq-1006-102742/`; overlap .625<.7. No update occurred. No threshold was lowered.

The promoted diagnostic and native FP32 loading are not interchangeable calibration protocols. The source Student contains596 BF16 and36 FP32 tensors, so native FP32 preserves values that a BF16 load can round. New production entropy/FD calibration is mandatory; old maps appear only in explicitly labelled no-update diagnostics.

## Live startup verification (not yet accepted)

`runs/cq_fp32_start_20261006T104139_validation/` was launched with fresh calibration, native FP32 Student, batch2,2 steps, save every step, response cap256, probe4×4, chunk64, block16, keep ratio.2,lr1e-6. Guard timeout3600s; floor5GiB and serial/flock protections unchanged. The request to use cap512 arrived after this command started; the running protocol was not interrupted or changed.

Required before acceptance:

1. Fresh bound entropy mapping and robust FD calibration pass.
2. Two actual CQ updates, not random/no-signal fallbacks; accepted delta unchanged and both current-policy rechecks pass.
3. Finite losses, gradients, LoRA and optimizer states; measurable parameter changes.
4. Checkpoint/timing sidecars, final cost summary, and real resumed update with restored RNG streams.
5. Phase times, memory headroom, and guard cleanup.

Response cap256 is a bounded smoke setting. High truncation or no-signal evidence would require a newly calibrated longer-response protocol; it must not be called a quality-ready setting. Do not start long training or evaluate exposed test data automatically.

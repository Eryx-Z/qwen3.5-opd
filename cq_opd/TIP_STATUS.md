# TIP-inspired adaptive top-k distillation

Status: deployed after source-hash verification and backup; bounded GPU preflight, five updates and checkpoint1 resume completed (2026-10-07). Exact replay holds within the saved window, not across fresh vLLM rollout generation. Full suite353 passed (2 external SWIG warnings), isolated `eryx:/tmp/cq-tip-cpu.vZnfgK/{pytest.log,results.xml}`. Seven TIP tests cover score/gradient references, stable ties, real EOS/padding, fresh masks, microbatch invariance, window2/4 call counts, no CQ calls, exact CPU mid-window resume and stale-mask refusal. Python compilation passed.

Default new CLI: `--objective adaptive_topk --selector tip --keep-ratio .5 --reuse-window 4 --distillation-topk 64`. Batch32 / micro4 / rollout32 / response2048 unchanged. Explicit `--selector cq` retains the previous CQ route; sampled-k1/mixed-kl retain CQ default and reuse1. TIP is rejected on those objectives.

## Meaning

On each current Student forward, compute Student entropy/log(K) and reverse KL on the Teacher's fixed top-K support. Both distributions are renormalized within that support. For each response separately, cap entropy at its98th percentile, min-max normalize entropy and reverse KL, and compute `h+d-h*d`. Constant ranges map to zero. Detach scores and select floor(ratio*N), at least one for nonempty responses. Stable ties prefer earlier positions. Prompt/pad are excluded, real response EOS remains eligible. Input sequences are never pruned.

This is TIP-inspired, not a full-distribution reproduction: original TIP uses batch entropy clipping; this variant normalizes per response for microbatch invariance. Conditional entropy may miss Student probability outside Teacher support. Selected positions still optimize the original adaptive forward/reverse conditional KL with frozen Teacher-entropy alpha, not TIP's pure reverse KL. Loss raw sums are divided once by total selected tokens across microbatches. No PPO ratio/clipping is added; existing gradient-norm clipping remains.

## Cost and reuse

Rollouts and Teacher targets/entropy/alpha are cached across four optimizer updates. Student-dependent scores/masks are recomputed EVERY update and are not saved in the replay snapshot. `selection_source_step` and `selection_age=0` distinguish current scores from `window_source_step/window_age` for old rollouts.

One current Student backbone forward per microbatch supplies both scores and backward hidden VJP. Scoring projects gathered K head rows for ALL valid positions under no_grad, then selected head rows are projected again for differentiable VJP. This duplicates selected head work, not the backbone; bounded chunk tensors avoid full-vocabulary Student logits. Loss masking does not promise proportional Transformer speedup.

TIP performs no probe or parameter perturbations/FD calibration. Teacher entropy calibration and numerical forward/backward precision checks remain. Delta metadata explicitly says CQ not performed; no CQ-valid success is fabricated. Resume carries Teacher targets and rollout state and recomputes current selection deterministically; stale selection tensors in TIP replay are refused.

## Primary method

TIP: https://arxiv.org/html/2604.14084v4#S6

## Bounded GPU evidence (2026-10-07)

Live backup: `runs/cq_tip_before_20261007_050014/before.tgz`, with downloaded source SHA256 manifest checked against live before deployment. Only trainer/replay_window/topk_trainer, TIP test, and TIP_STATUS/REUSE_WINDOW docs deployed; no algorithm fixes during validation.

- Preflight: `runs/tip_preflight_20261007_0500`; Guard `/home/eryx/cq-1007-050020`. Eight-token numerical gradient cosine0.99902324, norm-relative0.00550536; one-token vLLM sync error0.01026398. This short audit alone is not2048-token acceptance.
- Fresh train: `runs/tip_window5_20261007_0505`; Guard `/home/eryx/cq-1007-050404`. Five optimizer updates, saveevery1, response cap2048/batch32/micro4/concurrency32/top64/reuse4/TIP50%. Fresh entropy tau0.09939120, scale0.78003758. CQ explicitly unperformed (`cq_valid=false`, `performed=false`), no probe/FD scoring.
- Step wall seconds:110.371,33.505,33.550,33.678,105.455. Fresh rollout only steps1/5; warm steps2–4 generation/Teacher time exactly0. TIP scoring0.187–0.264s per update. Selection age0 throughout, window ages0/1/2/3/0. Tokens25475 with12730 selected in first window;20400 with10194 selected at refresh. Peak allocated45,198,149,120 bytes (~42.09GiB), not total unified-memory occupancy. Full run603.730s including startup286.275s and checkpointing.
- Resume: `runs/tip_resume1_20261007_0515`; Guard `/home/eryx/cq-1007-051431`. Resumed checkpoint1 through step5. Steps2–4 have identical saved rollouts and exactly identical LoRA tensors (max abs difference0). Step5 fresh generation differs and final max LoRA difference1.519606e-6; no bitwise cross-refresh replay claim. Selection masks are deliberately not persisted, so actual per-position GPU mask equality was not separately observed. Evidence `resume_comparison.json`. Operational resume succeeded.
- All three Guards child_returncode0, cleanup ok/errors[]/remaining[]; no active job after validation.

Remaining: full-token/random50%/TIP50% matched quality+wall-time comparison; investigation of fresh-generation reproducibility if exact GPU replay is required. These are bounded successful updates, not convergence/quality evidence or a controlled speedup comparison. No300-step run is authorized by these tests.

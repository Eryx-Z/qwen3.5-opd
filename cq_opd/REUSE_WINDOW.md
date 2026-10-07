# Adaptive top-k: bounded reuse windows

Latest staged default is TIP, not CQ. See `TIP_STATUS.md` for selection semantics and pending GPU gates.

## TIP windows (current default)

Reuse rollouts and frozen Teacher targets/entropy/alpha for four optimizer updates. Recompute current Student entropy, KL, Soft-OR scores and selected token masks on EVERY update, using the necessary gradient-bearing backbone forward. TIP snapshots store no score/mask tensors. Score age is always zero; rollout age is separate. No probe, direction or FD calibration is performed; Teacher entropy calibration and numerical precision gates remain. Selected adaptive mixed-KL gradients are recomputed normally, without PPO clipping.

## CQ windows (explicit legacy selector)

The following describes `--selector cq` only, not the current TIP default. This CQ route is preserved rather than silently relabeled TIP.

## Authorized objective

Direct forward/reverse KL on the Teacher top-64 conditional support, both distributions renormalized on that support. The forward-KL coefficient is a detached sigmoid of Teacher top-k entropy using a frozen median/IQR calibration. This is not full-vocabulary KL or full-vocabulary entropy. No PPO probability ratio or PPO clipping is used for this objective. Gradient-norm clipping remains a separate numerical safeguard.

## Window semantics

A window is measured in optimizer updates, not seconds. Initial proposal: four updates, with two and four to be validated. Response budget2048, batch32, microbatch4, rollout concurrency32 remain unchanged.

At the start of a window:
1. Generate a fresh training batch and compute a current-policy probe direction.
2. Perform required FD validation without relaxing thresholds.
3. Compute frozen Teacher top-k targets, entropy/alpha, CQ scores and selected token masks.

For remaining updates in the window:
- Reuse exactly those responses, Teacher targets, CQ scores and masks.
- Recompute current Student hidden states, loss and gradients on each update.
- Never cache Student graphs, hidden states or gradients across optimizer updates.
- Label CQ scores with the source optimizer step and actual age. They are intentionally stale, not current-policy CQ.

At expiry, discard the window and refresh all its data. A failed fresh probe/FD check must not silently extend the old window.

## Checkpoint semantics

Persist the frozen window data and source/expiry with normal model/optimizer/RNG/checkpoint bindings. Mid-window resume must use the saved window, not silently regenerate or relabel scores as fresh. Validate shapes, finite values, source/expiry, selected masks and protocol/model/data/calibration bindings before updating. Resume model/precision checks remain required; current-direction FD checks occur at refresh, where a current direction exists.

## Acceptance still required

- Exact fixed-target loss/VJP and direct selected-head projection references.
- Window2/window4 call counts and expiry behavior.
- Mid-window CPU resume equivalence, stale/tampered state refusal.
- Full regression suite.
- Guarded GPU adaptive-top-k preflight, fresh entropy/FD calibration and bounded real window runs.
- Honest cold-refresh vs warm-reuse timings and memory; real checkpoint resume.

The existing sampled-k1 GPU results do not accept either new algorithm. TIP additionally needs its fresh-mask CPU tests and actual CUDA preflight/window/resume validation; no FD calibration is required for TIP. No300-step run follows automatically from implementation.

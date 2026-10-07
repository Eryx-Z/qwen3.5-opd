# 单卡 OPD 吞吐优化验收范围

用户目标：监控进展，尽可能优化，为后续正式算法实验建立可靠运行基础。

不把有限短测称为全局最优；本轮验收是对用户明确提出的三类优化建立可复核证据、选出测得的可靠配置，并保留下一步实验边界。

## 当前结果

五步四组训练已完成，Student CUDA Graph + 并发32 + worker1 为用户认可的优先候选：扣除预热/保存79.07s每步、238.30输出token/s。详细证据与局限见 [`SPEED_OPTIMIZATION_RESULTS.md`](SPEED_OPTIMIZATION_RESULTS.md)。固定2048-token长回答Graph压力测试尚在运行，不能把整体优化验收标成完成。

## Deliverables / evidence checklist

1. Monitor serial GPU suite to completion, record each outcome/cleanup and progress. Evidence: suite.log and per-case guard outcome/telemetry. Never overlap experiments.
2. Compare concurrency16/32 and agent-worker1/4, explicitly establish existing continuous batching via source. Evidence: resolved config, fixed_train SHA/IDs, per-step rollout/timing, source lines. Do not introduce stale-policy cross-step pipelining.
3. Exercise CUDA Graph with real LoRA training weight updates and checkpoint save, not only initialization. Evidence: capture logs, 5 successful update metrics, checkpoint tensor/optimizer audit. If unsupported, preserve failure and diagnose narrow configuration fix before retry; no package/dependency upgrades by default.
4. Correct timing: exclude first step warmup; remove checkpoint save time from steady update means; report generation/teacher-scoring stage and output-token-normalized throughput. Report different sampled lengths and lack of repeated-seed confidence.
5. Choose a tested configuration and provide reproducible fresh-path launch command for later experiments, without running full training or changing accepted baseline/test scores.
6. Produce project optimization report and update HANDOFF with paths, tested invariants and known limits. CPU checks and actual GPU artifacts must cover the claim; checkpoint existence alone is not enough.

Non-goals: changing loss, data split, answer budget, LoRA capacity, or original accepted baseline; task-reward training; CQ implementation; tuning on exposed test results; asserting statistical significance or global optimality; disabling hardware protections.

Memory cutoff remains user-requested 5 GiB, temperature telemetry-only. Safety margin target is larger than 5 GiB; fast transient OOM remains possible. CUDA Graph must fit together with resident Teacher and training model.

# 单卡 OPD 吞吐优化记录

## 当前建议（2026-10-06）

用户同意将 **Student CUDA Graph + 并发 32 + agent worker 1** 作为后续实验的优先候选。五步真实训练验收已通过；固定长回答压力对照尚在运行。当前不是全局最优或长期稳定性证明。

| 配置 | 排除预热、保存后的整步均值 | 整步输出 token/s | 最低可用内存 GiB |
|---|---:|---:|---:|
| eager16 / worker1 | 98.04 s | 185.51 | 17.79 |
| eager32 / worker1 | 90.22 s | 230.12 | 19.49 |
| eager32 / worker4 | 85.71 s | 199.19 | 16.21 |
| graph32 / worker1 | **79.07 s** | **238.30** | 16.53 |

worker4 的回答更短，不能只看整步秒数判定收益；token 吞吐没有优于 worker1。各组回答长度不同，不能把以上差值全归因于配置，未做重复种子或显著性检验。generation 计时包括 Teacher scoring / 后处理，不是纯 Student decode。

## 可复核证据

- 五步对照：`runs/speed_20261006_072110/`，`comparison.json`、`checkpoint-audit.json`（实际审计输出文件名以脚本为准）、每组 resolved-config、rollout、guard outcome / telemetry。
- 每组相同 160 道冻结 train 问题，batch32 / microbatch4，2048 response cap，seed42，fresh base+LoRA；不使用 test 调参。
- 每组五次有限 loss / 正梯度更新，372 个有限 FP32 language-only LoRA 张量，186 个非零 B 张量；372 optimizer states 到 step5，全部有限。相同 prompt batches，rollout/actor 概率诊断有效；exit0、cleanupOK。
- Graph 的 FULL / PIECEWISE 捕获日志和真实同步后训练已验证，非仅初始化成功。
- CPU 框架生命周期/配置/guard 六组回归 **120 passed**；项目 summary / baseline tests **17 passed**。
- 两次 Graph 前置失败分别为共享池默认 eager 限制和 dataclass 缺少新字段；原始日志保留，不冒充成功 GPU 测试。

## 运行设置与边界

Student: `max_num_seqs=32`, `agent.num_workers=1`, `enforce_eager=False`, `cudagraph_capture_sizes=[1,2,4,8,16,32]`。其余保持 baseline（Student KV4 GiB / Teacher KV2 GiB，Teacher eager/concurrency8，batch32/microbatch4）。

实验 opt-in：`+distillation.allow_student_cuda_graph=True`。默认关闭，仅允许共享池 resident、unmerged LoRA Student；不允许 Teacher Graph 或不安全 sleep / 全权重组合。框架增量补丁：`baselines/gsm8k_k1/resident-student-graph.patch`，已应用；勿重复应用。含 runtime dataclass 字段及 Hydra 实例化回归。

已有 vLLM 连续批处理；`agent_loop.py` 对批内请求 create_task 后 gather。worker manager 分发请求；多 worker 未得到明确收益。不做跨训练步预生成，以免引入 stale-policy rollout。

温度只记录，available memory floor5 GiB；独立 guard、串行 GPU 运行和清理验收保留。5 GiB 不能防止瞬时 OOM。

## 固定长回答压力对照（未完）

`runs/speed_stress_20261006_082214/`：eager32 / graph32 各两步，固定 ignore_eos=True，使每条生成2048 token。仅用于吞吐/峰值压力，不是正式算法训练或模型质量结果。与正常 EOS 停止的实验协议分开记录。

已测 eager32：两步完成、保存成功、exit0/cleanupOK；第二步扣除保存144.12s，454.74输出 token/s，最低可用内存23.37GiB。graph32 正在运行；须完成保存、张量/optimizer审计和清理后再更新结论。

## CQ 公平比较

原 accepted 300-step baseline、checkpoint 和 test77.18%→78.85%证据保持不变。后续 CQ 与对应 baseline 应使用相同运行配置、数据、步数、token预算及评估协议；不能只给 CQ 换快配置后把 wall-time 差称为算法收益。用 dev 选择配置/模型，不再用暴露的 test 调参。

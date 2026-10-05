# Qwen3.5 单卡 OPD 环境记录

这是一个轻量的环境与模型记录仓库，方便后续在 **eryx** 上开展 on-policy distillation。
不包含模型权重、数据集或 Python 环境；不复制整个 verl，也不自动安装或启动训练。

## Teacher / Student

| 角色 | 模型 | eryx 上的路径 |
|---|---|---|
| Teacher | Qwen3.5-35B-A3B-FP8 | `/home/eryx/models/Qwen3.5-35B-A3B-FP8` |
| Student | Qwen3.5-2B | `/home/eryx/models/Qwen3.5-2B` |

Teacher 使用 checkpoint 自带的 FP8 量化元数据，不额外强制设置 `quantization=fp8`。
Student 推理使用 BF16；已验证的训练配置保留 FP32 参数和 Adam 状态。
两个模型的 `tokenizer.json`、`vocab.json`、`merges.txt` 一致；`tokenizer_config.json` 不同，各自保留原文件。配置、tokenizer、分片索引的 SHA-256 见 `environment.json`；没有对所有大权重分片重新计算哈希。

## 已验证环境

- 机器：`eryx-spark`，Linux `aarch64`，**一张 NVIDIA GB10**，约 121 GiB CPU/GPU 统一内存。
- verl 工作目录：`/home/eryx/verl`。
- Python 环境：`/home/eryx/verl/.venv`，Python **3.12.3**。
- NVIDIA 驱动：**580.173.02**；PyTorch CUDA：**13.0**。
- PyTorch：**2.13.0+cu130**；vLLM：**0.29.0**；verl：**0.10.0.dev0**。
- Transformers：**5.12.1**；Ray：**2.55.1**；FlashAttention：**2.8.3**；uv：**0.12.23**。

完整已安装版本见 `snapshots/installed-packages.txt`，原始锁文件见 `snapshots/uv.lock`。
这些是 ARM64/CUDA 环境记录，**不是任意机器上直接可安装的通用 requirements**。后续环境管理继续使用 uv。
当前环境已经可用，不需要再次 `uv sync`、升级依赖或下载模型。

### 必须保留的单卡配置

- Teacher 和 student 使用同一个真实 GPU placement group，不伪造第二张 GPU。
- Teacher 常驻：`free_cache_engine=false`、`enable_sleep_mode=false`。
- Student rollout 保留 `enable_sleep_mode=true` 和权重同步。
- Student Transformers attention 使用 **SDPA**，`use_remove_padding=false`；已装的外部 FlashAttention wheel 不支持 GB10 所需内核。vLLM 自身 attention backend 不变。
- 训练参数、优化器的 CPU offload 均关闭；AdamW `foreach=false`；reward worker 为 1。
- Teacher KV cache **512 MiB**，student KV cache **256 MiB**；两者均 eager，TP/DP/PP/EP 为 1。
- 使用 critic-free GRPO，不启用 reference/reward 模型；这里的 reward worker 是规则评分进程，不是 reward 模型。

完整配置快照见 `environment.json`。这些是已验证的小规模 smoke 配置，不代表大 batch、长序列或长训练已通过。

## 后续使用

`environment.env.example` 只记录路径。在 eryx 上可以这样重跑**两条合成数据的一步 smoke**：

```bash
cd /home/eryx/qwen3.5-opd
set -a
source environment.env.example
set +a
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
cd "$VERL_DIR"
export PYTHONPATH="$VERL_DIR/examples/on_policy_distillation_trainer/diagnostics:$VERL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPD_DIAGNOSTIC_DIR="$HOME/od-$(date +%H%M%S)"
"$PYTHON_BIN" examples/on_policy_distillation_trainer/diagnostics/guard.py \
  --log-dir "$OPD_DIAGNOSTIC_DIR" --timeout 600 --interval 1 \
  --min-mem-gib 20 --max-temp-c 75 -- \
  bash examples/on_policy_distillation_trainer/run_qwen3_5_single_gpu_smoke.sh
```

**正式数据集尚未准备。** Smoke 脚本每次都会创建两条 `What is 1+1?` 的合成数据，不是 GSM8K 原始数据。以后正式做 OPD 时，需另选训练/验证集和训练长度，保留上述单卡与安全配置，不要把 smoke 当成正式训练或效果评测。

### 内存保护不能省略

曾经的无保护运行耗尽统一内存并造成 SSH 无响应。所有 GPU 训练必须使用独立 guard：可用内存低于 20 GiB、ACPI 温度达到 75°C、超过 600 秒或传感器读取失败时停止，并清理本次进程。
短日志路径用于避免 Ray Unix socket 的 107 字节路径上限。轮询不能保证防止瞬时 OOM；ACPI 温度也不是 GPU junction 温度。

成功 smoke 的 teacher 转换期仍短暂保留 **83.32 GiB**，使用约 **11.55 GiB swap**；最小可用内存 **27.64 GiB**。关闭 sleep pool 解决的是临时分配留存，不是消除所有瞬时峰值。

## 已有验证

`/home/eryx/opd-log5/acceptance-proof.json` 记录：

- 完成 `training/global_step=1`。
- Actor loss `1.70812988`，distillation loss `1.70812994`。
- 梯度范数 `831.65027`，有限且非零（裁剪前）；学习率 `1e-6`。
- Optimizer step 完成，更新后的权重同步成功。
- Guard 返回 0，清理正常；GPU 空闲，可用内存恢复到 118 GiB。
- 144 项相关回归测试通过。没有验证长训练稳定性或训练质量。

精简的验收记录也保存在本仓库 `snapshots/acceptance-proof.json`。

## verl 代码来源

基础仓库为 <https://github.com/verl-project/verl>，基础提交见 `environment.json`。
已验证配置依赖当前 verl 工作树中的**未提交修改**，不能只 checkout 基础提交就认为代码一致。
`snapshots/verl-working-tree.patch` 保存这次单卡 OPD、resident teacher、诊断与回归测试修改，供以后恢复或审阅。
不要把它重新应用到已经修改好的 `/home/eryx/verl`。在干净的对应基础提交上，先运行 `git apply --check`，再应用。
源代码补丁遵循 verl 的原始许可，见 `snapshots/verl-LICENSE`。

本仓库只记录并备份现有设置，不修改原 verl 仓库、不复制模型、不提交大文件，也未创建远端 GitHub/GitLab 仓库。

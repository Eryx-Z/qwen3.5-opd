# 训练与测试集评估结果报告 (300 步实验)

## 1. 核心结果对比

在官方 GSM8K 完整测试集（1,319 道题，严格匹配 `####` 最终数值答案，greedy 采样 `temperature=0`）上的评测对比结果如下：

| 模型版本 | 正确题数 / 总数 | 测试集准确率 (Accuracy) | 相对 Base 提升 | 相对 verl 基线提升 | 截断率 (Truncation) | 平均输出长度 (Tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.5-2B (Base 基准)** | 1,023 / 1,319 | **77.56%** | - | - | 3.41% | 426.7 |
| **仓库标准 verl 基线 (`gsm8k_k1_300`)** | 1,040 / 1,319 | **78.85%** | +1.29% | - | 5.31% | 442.8 |
| **本次训练 (`Adaptive Top-64 + TIP 300`)** | **1,046 / 1,319** | **79.30%** | **+1.74%** | **+0.45%** | 6.44% | 460.0 |

### 答题表现明细 (Paired Analysis vs. Base)
- **修正数 (Improved)**：**107 题**（原 Base 答错、300 步训练后纠正成功）
- **退步数 (Regressed)**：**84 题**（净提升 +23 题）
- **有效解析率**：93.18%（不可解析格式率仅 6.82%）

---

## 2. 训练配置与吞吐表现

- **训练任务 ID**：`runs/tip300_diagnostic_20261007_074342`
- **核心算法机制**：
  - **自适应目标**：NeMo 风格 Teacher-Top64 截断归一化前向/反向双向混合 KL（根据 Teacher 截断熵动态调整权重）。
  - **Token 筛选**：TIP-inspired（根据 Student 截断熵与反向 KL 的 Soft-OR 分数保留前 50% 最具纠错价值的 token）。
  - **上下文复用 (Replay Window)**：每个 4 步更新为一个周期，第 1 步执行 Rollout 与 Teacher 目标计算，后 3 步仅复用已有序列做直接蒸馏更新（无 PPO ratio/clipping），大幅摊薄生成与前向成本。
- **训练规格**：
  - 硬件：单卡 NVIDIA GB10 (aarch64)
  - 学生模型：`Qwen3.5-2B` (BF16，FP32 master LoRA，FP32 output head)
  - 教师模型：`Qwen3.5-35B-A3B-FP8` (原生保持，无解量化)
  - 批次配置：`batch_size=32`, `micro_batch_size=4`, `rollout_concurrency=32`, `max_new_tokens=2048`
- **运行时间统计**：
  - **总耗时**：**15,888 秒**（约 **4 小时 24 分钟**）
  - **平均单步耗时**：~**52.9 秒 / 步**（刷新步 ~115s，复用步 ~33s）
  - **Loss 变化**：从初始 `0.5338` 稳定单调收敛至 `0.4166`（下降约 22%）
  - **梯度范数**：从初始 `0.8610` 稳步收敛至 `0.4978`
  - **检查点保存**：每 4 步保存一次，共生成 75 个原子持久化检查点，最终检查点为 `checkpoint-300.pt`。

---

## 3. 可复核产物路径

- **训练产物目录**：`eryx:/home/eryx/qwen3.5-opd/runs/tip300_diagnostic_20261007_074342`
  - 检查点：`checkpoint-300.pt`（以及 checkpoint-4 到 296）
  - 步频与训练指标：`metrics.jsonl`
  - 最终训练摘要：`training_summary.json`
- **评测产物目录**：`eryx:/home/eryx/qwen3.5-opd/runs/eval_tip300_20261007_121159`
  - 评测总结：`summary.json`
  - 评测明细：`base.jsonl`、`step300.jsonl`
  - 导出推理 Adapter：`adapter/adapter_model.safetensors`
- **对比基线数据**：`eryx:/home/eryx/qwen3.5-opd/runs/gsm8k_test_step300_20261006_061222/summary.json`

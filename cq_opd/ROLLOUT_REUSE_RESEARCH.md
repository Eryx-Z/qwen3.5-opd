# Rollout 多次更新、不用 PPO clipping：一手论文核查

## 结论先行

**可以复用 Student rollout，且不需要因此引入 PPO ratio / PPO clipping。最直接的论文先例是 DistiLLM 的 replay buffer；DistiLLM-2 则是每个 epoch 预生成一批数据，然后做多个 minibatch 更新。** 两者都直接优化 token 分布蒸馏损失，不是在旧轨迹上硬套 PPO。

但要分清：
- **GKD：**直接 token-level divergence，采样不反传；原始算法每步重新生成，不是 replay 实验证据。
- **DistiLLM：**真正的跨更新 replay buffer，无 PPO ratio clipping、无轨迹 importance correction。
- **DistiLLM-2：**epoch 级刷新，epoch 内固定数据多步训练；不能据此声称每条序列固定重复训练 K 次。
- **MiniLLM：**确实复用数据做 4 inner epochs，但用 importance weights 和 clipping，不能列为“不用 clip”的先例。

当前 CQ 若已切换成 **Student 上下文中的 top64 截断分布正/反向 KL 直接反传**，更接近 GKD / DistiLLM，而不是先前 sampled-k1 sequence-reward PG。沿用之前 PG 的 PPO 解释不合适。

## 1. 核实过的 4 篇论文

### A. GKD：直接条件分布蒸馏的基础，不是 replay 论文

**On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**。Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem。ICLR 2024；arXiv **2306.13649v3**。

来源：[论文](https://arxiv.org/abs/2306.13649v3)、[PDF](https://arxiv.org/pdf/2306.13649v3)。

- §3 Eq. (2)：对固定序列中的每个 prefix，计算 Teacher 与 Student **整个词表条件分布**的 divergence，再按 token 平均。
- §3.1 Eq. (4) 明说：**“we do not backpropagate through the student’s sampling distribution”**。
- Algorithm 1：每个训练 step 以概率 λ 生成当前 Student 序列，否则取固定数据；接着直接求 `∇θ D(pT || pSθ)(y|x)`。
- §3.1 允许 forward KL、reverse KL 等；**reverse KL 并不自动等于 REINFORCE/序列奖励训练**。
- 无 PPO ratio clipping 或轨迹 importance weights；但原始 Algorithm 1 **没有 replay buffer，也没有同批 K 次优化循环**。不能把后者冒充 GKD 原文设置。

原文算法的简化转写：
```text
每个更新步：
    概率 λ：用当前 Student 生成 B；否则从固定数据取 B
    对 B 的固定 prefixes 计算 Teacher/当前 Student 条件分布 divergence
    直接 backward + optimizer.step；不对采样过程反传
```

### B. DistiLLM：最直接的无 PPO clip replay 先例

**DistiLLM: Towards Streamlined Distillation for Large Language Models**。Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun。ICML 2024；arXiv **2402.03898v2**。

来源：[论文](https://arxiv.org/abs/2402.03898v2)、[PDF](https://arxiv.org/pdf/2402.03898v2)、[官方仓库](https://github.com/jongwooko/distillm)。

**机制：**§3.2、Algorithm 1、Fig. 4(c)。原文：**“we randomly draw samples from this pool”**，满容量后 **“replace the oldest samples”**。这是真正的 replay，不只是把 generation batch 调大。

Algorithm 1 的控制流（保留关键概率语义）：
```text
DR = 有限容量 replay buffer
每步 t：
    u ~ Uniform(0,1)
    若 u < φ(1-t/T)：当前 Student 生成一批，加入 DR，淘汰最旧数据
    若 u < φ：从 DR 随机取 minibatch
    否则：从固定数据 D 取 minibatch
    对固定序列计算 S(R)KL 并直接更新 Student
    验证时由 SGO scheduler 更新 φ
```

注意论文把 `ζ = 1-t/T` 称为 replay ratio，但它乘在**新生成/写入概率**上；不要误读为“ζ 越小，越少复用”。原文逻辑是后期新生成减少、更多复用旧 SGO。

- §4：初始 φ=0，buffer capacity **1000**；主设置使用 α=0.1 的 SRKL。
- 没有规定每条序列必须训练 2/4/8 次；随机抽样与淘汰决定实际复用次数。**固定 reuse count：未报告。**
- 官方代码核实：[`buffer.py` 固定提交](https://github.com/jongwooko/distillm/blob/d47e77ff9d27721783b32213be38c1204230cc0a/distillm/buffer.py#L16-L32) 使用 `deque(maxlen=args.capacity)` 和 `random.sample`；保存的是 input IDs、mask、labels，不是 old policy log-prob 或缓存 Teacher logits。
- [`losses.py`](https://github.com/jongwooko/distillm/blob/d47e77ff9d27721783b32213be38c1204230cc0a/distillm/losses.py#L14-L24) 的 reverse KL 对完整词表求和并直接反传；[SRKL 实现](https://github.com/jongwooko/distillm/blob/d47e77ff9d27721783b32213be38c1204230cc0a/distillm/losses.py#L80-L94) 同样如此。**这些蒸馏损失没有 PPO ratio clipping，也没有历史行为策略的 importance correction。**这不等于宣称整个训练不存在梯度范数裁剪等数值措施。

**效果与限制：**
- §5.2：相对 on-policy / mixed strategy，报告 **2.2–3.4×** 训练速度；摘要相对近期 KD 方法的整体比较为 **最高 4.3×**。两种口径不能混写，也不能外推为 CQ 的确定加速。
- Table 4：Dolly ROUGE-L，DistiLLM on→off 为 **26.37→26.12**；GKD 为 **23.75→22.89**。说明复用可能有损失，且与 divergence 选择存在交互，不能声称任意 KL 都无损。
- Appendix E.8 / Table 10：容量 250/500/1000/2000/4000；1000 综合最均衡，但不是每项都最好。原文明确提醒：过小有过拟合风险，过大引入过旧 SGO。

### C. DistiLLM-2：epoch 批量刷新，不应混同逐样本 replay

**DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs**。Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, Se-Young Yun。ICML 2025；arXiv **2503.07067v2**。

来源：[论文](https://arxiv.org/abs/2503.07067v2)、[PDF](https://arxiv.org/pdf/2503.07067v2)、[官方仓库](https://github.com/jongwooko/distillm-2)。

- §2.2 明确：**“collects SGO ahead of every training epoch ... rather than ... every training iteration”**。
- Algorithm 1：epoch 开始用 Teacher 与 epoch 起点 Student 生成响应，构造本 epoch 数据；随后 T 次取 minibatch 更新。
- §3 Eq. (2)：Teacher responses 用 SKL，Student responses 用 SRKL，配上动态 α/β。不是单一 sampled-action policy-gradient reward。

原文算法的简化转写：
```text
每个 epoch e：
    用 Teacher 与 Student_(e-1) 生成响应，建立 Dt
    对 τ=1..T：
        从 Dt 取 minibatch
        更新 αt、αs、β
        直接最小化 Teacher-response SKL + Student-response SRKL
```

**复用粒度：**epoch 内参数变化，而数据仍来自 epoch 起点策略；这支持“固定旧上下文跨多个优化步有效”。但它不是 DistiLLM 的 FIFO 随机 replay 机制，也未报告“同条 rollout 固定复用 K 次”。多 minibatch 步不意味着每条样本重复出现多次。Appendix C / Table 11 的 3/2/2 epochs 是任务总轮数，且每轮刷新；不能当作每条 rollout 的复用次数。

**clip 的精确区分：**Algorithm 1 第13行有 `clip`，但裁的是混合系数 β；Appendix B 的 curriculum 实现还裁剪 α。官方 [`distillm_trainer.py`](https://github.com/jongwooko/distillm-2/blob/fe4cf9bfbb4f83219dd1d69219800164b59685fa/src/distillm_trainer.py#L1138-L1185) 也能看到 α 的 `torch.clip`。**这些不是 PPO importance-ratio clipping。**不要笼统说“全文完全不用任何 clip”。论文与所查直接 divergence 实现没有为 rollout 陈旧性引入轨迹 importance weighting。

**实证：**Appendix D.1 / Table 12，GPT-2 + Dolly ROUGE-L：

| 方法 | 每步 on-policy | batched on-policy | off-policy |
|---|---:|---:|---:|
| DistiLLM-2 | 26.37 | 26.20 | 26.13 |
| GKD | 23.75 | 23.21 | 22.89 |

作者报告 batch generation 降低收集开销，便于使用 vLLM。**Table 12 本身没有给出可直接引用的速度倍率**；质量小幅下降是特定小模型/数据集证据，不是 Qwen2B←35B 保证。

### D. MiniLLM：重要反例——复用，但不是无 clip

**MiniLLM: On-Policy Distillation of Large Language Models**。Yuxian Gu, Li Dong, Furu Wei, Minlie Huang。arXiv **2306.08543v6**（2026-01-31 修订版；早期文献常简称 *MiniLLM: Knowledge Distillation of Large Language Models*）。本次标题与算法按实际抓取的 v6 PDF 核对，不把不同版本标题强行混用。

来源：[论文](https://arxiv.org/abs/2306.08543v6)、[PDF](https://arxiv.org/pdf/2306.08543v6)。

- §2.2 Eq. (3)：sequence reverse-KL 梯度分成单步词表求和项与长期回报 PG 项，不应简单描述为全是 sampled-k1。
- Teacher-mixed sampling Eq. (4)；Eq. (5) 引入 importance weights。正文说明前缀乘积方差高，实际近似为单 token 比率。
- §2.3 / Algorithm 1：长期回报更新包含 `min[ρ, clip(ρ,1-ε,1+ε)]`；原文明说 **“with a clipping strategy ... added to further improve stability”**。
- Appendix B.1 Training Details / Phase 2：**一次收集 256 sentences，4 inner epochs，ε=0.2**。

因此 MiniLLM 是“rollout 多次优化”的先例，但**不能用来佐证“不用 PPO clip”**。importance correction、ratio clipping 与直接 token KL 是不同层次的选择，不要统一归类。

## 2. 对当前 adaptive-top64 CQ 的可行推导（不是论文原实现）

设旧 Student 生成的 prefixes 集合为 B；Teacher 固定，Teacher top64 支持为 S(h)。可在 B 上定义：

`L_B(θ) = mean_h [ a_h KL(pT^S || qθ^S) + (1-a_h) KL(qθ^S || pT^S) ]`。

这里 pT^S/qθ^S 是约定支持 S 上的截断、归一化分布；若当前 CQ 有不同质量保留/尾部定义，应严格沿用实际定义。**top64 精确仅指选定截断目标，不等于全词表精确 KL。**

- B 固定时，每一步重新计算当前 Student logits，再对这项经验条件蒸馏损失反传，是有效的直接梯度。**不因 B 来自旧策略就必须乘 old/new policy ratio 或加 PPO clip。**
- 但 B 的状态分布不再等于当前 Student 的状态分布；应称 **replay / batched off-policy conditional distillation**，不能宣称是当前 on-policy sequence KL 的无偏梯度。
- 即使重新生成，GKD 原算法也主动不对采样分布反传；不要把它和“完整求导 sequence reverse-KL”混为一谈。若坚持估计当前策略状态分布下的目标，才涉及另外的分布校正问题，单 token ratio 也不会自动解决全部 prefix 分布偏移。
- **Teacher 可缓存：**同一冻结 Teacher、相同 prefix、相同数值/温度设定下，Teacher topk IDs、log-prob、相关统计可缓存。Teacher entropy 若需全分布熵，必须首次正确计算保存，不能假装 top64 就给出精确全词表熵。论文 buffer 本身没有证明或实现这种 Teacher cache 优化。
- **Student 必须刷新：**当前 logits、Student entropy、与 Student 有关的 CQ score / KL direction / 混合权重应按设计重新算。若 CQ gate 仅由冻结 Teacher 决定才可一起缓存；若支持集含 Student topk，也不能把整个支持永久冻结后仍声称目标不变。离散 direction/权重是否 stop-gradient 要明确，不能无意中改变算法。

最小验证方案（建议，不是已部署实现或论文最优超参）：
```text
1. 生成一批 Student rollout，固定 tokens/prefixes。
2. Teacher 对这些 prefixes 算一次并缓存必要 top64/熵/质量统计。
3. 对该批做 K 次优化：每次重跑当前 Student，刷新依赖 Student 的 CQ 量，
   直接算原截断 mixed-KL，反传；不加 PPO ratio/clipping。
4. 到 K 或预设的陈旧性/漂移阈值后重新生成。
```

先比较 **K=1/2/4**（探索值，不是论文推荐），不急着搭大型异步 buffer。要同时报告：
- 相同 wall-clock 的验证质量，以及相同 optimizer steps / 训练 tokens 的质量；
- generation、Teacher forward、Student forward/backward 各自耗时；复用不能省 Student backward；
- 新鲜 rollout 上的独立 KL/任务指标、CQ direction 翻转率、Teacher top64 覆盖质量、重复/退化生成；不能只看 replay batch loss 持续下降。

预期收益来自减少 rollout generation 与重复 Teacher 计算；理论上每步平均成本约 `Student_update + (generation + Teacher_eval)/K`，另加缓存/搬运成本。**没有上述论文直接证明 adaptive-top64 CQ、FP8 Qwen35B→Qwen2B 的最优 K 或收益倍率。**

## 3. 核验边界与材料

- 只研究上述 4 篇；下载 PDF 后用 `pdftotext -layout` 核查标题、作者、Algorithm、Eq.、实验表，并交叉核查官方 repo 的 buffer / loss。
- 抓取材料：`/tmp/cq-reuse-primary-202603/`。关键文件 `gkd.pdf/.txt`、`distillm.pdf/.txt`、`distillm2.pdf/.txt`、`minillm.pdf/.txt`、`buffer.py`、`losses.py`、`distillm_trainer.py`。
- 文件名 `minillm.html` 是初始抓取时的命名失误，内容实际上为 GKD（2306.13649）；正文以正确 ID 和 PDF 为准，不依据文件名辨识论文。
- 未运行训练、未使用 GPU、未修改项目代码、未部署任何改变。没有认证/配额阻塞；仓库默认分支核实为 master，初试 main 的 404 不代表仓库不可访问。未切换凭证或服务提供商。

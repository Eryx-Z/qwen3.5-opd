# 公开实现核查：sampled CQ 能否让 Teacher 也抽 token？

## 结论

**能，而且应区分两件事：在同一 Student prefix 上抽一个 Teacher token，和让 Teacher 自回归生成另一条轨迹。前者才直接补齐该 context 的 forward-KL 样本。** 本次核查的 TRL GKD、NeMo RL 标准 OPD、verl 蒸馏实现，没有找到“同一 Student context 下双向 token 抽样 + 熵自适应混合系数”的现成实现。它们分别提供 full-vocab GJSD、Teacher-top-k 双向 KL、Student-action sampled KL，可作为部件参考，不能描述成已经实现 sampled CQ。

以下只依据锁定 SHA 的源代码；不引用论文/博客代替实现证据，不涉及远程训练仓库修改或 GPU 实验。

## 对比

约定 Teacher 分布为 p，Student 为 q；forward=KL(p||q)，reverse=KL(q||p)。

|实现|实际分布/目标|系数是否自适应|Teacher 是否抽 token|梯度与成本|
|---|---|---|---|---|
|TRL `GKDTrainer`|完整词表；β=0 forward，β=1 reverse；中间值是 generalized JSD，不是 forward/reverse KL 的线性加权|`self.beta=args.beta`，固定；λ只是选择训练轨迹来源的固定概率|`seq_kd` 分支确实生成 Teacher **整条序列**；不是在每个 Student prefix 额外抽 Teacher action|Teacher `no_grad`；Student 对完整分布求导；两侧完整 logits、log-softmax，未解决大词表成本 [T1]|
|NeMo RL 标准 `DistillationLossFn`|Teacher top-k 集合；默认双方在此集合重新归一化，支持 forward/reverse/mixed|`mixed_kl_weight` 固定，默认0.5；不是按熵/置信度动态变化|Student 生成，Teacher 对这些序列执行 `get_topk_logits`；返回 logits/IDs 不是 Teacher 抽样序列|Teacher inference `no_grad`；Student top-k 概率显式反传；默认归一化仅 K 项，但模型仍先输出 logits，不能称绕过 full LM head [N1–N4]|
|verl 蒸馏|`k1/k2/k3` 使用已有 action 的两侧 logprob；另有 Teacher-top-k forward KL|无上述 entropy-adaptive 双向系数|sampled 分支消费已 rollout token 的 Teacher logprob；所审 loss 未额外抽 Teacher token|PG 分支将负 KL detach 为 advantage；top-k 保留全词表归一化，可分块减小临时 buffer；不是免除全 head [V1–V3]|
|OpenRLHF HEAD|本次 tree 未发现 KD/distillation trainer，`models/loss.py` 未发现 KD/distillation 实现|`AdaptiveKLController` 调的是 PPO KL penalty coefficient，不是 forward/reverse 蒸馏混合|不能据此声称实现了双向 Teacher sampling|未作为正面案例；范围限定当前 tree 和所读源文件 [O1]|

## 最值得借鉴的源代码细节

### 1. TRL：不要把 generalized JSD 当作 adaptive KL

`generalized_jsd_loss`：中间 β 时 m=βp+(1−β)q，返回 βKL(p||m)+(1−β)KL(q||m)。β 两端单独切换 KL；中间公式本身在 β→0/1 时趋零，不能把它理解为连续插值 αKL(p||q)+(1−α)KL(q||p)。代码未额外乘温度平方。[T1:270–297]

`compute_loss` 给 Teacher 和 Student **相同 input_ids**，Teacher 在 `torch.no_grad()` 下给出完整条件分布；这只是 Teacher 打分。`training_step` 首先以 λ 概率让 Student 生成；否则 `seq_kd` 才让 Teacher 从 prompt 生成整条序列。后者改变 context 分布，不等于同一 Student context 的双向 action 采样。[T1:324–357,406–446]

完整概率张量对 q 可微，因此固定 context 下的 KL/JSD 梯度没有 sampled-action 漏掉 score-function 的问题；但生成的离散 context 未反传，不能说它计算了 Student 轨迹分布变化的完整导数。

### 2. NeMo：确有双向 KL，但不是双向抽样，也不是自适应

`DistillationLossFn.__call__` 显式计算：

- forward：Σ pᵢ(log pᵢ−log qᵢ)
- reverse：Σ qᵢ(log qᵢ−log pᵢ)
- mixed：w·forward+(1−w)·reverse。[N1:1776–1850]

**关键限制在上游归一化**：`get_distillation_topk_logprobs_from_logits` 默认 `zero_outside_topk=False`，先 gather Teacher top-k IDs 对应的 Student logits，再在 K 维 log-softmax；Teacher 也在 K 维 log-softmax。因此默认算的是条件于截断集合的两个分布，不是原完整 p、q 的 unbiased KL estimator。[N2:1931–2035]

`zero_outside_topk=True` 才让 Student 使用 full-vocab 归一化；Teacher 仍是 top-k 归一化。reverse/mixed 在集合外用 `log_infinitesimal=-100` 加入近似尾部惩罚（代码 `H_all` 是 Σq log q，符号是负熵）。这是修改后的 Teacher 目标，不是恢复原 Teacher tail。[N1,N2]

`distillation_train` 的 rollout 使用 `student_generation`，随后 Teacher `get_topk_logits(train_data,k=...)`，只保存 `teacher_topk_logits/indices`。[N3:825–947] Teacher worker 将 inference 包在 `no_grad`，返回 CPU top-k 张量。[N4:744–842] 此路线可减少通信/保留张量，但两侧 model forward 的输出 logits 仍是 head 计算；默认 K 维归一化与“全词表 softmax 是必要成本”必须分开说。

### 3. verl：sampled KL 的数值正确，不自动代表反传正确

`compute_distillation_loss_reverse_kl_estimator` 只接受 Student action 的 `log_probs` 和 Teacher 对该 action 的 `teacher_logprobs`，调用 `kl_penalty`。令 d=log q(a)−log p(a)，未截断的理想形式为 k1=d，k2=d²/2，k3=exp(−d)−1+d；实现 k3 有数值 clamp。[V1,V3]

若 a∼q，则 E[k1]=KL(q||p)，但**直接对已抽出的 d 反传只得到 ∇log q(a)**，其期望为零，漏掉采样分布导数。当前 `distillation_loss` 的 PG 路径使用 `advantages=-distillation_losses.detach()`；配合 policy-ratio 梯度，k1 在匹配 on-policy、固定 context、无 clipping 等理想条件下给出所需 score-function 更新。过期 rollout、temperature/top-p、importance ratio、PPO clipping 会影响此解释，不能无条件宣称无偏。[V1:273–308]

`core_algos.kl_penalty` 自己也提醒 k1/k3 的 value estimator 不等于 gradient estimator，并提供 `+` straight-through 机制；但本次蒸馏 registry 仅注册无 `+` 的名字，不能直接宣称配置 `k3+` 就可用。[V1,V3]

`forward_kl_topk` 是 Σ_{i∈K}pᵢ(log pᵢ−log qᵢ)，**不在 K 内重归一化**。外层会 clamp_min(0)，所以进一步偏离原 full KL；这与 NeMo 默认版本不同。FSDP `_chunked_topk_log_probs` 计算 gather(logits)−logsumexp(full logits)，省 buffer 而非 full-vocab 正规化计算。[V1:380–384,V2]

## 对 sampled CQ 的最小下一步（建议，不是仓库现成功能）

保持 Student rollout 提供 context h，不启动 Teacher 整条自回归 rollout。在选定 h 上让 Teacher 从其完整条件分布 p(·|h) 抽 b，同时保留 Student action a。这两种样本可以各一个：

- forward 项：`L_F = −log qθ(b|h)`，b∼p。Teacher 概率/采样停止梯度。这是 forward KL 对 θ 的无偏梯度估计，省略的 Teacher logprob 是 θ 常数；**这个 CE 数值不是 KL 数值**。
- reverse 项：a∼q₀，用 `L_R = stopgrad(log q₀(a|h)−log p(a|h)) · log qθ(a|h)` 作为 on-policy θ=θ₀ 处梯度 surrogate；或复用现有 k1 PG。其标量值不能作为真实 reverse KL。多轮更新需明确 importance sampling/clip 约定。
- `L = α(h)L_F + (1−α(h))L_R`。若 CQ 依据当前模型算 α/选位置，先把 α 和选择 detach，明确这是冻结权重的局部目标；若不 detach，则还多出 ∇α 项，不再是同一 estimator。

**自适应部分仍需单独设计和验证。** Teacher 在原本打分 Student 序列时已拥有各位置的条件 logits，因此在同一行采一个 Teacher token 不必重走一条 Teacher trajectory；但 serving 接口需真能输出这种 token，并让 Student 计算该 token 的 logprob。仅有 `prompt_logprobs`（既定 Student token 的概率）不等于 Teacher 能通过同一 API 返回自抽 token。精确 categorical 抽样和精确 action logprob 一般仍依赖完整 head/归一化；采样节省的是保留/传输完整分布及损失张量，不承诺消除 248320 词表的 head FLOPs。

不能用一个 −log p(b) 的高方差 entropy 样本经非线性 α=f(H) 后声称复现原 CQ：E[f(Ĥ)]通常≠f(H)。下一步应先由父任务决定 α 的廉价可观测量与偏差容忍度，再做小词表 CPU 枚举梯度对照；本报告未实现或运行该实验。

## 固定版本一手引用

- [T1] Hugging Face TRL **e765a16e4293ae7fb083846398782db0b0429879**：[GKDTrainer 源码](https://github.com/huggingface/trl/blob/e765a16e4293ae7fb083846398782db0b0429879/trl/experimental/gkd/gkd_trainer.py#L238-L446)。
- [N1] NeMo RL **7409c28e0071194af12fb8e36ca2dd7a6eb725c0**：[DistillationLossFn](https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/nemo_rl/algorithms/loss/loss_functions.py#L1776-L1875)。
- [N2] [get_distillation_topk_logprobs_from_logits](https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/nemo_rl/distributed/model_utils.py#L1864-L2035)。
- [N3] [训练循环：Student generation → Teacher top-k](https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/nemo_rl/algorithms/distillation.py#L825-L967)。
- [N4] [DTensor worker get_topk_logits](https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py#L744-L842)。
- [V1] verl **8718ca30a3f002f93b7c4fd99b9b2506718681bc**：[distillation_loss / compute_distillation_loss_reverse_kl_estimator](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/distillation/losses.py#L233-L429)。
- [V2] [FSDP compute_forward_kl_topk / _chunked_topk_log_probs](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/distillation/fsdp/losses.py#L26-L150)。
- [V3] [kl_penalty / kl_penalty_forward](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/ppo/core_algos.py#L2188-L2270)。
- [O1] OpenRLHF **dc2a7ad326f619fc5a47737ea47d1920ce2c0b53**：[trainer tree](https://github.com/OpenRLHF/OpenRLHF/tree/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer)、[models/loss.py](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/loss.py)、[AdaptiveKLController](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/kl_controller.py)。这不是对所有历史版本/外部 fork 的不存在证明。

## 可复核产物与限制

原始 GitHub commit/tree API JSON 与上述 Python 文件保存在 `/tmp/cq-public-source-child/`；常规源码按 `组织/仓库/原路径` 保存，两个额外文件为 `nemo_model_utils.py`、`nemo_dtensor_worker.py`。使用匿名 GitHub API 固定 HEAD SHA 后按 SHA 抓 raw 内容；无登录、凭据切换或大仓库 clone。曾试错 NeMo 旧路径 `nemo_rl/algorithms/loss_functions.py` 得到404，随后根据同一 tree 找到实际路径 `algorithms/loss/loss_functions.py`。网络最终无阻塞。

验证仅限逐函数静态核查与保存源文件；未跑训练/性能测试，未检查远程 `eryx` 的实际 checkout，未断言这些最新版本适配当地依赖。FP8 精度与量化对采样分布的影响未在本次范围内验证。

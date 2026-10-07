# CQ-OPD V1.1：动态 divergence 整合版 · Codex 实现规格

版本：2026-10-06。替代上一版固定 forward-KL 主方案。目标平台：单台 DGX Spark。
状态：研究方案＋算法参考实现，不是已经完成的 Qwen 训练工程。
随附 cq_opd_reference.py 在 CPU/PyTorch 2.10.0 上通过了 23 个微型测试（9 项原始回归＋14 项动态损失测试）；尚未在 DGX Spark 或两个 Qwen 模型上实测。

## 0. 实现边界

只实现：**Teacher 熵驱动的动态混合 KL + 跨题任务方向 + 片段效用 Top-r 选择 + LoRA 更新**。

动态 divergence 从第一轮正式更新即启用，不是未来 TODO。
分工固定：**Teacher entropy 决定怎么教；CQ 只决定选哪些片段。**
一个片段只产生一个 U，只排序一次；不分别挑两种 KL，不让 CQ 修改混合比例。

Teacher = Qwen/Qwen3.5-35B-A3B，冻结。
Student = Qwen/Qwen3.5-2B，仅语言主干 LoRA 可训练。
不加入 Router、CQ 调整教法、Teachability 分数融合、额外 GRPO 主训练 loss。
Teachability、KL 排序、随机选择只作为基线 selector。

“以任务奖励构造梯度，再指导 OPD 监督”已经有相近研究，例如 Dr. OPD [1]。
本项目需验证具体的跨题、片段级设计及单机成本，而不是预先宣称首次提出或保证提升。

## 1. 一张流程图

```text
Student 生成训练回答 → Teacher 同一前缀分布 q_t
                                   │
                      H_t → alpha_t（本轮固定）
                                   │
             ell_t = alpha_t*F_t + (1-alpha_t)*R_t
                                   │
独立小测题 → 对错奖励 → d=-grad(L_probe)/norm
                                   │
             正反扰动 Student，但 q_t/alpha_t 不变
                                   │
            U_b = (ell_bad,b - ell_good,b)/(2*delta)
                                   │
              恢复参数，每条回答选 U 最大的20%片段
                                   │
          用同一 q_t、alpha_t 和混合 loss 正式更新 LoRA
```

其中 F=KL(Teacher||Student)，R=KL(Student||Teacher)。
“good / bad”只是相对于小测一阶下降方向的命名，不保证实际正确率升降。
原来的小测方向、LoRA 扰动方式、16-token 分片、固定预算都不变。

## 2. 数据和模型契约

### 2.1 数据集

使用四个不重叠的划分：

| 划分 | 用途 | 是否属于训练资源 |
|---|---|---|
| train | 生成被蒸馏的回答；正确答案不进入 Teacher/Student 的输入 | 是 |
| probe | 计算任务方向；答案仅供验证器判分 | 是 |
| dev | 数值校准、超参数和方法选择 | 否，但已用于模型选择 |
| test | 最终成功率 | 否；不得调参数或计算方向 |

题目规范化后按 ID 和内容哈希去重；不同解答但题干相同仍算重叠。
数据记录至少包含 `id, prompt, answer, verifier_type`。
第一版限能自动验证的短数学题；不要把字符串与整段参考解答直接匹配。
验证器支持明确的最终答案抽取、整数/分数/小数规范化；不对模型文本直接执行 Python eval。
解析失败或没有最终答案记 0；验证器崩溃等基础设施故障单独报错，不伪装成答错。

### 2.2 参数范围

- Teacher 所有参数 `requires_grad=False`，不生成自己的独立答案。
- Student 仅语言主干 LoRA 可训练；冻结 embedding、lm_head、视觉模块。
- 枚举真实模块后建立 LoRA 名称清单，禁止猜测层名或盲目给视觉层加 adapter。
- LoRA 权重和累计梯度保留 FP32；基座先 BF16。
- 输出头冻结是第 9 节低显存回传成立的前提；若输出头/其 LoRA 可训练，必须实现额外梯度路径，不能沿用当前优化。
- Qwen3.5-2B 配置有 tied embedding/head，冻结时验证共享参数也被冻结 [5]。
- LoRA dropout=0，其他 dropout 也关闭；不调用 merge_and_unload。
- 默认 LoRA 的 B 可初始化为零，因此初始化时 A 梯度为零不一定是 bug [8]。

### 2.3 分布对齐

启动必须验证：

1. Teacher / Student 的 token string -> ID 映射相同。
2. 特殊 token 映射、输出词表维度一致；只比较 vocab_size 不够。
3. 用同一套聊天模板生成一次 prompt IDs，并将实际 IDs 原样传给双方。
4. 师生都在 Student 真实前缀上评分，不重新套 Teacher 的聊天模板。
5. 记录模型 revision、tokenizer revision、模板和 thinking 配置的哈希。
6. 先用一个短输入比较 `project(final_hidden)` 与标准 forward logits，确认包括了最终 norm、实际输出头及任何输出变换。

模型适配器根据所固定的 Transformers 版本实现，禁止依赖未验证的 `.model.model...` 属性路径。
Qwen 文档区分 dense、MoE 以及 text-only 类；GB10 的加速路径也要实际测试 [4]。

## 3. Token 对齐、mask、分片

假设 unpadded 输入为：

```text
[prompt tokens ...][response tokens ...][真实采样的结束 token]
```

训练路径统一采用右 padding；本版不做跨样本 packing。
生成可以逐条运行以避免 padding 复杂性；若批量生成使用左 padding，返回后先去掉 padding，再构造统一训练记录。

张量：

```python
input_ids:       LongTensor[B, L]
attention_mask:  BoolTensor[B, L]
response_mask:   BoolTensor[B, L]   # 只标实际生成的 token，含真实结束 token
logits:          Tensor[B, L, V]

pred_logits = logits[:, :-1, :]
next_ids = input_ids[:, 1:]
valid = response_mask[:, 1:] & attention_mask[:, 1:]
# 以下片段位置、token loss、选择mask均位于 [B, L-1] 的预测位置空间
```

关键例子：

```text
输入位置：  0   1   2   3   4
token：     P   P   P   R0  R1
预测位置：          2→R0 3→R1
```

不能从 R0 所在位置的 logits 开始计分，否则会漏掉第一个回答 token。
prompt、padding、真实结束 token 之后的位置不得贡献 loss。
若 pad_id == eos_id，仍由 mask 区分真实结束和 padding，不能按 token ID 一律删除。
达到 max_new_tokens 时不补一个假的 EOS。
保留生成的思考与回答 token；不能只在某一方法中删除思考段。

按每个回答的有效生成 token 顺序每 B_len=16 个组成一片段。
最后不足 16 个的片段保留；片段不跨样本。片段 loss / utility 用片段内均值，不用总和。
短尾导致真实监督 token 比例偏离 20%，必须记录实际比例。

## 4. 动态蒸馏目标：Teacher 熵驱动的混合 KL

### 4.1 统一定义（不可交换 alpha 的含义）

在预测位置 t：

- q_t：冻结 Teacher 在固定 Student 前缀上的全词表分布。
- p_t：当前 Student 在完全相同前缀上的全词表分布。
- 蒸馏温度固定为 1，Teacher entropy 也使用这个温度。
- 所有 log 为自然对数，entropy 单位为 nats，不除以 log(vocab_size)。

\[
H_t=-\sum_v q_t(v)\log q_t(v),\qquad
\alpha_t=\sigma(k(H_t-\tau)).
\]

\[
F_t=\sum_v q_t(v)(\log q_t(v)-\log p_t(v)),\quad
R_t=\sum_v p_t(v)(\log p_t(v)-\log q_t(v)).
\]

\[
\boxed{\ell_t=\alpha_tF_t+(1-\alpha_t)R_t.}
\]

alpha 始终表示 **forward-KL 权重**：
高熵 → alpha 较大 → forward-KL 较多；
低熵 → alpha 较小 → reverse-KL 较多。

这是一个设计假设，不是“Teacher 越确定就越正确”的保证。
相关 EOPD 在高熵位置增加 forward-KL；本方案的连续加权混合不等于原论文具体公式 [13]。
此处“dynamic divergence”专指这两个 KL 的凸组合，不称为某个标准 alpha-divergence。

alpha 是 **token 级**；先计算每个 token 的混合 loss，再对16-token片段取平均。
不能先平均片段entropy再生成alpha，也不能先分别平均F/R再乘片段平均alpha；
一般情况下这些都改变了目标。

### 4.2 tau / k 怎么定：启动时校准一次，训练中不随 batch 改

不要凭空填一个entropy阈值，也不要每个batch重新做min-max/分位数归一化。

推荐可复现起点：
1. 初始 Student 在32道 **train** 题上生成回答，不更新参数。
2. Teacher 在这些固定前缀上计算entropy，只收集真实回答预测位置。
3. 固定随机种子，最多均匀抽取8192个有效位置；prompt/pad不参与。
4. `tau = Q50(H)`；`scale = max(Q75(H)-Q25(H), 0.1)`；`k = 1/scale`。
5. 保存 `entropy_calibration.json`，包含tau、scale、k、样本ID/位置、随机种子、
   Teacher/Student初始revision、tokenizer/template、温度、Teacher精度/量化配置的哈希。
6. 正式训练冻结tau/k；每个新前缀仍会有新的H和alpha，因此仍是动态token级loss。
7. 恢复checkpoint时读取同一校准，不重新拟合；Teacher精度/revision/温度等不符时显式报错。

32、8192和0.1都是启动规则，不是已验证最优值。
校准的生成和打分成本计入完整时间；不使用test、probe奖励或当前CQ排名来拟合。
参考 `calibrate_entropy` 只负责从有效entropy计算数值；采样、哈希、文件管理由trainer实现。

纯forward回归：`mixing.mode=constant, constant_alpha=1.0`。
纯reverse回归：`constant_alpha=0.0`。
固定混合消融：`constant_alpha=0.5`。
默认必须是 `mixing.mode=teacher_entropy`，不能静默走forward-only旧路径。
这里的alpha与LoRA的缩放参数`lora.alpha=16`是不同参数。

### 4.3 损失实现与梯度契约

```python
# 数学参考；生产路径从输出头之前按预测位置分块
# 单元测试保留FP64；真实低精度logits转FP32后再做log_softmax
with torch.no_grad():
    log_q = F.log_softmax(teacher_logits.float(), dim=-1)
    q = log_q.exp()
    entropy = -(q * log_q).sum(dim=-1)
    alpha = torch.sigmoid((entropy - tau) / scale).detach()

log_p = F.log_softmax(student_logits.float(), dim=-1)
p = log_p.exp()                         # 必须保留梯度！
forward_kl = (q * (log_q - log_p)).sum(dim=-1)
reverse_kl = (p * (log_p - log_q)).sum(dim=-1)
token_loss = alpha * forward_kl + (1 - alpha) * reverse_kl
```

要求：
- q、log_q、H、alpha 都固定，不进入 Student 梯度；alpha不训练、不由CQ修正。
- **reverse-KL中的p不能detach**，它本身随Student参数变化。
- log_q直接由Teacher logits的log_softmax取得，不先低精度softmax再取log或clamp。
- 不用Teacher argmax代替q；不用单个采样token的logprob代替完整词表loss。
- 不截断词表Top-K；不对NaN/Inf使用nan_to_num伪装成正常训练。
- 不将微小数值负KL逐token截成0，因为截断会改变差分/梯度；先检查误差和精度。
- 不传labels后误用模型内置hard-label CE `.loss`。
- 使用共享函数 `mixed_kl` 计算评分目标、训练目标和数值审计目标。

旧soft-CE捷径的适用范围：
\[
E_t=-\sum_vq_t(v)\log p_t(v),\qquad
\ell_t=\alpha_tE_t+(1-\alpha_t)R_t-\alpha_tH_t.
\]
只在q和alpha均固定时，最后一项是Student无关常数。
可以省略这一项以求梯度，但 **不能省略整个reverse-KL项，也不能只用soft CE评分**。
本版默认实现完整mixed KL，日志分别记录mixed/F/R，避免混淆。
旧 `soft_ce`、`chunked_hidden_gradient` 保留为forward-only回归测试，不是主训练入口。

KL-based baseline仍以true forward-KL `F_t` 排序，不按soft CE排序。
baseline选择指标与实际训练loss分开配置：比较选择器时，所有组使用相同动态loss和校准规则。

## 5. 小测方向：只提取梯度，不执行训练

### 5.1 采样

每轮从 probe 抽 M=4 题，每题 K=4 个独立回答。
所有回答来自本轮同一参数 psi_0。
第一版统一：

```yaml
do_sample: true
temperature: 1.0
top_p: 1.0
top_k: 0
repetition_penalty: 1.0
```

不使用 top-p 截断、重复惩罚、强制 EOS 或其他改变采样分布的额外 logits processor。
检查模型自带 generation_config，不要以为未显式传入的参数就不存在。
如果必须改变采样策略，probe 的 log probability 也必须对应同一策略；否则只能标为有偏代理。

最大长度是任务定义的一部分：在预算内给出可验证答案为成功，否则失败。
批量内不能做有意的相关采样；每个回答使用独立随机流。
生成结束后丢弃 generation cache；后续评分/训练重新计算完整前缀。

### 5.2 奖励和优势

\[
R_{ik}\in\{0,1\},\qquad
A_{ik}=R_{ik}-\frac{\sum_{j\ne k}R_{ij}}{K-1}.
\]

例：`[1,0,1,0] -> [2/3,-2/3,2/3,-2/3]`。
全对或全错的一题优势都为零，该题梯度贡献为零，不重新标成正/负。

### 5.3 任务代理损失

\[
L_{\text{probe}}=
-\frac{1}{MK}\sum_{i,k}A_{ik}
\sum_{t\in\text{generated}(i,k)}
\log p_{\psi_0}(y_{ikt}\mid x_i,y_{ik,<t}).
\]

- 外层除以全部 MK 个回答数，包括零优势回答。
- 内层是生成 token log probability 的**求和**，不除回答长度。
- `A`、奖励、token IDs 都 stop-gradient。
- 梯度条件是固定这些已采样轨迹；不对采样操作求导。
- 这只是策略梯度估计；小 batch 有噪声，梯度不是“绝对正确方向”。
- 不同 batch 的 probe loss 可正可负，其原始数值不能当作成功率，也不能跨 batch 直接比较大小。

计算：

\[
g=\nabla_\psi L_{\text{probe}},\qquad
d=-g/(\|g\|_2+\epsilon).
\]

`d` 是小测 loss 的下降方向，后面统一使用这个符号。
把整组 LoRA 参数视为一个向量归一化，但代码用张量列表计算范数，不创建巨大的 flattened copy。

使用 `torch.autograd.grad` 返回梯度，不累加到 Student 的 `.grad` [7]。
`create_graph=False`；逐个回答前向并累加返回的梯度，避免同时保留 16 个回答的计算图。
零优势回答可跳过前向，但分母仍是 MK。
按实际 LoRA 参数名称固定顺序；未使用参数返回 None 时转零，并报告名称用于排查。

如果所有优势为零或梯度严格无信号，本轮回退 random selector，记录 `fallback=no_probe_signal`。
若梯度 NaN/Inf，视为实现/数值故障，不能悄悄当作零信号。

严禁对 probe 调用 optimizer.step；probe 只通过混合loss的效用影响片段选择，不改变alpha，也不直接更新模型。

## 6. 片段效用：两次前向而非逐片段 backward

对固定片段 b：

\[
\ell_b=\frac{1}{|b|}\sum_{t\in b}\ell_t.
\]

理论目标：

\[
U_b^\star=
\left\langle\nabla_\psi \ell_b,\frac{g}{\|g\|_2+\epsilon}\right\rangle.
\]

若只用片段 b 做小 SGD 更新：
`L_probe(psi - eta*grad ell_b) ≈ L_probe(psi) - eta*<g,grad ell_b>`。
这是局部、SGD 意义的预测；最终 AdamW 更新另有动量与预条件化，不是精确同一个方向 [9]。

实际计算：

\[
\psi_\text{good}=\psi_0+\delta d,\qquad
\psi_\text{bad}=\psi_0-\delta d,
\]

\[
\boxed{U_b=\frac{\ell_b(\psi_\text{bad};q,\alpha)-\ell_b(\psi_\text{good};q,\alpha)}{2\delta}}.
\]

正分：朝小测下降方向移动时，同一alpha下的混合蒸馏loss也下降。
负分：局部方向冲突。
零附近：影响很小，或评分精度不足，需数值诊断。

### 6.1 参数生命周期

```python
# 伪代码；只快照LoRA，不复制整个模型
saved = clone_lora_parameters()
try:
    copy_lora_from(saved, offset=+delta * d)
    h_good = forward_last_hidden_no_grad(fixed_ids, use_cache=False)

    copy_lora_from(saved, offset=-delta * d)
    h_bad = forward_last_hidden_no_grad(fixed_ids, use_cache=False)
finally:
    copy_lora_from(saved)  # 精确恢复；即使前向异常也执行
```

- 每次从 saved 复制；不连续 `add_(delta); add_(-2delta); add_(delta)` 累积漂移。
- 评分前不能有尚未 backward 的主训练计算图。
- 评分后恢复参数，再重新前向建立主训练图。
- 之前已经累计的 `.grad` 可以保留，但评分不得改变它。
- 优化器 momentum、variance、step 不参与临时扰动；不创建 optimizer 更新。
- Teacher 参数、distribution、entropy、alpha、Student 生成 token、mask 全程不变。
- 每次扰动前向清空 KV / recurrent state；不能复用其他参数状态下的 Student cache。
- 不重采样回答，否则差分混入采样噪声。

### 6.2 两次Student前向，仍然只得到一个效用

Teacher主干前向一次，缓存最终hidden `h_T`。
Student的两个扰动只保留 `h_good, h_bad`，不缓存 `[B,L,V]` 全词表logits。

按64个预测位置一块投影：
- Teacher输出 `log_q`，由此计算当前固定前缀的H和alpha。
- good/bad logits各自做一次log_softmax。
- 同一份log_p同时用于F和R，不为两种KL分别跑Student主干。
- 缓存每个有效位置的H、alpha（标量），不长期保存全词表分布。

令 `lq=log_q`，`lg=log_p_good`，`lb=log_p_bad`：
\[
\Delta F_t=\sum_v q_t(v)(lg_t(v)-lb_t(v)),
\]
\[
\Delta R_t=\sum_v
\left[p_\text{bad,t}(v)(lb_t(v)-lq_t(v))-
p_\text{good,t}(v)(lg_t(v)-lq_t(v))\right],
\]
\[
\boxed{u_t=\frac{\alpha_t\Delta F_t+(1-\alpha_t)\Delta R_t}{2\delta}.}
\]

参考函数 `mixed_utility_from_logits` 实现这个等价式。
也可以先用共享 `mixed_kl` 算两侧loss再相减，二者必须通过等价测试。
最后片段内平均u_t得到U_b；不能只保留原先的Delta-F公式。

训练使用评分时缓存的同一alpha，Teacher log_q可从同一h_T按块重建。
若非CQ基线或probe无信号而跳过扰动，仍须准备H/alpha，正式训练仍用动态loss。
每块logits完成后释放。增加的是reverse-KL逐词表运算和少量标量缓存，
不是增加第三、第四次Student主干前向；真实耗时和峰值必须实测。

这利用目标更新与训练样本影响的一阶关系，与ToV的估计思路相关 [2]；
中心差分是本规格的实现，不声称是ToV原算法的原样复现。

## 7. delta：必须校准，不能随便填 1e-6

BF16 差分可能被舍入误差吞掉；LoRA master weights 是 FP32 也不保证整个前向没有量化误差 [10]。

先设：

\[
\delta=\rho\|\psi_0\|_2.
\]

起始校准候选 rho：
`[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]`。
范数包括全部 LoRA 张量，不按单个 LoRA B 的范数缩放（其初始化可能为零）。
若整组 LoRA 范数也是零，视为配置问题；先检查是否把 A、B 都错误初始化为零。

校准顺序：

1. 在 CPU FP64 微型模型上，比较差分 U 与逐片段显式梯度内积。
2. 在真实 Student 的少量 dev 轨迹上测重复前向噪声。
3. 比较 delta/2、delta、2delta 的片段排序。
4. 在噪声以外的片段上检查符号一致性；检查 Top20% overlap。
5. 选择稳定区间中较小的 delta，保存到 calibration.json。
6. 训练初期及明显分布变化后复查；重新校准开销计入训练时间。

可用 `Spearman >=0.8`、`Top20% overlap >=0.7` 作为启动时的工程检查线，
但这些阈值不是理论保证，也不是预先实验结果。近零分数大量并列时应报告不可判定，
不能单靠这两个数宣布评分有效。

如果 BF16 无稳定区间：
动态混合loss必须重新完成本节校准，不能直接沿用forward-only的delta报告。
若改动alpha映射/Teacher量化/评分dtype，同样重新校准。
先尝试提高 Student 评分前向的精度并校验与实际更新的一致性；
或在确认算子支持后采用 JVP。
`torch.func.jvp` 可能遇到不支持 forward-mode AD 的算子 [11]。
不得把明显不稳定的差分当作正常 CQ；可以继续跑基线，但必须标记 CQ 尚未通过验收。

## 8. 选择与最终损失：固定预算，不增加决策模块

每条回答有 n 个片段，选择：

\[
k=\max(1,\lceil rn\rceil),\quad r=0.2.
\]

按原始**带符号** U 降序选择 k 个。
不取绝对值，不 softmax，不乘 Teachability，不用正分过滤。
并列按原片段顺序稳定排序。
即使所有 U<0，固定预算版仍选相对最高的 k 个；同时记录负分占比。
这保证和随机、KL 基线比较的预算更明确，不代表负分片段一定有益。
“仅选择正分”是另一个可变预算消融，不要混入本版。

训练 loss：

\[
L_{\text{train}}
=\frac{\sum_t M_t \ell_t}{N_{\text{selected}}},
\qquad N_{\text{selected}}=\sum_t M_t.
\]

M 是展开到 token 的 bool mask，与 response_mask 取交集。
未选位置不直接产生 loss，但仍保留在完整上下文，梯度也可以经过其前缀计算。
不得删掉未选 token，不得只把选中片段当独立短文本送入模型。

r=1 必须与 **full dynamic OPD** 产生相同 LoRA 梯度（同轨迹、同alpha、同精度）。
alpha恒1时还必须回归到旧forward-KL梯度；不要把动态full误比较成纯forward full。
r=1 或 full baseline 可跳过没有意义的 probe/评分开销。

## 9. 显存关键实现：分块输出头 + 一次主干回传

### 9.1 为什么不能“分块后统一 backward”

下面的写法可能仍保留所有块的 softmax/输出头计算图：

```python
loss = sum(loss_for_chunk(chunk) for chunk in chunks)
loss.backward()
```

需要真正逐块释放输出头图，而不是只让表面上的 logits 变小。

### 9.2 正确路径（输出头冻结）

1. 正常 Student 主干前向，得到最终 `h: [B,L,D]`，保留主干图。
2. 按选中的预测位置切块；每块复制/分离 hidden，成为临时 leaf。
3. frozen head 投影，计算该块 mixed KL **总和**，使用已缓存的alpha。
4. `autograd.grad(chunk_loss, h_leaf)` 得到这一块的 hidden 梯度，立即释放块图。
5. 把各块梯度填回 `grad_h: [B,L,D]`，未选行填零。
6. 对原始 h 做一次向量-雅可比回传，更新主干 LoRA 梯度。

```python
h = student_adapter.forward_last_hidden(batch)  # 保留主干图
grad_h = zeros_like(h)
for rows in selected_prediction_chunks:
    h_leaf = h[rows].detach().requires_grad_(True)
    log_q = teacher_target_log_probs(rows).detach()
    alpha_rows = cached_alpha[rows].detach()
    logits = student_adapter.project(h_leaf)
    chunk_sum = mixed_kl(logits, log_q, alpha_rows).sum()
    dh = autograd.grad(chunk_sum, h_leaf, create_graph=False)[0]
    grad_h[rows] = dh.detach()
    # 块图在此释放
torch.autograd.backward(h, grad_h)
```

这不是截断主干梯度：最终回传仍使用原始 h 的完整计算图。
不要在第 1 步把整个 h detach 后忘记回传。
只 detach 临时输出头 leaf；Teacher log_q及alpha不需要梯度；Student p不得detach。

旧 `(p-q) @ W` 只对应forward部分，不能用于整个mixed KL。
普通frozen linear head `z=hW^T+b`、温度1、alpha固定时，
令 `r_v=log_p_v-log_q_v`、`R=sum_v p_v*r_v`，则：
`dell/dz = alpha*(p-q) + (1-alpha)*p*(r-R)`，
再做 `dell/dh = (dell/dz) @ W`。
先使用通用chunk VJP，并通过自动微分梯度等价测试，之后才考虑该加速。
有输出缩放/softcap时不能无条件套这个公式。
第一版不要手写 CUDA 或 fused KL 内核。

小测也可沿用这条路径：
小测仍用实际生成token的hard-label CE，每条回答乘 `A/(MK)`，不套入mixed KL或alpha；
只复用分块hidden VJP的工程机制。
主干使用 `autograd.grad(h, lora_params, grad_outputs=grad_h)` 返回梯度，
而不是写 `.grad`。不必真正分配 one-hot 张量，gather 目标 logprob 即可。

### 9.3 跨微批量归一化

每个微批 backward 的是 selected mixed KL 的**总和**，不先除该微批 token 数。
累计 A 个微批后：

```python
for p in trainable_lora_params:
    if p.grad is not None:
        p.grad.div_(total_selected_tokens_in_this_optimizer_step)
clip_grad_norm_(trainable_lora_params, 1.0)
optimizer.step()
```

不要再除 A，否则多除一次。
这样结果等价于把本轮所有选中 token 放一起计算平均 loss；
不是对长短不同的微批做等权平均。

## 10. 一次 optimizer step 的完整顺序

`step` 永远指一次 optimizer.step，不是一个微批。
正确性阶段每 step 刷新 probe（refresh_interval=1）。
只有通过验收后才测试每 8 step 刷新；这是 stale-direction 近似，必须记录 direction_age。

```python
# 启动：先读取或拟合一次entropy映射；恢复checkpoint时校验元数据哈希。
# 若mixing.mode=constant，则无需拟合entropy映射，直接固定指定alpha。
frozen_entropy_calibration = load_or_fit_entropy_calibration_on_initial_train()
# 随后用当前dynamic loss单独校准delta；两个校准不能相互替代。

for step in range(num_optimizer_steps):
    # 参数在这一整个轮次、包括所有microbatch，都是psi_0
    if selector == "cq" and keep_ratio < 1:
        if need_refresh(step):
            probe_rollouts = sample_probe_from_current_student()
            rewards = verify(probe_rollouts)
            direction = compute_probe_direction_without_touching_dot_grad(
                probe_rollouts, rewards
            )
        # zero-signal可正常回退；NaN/Inf/基础设施故障应显式报错
    else:
        direction = None

    optimizer.zero_grad(set_to_none=True)
    total_selected = 0
    loss_numerator_log = 0.0

    for micro in range(grad_accum_steps):
        # 此时没有上一micro的活跃计算图；前一micro的.grad可保留
        train_prompts = next_train_microbatch()
        trajectory = student_generate_current_params(train_prompts)
        batch = build_shifted_masks_and_blocks(trajectory)

        with no_grad():
            teacher_hidden = teacher_last_hidden(batch, use_cache=False)
        # 只缓存当前micro的Teacher最后一层hidden
        # 按块计算一次H/alpha。正式训练与正反评分共用同一alpha张量。
        target_meta = prepare_entropy_alpha(
            teacher_hidden, batch.valid_mask, frozen_entropy_calibration
        )

        if selector == "cq" and keep_ratio < 1 and direction is not None and direction.valid:
            # 会临时修改LoRA，必须在主训练前向之前完成
            token_utility = score_by_symmetric_offsets(
                batch, teacher_hidden, target_meta.alpha, direction, calibrated_delta
            )
            selected = top_fraction_of_blocks_per_response(token_utility)
        else:
            selected = baseline_or_random_fallback_selector(
                batch, teacher_hidden, selector, direction, keep_ratio
            )

        # 先检查参数已恢复，再建立主训练图
        assert_current_parameters_are_restored_in_debug_mode()
        h = student_last_hidden_with_grad(batch, use_cache=False)
        grad_h, mixed_sum, n_selected = chunked_mixed_kl_hidden_grad(
            h, teacher_hidden, target_meta.alpha, selected
        )
        if n_selected:
            backward(h, grad_h)  # 累计raw gradient sums
        total_selected += n_selected
        loss_numerator_log += mixed_sum
        release_microbatch_graphs_and_hidden_caches()

    if total_selected == 0:
        # 不推进optimizer或scheduler
        record_skipped_step()
        continue

    divide_accumulated_gradients_by(total_selected)
    assert_all_gradients_finite()
    clip_lora_grad_norm()
    optimizer.step()
    scheduler.step()
    # 下一轮重新生成；不对旧rollout做多轮epoch更新
    log_metrics()
```

实际实现应使用结构化 `DirectionResult(valid, tensors, source_step, reason)`，
不要让方向为 None 与网络故障混为一谈。
若选择器是 random/kl/teachability/full，不生成不必要的 probe 数据；但动态loss的H/alpha准备仍需执行。

## 11. DGX Spark 实现约束

DGX Spark 是 ARM + 128GB 统一系统内存，不是独立128GB显存 [3]。
不要把 CPU offload 当作增加了第二份物理内存。

- 先使用支持该硬件的 PyTorch 环境，锁定 torch/transformers/peft/CUDA/模型 revisions。
- Teacher BF16 frozen、Student BF16基座+FP32 LoRA，先测真实峰值再决定Teacher量化。
- Teacher量化不是透明优化：它改变 q，所有基线须使用同一量化Teacher。
- Qwen3.5 的 DeltaNet 路径、GB10 内核以及 text-only 加载要按官方文档核验；
  推理加速 kernel 不一定支持训练 backward，不强开未经测试的 remote kernel [4]。
- 先用同一 Student 实例生成/评分/训练；不额外常驻另一份 vLLM Student。
- 生成可以 use_cache=True；probe重算、Teacher打分、Student正反评分和正式训练用False。
- 所有独立问题清空 recurrent state；参数改变后更不能复用状态。
- 需要 checkpointing 时设置 non-reentrant (`use_reentrant=False`)，兼容 autograd.grad [12]。
- 缓存当前micro Teacher hidden，不缓存其全词表 logits；输出头按64个位置分块。
- 正反评分在主训练之前，无需保留其autograd图。
- 同一块log_q/log_p同时算两种KL，不为F/R复制模型或主干前向；临时张量按块释放。
- 动态loss只需新增H/alpha等标量缓存，但reverse-KL有额外词表运算；不能声称完全零开销。
- 可把H/alpha准备融合到评分的Teacher输出头分块循环；无CQ评分的基线在训练前准备。
  这样避免仅为entropy再额外投影一遍Teacher输出头，但必须保证训练复用同一alpha。
- 保留FP32 log_softmax，不先把Teacher概率转BF16再取log；否则reverse项易受极小概率影响。
- 不要每micro调用 empty_cache；它不释放仍被引用的计算图。
- 不在每个 token/chunk 上同步 `.item()`；日志按micro或step聚合，计时时在边界同步GPU。
- 不先开启 torch.compile/CUDA Graphs：先验证可变长度和临时参数更新，再单独测编译收益。
- 真正瓶颈可能是probe生成、Teacher前向、或Student打分；分别计时后再优化。
- 可在正确性通过后比较：刷新probe间隔、生成批量、输出头块大小、梯度检查点。
  任一数值目标变化（如top-K词表近似、长度归一化、量化Teacher）均作为新实验，不暗改。

两模型当前配置都列出 vocab_size=248320 [5]。
例如一个 `[1,2048,248320]` BF16 logits 张量仅元素存储就约1.02GB（十进制），
还不含logprob、Teacher分布、梯度等。因此应从输出头之前分块，不能只在softmax之后切片。

## 12. 默认配置

见 `cq_opd_dynamic.yaml`。关键原则：

- 所有数值是起始值，不是已验证最优值。
- probe刷新默认1；8是后续提速消融。
- delta不在配置里直接写死；rho由校准确定。
- 全词表动态混合KL，温度1；token级alpha；固定比例signed top-k。
- entropy映射先在train初始轨迹上校准并冻结；finite-difference delta另外校准，二者不是同一参数。
- 默认目标是动态loss；constant_alpha=1仅用于forward回归/消融。
- weight_decay=0便于减少混淆；所有比较组保持一致。
- 学习率是启动值，需要dev选择，不用test。

## 13. Codex 工程模块

```text
cq_opd/
  data.py           # 划分去重、题目读取、mask、shift
  verifier.py       # 可测试的最终答案验证；不执行任意模型文本
  model_adapter.py  # Qwen dense/MoE真实接口；hidden与head分离；dtype检查
  rollout.py        # 同一策略采样；当前LoRA同步；缓存清理
  blocks.py         # 回答内16-token片段及mask展开
  losses.py         # full F/R/mixed KL；head chunk VJP
  probe.py          # LOO优势；probe梯度；不触碰正式.grad
  utility.py        # 快照/恢复、正反扰动、delta校准
  selectors.py      # full/random/kl/teachability/cq，共用预算接口
  trainer.py        # snapshot级训练循环，按token总数归一化
  evaluate.py       # 固定解码协议，测试成功率和截断率
  metrics.py        # JSONL日志、阶段耗时、fallback、内存
  config.py
tests/
  test_alignment.py
  test_probe.py
  test_utility.py
  test_restore.py
  test_chunked_vjp.py
  test_accumulation.py
  test_end_to_end_tiny.py
```

最小接口：

```python
ModelAdapter.last_hidden(batch, with_grad, use_cache=False) -> Tensor[B,L,D]
ModelAdapter.project(hidden_rows) -> Tensor[N,V]
ModelAdapter.lora_named_parameters() -> list[tuple[str, Parameter]]

prepare_entropy_alpha(teacher_hidden, valid_mask, calibration) -> TargetMeta
# TargetMeta.entropy/alpha: [B,L-1]；无梯度，按mask有效。
compute_probe_direction(rollouts, rewards) -> DirectionResult
make_blocks(response_mask_shifted) -> list[Block]
score_token_utility(batch, teacher_hidden, alpha, direction, delta) -> Tensor[B,L-1]
select_blocks(scores, blocks, keep_ratio) -> BoolTensor[B,L-1]
chunked_mixed_kl_hidden_grad(student_hidden, teacher_hidden, alpha, selected) -> (
    grad_hidden, mixed_sum_detached, selected_count
)
```

接口中的with_grad不应仅用train/eval模式表达；eval()不等于no_grad()。
生产模型若train/eval选择不同数值kernel，应先做模式前向一致性检查。
不要依赖模型返回的完整 hidden_states tuple，优先直接取得最终hidden，避免保存所有层输出。

Teachability基线可另实现 `C_t=sum_{v in StudentTopK}q_t(v)`，
固定TopK大小并记录，`D_t`用true KL，片段内平均D*C；
它不进入CQ主评分。

## 14. 验收：先证明代码对，再检验方法是否有效

### 14.1 必须通过的正确性测试

| 测试 | 通过条件 |
|---|---|
| shift/mask | 第一回答token、真实EOS都计入；prompt/pad都不计入 |
| full词表分布 | q与p按相同ID对齐；概率归一；Teacher无梯度 |
| alpha端点 | alpha=1回归forward；alpha=0回归reverse，检查loss和梯度 |
| entropy/alpha | H高则alpha高；忽略prompt/pad；保存/恢复映射一致 |
| 梯度契约 | Teacher及alpha无梯度；reverse中的Student概率保留梯度 |
| 动态差分 | mixed loss差分、稳定公式和显式内积一致 |
| 旧CE/KL等价 | 仅作为forward-only回归；不用于整个动态目标 |
| probe | LOO数值正确；全对/全错贡献零；长度不被额外平均 |
| utility符号 | 一维构造中同向片段为正，反向为负 |
| 差分校验 | FP64微型模型与显式梯度内积吻合；误差随步长合理变化 |
| 恢复 | 正常/异常退出后LoRA逐元素恢复；optimizer和.grad不被评分修改 |
| 选择 | 固定每回答预算；负分按带符号排名；r=1与full一致 |
| chunk VJP | 动态mixed loss在多个chunk size下与完整反向梯度一致 |
| 累计 | 不等长microbatch与一次性token平均梯度一致 |
| cache隔离 | 改参数/换问题不复用Student recurrent状态 |
| 断点 | RNG、方向来源step、entropy映射及哈希、optimizer、scheduler、LoRA、config可重建 |

随附reference已有23项数学/张量测试（9项原回归＋14项动态测试）；生产Qwen仍需单独写适配与集成测试。

### 14.2 真实模型短期更新审计

在一个固定checkpoint：

1. 训练题产生候选片段；probe_A产生选择方向。
2. 从独立dev题probe_B生成一次审计轨迹和奖励，固定这些轨迹与优势。
3. 从相同模型与optimizer状态出发，分别按高U/随机/低U片段做一个小更新。
4. 比较固定probe_B上的代理loss变化；不能每次重采样B再比较原始loss。
5. SGD用于检查一阶推导；另用正式AdamW状态重复，检查代理与实际更新的相关性。
6. 多个checkpoint、多个batch重复，并匹配实际选中token数。
7. 短期audit通过不等于长期正确率提升；只是允许进入训练比较。

不要求每次高U都胜过随机；用重复实验估计相关性和波动。
若稳定无相关，不继续堆新选择模块，先修复方向、差分精度、优化器失配。
动态loss已是当前主目标；可用alpha=1做诊断对照，但不能把它静默当动态版提交。

### 14.3 最小端到端实验

共享初始checkpoint、LoRA范围、Teacher精度、训练问题序列、解码预算和超参数：

先做最小2×2实验，将选择器与loss变化分开：
| 选择器 | 固定forward（alpha=1） | 动态混合（Teacher entropy） |
|---|---|---|
| Random 20% | A | B |
| CQ 20% | C：旧版对照 | D：本版主方案 |

再补full、KL、Teachability基线；比较选择器时统一动态loss。
可增加CQ+constant_alpha=0.5，检查收益是否只是混合KL而非entropy自适应。
各动态组使用同一初始train校准及映射；alpha随各自on-policy前缀自然不同。
r=1、constant-alpha端点都要保留为数值验收模式。

各方法参数更新后rollout自然不同，不要强行用别的方法旧轨迹冒充on-policy。
诊断用同一静态轨迹，正式训练每个方法用自己的当前Student。

分别比较：
同optimizer步数、相近实际监督token数、相同总GPU时间。
额外加入等probe数据/奖励预算的训练基线，控制额外任务监督。
至少多个随机种子；报告均值和波动，不凭单次上涨断言有效。
测试只用于最终评估；周期检查使用dev。

## 15. 日志与停止条件

每step至少记录：

```text
student_version, direction_source_step, direction_age
probe_success_rate, informative_question_fraction, probe_grad_norm
delta, delta_rank_stability, cq_valid, fallback_reason
utility_mean, utility_positive_fraction, selected_utility_mean
selected_tokens, available_tokens, actual_keep_ratio
train_mixed_kl, train_forward_kl, train_reverse_kl, truncated_fraction
teacher_entropy_mean, alpha_mean, alpha_p10, alpha_p90
alpha_selected_mean, entropy_calibration_hash, tau, k
probe_generation_time, probe_gradient_time, train_rollout_time
teacher_forward_time, scoring_time, train_backward_time, optimizer_time
total_elapsed_time, system_memory_used, torch_peak_allocated
dev_success_rate（仅实际评估的step）
```

不能只记录mixed KL变小就宣称成功；最终指标是固定预算下的独立成功率。
不能把“20%监督位置”写成“20%总计算”；所有额外成本计入总时间。
NaN、参数恢复失败、cache越界、词表不一致必须fail-fast。
probe无信息可正常fallback；大量fallback要报告方法实际上接近随机选择。

## 16. 给 Codex 的执行顺序

```text
先跑随附23项CPU数学测试
        ↓
实现真实Qwen adapter：一条短样本的前向/反向、词表、shift
        ↓
实现full dynamic OPD与冻结输出头的mixed chunk VJP，检查梯度等价
        ↓
在train初始轨迹校准entropy映射并冻结（这不是delta校准）
        ↓
实现probe方向（不更新参数），验证.grad不被污染
        ↓
实现FD与参数恢复，在真实精度上校准
        ↓
实现固定预算CQ selector和token归一化训练循环
        ↓
跑20步smoke test：无数值错误、日志完整、内存峰值可控
        ↓
做独立题目单步更新审计
        ↓
有效后做200-step pilot，再决定扩大训练和刷新间隔
```

这些步数是建议的工程检查规模，不是预估运行时间或效果保证。
动态alpha已属于主实现，不能留TODO。
不要在这一阶段创建Router、CQ修正alpha、二阶meta梯度或额外GRPO训练分支。

## 参考来源

[1] Dr. OPD（2026-09-29）：任务奖励梯度与token蒸馏更新对齐。
https://arxiv.org/html/2609.38025v1

[2] Train on Validation（ToV）：通过目标方向的更新前后损失变化估计数据价值。
https://arxiv.org/html/2510.00386

[3] NVIDIA DGX Spark 硬件说明。
https://docs.nvidia.com/dgx/dgx-spark/hardware.html

[4] Transformers Qwen3.5 文档。其在线版本会更新；工程需固定版本并实测。
https://huggingface.co/docs/transformers/en/model_doc/qwen3_5
https://huggingface.co/docs/transformers/en/model_doc/qwen3_5_moe

[5] 官方模型配置。
https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json
https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json

[6] PyTorch kl_div 的输入语义与reduction。
https://docs.pytorch.org/docs/main/generated/torch.nn.functional.kl_div.html

[7] PyTorch autograd.grad 返回梯度但不累加到.grad。
https://docs.pytorch.org/docs/main/generated/torch.autograd.grad.html

[8] PEFT LoRA 初始化说明。
https://huggingface.co/docs/peft/en/developer_guides/lora

[9] AdamW 的动量、二阶状态和更新式。
https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html

[10] PyTorch 数值精度说明。
https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html

[11] PyTorch JVP。
https://docs.pytorch.org/docs/main/generated/torch.func.jvp.html

[12] PyTorch non-reentrant gradient checkpoint。
https://docs.pytorch.org/docs/main/checkpoint.html

[13] EOPD：Entropy-Aware On-Policy Distillation of Language Models，v3，2026-06-12。
https://arxiv.org/abs/2603.07079
已核对其高entropy位置增加forward-KL的动机；本方案不是该论文的逐行复现。

[14] PyTorch log_softmax：直接从logits计算，避免先softmax再log的数值问题。
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
在线文档可能重定向至更新版本；随附测试实际环境是CPU PyTorch 2.10.0+cpu。

# Topic 5 事件内构造性生成充分性审计 v0.1

日期：2026-07-30
状态：**工程验收通过；局部生成 Gate B 失败；全局双向 Gate C 与 SNN 对接按合同锁定**

## 1. 这轮真正问了什么

这轮没有继续搜索“更强的 RNN”，也没有把 IEI、发作倒计时或
early-ictal target 混进现有模型。唯一问题是：

> 已知 held-out 间期群体事件的第一个 rank set 后，患者特异的静态
> contact scaffold、冻结的短程 ordered transition 和独立的
> progress/STOP 规则，能否完全自由运行并生成真实的完整双向事件分布？

这里严格区分：

\[
\text{where：静态招募偏好}
\neq
\text{how：事件内传播}
\neq
\text{when：真实时间中的发作进入}.
\]

仓库的 Topic 2 已经系统分析过 IEI、事件率和发作附近变化；但它们没有与
一个跨事件持续的 RNN state 建立联系，而且当前论文合同将其留在 Paper 2。
因此审阅意见对“当前 RNN 没有建模 when”这一判断是正确的，但不能写成
“项目从未分析过 IEI”。

## 2. 冻结实验合同

- 队列：34 人，Epilepsiae 18 人、Yuquan 16 人。
- 数据：691,314 个 train80 事件、172,849 个 chronological heldout20
  事件。
- 模型：冻结的 `linear_state` checkpoint；34 人 × 3 seeds，共 102 个
  单元。
- 条件输入：held-out 事件真实的首个 rank set；首 rank 不计作模型预测成功。
- 生成：之后每一步只使用模型自己刚生成的 contact，直到 STOP 或所有 contact
  已出现。
- 三个生成分量：

  1. train80 contact participation log-frequency；
  2. 冻结 linear-state 相对 no-prefix field 的 ordered-history residual；
  3. train80 rank-progress termination hazard。

- 七个配对条件：`full_constructive`、`static_only`、
  `static_shuffle`、`history_h1`、`history_h2`、`constant_stop` 和
  `no_termination`。
- 所有条件共享完全相同的首 rank 和随机数。
- 训练与 rollout 全程禁止读取 A/B、物理轴、seizure label 和 ictal value。
- 无监督双模态和物理轴只在 rollout 冻结后由 train80 人体事件定义。

## 3. 工程验收

`102/102` 单元和 `714/714` condition rows 完成；没有 source 改写、重复
contact、非连续 rank、随机数错配或 target leakage。单元中位运行时间约
3.50 s，中位峰值 GPU allocation 约 39 MB，中位 RSS 约 0.96 GB；没有 OOM。

正式机器验收：

- `results/topic5_constructive_event_generation/analysis_v0_1/machine_acceptance.json`
- `results/topic5_constructive_event_generation/analysis_v0_1/gate_verdict.json`

## 4. 核心结果

### 4.1 历史分量能恢复局部 transition fingerprint

相对 `static_only`，完整模型明显改善：

| Endpoint | 中位收益 | 改善患者 | Wilcoxon P |
|---|---:|---:|---:|
| first-order transition MAE | 0.00277 | 28/34 | \(8.10\times10^{-6}\) |
| first-order transition correlation | 0.1026 | 30/34 | \(7.67\times10^{-7}\) |

Epilepsiae 和 Yuquan 中 transition-correlation 增益方向一致，不由单一数据集
驱动。相对 shaft-preserving static shuffle，完整模型也改善 transition
correlation（26/34，P=0.00015）和 participation（26/34，P=0.00013）。

这说明冻结的 history residual 不是完全无效；它确实保留了局部
“哪个 contact 之后更容易接哪个 contact”的统计。

### 4.2 局部转移没有组合成真实的完整事件

Gate B 的承重 endpoint 没有通过：

| Endpoint | 中位收益（正值有利 history） | 改善患者 | Wilcoxon P |
|---|---:|---:|---:|
| suffix rank Wasserstein | -0.00385 | 13/34 | 0.075 |
| suffix precedence correlation | -0.00876 | 16/34 | 0.761 |
| suffix precedence MAE | -0.00893 | 12/34 | 0.0328 |
| suffix participation MAE | 0.00444 | 20/34 | 0.370 |

precedence MAE 的显著性方向是完整模型更差，不是阳性结果。只有 9/34 患者
在 participation、rank 和 precedence 三项中至少两项落入人体
heldout-half vs heldout-half 的 `+10%` 经验变异范围；预设要求为至少
17/34。

`full` 相对 H1 或 H2 在 suffix rank 上均无增益（P=0.278 和 0.826）。
因此失败不是简单因为“历史窗口还不够长”；旧状态既没有显示必要性，也没有
修复 free-running error accumulation。

### 4.3 termination grammar 是独立且可靠的分量

train80 rank-progress hazard 相对 `constant_stop` 和 `no_termination`
均在 34/34 患者改善 event-length Wasserstein 和 STOP-hazard MAE，四项
检验均为 \(P=1.16\times10^{-10}\)。

这个结果只能解释为：

> 间期事件长度/结束位置具有稳定的 rank-progress grammar。

它不能解释为 RNN 发现了生物抑制恢复、真实时间常数或发作倒计时，因为 STOP
来自独立的经验 hazard，单位也是 rank step。

### 4.4 全局双向 read-back 没有被 history 恢复

train80-only read-back 得到：

- 31/34 人有可重复的无监督两模态表示；
- 25/34 人有可靠且 heldout 两侧均有支持的 source-to-sink PCA 物理轴；
- 两者交集为 22 人。

这不是把旧 A/B 当金标准，而是在更宽松的、label-free 坐标中询问生成事件
是否恢复双向组织。

在这 22 人中：

- template fidelity：完整模型只在 4/22 人优于 static-only，中位收益
  -0.0241，P=0.00366；方向为 history 更差；
- signed-axis fidelity：8/22 人改善，中位收益 -0.00344，P=0.222；
- 只有 2/22 人的 template 与 signed-axis 两项同时落入人体经验变异范围。

完整模型确实能在 22/22 人生成两个符号方向，但“生成了两边”不等于恢复了
正确的模板、比例和轴向位移分布。

由于 Gate B 已失败，这些数字只作诊断；Gate C 的正式状态是
`LOCKED_NOT_EVALUATED`，不是独立的第二个 FAIL。SNN fingerprint pipeline
也按合同保持 `LOCKED_BY_HUMAN_SUFFICIENCY_GATE`，没有新跑 SNN。

## 5. 对核心科学目标的判断

### 可以保留

1. 间期 rank 序列含有超出静态 contact participation 的局部一阶转移信息。
2. 该信息可被一个短程 linear-state residual 压缩，并在自由生成中恢复总体
   transition fingerprint。
3. 事件终止具有稳定但独立的 rank-progress grammar。

### 不能写

1. 当前 RNN 能自由生成真实的完整 A/B 传播事件；
2. 当前 RNN 恢复了患者病理轴；
3. 当前 hidden state 是可辨识的生物慢变量；
4. 当前结果连接了 IEI、发作倒计时或 early-ictal state；
5. 当前阴性否定真实数据中的 A/B 模板或 SNN 机制。

最安全的结论是：

> Ordered interictal history preserved local first-order transition
> statistics, but the frozen linear-state correction was not sufficient to
> compose those local statistics into realistic free-running whole-event
> propagation. Event termination was captured by a separate rank-progress
> grammar rather than by the recurrent state.

## 6. 为什么局部阳性与全局阴性不矛盾

模型每一步的小偏差会在 free running 中被再次作为输入，逐步改变 candidate
competition、sink 和后续路径。只匹配平均 first-order transition matrix，
不保证同时匹配：

- 每场事件的 rank 分布；
- 多步 precedence；
- 分支竞争；
- 模板 prevalence；
- source-to-sink signed displacement。

因此这轮排除的是“当前 additive static + linear history residual + empirical
STOP 已经构成充分生成模型”，不是排除 ordered history 本身。

## 7. 图与数据产物

主验收图：

- `results/topic5_constructive_event_generation/figures/topic5_constructive_event_generation_sufficiency_v0_1.png`
- 同名 PDF 和 metadata JSON。

图的六块依次回答：模型合同、局部与整场生成的分离、绝对 posterior
fidelity、termination 必要性、双模态/物理轴 read-back、预注册 gate。

统计产物：

- `cell_condition_metrics.csv`：102 seeds × 7 conditions；
- `patient_condition_metrics.csv`：先按患者合并 seeds；
- `empirical_variability_reference.csv`：人体 heldout split-half 误差；
- `readback_subject_inventory.csv`：双模态与物理轴资格；
- `paired_tests.json`：全部患者级配对检验。

## 8. 收口决定

本 goal 已运行到预注册停止点并应当结束：

1. 不调 history residual scale、temperature、hidden size 或 seed；
2. 不因 read-back 阴性改用旧 A/B 作监督标签；
3. 不开放 SNN fingerprint；
4. 不把这条线升级为主文 Figure 6 机制结果。

若论文需要保留，最合适的位置是 Extended Data / Supplementary
bounded computational result，用于说明“局部顺序信息存在，但不等于一个
可自由生成全局双向事件的可辨识 RNN state”。真正的 `when` 问题仍需另立
跨事件持续状态、显式 IEI 和可靠 seizure-specific target 的合同，不能在当前
event-reset 模型上继续加层。

# Topic 5 跨事件历史 RNN 到发作早期能量场 v0.1

## 1. 唯一科学问题

本合同只检验：

\[
\boxed{
\text{按真实时间积累的间期群体事件历史}
\rightarrow z_{\mathrm{history}}(t)
\rightarrow \text{下一次发作的 early-ictal contact-energy field}
}
\]

操作性主假设为：

\[
\operatorname{Perf}(M_2)>
\operatorname{Perf}(M_1),
\]

其中 `M1` 和 `M2` 看到完全相同的事件、contact、几何和静态 scaffold；二者唯一差别是 `M2` 保留跨事件的真实时间顺序与持续状态，`M1` 只做无序汇总。

本合同不把以下结果当作主终点：

- 单场事件内部 next-contact 预测；
- A/B 聚类或病理轴自动恢复；
- 患者平均间期场与患者平均发作场的静态相关；
- hidden-state PCA 的低维性；
- 完整事件自由生成。

这些既有结果只用于限定输入编码器或给出对照，不代替 `M2 > M1`。

## 2. 冻结术语

- \(h_{p,e,t}\)：一场间期群体事件内部、到第 \(t\) 个 rank set 为止的 prefix state。
- \(u_{p,e}\)：EventRNN 在整场事件结束后的 event embedding。
- \(z_{p,e}\)：跨事件持续、受真实 IEI 衰减的 history state。
- \(Y_{p,s,i}\)：患者 \(p\) 第 \(s\) 次发作在 contact \(i\) 上的 clinical-onset `[0,10] s`、`1–150 Hz` early-ictal energy。

既有 event-reset RNN 的正式名称固定为：

> `INTERICTAL_WITHIN_EVENT_SEQUENCE_ENCODER_QUALIFICATION`

旧的 early-ictal exporter/readback 只可称为：

> `WITHIN_EVENT_ORDER_BRIDGE_PILOT`

不得把其中任何 hidden state 写成 \(z_{\mathrm{history}}\)。

## 3. 数据合同

### 3.1 间期输入

主输入沿用 masked rank dataset 的 contact inventory、事件参与和事件内 rank-set 顺序。每个事件必须带：

- `event_abs_time`；
- 原始 `block_id` / recording identity；
- `segment_id`；
- 距上一场合格事件的 `delta_t_sec`；
- 是否在当前 segment 起点；
- contact participation；
- contact relative rank；
- EventRNN 输入所需的 contact features。

`segment_id` 在以下位置增加：

1. recording 不连续或 block coverage 存在不可解释 gap；
2. seizure/ictal interval；
3. postictal exclusion 结束后的第一场事件；
4. 数据合同规定的其他 fail-closed boundary。

不能仅因一段时间没有群体事件就重置；若 recording 连续，只通过时间衰减更新状态。

### 3.2 发作 target

Primary target 固定为 Epilepsiae strict clinical-onset cohort 中已接受的：

- anchor：`clinical_onset`；
- window：`[0,10] s`；
- band：`1–150 Hz`，保留既有 line-noise mask；
- unit：逐发作、逐 contact 能量场；
- contact join：事件 contact 必须精确映射到 target contact，同名、一一对应、顺序显式重排。

每次发作只能使用：

\[
t_e < t_{\mathrm{clinical\ onset}}-10\ \mathrm{min}
\]

的事件。10 min 为冻结 guard。任何 target 数值不得用于事件筛选、contact feature、offset calibration、early stopping 或超参数选择。

### 3.3 因果 prefix

对第 \(s\) 次发作，状态提取必须从其当前连续 segment 起点开始，按绝对时间依次输入所有合格事件，并在 onset minus guard 处停止。不得使用固定 `train80/heldout20` 代替 seizure-specific causal prefix。

若两次发作在 guard 前共享同一个最后事件，它们具有同一 \(z_{s^-}\)。统计中必须以“不同 patient-specific history state”为有效配对单位，不能把重复 state 的 seizure rows 当独立样本。

## 4. 两级递归计算图

### 4.1 EventRNN

EventRNN 保留已验收的 `LinearStateSequenceRNN` 形式，并在每场事件开始时归零：

\[
h_{e,t}=A_h h_{e,t-1}+B_h x_{e,t},
\qquad
u_e=h_{e,T_e}.
\]

它只编码单场事件内部 contact-rank propagation。正式 V1 中 EventRNN 必须：

- 使用外层 LOSO：训练 shared core 时排除 heldout patient；
- 不读取任何 ictal target；
- heldout patient 只允许用 onset guard 之前的间期事件做 local calibration；
- 对每场事件独立 reset；
- 导出 event embedding、contact embedding 和输入 fingerprint。

现有 checkpoint 可用于工程 smoke 和转导性诊断，不能作为确认性 G2/G3 的最终 encoder。

### 4.2 HistoryRNN

HistoryRNN 在事件之间持续：

\[
\tilde z_{e-1}
=
\exp[-\operatorname{softplus}(\gamma)\Delta t_e]\odot z_{e-1},
\]

\[
z_e=\operatorname{GRUCell}(u_e,\tilde z_{e-1}).
\]

约束：

- `delta_t_sec` 必须来自绝对时间；
- \(\gamma\) 只能解释为 event-history decay，不直接解释为细胞级生物时间常数；
- `z` 只在 §3.1 指定边界重置；
- 不允许 patient ID、time-to-seizure、ictal source、A/B label 或 target-derived contact parameter 进入 HistoryRNN。

## 5. G1：HistoryRNN 自监督资格

在打开 early-ictal target 前，HistoryRNN 必须预测下一场间期事件的 contact field：

1. participation：逐 contact BCE；
2. relative rank：只在真实参与 contact 上计算 Huber loss，并附 pairwise-order sensitivity。

输出由同一 history state 和冻结/contact-query embedding 生成：

\[
\hat y^{\mathrm{part}}_{e+1,i}
=
\sigma(b_i+\phi_i^\top A_{\mathrm{part}}z_e),
\]

\[
\hat y^{\mathrm{rank}}_{e+1,i}
=
\phi_i^\top A_{\mathrm{rank}}z_e.
\]

Matched nonrecurrent baseline 必须看到完全相同的 event embeddings 和时间范围，但只使用：

- permutation-invariant mean/max pooling；
- last-event embedding；
- 当前 history 的事件数、跨度和 last-event gap。

正式 across-event order sensitivity 使用 causal-prefix-matched null：对每个 heldout decision 固定完全相同的已观察事件、目标、contact mask、真实时间槽和 last-event embedding，只置换最近 64 个已观察事件中最后事件之前的顺序；更早历史状态保持真实并冻结。不得用整段全局置换替代正式 null，因为它会改变每个 decision 的 prefix event set。开发期整段置换只作工程诊断，不参与 gate。

G1 primary contrast 是 chronological HistoryRNN 相对 matched nonrecurrent baseline 的 heldout next-event participation BCE。relative-rank 为共同 secondary endpoint。结果按 patient-first 汇总，并分 Epilepsiae/Yuquan 报告。

三位 development patients（`epilepsiae_1073`、`epilepsiae_1146`、`yuquan_chenziyang`）不进入确认性 patient-first 推断。G1 primary cohort 为 31 位 development-excluded patients；34 人全体只作 supportive analysis。

G1 通过条件：

- cohort median `BCE(M1) - BCE(HistoryRNN) > 0`；
- patient-level one-sided paired test `p < 0.05`；
- 至少两个数据集方向一致，不能只由一个数据集驱动；
- across-event order shuffle 后增益显著降低。

G1 不通过时，停止 G2/G3；不得把状态命名为 \(z_{\mathrm{history}}\)。

## 6. G2：early-ictal field 的嵌套模型

### 6.1 公共基线

三组模型共享下列输入：

- 静态 causal contact participation prior；
- contact geometry / shaft nuisance；
- 当前 causal history 中事件数、时间跨度、last-event gap；
- last event 的 EventRNN embedding；
- 相同事件 embeddings 的 permutation-invariant pooling。

定义：

- `M0`：static + geometry + scalar context；
- `M1`：`M0` + unordered event pooling + last-event embedding；
- `M2`：`M1` + chronological HistoryRNN state。

Primary contrast 固定为：

\[
M_2-M_1.
\]

`M2 > M0` 或对 contact shuffle 显著不能替代 `M2 > M1`。

### 6.2 latent-to-contact readout

History state 只能通过跨患者共享的低参数 contact query 映射：

\[
L_{p,s,i}=\phi_{p,i}^{\top}Az_{p,s^-}.
\]

对每次发作在 contact 维中心化 \(L\) 和 target，以去掉任意全局 shift。V1 使用冻结 EventRNN/contact queries 和普通 shared readout；V2 才允许 rank 2–4 的低秩 readout sensitivity。

禁止：

- 为 heldout patient 用 ictal target 拟合自由 per-contact bias；
- 把 target 的患者平均场直接作为 heldout patient baseline；
- 使用未来发作或 onset 后数据做 calibration；
- 同时把 true-order latent 和 order-shuffled latent 放进同一模型。

### 6.3 外层确认

G2 使用 target-patient LOSO。`epilepsiae_1146` 因参与 development，只进入 supportive analysis；primary target cohort 为其余 15 位 strict clinical-onset patients：

1. 排除 heldout patient 后训练 EventRNN shared encoder；
2. 排除 heldout patient 后训练 HistoryRNN 和 ictal readout；
3. heldout patient 只用每次 onset guard 前的间期数据构建 causal features/state；
4. heldout patient 的 ictal field 只用于一次最终评分。

Primary metric 是 contact-centered field 的 heldout Spearman \(\rho\)；secondary 为 centered MSE / cosine。每位患者先在 seizure 内评分，再以不同 history state 为单位折叠，最后 patient-first 推断。

## 7. G3：state–seizure 特异配对

在同一患者、至少有两个不同 causal history states 时，比较：

\[
z_{s^-}\rightarrow Y_s
\quad\text{vs}\quad
z_{s'^-}\rightarrow Y_s,\ s'\ne s.
\]

Primary 是 correct-minus-wrong within-patient pairing。对于至少三次不同历史且 target residual 有足够支持的患者，增加 leave-one-seizure-out：

\[
\delta Y_s=Y_s-\bar Y_{-s},
\qquad
\delta z_s=z_s-\bar z_{-s}.
\]

G3 通过才允许使用“state-conditioned early-ictal prediction”。G2 通过而 G3 不通过时，只能写“chronological history improves patient-level early-ictal field prediction”，不能写 seizure-specific state。

## 8. 控制

至少保留：

1. within-event rank shuffle：破坏每场事件内部 propagation order；
2. across-event history order shuffle：保留 event embeddings 集合，破坏真实跨事件顺序；
3. matched unordered model：G1/G2 primary baseline；
4. within-patient history–seizure circular shift：G3；
5. contact shuffle：只解释空间场是否高于通道随机，不解释 history state；
6. guard sensitivity：5/10/30 min，10 min 为 primary；
7. decay sensitivity：固定为小网格，只能在 development patients 上冻结。

## 9. 版本边界

- `V1`：冻结的双级 self-supervised encoder + shared ictal readout；主版本。
- `V2`：rank 2–4 target readout；只作 sensitivity。
- `V3`：仅当 V1 的 G2/G3 通过，冻结 EventRNN，只微调 HistoryRNN low-rank adapter/head，并保留 next-event self-supervised loss。
- matched GRU/其他 architecture robustness 只在 V1 阳性后运行。

不再先扩展 architecture zoo。

## 10. Gates 与停止规则

| Gate | 问题 | 最低通过条件 | 失败后的动作 |
|---|---|---|---|
| G0 | 因果 prefix/segment/guard/contact join 是否可构建 | metadata audit 完整；无 target value；有效 subject/history 分母冻结 | 修数据合同，不训练 |
| G1 | 真正的跨事件状态是否存在 | HistoryRNN > matched unordered next-event predictor，且 order control 支持 | 停止 ictal bridge |
| G2 | history state 是否增量预测 early-ictal field | LOSO `M2 > M1` | bounded negative，不调 target head 刷结果 |
| G3 | 是否 seizure-state specific | correct pairing > within-patient wrong pairing | 限定为患者级 history association |
| G4 | 是否跨架构稳健 | 仅在 G2/G3 后做 | 不改变主结果 |
| G5 | 是否需要有限 joint alignment | 仅在 V1 阳性后做 | 不改变 V1 主结果 |

任何 gate 未通过都按其科学层级报告，不把未运行的下游 gate 写成失败。

## 11. 预期 Figure

仅在 G2 至少通过后制作主候选图：

| Panel | 科学含义 |
|---|---|
| A | EventRNN 与持续 HistoryRNN 的两级计算图，明确 event reset 与 cross-event persistence |
| B | G1：chronological state 相对 matched unordered 的 next-event field 增益 |
| C | history decay 与 across-event order-shuffle control |
| D | G2：LOSO `M2-M1` 的患者级 early-ictal field 增量 |
| E | G3：correct 与 wrong within-patient state–seizure pairing |
| F | 代表患者的 causal history trajectory、预测场与真实场，禁止选择性展示 |

若 G2 不通过，只生成带明确 bounded-negative 标注的诊断图，不进入主文 Figure 6。

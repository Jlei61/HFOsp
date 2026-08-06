# Topic 5 间期历史状态到发作早期场 direct transfer v0.2

## 1. 唯一科学问题

本合同直接检验：

\[
\boxed{
\text{某次发作前的 causal interictal history}
\rightarrow
\text{history-dependent state/field}
\rightarrow
\text{该次发作的 early-ictal contact-energy field}
}
\]

上一版 G1 只检验 next-interictal-event contact field。其状态固定为：

```text
G1_NEXT_EVENT_PROXY:
PROVISIONAL_BOUNDED_NEGATIVE
```

G1 与本轮 direct transfer 是并行证据，不再以 G1 阳性作为读取 early-ictal target 的必要条件。G1 阴性削弱“通用 chronology state”解释，但不能逻辑上否决发作特异的迁移任务。

## 2. 冻结结论边界

本轮若阳性，最多支持：

> 某次发作前的间期历史包含与随后 early-ictal spatial field 有关的预测信息。

单靠预测 RNN 不允许写：

- 已预测发作发生时间；
- 已证明间期活动因果塑造病理网络；
- latent state 是细胞级 E/I 状态；
- 学到的 decay 是生物时间常数。

“activity-dependent network shaping” 需要 human predictive bridge 与独立 SNN intervention mechanism 共同支持。

## 3. 数据与因果合同

Primary endpoint 保持冻结：

- `clinical_onset`；
- `[0,10] s`；
- `1–150 Hz`；
- 逐发作、逐 contact 能量场；
- onset 前 10 min guard；
- exact contact join。

对每次发作 \(s\)，所有输入必须由：

\[
\mathcal H_{p,s}=\{X_{p,e}:t_{p,e}<t_{p,s}^{onset}-600\,s\}
\]

构建。以下量必须逐发作从 causal prefix 重算：

- contact participation prior；
- normalization 和 contact query；
- unordered summary；
- EWMA state；
- RNN history state。

不能使用该次发作之后的间期事件，不能用发作 target 选择 history window、contact 或 checkpoint。

## 4. 模型阶梯

所有模型共享 exact contact set、geometry nuisance 和逐发作 causal static prior。

### M0：static causal scaffold

患者在该次发作前所有合格事件的 contact participation prior，加 geometry nuisance，并显式消费论文已冻结、target-blind 的 TA/TB earliness、support 和 template fields。后者只从 `results/interictal_propagation_masked/template_gradient_fields/per_subject/` 读取，禁止从 seizure data 重估。

冻结 TA/TB field 是全记录得到的患者级 trait covariate，不是逐发作动态状态；它只用于控制论文已知的 static morphology。由于该 target 已被既往分析读取，而且 frozen field 不是 prospective prefix-only 重估，本轮只能称为 reused-target internal cross-state validation，不能称为临床前瞻预测。所有会随发作时间改变的量仍必须严格从该次发作的 causal prefix 构建。

### M1：matched nonrecurrent history

M0 加：

- unordered mean/max event summary；
- last-event embedding；
- event count、history span、last-event gap。

### E0.5 / E2 / E6：单时间尺度活动积分器

分别用 0.5、2、6 h 衰减的 causal EWMA contact participation/rank field。它们检验可交换的 activity-load slow state，不要求复杂事件顺序。

### EM：multi-horizon activity state

同时使用 0.5、2、6 h EWMA fields，检验多时间尺度累积是否优于 M1。

### R2：chronological HistoryRNN

M1 加 target-blind HistoryRNN 输出的 contact field。HistoryRNN 使用真实 IEI 和事件顺序，权重来自 target-blind G1 训练，不按 ictal 结果再调参。

本版的 primary object 是 **frozen self-supervised state transfer**：HistoryRNN dynamics 不读取 early-ictal target；只有跨患者共享的低参数 ictal readout 在 outer-training patients 上拟合。当前结果无论阳性或阴性，都只判定这一 frozen representation 是否携带跨状态信息。不得因为 frozen transfer 阴性就直接写成“任何 RNN 都学不到 early-ictal field”，也不得在 primary 结果落地前端到端微调 recurrent dynamics。

Primary direct-transfer contrast：

\[
\operatorname{Perf}(R2)-\operatorname{Perf}(M1).
\]

并列机制比较：

\[
\operatorname{Perf}(E2)-\operatorname{Perf}(M1),
\qquad
\operatorname{Perf}(EM)-\operatorname{Perf}(M1).
\]

## 5. 必须控制

### 5.1 strict order control

固定：

- 相同事件集合；
- 每场事件内部 rank；
- event count；
- 总时间跨度；
- IEI 时间槽；
- last event；
- static/M1 branches。

只把 last event 之前的 event embeddings 重新分配到既有时间槽。用同一 R2 readout 比较 true 与 shuffled state。

### 5.2 state-seizure pairing

在同一患者至少两个不同 causal states 时，用同一 M1/contact layout，交换 R2 history fields：

\[
R2(z_{s^-})\rightarrow Y_s
\quad\text{vs}\quad
R2(z_{s'^-})\rightarrow Y_s.
\]

该比较不以患者间身份差异充当证据。

### 5.3 seizure-specific residual

对至少三次不同 causal states 的患者，评价：

\[
\delta Y_s=Y_s-\bar Y_{-s},
\qquad
\delta \widehat Y_s=\widehat Y_{R2,s}-\widehat Y_{M1,s}.
\]

该分析只作 patient-local、evaluation-only 的 residual check；heldout target 不参与 readout 拟合。

### 5.4 target headroom

必须报告：

- M0/M1 absolute heldout performance；
- 同患者不同发作 early-ictal field 的一致性；
- leave-one-seizure-out patient-mean oracle；
- 有多少患者具备 seizure-specific residual 的可靠分母。

## 6. G1 诊断收口

在把 v0.1 G1 称为正式 bounded negative 前，必须补：

1. synthetic chronology signal recoverability；
2. 3/10/30 coverage-cycle 局部收敛审计；
3. history-state variance；
4. history-to-output readout norm；
5. zero-state/state-field ablation；
6. decay 参数是否有有效梯度/是否离开初始化。

训练预算判定必须先于最终 RNN transfer 口径：

- 若 10→30 cycles 的 heldout BCE 变化中位绝对值 `<0.002` 且 chronological increment 方向一致，则 10-cycle 视为 target-blind plateau budget；16 个 direct-target folds 必须用该预算重训后再冻结 R2 结果；
- 若 10→30 cycles 仍不稳定，则 c3/c10 direct R2 只能记为 training-sensitive provisional，不得写成模型类正式阴性；此时用预先审计过的最长预算 c30 重训全部 16 folds，并把 c10 与 c30 并列作为 training-budget robustness，而不是根据 early-ictal target 选择其中较好者；
- 只有 c10 与 c30 的 R2 增量、absolute performance 和 temporal controls 方向一致，才允许把 direct-transfer 结果称为 training-budget robust。若两者不一致，最终状态必须保留为 `TRAINING_SENSITIVE_PROVISIONAL`，无论哪一个预算的单独 P 值如何；
- EWMA、target headroom 和 pairing denominator 不因 RNN checkpoint budget 改变，但最终图和 summary 必须明确记录所用 `history_cycles`。

诊断未完成前，v0.1 状态固定为：

```text
ENGINEERING_EXECUTION: PASS
G1_NEXT_EVENT_PROXY: PROVISIONAL_BOUNDED_NEGATIVE
CHRONOLOGY_SPECIFIC_STATE: NOT_SUPPORTED_UNDER_CURRENT_OBJECTIVE
```

## 7. 推断合同

- target-patient LOSO；
- primary cohort 排除 development patient `epilepsiae_1146`；
- 15 人 primary，16 人 supportive；这 16 人均来自当前具备逐发作 clinical-onset target 的 Epilepsiae inventory。34 人 interictal cohort 可用于 target-blind representation learning，但不能冒充 early-ictal transfer 分母；
- 先逐 seizure 评分，再按不同 history fingerprint 折叠，最后对每位患者取 seizure-level 指标中位数并进行 patient-first 推断；
- primary metric：contact-centered Spearman \(\rho\)；
- secondary：centered MSE、cosine；
- primary 统计：patient-level one-sided Wilcoxon，并同时报告效应量和 bootstrap CI；
- absolute spatial readout 的主 null 是每患者、每次 seizure 独立进行的 all-contact target-label shuffle，`n_perm=5000`；每个 draw 先逐 seizure 重算相同 Spearman，再以患者中位数折叠。报告 `rho_data - median(rho_shuffle)`、患者正 margin 数、Wilcoxon 和每患者 permutation p；
- channel shuffle 只回答预测场是否超过通道随机，不能替代 primary history contrast `R2-M1`；
- `EWMA-0.5h`、`EWMA-2h`、`EWMA-6h` 和 multi-horizon integrator 构成同一比较家族，必须做 BH-FDR；
- 相对 M1 的增量只有在对应模型的绝对 patient-level \(\rho>0\) 也通过时，才能称为有效预测；否则只记为相对改善候选；
- G3 和 residual 即使 primary direct contrast 阴性也照常报告，但固定为 secondary seizure-specific analysis；其阴性不能反向否定患者级 G2 transfer。

## 8. 结果分级

| 结果 | 允许结论 |
|---|---|
| M0 最好 | 主要是稳定 scaffold |
| EWMA > M1，R2 不优于 EWMA | 支持累积型 activity-load state，不需要复杂顺序 |
| R2 > M1 且 true > shuffle | 支持 chronology-sensitive history information |
| correct pairing > wrong pairing | 支持 seizure-conditioned state |
| residual check 也阳性 | 支持 seizure-specific early-ictal reconfiguration |
| 只有患者平均场阳性 | patient fingerprint/static scaffold，不是动态状态 |

任何层级均不得单独升级为因果“塑造”。

## 9. 执行和产物

新结果写入：

```text
results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/
```

旧 v0.1 产物只读保留，不覆盖。长任务使用可恢复 launcher，逐 fold 写 `DONE.json` / `FAILED.json` / log。图目录必须带中文 `figures/README.md`。

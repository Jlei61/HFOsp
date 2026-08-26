# Continuous Marked State R1.4 / T2-R2.0 合同

**冻结日期：** 2026-08-27
**状态：** development-only；formal test 与 sealed partition 保持关闭。
**一句话目标：** 先在六位预先固定患者中复现跨窗口 state→IED 预测，再检验在相同 pre-event state 下，最近 100 次 IED 的不可预测部分是否通过 event-triggered edge 改变下一事件和后续状态。

## 1. 科学位置

当前 closeout 支持三位 development 患者中的 persistent predictive memory，并支持 first subset 与 continuation 的 state dependence；raw waveform 尚无稳定独立增量。H3 只有 25–200 次事件历史与下一事件 STOP/extent 的前置信号，既有短 T2 为结构零，long/very-long T2 又受状态退化、免费截距、短有效核、重叠对照、伪重复和拟合发散影响。

因此下一版不继续超长 boxcar，而按以下顺序推进：

1. R1.4 六患者 H1/H2a 复现；
2. 稳定 T1 患者上的 N=100 一步 T2-R2.0；
3. 只有 R2.0 可估计并有增量后，才扩 N=50/200 与 physical-time sensitivity；
4. H2b 旧仪器修复后重跑一次，H3b 暂缓。

## 2. 冻结患者与分区

### R1.4 六患者

- `epilepsiae_620`
- `epilepsiae_958`
- `yuquan_huanghanwen`
- `epilepsiae_922`
- `yuquan_pengzihang`
- `yuquan_hanyuxuan`

选择在任何新 R1.4/T2-R2.0 结果前冻结。前三位来自 formal R1.3；E922、彭子航扩展数据来源与记录结构；韩宇轩作为预先指定的异质性患者，不按新结果选择。

### T2-R2.0 首轮

优先 `epilepsiae_620` 与 `epilepsiae_958`，因为两者已有 persistent 与 correct-time development 证据。其余患者只有在 R1.4 中形成可解释的 persistent/time-specific state 后才加入。没有 N=1,000 长整窗不构成 N=100 T2 的排除理由。

所有 epoch 选择仅使用 TRAIN 内 chronological inner-validation；development validation 只作一次最终评分。formal test、sealed partition 与 seizure labels 不参与训练或选择。

## 3. R1.4 输入与模型

每 30 s 构造 causal observation：

`O_k = {X[t_k-30s:t_k], S_k, M_k, C_p}`，

其中 `X` 为 IED-core-masked raw SEEG，`S` 为 per-contact spectral、variance、autocorrelation，`M` 为坏道与记录缺口 mask，`C_p` 为坐标与 shaft embedding。

固定事件历史 `r(t)` 包括 time since last IED、recent counts、last load、recent STOP/mark、time/session/coverage；不增加第二个 free event-history RNN。

### 主 observer

explicit spectral + variance + autocorrelation 经 contact MLP 与 coordinate-aware spatial encoder 得到 `o_k_exp`。

### raw sensitivity

raw Transformer 从同 seed explicit checkpoint 接入小 residual gate：`o_k = o_k_exp + alpha E_raw(X_k)`。不再比较 Transformer/Conformer 或扩大 encoder 家族；raw 是否有增量不阻断 H1/H2a。

### persistent state

`z_k^- = mu + exp(dt_k K)(z_{k-1}^+ - mu)`，其中 `K = Omega - Q`、`Omega^T = -Omega`、`Q >= 0`；`z_k^+ = z_k^- + U(o_k, z_k^-)`。主维数固定为 `d_z=8`。

## 4. R1.4 输出、损失与比较

模型联合输出 recorded-support intensity 与 exact tied-group sequential mark：STOP/group size、未募集 contacts 上的精确无放回 subset、later continuation 与 same-prefix continuation。

唯一训练目标是 recorded intervals 上的 timing survival likelihood 加 sequential mark likelihood。不加入频谱或 waveform 重建、contrastive/KL、seizure loss 或 latent consistency penalty。

主要比较：

1. persistent vs memoryless；
2. correct-time vs 每个 anchor 5–10 个 matched wrong-time donors；
3. explicit+raw vs explicit；
4. timing、STOP/size、first subset、later continuation、same-prefix continuation 分解。

wrong-time donor 必须来自同一记录覆盖段，并匹配 time of day、time since last IED、recent IED rate、last-event load/STOP、observation coverage 与 session position。

允许结论：

- persistent 胜 memoryless：`persistent predictive memory`；
- 再胜 matched wrong-time：`time-specific persistent state estimate`；
- 仅 STOP/size 改善：termination/extent memory；
- first subset 或 continuation 改善：state-dependent contact repertoire prediction；
- raw 胜 explicit：raw waveform 提供显式统计之外的增量。

这些仍是预测证据，不自动升级为生物因果或自主生理状态。

## 5. T2-R2.0 exposure 与 edge

主尺度固定 `N=100 events`，位于 H3-S0 的 25–200-event 稳定范围中部，不按患者或 seed 选择。

### 事件 innovation

对事件属性 `phi(m_e)` 在 TRAIN 内 cross-fit：

`phi_hat_e = E[phi(m_e) | z_e^-, r_e, o_e^-]`，
`eta_e = phi(m_e) - phi_hat_e`。

第一主 source 为 scalar load innovation；第二 source 为去除 total load 后的 participation composition，独立运行、独立报告。

### 累积 exposure

`x_e = exp(-1/N) x_{e-1} + eta_e`。

### event-triggered generator edge

事件 `e` 由 pre-event state `z_e^-` 预测；事件结束后才更新：

`z_e^+ = z_e^- + B x_e`。

T1 对应 `B=0`。T2 首轮从冻结 T1 checkpoint 克隆，observer、K、history baseline、timing/mark decoder 全部冻结，只训练低秩 `B`。`B` 允许正负方向，不预设 IED 只增加易感性。

## 6. T2-R2.0 四臂与端点

四个核心臂从同一 T1 checkpoint 与同一 pre-event state 出发：

1. T1 no-edge；
2. real cumulative exposure；
3. state-matched、历史不重叠的 donor exposure；
4. current-event-only jump。

fitted-intercept 仅作常数 offset 诊断，不称同容量对照；`real-no_edge` 单独不能承担 exposure 结论。

### 一级：next-event counterfactual

anchor event 后应用 real/placebo edge，关闭下一事件前 raw correction，比较事件 `e+1` 的 timing 与 exact mark。这是首要 estimability 与增量检验。

### 二级：one-shot persistence

只在 anchor event 应用一次 jump，之后关闭 raw correction、不再应用新 T2 jump，各臂使用相同真实 event-history covariates；比较 H5/H10 的状态与 mark 预测。

若差异只存在于 `e+1`，命名为 `exposure-conditioned next-event prediction`；只有差异通过冻结 generator 延续到 H5/H10，才允许称 `exposure-induced state update`。

## 7. 估计器与合成诊断

人体运行前必须持久化：

- `B=0` 处 TRAIN gradient norm；
- exposure variance、design rank 与边界压缩比例；
- edge 参数是否离开初始化；
- positive truth、zero truth、reversed-sign truth 的 synthetic recovery；
- donor 与 real 历史不重叠、pre-event state matching 质量；
- recorded segment、gap 与 TRAIN/validation 因果边界。

这些用于判定某个 T2 对比是否可估计，不是阻断 H1/H2a 或普通探索的总 gate。edge 留在零、T1 不稳定、readout 退化或 donor 不可构造时，记为不可估计，不作生物学阴性。

## 8. 统计与扩展规则

- seed 先在患者内取中位；患者为主要统计单位；
- event likelihood 用于拟合，不把事件数或滑动窗口数当患者数；
- 患者内不确定性用连续时间块 bootstrap；
- 不选择患者特异最佳 N；
- 不按结果剔除患者或 seed；
- ordinary negative results 继续完成，不把单一对照设成全项目 gate。

只有 N=100 的 real edge 确实离开零，并同时优于 donor 与 current-event，才扩到 N=50/200、按 TRAIN median IEI 匹配的 physical-time arm，以及 event merge/thinning sensitivity。N=1,000–2,000 与六小时 boxcar 暂不进入主实验。

## 9. H2b/H3b 收口

H2b 立即用修复后的 pseudo-onset 代码重跑旧 12 个产物一次并归档。待 R1.4/T2 冻结后，使用新 state 在 5/15/30/60/120 min lead 上做 patient-first probe。seizure loss 不反向训练 state。

H3b 只有在 T2 edge 可估计并冻结后才运行；否则保持未支持。

## 10. 执行与交付

所有长作业使用可恢复的独立进程，显式限制 CPU thread 与显存，OOM 时按 batch/chunk 降级而不改变科学合同。每个作业写原子 manifest、checkpoint、seed、代码版本和 split hash，可断点续跑。

交付包括：

1. R1.4 六患者 patient-first 表与端点分解；
2. T2-R2.0 synthetic recovery、estimability audit 与人体 next-event/H5/H10 表；
3. H2b 修复后一次性重跑报告；
4. 白话版与技术版报告；
5. 正式分区未打开、无 paper-ready 图变更、无无关工作树修改的审计。

## 11. 当前禁止表述

- raw SEEG 已识别出队列普遍的生理慢状态；
- IED 已因果塑造 epileptic network；
- N=100 是患者特异最佳时间尺度；
- 仅下一事件预测改善就证明 generator state 被长期改变；
- H2b 修复前数字是冻结论文结果；
- 超长 boxcar 阴性否定 H3。

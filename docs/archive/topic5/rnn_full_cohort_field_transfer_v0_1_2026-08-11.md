# Topic 5 RNN 全 cohort 间期学习与 early-ictal field transfer v0.1

日期：2026-08-11
状态：**执行完成并纳入 Topic 5 RNN closeout；正式口径见本报告与 Topic 5 主文档**

## 1. 这次修复了什么

此前 LBSS/RNN 外部评分沿用了旧 history-RNN `outer_*` cache 的 exact join，只剩 10 人、24 次发作。该分母不是“只有 10 人有发作数据”，也不是 Figure 3 的正式发作 cohort。

本轮重新锁定两个互不替代的分母：

- 间期患者内自监督 RNN：`dataset_v0_4` 全部 34 位 K=2 患者，每人 3 seeds；
- 发作外部测试：Figure 3D 同一母清单，17 位患者、全部 167 次 phenotype-matched seizures。

发作数据只在 17/17 个模型场冻结并写入 manifest 后读取，不参与训练、checkpoint、mode 匹配或场构造。

## 2. 模型和评分

本轮不重新训练。直接复用已经完成的 34×3 个 converged patient-within contact-space linear-state RNN 及其 heldout native rollouts。

每位有 Fig.3 target 的患者：

1. 只用间期 train events 拟合 K=2 mode read-back；
2. 将每个 seed 的 heldout native rollout 分到两个 mode；
3. 从自由生成事件计算 contact participation 与平均 normalized rank；
4. 用冻结经验间期 TA/TB rank 对两个 mode 做 target-free post-hoc 命名；
5. 在 Figure 3 已冻结的患者 plane、sigma、mirror/maxAB 合同上形成两个 RNN fields；
6. 使用 Figure 3 的 0–10 s phenotype-matched activation 和同步 all-contact channel-shuffle null 评分；
7. seizure 先在患者内折叠，再做 cohort paired inference。

因此本轮检验的是：患者内 RNN 生成出的间期空间场，是否与同患者发作早期能量场对应。模型没有从零独立发现物理平面，几何坐标仍来自已冻结的人体间期场。

## 3. 结果

### 3.1 间期传播生成，n=34

- RNN native rollout transition correlation 中位数：0.8669；
- static-only generator 中位数：0.4450；
- 患者配对差中位数：+0.3188；
- 33/34 患者为正；
- one-sided paired Wilcoxon：P=1.16×10⁻¹⁰。

这支持：患者内自监督 RNN 能从间期 contact-rank sequences 中学习可在 heldout events 自由生成的患者特异传播规律。

### 3.2 冻结 RNN field 与 early-ictal field，n=17 / 167 seizures

- RNN-field maxAB |r| 患者中位数：0.8166；
- synchronized all-contact shuffle 中位数：0.7967；
- 患者配对 margin 中位数：+0.0297；
- 11/17 患者为正；
- 预定义方向的 one-sided paired Wilcoxon：P=0.0443；
- two-sided patient sign-flip：P=0.116。

因此当前可以写：

> 只用间期事件训练并冻结的患者特异 RNN-generated fields，在 Figure 3 全 cohort 中显示高于 all-contact channel-shuffle 的正向跨状态对应。

但不能把这一结果写成逐发作预测、独立外部验证或连接机制已被证明。方向性 Wilcoxon 达到 0.05，而双侧 sign-flip 未达到 0.05，二者应一起保留在统计表。

### 3.3 E1146

- 15 个真实 SEEG field contacts；
- Figure 6-D 使用 canonical Figure 3-B 已冻结的 seizure 2；
- 该 target 严格为 clinical onset 0–10 s、1–150 Hz broadband energy，而不是把 gamma-only seizures 混入代表场；
- RNN-generated TA vs empirical TA：Spearman ρ=0.957；
- RNN-generated TB vs empirical TB：Spearman ρ=0.904。

代表图显示 RNN TA、RNN TB 和 canonical early-ictal energy field 使用同一真实 tissue-plane layout；经验 TA/TB 不在 D 中重复绘制，而放在 B 的 data–RNN event comparison 中。

### 3.4 连接结构回顾不是 C/E 的模型定义

Figure 6-C/E 使用的是本轮 34 人 patient-within `LinearStateSequenceRNN`（hidden size 32），不是空间布线 RNN。其 recurrence 是 hidden-state persistence，不应从 C/E 反推 tissue connectivity。

Figure 6-F–H 单独读取已冻结的 v0.4 connectivity-motif 结果：Dense、Sparse、Local、Spatial+cost 都能学习间期传播；真实顺序 recurrent arms 相对 no-recurrence 的 next-contact gain 为强阳性。Spatial+cost 在 10% active edges 下保持相近传播表现，但旧 exact-join early-ictal n=10 不能区分具体 connectivity motif；这部分只作为 connectivity sufficiency / efficiency 回顾，不替代 E 的 17 人跨状态统计。

## 4. 允许和不允许的结论

允许：

1. 34 位患者中，RNN 在患者内学会了比静态参与度更完整的间期传播结构；
2. 冻结 RNN-generated interictal fields 在 17 人/167 次发作上有正向跨状态对应；
3. E1146 的两个 RNN modes 分别复现经验 TA/TB，而不是一个折中单峰。

不允许：

1. RNN 独立恢复了真实解剖连接或物理轴；
2. RNN 预测了具体哪次发作何时发生；
3. 所有患者都存在相同强度的 cross-state reuse；
4. 10 人/24 seizures 是唯一可用发作 cohort。

## 5. 工程验收

- 34/34 patients；
- 3/3 seeds per patient；
- 17/17 model fields 在 target access 前冻结；
- 167/167 primary seizures；
- legacy `outer_*` cache reads = 0；
- target-based training/selection = false；
- 新增回归测试 4/4 通过；
- 无重新训练、无 GPU、无 OOM、无 NaN。

## 6. 产物

- 分母与输入审计：`results/topic5_rnn_full_cohort_field_transfer_v0_1/INPUT_AUDIT.json`
- target-free field manifest：`results/topic5_rnn_full_cohort_field_transfer_v0_1/MODEL_FIELD_MANIFEST.json`
- 间期患者表：`results/topic5_rnn_full_cohort_field_transfer_v0_1/interictal_patient_statistics.csv`
- 发作 event/patient/cohort 表：`results/topic5_rnn_full_cohort_field_transfer_v0_1/ictal_*_statistics.csv`
- 统计摘要：`results/topic5_rnn_full_cohort_field_transfer_v0_1/SCORE_SUMMARY.json`
- E1146 图：`results/paper-ready-figure/fig6_rnn_full_cohort_field_transfer/figures/topic5_rnn_e1146_field_transfer.*`
- cohort 图：`results/paper-ready-figure/fig6_rnn_full_cohort_field_transfer/figures/topic5_rnn_full_cohort_interictal_ictal.*`
- 八面板 Figure 6 候选：`results/paper-ready-figure/fig6_rnn_full_cohort_field_transfer/figures/topic5_figure6_rnn_full_cohort.*`
- 逐图中文说明：`results/paper-ready-figure/fig6_rnn_full_cohort_field_transfer/figures/README.md`

八面板候选严格按当前证据链排版：A 将 E1146 真实 contact layout 直接叠加在非规则 recurrent nodes 上，触点颜色为同一 rollout 的输出 rank，左右保留竖向 rank input/output；B 是同一批 chronological events 的 TA/TB data 与 same-start native rollout；C 是 34 人间期传播统计；D 是 E1146 RNN TA、RNN TB 与 canonical Figure 3-B seizure 2 的 0–10 s、1–150 Hz early-ictal energy，每个方形 field 后各有独立竖直色条；E 是 17 人、167 次发作的 cohort paired statistic；F–H 另起一行回顾 spatial connectivity constraints、21 人间期计算充分性和旧 exact-join n=10 的 motif-level early-ictal benchmark。F 统一以 edge length 着色，使 Dense、随机 Sparse、纯 Local 与 Spatial+cost 的连接范围可直接比较；B 的列逐事件匹配，不从 data/RNN 分布中各自挑选案例；H 与 E 的分母和模型族不得混写。

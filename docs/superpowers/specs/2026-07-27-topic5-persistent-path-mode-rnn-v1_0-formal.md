# Topic 5 / Figure 6：persistent path-mode graph RNN v1.0 formal contract

**日期**：2026-07-27
**状态**：执行合同
**目的**：纠正 v0.9 pilot 把完整传播顺序当主门、却没有完成 34 人触点级
rank-distribution 验证的问题。

## 1. 核心科学问题

模型的第一任务不是重新发现 A/B，也不是精确复刻每次事件的完整传播顺序，而是只从
间期 contact-rank 事件中学习患者内稳定的触点级分布：

1. 每个触点参与事件的概率；
2. 每个触点在事件内的完整归一化 rank distribution；
3. 由大量自由生成事件自然形成的常见传播路径；
4. 固定路径图、方向和抑制状态是否对上述分布有必要贡献。

pairwise precedence 和 whole-path distance 保留为 secondary dynamics
diagnostics，不再单独否决触点级主问题。

## 2. 冻结输入

- 34 位患者，使用 `dataset_v0_4` masked contact-rank events。
- 每位患者 chronological train80 / heldout20。
- axis、path bases 和 mode prior 只从 train80 构造。
- A/B label、IEI 和任何发作期量均不得进入构图、训练或模型选择。
- 所有输入 fingerprint 沿用 v0.9，不重建数据集或 path prior。

## 3. 冻结模型

- 固定 `K=2`：这是与结果无关、能表示多路径的最小结构。
- 每个事件选择一个 `(path mode, direction)` component 并持续到 STOP。
- 共享 leaky contact state、固定患者路径图、共享 inhibition、患者 local offset。
- component posterior 只能使用已经观察到的事件 prefix。
- 不再根据 v0.9 pilot 选择 K 或调动力学参数。

## 4. 正式运行

- 34 heldout subjects × 3 seeds。
- 每个 heldout fold 在其余 33 人上训练 shared parameters。
- shared training 完整扫 train80 两轮；每患者每轮固定 8 次 optimizer update。
- heldout local offset 完整扫其 train80 四轮；每轮固定 8 次 update。
- batch 1024；每个 condition 自由生成 5000 个事件。
- conditions：
  - `no_history`；
  - `merged_path`；
  - `K=2 intact`；
  - `K=2 weight_shuffle`；
  - `K=2 mode_shuffle`。
- intact 同时运行 graph、inhibition、forward、reverse、mode-collapse 和
  dominant-mode lesions。

共 `34 × 3 × 5 = 510` 个独立 LOSO runs。

## 5. 主终点和统计门

Primary metrics：

- participation MAE，越低越好；
- per-contact rank-distribution Wasserstein distance，越低越好。

每位患者先取三 seed 中位数。对 intact 相对每个主对照的 benefit，要求两个 primary
metrics 均满足：

1. cohort median benefit > 0；
2. 改善患者 >17/34；
3. patient-level directional Wilcoxon 经全部主比较 BH-FDR 后 `q<0.05`。

结构必要性要求 graph lesion 或 mode-collapse lesion 在两个 primary metrics 上满足
同一方向，并至少 18/34 患者恶化。Inhibition、方向和 dominant-mode lesions 为机制
分解，不单独阻断主门。

由于 v0.9 曾查看 `epilepsiae_1073`、`epilepsiae_1146` 和
`yuquan_chenziyang` 三位开发病例，34 人全队列仍是主合同，但必须并列报告排除这三人的
31 人 development-excluded sensitivity。该敏感性不改变已经冻结的 34 人硬门；论文中
不得把包含开发病例的全队列称为完全独立外部验证。

Secondary：

- heldout next-contact NLL；
- pairwise precedence MAE；
- whole-path distance；
- component posterior entropy；
- per-contact mean rank、early/middle/late probability；
- mode occupancy 与 event-prefix state/inhibition trajectory。

## 6. 发作期边界

只有触点级主门通过后，才允许在冻结模型上读取 clinical onset `[0,10] s`、
`1–150 Hz` baseline-robust-z 静态能量场。发作期任务是冻结 readout，不反向训练
RNN，不选择 K，不修改 path prior。

若主门失败，正式 34 人结果仍作为完成的 bounded-negative cohort test；不得用发作
结果挽救模型。

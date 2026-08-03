# Topic 5：history-conditioned early-ictal field refinement v0.4 结果报告

## 一句话结论

联合 RNN 残差没有在患者级改善已经有效的静态 A/B 场。

这句话只描述本轮 15 位 strict clinical-onset 患者的回顾性 LOSO 结果，不把它升级成唯一发作方向、发作时间预警或因果机制。

## 1. 这轮实际测了什么

本轮先冻结论文已经有效的患者特异静态 A/B 间期病理场，再问：发作前、同一连续记录段内的间期事件历史，能否通过一个受限残差支路，把这两个静态候选场修正得更接近该次发作 clinical onset 后 0–10 s 的 **1–45 Hz contact-energy field**。正式评分仍为两个候选中较高的绝对 Spearman（maxAB），不要求模型选择唯一 A/B 或正负号。

训练和评价严格按患者外层留一：每个 fold 的 event encoder 与 HistoryGRU 均未见过 heldout patient；14 位训练患者提供 early-ictal supervision，heldout patient target 只在最终评分时读取。三个固定 seed 的 A/B candidate fields 先逐 contact 平均，再计算 seizure maxAB，随后按 patient median 做队列统计。

## 2. 静态 A/B 锚点

- 15 位患者、31 次发作；每位患者的评分场覆盖 6–16 个触点，中位 9 个。
- M0 的患者中位 maxAB 为 **0.5545**。
- 相对每患者 5000 次 matched all-contact channel shuffle，患者中位 margin 为 **+0.1500**；11/15 位高于各自 null median，5/15 位超过各自 p95；患者级 exact P=0.01245。
- 因此，本轮不是让 RNN 从零生成发作场，而是在一个已存在信息的静态 A/B 基底上检验历史增量。
- **这个锚点的强度只对 all-contact 标签置换成立**。队列级显著来自 15 个患者 margin 的符号检验；单个患者只有 5/15 超过自己的 p95。评分触点跨 1–6 根电极杆（中位 3 根，1 位患者只有一根），所以本轮 null 没有把"同一根杆上相邻触点天然相似"这部分几何贡献扣掉；Topic 5 既往 peri-onset 工作已经显示 within-shaft null 会明显收紧这类对应。本轮把它当作"有信息的起点"，不当作静态场空间特异性的独立证明。

## 3. 四个模型

- **M0**：冻结 static A/B，不训练。
- **M1**：冻结 target-blind HistoryGRU，只训练 contact-query residual heads 与两个共享 gain；回答已有状态是否已经可读。
- **M2**：不用 recurrence，只使用固定 2 h EWMA、历史 mean/max、last event、event count 和 history span；回答简单历史内容/负荷是否足够。
- **M3**：从与 M1 完全相同的 30-epoch head checkpoint 分叉，再联合微调 HistoryGRU/decay 与 head 30 epochs；回答 early-ictal supervision 是否需要改变 recurrent dynamics。

所有模型共用同一静态 A/B、contact denominator、1–45 Hz target、patient-first loss 与 LOSO split。没有 architecture sweep、best-seed 选择或根据 heldout target early stopping。

## 4. Primary 和模型解释对比

- **Primary，M3−M0**：患者中位差 +0.0000，bootstrap 95% CI [-0.0500, +0.0000]；1 正 / 7 负 / 7 并列，exact P=0.03906（n=15）。
- **M3−M1**：患者中位差 +0.0000，bootstrap 95% CI [+0.0000, +0.0000]；2 正 / 3 负 / 10 并列，exact P=0.8125（n=15）。
- **M3−M2**：患者中位差 -0.0167，bootstrap 95% CI [-0.0273, +0.0000]；1 正 / 8 负 / 6 并列，exact P=0.05469（n=15）。
- **M1−M0**：患者中位差 +0.0000，bootstrap 95% CI [-0.0476, +0.0000]；2 正 / 7 负 / 6 并列，exact P=0.05469（n=15）。
- **M2−M0**：患者中位差 +0.0000，bootstrap 95% CI [+0.0000, +0.0059]；5 正 / 3 负 / 7 并列，exact P=0.8438（n=15）。

这些对比分开报告，没有复合 hard gate。M3−M0 只回答“联合 history residual 是否改善静态场”；M3−M1 和 M3−M2 才决定这种改善是否需要改变 recurrent dynamics、以及是否超过简单时间汇总。

### 4.1 训练集内确实学到了东西，只是没有迁移

45 个单元的训练集 soft-maxAB 中位轨迹：

| 训练阶段 | 起点 | 终点 | 中位增益 |
|---|---:|---:|---:|
| common frozen-recurrent head（30 epochs） | 0.5496 | 0.5608 | +0.0111 |
| M1 continuation（+30 epochs） | 0.5618 | 0.5980 | +0.0366 |
| M3 joint（+30 epochs） | 0.5615 | 0.5850 | +0.0235 |
| M2（60 epochs） | 0.5497 | 0.6026 | +0.0518 |

因此本轮阴性不是“残差支路根本学不动”：每个模型在自己的 14 位训练患者上都把 soft-maxAB 推高了，只是这些增益没有迁移到留出患者。

**M3−M1 必须带着这条限定读**：M3 在训练集上的增益（+0.0235）低于 M1 continuation（+0.0366），而两者起点相同、mini-batch 顺序相同、head 学习率相同。所以“留出集上 M3≈M1”与“联合微调 recurrent dynamics 在固定预算内更难优化”是混淆的，不能单独读成“改变 recurrent dynamics 没有价值”。可能的来源包括联合阶段的 state 每步都在移动，以及 head 与 GRU 共享同一个 gradient-clip 预算。

## 5. 历史是否真有特异性

- **M3 true order−完整历史顺序打乱**：患者中位差 -0.0006，bootstrap 95% CI [-0.0186, +0.0000]；2 正 / 9 负 / 4 并列，exact P=0.1016（n=15）。
- **M3 correct history−同患者其他发作 history swap**：患者中位差 -0.0037，bootstrap 95% CI [-0.0273, +0.0015]；2 正 / 5 负 / 3 并列，exact P=0.2969（n=10）。

顺序对照对整段 causal history 做事件身份置换并保留原时间槽，每个 seed 32 次；不是只洗最近 64 个事件。History-swap 保持患者、静态 A/B、contact set 和 target 不变，只替换成同患者另一场发作的历史；只有一场合格发作的患者不进入这一对比。

**顺序对照的两臂聚合方式不完全对称**：正式 M3 分数先把三个 seed 的候选场逐 contact 平均再评分，而打乱臂每个 seed 抽自己的置换、只能先评分再平均。把真实臂也改成“逐 seed 评分再平均”后，同构口径给出中位差 -0.0006，3 正 / 8 负 / 4 并列，exact P=0.123（n=15）。两种口径结论一致：真实事件顺序没有优势。History-swap 臂本来就是 seed 场平均，与正式分数同构。

## 6. 绝对信息和频带敏感性

- M3 的患者中位 observed maxAB 为 **0.5471**，matched channel-null 中位为 **0.3667**，中位 margin **+0.1083**；11/15 位高于 null median，5/15 位超过 p95。
- 1–150 Hz 只做 no-retrain sensitivity：M3−M0 患者中位差 **+0.0000**，2 正 / 6 负 / 7 并列。它没有参与模型、seed 或超参数选择。

## 7. 模型内部实际改了多少

- M1 heldout 状态范数中位 1.3048；candidate A/B 相对 static 的夹角中位分别为 3.96° / 5.09°。
- M2 candidate A/B 相对 static 的夹角中位分别为 3.67° / 5.14°。
- M3 heldout 状态范数中位 1.3059；candidate A/B 相对 static 的夹角中位分别为 2.71° / 3.78°。
- M3 学到的 rank-step/clock-time decay 在这里只是模型记忆参数，不解释为细胞级 E/I 时间常数。

这些量用于确认模型做的是“受限修正”还是重写静态场，不把 raw hidden coordinate 当作唯一神经流形。

## 8. 工程验收和资源

- 工程验收：**ACCEPTED**；45/45 单元完成，失败 0。
- 训练日志固定为每单元 150 行：common head 30、M1 continuation 30、M3 joint 30、M2 60。
- 最大单进程显存 100.4 MB；单元中位运行时间 15.2 min。
- 验收只检查 leakage、outer-fold 坐标、训练预算、有限数值、控制分母和 artifact 完整性；科学效应大小不是工程 gate。

## 9. 科学边界

1. 静态 A/B 不读取 early-ictal target，但由患者全记录间期事件回顾性估计，可能包含目标发作之后的事件。因此 residual-history 支路严格 causal，整体模型却不是完全前瞻预测器。
2. 输出是 A/B 两候选的无符号集合，不是唯一发作方向。
3. 该任务预测发作早期空间场，不预测发作何时发生。
4. 只有 correct-history 超过 within-patient swap，才把增量称为 seizure-matched；只有 true-order 超过整段 shuffle，才称为顺序特异。
5. 6–16 个触点的评分分辨率较粗，患者级精确并列是预期现象，统计已使用 1e-9 tie band 和 exact sign-rank null。
6. **这条阴性覆盖的历史尺度**：每次发作只读同一连续记录段内、cutoff 之前的事件，31 次发作的前缀事件数为 10–6125（中位 310），跨度中位 3.87 h，最后一个事件到 cutoff 的间隔中位 0.48 h；模型的时钟记忆半衰期中位约 2.11 h。因此本轮检验的是“发作前数小时之内的事件历史”，跨天或跨记录段的长程网络塑形不在被检验的模型族里。
7. 队列级阴性由 15 位患者的符号检验支撑，其中 M3−M0 有 7 位精确并列；本轮没有跑合成阳性对照，所以“检出真实增量的灵敏度下界”未被独立标定。
8. 静态 A/B 的信息量只对 all-contact 标签置换 null 成立（见 §2），电极杆几何贡献未被扣除。

## 10. 产物

- 六联图：`figures/history_conditioned_field_refinement_six_panel.png` 与同名 PDF。
- 中位效应代表病例：`figures/representative_history_refinement.png` 与同名 PDF。
- 患者级结果：`history_conditioned_field_patient_metrics.csv`。
- 5000-draw null：`history_conditioned_field_channel_null.csv`。
- raw state/residual：`history_conditioned_field_state_diagnostics.csv.gz`。
- 正式汇总：`HISTORY_CONDITIONED_FIELD_SUMMARY.json`。
- 工程验收：`ACCEPTANCE.json`。
- 可复现清单：`REPRODUCIBILITY_MANIFEST.json`。

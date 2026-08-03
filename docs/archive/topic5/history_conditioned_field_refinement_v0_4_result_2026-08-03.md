# History-conditioned early-ictal field refinement v0.4：正式结果

## 审阅结论

**科学合同执行完成，工程验收 `ACCEPTED`。静态 A/B 间期病理场在当前 15 人队列中可复现 early-ictal 空间信息，但加入发作前 causal history 的 RNN 残差没有进一步改善该静态场。**

这轮不是让 RNN 从零生成一个有符号的单场发作图，也不要求它识别唯一 A/B 方向。它冻结论文已经使用的 A/B 两个静态候选场，用同一套 sign-free `maxAB` 指标训练和评价受限历史残差，因此与已有数据结果直接对齐。

## 1. 数据和任务

- primary cohort：15 位 development-excluded strict clinical-onset 患者，31 次发作；外层按患者 LOSO。
- target：clinical onset 后 `[0,10] s` 的 `1–45 Hz` contact-energy field；`1–150 Hz` 只作 no-retrain sensitivity。
- 每位患者只在静态 A/B 与 early-ictal target 精确同名相交的 6–16 个触点上评分，中位 9 个。
- 每次发作只使用 onset−10 min 之前、同一连续记录段内的 causal interictal history；事件顺序和真实时间间隔均进入 history model。
- 三个固定 seed（11、29、47）的 candidate fields 先逐 contact 平均，再算 seizure `maxAB`，随后按患者 median 进入队列统计。
- 静态 A/B 不读取 early-ictal target，但来自患者全记录的回顾性间期数据，可能包含目标发作之后的事件；因此本分析不是完全前瞻预测器。

## 2. 四个冻结模型

- `M0_STATIC_AB`：静态 A/B，不训练。
- `M1_FROZEN_HISTORY_HEAD`：冻结 target-blind HistoryGRU，只训练 residual head 和两个共享 gain。
- `M2_TIME_AWARE_NONRECURRENT`：固定 2 h EWMA、history mean/max、last event、event count 和 span，经一个线性投影进入同一 residual head。
- `M3_JOINT_RNN`：与 M1 共享相同的 30-epoch head 起点，再联合微调 HistoryGRU/decay 和 head 30 epochs。

所有模型共用同一静态 A/B、target、contact denominator、patient-first loss 和 LOSO split。没有 architecture zoo、best-seed 选择或由 heldout target 驱动的 early stopping。科学效应不作为运行 hard gate，各对比独立报告。

## 3. 静态 A/B 锚点复现

- M0 患者中位 `maxAB = 0.5545`。
- 相对每患者 5000 次 all-contact channel shuffle，患者中位 margin `+0.1500`，11/15 位高于各自 null median，5/15 位超过各自 p95。
- 患者级 exact signed-rank `P = 0.01245`。

因此本轮起点不是无信息的静态场。后续 M3 若仍超过 channel null，首先可能只是继承 M0；历史信息必须由 M3−M0、顺序打乱和 history-swap 单独判断。

## 4. Primary 与模型分解

| 对比 | 患者中位 ΔmaxAB | bootstrap 95% CI | 正 / 负 / 并列 | exact P |
|---|---:|---:|---:|---:|
| **M3−M0（primary）** | `+0.0000` | `[-0.0500, +0.0000]` | 1 / 7 / 7 | 0.0391 |
| M3−M1 | `+0.0000` | `[+0.0000, +0.0000]` | 2 / 3 / 10 | 0.8125 |
| M3−M2 | `−0.0167` | `[−0.0273, +0.0000]` | 1 / 8 / 6 | 0.0547 |
| M1−M0 | `+0.0000` | `[−0.0476, +0.0000]` | 2 / 7 / 6 | 0.0547 |
| M2−M0 | `+0.0000` | `[+0.0000, +0.0059]` | 5 / 3 / 7 | 0.8438 |

M3 没有改善静态 A/B；差值方向反而以不变或下降为主。M3 也不优于冻结 history state 的 M1 或简单非递归的 M2，因此当前结果不支持“为了匹配 early-ictal field，必须联合改变 recurrent dynamics”。

## 5. 历史特异性对照

- `M3 true order − full-history shuffle`：中位 `−0.0006`，2 正 / 9 负 / 4 并列，`P = 0.1016`。
- `M3 correct history − within-patient seizure history swap`：中位 `−0.0037`，2 正 / 5 负 / 3 并列，`P = 0.2969`，n=10。

顺序 null 对整个 causal prefix 置换事件身份并保留原时间槽，每 seed 32 次，共 96 个 seed-draw realizations/患者；不是旧版只洗最近 64 个事件的弱扰动。History-swap 保持患者、静态 A/B、contact set 和 target 不变，只替换同患者另一场发作的 history。两项均未显示真实顺序或正确 seizure-history 配对的优势。

## 6. 绝对信息、模型行为和灵敏度

- M3 observed `maxAB` 中位 0.5471，matched channel-null 中位 0.3667，margin `+0.1083`；11/15 高于 null median，5/15 超过 p95。
- 该绝对阳性不能归因于历史支路，因为 M3−M0 为零且 M0 本身已经超过 null。
- M3 把静态 candidate A/B 改动的中位夹角仅为 2.71° / 3.78°；最终 gain 中位约 0.052 / 0.072。模型确实形成非零状态和受限修正，但这些修正没有带来 heldout 增益。
- M3 history half-life 参数中位约 2.11 h；这里只是模型的 clock-time memory 参数，不解释为细胞级 E/I 时间常数。
- `1–150 Hz` no-retrain sensitivity 的 M3−M0 中位仍为 0（2 正 / 6 负 / 7 并列）。

## 7. 工程验收与资源

- 45/45 formal units（15 LOSO × 3 seeds）完成；0 failed、0 OOM、无训练/预测 NaN。
- 每单元固定 150 行训练记录：common head 30、M1 continuation 30、M3 joint 30、M2 60。
- 最大单进程显存 100.4 MB；单元中位运行 15.2 min；正式运行使用 12 个并行 worker。
- 5000-draw matched channel null 共 300,000 行；full-history order control 共 1,440 行；swap 可用 26 个 patient-seizure entries。
- 45 项专项/回归测试通过。输入审计、训练、统计、画图和报告均有独立脚本与 manifest。

## 8. 论文可用结论

安全表述：

> 患者特异的静态 A/B 间期传播场在 strict clinical-onset 队列中保持与发作早期 contact-energy field 的 sign-free 空间对应。然而，在相同 maxAB 合同下，利用发作前 causal interictal history 学习的受限 RNN 残差没有改善静态场，也没有显示真实事件顺序或正确 history–seizure 配对的优势。

不能写成：RNN 成功预测了唯一发作方向、预测了发作时间、恢复了细胞级 E/I 状态，或证明间期历史因果塑造了发作场。当前结果适合作为 Supplementary bounded computational result；主信息是静态跨状态 scaffold 保留，而逐发作 history refinement 未建立。

## 9. 产物

- 主汇总：`results/topic5_history_conditioned_field_refinement_v0_4/HISTORY_CONDITIONED_FIELD_SUMMARY.json`
- 工程验收：`results/topic5_history_conditioned_field_refinement_v0_4/ACCEPTANCE.json`
- 患者表：`results/topic5_history_conditioned_field_refinement_v0_4/history_conditioned_field_patient_metrics.csv`
- 六联图：`results/topic5_history_conditioned_field_refinement_v0_4/figures/history_conditioned_field_refinement_six_panel.{png,pdf}`
- 代表病例：`results/topic5_history_conditioned_field_refinement_v0_4/figures/representative_history_refinement.{png,pdf}`
- 图说明：`results/topic5_history_conditioned_field_refinement_v0_4/figures/README.md`
- 可复现清单：`results/topic5_history_conditioned_field_refinement_v0_4/REPRODUCIBILITY_MANIFEST.json`

实现提交：`c6dc6503`；冻结 spec/plan 提交：`20983fe7`。

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

**这个锚点的适用范围（2026-08-03 审阅补）**：队列级显著性来自 15 个患者 margin 的符号检验，单个患者只有 5/15 超过自己的 p95。评分触点跨 1–6 根电极杆（中位 3 根，仅 `epilepsiae_139` 一位是单杆），而本轮 null 只做 all-contact 标签置换，没有扣除"同一根杆上相邻触点天然相似"的几何贡献 —— Topic 5 既往 peri-onset field similarity 工作已显示 within-shaft null 会把这类对应明显收紧（`docs/archive/topic5` 的 Fig3-B 系列）。因此本节只支持"本轮的静态起点带信息"，**不**支持"静态 A/B 与发作早期场的对应是杆几何以外的空间特异性"。后者需要 within-shaft null，本轮 6–16 触点的分母下多数患者不具备可行的杆内置换空间。

## 4. Primary 与模型分解

| 对比 | 患者中位 ΔmaxAB | bootstrap 95% CI | 正 / 负 / 并列 | exact P |
|---|---:|---:|---:|---:|
| **M3−M0（primary）** | `+0.0000` | `[-0.0500, +0.0000]` | 1 / 7 / 7 | 0.0391 |
| M3−M1 | `+0.0000` | `[+0.0000, +0.0000]` | 2 / 3 / 10 | 0.8125 |
| M3−M2 | `−0.0167` | `[−0.0273, +0.0000]` | 1 / 8 / 6 | 0.0547 |
| M1−M0 | `+0.0000` | `[−0.0476, +0.0000]` | 2 / 7 / 6 | 0.0547 |
| M2−M0 | `+0.0000` | `[+0.0000, +0.0059]` | 5 / 3 / 7 | 0.8438 |

M3 没有改善静态 A/B；差值方向反而以不变或下降为主。M3 也不优于冻结 history state 的 M1 或简单非递归的 M2，因此当前结果不支持“为了匹配 early-ictal field，必须联合改变 recurrent dynamics”。

### 4.1 训练集内拟合与 M3−M1 的混淆（2026-08-03 审阅补）

45 个单元训练集 soft-maxAB 的中位轨迹：common head `0.5496 → 0.5608`（`+0.0111`）、M1 continuation `0.5618 → 0.5980`（`+0.0366`）、M3 joint `0.5615 → 0.5850`（`+0.0235`）、M2 `0.5497 → 0.6026`（`+0.0518`）。

两个后果必须随结论一起讲：

1. **阴性不是"学不动"**。四个模型都在自己的 14 位训练患者上把目标函数推高了；失败点在跨患者迁移，不在优化。
2. **M3−M1 是混淆的**。M3 与 M1 起点相同、mini-batch 顺序相同、head 学习率相同，但 M3 的训练集增益（`+0.0235`）低于 M1（`+0.0366`）。所以"留出集 M3≈M1"同时兼容"改变 recurrent dynamics 没有价值"和"联合优化在 30 epochs 预算内更难收敛"。可能来源：联合阶段每步 state 都在移动，以及 `torch.nn.utils.clip_grad_norm_` 在联合阶段把 head 与 GRU 放进**同一个** clip 预算（回归测试 `test_m1_and_m3_stages_differ_only_in_recurrent_trainability` 只有在关闭 clip 时两臂才逐参数相等，正是这条差异的显式记录）。M3−M1 只能写成"未观察到联合微调带来增量"，不能写成"target supervision 不需要改变 recurrent dynamics"。

## 5. 历史特异性对照

- `M3 true order − full-history shuffle`：中位 `−0.0006`，2 正 / 9 负 / 4 并列，`P = 0.1016`。
- `M3 correct history − within-patient seizure history swap`：中位 `−0.0037`，2 正 / 5 负 / 3 并列，`P = 0.2969`，n=10。

顺序 null 对整个 causal prefix 置换事件身份并保留原时间槽，每 seed 32 次，共 96 个 seed-draw realizations/患者；不是旧版只洗最近 64 个事件的弱扰动（回归测试 `test_order_shuffle_permutes_the_entire_causal_prefix` 与 `test_order_shuffle_moves_identities_but_keeps_the_original_time_slots` 把这两条锁住）。History-swap 保持患者、静态 A/B、contact set 和 target 不变，只替换同患者另一场发作的 history。两项均未显示真实顺序或正确 seizure-history 配对的优势。

**两臂聚合方式的不对称（2026-08-03 审阅修）**：正式 M3 分数先把三个 seed 的候选场逐 contact 平均再评分，而打乱臂每个 seed 抽自己的置换、只能先评分再平均，两者不是逐项同构。补算的同构口径（真实臂也改成"逐 seed 算 patient median 再对 seed 取平均"）见 `HISTORY_CONDITIONED_FIELD_SUMMARY.json::comparisons.true_minus_order_shuffle_seed_matched`：中位 `−0.00056`、3 正 / 8 负 / 4 并列、`P = 0.1230`（原口径 `−0.00064`、2/9/4、`P = 0.1016`）。两种口径同向同量级，结论不变。History-swap 臂本来就按 seed 场平均，与正式分数同构，无需重算。

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
- `tests/test_topic5_history_conditioned_field_v0_4.py` 17 项（审阅后由 11 项扩到 17 项），加上 `tests/test_topic5_history_rnn.py` 的 v0.1/v0.2 遗留回归。输入审计、训练、统计、画图和报告均有独立脚本与 manifest。

## 8. 论文可用结论

安全表述：

> 患者特异的静态 A/B 间期传播场在 strict clinical-onset 队列中保持与发作早期 contact-energy field 的 sign-free 空间对应。然而，在相同 maxAB 合同下，利用发作前 causal interictal history 学习的受限 RNN 残差没有改善静态场，也没有显示真实事件顺序或正确 history–seizure 配对的优势。

不能写成：RNN 成功预测了唯一发作方向、预测了发作时间、恢复了细胞级 E/I 状态，或证明间期历史因果塑造了发作场。当前结果适合作为 Supplementary bounded computational result；主信息是静态跨状态 scaffold 保留，而逐发作 history refinement 未建立。

## 9. 代码审阅记录（2026-08-03）

对 spec/plan、4 个 src 模块、9 个 runner、45 个冻结单元与全部统计产物做了完整复核。**没有发现会改变四条结论方向的问题**；`HISTORY_CONDITIONED_FIELD_SUMMARY.json` 全部既有字段在重跑 summarizer 后逐位复现，`history_conditioned_field_{seizure_metrics,channel_null_summary}.csv` 逐字节相同。

已确认干净的部分：留出患者的 target 在训练器中不可见（固定 epoch，无 early stopping，因此 v2.6 那条"early stopping 以静态初始化为基准"的缺陷不重演）；每个 outer fold 的 event encoder 与 history 初始化 hash 唯一且来自同一 LOSO checkpoint；event embedding 的 mean/scale 来自同 fold 的 train-only 统计；`contact_features` 只含 causal participation 与纯电极几何，没有回顾性数据列；order shuffle 覆盖整段 causal prefix 且保留原时间槽（v0.2 的 64-事件窗口 bug 不重演）；channel null 每 draw 重做 A/B 与符号选择、同一置换用于 4 个模型；`g=0` 逐位还原 static A/B。

审阅修复（P1）：

1. **顺序对照两臂聚合不同构** → 补 `true_minus_order_shuffle_seed_matched`（§5）。
2. **5 条预注册工程判据没有单元测试** → 补 6 个回归测试：整段前缀置换、置换保留原时间槽、production 批量/padding 路径的 chunk parity（旧测试只覆盖 `run_history_to_cutoff`，而正式运行走的是 runner 内的 `_batch_history_states`）、anchor penalty 只作用于 recurrent 参数、M1/M3 恒等扰动同构、seed-matched 聚合。
3. **训练集拟合轨迹缺失**，导致 M3−M1 的混淆无法判读 → 补 §4.1。
4. **图内硬编码统计**（`Above p95: M0 5/15; M3 5/15`）→ 改为从 summary JSON 派生；同时补 panel C 图例、修 panel A 标签与散点重叠、把 panel F 的边界说明从数据点上移到轴标题下。
5. **静态锚点的 null 强度未标注** → §3 补 within-shaft 几何未扣除的适用范围。
6. **历史尺度未量化** → 见下方覆盖边界。

记录但未改动（P2，改动会使冻结产物失效或属设计选择）：

- `_anchor_penalty` 除以参数个数，与 spec §6.1 的 `λ‖θ−θ₀‖²` 写法不同，实际 λ 被削弱约三个数量级。方向上给 M3 更多自由度，对本轮阴性保守。下一版若重跑需与 spec 对齐或改写 spec。
- 联合阶段 head 与 GRU 共享同一个 gradient-clip 预算（`clip_grad_norm_` 传入合并参数列表），M1 只 clip head。实测 M3 joint 触发 clip 的 epoch 比例中位 0，最大 20%，与 M1 continuation（中位 3.3%，最大 20%）同量级，不是 §4.1 差距的主因，但它确实是 M1/M3 之间的第二处差异。
- `_time_summary` / `run_history_to_cutoff` 在 `src/` 与 runner/cache builder 中各有一份实现，数值一致但会漂移；新增的 production-path parity 测试是当前的护栏。
- 训练按患者逐个 `optimizer.step()`，spec §6.1 字面是"先按 seizure 平均再按患者平均"。患者等权这一科学意图保留，只是优化路径不同。
- `accept_*.py` 原先会**改写**它所审计的日志（把 `"history_half_life_hours": NaN` 就地替换为 `null`），使"无 NaN"检查自我实现且计数不可复现；已改为只计数不写回，字段更名为 `inapplicable_half_life_markers_counted_not_rewritten`。副作用需记录：`logs/train_*.log` 已被旧版审计器就地改写过一次（冻结 `ACCEPTANCE.json` 记为 2880 处），因此磁盘上的日志不是运行时原始字节；改版后重跑计数为 0，与当前 runner 直接输出 `null` 一致。日志只作进度记录，所有承重数字来自 `per_subject/*/training_log.csv` 与 `DONE.json`，不受影响。
- 本轮没有合成阳性对照，因此"能检出多大的真实增量"这一灵敏度下界未被独立标定（v0.2 有 `audit_topic5_history_rnn_synthetic_recoverability_v0_2.py`，可作为下一版模板）。

**覆盖边界（必须与结论一起讲）**：每次发作只读同一连续记录段内、cutoff 之前的事件；31 次发作的前缀事件数 10–6125（中位 310），跨度中位 3.87 h，最后一个事件到 cutoff 的间隔中位 0.48 h，而模型时钟记忆半衰期中位约 2.11 h。所以本轮阴性覆盖的是**发作前数小时之内的事件历史**，跨天、跨记录段的长程网络塑形不在被检验的模型族里。

## 10. 产物

- 主汇总：`results/topic5_history_conditioned_field_refinement_v0_4/HISTORY_CONDITIONED_FIELD_SUMMARY.json`
- 工程验收：`results/topic5_history_conditioned_field_refinement_v0_4/ACCEPTANCE.json`
- 患者表：`results/topic5_history_conditioned_field_refinement_v0_4/history_conditioned_field_patient_metrics.csv`
- 六联图：`results/topic5_history_conditioned_field_refinement_v0_4/figures/history_conditioned_field_refinement_six_panel.{png,pdf}`
- 代表病例：`results/topic5_history_conditioned_field_refinement_v0_4/figures/representative_history_refinement.{png,pdf}`
- 图说明：`results/topic5_history_conditioned_field_refinement_v0_4/figures/README.md`
- 可复现清单：`results/topic5_history_conditioned_field_refinement_v0_4/REPRODUCIBILITY_MANIFEST.json`

实现提交：`c6dc6503`；冻结 spec/plan 提交：`20983fe7`；代码审阅与修复见 §9。

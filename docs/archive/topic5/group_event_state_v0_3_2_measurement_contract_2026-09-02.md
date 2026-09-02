# Group-Event State v0.3.2：测量与评价合同（冻结件）

**日期：** 2026-09-02
**状态：** `V0_3_2_MEASUREMENT_CONTRACT_FROZEN`
**起点：** commit `4c2ed958`（branch `codex/topic5-group-event-state-v0-3`），工作分支 `codex/topic5-group-event-state-v032-eval`
**机器配置：** `config/topic5_group_event_state_v032_eval.json`（本文所有数字以该文件为准；两者冲突时以 JSON 为准并修订本文）
**角色边界：** 本合同只覆盖 Agent 2（measurement / evaluation）。Agent 1 的 `v032_model`（residual state 训练）不在本文修改范围；两者通过 `/data/hfosp_group_event_state_v0_3_2/shared/` 交换冻结产物。

> 本文在任何评价数字产生之前写成并冻结。§4 的 eligibility 阈值、§6 的 shift 集合、§7 的臂定义都不得在看到结果后修改；若必须修改，只能新开 v0.3.3 合同。

## 0. 一句话

所有比较始终只回答一个问题：在同一显式历史 `H(t)` 已经进入模型之后，正确时刻的动态状态 `S(t)` 是否还提供增量——

```
H   vs   H + S_correct   vs   H + S_shifted   vs   H + S_mean
```

不再用 `S` 单独对 `H`；不允许 development-test 上的任何表现反过来决定统计分母。

## 1. 时间分区（cumulative recorded time，不含 gap）

沿用 v0.2/v0.3 的 carry segment 构造（recorded session → 在 seizure onset 处切断 → 剔除 `offset + 3600 s` postictal → 丢弃 < 300 s 段）。在累计**有效记录时间**坐标上固定四个边界：

| 分区名 | 累计有效记录时间 | 用途 |
|---|---|---|
| `base_fit` | 0–60% | grammar 拟合、H 拟合、所有 normalization / PCA / contact vocabulary / group-size prior / calibration bias |
| `inner_val` | 60–70% | grammar epoch 选择、H 的 ridge 选择、adapter epoch 选择 |
| `base_refit` | 0–70% | 用 `inner_val` 已选配置**重拟合** state-free grammar、static calibration 与 H 的回归权重和 dispersion；normalization 常数不重估（仍来自 `base_fit`） |
| `dev_val` | 70–80% | Agent 1 的 state checkpoint 选择；本包只作次要 development validation 报告 |
| `dev_test` | 80–100% | 已在 v0.3.1 被消费，只能作 **development score**；本包的主报告分区 |
| sealed / formal | — | **保持关闭**，本包不读不写 |

约束：

- 所有 target window `[t, t+h)` 必须完整位于同一 segment 且同一分区内（label(t) == label(t+h−ε)）。
- 历史（H、S、控制臂状态）可以从 segment 左端 replay，跨过分区左边界；标签不得跨分区右边界。
- anchor 网格：每 300 s 一个 anchor，segment 起点后 300 s 起算（与 v0.2 一致）。
- horizon：300 / 1800 / 7200 s；**1800 s 为主**，7200 s 只在 §4 判定 coverage 合格的患者上作 secondary，300 s 只能称短程记忆。

## 2. 测量层泄漏审计（必须产出，先于任何模型）

### 2.1 contact vocabulary

固定 hardware montage 不可行：legacy 事件流（哪些事件存在、每个事件的参与触点）由全记录 refine/packing 定义，重建需要重跑检测与打包缓存，本包不做。因此按合同第二选项：**只用 `base_fit` 前缀确定 contact vocabulary**——在 `base_fit` 内参与 ≥ `vocab_min_events` 次事件的触点进入词表；之后分区中出现的词表外触点被标记为 `unseen`，其所在事件不进入 contact-identity likelihood（仍计入 count target）。写入 `nontransductive_support_manifest.json`，并明确 `measurement_layer_nested_contract = "prefix_vocabulary_on_legacy_event_stream"`（部分嵌套，不是完全嵌套）。

### 2.2 审计条目（`detector_provenance_audit.json`）

逐项记录 source / 是否使用未来或全记录 / v0.3.2 处理：detector threshold（legacy `rel_thresh=2` 相对批内中位数、`abs_thresh=2` 相对**全记录**中位数）、template（不适用）、refine/packing（全记录 `count > mean + 1·std` 选道；`chns_threh=0.5` 共激活打包）、contact selection、normalization、group-size prior、calibration bias、event feature normalization、tied-group statistics（固定 10 ms 容差，非数据自适应）、bad-channel support、checkpoint selection。

### 2.3 refractory / 窗口重叠（`detector_refractory_manifest.json`）

检测器 `min_gap=20 ms` 合并、packer 固定 `cut_t` core 长度 → 存在结构性最小事件间隔。逐患者报告：IEI 下限、IEI < core + min_gap 的比例、相邻事件 feature window（core ± 0.25 s ± 0.5 s filtfilt pad）重叠比例、zero-phase filter 的未来支持（core_end + 0.75 s）。规定：任一事件的内容最早在 `core_end + 0.75 s` 后才可进入状态；anchor 上使用的 S 只能来自 feature window 已结束的事件。

### 2.4 有效 exposure（`valid_exposure_manifest.json`）

只有 segment 内时间是有效 exposure；gap、seizure、postictal、session reset 不进入 count/timing target。逐患者列出 segment、分区标签、排除区间及理由、每分区有效秒数。

### 2.5 时变 contact support（`time_varying_contact_support.json`）

逐 segment 报告词表触点的观测参与率、`contact_ok`（波形有效）比例；bad-channel support 只用 `base_fit` 事件推导，之后只作描述。

## 3. 事前 eligibility（只看数据，不看任何模型结果）

`endpoint_eligibility.json` + `patient_learnability_table.csv`。逐患者输出：有效记录小时（总/各分区）、分区内事件数、非重叠 30/120 min block 数、count mean / variance / dispersion（var/mean）、grammar prefix 事件数、positive-K steps、contact entropy、support stability、seizure 数与 seizure cluster 数（cluster 间隔阈值 4 h）。

冻结阈值（见 JSON `eligibility`）：

- **30 min count（primary）**：`base_fit ≥ 24` 个非重叠 30 min block，`inner_val ≥ 4`，`dev_val ≥ 4`，`dev_test ≥ 8`，且 `dev_test` 事件数 ≥ 100。
- **120 min（secondary）**：`base_fit ≥ 8` 个非重叠 120 min block，`inner_val ≥ 2`，`dev_val ≥ 2`，`dev_test ≥ 4`。
- **H2a positive-K / prefix**：词表 ≥ 4 触点；`base_fit` 事件 ≥ 2000；`dev_test` 中 positive-K（K ≥ 2，即至少一个继续步）事件 ≥ 200；support stability（`dev_test` 参与质量落在词表内的比例）≥ 0.99。
- **H2b 数据支持**：只描述（held-out seizure 数、cluster 数），本轮不运行 seizure 模型。

不得因 ridge-edge、模型 loss、solver failure 或效果方向改变 eligibility。

## 4. 统一 H 基线

两个嵌套 baseline，都只用 anchor 之前（同 segment 内）的事件，按真实秒数衰减、segment 起点重置：

- **H_rate**：last IEI（log1p 距上次事件秒数 + 有无上次事件 + 上一个完整 IEI）；5/30/120 min 回看窗事件数与速率（含窗内 covered fraction）；time of day（24 h 与 12 h 正余弦）；session position（log1p 进入 segment 秒数、进入 session 秒数、距记录起点天数——**不使用** segment 终点或剩余时间，因为 segment 终点即 seizure onset，会泄漏未来发作时刻）；coverage（log1p 段内累计事件数）。
- **H_strong**：H_rate + recent extent（group size、tied-group 数的 EWMA）、spatial dispersion（参与 shaft 数、词表覆盖比例、delay span 的 EWMA）、multiband summaries（各可用频带 log 能量与峰时的 EWMA）、participation/repertoire summaries（逐触点参与场两尺度 EWMA、`base_fit` 冻结 PCA repertoire 坐标的 EWMA）。EWMA 尺度 300/1800/7200 s。

标准化：只用 `base_fit` anchor 的均值/方差；validation / test 冻结；不做 test-time recalibration。

30 min count 主模型：**negative-binomial（NB2）ridge GLM**。H 与 H+S 必须共享同一 target、同一 anchors、同一 NB family、同一 dispersion 规则（每臂在拟合行上按 ML 估计 α 后冻结；另报共享 H 的 α 作 sensitivity）、同一 exposure weighting（窗口 exposure 恒等于 h）、同一 block scoring。禁止一边 ridge 一边未校准 Poisson。

## 5. 评价臂（完全配对）

| 臂 | 定义 |
|---|---|
| `H` | H_rate 或 H_strong（主：H_strong） |
| `H+S_correct` | 加入 Agent 1 冻结 state 在正确时刻的轨迹 |
| `H+S_shifted` | 同患者同 session 内整段轨迹 circular shift；预定义 5 个 shift（session anchor 数 × j/6，j=1..5），只保留 |Δt| > horizon + 300 s 的 donor；每个 shift 单独拟合读出，报告逐 shift 与均值；不按结果挑 shift |
| `H+S_mean` | S 替换为 `base_fit` anchor 上的均值向量（静态） |
| `H+random reservoir` | 固定随机权重、无训练的 12 维 leaky bank，由事件 mark 驱动 |
| `H+times-only state` | 12 维 leaky bank，只由事件发生（单位冲击）驱动 |
| `H+linear marked EMA` | 事件 mark 向量的线性 EMA（300/1800/7200 s），无学习 |

规则：所有有限 test score 保留；ridge-grid edge 只标 flag；solver failure 单独报告；test 不重拟合 intercept / dispersion / regularization；不允许 post-hoc audit 删 seed 或患者；推断以 patient / block 为单位（block = 同 segment 内连续 `max(h, 1800 s)` 时间箱），重叠 anchor 不作独立样本；seed 先在患者内合并。

## 6. H2a frozen grammar probe

三层冻结：

1. **static grammar calibration**：`base_fit` 经验先验（每步 K 分布 + 触点频率），无网络；
2. **state-free grammar**：product-form tied-group grammar（沿用 v0.3 `FrozenContactGrammar`，legacy 只提供架构超参数、不加载权重），`base_fit` 拟合、`inner_val` 选 epoch、`base_refit` 重拟合后冻结；
3. **H-adapted grammar**：在冻结 grammar 上训练低秩 gated adapter，条件输入 = H(t_e⁻)。

拆开报告：continue vs observed STOP；positive size | continue；contact identity | K（**主要 H2a endpoint**：`subset_step_log_prob`，条件于 K、prefix、implanted support）；same-prefix later continuation（首个 tied group 与 `base_fit` 中出现 ≥ 20 次的 prefix 字典匹配的事件，其第 2 步起的继续项）。

对比臂（adapter 输入不同，其余完全相同）：grammar+H、grammar+H+S_correct、grammar+H+S_shifted（5 个 shift）、grammar+H+S_mean、grammar+H+random_state。adapter 在 `base_fit` 训练、`inner_val` 选 epoch、冻结后在 `dev_val` / `dev_test` 评分。

解释规则：count-trained state 若迁移到 contact identity，才支持“共享 interictal predictive state”；只改善 count → “residual event-burden state”；只改善 STOP/size → “residual observed-extent state”。

## 7. Agent 1 交接文件

读取 `/data/hfosp_group_event_state_v0_3_2/shared/frozen_state_registry.json`。本包期待的最小 schema 写在 `shared/frozen_state_registry.expected_schema.json`：逐患者逐 seed 给出 `anchor_time (A,)` 与 `anchor_state (A,D)`（open-loop、不读未来事件），可选 `event_pre_state (N,D)`；缺 `event_pre_state` 时 grammar probe 用最近一个 anchor 的 state（≤ 300 s 陈旧）并标记 `anchor_held_state`。anchor 按绝对时间对齐（容差 1e-3 s），缺失 anchor 对**所有**臂同时剔除以保持配对。registry 未到时先完成测量、eligibility、H、grammar 与不依赖 S 的控制臂；定时轮询，不重复训练 state。

## 8. 交付物

机器产物（`/data/hfosp_group_event_state_v0_3_2/{measurement,evaluation,shared}/`，镜像到 `results/group_event_state/v0_3_2/`）：`nontransductive_support_manifest.json`、`detector_provenance_audit.json`、`detector_refractory_manifest.json`、`valid_exposure_manifest.json`、`time_varying_contact_support.json`、`endpoint_eligibility.json`、`patient_learnability_table.csv`、`history_baseline_registry.json`、`grammar_headroom_report.json`、`paired_scoring_audit.json`、`v0_3_2_h1_h2a_summary.json`、`task_manifest.json`、`STATUS.json`。

图：只更新 H1 与 H2a 两张承重图，复用 `group_event_state_core_evidence_v2` payload 接口与现有 producer 的 panel 函数，输出到 `results/group_event_state/v0_3_2/core_evidence/`；仍是 candidate framework，不登记 Fig1–Fig4。

报告：`group_event_state_v0_3_2_measurement_plain_2026-09-02.md`、`group_event_state_v0_3_2_measurement_technical_2026-09-02.md`。

## 9. 完成条件

非 transductive support 与 eligibility 已冻结；所有 eligible 臂均能输出 score；无 test-time fit；无结果驱动的 NA / 删 seed；H 与 H+S 同分布同 anchors；frozen state 到达后完成 H1/H2a 配对评价；图与报告与机器 JSON 一致；targeted tests 通过；sealed partition、H2b、H3 不运行；提交并推送分支，不合并 main。

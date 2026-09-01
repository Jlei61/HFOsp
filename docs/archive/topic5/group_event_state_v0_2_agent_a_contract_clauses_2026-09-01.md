# Agent A — A0 合同条款逐条核验表（CLAUDE.md §6 仪式）

在写任何函数体之前枚举。每条给出：来源、实现位置、回归测试。
来源缩写：`CC`=`group_event_state_v0_2_common_contract_2026-09-01.md`，
`EI`=`..._engineering_invariants_...`，`SP`=`..._h1_h2a_spec_plan_...`，
`DC`=`group_event_state_v0_1_data_contract_2026-08-31.md`。

| # | 条款 | 来源 | 实现 | 测试 |
|---|---|---|---|---|
| C1 | TRAIN/inner-val/development-test 按**累计 recorded physical time** chronological 切；不按事件数；target 不跨 split | CC §7.1 | `v02/timeline.py::physical_time_split` | `test_split_is_by_recorded_time_not_event_count` |
| C2 | 状态不跨未记录 gap / 记录段边界传播；batch 并行维=不同 session；同 session chunk 严格有序、只 detach 不 reset | CC §7.2–7.3, EI §3 | `v02/timeline.py::carry_segments`, `v02/session_trainer.py` | `test_state_never_crosses_a_segment_boundary`, `test_single_pass_equals_chunked_carry` |
| C3 | exposure/target window 不跨 seizure onset；ictal 事件已剔除；发作后不静默桥接，新 segment 从 offset+60 min 起 | CC §7.4–7.6, EI §3 | `v02/timeline.py::carry_segments` | `test_segments_break_at_seizure_and_skip_postictal` |
| C4 | 每 5 min 固定物理 anchor；state 按真实 `dt` 从最后事件传播到 grid；每段真实时间只按网格贡献 | CC §5.2, SP §2 | `v02/timeline.py::anchor_grid` | `test_anchor_grid_is_uniform_in_time_not_in_events` |
| C5 | future target 用 anchor index + cumulative sums + prefix counts + sparse participation；**禁止**物化 事件×horizon×触点 dense 张量 | CC §8, SP §2 | `v02/targets.py::FutureTargetBuilder` | `test_targets_match_bruteforce_and_allocate_no_dense_tensor` |
| C6 | conditional mark 至少含 participation field / size-STOP / 连续 event-embedding 分布 / multiband field；cluster 只作可解释 secondary | SP §A2, CC §8 | `v02/marks.py`, `v02/targets.py` | `test_conditional_mark_covers_all_four_families` |
| C7 | repertoire、归一化、超参、checkpoint 选择只用 TRAIN/inner-validation | EI §1.3 | `v02/marks.py::fit_mark_embedding`(train_slice 必填) | `test_mark_embedding_refuses_non_train_fit` |
| C8 | causal prefix：anchor 只读该时刻及以前 | EI §1.1 | `v02/targets.py`, `v02/baseline.py` | `test_baseline_features_are_causal` |
| C9 | baseline 与 recurrent producer 使用同一 anchor/target/mask/normalization/评分代码 | SP §3 | `v02/scoring.py` 单一入口 | `test_all_arms_score_through_one_function` |
| C10 | 承重 null = 同患者同 session 的 block circular shift，平移量**严格大于** target horizon | CC §6 | `v02/readout.py::block_circular_shift` | `test_block_shift_exceeds_horizon_and_stays_in_session` |
| C11 | 长窗资格用与最终 estimator 相同的 coverage-segment 逻辑；滑窗数≠独立窗口数 | EI §2 | `v02/timeline.py::effective_independent_windows` | `test_independent_windows_is_time_over_horizon_not_anchor_count` |
| C12 | 绝对时刻用 float64 / 整数采样点，禁止远历元 float32 | EI §3 | 全部 `t_abs` 走 float64 | `test_absolute_times_stay_float64` |
| C13 | fixed jump 不得饱和成免费截距；intercept/常数漂移零真值进单元测试；ridge 按 Gram 尺度正规化；远坏于 intercept baseline 的拟合标为不可估计 | EI §2 | `v02/readout.py` | `test_constant_drift_truth_yields_zero_gain`, `test_ridge_is_gram_normalised` |
| C14 | 先写临时文件、完整校验后原子 rename；manifest 只在所有必需文件存在后更新；写 hash | EI §4 | `v02/registry.py::atomic_write_json` | `test_atomic_write_leaves_no_partial_file` |
| C15 | registry 每 producer 原子条目；缺失报 `not_available`，禁 silent fallback | CC §10 | `v02/registry.py` | `test_missing_producer_reports_not_available` |

## 三处必须向用户明说的判断（不静默处理）

1. **v0.1 的 `index.json::split_bounds_on_interictal_index` 是事件数切分，v0.2 明确禁止。**
   新代码不给 `split=None` 的默认回退路径（§6 "default=None silently restores the buggy path"）。
   `SubjectSequence.split_slice` 在 v0.2 路径上一次都不调用。

2. **v0.1 的 `SubjectSequence.new_session` 只在 session 变化处置位，不在发作处断开。**
   直接复用它会静默把发作前后接成一条状态链。v0.2 用 `carry_segments()` 重建，
   并且**所有** consumer（trainer / anchor grid / target builder / block shift）都改用它
   （CLAUDE.md §5 "apply safety fixes end-to-end"）。

3. **60 min postictal 排除的适用范围是我的判断。** `CC §7.5` 那一条以 "H2b 只读取
   seizure 前 trajectory" 开头，但 "发作后不静默桥接：从 seizure offset 后 60 min 起新 segment"
   是一般性措辞且位于面向三线的 §7。我按**一般规则**执行（对 A 线也排除 postictal 60 min），
   并把长度做成参数以便敏感性。若用户认为 A 线不应排除 postictal，只需改一个参数重跑。

## 预注册的队列（在任何模型结果之前固定）

- **A1 smoke（3 位固定长患者）**：按"总 recorded 覆盖小时数最长"选 →
  `epilepsiae_916` (424.3 h) / `epilepsiae_253` (260.1 h) / `epilepsiae_1073` (224.0 h)。
  规则与任何模型输出无关，且直接对准长 horizon 的需求。
- **A2（原中期 8 位）**：`epilepsiae_1073 / 1077 / 1096 / 1125 / 1146 / 253 / 384 / 548`
  （v0.1 技术报告 §8 已完成 5 臂×3 seeds 的那 8 位）。
- **A3（全部 development 可训练患者）**：已 consolidate 的全部 **27** 位。
  可训练判据（先于结果）：TRAIN/VAL/TEST 三段在 Δ=5 min 上各至少 1 个合格 anchor。
  覆盖短的患者只在长 horizon 上缺 anchor，记 `insufficient_coverage`，**不记阴性**。
- **A4**：承重配置 = `P_slow` 主配置，预先指定，补到 5 seeds；不看结果再选。

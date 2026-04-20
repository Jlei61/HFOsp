# PR-5-A Novel-Template Falsification Gate 中间报告 (2026-04-20)

> 性质：archive / PR-5-A gate 跑数与判定中间报告
> 触发：`pr5_template_recruitment_plan_2026-04-20.md` §3 写定 gate 合同后，按 TDD 红→绿，全 cohort 复跑 `--pr5-gate` 完成
> 上游入口：`docs/topic1_within_event_dynamics.md` §7.8 / §10
> 配套规则：`.cursor/rules/topic1-within-event-dynamics.mdc` PR Status 表
> 复跑命令：`python scripts/run_interictal_propagation.py --pr5-gate`

---

## 1. 一句话结论

**Gate PASS。** 主配置与辅助配置在 r / e / gap 三个维度上、pre vs baseline 与 post vs baseline 两组对比上**全部满足**事先写死的 PR-5 计划 §3.5 阈值，无一边界突破。最终口径必须收敛到一句老实话：在 **PR-5-A gate-eligible retained cohort**（main `n=23`, auxiliary `n=22`）中，peri-ictal 事件**不能拒绝**"仍属于全局稳定模板库 in-distribution 样本"的零假设；按预设的 `r/e/gap` diagnostics，**未观察到支持** `H_OOD`（peri-ictal 来自全 cohort 训练 dominate 的"新模板族"）或 `H_assignment_drift`（归属置信度系统性下降）的 cohort-level evidence。**PR-5-B 主分析 (`--pr5-recruitment`) 解锁。**

---

## 2. 数据合同与执行参数

- 模板来源：每 subject 的 `T_global = build_cluster_templates(v_ranks, v_bools, labels=adaptive_labels, n_clusters=stable_k)`，与 PR-4D 完全一致；**未**对 peri-ictal 子集重新聚类。
- 窗口来源：实现上没有直接读取 `pr4c_seizure_proximity*.json`，而是在 `_run_pr5_gate` 内部用 `_build_seizure_proximity_windows` 在内存里重建（参数与 PR-4C 完全一致）。这是一次**有意识的实现偏离**：PR-4C JSON 没有持久化 `state_event_indices`，若再补一份 reload 逻辑，反而会制造第二份 ownership 口径。当前选择是复用 PR-4C 已验收的同一 helper，保持"单一真实来源"。
- 事件向量：`v_rel = _compute_relative_lag_matrix(lag_raw[:, valid_event_indices], v_bools)`，`min_participating_l3 = 5`，`min_shared_channels = 3`，与 PR-4C / PR-5 计划完全一致。
- 配置：`SEIZURE_PROXIMITY_CONFIGS["main"]` + `SEIZURE_PROXIMITY_CONFIGS["auxiliary"]` 同步跑。
- per-state 入门门槛：`min_state_events_for_gate = 30`（pool across usable seizures）；不达标 subject 标 ineligible，cohort n 缩水如实报告。

---

## 3. cohort 概况

| 配置 | 总 subject | gate-eligible | ineligible | ineligible 主因 |
|---|---|---|---|---|
| main | 26 (有 PR-4C usable) | **23** | 3 | `epilepsiae/818`, `yuquan/huanghanwen`, `yuquan/zhangjinhan`：per-state event 数 < 30 |
| auxiliary | 26 | **22** | 4 | 同上 + `yuquan/chenziyang`（auxiliary 窗口下 pre 段事件不足） |

ineligible 列表与 PR-5 计划 §7 风险条目"预计 PR-5 gate PASS 后的可分析 cohort ≈ 20 subjects"基本吻合（实际 22–23，略好于预期）。无 subject 因 `no_gate_eligible_events` 之外的原因被剔除。

---

## 4. cohort-level gate 数值

cohort inferential 层按 PR-5 计划 §3.4 同时报告 **sign test + Wilcoxon**；gate 判定仍只由 §3.5 写死阈值决定，**未事后调整**：

- `r` 漂移：cohort `|median Δ_r| ≤ 0.05` 且 paired Wilcoxon p ≥ 0.05
- `e` 漂移：cohort `|median Δ_e / median(e_baseline)| ≤ 0.10` 且 paired Wilcoxon p ≥ 0.05
- `gap` 漂移：不出现"peri-ictal 显著低于 baseline"（`p < 0.05` 且 median delta < 0）→ FAIL；其余 PASS

### 4.1 主配置 (`main`)

`median(e_baseline) across subjects = 4.3220`

| 对比 | 维度 | n | median Δ | wilcoxon p | 阈值 | 通过 |
|---|---|---|---|---|---|---|
| pre vs baseline | r | 23 | -0.0088 | 0.243 | \|Δ\| ≤ 0.05, p ≥ 0.05 | ✅ |
| pre vs baseline | e | 23 | +0.0729 (rel = 0.0169) | 0.273 | rel ≤ 0.10, p ≥ 0.05 | ✅ |
| pre vs baseline | gap | 23 | +0.0129 | 0.615 | 不显著向下 | ✅ |
| post vs baseline | r | 23 | -0.0071 | 0.615 | 同上 | ✅ |
| post vs baseline | e | 23 | +0.0157 (rel = 0.0036) | 0.580 | 同上 | ✅ |
| post vs baseline | gap | 23 | +0.1490 | 0.211 | 同上 | ✅ |

补充方向检验（sign test，双侧）：

- `main / pre_vs_baseline`: `r` 7+/12- (`p=0.359`), `e` 15+/8- (`p=0.210`), `gap` 12+/10- (`p=0.832`)
- `main / post_vs_baseline`: `r` 6+/13- (`p=0.167`), `e` 12+/11- (`p=1.000`), `gap` 14+/9- (`p=0.405`)

`config_pass = True`。

### 4.2 辅助配置 (`auxiliary`)

`median(e_baseline) across subjects = 4.4237`

| 对比 | 维度 | n | median Δ | wilcoxon p | 阈值 | 通过 |
|---|---|---|---|---|---|---|
| pre vs baseline | r | 22 | +0.0000 | 0.836 | \|Δ\| ≤ 0.05, p ≥ 0.05 | ✅ |
| pre vs baseline | e | 22 | +0.0116 (rel = 0.0026) | 0.799 | rel ≤ 0.10, p ≥ 0.05 | ✅ |
| pre vs baseline | gap | 22 | +0.0245 | 0.406 | 不显著向下 | ✅ |
| post vs baseline | r | 22 | +0.0000 | 0.711 | 同上 | ✅ |
| post vs baseline | e | 22 | -0.0294 (rel = 0.0066) | 0.775 | 同上 | ✅ |
| post vs baseline | gap | 22 | +0.1615 | 0.425 | 同上 | ✅ |

补充方向检验（sign test，双侧）：

- `auxiliary / pre_vs_baseline`: `r` 9+/7- (`p=0.804`), `e` 12+/10- (`p=0.832`), `gap` 11+/11- (`p=1.000`)
- `auxiliary / post_vs_baseline`: `r` 9+/9- (`p=1.000`), `e` 10+/12- (`p=0.832`), `gap` 13+/9- (`p=0.523`)

`config_pass = True`。

### 4.3 总判定

`overall_pass = True`（main_pass ∧ aux_pass）。无任何边界条件踩到 PR-5 计划 §3.5 的红线。

---

## 5. 解读

- **几何稳定性**：`Δr` 中位数三位有效数级别（0.00–0.07，远低于 0.05 阈值），而且 sign test 方向也接近对半，不构成系统漂移。peri-ictal 事件 best-template Spearman r 与 baseline 不可区分。
- **重建残差稳定性**：相对漂移最大也只有 1.7%（main pre），远低于 10% 阈值；post 相对漂移 < 1%。Wilcoxon 全部 p ≥ 0.27。
- **归属置信度稳定性**：所有 `gap` 漂移方向都是 ≥ 0（peri-ictal 比 baseline gap 略高一点点），与 `H_assignment_drift` 的预测方向相反；无任何 cohort 显示"peri-ictal gap 显著低于 baseline"。
- **跨配置一致性**：main / auxiliary 在符号、量级上一致；不存在"主 PASS / 辅 FAIL"的 sensitivity 矛盾。
- **事件量充裕**：23 个进入主配置的 subject 在 baseline / pre / post 三个 state 都各自有 ≥ 30 事件，`epilepsiae/922` 等 subject 单 state 事件量过万，cohort-level Wilcoxon 没有受样本不足限制（只是当前 effect size 本来就接近零）。

---

## 6. 探索性诊断（不入主结论）

PR-5 计划 §3.6 提到的 `peri_pool_recluster` Hungarian + Spearman 匹配诊断**本次未触发**。理由：gate 主读数全部健康，没有任何 cohort-level 信号提示需要追查 OOD 候选；按 §3.6 "只进 archive，不进主结论" 的合同，无健康信号时跳过即可，避免无意义的 multiple-testing 噪声。如未来 PR-5-B 出现"recruitment shift 明显但解释不通"的反例，可单独立 archive 触发该诊断。

---

## 7. 与 PR-4C / PR-4D 的边界

- 本 gate **不**对 PR-4C 五指标几何 cohort null 结论作任何修订；它只验证"如果有 recruitment shift，那它发生在固定模板库内、而非新模板族"的前提。
- 本 gate **不**触碰 PR-4D `rate × type` 描述层；PR-5-B 才会把 `rate_by_template` 信号在与 gate 同事件池上重算。
- 本 gate 的 ineligible 名单（main: `epilepsiae/818`, `yuquan/huanghanwen`, `yuquan/zhangjinhan`；auxiliary 多一个 `yuquan/chenziyang`）会原封传给 PR-5-B 的 cohort 缩水说明，**不**允许在 PR-5-B 阶段补救。

---

## 8. 实现与产物

- 新增函数（`src/interictal_propagation.py`）：
  - `_per_event_gate_metrics`（私有）：单事件 × 全模板的 r / e / gap
  - `compute_novel_template_gate`：per-subject pool 三个 state 的分布与 deltas
  - `_wilcoxon_safe`（私有）：cohort-level Wilcoxon 包装，安全处理 `n=1`、全零、scipy 退化分支
  - `_evaluate_pr5_gate_thresholds`（私有）：PR-5 计划 §3.5 阈值的 deterministic 判定
  - `summarize_pr5_novel_template_gate`：cohort-level 汇总 + 双配置 overall_pass
- 新增 runner 路径（`scripts/run_interictal_propagation.py`）：
  - `--pr5-gate`：跑 main + auxiliary
  - `--pr5-min-state-events`（默认 30）：与计划 §2.3 fail-fast 合同对齐
  - 内部 `_run_pr5_gate`：在内存里复用 `_build_seizure_proximity_windows`，避免新口径
- 新增测试（`tests/test_interictal_propagation.py`，TDD 红→绿）：
  - `test_pr5_gate_passes_when_states_are_resampled_baseline`
  - `test_pr5_gate_fails_when_post_events_drawn_from_orthogonal_template`
  - `test_pr5_gate_uses_only_l3_eligible_events`
  - `test_pr5_gate_summary_reports_sign_test_and_passes`
  - `test_pr5_gate_summary_fails_when_gap_is_significantly_lower`
  - 全套 36 + 5 = 41 项 pytest 全绿
- 输出落位：
  - `results/interictal_propagation/pr5a_novel_template_gate.json`（cohort + per-subject）
  - `results/interictal_propagation/per_subject/pr5a/<subject>.json`（per-subject 明细）
  - `results/interictal_propagation/pr1_cohort_summary.json` 新增 `novel_template_gate` 字段，与 `seizure_proximity_analysis*` 同级

---

## 9. 下一步

1. 启动 PR-5-B `compute_template_recruitment_shift` 实现（PR-5 计划 §4 + §5.2）；TDD 5 项（计划 §5.3 #4–#8）红→绿。
2. `--pr5-recruitment` runner 路径接入；要求 `pr5a_novel_template_gate.json` 的 `cohort.overall_pass=True`，否则 `SystemExit`（已被本 gate 满足）。
3. 主 + 辅两配置全 cohort 复跑，bootstrap CI。
4. Topic 1 §7.8 / PR Status 表与 `.cursor/rules/topic1-within-event-dynamics.mdc` 在 PR-5-B 完成后再统一一次性升级（本 archive 只刷"PR-5-A 已完成、gate PASS"的状态条目）。

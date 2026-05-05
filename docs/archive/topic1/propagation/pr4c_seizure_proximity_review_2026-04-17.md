# PR-4C 发作邻近分析审阅 (2026-04-17)

> 性质：archive / 历史归档报告
> 触发：用户在拿到 PR-4C 主+辅助配置全量结果后要求严肃审阅，并据此规划 Topic 1 后续方向
> 上游入口：`docs/topic1_within_event_dynamics.md` §3.1c / §5 / §7.2 / §7.6 / §7.7 / §10
> 配套规则：`.cursor/rules/topic1-within-event-dynamics.mdc` PR Status 表

---

## 1. 一句话结论

当前全量结果**不能直接下"固定传播模板没有 seizure proximity"的结论**。
更准确的版本是：**就当前这版 PR-4C 输出而言，固定传播模板的几何结构没有表现出稳健、可复现的发作邻近调制；但实现层面存在三处会系统性朝假阴性方向丢信号的合同问题，因此该阴性还不能作为正式封板结论。** 真正稳健的发作邻近信号在 absolute event rate 层 (`rate_by_template`)，对应"模板招募频率"而非"模板内部几何变形"。

---

## 2. 全量配置与统计单元

- 主配置 (`config_name="main"`)：baseline `(-4, -1) h`，pre_ictal `(-1, -0.25) h`，post_ictal `(0.25, 1) h`
- 辅助配置 (`config_name="auxiliary"`)：baseline `(-2, -0.5) h`，pre_ictal `(-0.5, -1/12) h`，post_ictal `(1/12, 1) h`
- 统计单元：subject
- 共 26 个 subject 有 PR-4C 输出（扣除无 `lagPat` / 无 `seizure_times` 的样本），24 个 subject 在 cohort 上提供成对样本
- 主配置 187 个可用窗口（合同 §4.1 过严，详见后文），辅助配置 245 个

---

## 3. 全量数值结果

### 3.1 主配置 propagation pattern 五个指标 (cohort-level)

| 比较 | 指标 | median delta | wilcoxon p | n_positive / n |
|---|---|---|---|---|
| pre_ictal vs baseline | raw_tau | +0.0027 | 0.864 | 13/24 |
| pre_ictal vs baseline | centered_tau | +0.0024 | 0.099 | 16/23 |
| pre_ictal vs baseline | lag_span | -0.0006 | 0.215 | 10/24 |
| pre_ictal vs baseline | pearson_r | -0.0082 | 0.461 | 9/24 |
| pre_ictal vs baseline | dominant_cluster_fraction | +0.0006 | 0.989 | 12/23 |
| post_ictal vs pre_ictal | raw_tau | +0.0079 | 0.583 | 15/24 |
| post_ictal vs pre_ictal | centered_tau | +0.0000 | 0.659 | 12/22 |
| post_ictal vs pre_ictal | lag_span | +0.0016 | 0.230 | 16/24 |
| post_ictal vs pre_ictal | pearson_r | +0.0108 | 0.916 | 13/24 |
| post_ictal vs pre_ictal | dominant_cluster_fraction | +0.0219 | 0.220 | 16/24 |
| post_ictal vs baseline | raw_tau | -0.0003 | 0.530 | 12/24 |
| post_ictal vs baseline | centered_tau | +0.0006 | 0.510 | 13/24 |
| post_ictal vs baseline | lag_span | +0.0009 | 0.726 | 14/24 |
| post_ictal vs baseline | pearson_r | +0.0107 | 0.638 | 14/24 |
| post_ictal vs baseline | dominant_cluster_fraction | +0.0157 | 0.768 | 14/24 |

**15 项 cohort-level 比较 0/15 显著 (p < 0.05)**。effect size 接近 0，最大 |median delta| = 0.022。

### 3.2 辅助配置 propagation pattern 五个指标 (cohort-level)

| 比较 | 指标 | median delta | wilcoxon p | n_positive / n |
|---|---|---|---|---|
| pre_ictal vs baseline | raw_tau | -0.0167 | **0.0005** | 5/24 |
| pre_ictal vs baseline | centered_tau | +0.0007 | 0.250 | 14/20 |
| pre_ictal vs baseline | lag_span | -0.0002 | 0.598 | 10/24 |
| pre_ictal vs baseline | pearson_r | -0.0036 | 0.560 | 10/24 |
| pre_ictal vs baseline | dominant_cluster_fraction | +0.0117 | 0.298 | 15/24 |
| post_ictal vs pre_ictal | raw_tau | +0.0145 | 0.225 | 17/24 |
| post_ictal vs pre_ictal | centered_tau | -0.0004 | 0.430 | 9/21 |
| post_ictal vs pre_ictal | lag_span | +0.0012 | 0.157 | 14/24 |
| post_ictal vs pre_ictal | pearson_r | +0.0226 | 0.243 | 14/24 |
| post_ictal vs pre_ictal | dominant_cluster_fraction | +0.0165 | 0.485 | 14/24 |
| post_ictal vs baseline | raw_tau | +0.0046 | 0.876 | 13/24 |
| post_ictal vs baseline | centered_tau | -0.0005 | 0.310 | 8/22 |
| post_ictal vs baseline | lag_span | -0.0002 | 0.901 | 11/24 |
| post_ictal vs baseline | pearson_r | -0.0077 | 0.857 | 11/24 |
| post_ictal vs baseline | dominant_cluster_fraction | +0.0258 | **0.0135** | 19/23 |

15 项 cohort-level 比较 2/15 名义显著，但：

- `pre_ictal vs baseline raw_tau`：主配置 +0.003 / 辅助配置 -0.017，**方向相反**，sensitivity check 失败
- `post_ictal vs baseline dominant_cluster_fraction`：主配置 p=0.768 完全 null，辅助配置才出 p=0.0135，**跨配置不一致**
- 按 Bonferroni 校正 15 次比较 (alpha=0.0033)，辅助配置 raw_tau 仅边缘通过；且方向跨配置反转，不能作为稳健信号

### 3.3 rate_by_template (描述层)

主配置：

- `pre_ictal vs baseline` dominant_template_rate Δ_med = +9.3 events/h, p=0.042
- `post_ictal vs baseline` dominant_template_rate Δ_med = **+35.7 events/h, p=0.00085**

辅助配置：

- `post_ictal vs baseline` dominant_template_rate Δ_med = +22.8 events/h, p=0.019

rate 这一层在主配置和辅助配置都看到 post_ictal 显著升高，**方向一致**。但这是 absolute event rate 调制，不是 propagation pattern 调制；它对应"哪些模板被招募的频率"，不是"模板内部几何变形"。

---

## 4. 三处实现合同问题

### 4.1 严重：window usability gate 过严

`_build_seizure_proximity_windows` 把 window 的 `usable` 判定为"baseline + pre_ictal + post_ictal 三段都必须非空"：

```
src/interictal_propagation.py L2597-L2608
usable = all(state_counts[name] > 0 for name in state_names)
...
if usable:
    usable_windows.append(usable_window)
```

随后 `compute_seizure_proximity_coupling` 对非 usable window 直接 `continue`：

```
src/interictal_propagation.py L3209-L3215
if not window["usable"]:
    seizure_windows.append(window_out)
    continue
```

科学问题：一个 window 即便只缺 post_ictal 段，本可以参与 `pre_ictal vs baseline` 比较，现在被整体丢弃。**这是纯粹降功效，且方向单一：更容易把真实信号洗成阴性。**

修复方向：每个 pair 独立判定 usability。`pre_ictal vs baseline` 只要这两个状态都有事件就纳入，不管 post_ictal 是否为空。

### 4.2 严重：nearest seizure 归属规则错位

`_build_seizure_proximity_windows` 现在的归属逻辑是"先选最近 seizure，再判断该事件是否落在该 seizure 的窗口里"：

```
src/interictal_propagation.py L2573-L2584
sz_id = int(np.argmin(np.abs(sz_times - float(event_time))))
delta_h = (float(event_time) - float(sz_times[sz_id])) / 3600.0
state = _state_for(delta_h)
if state is None:
    continue
windows[sz_id]["state_event_indices"][state].append(int(ev))
```

科学问题：在密集发作场景下，一个事件对最近 seizure 落不进任何窗口（比如距离最近 seizure +1.2h，超出 post_ictal），但对次近 seizure 可能正好落在 baseline 或 pre_ictal 范围内，按当前实现这类事件会被直接丢弃。**这种丢弃会系统性削弱晚 baseline / 早 pre_ictal 事件，正是最关键的边界事件。**

修复方向：先枚举所有合法 `(seizure, state)` 候选，再在候选中按"离该 seizure 中心最近"或"按状态优先级"决定归属，而不是先做最近邻硬切。

### 4.3 中等：rate_by_template 没用 gap-aware exposure

Topic 1 文档明确要求 PR-4C 次级描述层应复用 PR-4D 的 gap-aware `rate×type`，暴露时间必须按真实 `coverage_ranges` 算。但当前 `_rate_by_template_for_window` 直接拿固定窗宽做分母：

```
src/interictal_propagation.py L2639-L2654
for state, duration in state_durations_hours.items():
    ...
    dur = float(max(duration, 1e-9))
    rate = counts / dur
    out[state] = {
        "duration_hours": float(duration),
        ...
        "rate_total_per_hour": float(total / dur),
```

科学问题：发作邻近窗口里如果有停录 / 缺口，固定分母会让 rate 偏低，向"无 rate 调制"方向漂。这一层目前已经看到方向一致的显著效应，所以这个问题暂时没掩盖结论；但要拿这层做正式 inferential 论述前必须修。

修复方向：`_rate_by_template_for_window` 接收 `coverage_ranges` 列表，按真实 covered seconds 算分母，与 PR-4D 对齐。

---

## 5. 这个结果能不能下"无 seizure proximity"结论

**不能。** 更准确的版本是：

- 就当前这版 PR-4C 输出而言，固定传播模板的几何结构（`raw_tau / centered_tau / lag_span / pearson_r / dominant_cluster_fraction`）没有表现出稳健、可复现的发作邻近调制
- 但因 §4.1 / §4.2 两个合同问题朝同一方向系统性丢信号，该阴性不能作为最终封板结论
- 真正稳健的发作邻近信号在 absolute event rate 层（`rate_by_template`），对应"模板招募频率"而非"模板内部几何变形"
- 这与 Topic 1 的核心叙事自洽：刻板模板反映的是结构性病理网络，发作邻近时网络的招募强度（rate）变化，而模板本身的几何结构不必随之变形

---

## 6. 推荐下一步

### 6.1 P0：修 PR-4C 实现，复跑（硬性前置）

1. window usability 改为 pair-wise 判定（fix §4.1）
2. 事件归属改为"先枚举合法候选，再决定归属"（fix §4.2）
3. `rate_by_template` 接入 gap-aware coverage（fix §4.3）

修完跑两个配置全量后再判定阴性是否成立。

### 6.2 P1：Topic 1 × Topic 3 桥接 — 稳定模板的空间锚定

详见 `docs/topic3_spatial_soz_modulation.md` "可选方向：稳定传播模板的空间锚定（Topic 1 × Topic 3）"。

核心问题：30/30 subject 都有 stable adaptive solutions、`23/30 strong` 跨时间复现，这套刻板模板的 source/sink 通道是否在解剖上锚定到 SOZ / lesion / extra-focal？这是把 Topic 1 的"刻板"直接挂到 Topic 3 的"病理空间"，不依赖发作邻近窗口的功效。

### 6.3 P2：Topic 1 × Topic 2 桥接 — 模板招募频率，而非模板几何

详见 `docs/topic1_within_event_dynamics.md` §7.6。

核心问题：本审阅结果中，rate_by_template 在 main + auxiliary 都显示 post_ictal 主导模板事件率显著升高。下一步应该把"哪个模板在发作邻近被招募更多"作为正式问题，而不是继续找模板内部几何形变。这条路与 Topic 2 的 seizure-triggered rate 框架天然衔接。

### 6.4 P3：高置信子集的窄 exploratory 分支

详见 `docs/topic1_within_event_dynamics.md` §7.7。

核心问题：8/30 高置信 subject (dominant_r > 0.7) 和 9 个 forward/reverse 复现 subject 是 Topic 1 中信号最强的子集。在这部分 subject 上做条件性的发作邻近分析（修复 §4 后），不强求 cohort-level 普适结论。

---

## 7. 数据与代码入口

- 主配置 cohort 输出：`results/interictal_propagation/pr1_cohort_summary.json` → `seizure_proximity_analysis`
- 辅助配置 cohort 输出：`results/interictal_propagation/pr1_cohort_summary.json` → `seizure_proximity_analysis_auxiliary`
- 主配置 per-seizure：`results/interictal_propagation/pr4c_seizure_proximity.json`
- 辅助配置 per-seizure：`results/interictal_propagation/pr4c_seizure_proximity_auxiliary.json`
- 主代码：`src/interictal_propagation.py`
  - `compute_seizure_proximity_coupling`（含 §4.1）
  - `_build_seizure_proximity_windows`（含 §4.1 / §4.2）
  - `_rate_by_template_for_window`（含 §4.3）
  - `_summarize_seizure_proximity` / `summarize_propagation_cohort`
  - 命名常量：`SEIZURE_PROXIMITY_CONFIGS`
- 运行脚本：`scripts/run_interictal_propagation.py --pr4c` / `--pr4c-auxiliary`
- 单元测试：`tests/test_interictal_propagation.py`（33 passed）
- 窗口 sweep 工具（一次性数据驱动决策）：`scripts/pr4c_window_sweep.py`、`scripts/pr4c_window_sweep_report.py`

---

## 8. 与 Topic 1 主文档的关系

本归档是对单次审阅事件的完整记录。Topic 1 主文档 `docs/topic1_within_event_dynamics.md` 只保留：

- §3.1c：在 PR-4C 段尾指向本归档
- §5：在风险点列出"PR-4C 当前实现存在合同问题，阴性结论待修复后复核"
- §7.2：PR-4C 章节末尾指向本归档
- §7.6 / §7.7：把本归档 §6.3 / §6.4 落成 Topic 1 后续可选方向
- §10：在历史文档索引中收录本归档

主文档不重复本归档的全量数值表与代码 line 引用；具体数字与代码定位以本归档为准。

---

## 9. 2026-04-19 更新：P0 修复完成，全量复跑结论刷新

### 9.1 三处合同问题已落地（TDD 红绿验证）

| 合同 | 红灯测试 | 修复后实现 |
|---|---|---|
| §4.1 window usability 改 pair-wise | `test_seizure_proximity_window_missing_post_still_supports_pre_vs_baseline` | `_PR4C_PAIRS` + `pair_usability` 字典；`compute_seizure_proximity_coupling` 对每个 pair 独立判定 |
| §4.2 事件归属枚举候选再 tie-break | `test_seizure_proximity_assigns_event_to_non_nearest_seizure_when_nearest_has_no_state` | `_build_seizure_proximity_windows` 内层换成"枚举所有合法 (sz, state)，按 \|Δh\| 最小决胜" |
| §4.3 rate denominator 接 gap-aware coverage | `test_rate_by_template_uses_gap_aware_coverage_when_available` | 新 helper `_intersect_seconds`；`compute_seizure_proximity_coupling` 接 `coverage_ranges`；runner 透传 `loaded["block_time_ranges"]` |

测试套件 36 passed，新增三项失败合同先行钉死。Epilepsiae 548 单 subject smoke 验证：`coverage_aware_rate=True`、`n_seizures_pair_usable={pre_vs_baseline:25, post_vs_pre:28, post_vs_baseline:28}`、首个 window 实测 covered 0.63/0.62/0.73 h（名义 3.0/0.75/0.75 h），印证 Epilepsiae 数据的 gap 严重。

### 9.2 全量复跑统计单元变化

| 维度 | 修复前主配置 | 修复后主配置 | 修复后辅助配置 |
|---|---|---|---|
| `n_subjects_with_usable_windows` | 24 | 25 | 26 |
| `n_usable_windows_total` | 187 | **360**（+92%） | 370 |
| Wilcoxon 平均样本数 / 项 | ~24 | ~24 | ~24 |

n_usable_windows_total 翻倍主要来自 §4.1（缺 post_ictal 段也参与 `pre_vs_baseline`）+ §4.2（救回边界 events）。Wilcoxon n 没有显著变化，因为统计单元仍是 subject。

### 9.3 修复后 propagation pattern 五指标 cohort 数值

主配置（n_w 多在 23-25）：

| 比较 | 指标 | median delta | wilcoxon p | n_b>a / n_w |
|---|---|---|---|---|
| pre_ictal vs baseline | raw_tau | +0.0109 | 0.264 | 15/25 |
| pre_ictal vs baseline | centered_tau | +0.0020 | 0.121 | 15/24 |
| pre_ictal vs baseline | lag_span | +0.0000 | 0.853 | 13/25 |
| pre_ictal vs baseline | pearson_r | +0.0177 | 0.303 | 15/24 |
| pre_ictal vs baseline | dominant_cluster_fraction | +0.0005 | 0.895 | 13/25 |
| post_ictal vs pre_ictal | raw_tau | -0.0008 | 0.855 | 12/24 |
| post_ictal vs pre_ictal | centered_tau | +0.0005 | 0.643 | 12/23 |
| post_ictal vs pre_ictal | lag_span | +0.0011 | 0.229 | 16/24 |
| post_ictal vs pre_ictal | pearson_r | -0.0010 | 0.877 | 12/24 |
| post_ictal vs pre_ictal | dominant_cluster_fraction | +0.0206 | **0.020** | 17/23 |
| post_ictal vs baseline | raw_tau | -0.0036 | 0.846 | 10/23 |
| post_ictal vs baseline | centered_tau | +0.0020 | 0.264 | 15/24 |
| post_ictal vs baseline | lag_span | -0.0000 | 1.000 | 11/24 |
| post_ictal vs baseline | pearson_r | +0.0185 | 0.252 | 16/24 |
| post_ictal vs baseline | dominant_cluster_fraction | +0.0132 | 0.252 | 15/24 |

辅助配置：

| 比较 | 指标 | median delta | wilcoxon p | n_b>a / n_w |
|---|---|---|---|---|
| pre_ictal vs baseline | raw_tau | -0.0037 | 0.141 | 10/25 |
| pre_ictal vs baseline | centered_tau | +0.0000 | 0.893 | 12/23 |
| pre_ictal vs baseline | lag_span | -0.0002 | 0.916 | 12/25 |
| pre_ictal vs baseline | pearson_r | +0.0186 | 0.229 | 14/24 |
| pre_ictal vs baseline | dominant_cluster_fraction | +0.0027 | 0.491 | 14/25 |
| post_ictal vs pre_ictal | raw_tau | +0.0108 | 0.160 | 16/24 |
| post_ictal vs pre_ictal | centered_tau | -0.0005 | 0.539 | 8/21 |
| post_ictal vs pre_ictal | lag_span | +0.0006 | 0.508 | 13/25 |
| post_ictal vs pre_ictal | pearson_r | +0.0356 | 0.197 | 14/24 |
| post_ictal vs pre_ictal | dominant_cluster_fraction | +0.0162 | 0.178 | 16/24 |
| post_ictal vs baseline | raw_tau | -0.0059 | 0.390 | 12/24 |
| post_ictal vs baseline | centered_tau | -0.0003 | 0.656 | 12/22 |
| post_ictal vs baseline | lag_span | +0.0006 | 0.768 | 13/24 |
| post_ictal vs baseline | pearson_r | -0.0005 | 0.684 | 12/24 |
| post_ictal vs baseline | dominant_cluster_fraction | +0.0260 | **0.002** | 20/24 |

跨配置一致性：
- `pre_ictal vs baseline raw_tau`：旧 aux 名义显著 (p=0.0005)、修复后 aux p=0.141 → **该信号消失**，与 Fix B 改变 baseline event composition 一致；说明旧的 raw_tau 信号是合同 bug 造的，不是真实生理调制
- `dominant_cluster_fraction`：main 在 `post_vs_pre` 名义显著 (p=0.020)、aux 在 `post_vs_baseline` 名义显著 (p=0.002)，**配置不一致**；按 Bonferroni（30 项比较，alpha=0.0017）只有 aux post_vs_baseline 边缘通过，跨配置一致性差，仍属 noise-floor 边缘
- 其余 14 个传播形状指标在两个配置下 p 全部 >= 0.12，方向跨配置混乱 → **传播模式几何无稳健发作邻近调制**这一阴性可以正式封板

### 9.4 修复后 rate_by_template 数值

主配置（gap-aware）：

| state | median total rate (events/h) | median dominant template rate |
|---|---|---|
| baseline (-4..-1 h) | **72.2** | 36.4 |
| pre_ictal (-1..-0.25 h) | **176.8** | 82.3 |
| post_ictal (0.25..1 h) | **170.5** | 107.2 |

辅助配置：

| state | median total rate (events/h) | median dominant template rate |
|---|---|---|
| baseline (-2..-0.5 h) | 109.4 | 52.2 |
| pre_ictal (-0.5..-1/12 h) | 155.7 | 85.8 |
| post_ictal (1/12..1 h) | 156.7 | 84.4 |

修复前主配置 baseline 误报为 ~152/h，修复后 72/h（gap-aware 让分母从名义 3 h 缩为实测 ~1.4 h）。peri-ictal/baseline 倍数：主 ~2.4×、辅 ~1.4×。倍数差异符合预期：辅助配置 baseline 段更靠近 seizure，本身已经处于上升期。

cohort 比较：

| 比较 | 主配置 dominant rate Δ_med | 主 p | 辅助 dominant rate Δ_med | 辅 p |
|---|---|---|---|---|
| pre_ictal vs baseline | +4.8 events/h | 0.300 | +9.9 events/h | 0.099 |
| post_ictal vs pre_ictal | +11.3 events/h | **0.034** | -1.0 events/h | 0.635 |
| post_ictal vs baseline | **+39.9 events/h** | **0.0009** | **+22.7 events/h** | **0.0067** |

`post_ictal vs baseline` 主导模板事件率升高在两个配置都通过 Wilcoxon (p < 0.01)，方向一致，**这是当前 PR-4C 唯一稳健的发作邻近信号**。

### 9.5 结论刷新

1. **传播模式几何（template stereotype）正式封板为阴性。** 修复 §4.1/§4.2/§4.3 后，n_usable_windows 接近翻倍，传播形状五指标在主+辅助两配置下 cohort Wilcoxon 仍然 null（仅 dominant_cluster_fraction 出现 1 项 main 名义显著、1 项 aux 名义显著，跨配置不一致）。这与"刻板模板反映结构性病理网络、网络几何不随发作邻近变形"的核心叙事一致
2. **rate-by-template 升级为正式发作邻近信号。** post_ictal vs baseline 的主导模板事件率在主+辅助两配置都通过 Wilcoxon (p < 0.01)，方向一致。这是 Topic 1 × Topic 2 桥接（§7.6 / 本档 §6.3）的实证基础
3. **P0 完成；P1/P2/P3 继续作为后续可选方向。** 阴性结论已 clean，下一步是把信号搬到 Topic 1 × Topic 2（rate 调制）和 Topic 1 × Topic 3（空间锚定），不再继续在 PR-4C 内部追加 propagation geometry 假设

### 9.6 数据与代码入口（更新）

- 主配置 cohort 输出：`results/interictal_propagation/pr1_cohort_summary.json` → `seizure_proximity_analysis`
- 辅助配置 cohort 输出：同文件 → `seizure_proximity_analysis_auxiliary`
- per-seizure：`results/interictal_propagation/pr4c_seizure_proximity{,_auxiliary}.json`，每个 window 新增 `pair_usability`、`state_covered_hours` 字段
- 顶层 `seizure_proximity_coupling*` 新增 `n_seizures_pair_usable`、`coverage_aware_rate` 字段
- 全量复跑日志：`logs/pr4c_full_rerun_v2.log`（main 2h05min，aux 1h54min，`--n-sample 100 --n-seeds 3`）
- 测试入口：`tests/test_interictal_propagation.py`（36 passed），新增三项 §9.1 中的失败合同测试

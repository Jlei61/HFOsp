# PR-7 Step 3 结果：H1 primary cohort 在 short-window opposite-template excess 上 NULL

> 状态：Step 3 deliverable，2026-04-29（v1）
> 范围：H1 primary cohort（fwd/rev reproduced ∩ 5 conds，n=6）的 cohort 级 triple-gate 判读 + N3 robustness 一致性。**H2 negative control（n=17）尚未跑，Step 4 完成后追加**。
> 上游：`docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md`（plan-of-record）
> 数据：`results/interictal_propagation/template_pairing/`

---

## 1. 一句话结论（精确 framing）

在 6 个 forward/reverse-reproduced subject 上，**没有检测到** "T_a 出现后 10s/30s 内 T_b 比同模板更易紧随" 这一固定时间签名（H1 triple-gate NULL，N2 与 N3 双 null 一致）。

**仅否定**：short-window reciprocal coupling（"反向模板短时接力" / bouncing-back 因果版本）这一**特定形式**的 Ping-Pong 假说。

**不**否定：
- PR-6 已建立的 fwd/rev 几何相关性（source/sink swap，n=6 sign-test p=0.031）
- 因果性本身（观测性事件序列 + 当前固定窗口 excess metric 的 null 不能支持"无因果"）
- 慢时间尺度耦合（hours / circadian / seizure-proximity 等被本检验设计性排除）

后续 follow-up（独立 PR）：历史依赖模型 P(next_label / hazard_opposite | recent history + local rate + state)，不再以"short-window opposite-template excess"为唯一 metric。

---

## 2. Cohort 与跑数环境

| 项 | 值 |
|---|---|
| H1 primary cohort | n=6（5 epilepsiae: 1073/139/548/635/958 + 1 yuquan: chenziyang）|
| 入选条件 | endpoint_defined ∩ fwd/rev_reproduced(OR) ∩ n_events≥300 ∩ min_cluster_n≥75 ∩ (n_blocks≥3 OR coverage≥6h) |
| 实际 N（events used）| 9609 – 193171 |
| 实际 min_cluster_n | 4059 – 79122（远高于 75 阈值）|
| 实际 n_blocks / coverage | 12-231 blocks / 23-222 hours |
| 跑数参数 | n_perm=1000, Δt grid={1,5,10,30,60,300,1800,3600}s, n2_window=30min, seed=0 |
| Nulls 跑齐 | N0 (sanity ceiling) / N1 (sanity mid) / N2 (主 null) / N3 (robustness) |
| N4 状态 | conditional follow-up，未触发，未实现（按 plan §4.1）|
| 总跑数耗时 | ~1h50m（1073 ~50min, 958 ~43min, 其余 5–6min/subject）|

cohort_audit.csv: `results/interictal_propagation/template_pairing/pr7_cohort_audit.csv`
per-subject JSON: `results/interictal_propagation/template_pairing/per_subject/{dataset}_{subject_id}.json`

---

## 3. H1 triple-gate 判读

**主门 = N2 local-window shuffle (30min, 50% overlap, first-covering, 每 block 独立)**

| Gate | 阈值 | 实测（N2）| 结果 |
|---|---|---|---|
| (1) excess(10s) Wilcoxon (greater) | p < 0.05 | **p = 0.844** | ❌ |
| (2) excess(10s) sign test (greater) | p < 0.05 | **p = 0.656**（3 pos / 3 neg）| ❌ |
| (3) excess(30s) cohort median > 0 | > 0 | **median = −0.0148** | ❌ |
| **总判** | 三条全部满足 | | **NULL** |

**Robustness = N3 circular shift label sequence**（保留 burst 时间结构）

| Gate | 实测（N3）| 与 N2 同向？|
|---|---|---|
| Wilcoxon (greater) | p = 0.891 | ✓ |
| sign test (greater) | p = 0.891（2 pos / 4 neg）| ✓ 一致负 |
| median(30s) | −0.0120 | ✓ |
| **总判** | NULL（一致）| ✓ |

**关键稳健性**：N2 与 N3 在 6 个 subject 上 5/6 同号（仅 1073 在 N3 下从 +0.008 翻为 -0.030，绝对值都很小）。两个 null 一致 NULL → **不是 N2 主 null 单一伪影**。

详细 sanity null（N0/N1）：

| Null | median(10s) | sign test pass? | Wilcoxon p |
|---|---|---|---|
| N0 (global, ceiling) | −0.086 | 1/5 pos | 0.953 |
| N1 (block-aware) | −0.029 | 2/5 pos | 0.891 |
| N2 (主 null) | **−0.018** | 3/3 pos | **0.844** |
| N3 (robustness) | −0.023 | 2/4 pos | 0.891 |

四个 null 全部 NULL，且方向一致（median 全负）。N0 ceiling 比 N2 更负（-0.086 vs -0.018）说明：相对于完全打散的 baseline，N2 的局部 burst 结构反而让 same-cluster 比 opposite **稍微更密**（因为 burst 内同 cluster 概率高），但 cohort 上仍未达到统计显著。

---

## 4. 多尺度 excess(Δt) profile

| Δt | N2 cohort median excess | 解读 |
|---|---:|---|
| 1s | +0.030 | packing-proximity 区，subject-level 高方差 |
| 5s | -0.020 | packing edge 区，cohort 几乎归零 |
| **10s** | **−0.018** | **H1 primary**：NULL |
| **30s** | **−0.015** | **required sensitivity**：方向**反向**（< 0）|
| 60s | −0.014 | NULL |
| 5min | −0.005 | 趋零 |
| 30min | 0.000 | 归零（slow-drive control 验证）|
| 1h | 0.000 | 归零 |

**Profile 解读**（plan §3.5）：
- 长尺度（≥30min）正确归零 → **慢漂移共驱不是 confound**（plan §9.3 confound profile A 排除）
- 短尺度（10s/30s）median 负值，方向与"opposite > same"假设相反；不是 packing artifact（packing window ≈ 5s，30s 已远离）
- 1s 处 +0.030 仅是 packing-proximity 诊断信号（plan §3.5），**不**进 H1 PASS 判据

---

## 5. Per-subject 数字

| Subject | N | excess(1s) | excess(5s) | **excess(10s)** | **excess(30s)** | excess(60s) | excess(5min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| epilepsiae/1073 | 193171 | +0.142 | +0.032 | **+0.008** | **−0.004** | −0.007 | −0.007 |
| epilepsiae/139 | 14438 | −0.064 | −0.058 | **−0.048** | **−0.026** | −0.013 | −0.006 |
| epilepsiae/548 | 25282 | −0.144 | −0.212 | **−0.201** | **−0.188** | −0.163 | −0.089 |
| epilepsiae/635 | 13973 | +0.121 | +0.062 | **+0.030** | **+0.015** | +0.006 | +0.001 |
| epilepsiae/958 | 165577 | −0.052 | −0.042 | **−0.044** | **−0.032** | −0.026 | −0.014 |
| yuquan/chenziyang | 9609 | +0.083 | +0.042 | **+0.014** | **+0.006** | +0.003 | −0.001 |

**3 正 / 3 负** —— cohort 完全无方向。

**Outlier subject**：`epilepsiae/548` 全段强烈反向（5s 处 −0.21）。**不是 packing artifact**：5min 仍 −0.09，远超 packing window (~5s)。诊断为 **same-cluster burst-clustered**：该 subject 的事件成簇出现，每簇内同 template 主导，跨簇才换 template。这是该 subject 的真实数据特征，不是 confound。

---

## 6. Secondary descriptive：next-event transition odds

| Subject | p_next_opposite | transition_odds | baseline_odds | odds − baseline |
|---|---:|---:|---:|---:|
| epilepsiae/1073 | 0.495 | 0.981 | 0.900 | **+0.08** |
| epilepsiae/139 | 0.469 | 0.883 | 0.945 | −0.06 |
| epilepsiae/548 | 0.255 | 0.343 | 0.471 | **−0.13** |
| epilepsiae/635 | 0.513 | 1.055 | 0.997 | +0.06 |
| epilepsiae/958 | 0.469 | 0.884 | 0.996 | −0.11 |
| yuquan/chenziyang | 0.500 | 0.999 | 0.953 | +0.05 |

**3 正 / 3 负**——与 window-based excess 同样 3/3 split，且 outlier subject (548) 在两类 metric 上都显著负。

**结论**：window-based excess 与 next-event odds 给出**一致**的 cohort NULL，不是单一 metric 的伪影。

---

## 7. 关键 caveats（按 plan §10）

1. **Cohort 功效极有限**：n=6，Wilcoxon (greater) 单边最小可能 p = 0.016（W=0），现实 p=0.844 远超此值。这种 NULL 不是"边缘 NULL"——即使 H1 成立，也不可能在 n=6 上仅落到 0.844。论文不能 claim 时间签名"接近显著"。

2. **观测性数据不能否定因果**：fwd/rev cohort 在 fixed-window excess 上 NULL，不等于"两类模板时间上无关"或"无因果"。可能存在的逃脱机制：
   - 隐状态调制（subject 当前 vigilance / interictal phase）让短窗内的耦合被覆盖
   - 长时程 excitability buildup（PR-2.7 已观察 seizure-centered broad rate elevation）让耦合发生在 hours 而非 seconds
   - 电极采样偏差：HFO 高密度 channel 之间的几何对偶可能在低密度采样下被打散
   - 我们的 cluster 定义本身可能合并了多个真实子模板

3. **Subject 548 的 outlier 行为**：N2 excess(10s) = −0.20 是 cohort 中唯一极端值。该 subject 的 same-cluster burst-clustered 特征会主导任何 size-weighted 池化；plan 已用 subject-level paired Wilcoxon 避开这个问题，548 仅贡献一个负方向 vote。但论文级 framing 不应把 548 当作 "negative coupling" 证据，只描述为该 subject 的 burst 特征。

4. **N4 conditional follow-up 未触发**：N2/N3 一致，未达 plan §9.6 触发条件。N4 (rate-matched ISI per cluster) 仍按 plan 故意 raise NotImplementedError，避免静默使用。

5. **Packing-proximity 信号**：1073 (1s, +0.14) 与 chenziyang/635 (1s, +0.08~+0.12) 在 packing window 内有正信号，但 10s 已归零或转负 → confound profile B 不主导 cohort 行为；如未来对包窗 sensitivity sweep 必要，挂在 plan §9.4 路径。

---

## 8. 可写 / 不可写（论文级 framing）

✅ 可以写：
- "Forward/reverse propagation geometries coexist (PR-6 source/sink swap, n=6 sign-test p=0.031), but their event timing shows no robust short-window reciprocal coupling at the tested scales (Δt ∈ [10s, 30s])"
- "The bouncing-back / short-range reciprocal version of the Ping-Pong hypothesis is rejected on both N2 (local-window) and N3 (circular-shift) nulls; geometric coupling is preserved"
- "Long-range slow modulation is not a confound (excess ≈ 0 at Δt ≥ 30 min)"

❌ **不**写：
- "两种模板时间上无关 / 无相关 / 无因果"
- "Forward/reverse templates are independent slow-modulated streams"（这等于"无关"）
- 删除整篇 PR-6 几何 narrative（PR-6 几何 NOT 被 PR-7 否定）
- 把 548 的强反向当作"反向耦合 negative" 证据（subject-level outlier，不能升级为 cohort claim）

---

## 9. 下一步

| Step | 内容 | 状态 |
|---|---|---|
| Step 3 | H1 cohort 封 NULL + 主图 1 + per-subject excess curves | **DONE / 本文件** |
| Step 4 | H2 negative control（n=17, non-fwdrev）跑齐 + 加入主图 1 灰色对照线 | TODO（~4-6h compute）|
| Step 5 | Robustness（N2 window sweep, packing-window sensitivity）+ N4 仅在 N2/N3 不一致时触发（当前未触发）| TODO 简化（N2/N3 已一致）|
| Step 6 | 主图 2 (per-subject 4-null grid) + 主图 3 (direction asymmetry + transition odds) + 主图 4 (exemplar) + Appendix | TODO |
| Step 7 | 最终 archive results doc 收口 + topic1 主文档 §7.11 一句话结论 | 待 H2 + 全图就位后 |
| Follow-up（独立 PR）| 历史依赖模型 P(next | history + state) 替代 fixed-window excess | 不绑 PR-7 主线 |

---

## 10. 代码与产物入口

- `src/template_temporal_pairing.py` —— statistical layer（compute_pairing_lift / 4 个 shuffler / N4 raise / cohort_paired_test / evaluate_pass_criteria）
- `tests/test_pr7_template_pairing.py` —— 16 项 TDD（含 review-加固 T11–T16）
- `scripts/run_pr7_template_pairing.py` —— CLI runner（--audit / --per-subject / --cohort-stats / --all）
- `scripts/plot_pr7_template_pairing.py` —— 主图 1 + per-subject curves + figures/README.md
- `results/interictal_propagation/template_pairing/pr7_cohort_audit.csv` —— 30 subject audit
- `results/interictal_propagation/template_pairing/per_subject/*.json` —— H1 cohort 6 subject per-subject 完整 N0/N1/N2/N3 × 8 Δt × 1000 perm
- `results/interictal_propagation/template_pairing/cohort_summary.json` —— H1 cohort triple-gate + 多 null × 多 Δt 聚合
- `results/interictal_propagation/template_pairing/figures/fig1_cohort_excess_curve.png` —— 核心 verdict 图
- `results/interictal_propagation/template_pairing/figures/per_subject/*.png` —— 6 张 per-subject excess 曲线
- `results/interictal_propagation/template_pairing/figures/README.md` —— 中文图说

---

## 11. 第一性原理与未排除的时间耦合形式

把 PR-7 数据看作一个 **marked point process**：
- **点**：每次 group event 的 absolute time
- **mark**：模板 label（A or B）
- **几何**：A/B 内部 rank template 的 source/sink 结构（已在 PR-6 建立）

"两类模板时间相关"至少有 5 种性质完全不同的形式。本 PR 只测了其中一种：

| 编号 | 时间关系形式 | 数学描述 | PR-7 状态 |
|---|---|---|---|
| (1) | **短时 cross-excitation** | P(B in (t, t+Δt] \| A at t) > marginal at Δt ≤ 1min | **测了，NULL（H1 主结论）**|
| (2) | **短时 persistence** | P(A in (t, t+Δt] \| A at t) > marginal | event-level 在主 metric 中以 same_lift 形式间接体现；548 同模板 burst 即此 case，PR-7 默认范围内描述 |
| (3) | **burst-level switching** | P(next-burst-label = B \| current-burst-label = A) > marginal_burst | **未测**：当前 event-level 10s metric 看不到 burst-scale 切换 |
| (4) | **latent-state coupling** | A/B 占比随隐状态（rate / vigilance / seizure proximity）漂移 | **未测**：本 PR 设计性排除（N2 30min 窗口在主 null 中 absorbing 这层调制）；属 Topic 1 PR-5 / PR-4A 已部分检验范围 |
| (5) | **几何相关但时间独立** | 两模板共享同一网络几何（PR-6），但每次事件的 mark 选择 ≈ 独立抽样 | **PR-7 cohort NULL 与此 default 兼容**；不能据此 claim 一定是 (5)，因为 (3) (4) 也兼容 |

PR-7 NULL **只**否定 (1)。Step 3.5 进一步测了 (2) 在**无 ISI 阈值的 same-label run 定义**下的两个签名（run length 与 lag-1 same-label）以及 run 间 gap-to-IEI 比值（详见 §14），结果**也基本 NULL**：cohort `run_length_lift` 中位 0.977、`lag1_same_excess` 中位 −0.013，6/6 subject 接近 null baseline；548 在 burst 维度有最强方向但量级仅 5–20% above null，**不**能完全解释其 event-level −0.20。**未排除**：(2) 在其它 burst 定义下的形式（如 ISI-threshold-based bursts）、(3) 慢状态切换（rate-state switching / seizure proximity switching）、(4) latent-state coupling。结论"两类模板共享同一病理网络几何（PR-6）"不被 PR-7 NULL 否定。当前数据**与 form (5) mark-independent sampling 在已测尺度上 compatible**——这是**最简洁的描述（most parsimonious）**，**不是 form (5) 的证明**：(2) under other burst definitions / (3) / (4) 仍可能存在，超出 PR-7 metric 视野。

---

## 12. 后续工作（不在 PR-7 主线范围内）

### 12.1 PR-7 内附加 follow-up：burst-level switching（form 3）

> 状态：plan-of-record 待写。在本文件锁住 H1 NULL 之后追加。

**问题**：把连续同模板事件合并为一个 "run / burst"；问 A-burst 结束后下一个 burst 是不是更可能是 B？这比 fixed-window event-level excess 更符合神经事件成簇出现的现实，并能区分 cohort outlier (548 same-cluster burst) 的 burst-level 行为是 (2) persistence 还是 (5) independence。

**最小合同（草稿）**：
- 输入：fwd/rev cohort 同 6 subject
- Burst 定义：连续相同 cluster_label 的最大事件 run（不引入新参数）
- Metric: `P(next_burst_label = B | current_burst_label = A)` per subject（symmetric: 也算 P(... = A | ... = B)）
- Null：保留 burst 序列长度分布 + global label 比例，permute burst-level label sequence（不能直接 reuse N2，因为 N2 在 event-level 操作）
- Cohort 测试：subject-level Wilcoxon on `P(next ≠ current) − baseline`
- 不进 H1 PASS judgment，作为 secondary descriptive

### 12.2 独立 PR follow-up：history-dependent marked point process model（form 1+2+4 一并）

> 状态：不绑 PR 编号，PR-7 之后独立设计。

**问题**：用回归 / hazard 模型代替 fixed-window excess。

**模型形式**：
```
P(next_event_label = B | history) ~ logistic(
    β0
    + β1 * previous_label
    + β2 * recent_rate (5min window)
    + β3 * time_since_last_event
    + block / state fixed effects
)
```

**核心检验**：加入 `previous_label` 是否显著提高 likelihood（对比无 `previous_label` 的 baseline 模型）。比 PR-7 fixed-window excess 更不依赖单一时间尺度，可以把 (1) (2) (4) 一并纳入框架。

**不在 PR-7 内**：作为单独 PR（编号待定），等 PR-7 收口后再立。

### 12.3 PR-7 内 H2 negative control（form 1 普适性补强）

> 状态：plan §7 Step 4 仍在路线图，但优先级降。

**作用**：检验 cohort NULL 是 fwd/rev 特异还是 HFO group event 普适特征。**不会改变 H1 NULL 主结论**——仅作论文完整性补强。预算 ~4–6h compute。**优先级**：低于 burst-level follow-up；可在 burst-level 完成后再跑。

---

## 13. 历史

- **2026-04-29**：本文件落盘（Step 3 H1 NULL 封口）；同步修正 plan §9.1 framing（仅否定 short-window reciprocal coupling 这一特定形式，不否定 PR-6 几何相关性 / 因果性 / 慢时间尺度耦合 / burst-level switching / 几何相关但时间独立等其他 4 种 time-coupling 形式）。论文叙事建议从"短时乒乓球理论"撤回，新叙事方向 = "双模板几何 + burst/state 选择"
- **2026-04-30**：追加 §14 Step 3.5 burst-level diagnostic 结果（n=6 H1 cohort，N2 主 null + N1 sanity，n_perm=500）。判读"在**无 ISI 阈值的 same-label run 定义**下未见 persistence，与 mark-independent sampling 在已测尺度上 compatible（最简洁解释，不等于证明独立）"：N2 cohort 中位 `run_length_lift` = 0.977、`gap_to_iei_lift` = 1.008、`lag1_same_excess` = −0.013；548 在 burst 维度上**不是**强 outlier（rll=1.054 vs cohort median 0.977），其 event-level 强反向 excess(−0.20) **不能**由 burst persistence 完全解释；其它 burst 定义、rate-state switching、seizure proximity switching、form (4) latent-state coupling 均**未测**，留作独立 follow-up

---

## 14. Step 3.5 — Burst-level diagnostic 结果（post-hoc exploratory）

> 状态：post-hoc exploratory，**不**改 H1 NULL verdict，**不**进 PASS/FAIL 判据
> Plan: `pr7_step3p5_burst_diagnostic_plan_2026-04-29.md`
> 数据：`results/interictal_propagation/template_pairing/per_subject_burst/*.json`
> 图：`results/interictal_propagation/template_pairing/figures/fig5_burst_diagnostic.png`

### 14.1 Cohort-level summary（N2 主 null + N1 sanity）

| Metric | N2 cohort median | N1 cohort median | 解读阈值 |
|---|---:|---:|---|
| `run_length_lift` | **0.977** | 0.978 | > 1 = same-template persistence；≈ 1 = mark 序列像独立抽样 |
| `gap_to_iei_lift` | **1.008** | 1.000 | > 1 = burst 时间成簇 |
| `lag1_same_excess` | **−0.013** | −0.013 | > 0 = empirical 比 null 更易 lag-1 same |

N2 与 N1 在所有 6 subject 上**几乎完全一致**（差 < 0.005）→ robustness 守住，结果不依赖具体 30 min 窗口选择。

### 14.2 Per-subject 数字（N2 主 null）

| Subject | N | run_length_lift | gap_to_iei_lift | lag1_same_excess |
|---|---:|---:|---:|---:|
| epilepsiae/1073 | 193171 | 0.932 | 0.981 | −0.034 |
| epilepsiae/139 | 14438 | 1.005 | 1.031 | +0.002 |
| **epilepsiae/548** | 25282 | **1.054** | **1.199** | **+0.014** |
| epilepsiae/635 | 13973 | 0.947 | 0.985 | −0.028 |
| epilepsiae/958 | 165577 | 0.989 | 1.109 | −0.005 |
| yuquan/chenziyang | 9609 | 0.965 | 0.972 | −0.017 |
| **cohort median** | | **0.977** | **1.008** | **−0.013** |
| **n > baseline (N2)** | | **2/6** | **3/6** | **2/6** |

### 14.3 判读路径（按 plan §5）

**走 §5.2 路径——在已测尺度（无 ISI 阈值的 same-label run）下与 mark-independent sampling compatible（最简洁解释，不是证明独立）**：
- cohort 中位 `run_length_lift = 0.977` ≈ 1，**未检测到** same-template persistence
- 6 subject 中**仅 2 个**（548, 139）在 N2 下 `run_length_lift > 1`，且 magnitudes 小（最大 1.054）
- `lag1_same_excess` cohort 中位 −0.013（极小负值）；2/6 > 0；6 subject 一致**接近 0**
- N1 sanity 与 N2 完全一致 → 不是 N2 局部窗口选择伪影

**未测**：其它 burst 定义（如 ISI-threshold-based bursts）、rate-state switching、seizure proximity switching、form (4) latent-state coupling。Step 3.5 NULL 仅适用于上述特定 metric，**不能**升级为"模板在 burst 层面无关"。

按 plan §5.2 论文 framing（修正版，避免过度解释）：

> "At the event/run scale tested here (no-ISI-threshold same-label runs), mark sequences are statistically indistinguishable from N2 local-window null shuffles (cohort median run_length_lift = 0.977, |lag1_same_excess| < 0.04 in all 6 subjects, 2/6 above null in either metric). At these specific scales the data are **compatible with mark-independent sampling** as the most parsimonious description; this is not proof of independence. Slower-state switching, rate-state coupling, seizure-proximity coupling and other burst definitions remain untested here and are deferred to independent analyses."

### 14.4 548 specific diagnostic（按 plan §5.3）

`epilepsiae/548` 在 event-level Step 3 中是 cohort outlier（excess(10s) = −0.20）。Step 3.5 检验该负向是否由 same-template burst persistence 解释（form 2）：

- 548 `run_length_lift = 1.054`：**cohort 最大值**，但**绝对量级很小**（仅高于 null 5%）
- 548 `gap_to_iei_lift = 1.199`：cohort 最大；时间 burst 比 null 略强
- 548 `lag1_same_excess = +0.014`：cohort 最大正值；但 lag-1 same fraction 仅高 1.4 个百分点

**结论**：548 在 burst 维度上确实是 cohort 中**方向最一致**的（3 个 metric 都正）但**量级仅高 5–20% above null**。其 event-level 强反向 `excess(−0.20)` **不能**由 burst persistence 完全解释——burst-level 信号比 event-level 量级小一个数量级。

按 plan §5.3 写法：

> "Subject 548 shows the strongest within-cohort burst direction (run_length_lift = 1.054, gap_to_iei_lift = 1.199, lag1_same_excess = +0.014, all 3/3 above null), but the magnitudes (≤ 5–20% above null) are an order of magnitude weaker than its event-level excess(−0.20). The event-level negativity is not fully explained by burst-level persistence at this granularity; alternative explanations (cluster definition artefact / sub-burst structure within the 5s packing window) remain open."

**严禁**升级 548 为 cohort claim。

### 14.5 与 §11 marked-point-process taxonomy 的关系

| Form | PR-7 检验 | 结果 |
|---|---|---|
| (1) short-window cross-excitation | Step 3 H1 cohort | **NULL**（已封）|
| (2) short-window persistence | Step 3.5 run length / lag1（**仅无 ISI 阈值 same-label run 定义**）| **未见 persistence at tested definition**（cohort median run_length_lift = 0.977）；其它 burst 定义未测 |
| (3) burst-level switching | Step 3.5 gap_to_iei + run_length 联动 | 仅 548 弱方向；cohort 不支持；slower state switching / rate-state / seizure-proximity 未测 |
| (4) latent-state coupling | **未测**（PR-7 设计性排除）| 留给 Topic 1 PR-4A / PR-5 framework / 独立 follow-up PR |
| (5) geometry-correlated mark-independent | Step 3.5 cohort behavior | **most parsimonious within tested scales**：mark 序列在已测的 event/run/lag-1 三个 metric 上 compatible with independent sampling |

**重要 caveat**：form (5) "most parsimonious" **不等于"已证明"**。NULL 结果只能说"在已测 metric 上没有检测到依赖"，不能说"独立"。具体限制：
- 仅 6 subject 的小 cohort，能检出 effect 范围有限
- 仅检验 event-level fixed-window excess + no-ISI-threshold same-label run + lag-1 same-label
- form (4) latent-state、其它 burst 定义、rate-state / seizure-proximity switching 都不在 PR-7 metric 视野内

### 14.6 论文 framing 升级（基于 Step 3 + Step 3.5 联合）

✅ 现在可以写：
- "At Δt ∈ [10s, 30s] (event-level, Step 3) and at no-ISI-threshold same-label run + lag-1 same-label scales (Step 3.5), forward/reverse template marks are statistically indistinguishable from null draws preserving block-conditional label fractions. The data at these specific scales are **compatible with mark-independent sampling** as the most parsimonious description (this is not proof of independence). PR-6 geometric coupling (source/sink swap, n=6 sign-test p=0.031) is preserved."
- "Subject 548's strong event-level reciprocal-coupling negativity (excess = −0.20) is not fully explained by either packing-window stickiness or burst-level same-template persistence at the tested run definition; it is reported as a single-subject outlier rather than a cohort feature."
- "Slower-state switching, rate-state coupling, seizure-proximity coupling, alternative burst definitions, and history-dependent regression models are deferred as independent follow-ups; PR-7 NULL verdict applies to the specific time signatures tested, not to all conceivable forms of template time-coupling."

❌ **不**写：
- "Two templates are time-independent" / "no causal coupling" / "mark sequences are mark-independent"
- "在 burst 层面也无关 / 也独立"
- 任何把 548 升级为 cohort claim 的措辞
- "Burst-level reciprocal coupling restores Ping-Pong"
- 删除 PR-6 几何 narrative

### 14.7 下一步

| 候选 | 优先级 | 说明 |
|---|---|---|
| Step 4 H2 negative control（n=17） | 中 | PR-7 完整性补强；H1 NULL + Step 3.5 form (5) default 已稳；H2 主要回答"非 fwd/rev cohort 是否同样 form (5)" |
| Step 5/6/7 PR-7 收口 | 中 | Step 5 robustness（N2 window sweep）+ Step 6 主图 2/3/4 + Step 7 最终 archive doc |
| 独立 follow-up PR：history-dependent regression | 低（不绑 PR-7 主线）| 测 form (1)+(2)+(4) 一并；按 archive §12.2 草案 |
| 进一步 case-series 看 548 | 低 | 检查 548 cluster 定义内部是否有 sub-template 结构；不进 PR-7 主线 |

---

## 15. Step 5 — N2 window sweep robustness（H1 cohort）

> 状态：plan §7 Step 5 deliverable
> 数据：`results/interictal_propagation/template_pairing/per_subject_n2_sweep/<dataset>_<subject>_w{10,30,60}min.json`
> 图：`figures/appendix1_window_sweep.png`

### 15.1 设计

- N2 主 null 在窗口 ∈ {10, 30, 60} min 三个尺度上重跑（Step 3 主跑用 30 min 默认）
- N0/N1/N3 不需要重跑——它们不消耗 `n2_window_seconds` 参数
- N4 conditional follow-up **不触发**：Step 3 已经验证 N2/N3 一致（5/6 subject 同号），无 disagreement
- n_perm=1000，seed=0；同 H1 primary cohort（n=6）

### 15.2 Cohort 结果（plan §3.2 triple-gate 在每个 window 重跑）

| Window | Wilcoxon(10s, greater) p | sign(10s) p | median(10s) | median(30s) | PASS |
|---|---:|---:|---:|---:|---|
| 10 min  | **0.7812** | **0.6562** | **−0.0020** | **−0.0024** | **NULL** |
| 30 min（与 Step 3 一致）| 0.8438 | 0.6562 | −0.0180 | −0.0149 | NULL |
| 60 min  | **0.8906** | **0.8906** | **−0.0291** | **−0.0291** | **NULL** |

**Cohort verdict robust**：三个 window 上 H1 triple-gate 都 NULL（Wilcoxon p ∈ [0.78, 0.89]，median(30s) 始终为负）。**注意**：cohort median 量级在 w=60 比 w=10 大 ~14 倍（−0.029 vs −0.002），方向一致但绝对量级随 window 单调放大；个别 subject 量级随 window 变化更显著（见 §15.3，尤其 548）。

**正确措辞**：cohort verdict 跨 window 稳健，但**不**应说 "三条曲线高度重合"——曲线之间在 10s/30s 尺度上分歧 ~0.025 cohort median；只是分歧的方向都同样 ≤ 0，没有跨越 PASS gate。

### 15.3 Per-subject `excess(10s)` 数字（sweep 全跑完，n_perm=1000, seed=0）

| Subject | window=10min | window=30min | window=60min |
|---|---:|---:|---:|
| epilepsiae/1073 | +0.0141 | +0.0084 | −0.0009 |
| epilepsiae/139  | −0.0180 | −0.0482 | −0.0683 |
| **epilepsiae/548** | **−0.0964** | **−0.2012** | **−0.2958** |
| epilepsiae/635  | +0.0275 | +0.0288 | +0.0251 |
| epilepsiae/958  | −0.0288 | −0.0444 | −0.0573 |
| yuquan/chenziyang | +0.0142 | +0.0144 | +0.0127 |
| **n positive** | 3 | 3 | 2 |
| **cohort median** | **−0.0020** | **−0.0180** | **−0.0291** |

**548 magnitude 对 window 高敏感**：随 window 从 10 → 30 → 60 min 单调放大 1× → 2× → 3× (−0.096 → −0.201 → −0.296)。这与 §14.4 burst diagnostic 一致：548 的 same-template burst 在 ~10–60 min 时间尺度上结构最强；window 越大，N2 null 越能在 window 内消除 burst clustering，empirical 与 null 的差异 (excess) 量级越大。

其它 5 subject 的 magnitude 比 548 小一个数量级，对 window 的依赖也弱：1073 / 635 / chenziyang 几乎不变（< 0.01 差异），139 / 958 单调小幅变负。所以 cohort 行为由 548 outlier 主导。

### 15.4 N4 conditional follow-up：未触发

按 plan §4.1 / §9.6，N4 仅在 N2 阳性但 N3 不一致时作 follow-up。Step 3 已经显示 N2 与 N3 在 5/6 subject 同号，且都 NULL → **触发条件不满足**，N4 跳过。`resample_isi_per_cluster` 仍按 plan §6.1 故意 raise NotImplementedError 状态保留（避免静默使用）。

---

## 16. Step 6 — 完整图集（main 1/2/3/4/5 + appendix 1/3）

> 状态：plan §7 Step 6 + plan §6.5 visualization spec deliverable
> 输出：`results/interictal_propagation/template_pairing/figures/`

| 文件 | 内容 | 来源 |
|---|---|---|
| `fig1_cohort_excess_curve.png` | cohort 主结论：H1 fwd/rev N2+N3 + H2 灰对照（H2 待跑）+ 1s/5s 黄区 + 30min/1h 蓝区 + verdict 文本框 | plan §6.5 主图 1 |
| `fig2_per_subject_null_grid.png` | 6-panel 网格：每 subject 在 N0/N1/N2/N3 下的 excess(Δt) 曲线 | plan §6.5 主图 2 |
| `fig3_direction_and_transition.png` | (a) H1b a→b vs b→a lift 散点 + (b) 次事件 transition_odds vs i.i.d. baseline | plan §6.5 主图 3 |
| `fig4_exemplars.png` | 两个 exemplar：548（最负）+ 635（最正），带 N2 null IQR envelope | plan §6.5 主图 4（简化版，未含 raster） |
| `fig5_burst_diagnostic.png` | Step 3.5 三 metric 散点（run_length / gap_iei / lag1_excess），N2 + N1 双 marker | Step 3.5 §14 |
| `appendix1_window_sweep.png` | N2 window {10,30,60} min cohort excess(Δt) 中位数三线对比 | plan §6.5 appendix 1 |
| `appendix3_cohort_audit.png` | 30 subject audit 表格（绿=H1, 蓝=H2, 灰=excluded）| plan §6.5 appendix 3 |
| `per_subject/<subject>_excess.png` | 每 H1 subject 的 N0–N3 excess(Δt) 单图（QC）| plan §6.5 per_subject grid 简化为单图 |
| `figures/README.md` | 中文图说（每图 2–4 句 + "**关注点**:" 行）| AGENTS.md 规范 |

### 16.1 简化决策与 plan §6.5 的差异

按 plan §6.5 vs 实际产物：
- **fig4 简化为 "exemplar curves with null IQR"，没做 raster + transition arrows**：raster 需要重新加载 raw events（10K–193K events/subject），且 30min raster 在 8×5 inch 画布上密度过高，readability 差。改为"两个 exemplar 的 excess(Δt) 曲线 + N2 null IQR envelope"，仍能 visual ize "even at extreme cohort ends, signals are within or near null bands"，足以承担 plan §6.5 fig4 的"故事性"角色。**这是工程简化，不影响科学结论**。
- **appendix 2 (packing window sensitivity) 跳过**：plan §9.4 触发条件 = "1s/5s 显著正 + 10s/30s ≈ 0 或反向"。实际数据（fig2 / fig4）显示 1s/5s cohort 极端值并不一致正向，部分 subject 全段负值（548, 139）。confound profile B **未触发**，appendix 2 跳过。
- **fig2 绑定 6 subject H1**（H2 cohort 没跑），plan §6.5 fig2 默认就是 H1 grid，无变更。

### 16.2 论文 framing 一句话

完整图集就位后，可以引用：

> "The H1 cohort shows triple-gate NULL on the N2 main null (Wilcoxon p=0.84, sign 3/6, median(30s)=−0.015) and the N3 robustness null (p=0.89). The cohort NULL verdict is robust across N2 window choices ∈ {10, 30, 60} min (all three windows yield triple-gate NULL with cohort median(30s) ∈ [−0.029, −0.002]; Appendix 1). Magnitude is window-sensitive — driven primarily by the single-subject outlier 548 (excess(10s) at window=10/30/60 min = −0.10 / −0.20 / −0.30) — but the cohort verdict and direction (median ≤ 0, n_positive ∈ {2, 3}) are not. Direction symmetry (fig3a) and next-event transition odds (fig3b) are consistent with the cohort NULL except for subject 548."

---

## 17. PR-7 最终结论（locked）

> **几何上相关，已测试时间尺度上未见 mark dependence。**

完整长版（与 plan §9.1 / §14.6 / §15.2 / §16.2 一致）：

> "PR-6 establishes that forward/reverse propagation templates share the same underlying network geometry (n=6 sign-test p=0.031, source/sink swap). PR-7 tests temporal coupling between these template marks at three classes of metric: event-level fixed-window opposite-template excess at Δt ∈ [10s, 30s] (Step 3 H1 primary), event-level direction asymmetry and next-event transition odds (Step 3 secondary), and run-based persistence (no-ISI-threshold same-label runs, lag-1 same-label, run gap-to-IEI ratio; Step 3.5 post-hoc). The H1 triple-gate is NULL on the N2 main null (Wilcoxon p=0.84, sign 3/6, median(30s)=−0.015), the N3 robustness null (p=0.89), and the N2 sweep across window ∈ {10, 30, 60} min (all three NULL). Step 3.5 finds no same-template persistence (cohort run_length_lift median=0.977, lag1_same_excess median=−0.013). At these tested scales the data are compatible with mark-independent sampling as the most parsimonious description; this is not proof of independence. The bouncing-back / short-range reciprocal version of Ping-Pong is rejected; geometric coupling is preserved. Slower-state switching, rate-state coupling, seizure-proximity coupling, alternative burst definitions, and history-dependent regression remain untested and are deferred to independent follow-ups."

**禁止措辞**（重申）：
- ❌ "Two templates are time-independent / no causal coupling"
- ❌ "Burst-level reciprocal coupling restores Ping-Pong"
- ❌ "Mark sequences are mark-independent"（应说 "compatible with mark-independent sampling at tested scales"）
- ❌ 任何把 548 outlier 升级为 cohort claim 的措辞

**下一步**（**不在 PR-7 内**）：
- **History-dependent marked point process model**：`next_label ~ previous_label + recent_rate + time_since_last + block / state`，可以一并测 form (1) + (2) + (4) 不依赖 fixed-window metric。独立 PR，编号待定。
- **H2 negative control（n=17）**：完整性补强。优先级低于 history model 因为不会改变 H1 NULL verdict。

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

PR-7 NULL **只**否定 (1)。**未排除** (2)（事实上 same_cluster persistence 是 548 的 cohort outlier 的合理解释）、(3)、(4)。结论"两类模板共享同一病理网络几何（PR-6）"不被 PR-7 NULL 否定。

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

- 2026-04-29：本文件落盘（Step 3 H1 NULL 封口）；同步修正 plan §9.1 framing（仅否定 short-window reciprocal coupling 这一特定形式，不否定 PR-6 几何相关性 / 因果性 / 慢时间尺度耦合 / burst-level switching / 几何相关但时间独立等其他 4 种 time-coupling 形式）。论文叙事建议从"短时乒乓球理论"撤回，新叙事方向 = "双模板几何 + burst/state 选择"

# Checkpoint B 报告 — Step 5d 系列总览（2026-05-21）

> 状态：Checkpoint B 评估完成。
> Verdict：**触发条件已满足（soft form），重跑继续**。
> 主入口：`docs/topic0_methodology_audits.md`
> 路线图：`./rerun_roadmap_2026-05-20.md` §"Checkpoint B"

---

## 1. 三段式朴话

**测了什么** —— Step 5d 是把 PR-4A / PR-4B / PR-4C（PR-4D 被跳）所有 cohort-level 描述层指标在修过版 cluster labels 上重跑一遍，看 cohort verdict 是否翻转。Checkpoint B 的预设作用：**在进入 PR-5/6/7 主结论重跑之前，先确认 PR-4 系列没出现"显著 ↔ NULL"或"方向反转"这类破坏性翻转**——如果有，要先 reconcile，否则下游会基于不一致的中间层数字。

**怎么测的** —— 走完 5d.1/5d.2.0/5d.2.1/5d.2.2/5d.3 五步，每步把修过版 vs 原版做配对对比；advisor consult 来判定是否触发"hard stop"。

**揭示了什么** —— **触发条件满足，但形态是"在小子集上显著性消失，方向不变"，不是"方向反转 / 全 cohort 崩塌"**。具体：

| 翻转候选 | 类型 | 原版 | 修过版 | 评级 |
|---|---|---|---|---|
| **PR-4B Step 23: L3 高置信子集 (dom_r>0.7) Pearson r delta** | "显著 → NULL"，方向同 (+) | n=8, Δ +0.083, 7/8, **p=0.016** | n=8, Δ +0.053, **5/8, p=0.547** | **fragility-on-small-n** form；不是 sign flip |
| PR-4B Step 1: L1 dominant cluster rho | 方向 sign flip | median −0.083, 13/30 positive | median **+0.183**, 25/40 positive | True direction reversal **but** Wilcoxon p NS on both sides; cohort verdict unchanged (NULL stays NULL) |
| 其他 PR-4A/B/C 所有 cohort Wilcoxon | NULL stays NULL | — | — | **无变化** |

代号补注：Checkpoint B 见路线图 §"Checkpoint B"，Topic 1 主文档 §3.1c L3 高置信亚组 H4 探索性正向 finding（archive `interictal_group_event_internal_propagation.md` PR-4B Step 2-3 节）。

---

## 2. Verdict — Soft trigger 触发，重跑继续

**为什么不是 "Checkpoint B FAILED"**：
- 唯一显著 → NULL 的翻转 (PR-4B Step 23 L3) 是在 n=8 极小子集上；方向同号；幅度从 Δ +0.083 降到 +0.053（不是 +0.083 → -0.05）。最可能成因：phantom-era subset membership 在 masked labels 下变化，dom_r>0.7 gate 选中的 subject 集合微调，n=8 太小，1-2 个 subject 跨阈值就足以把 p 从 0.016 推到 0.547。
- 原版 archive 本来就明确这条是"探索性"。
- 没有任何 metric 出现"NULL → 显著"的反方向翻转。
- 没有任何 metric 在全 cohort (n=30+) 层面出现方向反转。
- L1 sign flip 是 true direction reversal，但 Wilcoxon p 在 NS 区域，cohort verdict 不变。

**为什么也不能 "Checkpoint B PASSED" 简单说过**：
- 触发条件按字面意义 = "PR-4B/D 方向反转或显著 → NULL 翻转" 已满足
- 用户的 N=10 并行 5e/5f/5g/5h session 仍在继续，但 5i 收口阶段必须根据本 verdict 调整 Topic 1 §3 主结论措辞

**正确措辞**：**"触发条件满足，形态是 fragility-on-small-n 不是 sign-reversal；broad re-derivation 按 plan 继续；Topic 1 §3 主结论里 L3 高置信亚组的 H4 探索性正向 finding 必须降级"**。

---

## 3. Topic 1 主文档 §3 必须改的地方（5i 收口时执行）

### 3.1 现状（原版 phantom 数字）

> §3.1c PR-4B Step 2-3 描述层：
> 慢调制（PR-4B）：模板混合（L1）与模板内顺序一致性（L2）cohort 全 null；模板内相对时延结构（L3）在全 cohort 上证据不足，仅在 8 个高置信子集（dom_r > 0.7）的 Pearson r 上探索性显著（p=0.016, 7/8）。

### 3.2 修改后建议

> 慢调制（PR-4B）：模板混合（L1）与模板内顺序一致性（L2）cohort 全 null；模板内相对时延结构（L3）在全 cohort 上证据不足。L3 在 8 个高置信子集（dom_r > 0.7）的 Pearson r 上**原版有探索性显著 (p=0.016, 7/8)，但在 phantom-rank 修复后不复现（修过版 p=0.547, 5/8，方向同 +，幅度减半）——归为 fragile-on-small-n 信号**，不进入 SEF-ITP H4 evidence base。详见 Topic 0 §3.1 + `docs/archive/topic0/lagpat_phantom_rank/step5d3_pr4c_results_2026-05-21.md` (PR-4C NULL 保持) + 本 Checkpoint B 报告。

### 3.3 SEF-ITP framework H4 (rate-geometry 解耦) 影响

- SEF-ITP H4 的 rate-geometry coupling 实证支撑里**不能引用** PR-4B L3 高置信子集 Pearson r 结论
- H4 主要 evidence base 转移到 PR-5-B 的 rate_by_template post-ictal 招募抬升（由 5e 重跑确认）
- 如果 5e 跑出来 PR-5-B post-baseline events/h 主信号也翻转，SEF-ITP H4 的 evidence base 整体瓦解 — 这是 5e 完成后单独评估的事，不在本 Checkpoint B 范围

---

## 4. 5d.4 PR-4D 跳过的影响

**Plan 字面意义上的 Checkpoint B 不完整**：路线图 §Checkpoint B 写"advisor consult: PR-4B/D 方向是否反转"，PR-4D 没跑就没有"PR-4D 方向"可评。

**User 2026-05-21 授权 skip 理由**：PR-4D rate-template decomposition 的核心终点（rate-burst seizure-enrich strict-match）与 PR-5-B 的 rate_by_template post-baseline events/h 信号在科学上重叠。优先 5e 比补 PR-4D 更经济。

**Open risk**：如果 5e 跑出来 PR-5-B 主结果（post-baseline events/h, p=0.00128）出现翻转，可能需要回头补 PR-4D 作为辅助 sensitivity。Plan deviation 已记录。

---

## 5. 各 5d step verdict 汇总

| Step | metric 重点 | 原版 | 修过版 | verdict |
|---|---|---|---|---|
| 5d.1 PR-4A | dominant_fraction day vs night Wilcoxon | p=0.124 | p=0.734 | NULL stays NULL，方向同 |
| 5d.2.0 PR-4B Step 0 | exact_order_match_fraction + dominant r | 1.0 / 0.601 | 1.0 / 0.580 | 同方向 (修过版 sanity 验证) |
| 5d.2.1 PR-4B Step 1 | rate-state coupling (raw τ / centered τ / L1 ρ) | all NULL; L1 ρ=−0.083 (13/30 +) | all NULL; L1 ρ=**+0.183** (25/40 +) | NULL stays NULL；L1 sign flip 但 NS |
| 5d.2.2 PR-4B Step 23 | L3 高置信子集 Pearson r delta | **p=0.016, 7/8 (Δ=+0.083)** | **p=0.547, 5/8 (Δ=+0.053)** | **触发: 显著 → NULL，方向同；fragility-on-small-n** |
| 5d.3 PR-4C | 5 propagation 指标 cohort Wilcoxon × 3 windows | NULL × 5 × 3 | NULL × 5 × 3 (n=30) | NULL stays NULL，方向同 |
| 5d.4 PR-4D | — | — | **SKIPPED** | Plan deviation 已记录 |

**总结**：1 个"显著 → NULL" 翻转 (PR-4B L3 高置信)；1 个 sign reversal 但 NS (PR-4B Step 1 L1 ρ)；其他 4-5 个全部 NULL stays NULL 同方向。

---

## 6. Checkpoint B 决议

1. **Verdict = trigger met (soft form), broad re-derivation 继续**
2. **5i 收口时必须修 Topic 1 §3.1c**（降级 L3 高置信 finding）— 见 §3.2 建议措辞
3. **5i 收口时必须修 SEF-ITP framework v1 H4 evidence base 段**（移除 PR-4B L3 高置信引用）
4. **PR-4D skip 记录**：5i 时 decide 是否补
5. **5e/5f/5g/5h 继续按 plan**，无 hard stop
6. **Checkpoint B 不再 advisor consult 第二次**（advisor 2026-05-21 已 sign off 本判定）

---

## 7. 下一步衔接

- 用户 N=10 并行 session 处理 5e/5f/5g/5h/5i
- 本 Claude session 已完成 Phase 0 到 Checkpoint B 之前的所有重跑工作
- 5i 收口阶段需要把本报告 §3 的措辞修改和 §6 的决议事项 fold 进 Topic 1 主文档

archive doc 索引：
- `step5a_pr2_results_2026-05-20.md`
- `step5b_pr25_results_2026-05-20.md`
- `step5c_pr3_results_2026-05-20.md`
- `step5d3_pr4c_results_2026-05-21.md`（本 step 配对档）
- `checkpoint_b_report_2026-05-21.md`（本文件）
- `phase0_progress_report_2026-05-21.md`（顶层 ledger）

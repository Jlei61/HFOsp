# 乒乓球假说（Ping-Pong）审阅与 PR roadmap

> 状态：working hypothesis review，2026-04-28
> 目的：把当前 Topic 1 已经建立的 forward/reverse template 现象，与"间期事件作为 SOZ 兴奋-抑制对偶的 biomarker" 这一更进一步的机制叙事对齐；明确哪些已被支撑、哪些是工作假说、哪些必须靠新实验验证。
> 性质：roadmap，不是结论；后续 PR 的 plan-of-record 单独归档（参见 §6）。
> 上游：`docs/topic1_within_event_dynamics.md` §3 / §7
> 下游：`docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md`（PR-7 Antagonistic Temporal Pairing）

---

## 1. 假说陈述（user 2026-04-28）

间期 HFO 群体事件的 forward / reverse 时序模板（Topic 1 PR-2 + PR-6 已建）是 SOZ 内部兴奋-抑制对偶的 biomarker：

- 每次 SOZ 内部兴奋性冲动（"Ping" = forward propagation）都伴随抑制性高墙的反弹（"Pang" = reverse propagation）
- SOZ 兴奋性逐渐积累 → 抑制平衡破缺 → 癫痫发作
- 间期事件刻板时序 = 病理网络的环路标志物
- post-ictal rate elevation = "比赛重置"

**类比**：乒乓球比赛中 SOZ 是球台，抑制性高墙是球网，每次发球（Ping）都被回击（Pang）；越接近发作，回合越快越激烈；最终一方失败（抑制崩溃）= 癫痫发作。

---

## 2. 必须三层分离（**这是最关键的方法学修正**）

把上述假说的论断按可证性强度分三层：

| 层 | 论断 | HFO 数据能否支撑 | 当前证据 / 状态 |
|---|---|---|---|
| **A 现象学** | 反向 rank template 在 30 subject 中 6–8 个跨时间复现 | ✅ 已支撑 | PR-2.5 forward/reverse OR rule reproduced n=8/9；PR-6 Step 4 节点级 source/sink swap 几何（n=6, swap−same sign-test p=0.031）已建 |
| **B 功能耦合** | T_a 与 T_b 在时间上配对出现（不是各自独立的两个 mode） | ⚠️ 数据足够，**完全没测** | PR-7 计划的核心检验目标 |
| **C 机制** | 反向 = 抑制性高墙反弹（excitation / inhibition 对偶） | ❌ HFO 80–250 Hz 不区分 E/I | 仅作文献语言对接（Schevon / Trevelyan inhibitory restraint），**不**作 HFO 数据结论 |

**核心结论**：
- A 已经稳；C 在 HFO 上原则上不可证伪
- **B 是从 A 走向 C 的唯一可量化桥梁**
- 不解 B，整个 Ping-Pong 故事就停留在"两个反向模板共存"，离"乒乓 = 因果对偶"差一整层
- 论文 framing 必须按 A + B 写，机制层 C 仅作文献借词，**不**写成"证明 SOZ 抑制墙"

---

## 3. 数字校核（与 user 直觉对账）

| user 回忆 | 当前文档实际 | 来源 |
|---|---|---|
| ~10 subject 完全反向模板 | candidate forward/reverse = **11–12/30**（inter-cluster r<−0.5 of which **6–8/30 跨时间复现**） | `interictal_group_event_internal_propagation.md` §7.3 + PR-2.5 |
| 7 个 subject "前三反向" 显著 | **PR-6 H2 forward/reverse-reproduced subset n=6**（OR 规则），swap geometry sign-test p=0.031 ✓ | `pr6_template_endpoint_anchoring_plan_2026-04-25.md` §15 Step 4b |
| post-ictal rate 提高 | dominant template post-ictal rate elevation Bonferroni-pass | PR-5 §11 |
| SOZ 解剖锚定 | H1 cohort NULL（p=0.42, n=21）；endpoint vs middle SOZ frac 不显著 | PR-6 Step 3 |

**论文写作口径**：用"forward/reverse-reproduced subset n=6"或"swap-leaning subset n=6"，**不**写"7 个"或"10 个"。这些数字在不同口径下会变化，必须固定到一个 audit-derived count。

---

## 4. user 提的补充实验，逐项与现有 PR 对账

### 4.1 已经做过 / 已部分做过的（不重复）

| user 提议 | 现状 | 升级方向 |
|---|---|---|
| 反向节点 vs SOZ 重叠 | PR-6 H1 cohort NULL；node-level swap-vs-template-specific SOZ frac Wilcoxon p=0.19 | PR-6 §15 Step 6 候选 #1：用 ER ratio leading channel 替代 SOZ JSON（已列入 backlog） |
| 节点级 swap geometry | PR-6 Step 4b 已离散计数（swap_node count, fwd/rev sign-test p=0.031） | PR-8 candidate：升级到连续 signed displacement（见 §6） |
| post-ictal rate ↑ | PR-5 主结论已封板（dominant template +65 ev/h Bonferroni-pass） | PR-9 candidate：按 fwd/rev vs dominant subject 拆分（见 §6） |

### 4.2 真正的 gap（必须做新实验）

| user 提议 | 现状 | 计划归属 |
|---|---|---|
| **拮抗性时间配对**（T_b 是否紧跟 T_a） | ❌ 完全没测 | **PR-7**（已立 plan-of-record） |
| **节点级 signed rank displacement metric** | ⚠️ PR-6 是离散；连续版没做 | **PR-8 candidate**（PR-7 之后） |
| **subject typology × PR-5 split** | ❌ 没按类型拆 | **PR-9 candidate**（依赖 PR-8 分类） |
| **降维可视化两类模板 + 散落事件** | ❌ 没做 | **Supplementary figure**，**不**进 PR 主线 |
| **longitudinal SOZ 扩张**（hHI channel set 时间扩展） | ❌ 没做 | **不进当前论文主轴**（数据时长不足；月-年级 longitudinal 才有意义） |

### 4.3 user 提议中需要谨慎的部分

- **"前三反向 = 完全反向"硬阈值** — 不预注册容易调参。**PR-8 candidate 必须先报告连续谱**（slope / Spearman / normalized displacement / swap-same count），分类只作描述层
- **PCA / UMAP 直接对 rank 向量** — rank 是序数对象，欧氏距离不合理。改用 Kendall / Spearman / footrule 距离 + PCoA / MDS。这条由 supplementary figure 承担，**不**作核心证据
- **"完全抑制"= 反向传播** — 反向传播也可能是兴奋性 back-projection 或 spreading-depression-like。HFO 不能区分；必须按 §2 的三层分离写

---

## 5. 主要新实验：PR-7 Surrogate 设计要点（来自 user 2026-04-28 的 push back）

PR-7 完整合同见 `pr7_template_antagonistic_pairing_plan_2026-04-28.md`。这里只摘要 surrogate 设计原则，避免双源漂移。

**核心问题**：单纯"固定时间戳 + shuffle template label"太弱 — 总事件率有 burst 和慢漂移，全局 label shuffle 会把"高 rate 时段事件天然更密"误判成 T_a → T_b 配对。

**PR-7 Null hierarchy 与角色**：

| 层 | Null | 何时跑 |
|---|---|---|
| 主 null | **N2** local-window shuffle (30 min) | 必跑（H1 PASS / NULL 判据基于此） |
| Robustness | **N3** circular shift label sequence | 必跑（N2 与 N3 一致 → robust） |
| Sanity | **N0 / N1** global / block-aware shuffle | 必跑（ceiling + mid-strength 对照） |
| Conditional | **N4** rate-matched ISI per cluster | **仅在 N2 阳性但 N3 不一致时**作 follow-up；不作主交付（参数化太重） |

**主 H1 判读用 N2**；同时报告 opposite-template 与 same-template excess，避免 burst-induced clustering 被误读为 fwd/rev pairing。

**PR-7 PASS 三条门**（pre-registered，写死）：
1. `excess(10s)` Wilcoxon p < 0.05 (primary)
2. `excess(10s)` sign test p < 0.05 (primary)
3. `excess(30s)` 中位数 > 0 (required sensitivity，不必显著但**不能反向**)

10s primary + 30s sensitivity 的设计避免单点显著被 packing-proximity stickiness 主导（5s packing window edge 太近）。1s / 5s 仅作 packing-proximity diagnostic，不进 PASS 判据。

---

## 6. PR roadmap（按依赖关系）

```
PR-7 (Template Antagonistic Temporal Pairing) ← 当前
   │
   │ PASS (lift > 1 in short window, ≈1 in long window)
   ↓
PR-8 candidate (Signed rank displacement / continuous pair-geometry)
   │
   │ 完成 subject typology
   ↓
PR-9 candidate (Subject typology × PR-5 split)
   │
   ↓
Supplementary figures (PCoA/MDS visualization, packing-window sensitivity)
```

**优先级 / 时间线**：

| PR | 内容 | 依赖 | 性质 | 预算 |
|---|---|---|---|---|
| **PR-7** | Antagonistic Temporal Pairing（功能耦合层） | 已就绪（复用 PR-6 cohort） | 决定 Ping-Pong 假说生死 | 4.25 d |
| **PR-8 candidate** | Continuous signed rank displacement + 三型连续坐标 | 不依赖 PR-7 结果，但 PASS 后更有价值 | 论文 Fig 2 candidate；不预注册分类阈值 | ~3 d |
| **PR-9 candidate** | Subject typology × PR-5 post-ictal split | 依赖 PR-8 typology | PR-5 narrative 加强 | ~1 d |

**显式不做** / 不进主线：
- 降维可视化（PCoA/MDS）→ supplementary figure，PR-7 完成后做
- Longitudinal SOZ expansion → 数据时长不足，下一篇论文
- 任何 ER / CUSUM / ictal anchor 复辟 → PR-6-A 已封板冻结

---

## 7. 论文写作口径（pre-registered）

Conditional on PR-7 结果：

### 7.1 PR-7 PASS（H1 cohort + H2 negative control 都守住）

可以写：
- "Forward/reverse template pairs are temporally coupled at short timescales (Δt ≤ 1 min) but uncoupled at long scales (Δt ≥ 30 min), supporting an antagonistic functional pairing"
- "This pattern is consistent with literature on excitatory-inhibitory restraint and rebound (Schevon 2012, Trevelyan 2013), but cannot directly distinguish E vs I from HFO 80–250 Hz"
- "We propose interictal forward/reverse template pairing as a candidate biomarker of pathological network excitability balance"

**不**可以写：
- ~~"We demonstrate that forward HFO propagation triggers an inhibitory rebound"~~
- ~~"The reverse template represents firing of the inhibitory restraint wall"~~

### 7.2 PR-7 NULL（实际结果，2026-04-30 lock）

PR-7 H1 三条 metric 全部 NULL 已验收（详见 `pr7_template_pairing_results_2026-04-29.md` §17 + topic1 §7.11）。**预先写在本节的 framing（"independent slow-modulated streams" / "删除整篇 Ping-Pong metaphor"）2026-04-30 user review 推翻——太强**。

正确写法（locked across §17 / topic1 §7.11）：

✅ **可以写**：
- "Forward/reverse propagation geometries coexist (PR-6 source/sink swap, n=6 sign-test p=0.031), but their event timing shows no robust short-window reciprocal coupling at Δt ∈ [10s, 30s] (Step 3 NULL on N2 + N3 + window sweep)"
- "No same-template persistence detected at no-ISI-threshold run scale (Step 3.5)"
- "At these tested scales the data are **compatible with mark-independent sampling** as the most parsimonious description; this is not proof of independence"
- "The bouncing-back / short-range reciprocal version of Ping-Pong is rejected; geometric coupling is preserved"

❌ **不**写：
- ~~"Two templates are independent slow-modulated streams"~~（等于"无关"）
- ~~"Mark sequences are mark-independent"~~（NULL ≠ proof of independence）
- ~~"删除整篇 Ping-Pong metaphor"~~——只撤回"短时接力"叙事，几何相关性 + 慢时间尺度 ping-pong（form 4/5）仍然开放
- ~~把 548 outlier 升级为 cohort claim~~

**未测**（archive results §11 / §17 + topic1 §7.11 锁死）：
- alternative burst definitions（ISI-threshold-based bursts）
- rate-state switching、seizure-proximity switching
- form (4) latent-state coupling
- history-dependent regression on `next_label ~ previous_label + recent_rate + time_since_last + block / state`

### 7.3 PR-7 H2 FAIL（non-fwdrev cohort 也显著）

写：
- "Short-range opposite-cluster pairing is a general feature of HFO group event clustering, not specific to forward/reverse subjects"
- 降级为 "general short-range clustering"，不进 fwd/rev 机制叙事

---

## 8. 与其他 topic 的边界

- **"~2 Hz peak 是不是真的"** → `docs/topic2_between_event_dynamics.md`（Topic 2 已封板：dead-time + slow modulation，不是 oscillator）
- **"慢调制是否 SOZ-specific"** → `docs/topic3_spatial_soz_modulation.md`（PR-1 Yuquan 9 subject：detrend_fraction SOZ 更低；PR-2 Epilepsiae underpowered）
- **PR-7 不回答** "fwd/rev 与 SOZ 解剖关系"（PR-6 H1 cohort null 已封板，单一 metric 不能锚定）

---

## 9. 历史链接

- `docs/topic1_within_event_dynamics.md` — Topic 1 主文档
- `docs/archive/topic1/interictal_group_event_internal_propagation.md` — PR-2 / PR-2.5 完整结果
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — PR-6 cluster centroid 几何（PR-7 输入合同的来源）
- `docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md` — **PR-7 plan-of-record**
- `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md` — PR-5 dominant template post-ictal rate elevation（PR-9 candidate 依赖）

---

## 10. 当前 commit 与版本

- 2026-04-28：本 review + PR-7 plan-of-record 同步落盘
- 2026-04-29：PR-7 Step 0–3 完成；H1 cohort NULL 已封；新增 archive results doc + Step 3.5 burst diagnostic plan
- 2026-04-30：PR-7 Step 3.5 / 5 / 6 完成 + 验收；framing locked across §17 / topic1 §7.11 / 本文 §7.2；PR-7 主线封板。本 review §6 提到的 PR-8 / PR-9 candidates **未启动**；后续 follow-up 改名为 history-dependent marked point process model（与 PR-8 candidate 不重叠）

---

## 11. PR-7 收口后的状态判读（2026-04-30）

### 11.1 已被 PR-7 否定的具体形式

仅一种：**event-level fixed-window opposite-template excess at Δt ∈ [10s, 30s]**。这是 user 原始 ping-pong 假说中的"短时接力"版本。

### 11.2 PR-7 NULL **不**否定的 6 类时间结构

archive results §11 marked-point-process taxonomy + §17 final conclusion 列出了**仍开放**的所有时间耦合形式：

| 形式 | 是否被 PR-7 测过 |
|---|---|
| (1) short-window cross-excitation at 10s/30s | ✅ NULL（H1） |
| (2) short-window persistence (run length / lag-1) | ✅ NULL（Step 3.5）但仅在**无 ISI 阈值 same-label run** 定义下 |
| (3) burst-level switching at hours/days | ❌ 未测 |
| (4) latent-state coupling（rate-state / vigilance / seizure proximity） | ❌ 未测 |
| (5) geometry-correlated mark-independent | 当前数据 compatible（最简洁解释，非证明） |
| **NEW**：spatial within-event dynamics（intra-event SOZ → non-SOZ ping-pong）| ❌ 未测，本 review 11.3 提出 |
| **NEW**：长时间尺度 template ratio 趋势（peri-seizure）| ❌ 未测，PR-2.7 + PR-5 数据可重用 |

### 11.3 user 原始 ping-pong 直觉的多层结构

**重新读 §1 假说陈述，user 直觉实际包含至少 4 个独立 testable 层级**：

1. **现象学（已建）**：fwd/rev template 几何上对偶 — PR-6 source/sink swap geometry
2. **短时功能耦合（已测 NULL）**：T_a 与 T_b 在 ≤ 1min 上 reciprocal triggering — PR-7 Step 3
3. **长时系统动力学（未测）**：随时间推进，Ping-Pong "回合速度" 加快 → 越接近发作越频繁切换 → 抑制崩溃 → seizure
4. **空间内耦合（未测）**：单次事件内部，SOZ 通道兴奋 → 抑制 wall 通道反弹（intra-event spatial dynamics）

PR-7 NULL **仅否定了 #2**。user 的 *system claim* 包含 #3 和 #4，这两层根本没测。

---

## 12. 后续实验建议（第一性原理）

### 12.1 优先级最高：history-dependent marked point process model

> Already named as next-step in topic1 §7.11 + archive results §17。

**核心检验**：把"两类模板时间耦合"从 fixed-window metric 升级到 likelihood ratio：

```
M_full:    P(next_label | previous_label, recent_rate, time_since_last, block_id, state) 
M_reduced: P(next_label | recent_rate, time_since_last, block_id, state)   (drop previous_label)

LRT: Δ deviance = -2 (logL(M_reduced) - logL(M_full))  ~ χ²(df = #removed terms)
```

**为什么这个更好**：
- 不依赖单一 fixed window；自然包含多 timescale
- 同时检验 form (1) + (2) + (4)；如果 `previous_label` 显著贡献 → 时间依赖存在
- 可加 `interaction(previous_label, time_since_last)` → 检验依赖在哪个时间尺度
- `recent_rate` / `state` 控制慢调制 confound

**失败合同**：cohort-level LRT 不显著且 `previous_label` 系数 cohort 集中在 0 → 与 PR-7 NULL 一致。预算：~3 d。**独立 PR**，不绑 PR-7。

### 12.2 第二优先：长时系统动力学（peri-seizure template ratio）

**核心检验**：user 直觉 #3——"回合速度随发作邻近加快"。

**操作化**：
- 对每 seizure，画 [-12h, +12h] 时间窗口内每 5-min bin 的 (T_a count, T_b count)
- 计算每 bin `template_ratio = T_a / (T_a + T_b)`
- 检验：peri-seizure 是否 ratio 偏移（向 T_a 或 T_b 倾斜）？是否 variance 增大（cycle 速度加快）？
- Cohort：跨 subject paired comparison `pre_ictal vs baseline ratio` 与 `pre_ictal vs baseline variance`

**与已有 PR 的边界**：
- PR-5 已发现 dominant template post-ictal rate ↑（绝对率），**未拆 ratio**
- PR-2.7 已发现 seizure-centered broad rate elevation
- 本检验把这两个发现联动：rate 升 ≠ ratio 偏 ≠ variance 升，三者独立测才能说"ping-pong cycle speeds up"

**失败合同**：cohort-level paired test 不显著 → user 直觉 #3 不成立；rate elevation 由共同 driver 拉升。预算：~1.5 d，复用 PR-2.7 / PR-5 数据。**独立 PR**。

### 12.3 第三优先：548 单 subject 深度 case-study

**理由**：548 是 cohort 中**唯一**显示一致 burst 方向的 subject（Step 3.5 三 metric 都正向，PR-7 sweep 上 magnitude 单调放大 1×→3×）。但 PR-7 cohort 把它平均掉了。

**操作化**：
- 单独画 548 的 lagPat raster（30 min 局部窗口） + 时间轴上 T_a/T_b 着色
- 看是不是真存在 "burst of T_a, then burst of T_b" 这种模式
- 如果存在，按 burst 间隔时间分布做 inter-burst ISI distribution
- 可能能直接看到 "ping-pong cycle period"

**性质**：**case-study only**，**不**升级 cohort claim。论文层面写法：
> "Single-subject 548 exhibits a burst-clustered template structure consistent with the user's hypothesis at the within-cohort outlier level; this is exemplary, not generalizable."

预算：~0.5 d。

### 12.4 第四优先：intra-event spatial dynamics（user 直觉 #4 的最直接形式）

**核心检验**：在**单次群体事件内部**，SOZ 通道激发是否在时间上**领先**于非 SOZ 通道？

**操作化**：
- 取每个 group event 的 lagPatRaw（每通道在事件内的相对时间）
- 用 PR-6 已建的 SOZ vs non-SOZ matched_bipolar 划分
- 计算 `Δ_event = mean(lag_SOZ_channels) - mean(lag_nonSOZ_channels)`
- 如果 `Δ_event < 0` → SOZ 领先（"Ping" from SOZ）
- 如果 fwd template 上 `Δ_event < 0` 且 reverse template 上 `Δ_event > 0` → **直接 ping-pong 几何**

**为什么这是 user 直觉的最直接形式**：
- user 原话："SOZ 内部产生兴奋性冲动 → 高墙反弹"
- 如果 fwd = SOZ-first 且 reverse = SOZ-last，那 reverse template 就是空间上"从外向内的反弹波"
- 这是 *intra-event* 的 ping-pong，不是 *inter-event*；PR-7 测的是后者

**与 PR-6 的边界**：
- PR-6 H1 测 endpoint 是不是 SOZ（cohort NULL）
- 本检验测 endpoint 在事件内的**时间顺序**（lagPatRaw 极性 vs SOZ 标注）
- 数据完全不同：PR-6 用 centroid rank（averaged across events），本检验用 per-event lag

**失败合同**：cohort 上 `Δ_event` 不区分 fwd 和 reverse → user 直觉 #4 不成立。预算：~2 d。**独立 PR**。

### 12.5 不再做的方向

- ~~PR-8 candidate（continuous signed displacement）~~：现在没必要——PR-6 Step 4b 离散计数已足，连续版只是 visual 上更精细，不会改变 cohort verdict
- ~~PR-9 candidate（subject typology × PR-5）~~：现在没必要——PR-7 NULL + Step 3.5 NULL 已经表明 fwd/rev 不是有意义的 cohort split
- ~~longitudinal SOZ expansion~~：数据时长不足
- ~~重启 short-window reciprocal coupling 的更细粒度版本~~：plan §17 已封禁
- ~~burst-level reciprocal coupling 测试~~：trivial（run 边界恒为 switch）

### 12.6 综合建议

**真正的 follow-up paper （或 PR）应该做**：12.1（history-dependent regression） + 12.2（peri-seizure ratio） + 12.4（intra-event spatial dynamics）。三个互相独立、互相互补，**任一**显著都重新打开 ping-pong 叙事的某个特定层级。三个都 NULL 才能真正 close the door。

**论文写作策略**：当前 PR-7 NULL 保留 PR-6 几何 + PR-5 rate 结论；下一个工作把这三个 follow-up 任意 1–2 个跑完后写新文章。**不把当前 PR-7 NULL 当作 "ping-pong 死亡证明"**，仅当作"短时接力签名不成立"。

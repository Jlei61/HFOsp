# Stage 3 探索阶段 — 二端等强单网络自发事件分型（exploratory phase record）

**Date:** 2026-06-14 · **Run:** `scripts/run_stage3_regime_screen.sh` (user-authored), 58/61 readouts (Phase 2 `twoend_deph` arm 3/3 failed, non-critical) + 候选确认 `candidate_confirm/` + 事件分型 `scripts/analyze_stage3_event_types.py`. Raw: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/{regime_screen/regime_screen_raw.csv, event_types.csv, event_type_by_cell.csv, readable_global_events.csv}`.
**Status:** EXPLORATORY PHASE，已刻画收束。**不进建模主文档主结论；Stage 3 的"单网络标签-时序独立性"主问从未被检验**（没造出测试床），这里记录的是这条路探到哪、为什么停、以及探到的机制信息。

---

## 导读（先读这段；下面的 regime screen → candidate confirm → "NULL" 是走到这里的证据路径，其 pass/fail/NULL 措辞已被本文最后的 reframe 段撤回）

**测了什么** —— 在一张 cm 尺度脉冲网络里两端各放一块"容易自己点着的病灶组织"，只靠背景噪声让它们自发点火，把 621 个自发群体事件逐个贴标签（哪头先点着=源 / 有没有传开到够多虚拟电极触点能读出方向 / 点火多旺 / 持续多久）。原本想看：两头能不能各自独立点火、产出一条"两套相反模板交替、且交替顺序与标签无关"的长记录（对应真实间期 HFO 的正反模板现象）。

**怎么测的** —— 先扫遍工作点（核间距 / 异质宽度 / 平均门槛 / 背景驱动 / 错相位）找"两头都干净自发、低碰撞、平衡双向"的区；没找到 → 围最像的候选格做更长确认；都不成 → 改问"为什么多数事件是局部的"，把每个事件的源核点火强度 vs 另一核、局部小事件 vs 可读大事件的持续时间/扩散逐个比。

**揭示了什么** —— ① 两头确实**都能各自成核**（源层面），但"平不平衡"是**逐个工作点不一样的，不是一个锁定的平衡双向区**；② 局部事件**不是没点旺**（点火强度 0.328 ≈ 可读大事件 0.324），而是**点得很旺却传不远**（"点着了但传不动"，contained / relay-failure）——区分局部↔全局的是**持续时间 + 扩散范围，不是成核能量**；③ 能传开的少数大事件**带方向但不对称**（负端起的几乎全读成一个方向，正端起的方向读出一半一半）。**这支持**"正反模板可能来自同一条轴两端的随机成核"（源层面直接支持），**不支持**"平衡、独立、可长时序的双源列车"。**局部事件是机制信息，不是失败** —— "很多小局部事件 + 少数模板化大事件"这个层级本身比强行的平衡双向列车更像真实 HFO。

**当前定位**：Stage 3 整条线 = 建模 Step 4（spiking 层）的一个探索性观测层结果，**回落到 Stage 2 已证的结构层**（两套相反模板可被真实 masked pipeline 复现 + 端点互换，见 `snn_cm_spontaneous_bidirectional_2026-06-11.md`）；Stage 3 的 timing 层不写主结论。

（内部归档代号：`twoend_equal` lesion / `sep_frac`·`core_std`·`core_mean`·`drive` 网格 / `pilot_gate` / hidden source 1%-core-onset 标签 / `readable_global`·`local`·`collision`·`readable_unknown_source` / `source_core_ignite_frac`·`core_ignite_asymmetry` / candidate `sep0.7/std1.0/m17.5`）

---

## What was screened

`twoend_equal` grid: core-spacing `sep_frac{0.6,0.7,0.8}` × spread `core_std{0.5,1.0,1.5}` × `core_mean{17.0,17.5}` × 3 seeds (54 runs, T=3000) + Phase 0 sign calibration via `oneend` (known source). Question: is there a working point with a BALANCED bidirectional regime (both ends fire their own events in one network, low collision)?

## Findings

**Phase 0 sign calibration — clean.** `oneend_neg` → all forward (+1) [13/0, 14/0]; `oneend_pos` → all reverse (−1) [0/13, 0/14]. So the read-out sign tracks the known source end deterministically. The earlier "direction/source decoupling" worry is retired — sign is sound.

**No gate passed.** Strict direction gate (`n_fwd≥3 AND n_rev≥3 AND collision_rate<0.30`) = **0/54**. Hidden-source gate (`neg_clean≥3 AND pos_clean≥3`) = **0/54**. Spec gates are far off: pilot needs ≥30/end; formal needs ≥50/end AND total clean ≥100. Nothing here is near formal.

**Source/sign agreement audit (the hidden-source label is regime-sensitive, NOT globally invalid).** On clean hidden-source events, hidden label vs sign-implied source agree:
- overall **39/53 = 74%**
- **sep0.6 = 2/16 = 12%** (disagrees, concentrated at sep0.6 std1.0/1.5 m17.0 = 0/6, 0/6)
- **sep0.7 = 30/30 = 100%**, **sep0.8 = 7/7 = 100%**

So `sign` is a NECESSARY cross-check, not a global source ground truth; and the 1%-core-onset hidden label must NOT be blanket-discarded — it needs threshold/regime sensitivity auditing (this table is that audit). It is trustworthy at sep0.7/0.8 and the candidate cell, untrustworthy at sep0.6 std1.0/1.5 m17.0.

## Frozen conclusions

1. **Old op-point (sep0.6 / drive0.6 / m17.0) = NO-GO** (one-end dominance + label disagreement).
2. **The only priority candidate = `sep0.7 / std1.0 / m17.5`**: 3 seeds all `collision_rate=0`, hidden↔sign agreement 15/15=100%, counts ~3/2, 3/2, 4/1 (forward:reverse). **This is a candidate, NOT a victory.** Evidence is thin (3 seeds, ~3:2, ~5 clean directional events/run at T=3000).
3. Allowed statement: *"found a low-collision, dual-end-nucleation-capable candidate working point; not yet a balanced/formal regime."* Forbidden: *"Stage 3 reproduces balanced bidirectional templates."*

## Next (gated)

- **Short confirmation around the candidate** (`sep0.7/std1.0/m17.5`, longer T and/or more seeds): goal is NOT a formal claim, but to estimate the per-end clean rate and whether the ~3:2 imbalance is stable. RAM-safe (one cell, low parallelism; 13GB/sim, ≤14 concurrent on the 251GB box).
- **Formal long run gate:** enter Stage 3 timing / positive-controls ONLY when a single run extrapolates to ≥50/end, collision <20–30%, AND source-cluster purity is measurable. None established yet.

---

## Addendum 2026-06-14 — drive axis + deph fallback folded in (gate-as-code)

Two more arms were folded into the same gate (`src.sef_hfo_stage3.pilot_gate`, unit-tested `tests/test_stage3_gate.py`; combined aggregator `aggregate_stage3_regime_screen.py` → `regime_screen_cells.csv`). **Combined now 92 runs / 23 cells / 0 pass.** Both arms kill *other* regions; **neither touches the `sep0.7/std1.0/m17.5` candidate.**

- **Drive axis** (collaborator `twoend_sweep`: `drive{0.40,0.45} × core_mean{17.0,17.5}`, 31 runs, folded in): every cell → **`no_events`** (median ~0 events; even `drive0.45/m17.5` only ~2.5 events, all ambiguous). Lowering background drive does not de-collide the two ends — it just makes the network **too cold to ignite at all**. So the failure is bounded on *both* sides: default drive 0.6 → co-ignition; drive ≤0.45 → silence.
- **Deph fallback** (`twoend_deph`, 0.3 mV, seeds 1–3 — the 3 runs the original screen left incomplete, now run): collision 0.20 / 0.43 / 1.00 (median 0.43), `neg_clean = pos_clean = 0` for all three seeds, all-forward → fails **`high_collision`** + no clean single-source events. A 0.3 mV dephase is **too weak to decouple** the two ends on this coupled sheet; it does NOT recover a balanced regime (it is not the Stage-2 win, because Stage 2 ran the two ends in *separate* networks and interleaved, not as two coupled foci in one).

**Net (reconciled with §3 above):** everything except the one candidate is now triply dead (co-ignition at default drive, silence at low drive, collision under dephase). The `sep0.7/std1.0/m17.5` candidate (`pilot_gate` reason = `source_imbalance`, i.e. fails the balanced ≥3/≥3 bar by one +end event, but 0 collision + 100% hidden↔sign agreement) is the ONLY surviving lead. **Correct next step is the §"Next" short candidate confirmation (longer T / more seeds), NOT a design change yet** — the design fork (revert to Stage-2 separate-network interleave / decouple harder / drop the timing layer) is only forced if the candidate confirmation also fails.

*(internal: regime_screen_raw.csv + regime_screen_cells.csv; gs_sign/gs_te/gs_deph + twoend_sweep drive arm; pilot_gate reasons no_events/high_collision/source_imbalance/unidirectional; n_fwd/n_rev=sign; neg_clean/pos_clean=1%-core-onset hidden label; candidate sep0.7/std1.0/m17.5.)*

---

## Candidate confirmation 2026-06-14 — `sep0.7/std1.0/m17.5` does NOT survive (no-go)

Ran the candidate at **T=6000, 6 fresh seeds** (`candidate_confirm/`, RAM-safe) to estimate per-end clean rate + imbalance stability (NOT a formal claim). Result: the candidate fails — but for read-out + source-dominance reasons, NOT "the cores stop firing" (mechanism corrected 2026-06-14 after a diagnostic re-read; see the corrected bullet below).

- **Readable directional rate is too low** (median sign-based n_fwd/n_rev = **1.5 / 1.0** per 6 s ≈ 0.25/s fwd, 0.17/s rev → **≥50 readable/end needs T ≈ 200 s / 300 s** = hours per run). **BUT this is the read-out-limited rate, not the source-event rate** — see the corrected mechanism below. Longer T did not raise the readable rate; it raised the *unreadable small-event* fraction.
- **CORRECTED mechanism — `neg_clean=pos_clean=0` is a READ-OUT bottleneck, not a source-dynamics failure.** The 66% "ambiguous" is overwhelmingly `unreadable_axis` (**69/83 events**), NOT `no_core_crossing` (**1/83**). The cores DO ignite cleanly and singly — pooled **30 neg + 38 pos** clean single-core onsets across the 6 seeds (e.g. `None/159`=clean pos, `859/None`=clean neg, alternating). But events are **tiny: `n_part` median = 2 contacts** (only 2–4 of 12 light per event; the endpoint-centroid axis needs ≥7), so `clean_for_timing` (which requires a readable axis) drops them. Mechanistically: at the colder `m17.5` op-point events stay **local** and don't propagate far enough to light ≥7 contacts → low collision (good) but unreadable (the failure). So `neg_clean=pos_clean=0` means **this 2-shaft virtual SEEG cannot see these small local events**, not that the cores are silent.
- **At the source level (bypassing the read-out) it is STILL not a balanced independent train.** Using core onsets directly, each seed is **one-end-dominant** with **which-end flipping across seeds** (P-dominant 1/3/4/6, N-dominant 2/5; per-seed ratios up to 11:3). So even the ground-truth source labels show the broad screen's same one-end-dominance — the candidate is no-go at BOTH layers: read-out (events unreadable) AND source dynamics (one end dominates, flipping by seed).
- **`pilot_gate(median)` = FAIL, reason `source_imbalance`** (collision_ok True 0.15, enough_events True, bidir_ok True, but source_balance_ok False; ambiguous_rate 0.66).

**Frozen verdict:** the candidate is no-go. With the screen's other regions already triply-dead (co-ignition at default drive, silence at low drive, collision under dephase) and this last low-collision lead now failing confirmation, **two equal coupled foci in ONE cm-SNN do not yield a usable balanced bidirectional spontaneous regime at a feasible event rate anywhere in the screened box.** The Stage-3 primary question (one-network label-timing independence) is **not tractable on this substrate as posed** → the design fork is now forced (it was gated on "only if the candidate confirmation also fails"). Options: (a) document the NULL + fall back to the Stage-2 structure result (one-network timing layer not testable here); (b) decouple harder / change substrate (weaker coupling, two sub-networks, larger separation); (c) drop the timing layer. No main-doc claim either way. **Not a Stage 3 pass; not a failure of the model — a bounded, characterized NULL of this specific construction.**

---

## Reframe + event-typing (2026-06-14, SUPERSEDES the "NULL / no-go / pass-fail" wording above)

The "NULL / no-go / pass-fail" language above overstates it and is retired. This is **not** a tested-hypothesis null (the timing-independence test was never run — we couldn't build its test bed) and **not** a model failure. Correct framing: **an exploratory, read-out-limited result of the one-network two-foci construction.** Stage 2 (two SINGLE foci run SEPARATELY, clean global events filtered, then pooled) and Stage 3 (two foci COUPLED in one network, spontaneous) are **different scientific questions**, not a natural extension — Stage 3 adds competition / co-ignition / refractoriness / local events / observation sparsity.

**Event typing (`scripts/analyze_stage3_event_types.py` → `event_types.csv` + `event_type_by_cell.csv` + `event_type_by_seed.csv` + `readable_global_events.csv`, `figures/stage3_event_typing.png`; 621 events across 60 two-foci runs) answers WHY most events are local. Pooled propagation_class: local 377 / collision 187 / readable_global 54 / readable_unknown_source 3.**

- **local ≠ weak nucleation — and the source core fires the SAME on local and global.** P1-3 split `core_ignite_frac` into the **source** core vs the **other** core. On local events (source∈{neg,pos}, n=347): the **source core hits 0.328** active-fraction while the **other core stays 0.000** (asymmetry median **0.328**) — i.e. a strong, sharply ONE-SIDED ignition that simply does not spread. Crucially, readable_global events fire the source core at **0.324** — **statistically identical** to local. So whether an event becomes a readable template is **NOT** about ignition energy.
- **What separates global from local is DURATION + SPREAD, not energy.** Median `duration` local **24 ms** vs global **66 ms** (≈2.7×); `n_part` local 3 vs global 8. Same ignition strength, but the global ones SUSTAIN and RELAY outward while the local ones do not. This is **contained propagation / relay failure** (the wave ignites hard at one end but the cool m17.5 surround does not carry it the length of the axis) — we deliberately retire the word "truncation": we have not yet shown a wavefront that travels outward and then stops (that needs a spatiotemporal snapshot, see plan); we have shown ignition-without-spread.
- **Refractory-shadow is NOT the main driver:** 1/377 local events fall within 250 ms after a readable_global; local inter-event gap (median 268 ms) ≈ global (283 ms).
- **Source balance is PER CELL, not a single pooled "balanced regime."** Pooled across 19 cells the source mix looks even (neg 187 / pos 214 / both 187 / none 33), **but this average hides wide per-cell variation** (`event_type_by_cell.csv`): some cells are all-collision (sep0.6/std1.0/m17.5 = both 21/22), some one-end-dominant (sep0.7/std1.0/m17.0 = pos 19 / neg 8), and the candidate's own longer-T confirmation (conf:sep0.7/std1.0/m17.5, T=6000) is pos-leaning (neg 41 / pos 56). **So at the source layer two-end nucleation genuinely occurs, but it is NOT a locked balanced regime — it is regime/cell-dependent.**
- **The readable templates DO carry source→direction structure, asymmetrically.** Among the 54 readable_global, neg-source → forward(+1) holds **31/35 = 89%** (consistent with the oneend sign calibration), but pos-source → reverse(−1) is only **9/19 = 47%** (≈coin flip) — the pos-end read-out is direction-ambiguous (likely the still-present neg core biases the endpoint-centroid axis). readable_global concentrate in sep0.7/std1.0/m17.5 (15, T=3000 screen) and sep0.7/std0.5/m17.0 (10); **the candidate's longer T=6000 confirmation produced only 1**, confirming the screen's apparent richness was a short-T slice.

**Scientific statement (locked):** two-end excitable foci in one cm-SNN **do produce two-end nucleation (per-cell, not a single balanced regime) and a few readable forward/reverse propagation events**; but under the current coupling / drive / virtual-electrode / gate, **most spontaneous events stay local — strong one-sided ignition, contained propagation — and hotter/closer conditions turn them into co-ignition/collision.** The local→global distinction is set by **propagation (duration + spread), not nucleation energy**, and the readable templates carry source→direction structure on the neg end but ambiguously on the pos end. This **supports the direction** "forward/reverse templates may arise from random nucleation at the two ends of one propagation axis" (supported at the source level), but does **NOT support** "a balanced, independent, long-timeseries two-source train." **Local events are mechanism information, not failure** — a "many small contained events + a few template-able large events" hierarchy, itself closer to real HFO than a forced balanced bidirectional train. Per-event + per-cell + per-seed fields available for re-analysis in the four CSVs (source / readout sign·n_part·axis_err / propagation_class / source_core·other_core·asymmetry ignition / time_since_prev / previous_source / within_recovery / core_onset_diff / duration; per-cell P(global|neg)·P(global|pos)).

---

## Step 2/3 局部→全局层级定量（2026-06-14, review-corrected）

**测了什么** —— 上面已经把每次自发放电分成了"只在一头点着、传不开的小事件"和"少数传开到够多虚拟电极触点、能读出方向的大事件"。这一步不是去问"两套相反的行波模板存不存在"（那个主问的测试床我们一直没造出来），而是把这两档摆在一起，逐项量一下它们到底差在哪：差在点火点得旺不旺，还是差在传得远不远、撑得久不久。

**怎么测的** —— 第一步（分源层级量化）：把"小局部"和"少数可读大"两档放在一起比三件事 —— 撑多久、点亮几个触点、起点那块组织点得多旺。关键是**必须按"哪头先点着"分开比**，因为合在一起看像"两档点火一样旺"，分开看其实两头方向相反。如果点火能量真是无差别的，那么按起点端分开后两档应当各自重叠；实测发现负端起的小事件比大事件点得还旺，正端起的反过来 —— 这正说明"点火能量两档相等"是合并假象。第二步（可读大事件的描述性模板复核）：先把那 54 个可读大事件逐个对回它在记录文件里的精确那一列（验证列映射），再用和真实病人完全相同的去伪聚类机器，看它们能不能聚成"两套相反模板"。但这一步只当**描述性核对**，不当主问的答案。

**揭示了什么** —— ① 区分小局部和可读大事件的是**传播（撑得久 + 传得广），不是成核能量**：撑的时长局部约 22 ms、全局约 67 ms（≈3×），点亮触点局部 3 个、全局 8 个，两档差距大且方向明确；点火能量则是**分源相反、不是统一相等**（负端起的小事件比大事件点得旺，正端起的反过来）。② 那 54 个可读大事件做描述性模板核对时，按起点端分开后**每端可区分的形状种类太少**（负端 35 个事件里只有 6 种不同形状、正端 19 个里只有 7 种），所以即便聚类机械上聚出了"两类"，也**不能**把它读成"找到了稳定的两套相反模板"——只能记成"数据太少（形态多样性不足）"。③ 因此小局部事件是"点得旺却传不远"的**受限传播 / 接力失败**（contained / relay-failure），是机制信息不是失败。

（内部归档代号：Step 2 = 分源 `local` vs `readable_global` 效应量；`source∈{neg,pos}` 分面 + bootstrap 95% CI + label-shuffle permutation；`duration` / `n_part` / `source_core_ignite_frac` / `sign` 四面板；Step 3 = `readable_global` masked 模板复核，复用 `compute_adaptive_cluster_stereotypy(use_masked_features=True)` + `compute_swap_score_sweep`；`n_unique_masked_patterns` / `diversity_limited` / `swap_class` / `decision_k` / `chosen_k`）

### (a) Step 2 分源效应量（C1 —— duration/n_part 是真正分界，source_core 分源不对称）

脚本 `scripts/plot_stage3_local_global_hierarchy.py` → 图 `figures/stage3_local_global_hierarchy.png` + `stage3_hierarchy_effect_sizes.json`。每个 (source × class) 报 median + bootstrap 95% CI；每 source 内做 local-vs-global 的 label-shuffle permutation（B=5000，descriptive p，**不是 gated test**）。

| 量 | source | local median | global median | Δ (global−local) | perm p | n_local / n_global |
| --- | --- | --- | --- | --- | --- | --- |
| `duration` (ms) | neg | 22.0 | 67.0 | **+45.0** | **0.0002** | 152 / 35 |
| `duration` (ms) | pos | 26.0 | 25.0 | −1.0 | 1.0 | 195 / 19 |
| `n_part` (/12) | neg | 3.0 | 8.0 | **+5.0** | **0.0002** | — |
| `n_part` (/12) | pos | 3.0 | 8.0 | **+5.0** | **0.0002** | — |
| `source_core_ignite_frac` | neg | 0.3659 (CI[0.3339,0.3786]) | 0.3146 (CI[0.3087,0.3373]) | **−0.0513** | 0.04 | — |
| `source_core_ignite_frac` | pos | 0.2777 (CI[0.2261,0.3134]) | 0.3344 (CI[0.2933,0.3656]) | **+0.0567** | 0.4115 | — |

`sign × source`（54 个 `readable_global`）：neg → forward 31 / reverse 4（89% 顺向）；pos → forward 10 / reverse 9（≈抛硬币）。

**读法（C1）**：`n_part` 在两源上都是 local 3 / global 8、干净显著（这是两档的真正分界）；`duration` 在 neg 源大幅拉开（+45 ms, p=0.0002）。而 `source_core_ignite_frac` 是**源-不对称**的 —— neg 源 local **高于** global（Δ=−0.0513），pos 源 local **低于** global（Δ=+0.0567，且 p=0.41 不显著），两源方向相反。所以**允许**写"local 不是弱点火"；**禁止**写成"所有源上点火能量完全相等"（合并掉源就会得到这个假象）。C1 directional sanity 已确认：neg Δ<0（local>global）、pos Δ>0（local<global）。

### (b) Step 3 三层 × 三档判定（C2/C9 —— swap_class 是描述性复核，非 Stage 3 证据）

脚本 `scripts/analyze_stage3_readable_templates.py`（先 dry-run 焊死 `event_id → lagPat 列` 映射，再 masked 三层聚类）→ `stage3_readable_templates_summary.json`。

**Dry-run（列映射门，已过）**：54 个 readable_global、source `{neg:35, pos:19}`、sign `{1.0:41, -1.0:13}`；全部 54 事件精确对回 record 列，montage 一致（12 触点）。两个有重复 record 的 tag（`gs_te_sep0.6_std1.0_m17.0_s1`、`gs_te_sep0.7_std1.0_m17.5_s1`）经 **sidecar→同级 record** 唯一解析消歧（**不是**全局 glob 取第一个），见 C7。

| 层 | n_events | n_unique_masked_patterns | chosen_k (stable_k) | swap_class（描述性） / decision_k | diversity_limited | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| pooled_sanity | 54 | 13 | 2 (=2) | strict / 2 | False | 可复核（仅 sanity；混了 cell/seed/source，不作模板结论） |
| source_neg | 35 | **6** | 2 (=2) | strict / 3 | **True** | 数据太少（形态多样性不足: n_unique=6） |
| source_pos | 19 | **7** | 5 (=5) | None / None | **True** | 数据太少（形态多样性不足: n_unique=7） |
| top_cell `sep0.7/std1.0/m17.5` | 15 | **3** | 2 (=2) | strict / 2 | **True** | 数据太少（形态多样性不足: n_unique=3） |

**读法（C2/C9）**：`swap_class` 一律只是**描述性模板复核**，**不是** Stage 3 证据、**不是**主问的答案（JSON 顶层已注：「swap_class 是描述性模板复核，不是 Stage 3 证据」）。两个 source 分层的可区分形状都太少（neg 6/35、pos 7/19，均 `diversity_limited=True`），因此它们机械聚出的 `stable_k` **不能**读成"找到稳定双模板"——这是挡 stable_k 过度解释的硬闸（C9）。pooled 层 distinct=13≥10 形式上进"可复核"，但混了 cell/seed/source，仅作 sanity，不作模板结论。整段无任何一层被标 PASS（grep clean）。

### (c) baseline 对照（C4）

相对 Stage 2（两个**单灶分别**在独立网络里跑、过滤出干净全局事件后再合并）已经能用真实 masked pipeline 复现两套相反模板 + 端点互换（见 `snn_cm_spontaneous_bidirectional_2026-06-11.md`），本步的两灶**耦合在同一张网络里自发**是不同的科学问题，多了竞争 / 共点火 / 不应期 / 局部事件 / 观测稀疏。Phase 0 的 oneend sign calibration（已知源）干净（neg→全顺向、pos→全逆向），所以读出符号本身可信；而本步的 hot/collision 条件（更近/更热的 cell 转成共点火、碰撞）说明"传不开"的局部档与"两头同时点着"的碰撞档是同一耦合机制的两端。

### (d) 科学边界（C4）

可读大事件与局部事件成核能量相同、只在传播上分开 —— 这是**受限传播 / 接力失败**（点火强、不扩散）。**未证明波前先向外走再停下被截断**（那需要事件级时空快照，见 plan）；只能说"点着了但没扩散开"。

### (e) Step 4 重跑门记录（C3 —— 默认不开仿真）

重跑门 = **每个 source ≥20 readable_global 且每个 source 跨 ≥2 seeds**。复核（`readable_global_events.csv` 按 cell×source 分组）：**当前 0 个 cell 达标**（20 个 cell-source 组合全部不足；最好的 `sep0.7/std1.0/m17.5` 也只有 neg 10（3 seeds）/ pos 5（3 seeds），事件数远不及 20）→ **不开仿真**。只有 Step 3 明确显示"事件数不足但形态有信号"（某 source 层 chosen_k==2 且 swap 方向清晰但 n 未及阈值）才重提；本步三个 source/cell 分层均 `diversity_limited=True`，无此情形，门维持关闭（C6：不在本计划内自动触发任何长仿真）。

**tier 纪律（重申）**：Stage 3 主问（单网络标签-时序独立性）= 未检验；本步 = 探索性机制刻画，进 archive、**不进**建模主文档主结论。

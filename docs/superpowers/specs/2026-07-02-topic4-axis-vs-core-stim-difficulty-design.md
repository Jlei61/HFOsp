# Axis-vs-core stimulation across substrate situations + the "why spontaneous can't self-train" difficulty figure — Design

**Date:** 2026-07-02
**Topic:** Topic 4 (SEF-HFO / cm-SNN), M3A-v2.2 q_I/g_K slow-variable line
**Status:** design (brainstormed, pending user review → writing-plans)
**Supersedes/redirects:** the Phase-2 half of `docs/superpowers/plans/2026-07-02-stage4-v2-spontaneous-qI-stim.md` (a spontaneous self-generated train does NOT exist — 12/12 + 6/6 burst; see `docs/archive/topic4/stage4_v2_workpoint_2026-07-02.md`), so the stim story pivots from "GIF on a self-made train" to "where to stimulate: axis vs core, across situations."

---

## Abstract (第一性原理)

**核心要讲的一句话（临床落点）**：一块会发作的脑组织，有"灶"(发作起始的源)和"轴"(活动往外传的必经窄通道)。我们要用模型说明——**在固定电极预算下，把刺激挡在"轴"上，至少不弱于打在"灶"上**；而且这在不同灶形态下都成立。临床含义：不一定要精确命中发作灶，挡住传播通道一样管用（甚至更省电极）。

**为什么先要一张"铺垫图"**：读者会问"那为什么不直接让一个自发的灶自己发作、就地做刺激？" 铺垫图诚实回答：一块**大而均匀**的灶自发时只有两种命运——**要么整片同时炸(太热)、要么哑火(太凉)**，中间那条"一串分开的小事件慢慢累积到发作"的路**不存在**；把灶**缩小**也不行(局部点火了，但前锋照样铺满整片、停不下来)。根子是一个矛盾：**要它自己点着就得"易燃到能自燃"，可一旦易燃到能自燃，点着后就铺满停不下来**。所以模型里的离散事件串必须**外部戳**出来(kick)——这也解释了为什么刺激实验用 kick 驱动的底物才有"一串"可谈。

**怎么测"轴≥灶"（固定预算的公平对比）**：两种刺激**用同样多的电极**。打灶=把电极压在源上(但源通常比预算大、盖不全，剩下的源照样点着)；打轴=同样多电极压在下游那条窄通道上(窄到能整条堵死)。比谁把发作(runaway)推得更晚 / 干脆在窗口内拦住。**"轴≥灶"能成立的前提正是"灶盖不全、轴堵得死"** —— 这就是为什么小核也必须**盖住足够多的电极**(大到一撮 footprint 盖不全源、又留出下游可堵的轴)。

**诚实边界**：这是模型内的**机制/效率示意**(visual + 单条轨迹 + 小扫描)，不是"电刺激治发作"的临床证明；runaway/tonic 不是真的 ictal 事件。"轴≥灶"在**多源/单一咽喉**几何(E1146 两灶+中段走廊)里明确成立；在**单个中心灶**(径向漏、无单一咽喉)里是**诚实检验**——可能只是"打平"甚至"轴略弱"，那也如实报告，不硬拗。

（内部归档代号：M3A-v2.2 `SpatialSlowField` q_I/g_K；`_build_stage4_patch`；`intervention_vth_at_time` V_th clamp；E1146 `fig_m3a_v2_2_qI_stim_*` endpoint_vs_middle=+414/+834、both_foci=+848；Stage-4 v2 fast-gate 12/12 + small-core scan 6/6 one_shot_burst）

---

## 1. Goal & claim scope

**Primary deliverable (Figure B):** a paper-grade figure showing, across ≥2 substrate situations, that **at a fixed stimulation footprint, blocking the propagation axis delays runaway at least as much as stimulating the core** — with the honest caveat that the strong form holds in the multi-source/chokepoint geometry and is *tested* (not assumed) in the single-core geometry.

**Supporting deliverable (Figure A):** a 3-row difficulty figure establishing *why* a spontaneous single focus cannot self-generate the discrete-event train (so the stimulation experiments necessarily ride on either kick-driven trains or single fill events) — the self-ignite↔self-terminate tension.

**Claim tiers (locked):**
- **Established, reuse:** E1146 kick two-foci — middle/axis stim (+834 ms) ≥ endpoint/core stim (+414 ms), ≈ both-foci (+848 ms) at 2× footprint. (Existing figures/metadata.)
- **Primary new test:** small central core (r≈3) — axis-stim vs core-stim at fixed footprint. Result reported honestly; the cross-situation claim is "axis ≥ core holds where there is a shared chokepoint; single-core is a stress test."
- **Descriptive:** Figure A regime characterization (synchronous blast / fill / train).

**Forbidden language:** "proves seizure mechanism", "electrical stimulation treats seizures", "closed-loop/recovery". Runaway/tonic ≠ ictal event. Everything is within-model, single-trajectory + small screen.

---

## 2. Background (what is fixed going in)

- **Substrate = canonical Stage-4 spontaneous runner** (`run_sef_hfo_snn_cm_spontaneous_readout.py:520-525`): `g=3.6`, `AR=2.0`, `theta=45°`, `density=100`, `L=20`, `drive=0.6`. Anisotropic E→E (AR=2 along 45°) → spread is elongated along `axis_unit=[cos45,sin45]`, i.e. a dominant axial direction (not isotropic) — this is what makes an "axis" meaningful.
- **Slow variables:** q_I (slow across-event depletion), g_K (fast per-event fatigue brake, coupled `eta_K>0`).
- **Established negatives (this session):** big core (r=6) → synchronous blast 23–32 ms, 12/12; small central core (r=2–4) → focal ignition but front fills the sheet, 6/6 burst (runaway later for smaller/cooler: r=2 cool 50 ms → r=4 hot 21 ms). g_K (even eta_K=0.8/tau_K=150) never contains the front. `results/topic4_sef_hfo/stage4_v2_workpoint/{screen_fast,scan_small_core}.json`.
- **Established stim result (E1146, reuse):** `results/paper-ready-figure/fig_m3a_v2_2_qI_stim_{site_compare,both_foci}_*` — middle +834, endpoint +414, both-foci +848; parity-tested `intervention_vth_at_time`.

---

## 3. Substrate & montage design

### 3.1 Situations
| id | substrate | build | ignition | why in the story |
|----|-----------|-------|----------|------------------|
| `big` | single r=6 core, spontaneous | `_build_stage4_patch(core_radius=6)` | synchronous whole-disk blast | Fig A row 1; NOT used for stim (core covers ~all contacts → no distinct axis) |
| `small` | single r=3 core, spontaneous | `_build_stage4_patch(core_radius=3)` | focal → fills along axis | Fig A row 2 **and** Fig B situation (ii) |
| `kick` | E1146 two foci, kicked | `_build_subject1146` + pulse schedule | discrete train → runaway | Fig A row 3 **and** Fig B situation (i) — reuse existing |

### 3.2 Virtual-SEEG montage for the single-core situations (`small`, `big`)
Synthetic linear montage of `n_contacts=11` along `axis_unit` through the sheet centre:
`contact_i = center + (i - (n-1)/2) * pitch * axis_unit`, `pitch=1.2 mm` (spans ±6 mm).
- **source contacts** = contacts within `core_radius` of centre. For r=3 / pitch 1.2 → indices {3,4,5,6,7} (5 contacts). **Must be ≥ 5** (this is the "覆盖足够多的电极" gate — a fixed footprint can't cover all 5).
- **downstream/axis contacts** = the rest {0,1,2,8,9,10} (6 contacts), symmetric along the two axial fronts.
- E1146 keeps its own real montage (ICL1-11) from figdata.

### 3.3 Stim targets (E-cell masks; reuse `Q._electrode_e_mask`)
- **core-stim:** clamp E cells within `r_stim=2.0 mm` of the **N source contacts nearest the centre** (partial cover of the source — must leave residual).
- **axis-stim:** clamp E cells within `r_stim=2.0 mm` of **N downstream contacts split symmetrically** (N/2 nearest-downstream each side) to block both axial fronts.
- **Fixed footprint N (even), identical for both arms**, with the contract `N < n_source_contacts` so core-stim leaves residual source. Exact `N` (candidate: 2 for a large visible residual, or 4) is locked in the plan; equality across arms is asserted. Tie-breaking on contact selection is deterministic (nearest-by-distance, lower index first) and documented in the plan.
- E1146: reuse existing `_stim_site_center`/`_select_middle_contacts`/`_select_both_foci_contacts` (core=endpoint, axis=middle corridor); footprints already matched (4 contacts each in the site_compare figure).

---

## 4. The two figures

### 4.1 Figure A — difficulty (3 rows × 2 cols), fig4 visual grammar
Rows: `big` / `small` / `kick`. Columns:
- **col1 空间点火 (source-space onset-time, viridis 早紫→晚黄):** per-E-cell time of first spike. `big`: near-uniform colour (synchronous, no gradient). `small`: centre→edge gradient **reaching the sheet boundary** (fills). `kick`: a representative evoked event that **does not reach the boundary** (contained). Core outline (red) + montage contacts (orange/cyan) annotated.
- **col2 时间 (rate + q_I + g_K):** left y = E-rate (Hz), right y = q_I & g_K (0–1). `big`/`small`: one spike → runaway wall; q_I one-step to floor; g_K ≈ 0 at the blast. `kick`: a train of bumps → q_I staircase → runaway; g_K accumulates across the train.
- **Punchline** in title/footer: self-ignite↔self-terminate tension; shrinking only delays the fill; kick supplies the train externally.

### 4.2 Figure B — axis ≥ core across situations
Per situation (rows: `kick`, `small`), columns:
- **col1 底物 + 两种刺激位点 (source space):** substrate + core outline + montage; **core-stim contacts** and **axis-stim contacts** highlighted in two distinct colours (same footprint N shown).
- **col2 runaway 推迟对比:** for {no-stim, core-stim, axis-stim} overlay the E-rate (or a compact bar of `runaway_delay_ms`), annotated with the delay each achieves. Shared legend.
- **Punchline:** at fixed footprint N, `delay(axis) ≥ delay(core)` where it holds; honest per-situation numbers.

Both figures: paper-grade self-contained (no §X / cluster_id / bracketed axis labels), shared legend, tight coords, plasma substrate / viridis time / orange(A)-cyan(B) contacts. Output `results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/` (png+pdf+metadata+README 中文). Script(s) under `scripts/paper_figures/`.

---

## 5. Stimulation protocol & fairness contract (LOCKED)

- **Parity:** both arms share substrate/seed/drive; stim only changes the V_th comparison via the parity-tested `intervention_vth_at_time` (no extra RNG) → byte-identical until `stim_on`.
- **Fixed footprint:** `n_contacts(core-stim) == n_contacts(axis-stim) == N`; assert in code. (E-cell counts under each mask reported in metadata.)
- **Core not fully coverable:** `N < n_source_contacts` (so core-stim leaves residual source). Assert.
- **Window:** `[stim_on, stim_off)` starts before the no-stim runaway onset and is long enough to show hold + post-release rebound. Single-core: `on≈0`, `off` tuned so the danger zone (< baseline ~50 ms) is covered and rebound is visible within `T`. Kick: reuse the existing `[500,1400]` window. Exact values locked in the plan.
- **Metric:** `runaway_delay_ms = (runaway_stim or T) - runaway_nostim`, using the shared runaway criterion (`_smooth_rate` 20 ms + `_first_sustained` 120 Hz/100 ms). "Held for the whole window, ran away only after release" and "prevented within T" are reported as such.

---

## 6. Acceptance gates (numeric — encode the conclusions, not just existence)

**Figure A (regime):**
- G-A1 `big`: `n_events==1` AND `runaway_onset < 60 ms` (single fast blast). [screen: 23–32 ✓]
- G-A2 `small`: `n_events==1` AND `max_active_fraction > 0.5` (front fills ≥ half the sheet) AND runaway present. [scan ✓]
- G-A3 `kick`: `n_events ≥ 3` before runaway (a genuine train). [existing qI figure ✓]
- G-A4 (source-space, big vs small): fraction of core E cells igniting within a 10 ms window `big > small` (big more synchronous); `small` onset gradient reaches within `pitch` of a boundary contact (fills). Report values.

**Figure B (the core claim):**
- G-B1 **fairness:** `n_contacts(core)==n_contacts(axis)==N` (assert). Bad-data regression: unequal N must raise.
- G-B2 **core-partial:** `N < n_source_contacts` (assert; else the single-source comparison is trivial).
- G-B3 **parity:** two arms byte-identical until `stim_on` (test: `E_spk_bool` equal on `[0, stim_on)`).
- G-B4 **effect present:** `delay(core) > 0` AND `delay(axis) > 0` (both do something). Bad-data regression: empty-mask stim → `delay == 0`; full-core clamp → runaway prevented within `T`.
- G-B5 **claim (reported, not asserted-true):** `delay(axis) ≥ delay(core) - tol` with `tol` locked (e.g. 10 ms). For `kick` this is expected PASS (existing +834 ≥ +414). For `small` it is a **test**: PASS / TIE / FAIL all reported honestly in metadata + README; a FAIL is a scientific result ("single radially-spreading core has no shared chokepoint"), not a bug.

All load-bearing params (`core_radius` per situation, `n_contacts`, `pitch`, `r_stim`, `N`, `stim_on/off`, `T`, `RUNAWAY_HZ/DUR`) are **locked in the implementation plan** and echoed in figure metadata.

---

## 7. Data & machinery reuse (do not re-invent)

- Builds: `H._build_stage4_patch(core_radius=…)` (exists), `H._build_subject1146` (exists).
- Sim + stim: `H._simulate_continuous(..., vth=S["patch_vth"], stim_target=mask, stim_on, stim_off, record_gif=True)` (exists; parity-tested).
- Stim helpers: `Q._electrode_e_mask`, `Q._select_middle_contacts`, `Q._select_both_foci_contacts`, `Q._stim_site_center` (exist for E1146; add a single-core montage + source/axis contact selector).
- Runaway criterion: `H._smooth_rate` + `H._first_sustained` (exist).
- Onset-time field / activity field for col1: derive from `res["E_spk_bool"]` (first-spike time per E cell); reuse `Q._activity_fields`/`_zlfp` patterns.
- E1146 stim delays: reuse existing metadata JSONs (no re-run) OR re-run for a common code path — plan decides (default: reuse to save cost).
- New code: single-core linear montage builder + source/axis contact split; the two figure renderers; TDD for the montage/contact-split + fairness asserts + parity.

---

## 8. Cost & checkpoints

- Fig A renders: `big` 1 run (~T=300, ~5–8 min), `small` 1 run (~T=600, ~5–10 min), `kick` reuse-or-rerun (~10–15 min if rerun). 
- Fig B: `small` core-stim + axis-stim + no-stim (3 runs, ~T=600 each) ≈ 15–25 min; `kick` reuse existing delays (0) or rerun 3× (~30–45 min).
- **Cost checkpoint before any render/stim sims** (report survivors-style estimate + get user go), consistent with Task-4 discipline. Cheap code + TDD first.

---

## 9. Out of scope
- No cohort claim; no new subjects.
- No engine edits (`kick_probe.py`/`slow_field.py` untouched).
- No re-opening the spontaneous working-point search (closed NEGATIVE).
- `big` situation is NOT stimulated (core covers ~all contacts → no distinct axis); it appears only in Fig A.

## 10. Risks / open choices (flag for user review)
- **R1 (single-core may not show axis≥core):** a central radially-spreading core has no single chokepoint and leaks perpendicular to the axis; axis-stim (even split both sides) may only TIE or lose to core-stim. This is an honest outcome (G-B5 reported, not forced). If the user wants the single-core case to *demonstrate* axis≥core more cleanly, an alternative geometry is a **core at one end + a downstream tract (single chokepoint)** — deviates from "central" but gives axis-stim a real chokepoint to block. Default: central (as requested), report honestly.
- **R2 (window for the fast single-core event):** the small core fills in ~50 ms; the stim window/metric must resolve a delay on this fast timescale — locked in the plan; if too compressed, extend `T` and lower `drive` slightly (documented, not silent).
- **R3 (E1146 reuse vs rerun):** reuse keeps cost low but mixes code paths in Fig B; rerun is cleaner but ~30–45 min. Default reuse; switch on user preference.

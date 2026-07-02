# Stage-4 v2 spontaneous big-focus working-point search — NEGATIVE (fast-gate fail)

**Date:** 2026-07-02
**Branch:** `topic5-v2-phase1` (topic4 work; commits `20d4cd8` plan, `0766b0a` Phase-1 code)
**Status:** Phase 1 go/no-go gate FAILED at the fast stage. Confirm stage NOT run. Phase 2 (stim GIF) NOT run.
**Plan:** `docs/superpowers/plans/2026-07-02-stage4-v2-spontaneous-qI-stim.md`

## Abstract (第一性原理)

我们把一块很容易兴奋的**大脑组织大斑**放在那儿，让它只靠背景噪声**自己放电**（没有任何外部
触发），想看它会不会先冒出**一串分开的小火花**、把一个"抑制油箱"一点点耗干、最后才塌进
一次停不下来的大风暴——这一串结构正是"刺激版 GIF"要压制的对象。

怎么测的：试了 **12 组设定**，扫"组织有多易燃 × 一个'每次放完就自己疲劳'的刹车有多强、
多快"。被测的赌注是：如果每次小火花能靠这个**快刹车**自己熄掉，整片斑就该吐出**一串**火花
而不是**一次爆**。判据：塌进风暴之前有没有 ≥3 次分开的火花（成串 = 可用工作点），还是一
下就整片爆。每次一旦探到持续风暴就立即中止仿真，省算力。

揭示了什么：**12 组全都不成串**——每一组都是在 **23–32 毫秒**内整片**同时**点着、**一次**就
炸成风暴；那个"快刹车"**完全没用上**，因为整片组织在 ~30 毫秒内一起越过阈值，比任何"放完
之后才慢慢累积"的刹车快得多。所以在够得着的参数范围里，一块**大而均匀的热斑**自发只有
"**一次同步爆**"（把它调凉就变成"**哑火**"），中间**没有**做刺激图所需要的那种分开的火花串。
这跟本实验室一贯的结论一致：均匀 / 大核衬底自发只会整片爆或哑，中间没有稳健的自终止离散
事件档。

**结论口径**：这是一个 **go/no-go 快筛**（12 组、单一随机种子、只扫了易燃度 × 刹车强度 ×
刹车速度），**不是**"任何大灶设置都不可能"的证明。它的作用是：别再在"刹车"这个旋钮上继续
调了，下一根杠杆应该换地方（见下方 fallback）。

（内部归档代号：Stage-4 `extended_patch` 单大灶自发；M3A-v2.2 `SpatialSlowField` 的 `q_I`
慢油箱 + `g_K` 快疲劳场作为 per-event 刹车；`eta_K`/`tau_K` fast-brake 赌注；
`classify_workpoint` 判定 `one_shot_burst` 12/12；`results/topic4_sef_hfo/stage4_v2_workpoint/screen_fast.json`）

## What ran

**Substrate = canonical Stage-4 spontaneous runner** (source of truth
`run_sef_hfo_snn_cm_spontaneous_readout.py:520-525`): `Params(g=3.6)`, CLI defaults
`AR=2.0`, `theta=45°`, `density=100`, `drive=0.6`, `L=20`. Single isotropic excitable disk
(`extended_patch`) at sheet centre, radius `core_r=6.0` mm, built directly via
`sample_core_field` (the runner's `build_lesion_vth` extended_patch path is broken against the
current `sample_core_field` signature — pre-existing, documented in the plan Global Constraints).
Spontaneous = `n_pulses=0` / `KICK_BOOST=0` (background OU/Poisson drive only).

**Slow variables (M3A-v2.2):** `q_I` slow across-event inhibitory-resource depletion
(`k_q=0.25`, `tau_q=5000`, `sigma_q=1.5`, `q_min=0.05`); `g_K` COUPLED as a fast per-event
fatigue brake (`use_gK=True`, `eta_K>0`, short `tau_K`, `k_K=1.5`, `sigma_K=0.5`) — this is the
new bet vs the kick-driven `fig_m3a_v2_2_qI_runaway_transition` where `g_K` was visualized only
(`eta_K=0`).

**Grid (12):** `core_mean ∈ {16.5, 17.0}` × `eta_K ∈ {0.3, 0.5, 0.8}` × `tau_K ∈ {150, 400}` ms.
`FAST_T=900` ms with early-abort on the shared runaway criterion (`_smooth_rate` 20 ms +
`_first_sustained` 120 Hz / 100 ms, 80%-rule). Total wall time ≈ **22 min** (early-abort keeps
burst configs cheap).

## Results

| metric | value |
|---|---|
| verdicts | **`one_shot_burst` × 12** (0 `train_then_runaway`, 0 near-miss, 0 `train_no_runaway`, 0 silent) |
| survivors | **0 / 12** |
| `n_events` per run | 1 (every config) |
| runaway onset | 22.9 ms (`cm=16.5`) → 31.7–31.9 ms (`cm=17.0`) |
| `q_min_final` | 0.05 (floor) every run — the single blast drains `q_I` in one shot |
| brake effect | none: within an `eta_K`/`tau_K` block the onset is identical to 3 sig figs |

Full per-config rows: `results/topic4_sef_hfo/stage4_v2_workpoint/screen_fast.json` (gitignored).

## Mechanism interpretation

The failure is **not** "the brake is too weak" — it is "there is only **one** event." The big
homogeneous hot disk (r=6 mm, ~9k E cells, `core_mean` 16.5–17.0) crosses threshold
**synchronously** within 23–32 ms of the first noise-driven ignition, so the whole disk detonates
at once rather than nucleating a focal event that spreads. `g_K` can only raise a cell's effective
threshold *after* it has fired enough to accumulate fatigue; at ~30 ms `g_K ≈ 0` regardless of
`eta_K` (which merely scales `g_K`) or `tau_K` (which sets how fast it *would* build). Hence the
brake is structurally unable to act before the detonation, and there is no second event for it to
gate. Raising `core_mean` from 16.5→17.0 only delays ignition ~9 ms (higher threshold, slightly
later noise-crossing); the pilot's `core_mean=17.5` also burst and `18.0` = base_mean = silent, so
the burst→silent wall sits at ~17–18 with **no discrete-train regime between**.

This is the same structural wall as: `project_topic4_sef_hfo_step0_outcome` (homogeneous rate
field has no robust spontaneous discrete-event regime), the M2 ahead-of-front brake
(`project_topic4_m2_task0_event_extent_2026-06-19`: a subtractive brake either doesn't gate or
gates the whole sheet dead), and the M3A-v2 closed-loop closure
(`project_topic4_m3a_v2_spatial_field_plan`: the current SNN regime is all-or-nothing / whole-field).

## Gate decision + fallbacks

**GATE FAIL (fast) → STOP.** The confirm stage (would have been ~1 hr/run) was not launched
(0 survivors → nothing to confirm). Phase 2 stim GIF is not produced (no working point exists).

Fallbacks (NOT auto-picked; for user decision):
- **(b) Keep the kick-driven figures as the model's story.** The existing
  `fig_m3a_v2_2_qI_stim_*` (middle / endpoint / both-foci) already demonstrate the q_I brake
  mechanism with an externally-imposed discrete-event structure — which is exactly why they work
  (the scheduled kicks *supply* the train that the homogeneous substrate cannot self-generate).
- **(c) Small-core + weak-background variant** (a focal nucleate-and-spread source, NOT a "big"
  focus) — changes the scientific object away from what the user asked ("大灶自发"), so flagged
  as a scope change.
- **(d) Structured / heterogeneous core** so events nucleate locally and self-terminate, or a
  genuinely self-terminating recurrent-excitation (E→E depression) mechanism — a substrate
  redesign, i.e. the "reopen the carrier" lever, not a knob within this grid.

## Provenance / reproduce

```
python scripts/run_stage4_v2_workpoint_search.py --stage fast
# -> results/topic4_sef_hfo/stage4_v2_workpoint/screen_fast.json ; SURVIVORS 0 / 12
```

Code: `scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py`
(`_build_stage4_patch`, `_simulate_continuous` `vth=`/`abort_on_runaway`),
`scripts/run_stage4_v2_workpoint_search.py` (`classify_workpoint`, `run_one`, `--stage fast`),
`tests/test_stage4_v2_spontaneous.py` (10/10 green). Superseded feasibility pilot:
`scripts/pilot_stage4_spontaneous_qI.py` (6/6 `one_shot_burst`, motivated the g_K-fast-brake bet
tested and rejected here).

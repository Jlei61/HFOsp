# Stage 3 — Long-record stochastic template train (cm-SNN, two coexisting foci)

**Date:** 2026-06-13
**Topic:** Topic 4 SEF-HFO observation layer → spontaneous bidirectional read-out
**Status:** DRAFT — pending user review before an implementation plan is written
**Predecessor:** Stage 2 (pooled two-run synthetic bidirectional subject), `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/`
**Reviewer fixes folded in:** P1-1 … P1-5 + structure-gate active-contact correlation (review 2026-06-13)

---

## §0 Scientific framing and tier discipline

**测了什么（一句话）** — 把"两个等强易激病灶坐在同一条几何轴的两端、靠背景噪声各自自发点火"跑成一条长记录，问它**自己长出来的事件序列**：(a) 真实盲分析流程能不能把它分成两个互为反向、端点互换、左右半程可复现的传播模板；(b) 在时间上，"这一次是哪一头先点着"是不是像抛硬币一样**没有强制交替、也没有强黏连**；(c) 每个事件的最早入口是不是在一个小入口群里来回抖动，而不是死锁在单个触点。

This is a **model-sufficiency demonstration**, NOT a causal proof about real patients. Real-data phenomenology is already established (Topic 1 + the entry-dispersion work). Stage 3 only shows this specific substrate *can* reproduce the full picture including the timing layer that the pooled Stage 2 was structurally unable to test (Stage 2's event order was hand-interleaved).

**Pre-registered tiers (fixed at planning time, per CLAUDE.md §5):**

| Layer | Tier |
|---|---|
| Structure (k=2 / opposite templates / endpoint swap / split-half + odd-even reproducible) | **precondition** (already shown in pooled form; re-showing in one real network is a sanity gate, not the headline) |
| Label-independence (no strong ping-pong / no strong persistence at model scale) | **primary new model claim**, made falsifiable by the positive controls |
| Entry-jitter (nucleation site wanders, not a fixed single contact) | **secondary** |
| Slow-state rate modulation | **out of scope** — deferred to a later step |

**Allowed conclusion wording:**
> 双端等强异质性病灶在同一个 cm-SNN 中能自发产生两类互为反向的传播模板；模板几何稳定、端点互换、左右半程可复现；事件的"哪一头"标签在模型尺度上**未见强制交替或强黏连**（不是"证明独立"）。

**Forbidden wording:**
> ~~SNN 证明了真实病人的正反模板就是两个病灶独立随机点火。~~ （端点、SOZ、E/I、病理来源都没有被这个模型直接识别 → 机制充分性演示，不是因果证明。）

Small-n honesty (recurring lesson): with ~6 runs the combined timing CI is wide. The primary timing statement is capped at **"未见强依赖"**; we do not write "证明独立".

---

## §1 Simulation design

One cm-SNN on the validated substrate: `L=20`, `density=100`, `theta_EE=45°`, `AR=2`, `g=3.6`, `dt=0.1 ms`, `nu_ext_ratio=0.6` (DRIVE). Engine = the guarded `results/topic4_sef_hfo/lif_snn/engine` (checksum guard via `assert_versions`).

**Two equal low-threshold cores (`twoend_equal`, NEW lesion mode):**
- core mean = **17.0 mV** (the validated-clean operating point; 16.5 over-heats → fragmented/quasi-continuous; we stay above it), spread = **1.5** ("wide"), radius = **1.5 mm**.
- one core at the −axis-end (`center − 0.6·half·axis_unit`), one at the +axis-end (`center + 0.6·half·axis_unit`) — same offsets as the existing `twoend_deph`.
- `vth = min(cf_neg.vth, cf_pos.vth)`; surround stays at base 18.0 mV (sub-critical → bare sheet quiet → events nucleate ONLY from the two cores).
- **Equal means** (NOT the `twoend_deph` +0.3 mV offset). The 2026-06-10 diagnostic found equal means can collide/merge — that is handled by collision detection + censoring (§2), NOT by dephasing.

**No external kick:** `simulate_kick(KICK_BOOST=0.0, t_kick=1e9)`. Background noise drives spontaneous ignition.

**Read-out montage:** unchanged from the validated cm read-out — 2 shafts, ∥ (theta_EE) and ⊥ (theta_EE+90), pitch 4 mm, `nc` contacts each, `valid_mask` (on-sheet ∧ ≥1 neuron within `Rr`). `endpoint_centroid_axis`, `k_dir=3`, `margin_frac=0.10`. All contacts must be on-sheet (refuse boundary-extrapolated records).

**Per-event hidden source label (P1-4 fix) — core-level onset, NOT single-cell first spike:**
- For each detected event window, compute each core's **E-neuron active fraction** time series (fraction of that core's E cells that have fired, binned at `BIN_MS`).
- `core_onset_X` = first time core X's active fraction crosses a threshold = `max(0.01, N_min/n_core_cells)` (≥1% of core cells OR ≥`N_min` cells, whichever is larger; `N_min` ~ a small fixed count, e.g. 5 — pinned in the implementation plan after the pilot).
- `hidden_source_label`:
  - `neg` if `core_onset_neg + Δ_onset < core_onset_pos`
  - `pos` if `core_onset_pos + Δ_onset < core_onset_neg`
  - `collision` if `|core_onset_neg − core_onset_pos| ≤ Δ_onset` (both cores ignite near-simultaneously)
  - `ambiguous` if neither core crosses threshold inside the window, or the read-out axis is unreadable (`axis_err is None` / `n_part < PART_MIN`).
- `Δ_onset` pinned after the pilot (target: well below the typical inter-event interval, above one bin). Both `core_onset_neg` and `core_onset_pos` are written to the sidecar so the threshold can be re-cut offline without re-simulating.

---

## §2 Collision handling = censor boundary (P1-1 fix)

**Collisions and ambiguous events are NOT deleted from the middle of the sequence.** Deleting them would splice the event before and the event after into a fake "adjacent" pair, manufacturing or masking lag-1 / run-length / transition structure — and the label sequence is the primary object under test.

**Mechanism (uses the real machinery unchanged):** the time axis is partitioned into **maximal collision-free / ambiguity-free segments**. Each segment becomes one entry in `block_time_ranges`. Collision/ambiguous events fall *between* blocks → `_assign_blocks` maps them to −1 → they are skipped as anchors AND `compute_runs` closes the current run at the boundary (`in_block_mask[i]` is False). No transition is ever counted across a collision. No ad-hoc code; this is exactly what `block_time_ranges` is for.

**Two views reported (P1-1 reporting requirement):**
1. **collision-as-boundary** (PRIMARY): segments = blocks as above. The primary conclusion comes from this view.
2. **clean-only naive** (robustness contrast): drop collision/ambiguous events, concatenate the survivors as a single block. This is the *wrong* concatenation; reporting it alongside view 1 shows whether the censoring actually matters. If the two agree, the result is robust to collision handling.

**Also reported:**
- **collision rate** = `n_collision / n_events` (gate: < 20–30 %, else "two independent spontaneous sources" is not a stable regime — fail the cell).
- **collision end-bias**: are collisions preferentially preceded by one end firing? (a collision that systematically follows `neg` events is itself a dependence signal — must be inspected, not silently censored).
- segment-length distribution (each timing test needs enough events *per segment* to be meaningful).

---

## §3 Three analysis layers

### §3.1 Structure (precondition) — per-run, masked

Each network run = one synthetic subject (long runs give ≥60–100 events/run, enough for per-run clustering — no Stage-2-style pooling across networks). Reuse the masked pipeline body refactored out of `pool_and_cluster_spontaneous.py` to take ONE single-network record:
- masked PR-2 adaptive cluster (`compute_adaptive_cluster_stereotypy(..., use_masked_features=True)`) → `stable_k`, labels, cluster sizes.
- masked PR-2.5 reproducibility (`compute_time_split_reproducibility(..., use_masked_features=True)`) → split-half **and** odd-even are **both computed and reported**; the forward/reverse-reproduced **pass criterion is split-half OR odd-even** per the Topic 1 `forward_reverse_reproduced` contract (checking only split-half undercounts — 8/9 subjects in real data). NOT an AND.
- rank-displacement swap sweep (`compute_swap_score_sweep`) on the two cluster templates over their **shared valid contacts** → `swap_class`, `decision_k`.

**Inter-cluster correlation — descriptive sanity check, NOT a hard gate.** Computed on **shared active contacts only** = intersection of the two cluster templates' valid masks (`vm0 & vm1`), NOT on masked-imputed values (the same inflation that pushed the misaligned-montage corr to +0.74). Report `n_shared` and the value; if `n_shared` is too small to trust, report but **do not interpret**. Endpoint reversal is judged by **rank-displacement `swap_class` ∈ {strict, candidate}**, not by the correlation — the model template space is near-1D, so a correlation is easily over-read by eye, which is why it stays descriptive.

**cluster ↔ hidden-focus purity:** contingency between the blind cluster label and the hidden source label (collisions excluded). Gate: ≥ 90 % (the two clusters really are the two ends).

### §3.2 Label-independence (PRIMARY) — model-scale PR-7 analog (P1-2, P1-3 fixes)

Runs on the **blind cluster labels** (what a real analyst sees — matches how real Topic 1 ran PR-7), with the hidden source labels as a mechanism cross-check.

**Model-scale Δt grid (P1-2):** measure `compute_pairing_lift(..., delta_t_seconds=Δt, block_time_ranges=segments)` over a grid of Δt covering ~**1, 2, 4 typical inter-event intervals** (typical IEI estimated from the run; "seconds" here is the model time axis). The patient-second windows (10/30/60 s) and the local-window null's **1800 s default are NOT used** — on a ~15 s record they degenerate to near-global statistics. Documented as a *model-scale PR-7 analog*, not literal patient-time PR-7 replication.

**Null models:** `compute_pairing_with_nulls` / `compute_burst_diagnostic_with_nulls` with:
- **N1** block-aware shuffle (composes naturally with collision censoring — shuffles labels within each collision-free segment),
- **N3** circular shift,
- **N0** global shuffle (marginal-preserving baseline).
- **N2** local-window only if `n2_window_seconds` is rescaled to a model-scale window (a few IEIs); otherwise omitted. Never the 1800 s default.

**Gate (P1-3 — corrected null points):**
- `lag1_same_excess` (empirical − null_mean): CI **includes 0**.
- pairing excess (opposite/same vs null): CI **includes 0**.
- `run_length_lift` (empirical/null **ratio**, `template_temporal_pairing.py:855/857`): CI **includes 1**.

Per-run statistics, then combined across runs. **Combination default (small-n honest):** report each run's `lag1_same_excess` / `run_length_lift` plus a sign-consistency summary (how many of N runs fall inside the independence CI). A formal across-run test (sign/Wilcoxon) is reported only if N is large enough to mean anything — with N≈6 the honest output is the per-run spread, not a single pooled p-value. Primary statement capped at "未见强依赖".

### §3.3 Entry-jitter (secondary) — NEW small helper

Per event: the first-active contact (and/or first-firing core cell, from the core-onset machinery). Across events of a given direction:
- the first-active contact is NOT a single fixed contact (dispersion > 0);
- a small lead group dominates (echoing real "top-3 cover ~74 %").
Report the lead-contact distribution per direction.

---

## §4 Positive controls (P1-5 fix) — falsifiability of the no-ping-pong claim

The independence claim is reported only if the main run looks independent AND the controls below are caught as dependent by the *identical* test.

1. **Engine-level forced ping-pong (physical existence control).** A second simulation where, after one end fires a global event, an **asymmetric cross-coupling** briefly suppresses that end / boosts the opposite end → forces alternation. The same timing test must flag it dependent.
   - The coupling magnitude/duration is tuned only to *produce* alternation (a control's job is to fail the gate) — NOT tuned to the main run's answer.
   - *Caveat (the reason P1-5 was raised):* this coupling also perturbs event rate, collision rate, and event size, so what the test catches here may be partly a rate artifact, not pure label dependence. It demonstrates a *physical* mechanism can produce detectable dependence; it is NOT the test-validity control.

2. **Synthetic-label controls (test-validity controls — cheap, the decisive ones).** Take the MAIN run's event times and block structure **unchanged**, keep the **marginal label counts unchanged**, and only re-arrange the label order:
   - **forced-alternating** (ping-pong) → must yield `lag1_same_excess` strongly negative, `run_length_lift` < 1 → flagged dependent.
   - **forced-sticky** (persistence) → `lag1_same_excess` positive, `run_length_lift` > 1 → flagged dependent.
   - **independent shuffle** (sanity) → should read as no-dependence (the test does not false-positive on a genuinely independent sequence at the run's actual time/marginal structure).
   These isolate "the statistical test has teeth at the model's real event-time/marginal structure" from any rate/collision/size confound the engine control introduces.

---

## §5 Acceptance gates (each encodes its conclusion)

Per-cell (per operating point), aggregated across runs:

| Gate | Threshold |
|---|---|
| clean events per end | ≥ **30** (≥ 50 preferred) |
| collision rate | < **20–30 %** |
| `stable_k` | == 2 (across runs) |
| cluster ↔ hidden-focus purity | ≥ **90 %** |
| inter-cluster corr (shared active contacts) | **descriptive sanity only — NOT a gate**; reported, not interpreted if `n_shared` small |
| forward/reverse reproduced | **split-half OR odd-even** (both computed + reported; pass = OR, Topic 1 contract) |
| rank-displacement (primary endpoint-reversal judge) | `swap_class` ∈ {strict, candidate} |
| independence (primary) | main run: `lag1_same_excess` CI∋0, pairing excess CI∋0, `run_length_lift` CI∋1 **AND** all positive controls flagged dependent |
| entry-jitter (secondary) | first-active contact dispersion > 0 with a small lead group |

Timing conclusion wording capped at **"未见强依赖"**, never "证明独立".

---

## §6 Sidecar schema (per event)

Minimal fields required for offline re-analysis without re-simulation:

```
event_id
t_on / t_off
event_peak_t
hidden_source_label          ∈ {neg, pos, collision, ambiguous}
core_onset_neg / core_onset_pos   (ms; raw, so the Δ_onset cut can be redone offline)
collision_reason             (simultaneous_onset | unreadable_axis | no_core_crossing | none)
cluster_label                (assigned post-hoc by the blind pipeline)
clean_for_timing             (bool: enters the labeled timing analysis)
block_id_after_collision_censoring
n_part / axis_err / sign / readability   (existing read-out fields)
```

The single-network legacy lagPat record (`*_lagPat_withFreqCent.npz` + `*_packedTimes_withFreqCent.npy` + `*_montage.json`) is written with the REAL spontaneous event times (NOT Stage 2's interleaved synthetic times), so the timing layer is meaningful.

---

## §7 Code units (reuse-first)

- **Engine layer** — extend `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`: add `twoend_equal` lesion mode (reuse `build_lesion_vth` / `sample_core_field`); core-level onset + collision detection; sidecar writer; `--control engine_pingpong` flag adding the asymmetric coupling; write ONE single-network record + sidecar.
- **Structure analysis** — refactor the masked-pipeline body of `pool_and_cluster_spontaneous.py` into a function taking one record (drop the two-run concatenation); add shared-active-contact corr + cluster↔focus purity.
- **Timing analysis** — NEW thin wrapper over `src/template_temporal_pairing.py` (`compute_pairing_with_nulls`, `compute_burst_diagnostic_with_nulls`, `compute_lag1_same_fraction`, `compute_runs`/`compute_run_metrics`) driven by collision-free `block_time_ranges`, model-scale Δt grid, N0/N1/N3 nulls; runs main + synthetic controls + engine control; per-run then combined.
- **Synthetic-label controls** — NEW small helper: re-arrange labels (alternating / sticky / independent) holding times + marginal counts fixed.
- **Entry-jitter** — NEW small per-event first-contact helper.
- **Orchestrator** — NEW `scripts/run_sef_hfo_snn_cm_spontaneous_stage3.py`: pilot mode + multi-seed main + controls, gate each, write `stage3_summary.json`.
- **Figure** — extend the core figure with a timing panel (label sequence + lag-1 / run-length vs null, main vs controls side by side).

Reuse discipline (CLAUDE.md §6.1): the timing helpers' null contract is "is the label order independent given block structure" — which matches the model question. `block_time_ranges` MUST be passed explicitly everywhere (no default), per the `compute_transition_odds` boundary-parameter rule (§6).

---

## §8 Execution route (minimal, pilot-gated)

1. **Spec** (this doc) with all P1 fixes — pending user review.
2. **Short pilot, NO conclusions:** confirm `twoend_equal` at 17.0 wide does not mush, and the collision rate is acceptable (< 20–30 %); pin `N_min`, `Δ_onset`, `T`, and the typical IEI for the Δt grid.
3. **Multi-seed long records:** ~6 independent networks (connection seed FREED — reusing one skeleton is not a real multi-seed, per the heterogeneity-result lesson), `T` sized so each run yields ≥ 30 (ideally 50) clean events/end with usable segment lengths.
4. **Blind clustering + timing test + positive controls**, then gates + figure.

---

## §9 Resourcing defaults (tunable, not scientific forks)

- operating point: 17.0 mV wide (above the 16.5 over-heating point).
- `T ≈ 15 s/run` as a starting estimate (Stage 2 gave ~15 clean/focus at 3.5 s single-focus; two shared-sheet foci lose some to collisions + mutual dead-time) — re-pinned by the pilot.
- seeds: 6 independent networks; structure per-run; independence per-run then combined (a label *sequence* exists only within one continuous run — pooling sequences across runs would fabricate transitions).

---

## §10 Open confirmations (resolved during brainstorming)

- independence tested on **cluster labels** (primary), hidden labels as cross-check — confirmed.
- positive control: **yes**, engine-level + synthetic-label (P1-5).
- slow-state rate modulation: **out of scope** for Stage 3.
- two equal foci + collision censoring (A1) — confirmed.

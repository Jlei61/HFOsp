# FCXR-HEO2 — Broadband-spiky ictal phenotype: mechanistic diagnosis (design spec)

**Date** 2026-07-24 ｜ **Branch** `codex/topic4-mz-fcxr-heo1` (base `01225ff7`) ｜ **Type** diagnostic (picks the
mechanism for a later sprint), NOT a GO/NO-GO acquisition sprint. Follows FCXR-HEO1 + its two review
rounds (archive `docs/archive/topic4/sef_hfo/mz_fcxr_heo1_2026-07-24.md`).

## 0. Why this exists (the reframe)

HEO1 asked "can the cooperative gate make a **sustained 30–150 Hz broadband platform**?" — answer:
no (0/48), it makes a bounded **~16 Hz narrowband** coherent limit cycle (low bands suppressed). Two
review rounds established:

1. The **HEO1 gate is mis-specified** for this subject. A real E1146 seizure (`pat_114602`, CP,
   origin SCL7/8) is a **~3 Hz spiky broadband, intermittent** pattern: it passes Gate B/C (broadband +
   12–14/15 contacts in *some* windows) but **fails Gate A** (no sustained ≥1 s plateau — the spike-wave
   returns to baseline ~every 300 ms; `passes_full_HEO1_gate=False, gate_A_plateau=False`). Neither the
   model's ~16 Hz state nor the real seizure passes the HEO1 gate.
2. The model **already contains both halves** of the answer: (a) a **sparse ~3.5 Hz broadband-spiky**
   event family (e.g. `gq0.9999_A1_D0.15_nokick`: six-band ΔdB all positive `[4.0,6.7,5.8,6.0,4.1,7.4]`
   but only **3/15** coverage, ~18% duty), and (b) a **sustained 16 Hz narrowband** family (A4 cells).

**HEO2's real question is therefore not "16 Hz → 3 Hz" but: what merges the sparse-spiky-broadband
*waveform* with the sustained-16 Hz *persistence + spatial coverage*, to match the real seizure's
broadband-spiky phenotype?** This diagnostic decides the mechanism knob empirically before a sprint.

## 1. Target contract (Phase −1, no model runs) — LOCKED

The target is the **empirical broadband-spiky phenotype**, a set of continuous metrics vs the real
E1146 seizure — **NOT** the HEO1 binary gate (which neither passes).

- **Real E1146 six-band ΔdB reference vector** (measured, `real_e1146_seizure_gate.json`, local-CAR,
  onset+3..18 s vs interictal): `R = [12.0, 10.4, 8.6, 8.3, 5.0, −1.2] dB` for
  `[1-4, 4-8, 8-13, 13-30, 30-80, 80-150] Hz`. Real per-contact dominant **~3 Hz**.
- **Target signature**: dominant **3–8 Hz**; low bands (1–4, 4–8) and 13–30 strongly positive; 30–80
  moderately positive; 80–150 descriptive only (not required); spiky (non-sinusoidal) waveform;
  spatial coverage retained.
- **Spectral distance to real** `d = ||M − R||₂` (L2 in dB, six-band) + cosine similarity of the ΔdB
  profiles, where `M` is a model cell/arm's six-band territory ΔdB. Lower `d` / higher cosine = closer.
- Explicitly retract (done, committed) any "real seizure passes the HEO gate / all bands sustained"
  wording in HEO1 artifacts.

## 2. Phase 0 — re-map the 48 cells (zero new compute), fixed measurement

**Problem fixed:** the HEO1 `oscillation_probe` uses `nperseg=512@1000 Hz` (~1.95 Hz bins); with the
`f>2` floor the first bin is 3.906 Hz, so all sparse cells read exactly 3.91 Hz — a **resolution-floor
artifact** (2 s Welch shows the real value ~3.5 Hz). Phase 0 uses proper estimation.

Per cell (all 48 `screen_cells/*_trace.npz`), compute:
- **dominant_hz**: 2 s Welch (`nperseg=2·fs`, ~0.5 Hz) on population rate + per-contact LFP, peak 1–200 Hz.
- **event_ipi_hz**: detect population-rate events (peaks over a rolling-median gate) → median
  inter-peak interval → `1/IPI`. A frequency is a "rhythm" only if `dominant_hz` and `event_ipi_hz` agree.
- **autocorr_period_ms**: first non-zero-lag autocorrelation peak of the rate.
- **spectral_entropy** (Shannon of normalized PSD) + **bw90** (90 %-power bandwidth): narrow vs broad.
- **spikiness**: kurtosis of the population rate (spike-wave = high kurtosis; sinusoid ≈ 0) + harmonic
  ratio (power in ≥2× harmonics / fundamental).
- **six-band ΔdB** (territory median, `band_db_field`) + **spectral_distance_to_real** + cosine.
- **duty_cycle**: fraction of 300 ms-rolling-rate above a mid-ictal threshold.
- **max_silence_gap_ms**; **rolling_ictal_occupancy** (fraction of 1 s windows broadband-high).
- **coverage**: max broadband-high contacts per window + SCL (continuous, from the HEO1 Gate-B logic —
  reused, reported as a number, not a binary gate).
- **class ∈ {sparse_event_train, transitional, tonic_16Hz_cycle, target_like_spiky}** — pre-registered:
  `tonic_16Hz_cycle` = dominant>13 Hz ∧ duty>0.7 ∧ coherent; `sparse_event_train` = dominant<8 Hz ∧
  duty<0.4; `target_like_spiky` = dominant 3–8 Hz ∧ all six ΔdB>0 ∧ duty≥0.6 ∧ coverage≥8/15; else
  `transitional`.

Anchors compared directly: `A1_D0.15_nokick` (sparse spiky) vs `A4_D0.15_nokick` (16 Hz). Output
`broadband_diagnostic/phase0_state_map.json` + a figure (dominant vs duty vs coverage, colored by class;
spectral-distance-to-real per cell). **If any existing cell is already `target_like_spiky`, Phase 1's
question narrows to "can adaptation move A4's sustained/covered high state onto that spiky rhythm?".**

## 3. Phase 1 — delayed force-matched adaptation wedge (small sims)

**Mechanism wedge:** cooperative positive feedback (high state) + slow spike-frequency adaptation
(delayed negative feedback) → candidate relaxation bursting. NOTE: in `full_conductance`, `m` is already
mapped to `g_M(E_K−V)` (spike-triggered slow-K/sAHP) — "adaptation" *is* the slow-K here; a genuinely
new slow-K would need Ca-dependence / a saturating pool / voltage-dependent M-current (deferred). Honest
prior: linear `m` likely yields **prevention/stalling → sparse → silence**, so this is a **cheap
falsification probe**, not "most likely to succeed".

**Two design pitfalls this phase must avoid:**
- **Confounding timescale with strength.** Sweeping `tau_adp` alone changes recovery speed *and*
  steady-state `m` *and* mean adaptation current. → **Force-match** `eta_m` per `tau` (reuse
  `src.topic4_mz_slowvars.eta_m_from_frac(frac, I_EE_scale, peak_m)` + `replay_adaptation_peak`) so the
  in-high-state adaptation-current peak is a fixed fraction of recurrent drive.
- **Opening adaptation from t=0 answers the wrong question** (it may just prevent ignition; prior MZ
  work: linear adaptation = prevention, not containment). → **Delayed onset**: run m-off for the first
  1 s to establish + confirm the 16 Hz high state, THEN enable adaptation.

**Grid (seed1 screen):** representative cell = the Phase-0-chosen sustained/covered anchor (default
`A4_D0.15_nokick`). 6 arms:
- `m_off` (control = the ~16 Hz state),
- `tau_adp ∈ {250, 750 ms}` × `eta_m` force-matched to `{5%, 10%}` of recurrent drive (4 arms),
- `static_K` control: `m_frozen_E` held at a level matching the 10 %-arm's mean `g_M` (distinguishes
  dynamic adaptation from a mean K-load shift).

Each `(tau, frac)` first passes a **D=0 workpoint gate** (adaptation must not erase normal interictal
IED at baseline). **Run structure:** first 1 s m-off (assert 16 Hz high state established), then enable
`m` for `max(4 s, 5·tau)` → total ~5–6 s (per-arm length scales with tau). seed1 screen; **seed3 only
for arms that pass the success gate.**

**Pre-registered success (all 8 conjunctive; a clean fail on 5/6/7 = valid falsification):**
1. pre-adaptation (0–1 s): 16 Hz high state established (coherent, mean rate > 60 Hz);
2. post-adaptation (last ≥2 s): dominant 3–8 Hz AND `dominant_hz` ≈ `event_ipi_hz` (real rhythm, not
   floor artifact);
3. low (1–4, 4–8) AND 13–30 ΔdB positive;
4. spectral_distance_to_real strictly improved vs the m-off arm;
5. NOT collapsed: post-adaptation mean rate stays above the interictal band (not sparse-IED ~4 Hz mean);
6. coverage NOT degraded to focal: max broadband-high contacts ≥ the m-off arm's (not < the sparse
   family's 3/15);
7. adaptation dynamics: `m_mean` lags high-activity onset and recovers in the low phase (bursting
   signature), not monotone suppression;
8. numerical safe: `clip_frac==0`, no runaway, finite, `tau_eff_min≥2·dt`.

## 4. Engineering contract

- **6 blessed engine files untouched** (`kick_probe/params/model/connectivity/connectivity_rot/lfp`);
  all changes in non-blessed `mz_slow_vars.py` + new HEO2 module/runner/plotter/tests.
- **New `m_enable_ms` (off-by-default) in `mz_slow_vars.py`** (multi-clause contract, TDD each):
  (a) default `None` → byte-identical to current `use_m` behavior (existing suites stay green);
  (b) when set, `m ≡ 0` for `step_i·dt < m_enable_ms`, then accumulates normally (gate the
  `self.m[spk & is_E] += 1.0` line + the decay); (c) `apply_currents`/`membrane_terms` see `m=0`
  before enable → the pre-enable window is exactly the m-off high state; (d) deterministic.
- **New `m_frozen_E` (off-by-default)** mirroring `z_frozen_E`: holds `m` at a preset per-E field
  (requires `use_m=False`, `m_enable_ms=None`), giving a static `g_M` for the matched-K control.
- **Force-match**: reuse `eta_m_from_frac` + `replay_adaptation_peak` (grep-confirmed) — do not reinvent.
- **Reuse** HEO1 `build_baseline_reference` / `band_db_field` + new Phase-0 estimators; substrate via
  `build_substrate` + frozen-Z field; classifier caveat (model synthetic |current| proxy vs real iEEG)
  stays.
- **Per-arm traces saved**: `m_mean, gM_mean, rate_E, per-contact energy, coop_engaged_frac` + LFP.
- **Resource/nohup** (as HEO1): `OMP=1`, flock launcher, ≤2 workers for T≤8000, `setsid nohup`,
  RUNNING/DONE/FAILED sentinels, resource_log, swap/RSS gates. ~6 arms × ~5–6 s ≈ 60–80 min at 2 workers.
- **Results root** `results/topic4_sef_hfo/mz_full_conductance_spatial_relay/broadband_diagnostic/`.

## 5. Decision gate + forbidden claims

**Output → decision:** Phase 0 says whether a `target_like_spiky` state already exists and its
parameters; Phase 1 says whether **delayed adaptation transforms** the sustained 16 Hz state onto a
spiky 3–8 Hz broadband rhythm **while keeping coverage** (all 8 criteria), or merely **stalls/collapses**
it. That verdict picks the mechanism for the fuller broadband sprint (its own spec): if transform →
lock adaptation; if collapse → next knob (Ca-dependent slow-K / heterogeneity / waveform).

**Forbidden:** calling adaptation-induced collapse or sparse-IED "success"; reusing the HEO1 gate as
the HEO2 target; "seizure reproduced / broadband platform achieved"; reporting a `dominant_hz` from the
coarse probe as a rhythm without event-IPI agreement; attributing broadening to "slow timescale" without
force-matched `eta_m`; claiming "adaptation transformed the high state" from a t=0-adaptation run.

## 6. Out of scope (YAGNI)

No new slow-K mechanism (Ca-dependent / saturating pool / M-current) this round; no multi-core geometry;
no connectivity/axial changes; no dynamic Z; no full seizure lifecycle. This is a mechanism-selection
diagnostic only.

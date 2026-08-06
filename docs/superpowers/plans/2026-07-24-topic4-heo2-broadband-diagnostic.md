# FCXR-HEO2 Broadband-Diagnostic Implementation Plan

> **For agentic workers:** executed INLINE autonomously (user: "go, ~8h, P0-only interruptions"). TDD
> per task; commit per task. Spec: `docs/superpowers/specs/2026-07-24-topic4-heo2-broadband-diagnostic-design.md`.

**Goal:** Decide empirically (Phase 0 re-map + Phase 1 delayed-adaptation wedge) which knob turns the
model's ~16 Hz narrowband state into the real E1146 broadband-spiky (~3–8 Hz) phenotype.

**Architecture:** Phase 0 = pure re-analysis of the 48 existing HEO1 traces (new estimators + 4-class
map, zero compute). Phase 1 = new off-by-default `m_enable_ms` (delayed adaptation) + `m_frozen_E`
(static-K control) in the non-blessed `mz_slow_vars.py`, a runner that force-matches `eta_m` per `tau`
and screens 6 arms with 8 conjunctive success criteria.

**Tech Stack:** numpy, scipy.signal (welch/find_peaks), existing `mz_slow_vars`, `topic4_mz_slowvars`
(`eta_m_from_frac`/`replay_adaptation_peak`), HEO1 `topic4_mz_fcxr_heo1` (`band_db_field`,
`build_baseline_reference`), `run_topic4_mz_fcxr_heo1` scaffolding, `build_substrate`.

## Global Constraints (verbatim)

- **6 blessed engine files NEVER touched**: `kick_probe.py, params.py, model.py, connectivity.py,
  connectivity_rot.py, lfp.py` (SHA-guarded by `engine_versions.json`). All changes non-blessed.
- **`m_enable_ms=None` (default) → byte-identical** to current `use_m` behavior; existing mz suites green.
- **`m_frozen_E`** mirrors `z_frozen_E`: requires `use_m=False` AND `m_enable_ms=None`; static `g_M`.
- **Force-match** `eta_m` per `tau` via existing `eta_m_from_frac` + `replay_adaptation_peak` (reuse).
- **Target = continuous metrics vs real E1146 six-band ΔdB `R=[12.0,10.4,8.6,8.3,5.0,−1.2]dB`**, NOT the
  HEO1 binary gate.
- **Substrate anchors unchanged** (E1146/narrow/L=20/N=40000/dt=0.05, frozen-Z field, `_fc_cfg` arm-C).
- **Resource/nohup**: OMP=1, flock, ≤2 workers (T≤8000), `setsid nohup`, sentinels, swap/RSS gates.
- **Results root** `results/topic4_sef_hfo/mz_full_conductance_spatial_relay/broadband_diagnostic/`.
- **Forbidden**: calling adaptation collapse/sparse-IED "success"; reusing HEO1 gate as target;
  "seizure reproduced"; `dominant_hz` without event-IPI agreement; broadening without force-matched
  `eta_m`; "transformed high state" from a t=0-adaptation run.

## Diagnostic P0 stop conditions (interrupt the 8h run only for these)

byte-parity broken (`m_enable_ms=None` ≠ current) ; a blessed engine SHA changes ; the frozen-Z D=0
workpoint gets erased by an m-param (all arms fail workpoint) ; resource hard-abort (MemAvailable<32GiB
or swap Δ≥512MiB) ; the real target vector can't be reproduced. Otherwise run to completion; a clean
"adaptation stalls/collapses" is a valid diagnostic result, not a stop.

---

## Task 1 — Phase 0 estimators + 48-cell state map (zero compute)

**Files:** Create `src/topic4_mz_fcxr_heo2.py`, `scripts/run_topic4_heo2_phase0.py`,
`tests/test_topic4_mz_fcxr_heo2.py`. Reads `broadband_diagnostic/` = new; reads existing
`high_energy_oscillatory_branch/screen_cells/*_trace.npz` + `baseline_lfp_seed1.npz`.

**Produces (signatures later tasks/consumers rely on):**
- `dominant_2s(sig, fs, lo=1, hi=200) -> float` (2 s Welch peak; fixes the 3.906 floor)
- `event_ipi_hz(rate, fs) -> float` (find_peaks over rolling-median gate → 1/median-IPI)
- `spikiness(rate) -> float` (excess kurtosis) ; `spectral_entropy(sig, fs) -> float` ; `bw90(sig, fs) -> float`
- `spectral_distance_to_real(six_band_ddb) -> (l2, cosine)` with `REAL_E1146_DDB=[12.0,10.4,8.6,8.3,5.0,-1.2]`
- `duty_cycle(rate, fs) -> float` ; `max_silence_gap_ms(rate, fs) -> float`
- `classify_state(metrics) -> str` in {sparse_event_train, transitional, tonic_16Hz_cycle, target_like_spiky}
  per spec §2 thresholds.

**TDD contracts (each a test):**
1. `dominant_2s` on a synthetic 3.5 Hz sine → 3.5±0.5 Hz (NOT 3.906); on 16 Hz → 16±0.5.
2. `event_ipi_hz` on a 4 Hz pulse train → ~4 Hz; agrees with `dominant_2s`.
3. `spikiness`: high for a spike-wave (sharp), ~0 for a sinusoid.
4. `spectral_entropy`/`bw90`: low/narrow for a pure tone, high/broad for white noise.
5. `spectral_distance_to_real`: 0 for `R` itself; larger for a narrowband `[-20,-14,-9,+16,+3,0]`-like vector.
6. `classify_state`: the A1 anchor (~3.5 Hz, all-ΔdB>0, duty≈0.18, cov 3) → `sparse_event_train`;
   A4 (16 Hz, duty>0.7, coherent) → `tonic_16Hz_cycle`; a synthetic 5 Hz all-band-up duty0.7 cov10 →
   `target_like_spiky`.

- [ ] tests → RED → implement estimators + `classify_state` → GREEN → commit.
- [ ] `run_topic4_heo2_phase0.py`: over all 48 cells, compute the metric row + class, write
  `broadband_diagnostic/phase0_state_map.json`; print class tally + the closest-to-real cells +
  the A1/A4 anchors; render `broadband_diagnostic/figures/phase0_state_map.png` (dominant vs duty,
  size=coverage, color=class; + spectral_distance bar). Run it. Eyeball figure. Commit.
- [ ] **Decision recorded in the run output:** does any existing cell classify `target_like_spiky`? If
  yes, Phase-1 anchor = the best sustained/covered cell whose knobs are closest to it.

## Task 2 — `m_enable_ms` delayed adaptation (mz_slow_vars.py) + TDD

**Files:** Modify `src/snn_engine/mz_slow_vars.py` (config field + `step()` gate). Test in
`tests/test_topic4_mz_fcxr_heo2.py`.

**Contract (multi-clause, hfosp-deep-contract-verify):**
- `m_enable_ms: float|None = None`. `None` → current behavior EXACTLY (byte-parity).
- When set + `use_m`: `m` stays 0 while `step_i·dt < m_enable_ms`; the `m += 1` spike accumulation AND
  the decay are both skipped before enable; after enable, normal. So `apply_currents`/`membrane_terms`
  see `m=0` (no adaptation current) in the pre-enable window.
- Validation: `m_enable_ms` requires `use_m=True`; incompatible with `m_frozen_E`.

**TDD:**
1. `m_enable_ms=None` unit: `step()` behavior identical to current (m accumulates from step 0).
2. Engine byte-parity: full `simulate_kick` with `use_m,eta_m>0,m_enable_ms=None` == `m_enable_ms` absent.
3. Delayed: `m_enable_ms=100.0, dt=0.1` → `m==0` for steps 0..999, then accumulates; trace_m_mean flat 0 then rises.
4. Pre-enable adaptation current is 0 (apply_currents returns `I_E-I_I` in the window even with eta_m>0).
5. Existing `test_mz_slow_vars.py` + `test_mz_full_conductance_spatial_relay.py` green.
- [ ] RED → implement → GREEN → run full mz suites green → commit.

## Task 3 — `m_frozen_E` static-K control (mz_slow_vars.py) + TDD

**Files:** Modify `src/snn_engine/mz_slow_vars.py` (config + `__init__` inject, mirror `z_frozen_E`).

**Contract:** `m_frozen_E: np.ndarray|None = None`, length NE, ≥0. Requires `use_m=False` AND
`m_enable_ms=None`. Sets `self.m[:NE]=m_frozen_E` and never updates it → static `g_M` in
`membrane_terms`. Default None → byte-parity.

**TDD:**
1. `m_frozen_E=None` → byte-parity (existing green).
2. `m_frozen_E=const` in full_conductance → `_gM_mean_last` == `m_conductance_gain·eta_m·const/(v_match−e_k)`
   (static, no evolution over steps).
3. Validation raises: `m_frozen_E` with `use_m=True`, or with `m_enable_ms` set, or wrong shape/negative.
- [ ] RED → implement → GREEN → full mz suites green → commit.

## Task 4 — Phase 1 runner: delayed force-matched adaptation wedge

**Files:** Create `scripts/run_topic4_heo2_phase1.py` (mirror `run_topic4_mz_fcxr_heo1.py` scaffolding).

**Logic:**
- Build substrate(seed1) + frozen-Z field (D from the Phase-0 anchor, default D=0.15) + montage + HEO1
  baseline ref. Anchor cell config = its `_heo_cfg` (cooperative gate on).
- **Force-match**: run a short m-off high-state, get `E_spk_bool` + `I_EE` scale, `peak_m =
  replay_adaptation_peak(...)`, `eta_m = eta_m_from_frac(frac, I_EE, peak_m)` for frac∈{0.05,0.10}.
- **D=0 workpoint gate** per `(tau, frac)`: 4000 ms D=0, `use_m=True, m_enable_ms=None`, classify
  workpoint (must stay INTERICTAL). Drop m-params that erase baseline.
- **6 arms** on the anchor (D=anchor): `m_off`; `tau{250,750}×frac{5%,10%}` with `use_m=True,
  m_enable_ms=1000.0` (1 s establish then enable); `static_K` = `m_frozen_E` at the 10%-arm mean-m.
  T = `max(5000, 1000 + 5·tau)` ms (~5–6 s). seed1.
- Save per-arm `summary.json` + `traces.npz` (rate_E, lfp_trace, m_mean, gM_mean, coop_engaged_frac).
- setsid nohup, flock, sentinels (RUNNING/DONE/FAILED), resource_log, ≤2 workers.
- [ ] Smoke one arm (short) to verify wiring (m stays 0 for 1 s then rises; force-match runs). Commit runner.
- [ ] Launch the 6-arm screen via `setsid nohup`; poll to completion.

## Task 5 — Phase 1 classification (8 criteria) + figure

**Files:** add `classify_phase1_arm(traces, ref, m_off_arm) -> dict` to `topic4_mz_fcxr_heo2.py` + tests;
`scripts/plot_topic4_heo2_phase1.py`.

**Contract:** the spec §3 8 conjunctive criteria → per-arm verdict ∈ {transformed_broadband_spiky,
stalled, collapsed_sparse, silenced, unchanged_16Hz, unsafe}. `transformed` requires all 8.

**TDD:** synthetic traces for each verdict (establish-then-3Hz-spiky-broadband → transformed;
establish-then-decay-to-sparse → collapsed; establish-then-unchanged → unchanged_16Hz; etc.).
- [x] tests RED→GREEN→commit. Run classification over the 6 arms → `phase1_verdict.json`. **0/4 transformed.**
- [x] `plot_topic4_heo2_phase1.py`: per-arm (rate + m_mean, m-enable line) + six-band ΔdB vs real summary. Eyeballed. Committed `7caff82`.

## Task 6 — archive + STATUS + README + memory + finalize

- [x] `docs/archive/topic4/sef_hfo/mz_fcxr_heo2_2026-07-24.md` (bounded-negative: adaptation = brake not
  frequency-converter; gap = synchrony↔coverage vs desync↔broadband-spiky; next lever = spatial desync / HEO3).
  `broadband_diagnostic/STATUS.md` + `figures/README.md` (中文) written. Memory updated (project file + MEMORY.md).
- [x] Final: docs committed; 6 blessed SHAs unchanged; 17 heo2 + 94 existing tests green; no stray sim procs.

## Self-review

- Spec coverage: §1 target→Task1(`spectral_distance_to_real`/`REAL_E1146_DDB`)+Global; §2 Phase0→Task1;
  §3 Phase1 wedge→Tasks2-5; §4 engine→Tasks2-3+Global; §5 decision→Task5/6; §6 YAGNI→Global forbidden.
- No placeholders (each task has concrete signatures + TDD assertions + commands).
- Type consistency: `REAL_E1146_DDB`/`six_band_ddb` order `[1-4,4-8,8-13,13-30,30-80,80-150]` throughout;
  `m_enable_ms`/`m_frozen_E` names consistent Tasks 2-4.

# FCXR-HEO2.1 Closeout Plan (review P1 fixes before HEO3)

> Small closeout on `codex/topic4-mz-fcxr-heo1` responding to the HEO2 review's three P1s. NOT HEO3.
> Fixes metric conflation + control cleanliness + labels, then re-states claims at the review's §4 layer.

**Goal:** de-conflate the spatial readout, add clean controls, relabel the two strong arms, and rewrite the
HEO2 claims to the review's safe layer — so HEO3 (causal 2×2) starts on honest footing.

## Global constraints (unchanged from HEO2)
- 6 blessed engines never touched (SHA-guarded); non-blessed edits only; byte-parity default.
- setsid nohup / flock / sentinels / ≤2 workers for any sim; a clean bounded-negative is a valid result.
- No cohort claims; no "spatial desync proven"; no "seizure reproduced".

## P1-a — three de-conflated spatial readouts (compute-free; per-contact field IS stored, lfp_trace (T,15))
New in `src/topic4_mz_fcxr_heo2.py` on `band_db_field` (per-contact six-band ΔdB, bands 0–4 = 1–80 Hz, 5 = 80–150):
- `active_recruitment(ddb_field, thr_db=3.0) -> int`: # contacts with **any** band 0..4 ≥ thr_db.
- `broadband_coverage_1_80(ddb_field, thr_db=3.0, k_bands=3) -> int`: # contacts with **≥k_bands of bands 0..4** ≥ thr_db (EXCLUDES 80–150; the real E1146 target has 80–150 ≈ −1.2 dB, not up).
- phase synchrony: **reuse** `oscillation_probe(...)["coherence_med"]` (cross-contact coherence at the common freq) + `phase_span_deg`; dispersion = 1 − coherence_med. Do NOT reinvent.
- TDD incl. bad-data regressions: flat field → recruit 0 / broadband 0; all-bands +10 dB → 15 / 15; narrowband (only band-3 up) → recruit 15 / broadband 0; real-E1146 vector broadcast → 15 / 15.
- Recompute over the 48 Phase-0 cells + 6 Phase-1 arms → augment `phase0_state_map.json` + `phase1_verdict.json`
  with `active_recruitment`, `broadband_coverage_1_80`, `phase_coherence`. **Expected:** 16 Hz anchor recruit≈15,
  broadband≈0, coherence≈0.97 (widely-recruited-but-synchronous-narrowband — the honest decomposition).

## P1-c relabel (compute-free) — tail/state-segmented labels for the two strong arms
`segment_state_label(rate, fs, m_enable_ms, dt) -> str` on the post-enable window:
- `terminated_no_recovery_in_window`: high state early (max smoothed rate > 60), last 25 % of window mean < 5 Hz, no re-ignition. → replaces the wrong `unchanged_16Hz` for 750 ms/10 %.
- `intermittent_fast_bursting`: ≥3 on→off envelope crossings AND re-ignition after a silent gap AND event-IPI ≫ envelope freq. → replaces the wrong `collapsed_sparse` for 250 ms/10 %.
- TDD: synthetic terminate-and-stay → terminated; synthetic burst-gap-burst → bursting; sustained → neither.

## P1-c controls (2 new sims, parallel) — needs a delayed-static-K engine mode
- Engine (non-blessed `mz_slow_vars.py`): add `m_frozen_enable_ms: float|None=None`. With `m_frozen_E` set:
  None → inject at t=0 (current, byte-parity); set → m=0 until enable then m=m_frozen_E. Mirrors `m_enable_ms`
  gating. TDD + byte-parity (None ≡ current).
- **Mean-matched delayed static-K**: freeze m at the **post-enable time-mean** of the 750 ms/10 % dynamic arm's
  `trace_m_mean` (population mean, applied uniformly, NOT the p90 peak), delayed at m_enable_ms=1000. Isolates
  dynamics-vs-mean-load cleanly (the old peak-matched control over-applied K → silenced).
- **Extended slow-strong**: re-run 750 ms/10 % at T=9000 ms → distinguish true termination vs no-recovery-in-5 s.
- Also: report interictal-preservation as event stats (count/duration/participation/peak/irregularity) at the D=0
  workpoint for the 4 arms, not just the coarse workpoint label (the arms drop mean rate 37–72 %).

## P1-b claim rewrite (docs) — to the review §4 safe layer
Archive §3.3/§4/§5 + STATUS + memory: replace "coverage bound to synchrony" / "gap = sync-vs-desync" with:
- accepted: FCXR makes a bounded, sustained, widely-recruited but too-fast/too-synchronous high state; real
  spectral shape and wide sustained recruitment live in different state families; uniform slow-K adaptation on
  the tested anchor does not merge them; strong adaptation yields bursting (fast τ) or termination (slow τ) =
  reusable lifecycle parts.
- NOT claimed: seizure-consistent state; broadband proven to require spatial desync; interictal↔ictal loop;
  FCXR architecture disproven.
- next: bottleneck narrowed to **joint** control of recruitment + phase organization; spatial heterogeneity/desync
  = priority hypothesis, to be tested causally in HEO3 (homogeneous vs patchy adaptation × off vs on, global mean
  K held equal, + spatial-shuffle/patch-rotation control, requiring recruitment↑̸↓ ∧ dispersion↑ ∧ 1–80 coverage↑).

## Order
A (compute-free, commit): metrics + relabels + TDD → recompute 48+6 → augmented maps + figure.
B (engine + 2 sims): delayed-static-K mode + TDD → launch mean-matched + extended (parallel, nohup).
C (docs): re-analyze → figure → archive/STATUS/memory claim rewrite. Verify blessed/tests/tree.

# Stage 3 regime screen — frozen screen conclusion (NOT a Stage 3 verdict)

**Date:** 2026-06-14 · **Run:** `scripts/run_stage3_regime_screen.sh` (user-authored), 58/61 readouts (Phase 2 `twoend_deph` arm 3/3 failed, non-critical). Raw: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_screen/regime_screen_raw.csv`.
**Status:** DIAGNOSTIC SCREEN, frozen. **Does NOT enter any main doc. Stage 3 has NOT passed any gate.**

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

**Event typing (`scripts/analyze_stage3_event_types.py` → `event_types.csv`, `figures/stage3_event_typing.png`; 621 events across 60 two-foci runs) answers WHY most events are local:**
- **local ≠ weak nucleation.** Local events fire the source core HARD — `core_ignite_frac` median **0.287**, essentially the same as readable_global's 0.320 (70% of locals ≥0.10) — but light only 2–3 of 12 contacts. On the energy-vs-spread plot they sit BOTTOM-RIGHT (strong ignition, no spread = **contained propagation**), not bottom-left (low energy). So locality is a **propagation/containment** property of the cold op-point (m17.5 keeps collision low but the cool surround does not relay the wave outward), NOT a nucleation-energy deficit.
- **Refractory-shadow is NOT the main driver:** 0/379 local events fall within 250 ms after a readable_global, and local inter-event gap (median 268 ms) ≈ global (283 ms) — locals do not cluster at short gaps. (Competition/refractoriness may still modulate, but it is not what makes events local here.)
- **At the SOURCE level both ends nucleate, roughly balanced across the dataset:** neg 187 / pos 214 / both(collision) 187 / none 33. So "two-end nucleation" genuinely happens and is fairly balanced; the bottleneck is the READ-OUT (most events don't spread to ≥7 contacts) + the formal ≥50/end gate — both our instrument/criteria.

**Scientific statement (locked):** two-end excitable foci in one cm-SNN **do produce two-end nucleation (source-balanced) and a few readable forward/reverse propagation events**; but under the current coupling / drive / virtual-electrode / gate, **most spontaneous events stay local (strong ignition, contained propagation), and hotter/closer conditions turn them into co-ignition/collision.** This **supports the direction** "forward/reverse templates may arise from random nucleation at the two ends of one propagation axis" (directly supported at the source level), but does **NOT support** "a balanced, independent, long-timeseries two-source train." **Local events are mechanism information, not failure** — a "many small events + a few template-able large events" hierarchy, which is itself closer to real HFO than a forced balanced bidirectional train. Per-event fields available for re-analysis in `event_types.csv` (source / readout sign·n_part·axis_err / propagation_class / time_since_prev / previous_source / core_ignite_frac / within_recovery / core_onset_diff / duration).

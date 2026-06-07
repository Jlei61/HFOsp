# SEF-HFO 虚拟 SEEG 观测层 — Increment 3 Design (rate-model parity + heterogeneous-core → bi-model modality for Step 3)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:writing-plans → executing-plans.
> **STATUS 2026-06-07:** Tasks 1/2/3 DONE (TDD pass). Task 4 smoke in progress.
> Increment 3b (heterogeneous core) is a DESIGN whose full TDD is written when 3a lands + Step-3 mechanism is locked.

**Goal:** Prove the virtual-SEEG observation modality (Increment 1 contract + Increment 2 SNN
baseline) **runs correctly on the rate-field model** (3a, engineering parity), then use it to
**read out a heterogeneous-core rate field** (3b: nucleation location + propagation axis + channel
order). Once the modality reads cleanly on the rate field (3a), it becomes a **validated test
modality** that Step 3 (heterogeneity) consumes. SNN cm-scale confirmation is deferred (see §SNN
status below).

**Why this matters (the user's framing):** the heterogeneity case parameters ("where it's easy to
ignite") can only be compared to data through the observation space. A modality validated on the
rate field (natively cm-scale, deterministic, same connectivity length constants as SNN) is
trustworthy infrastructure for Step 3.

**Reuse:** the ENTIRE Increment-1/2 chain — `montage / sample_envelopes / extract_lagpat /
endpoint_centroid_axis / direction_readability`. The rate model needs only a thin adapter (field
frames → contacts) + one additive edit to expose frames. **Spec:** observation-layer spec §5/§6 +
pathology-mapping spec §5.2 (`Var(V_th,E)` patch) / §6 (LIF rate ∥ SNN isomorphism).

---

## §SNN status (corrected from earlier over-claim)

Increment-2 SNN status is **NOT "passed"**. Correct statement:

- **Toy-wave contract gate (Increment 1):** PASSED — `endpoint_centroid_axis` recovers known
  direction on synthetic traveling wave at real 3.5 mm spacing; 56 tests pass.
- **SNN small-scale oracle:** small-scale (L≈3mm, dense) anisotropy-front oracle confirmed
  anisotropic front at θ_EE=44.9° — the engine works.
- **SNN real-electrode-scale current-LFP smoke:** NOT PASSED. `density=100` at L=20mm placed
  only ~6 E neurons in the 0.15mm kick disk → no event ignited (`max_active_frac=0.000`,
  `n_part=1`). **Conclusion: this density/size parameter combination cannot read events; it does
  NOT prove SNN is incompatible with real-electrode-scale observation.** The per-contact-baseline
  readout fix (z-score vs pre-kick baseline, no on−off subtraction) is correct; the failure was
  purely the kick seed being too small for the sheet density chosen.
- **SNN real-scale next step (deferred, does NOT block rate 3a):** requires adequate density
  (≥1800/mm²); sheet size determined by the rate field's event extent (measured first via 3a);
  one informed run at those parameters, reading the oracle before the electrode pipeline. Until
  then SNN cm-electrode-scale = unresolved.

---

## §0. Contract

### Increment 3a — rate-model parity adapter (engineering validation, Tasks 1-4)

- **Same interface, rate substrate.** The rate field already produces `r_E` on an `n×n` grid
  (`src.sef_hfo_lif.integrate_lif_field`, anisotropic E→E kernel with rotatable `theta_EE`).
  Sample it at montage contacts with the SAME `sample_envelopes` (it natively takes grid frames +
  grid coords), then the SAME `extract_lagpat → endpoint_centroid_axis`.

  **ESTIMATOR: `endpoint_centroid_axis` (sparse-friendly).** NOT `onset_front_axis` — that
  estimator requires a dense-neuron oracle to recover the anisotropic lobe and failed on sparse
  montages (WF-A lesson). `endpoint_centroid_axis` = centroid(k_dir earliest-rank contacts) →
  centroid(k_dir latest-rank contacts); works on sparse 2-shaft montages; same estimator used
  in all Increment-2 Option-B code. The four controls use this estimator with the same locked
  thresholds (AXIS_ERR_MAX=25°, KDIR=3, PART_MIN=7).

- **One additive edit (LOCKED, backward-compatible, TRACKED, DONE):** `integrate_lif_field(...,
  return_frames=False)` — when True, also returns the per-timestep field stack `rE_frames
  (nsteps, n, n)` as the LAST element. Default False = byte-identical to today. TDD PASSED
  (`tests/test_sef_hfo_lif.py::test_return_frames_position_and_last_frame`).

- **Finite-pulse kick = tracked `pulse_stim_fn(center,...)` in `src/sef_hfo_rate_adapter.py`
  (DONE):** drives `integrate_lif_field`, NOT `src.sef_hfo_pulse` (that drives the old sigmoid
  `integrate_field` — wrong substrate). TDD PASSED
  (`tests/test_sef_hfo_rate_adapter.py::test_pulse_stim_fn_disk_placement_and_gating`).

- **`rate_event_envelope` adapter (DONE):** `rate_event_envelope(rE_frames, n, L, montage,
  kernel_width)` reshapes frames and calls `sample_envelopes(grid_xy=grid_coords(n,L))`. TDD
  PASSED (core property + wiring guard). Grid-alignment contract: C-order ravel + ij-indexed
  `_grid` = same convention as `grid_coords`.

- **Same four controls, same estimator (`endpoint_centroid_axis`), same thresholds as
  Increment 2** (reframe applies identically):
  - C-track: CENTER pulse, rotate `theta_EE`∈{0,45,90} → axis tracks `theta_EE` (err<25°)
  - Kick-track: theta_EE fixed, OFF-center pulse → axis stays at theta_EE
  - Shaft-invariance: rotate montage → axis stays
  - Iso must-fail: `ell_par==ell_perp`, CENTER pulse → n_part<7 OR no readable axis
  **UNITS:** call sites pass `np.deg2rad(deg)` — `integrate_lif_field`'s `theta_EE` is radians.
  **Operating point:** Step-0b candidate window (deterministic pulse gives
  `self_limited_propagation`; the homogeneous rate field's noise-spontaneous events were
  honest-NULL, so parity uses the DETERMINISTIC pulse, not noise).

- **Acceptance = engineering parity, not a new science claim:** chain runs on rate field AND
  `endpoint_centroid_axis` gives SAME qualitative verdict (axis tracks θ_EE, iso fails / n_part
  low). Confirms modality is substrate-validated. **No homogeneous-rate-field science claim.**

### Increment 3b — heterogeneous-core rate field (Step-3 read-out) — DESIGN only

- **The mechanism (pathology-mapping spec §5.2):** add a heterogeneity patch — narrow
  `Var(V_th,E)` at a location `x_patch`, radius `r_patch`. Effect on local gain COMPUTED at op,
  NOT presupposed. Implementation = spatial operating-point field. Detailed TDD when Step-3 params
  lock.
- **Mean-matched control (REQUIRED, pathology-mapping §5.0 contract-2):** narrowing `Var(V_th,E)`
  shifts mean rate via Jensen (curve convexity). Every heterogeneity read has two layers: (a) raw
  patch, (b) mean-rate-matched control. Only (a)−(b) is the distribution-narrowing effect.
- **What the virtual electrodes must read out (three Step-3 observables):**
  1. **Nucleation position** — earliest-onset contacts' centroid localizes at/near `x_patch`;
     moves when `x_patch` moves.
  2. **Propagation axis** — `endpoint_centroid_axis` still tracks `theta_EE` (connectivity sets
     AXIS; heterogeneity sets WHERE it ignites). Two dissociable knobs.
  3. **Channel order / template** — per-event rank template stable across events seeded at patch.
- **Acceptance (Step-3 tier):** nucleation centroid within `r_patch + margin` of `x_patch` (moves
  with `x_patch`); axis tracks `theta_EE` independent of `x_patch`; template stability above
  matched null; all read as raw−(mean-matched control). **Circularity red line:** `x_patch` is a
  model knob, NOT fitted from observed templates.

### The rate-field modality confirmation (deliverable that gates Step 3)

The observation layer is declared **validated for Step-3** iff the rate-field 3a smoke + four
controls pass with `endpoint_centroid_axis` (AXIS_ERR_MAX=25°, PART_MIN=7). SNN cm-scale
confirmation (when it runs at adequate density) is additional evidence but does NOT block Step 3.

---

## File structure

- **`src/sef_hfo_lif.py`** — `return_frames` additive opt-in. **DONE.**
- **`src/sef_hfo_rate_adapter.py`** — `pulse_stim_fn` + `rate_event_envelope`. **DONE.**
- **`tests/test_sef_hfo_lif.py`** — `test_return_frames_position_and_last_frame`. **DONE.**
- **`tests/test_sef_hfo_rate_adapter.py`** — Tasks 2+3 TDD. **DONE.**
- **`scripts/run_sef_hfo_obs_increment3a_smoke.py`** — rate-field smoke + four controls runner.
  **Task 4 — IN PROGRESS.**
- **(3b, later)** spatial-heterogeneity field module + Step-3 read-out runner.

---

## Increment 3a remaining tasks

- **Task 4 — rate smoke + four controls (CURRENT):** rate-field `endpoint_centroid_axis` smoke at
  θ_EE=45°, real 3.5mm contacts, two non-parallel shafts, same lag→axis chain as Increment 2.
  Produces: verdict JSON + "electrode overlay + per-contact trace + recovered axis" diagnostic
  figure (`plot_readout_diagnostic`, §13). Smoke clean → then four controls.
- **Task 5 — bi-model modality verdict (after Task 4):** short doc/JSON asserting "rate-field
  observation modality validated" — gates Step 3. SNN cm-scale is listed as "deferred pending
  adequate-density design."

## Increment 3b (design only; full TDD when Step-3 params lock)

Spatial-heterogeneity field + read out nucleation centroid + axis + template stability via the
SAME `endpoint_centroid_axis` chain. Acceptance + circularity red line per §0. Do not implement
until 3a passes AND Step-3 heterogeneity mechanism is locked.

---

## Self-Review / open decisions

- **Two repeated-mistake guards (hard lock):** (1) all controls use `endpoint_centroid_axis`, NOT
  `onset_front_axis` (sparse-montage failure; WF-A lesson); (2) always drive `integrate_lif_field`,
  never `sef_hfo_pulse.run_pulse` (sigmoid `integrate_field`, wrong substrate).
- **3a uses DETERMINISTIC pulse** (homogeneous noise → honest-NULL; parity ≠ noise-spontaneity).
- **SNN deferred:** until rate-field 3a confirms event-extent, SNN cm-scale parameters are
  undetermined. The rate field event-extent measurement in Task 4 informs the eventual SNN
  density/sheet-size design.
- **Open for user:** (1) Is engineering parity on the rate field the gate for Step 3, or do you
  want bi-model (SNN + rate) confirmation first? (2) 3b nucleation-localization null — lock when
  Step-3 starts.

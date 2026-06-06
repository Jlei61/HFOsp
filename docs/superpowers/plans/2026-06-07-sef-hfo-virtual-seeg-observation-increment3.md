# SEF-HFO 虚拟 SEEG 观测层 — Increment 3 Design (rate-model parity + heterogeneous-core → bi-model modality for Step 3)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:writing-plans → executing-plans. **STATUS: design 2026-06-07, for USER REVIEW.** Increment 3a (parity) is execution-ready after a smoke; Increment 3b (heterogeneous core) is a DESIGN whose full TDD is written when 3a lands + Step-3 mechanism is locked.

**Goal:** Prove the virtual-SEEG observation modality (Increment 1 contract + Increment 2 SNN slice) **runs identically on the rate-field model** (3a, engineering parity), then use it to **read out a heterogeneous-core rate field** (3b: nucleation location + propagation axis + channel order). Once the modality reads cleanly in BOTH models (SNN ✓ Increment 2, rate ✓ 3a), it becomes a **validated test modality** that Step 3 (heterogeneity) consumes.

**Why this matters (the user's framing):** the heterogeneity case parameters ("where it's easy to ignite") can only be compared to data through the observation space. A modality that reads direction the same way in two independent model substrates (spiking + rate) is trustworthy infrastructure for Step 3; one that only works in one substrate is suspect.

**Reuse:** the ENTIRE Increment-1/2 chain — `montage / sample_envelopes / extract_lagpat / onset_front_axis / direction_readability / event_window_for_run`. The rate model needs only a thin adapter (field frames → contacts) + one additive edit to expose frames. **Spec:** observation-layer spec §5/§6 + pathology-mapping spec §5.2 (`Var(V_th,E)` patch) / §6 (LIF rate ∥ SNN isomorphism).

---

## §0. Contract

### Increment 3a — rate-model parity adapter (engineering validation FIRST)

- **Same interface, rate substrate.** The rate field already produces `r_E` on an `n×n` grid (`src.sef_hfo_lif.integrate_lif_field`, anisotropic E→E kernel with rotatable `theta_EE`). Sample it at montage contacts with the SAME `sample_envelopes` (it natively takes grid frames + grid coords), then the SAME `extract_lagpat → onset_front_axis`.
- **One additive edit (LOCKED, backward-compatible, TRACKED code):** `integrate_lif_field(..., return_frames=False)` → when True, also return the per-timestep field stack `rE_frames (nsteps, n, n)`. Today it returns only the final/peak snapshot + `ext`. Default False = byte-identical to today. (Unlike the gitignored SNN engine, this edits tracked `src/sef_hfo_lif.py` — committable.)
- **Finite-pulse kick, parametrized center (LOCKED):** `src.sef_hfo_pulse._disk_mask` is center-only; add a `center` param (default origin → identical) so C-track/iso use a CENTER disk and kick-track uses OFF-CENTER (mirrors the SNN `kick_center` fix — same confound discipline).
- **Same four controls, same estimator, same thresholds as Increment 2** (reframe applies identically): C-track (CENTER pulse, rotate `theta_EE`∈{0,45,90}) → undirected `onset_front_axis` tracks `theta_EE` (err<25°, ratio>1.3); kick-track (theta_EE fixed, OFF-center pulse) → axis stays at theta_EE; shaft-invariance (rotate montage) → axis stays; iso must-fail (`ell_par==ell_perp`, CENTER pulse) → ratio<1.3. **Operating point: a Step-0b candidate window** (the deterministic pulse produces `self_limited_propagation` — the rate field's noise-spontaneous events were honest-NULL, so the rate-parity uses the DETERMINISTIC pulse, not noise).
- **Acceptance = engineering parity, not a new science claim:** the chain runs on the rate field AND the discriminator gives the SAME qualitative verdict as the SNN (axis tracks `theta_EE`, iso fails). This confirms the modality is substrate-independent. **No homogeneous-rate-field science claim** (D5: homogeneous rate = engineering parity only).

### Increment 3b — heterogeneous-core rate field (the Step-3 heterogeneity read-out) — DESIGN

- **The mechanism (pathology-mapping spec §5.2):** add a low-threshold-heterogeneity patch — narrow `Var(V_th,E)` at a location `x_patch`, radius `r_patch` — which (computed at the operating point, NOT presupposed) raises local gain → a **nucleation hotspot** ("where it's easy to ignite"). Implementation = a **spatial operating-point field** (`op` varies per pixel; the deferred "0e" layer `Φ_eff = ∫Φ_LIF(μ,σ;θ)p(θ)dθ` with `p` narrower inside the patch). This is a real model extension (Step-3 mechanism) — its detailed TDD is written when Step-3 params lock; here we fix what the OBSERVATION must read.
- **What the virtual electrodes must read out (the three Step-3 observables):**
  1. **Nucleation position** — across noise-triggered (or near-threshold) events, where do events START? The read-out (earliest-onset contacts' centroid across events) should localize **at/near `x_patch`**, not uniformly. Control: move `x_patch` → the read-out nucleation centroid moves with it (the pathology-mapping spec's "移动 patch 后 source density 随之移动").
  2. **Propagation axis** — `onset_front_axis` should still track `theta_EE` (heterogeneity sets WHERE it ignites; connectivity still sets the AXIS). Two knobs, two read-outs, dissociable.
  3. **Channel order / template** — the per-event rank template (through the real pipeline) should be **stable across events** seeded at the patch (the H1/H3 identity-bias + stable-geometry phenomenon), readable as a reproducible source→sink ordering for same-end seeding.
- **Acceptance (Step-3 tier, encode the conclusion):** nucleation centroid within `r_patch + margin` of `x_patch` (and moves when `x_patch` moves, beats a uniform-nucleation null); axis tracks `theta_EE` independent of `x_patch`; template rank stability (split-half / repeat-seed Jaccard) above a matched null. **Circularity red line (spec §5/§7):** `x_patch` is set as a model knob, NOT fitted from observed templates; the read-out is COMPARED to it, never used to set it.

### The bi-model modality confirmation (the deliverable that gates Step 3)

- The observation layer is declared a **validated Step-3 test modality** iff the SAME interface + SAME estimator + SAME thresholds give a consistent direction read in BOTH substrates: **SNN (Increment 2)** and **rate field (Increment 3a)**. Cross-model parity (axis-tracks-θ_EE, iso-fails in both) is the gate — a substrate-specific pass is NOT enough.
- Only after that does 3b (heterogeneous-core read-out) carry weight as the Step-3 heterogeneity observation, because we then know the read-out reflects the field, not the substrate or the electrodes.

---

## File structure

- **Modify** `src/sef_hfo_lif.py` — additive `return_frames` on `integrate_lif_field` (3a).
- **Modify** `src/sef_hfo_pulse.py` — `center` param on `_disk_mask` / `run_pulse` (3a kick-track).
- **Create** `src/sef_hfo_rate_adapter.py` — `rate_event_envelope(rE_frames, n, L, montage, kernel_width)` → reuse `sample_envelopes(grid_xy=grid_coords(n,L))` (3a; pure, unit-testable on a synthetic field).
- **Create** `scripts/run_sef_hfo_obs_increment3a.py` — rate-field discriminator runner (the four controls; mirrors Increment 2; cross-model parity panel vs the SNN verdict).
- **(3b, later)** spatial-heterogeneity field module + Step-3 read-out runner — full TDD when Step-3 params lock.

---

## Increment 3a tasks (execution-ready after a smoke)

- **Task 1 — `return_frames` (TDD):** failing test = `integrate_lif_field(..., return_frames=True)` returns an `(nsteps, n, n)` stack whose last frame equals the `return_field=True` snapshot. Implement (accumulate `rE` per step into a list, stack, append to the return tuple). Commit.
- **Task 2 — `_disk_mask`/`run_pulse` center param (TDD):** failing test = off-center `center=(cx,cy)` puts the disk there; `center=None` → origin (identical). Commit.
- **Task 3 — `rate_event_envelope` adapter (TDD, synthetic field):** a synthetic traveling-Gaussian field (the Increment-1 `traveling_wave` source IS already grid frames!) → `sample_envelopes(grid_xy=grid_coords(n,L))` → `extract_lagpat → onset_front_axis` recovers the imposed direction (ratio>1.3, err<25°). This reuses Increment-1 machinery almost entirely — the adapter is a ~10-line wrapper. Commit.
- **Task 4 — rate discriminator runner + smoke + freeze:** mirror Increment-2 Task 5/6 on the rate field (deterministic pulse at a Step-0b candidate op). Smoke on θ_EE=45° (estimator pre-lock + knob freeze), then the four controls. Verdict JSON + figure + Chinese README. **Cross-model parity panel:** put the rate-field per-θ_EE axis errors next to the SNN ones — they should agree.
- **Task 5 — bi-model modality verdict:** a short doc/JSON asserting "modality validated in both substrates" iff SNN (Inc2) AND rate (3a) both pass; this is the artifact Step 3 cites.

## Increment 3b (design only here; full TDD when Step-3 params lock)

- Spatial-heterogeneity field (per-pixel `op` via narrowed `p(V_th,E)` in the patch); noise/near-threshold triggering; read out nucleation centroid + axis + template stability through the SAME observation chain. Acceptance + circularity red line per §0. **Do not implement until 3a's bi-model parity passes and the Step-3 heterogeneity mechanism is locked in the pathology-mapping spec.**

---

## Self-Review / open decisions

- **3a is mostly free:** the rate field is already grid frames, so `sample_envelopes` applies directly; the only new code is `return_frames` (additive, tracked), the `center` param, and a ~10-line adapter. The four controls + estimator + thresholds are reused verbatim from Increment 2.
- **3a uses the DETERMINISTIC pulse**, not noise (the homogeneous rate field's noise-spontaneous events were honest-NULL, 2026-06-04). Parity is about the read-out chain, not re-litigating noise-spontaneity.
- **3b is the Step-3 bridge, not this round's execution.** It needs the spatial-op heterogeneity layer (deferred "0e") which is a Step-3 mechanism; surface it here so the observation contract (nucleation/axis/template + circularity red line) is fixed before the mechanism is built.
- **Open for user:** (1) is engineering parity (3a same qualitative verdict as SNN) the right bar, or do you want a quantitative axis-error agreement threshold between substrates? (2) 3b nucleation-localization null (uniform-nucleation vs shaft-matched) — lock when Step-3 starts.

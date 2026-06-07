# SEF-HFO Increment-2 Option-B Execution Plan (single-end kick + endpoint-centroid, geometry-smoke first)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.
> **STATUS framing (user-locked 2026-06-07):** the core read-out ROUTE is switched correctly (Option B); this is NOT "Increment 2 complete". Next = a clean geometry smoke. This plan REPLACES the center-kick Task-5/6 of `2026-06-07-...-increment2.md` (that file is audit-only for the superseded onset_front_axis approach). No more patching the old plan.

**Goal:** Geometry-smoke the validated Option-B read-out so all direction conditions read cleanly, freeze the geometry, advisor-review, THEN run the full four-control. Only four steps (below); nothing else.

**Why B (settled):** the SNN center-kick event IS elongated along θ_EE (dense-neuron oracle `anisotropy_front.principal_axis` = 44.9°, ratio 3.65, err 0.1° — model+kick fine). `onset_front_axis` (early-contact SET principal axis) read 100° (err 55°) on 16 sparse contacts — it needs dense sampling no electrode array has, so it is demoted to a dense-oracle parity reference, NOT a sparse gate. **Option B** = the sparse-friendly Increment-1 estimator: single-end kick (unidirectional front) → `endpoint_centroid_axis` (source→sink; compared UNDIRECTED mod-180° vs θ_EE). Validated: θ=45 → err **6.4°**, readability 0.76, n_part 16. The reframe CLAIM ("connectivity sets the undirected axis; seed sets the sign") is UNCHANGED.

---

## §0. Contract (LOCKED)

- **Estimator** = `endpoint_centroid_axis` (Increment-1, exists; k_dir=3, eps_deg=0.5·pitch, degenerate/collinear→None), compared to θ_EE **undirected (mod 180°)** via `axis_angle_error_deg`. `direction_readability` (max-axis Spearman) reported alongside. `onset_front_axis` is NOT used as a gate (dense-oracle parity only).
- **Kick** = single-end, via the patched `simulate_kick(kick_center=...)`. C-track: kick at the θ_EE end. **kick-track (LOAD-BEARING)**: θ_EE FIXED, kick OFFSET PERPENDICULAR to θ_EE (so the unidirectional wave SWEEPS the montage, not off-sheet) — the recovered undirected axis must STAY at θ_EE, NOT follow the kick. Two end-kicks = H2 forward/reverse (secondary).
- **Verdict thresholds — NEVER tuned**: axis err **< 25°**; `k_dir = 3` (so participation gate ≥ 2·k_dir+1 = 7); `n_seed_min` (per condition); iso ratio/readability `< τ_fail = 0.3`. The geometry smoke tunes ONLY observation/substrate geometry (montage pitch / contact count / placement, kick end-distance, sheet L, density) — never these bars, never seed-picking.
- **iso reality (report honestly, do NOT force):** AR=1 fizzles (anisotropic reach is necessary for propagation — spiking_gt_validation). So the iso must-fail in the SNN is "few/no events → INSUFFICIENT", which itself supports "anisotropy necessary"; report it that way, never as a clean "event-but-no-direction".

---

### Task 1 — Tracked Option-B smoke runner

**Files:** Create `scripts/run_sef_hfo_obs_inc2_smoke.py` (TRACKED).

- [ ] **Step 1:** Write a runner that, given a geometry config `(L, density, pitch, n_contacts, shaft_angles, kick_end_frac)`, builds the net (`build_connectivity_rot`), single-end kicks, runs the full chain (`snn_event_envelope → event_window_for_run → extract_lagpat → endpoint_centroid_axis`), and reports **per condition**: `n_part`, `axis_err` (undirected vs θ_EE), `readability`, `event_window`. Conditions: C-track θ_EE∈{0,45,90} (kick at the θ_EE end); kick-track θ_EE=45 with kick offset perpendicular at a few distances; iso AR=1.
- [ ] **Step 2:** The kick-track kick positions are **perpendicular offsets** from the montage center: `kick = center + d·(−sinθ_EE, cosθ_EE)` for `d ∈ {±off}` — so the θ_EE-directed wave crosses the montage (the 90-end kick failed because it sent the wave off-sheet). Output each kick position's `n_part` + recovered axis.
- [ ] **Step 3:** Emit `smoke_verdict.json` (per-condition table) + print a one-line PASS/FAIL per the §2 criteria. Commit the runner (tracked).

---

### Task 2 — Small-grid geometry smoke (observation/substrate geometry ONLY)

- [ ] **Step 1:** Sweep a SMALL grid over **geometry only**: `pitch ∈ {…}`, `n_contacts ∈ {…}`, montage `placement/shaft_angles`, `kick_end_frac`, `sheet L ∈ {3,4}`, `density`. NEVER the verdict bars (25°/k_dir/τ_fail) and NEVER seed selection. Build is the cost → reuse a built net across the cheap observation-knob sub-sweep where possible (the diagnostic pattern: build once per (θ,AR), sweep montage/front knobs on the saved spikes).
- [ ] **Step 2:** Goal config = one geometry where **θ=45 AND θ=90 both have ≥7 participating contacts AND axis err < 25°**, kick-track stays at θ_EE, iso is honestly fizzle/INSUFFICIENT. Record the full grid to `smoke_grid.json` (so the choice is auditable, not cherry-picked).

---

### Task 3 — Verify the three things

- [ ] **θ tracking:** θ_EE=45 AND θ_EE=90 each: `n_part ≥ 7` AND `axis_err < 25°` (undirected). (θ=0 too if in scope.)
- [ ] **kick-track (no seed-following):** θ_EE fixed 45°, the perpendicular-offset kicks each recover an axis within 25° of **45°** (NOT of the kick direction) — i.e. the axis follows connectivity, not the seed.
- [ ] **iso honest:** AR=1 reports fizzle/INSUFFICIENT (too few contacts) or ratio/readability < τ_fail — reported as "supports anisotropy-necessary", never forced to a clean must-fail.
- [ ] If any of the three can't be met by ANY geometry in the small grid → **STOP + report** (it means sparse virtual electrodes can't read this event's direction even with B; a genuine finding). Do NOT widen thresholds or hand-pick.

---

### Task 4 — Freeze → advisor → full four-control

- [ ] **Freeze** the winning geometry into the runner constants + record in `smoke_verdict.json::frozen_geometry`. No geometry change after this.
- [ ] **Advisor review** of the frozen smoke result (per user instruction: advisor before the full run).
- [ ] **THEN** the full four-control run (the Increment-2 §6 four controls, with `endpoint_centroid_axis` + single-end kick + the frozen geometry): C-track θ_EE∈{0,45,90}, kick-track (perpendicular offsets), shaft-invariance (rotate montage), iso must-fail. Verdict JSON + figure + Chinese README. INSUFFICIENT conditions surfaced, never forced.

---

## Self-review / discipline

- **Not "complete":** the route is switched correctly + θ=45 validated; "Increment 2 complete" requires Task 4's full four-control to pass with frozen geometry.
- **Geometry vs verdict:** the smoke tunes geometry (pitch/count/placement/kick-end/L/density); the verdict bars (25°/k_dir=3/τ_fail/n_seed_min) are immutable; no seed-picking. The grid is recorded (`smoke_grid.json`) so the frozen choice is auditable.
- **iso fizzle is a finding, not a fix.** Report it; it supports anisotropy-necessary.
- **kick-track is load-bearing** (single-end reintroduces the seed confound); its kick positions are perpendicular offsets so the wave sweeps the montage.
- Reuse: `endpoint_centroid_axis` / `axis_angle_error_deg` / `direction_readability` / `snn_event_envelope` / `event_window_for_run` all exist; `kick_center` patched. No new estimator code.

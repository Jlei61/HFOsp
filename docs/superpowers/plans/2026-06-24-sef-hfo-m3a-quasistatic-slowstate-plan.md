# M3A-A1 quasi-static slow-state mechanism plan

> Status: new hard-boundary plan, 2026-06-24.
> Scope: M3A only. This plan tests whether fixed slow-variable states can make the SNN spontaneously produce distinguishable interictal-like vs seizure-like event phenotypes.

## 0. Hard Boundary

M3A answers only this question:

```text
slow-variable state s  ->  spontaneous event phenotype
```

M3A does not define `W`, does not use `h(W)`, and does not use `W`-coupled threshold permissivity as a mechanism. The old static `V_th_eff = V_th0 - delta * mu * h(W)` result is treated as negative evidence: it raised event rate but did not change event size, duration, or R-class in a mechanistically useful way.

Primary input is no-kick spontaneous activity. Kick or finite-pulse probes are allowed only as secondary stability checks, not as the definition of seizure-like transition.

## 1. Scientific Goal

Test whether a biologically motivated slow state can move the SNN through at least two distinguishable regimes:

- interictal-like returned finite events: R2/R3, return-to-baseline preserved;
- seizure-like sustained recruitment: R4a only, with spatial/recruitment structure still present;
- tonic full-field runaway: R4b, explicitly not accepted as seizure-like bridge.

M3A-A1 is quasi-static. It freezes or clamps slow-state values and asks: if the tissue were already at this slow state, would spontaneous events change phenotype?

## 2. Mechanism Candidates

Priority order:

1. `e_GABA` / depolarizing GABA / chloride accumulation proxy.
2. `z` disinhibition: current-based `I_net = I_E - z * I_I`, where lower `z` weakens inhibition.
3. `phi` adaptive threshold.
4. `g_K` sAHP outward current.

`src/snn_engine/slow_vars.py` contains placeholder parameter values. The plan must not draw biological conclusions from defaults. Defaults are only smoke-test starting points.

## 3. Engine Path Gate

Before any scientific run:

- confirm the active worktree has `SlowVars` hooks in the execution path being used;
- current mainline `src/snn_engine/model.py` supports `simulate(..., slow=...)`;
- current mainline `src/snn_engine/kick_probe.py` also exposes `simulate_kick(..., slow=...)`, but this must be rechecked in the target worktree;
- `slow=None` must be bit-parity with the previous no-slow baseline;
- if `model.simulate()` lacks full event readouts needed for R0-R4 classification, using `simulate_kick(KICK_BOOST=0, t_kick=1e9, slow=...)` is acceptable, but the run must be documented as no-kick spontaneous.

Do not implement new engine semantics until the path audit says which hook is missing. If a hook must be added, it is off by default and requires parity tests.

### Engine-path audit RESULT (2026-06-24, this worktree; src/snn_engine is git-tracked so it carries over)

- `kick_probe.py::simulate_kick(p, net, KICK_BOOST, slow=None, ...)` **DOES wire slow** (calls
  `slow.apply_currents / slow.threshold / slow.step`, lines 230-251); `model.py::simulate()` also does.
  Either path works for no-kick spontaneous (`simulate_kick(KICK_BOOST=0.0, t_kick=0.0, slow=...)`).
- `SlowVars` (slow_vars.py) `apply_currents/threshold/step` are **implemented** (z/φ/g_K equations present);
  only the *parameters* are PLACEHOLDER (must calibrate).
- **Smoke-verified (L8)**: `slow=None` is BYTE-IDENTICAL to no-slow-arg (bit-parity ✓);
  `slow=SlowVars(z=0.3)` changes spikes (125 → 404988 → z works but z=0.3 is uncalibrated runaway);
  static depolarized `e_gaba` (`shunt_gaba=True, e_gaba=...`) changes activity with **no engine change**.
- **Therefore**: Task 1 bit-parity gate is already met for `simulate_kick(slow=)`. The existing
  `scripts/run_m3_static_mu_spontaneous.py` (= `simulate_kick(KICK_BOOST=0)`) only needs `--slow-mode /
  --e-gaba / --shunt-gaba` pass-through to become the A1 runner — NO new engine semantics for
  quasi-static z/φ/g_K or static e_GABA. **Only DYNAMIC e_GABA (Cl⁻ accumulation, the prime A2 candidate)
  needs a new SlowVars e_GABA state + per-neuron time-varying e_gaba in membrane_step (off-by-default).**
- Concrete A1 first step (lowest effort, no engine change): **quasi-static e_GABA scan = sweep the
  `e_gaba` param (`shunt_gaba=True`)** over Task-4's levels via the spontaneous runner; z/φ/g_K
  quasi-static = `SlowVars` with frozen values. Calibrate before any conclusion (z=0.3 already runaway).

## 4. Outputs

Canonical output root:

```text
results/topic4_sef_hfo/m3a_slowvars/quasistatic/
```

Required files:

- `config.json`: engine SHA, Params, substrate, slow-state values, detector thresholds.
- `per_event.csv`: one row per spontaneous event.
- `slow_state_samples.csv`: state at `pre`, `onset`, `peak`, `end`, and post-event windows.
- `summary.json`: event rate, size, duration, return probability, R-class fractions.
- `figures/README.md`: Chinese description for every generated figure.

Required per-event fields:

```text
event_id, seed, state_label, onset_ms, end_ms, duration_ms,
size_bins, active_mass, return_to_baseline,
R_class, sustained_front_score,
z_pre, z_onset, z_peak, z_end,
phi_pre, phi_onset, phi_peak, phi_end,
gK_pre, gK_onset, gK_peak, gK_end,
e_gaba_pre, e_gaba_onset, e_gaba_peak, e_gaba_end
```

Fields can be `NA` only when the mechanism is not active or not implemented; do not write unknown as 0.

## 5. Tasks

### Task 0: Freeze the M3A-A1 contract

- [ ] Write a short `STATUS.md` saying A1 excludes W/h(W), excludes external kick as primary, and treats static vth-mu as negative evidence.
- [ ] Record the old negative boundary: rate-only increase is not success.
- [ ] Record R4a vs R4b definitions before running.

### Task 1: Engine-path and bit-parity audit

- [ ] Add or run tests proving `slow=None` is bit-identical for the chosen no-kick execution path.
- [ ] Confirm `SlowVars` can alter dynamics when enabled.
- [ ] Confirm the runner can emit full event traces needed by R0-R4 classification.
- [ ] If the target worktree lacks `slow` support in `simulate_kick`, add only an off-by-default hook and re-bless parity.

Pass condition: no-slow path unchanged; slow-on path measurably changes either currents, threshold, or spikes in a small smoke case.

### Task 2: Quasi-static state helper

- [ ] Implement a small helper that can freeze or clamp slow-state values per neuron/bin without using W.
- [ ] Support at least:
  - fixed `z`;
  - fixed `phi` offset;
  - fixed `g_K`;
  - fixed `e_GABA` if the conductance-shunt path is available.
- [ ] Add unit tests for sign semantics:
  - lower `z` weakens inhibition;
  - higher `phi` raises threshold;
  - higher `g_K` suppresses excitability;
  - depolarized `e_GABA` makes inhibition less protective only in the shunting path.

### Task 3: No-kick spontaneous detector reuse

- [ ] Reuse existing event detection / R0-R4 classification where possible.
- [ ] Ensure the detector reads raw spontaneous activity, not kick-minus-sham fields.
- [ ] Ensure R4a requires sustained/recurrent recruitment with nontrivial front or spatial structure.
- [ ] Ensure R4b tonic saturation is reported but never counted as seizure-like success.

### Task 4: Tiny quasi-static sweep

Run a small, cheap pilot before any full grid:

```text
substrate: one accepted Stage-3/M3 base substrate
T: 8-20 s per seed
seeds: 3-5
states:
  e_GABA: baseline plus 3-5 depolarized levels, if available
  z: 1.0, 0.8, 0.6, 0.4
  phi: baseline plus 2-3 offsets
  g_K: 0 plus 2-3 levels
```

Only one slow variable is varied at a time. Other mechanisms remain off or baseline.

### Task 5: A1 verdict

Answer these six questions:

1. Does event rate change?
2. Does event size distribution shift?
3. Does duration distribution shift?
4. Does return probability drop?
5. Does R-class composition change toward R4a?
6. Does the effect differ from simple rate-only heating?

Success requires more than event-rate increase. A minimal A1 candidate needs size/duration or return/R-class movement in addition to rate change.

Failure modes:

- rate increases but size/duration/R-class stay flat: repeat of static vth-mu negative result;
- only R4b appears: tonic runaway, not seizure-like bridge;
- detector threshold changes alone explain the R-class shift: invalid;
- all variables are silent or unstable: no quasi-static mechanism support.

## 6. Stop Rule

Stop after A1 tiny pilot and write a recap before A2. Do not start dynamic slow-variable runs until a slow-state range has either:

- a positive phenotype-shift candidate, or
- a clearly documented negative boundary worth testing dynamically for history effects.


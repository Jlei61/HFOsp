# M3B W / W_eff field-readout and SEEG bridge plan

> Status: rewritten as M3B after the 2026-06-24 M3 split.
> Supersedes the old single-line "local-W + W-coupled slow permissivity" plan.
> Scope: M3B only. M3B asks whether W / W_eff can quantify field structure and bridge model output to SEEG propagation axes. It does not test the slow-variable seizure mechanism.

## 0. New Hard Boundary

The old M3 plan mixed two different questions:

```text
W measurement -> h(W) threshold permissivity -> slow state -> seizure-like transition
```

That chain is no longer allowed as the main M3 design.

New split:

- **M3A**: slow-variable mechanism. Does the SNN spontaneously produce interictal-like vs seizure-like discharge differences as slow state changes?
- **M3B**: W / W_eff readout. Can W quantify field scaffold / effective recruitment and connect model axes to SEEG interictal and ictal rank axes?

M3B may later consume M3A's slow-state variable `s_slow`, but it does not invent `s_slow` and does not use `h(W)`-coupled threshold as the mechanism.

## 1. What Is Retained From The Old Plan

Retain:

- base SHA / bit-parity discipline;
- `W_resp / W_step / W_shape` separation;
- source-column, target-row direction convention: `W[p, q] = q -> p`;
- kick / event-aligned window calibration as an instrument problem;
- ordering predictivity against distance and rate;
- binning / resolution sensitivity;
- real masked lag/rank/KMeans readout before any data-facing claim;
- R4a vs R4b distinction when M3B consumes M3A events.

Demote or remove:

- `permissivity_vth_delta(h(W), mu)` as a main mechanism;
- `Lambda0 x mu` where `mu` is an artificial W-derived threshold knob;
- claims that `W_event` is already a directional propagation operator;
- any statement that kick basin defines seizure onset.

## 2. W Objects

### 2.1 Structural W

`W_struct` is the fixed substrate scaffold. It should not change with slow state.

Candidate definition:

```text
W_struct[p, q] = mean E->E coupling from source bin q to target bin p
```

It can be computed from the model connectivity matrix, geometry, delay, and E->E anisotropy. It is a structure readout, not a phenotype readout.

### 2.2 Baseline response W

At baseline state, estimate response fields:

```text
W_resp[p, q] = [ E(A_p | perturb/source q) - E(A_p | sham) ]_+
```

Then derive:

```text
W_step[p, q] = W_resp[p, q] / (source_mass_q + eps)
Lambda0 = rho(W_step)
W_shape = normalized W_resp for axis/order only
```

Do not use a single row-normalized matrix for both `Lambda0` and shape. `W_resp` carries recruitment strength; `W_step` carries gain; `W_shape` carries axis/order.

### 2.3 Event-conditioned W

Because small-kick W was negative and finite events are nonlinear, M3B must estimate event-conditioned W:

```text
W_event[p, q] = E(early recruitment in p | successful returned event initiated at q)
```

Primary event window is event-aligned, not fixed-delay. Fixed windows are sensitivity only.

Current evidence status must be preserved:

- `W_small`: not supported.
- `W_event`: supports finite recruitment / susceptibility at current scale.
- Directional W claim: not established; high-res W is resolution-sensitive but not resolved.

### 2.4 State-dependent effective W

`W_eff(s)` is only defined after M3A provides a slow state `s`.

Conservative model-based form:

```text
W_eff(s) = D_post(s) * W_step * D_pre(s)
Lambda_eff(s) = rho(W_eff(s))
```

Event-conditioned empirical form:

```text
W_eff[p, q | s_bin] =
  E(delta A_p(t + delta) | A_q(t) > 0, s(t) in s_bin)
```

Core claim to test:

```text
W_shape(s) remains similar to W_shape(0)
while Lambda_eff(s) or R_event(s) increases.
```

This means effective gain changes; structural W does not.

## 3. Required Inputs

M3B can proceed in two lanes.

Lane B1 can start now:

- model connectivity and geometry;
- baseline / event-conditioned model events;
- high-resolution W artifacts;
- SEEG axis artifacts from Topic 5 / ictal recruitment docs.

Lane B2 waits for M3A:

- slow-state scalar or vector `s_slow(t)`;
- per-event state bins;
- R-class labels;
- pre/onset/peak/end slow-state samples.

Before using Topic 5 numbers or axes, read:

```text
docs/topic0_methodology_audits.md
docs/topic5_seizure_subtyping.md
docs/paper_overview.md
```

Do not rely on stale summaries.

## 4. Tasks

### Task 0: Rewrite and freeze M3B scope

- [ ] Write `STATUS.md` stating M3B excludes slow-variable mechanism claims.
- [ ] Archive the old `h(W)->threshold mu` path as historical negative/control only.
- [ ] Record current evidence: high-res W is resolution-sensitive but not resolved; static mu was rate-only negative.
- [ ] Confirm no M3B script writes "seizure-like transition" without M3A input.

### Task 1: Audit W object implementation

- [ ] Confirm or implement `W_struct`, `W_resp`, `W_step`, `W_shape` helpers.
- [ ] Keep source-column / target-row direction tests.
- [ ] Keep row-normalization anti-trap tests.
- [ ] Keep low-source-mass exclusion tests.
- [ ] Emit provenance: binning, source mask, event window, perturbation mode, seed set.

Pass condition: each W object has a separate file/output field and cannot be silently substituted for another.

### Task 2: Baseline and event-conditioned W estimation

Estimate W in three ways:

1. structural connectivity: `W_struct`;
2. perturbation response: `W_resp/W_step/W_shape`;
3. spontaneous or finite returned event early recruitment: `W_event`.

Required comparisons:

- `W_event` reproducibility across seeds;
- `W_event` vs `W_struct` axis/shape;
- `W_event` vs distance;
- `W_event` vs local rate where rate is available;
- `W_event` vs K_min / susceptibility map where available.

Do not call W a directional propagation operator unless it beats distance/rate and has stable axis evidence.

### Task 3: Resolution and binning sensitivity

Run or aggregate:

```text
n_bins_per_axis: 5, 9, 11
substrates: bare, core/susceptible substrate where available
source: center first, then non-edge off-center sources
```

Required verdict:

- anisotropy trend;
- significance vs spatial-shuffle null;
- `rho_W - rho_dist`;
- whether apparent axis is boundary-driven;
- whether higher resolution changes the claim or only weakens/strengthens a caveat.

If W improves with resolution but remains non-significant, write "resolution-sensitive but not resolved", not PASS.

> **Partly DONE (2026-06-24)**: center bare@1.6 / n17.6@1.1 already run + analyzed at n_bins 5/9/11
> (`run_m3_highres_w.py`, results/topic4_sef_hfo/m3_local_w/highres_w/). Verdict = **RESOLUTION-SENSITIVE
> but NOT RESOLVED**: anisotropy rises with resolution (bare 1.19→2.47, n17.6 1.17→1.82 — 5×5 was
> washing out structure) but stays non-significant vs the spatial-shuffle null (p≈0.6–0.8) and W does
> not robustly beat distance even at 11×11. So at the CURRENT (interictal-baseline) working point W only
> weakly captures the E→E gradient. Still TODO for Task 3: off-center non-edge sources + the
> susceptible/core substrate at high-res + (Task 2) the EVENT-CONDITIONED / spontaneous-early-recruitment
> W estimator (the kick probe injects radially, which may bias W toward distance).

### Task 4: Model-to-SEEG axis bridge

Goal:

```text
model W axis / rank order
  vs
SEEG interictal template rank axis
  vs
SEEG seizure early rank axis
```

Required steps:

- [ ] Load accepted Topic 5 / ictal axis contract from docs before coding.
- [ ] Define a common axis metric: rank correlation, signed projection, endpoint-side agreement, or angular alignment.
- [ ] Apply the same metric to model output and SEEG artifacts.
- [ ] Use masked lagPat/rank pipeline, not raw phantom-contaminated ranks.
- [ ] Report subject-level results before cohort summary.

Pass condition: W-derived model axis is quantitatively comparable to SEEG interictal/seizure rank axes and beats appropriate geometry/rate nulls.

### Task 5: W_eff(s) after M3A

Only after M3A provides `s_slow`:

- [ ] Bin events by slow-state quantile or physiologic state.
- [ ] Estimate `W_eff(s)` empirically or via `D(s) * W_step`.
- [ ] Compute `Lambda_eff(s)` or finite-event recruitment gain `R_event(s)`.
- [ ] Compute shape stability:

```text
corr(vec(W_shape(s)), vec(W_shape(0)))
axis_delta(W_shape(s), W_shape(0))
```

Expected positive pattern:

```text
s_slow increases
Lambda_eff or R_event increases
W_shape stays stable
R-class moves R2/R3 -> R4a
```

Negative patterns:

- `Lambda_eff` flat and phenotype changes: W is not the mediator/readout.
- W shape changes completely: slow state rewrites the scaffold, contradicting stable-field claim.
- only R4b appears: tonic runaway, not bridge.

### Task 6: Event accumulation along W

This is the M3B version of "multiple interictal events influence seizure":

```text
load_p(t) = [W_shape * a(t)]_p
cumulative_load_p(t + dt) = lambda * cumulative_load_p(t) + load_p(t)
```

If M3A has chloride/GABA state:

```text
e_GABA_p(t + dt) =
  e_GABA_p(t) + eta_Cl * cumulative_load_p(t)
  - dt / tau_Cl * (e_GABA_p(t) - e_GABA_0)
```

This is not a new M3A mechanism unless M3A has already validated the slow variable. In M3B it is a field-load readout: does repeated activity along the same scaffold predict later slow-state susceptibility?

Required controls:

- time-shuffled event load;
- spatially shuffled W_shape;
- distance-only load;
- rate-only load.

### Task 7: Figures and verdict

Required figures:

1. `W_struct / W_event / W_shape` maps with axis.
2. `rho_W` vs `rho_dist` / rate.
3. resolution sensitivity summary.
4. model W-axis vs SEEG interictal/seizure rank-axis bridge.
5. after M3A: `s_slow` vs `Lambda_eff/R_event` colored by R-class.
6. after M3A: `s_slow` vs W-shape stability.
7. after M3A: cumulative W-load vs slow-state change.

Every `figures/` directory must include a Chinese `README.md`.

Final M3B verdict categories:

- **B-PASS field bridge**: W axis/readout bridges model and SEEG and beats nulls.
- **B-PASS effective-gain only**: W shape stable and `Lambda_eff/R_event` tracks slow state, but axis bridge weak.
- **B-BOUNDED NEGATIVE**: W is local recruitment/susceptibility only, not a directional field operator.
- **B-FAIL**: W does not beat distance/rate and does not bridge SEEG axis.

## 5. Stop Rules

Stop and recap before any expensive grid if:

- W does not beat distance/rate at the planned resolution;
- Topic 5 axis contract is unresolved or stale;
- M3A has not produced a valid slow-state variable but the task attempts `W_eff(s)`;
- R4b is being counted as seizure-like bridge.

## 6. Plain-Language Claim Template

Allowed if supported:

> The structural field scaffold remains stable. Slow state does not create a new route; it changes effective recruitment gain along the same scaffold. Repeated interictal events load that scaffold and can be tested as a driver of later susceptibility.

Not allowed:

> W causes seizure.

Not allowed:

> W_event is a proven directional propagation operator.

Not allowed:

> h(W)-threshold mu reproduced the interictal-to-seizure transition.

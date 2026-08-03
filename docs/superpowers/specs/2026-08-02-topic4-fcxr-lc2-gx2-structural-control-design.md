# FCXR-LC2-GX2 — causal entry-gate × shared-offset-path architectures

Status: **CONDITIONAL CONTINGENCY — NOT AUTHORIZED BY GX1**

Date: 2026-08-02
Upstream: `2026-08-02-topic4-fcxr-lc2-gx1-entry-offset-diagnostics-design.md`

## 1. Scientific question

GX2 is not another parameter search. It is authorized only if GX1 shows both:

1. no adjacent natural H parameter window separates healthy-low, susceptible-low and susceptible-high;
2. theoretical maximal X shutdown cannot remove the H-supported high state.

Under that joint result, GX2 asks which missing structural edge is causal:

```text
susceptibility gate on H gain       (entry control)
shared X control of the whole H path (offset authority)
```

The archived additive H architecture is the `(gate off, shared path off)` negative control. GX2 adds the
other three cells of this architecture 2×2. It remains a frozen-geometry experiment, not a dynamic
lifecycle.

### 1.1 GX1 adjudication (2026-08-03)

GX1 completed after this contingency was drafted and returned:

```text
S1 = NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP
X1 = X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH
authorized_next_hypothesis = LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE
```

The maximal-bypass prerequisite is false: availability `0.1` and `0.0` both returned to the accepted
interictal workpoint for the final 2 s, while `0.5` and `1.0` remained high. Therefore the full
`D gate × shared X/H path` 2×2 below is **not authorized for execution**. It is retained as a
pre-registered contingency only, so a future independent result that demonstrates maximal-X bypass
cannot lead to a post-hoc architecture design.

The currently authorized route is narrower and separable: test local D-dependent H gain as an entry
geometry hypothesis, and calibrate/justify the X dynamic range independently. GX1 does not prove that a
D gate is sufficient: H1 at high theta already showed D-selective one-way ignition, but did not preserve
a susceptible low basin.

## 2. Locked scope

Allowed:

- RC1 recurrent-only conductance and smooth saturation;
- the same local H state, theta, k and tau selected upstream;
- frozen depletion `D={0,0.15}` and frozen relay availability;
- exactly three new architectures and five frozen arms per architecture;
- development noise 401 and out-of-sample confirmation noise 402;
- an off-by-default implementation in non-blessed `mz_slow_vars.py` only.

Locked out:

- dynamic Z/X, M/K/A/ELR, kick, reset, time switch or new E→E edges;
- core masks, seizure labels or fitted spatial maps in either new gate;
- tuning any structural parameter from GX2 outcome;
- 40k lifecycle, patient morphology or paper-ready lifecycle claims.

## 3. Entry gate

Let `D=1-z` and define one analytic, local, monotone susceptibility gate:

```text
D_ref = 0.15
D_H   = D_ref/2 = 0.075
k_D   = D_ref/6 = 0.025
G_D(D) = sigmoid((D-D_H)/k_D) / sigmoid((D_ref-D_H)/k_D)
```

Thus `G_D(0)≈0.0498` and `G_D(0.15)=1`. No parameter is estimated from GX2. The nonzero healthy
leakage makes healthy-high a real falsification arm rather than a hard-coded pass.

The D-gate-only architecture is:

```text
u_pre_sat_i = gA_postX_i + rho_H * G_D(D_i) * S_H(h_i)
```

It changes only entry selectivity. Existing H remains additive and may still bypass X.

## 4. Shared offset path

The shared-path architecture must make theoretical `x=0` remove both fast recurrent drive and H's
amplification of that drive. It is not a second target-side inhibitory current:

```text
u_pre_sat_i = gA_postX_i * (1 + a_H * S_H(h_i))
```

The combined architecture is:

```text
u_pre_sat_i = gA_postX_i * (1 + a_H * G_D(D_i) * S_H(h_i))
```

`gA_postX_i` is the already validated recurrent conductance after the presynaptic X scatter path. Hence
`x=0 -> gA_postX=0 -> u_pre_sat=0` by construction, without a separate H-off switch.

### 4.1 Force matching

`a_H` is not scanned. Before GX2, take the final 1 s median of the spatial-mean `gA_trace` from the
canonical archived susceptible-high/no-X fork, call it `gA_ref>0`, and set:

```text
a_H = rho_H / gA_ref
```

At `D=0.15`, `S_H≈1` and `gA=gA_ref`, the mean added pre-saturation drive matches the old additive
`rho_H` anchor. Failure of `gA_ref>0` blocks execution. The force match is aggregate; per-cell effects are
expected to differ and must be reported.

## 5. Architecture 2×2 and scientific arms

The archived current architecture is not rerun. Test these three new architectures:

| architecture | D gate | shared X/H path |
|---|---:|---:|
| D_only | on | off |
| shared_only | off | on |
| combined | on | on |

Each architecture requires five arms:

| arm | D | H(0) | x availability | required role |
|---|---:|---:|---:|---|
| healthy_low | 0 | 0 | 1 | low basin |
| healthy_high | 0 | 2 theta | 1 | entry-gate leakage challenge |
| susceptible_low | 0.15 | 0 | 1 | spontaneous-low basin |
| susceptible_high | 0.15 | 2 theta | 1 | finite high basin |
| susceptible_high_X | 0.15 | 2 theta | 0.786 | strongest archived physiological-load challenge |

Run all five at noise 401 and 402: `3 architectures × 5 arms × 2 noises = 30` branches. Shared prefixes
may save compute but do not reduce arm count. Noise 402 never selects parameters.

## 6. Gates

### 6.1 Entry-positive

An architecture has entry control only when both seeds satisfy:

- healthy_low and healthy_high return to `INTERICTAL_WORKPOINT`;
- susceptible_low remains `INTERICTAL_WORKPOINT`;
- susceptible_high remains finite-high for the locked tail window;
- all four arms are finite, zero-clip and below refractory ceiling.

This is a frozen basin/selectivity result, not spontaneous onset.

### 6.2 Offset-positive

An architecture has physiological-load offset authority only when `susceptible_high_X` returns to
`INTERICTAL_WORKPOINT` for at least `max(2000 ms,3 tau_H)` in both seeds while its matched
`susceptible_high` remains high.

If combined passes entry but not offset, report `SHARED_PATH_POSITIVE_RANGE_NEGATIVE`; do not weaken the
return window or substitute x=0. If shared-only offsets but ignites healthy arms, it is offset-positive
and entry-negative.

### 6.3 Joint gate

Only `combined` may earn:

```text
FROZEN_ENTRY_OFFSET_ARCHITECTURE_CANDIDATE
```

It requires entry-positive and offset-positive in both noises. This label still does not authorize a
lifecycle claim; it authorizes a separate dynamic Z/H/X sprint.

## 7. Causal interpretation

- D_only positive, shared_only negative: susceptibility gate is sufficient for entry; X execution still
  lacks practical authority.
- shared_only positive for offset but negative for entry: shared path repairs termination but not state
  selectivity.
- combined exceeds both single structures: interaction is required.
- combined equals the union of single structures: the two repairs are separable.
- all negative: the local H state/effect form is inadequate; do not add M/K/A to rescue it.

## 8. Resources and stop rules

- First one-row smoke remeasures RSS; at most two 40k workers.
- `MemAvailable>=96 GiB + 2×1.35×RSS_single` before worker 2; OMP/BLAS threads fixed to one.
- Swap +256 MiB stops submission; +512 MiB and rising terminates only GX2.
- All long work uses `setsid nohup`, exact PID/SID watchdog, stage lock and sentinels.
- Numerical corruption stops the affected matched architecture; scientific negatives do not stop the
  registered 30-arm matrix.
- GX2 cannot start unless a future signed adjudication explicitly supersedes GX1 and authorizes
  `CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH`.

## 9. Required outputs

```text
results/topic4_sef_hfo/fcxr_lc2_core/gx2_structural_control/
  execution_lock.json
  force_match.json
  architecture_manifest.json
  architecture_map.json
  candidate_verdict.json
  STATUS.md
  resource_log.jsonl
  figures/architecture_2x2.png
  figures/entry_offset_traces.png
  figures/README.md
```

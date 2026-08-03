# FCXR-LC2 GX1 review acceptance

Date: 2026-08-03

## 1. Acceptance

GX1 is accepted as a frozen mechanism map, not as a lifecycle result:

```text
GX1_MECHANISM_MAP_ACCEPTED
FINITE_H_HIGH_STATE_POSITIVE
D_SELECTIVE_ONSET_CANDIDATE
SAME_D_BISTABILITY_NOT_FOUND
X_OFFSET_PATH_REACHABLE
X_FIXED_D_DYNAMIC_RANGE_INSUFFICIENT
COUPLED_D_X_OFFSET_UNTESTED
DYNAMIC_LIFECYCLE_UNTESTED
SPATIAL_INSTABILITY_UNTESTED
```

The accepted scientific reading is that local H can support finite high activity and that a high
threshold places the healthy and susceptible substrates on opposite sides of an H activation boundary.
GX1 did not find a low/high dual basin at the same susceptible D, but that geometry is not required for
an autonomous seizure lifecycle.

The X probe shows that the high state returns low at frozen availability 0.1/0.0, while it remains high
at 0.5/1.0 and at the same-anchor archived loads 0.872/0.786. This proves a structurally reachable offset
path only. It does not test the coupled route in which X lowers activity, Z recovers, D decreases and the
two variables jointly remove the high state.

## 2. Corrections accepted

- X verdict now compares against same-anchor archived fork rows rather than a narrative constant.
- The susceptible-high strip arm is marked as H-gate-saturated: 12 nominal points reduce to three
  independent rho trajectories for that arm.
- The X experiment is labelled as starting from analytic `H=2 theta`, not a converged high microstate;
  the measured timing margin is preserved as a bound, not promoted to exact-state evidence.
- Result/archive numbers, resource data and archived relay loads are derived from artifacts.
- Mechanism-module hashes supplement the six-file blessed-engine set.
- Figures use the actual interictal decision line and explain the repeated susceptible-high column.
- The pre-registered GX2/D-gate route is retained only as an audit trail and is superseded for execution.

## 3. Engineering status

- 40/40 archived trajectories are finite, zero-clip and unchanged; no simulation was rerun for the
  verdict correction.
- The final X map was re-aggregated read-only from archived cells.
- Targeted GX1 tests pass; the post-review relevant regression and six blessed-engine hash checks remain
  recorded in the signed run manifest.
- The +256 MiB soft stop was not implemented in GX1's already-completed eager process pool. LC3 requires
  bounded submission before any new matrix.

## 4. Claim boundary

Allowed:

> In the bounded RC1 SNN, local H supports finite high activity and shows a D-selective high-state
> candidate. Strong experimental relay depression can remove the state, but the natural relay range is
> insufficient at fixed pathological D.

Not allowed:

- bistability, hysteresis or a proven bifurcation;
- a spontaneous seizure lifecycle;
- physiological validity of availability 0.1/0.0;
- insufficiency of X in the coupled D-X system;
- axial/local onset instability;
- patient-like ictal morphology or statistical recovery.

## 5. Superseding next program

The authorized next design is FCXR-LC3:

1. derive a D axis from archived no-kick Z trajectories;
2. map the unchanged H equation in the frozen D-aX plane from exact low/high microstates;
3. audit axial/local versus global spatial response before a dynamic long run;
4. calibrate X only against the measured offset surface;
5. run a no-kick lifecycle pilot only if the temporal and spatial gates both support it.

Explicit D gating, H retuning, M morphology, K/A/ELR and paper-ready figures remain locked out.

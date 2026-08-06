# FCXR-LC2-GX1 status

Status: **ACCEPTED — GX1 frozen mechanism map**

- Canonical verdict: `GX1_MECHANISM_MAP_ACCEPTED`
- S1: `NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP`
- X1: `X_PATH_REACHABLE_RANGE_INSUFFICIENT`
- Pre-registered local routing result: `LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE`
- Post-review authorized next program: `LC3_DX_STATE_PLANE_AND_SPATIAL_INSTABILITY_AUDIT`
- Numerical safety: 40/40 rows
- Entry component: `D_SELECTIVE_ONE_WAY_IGNITION_WITHOUT_DUAL_BASIN`
- Natural low/high dual-basin window: **no**
- Same-D bistability required for lifecycle: **no**
- X path reachable: **true**; tested return bracket:
  `[0.1, 0.5]`; archived loads at this anchor:
  `INSUFFICIENT_FOR_THIS_H_BRANCH`
- Coupled D-X offset: **untested**
- Spatial instability / eigenmode: **untested**
- Strip resolution: the `susceptible_high` arm(s)
  run with the H gate pinned open, so they resolve `rho` only
- Offset arms started from `analytic 2*theta head start, not the converged high branch`
- Dynamic lifecycle: **not tested**
- M/K/A/ELR: **not used**

GX1 identifies a D-selective monostable-onset candidate and a structurally reachable X offset path.
It does not establish a spontaneous interictal-ictal-interictal lifecycle, a coupled D-X return path,
an axial onset mode, or patient-like ictal morphology.

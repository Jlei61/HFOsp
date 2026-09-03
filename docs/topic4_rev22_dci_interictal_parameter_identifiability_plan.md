# Topic 4 rev22-DCI implementation plan

**Depends on:** `docs/topic4_rev22_dci_interictal_parameter_identifiability_spec.md`

**Execution state:** documentation only; stop before new SNN simulation for collaborator review

## Task 0: freeze the revision boundary

- Create a rev22 config with the exact rev20 `dualcore_s39`, Node, OU, detector,
  causal-family producer and patient input hashes.
- Rename machine-facing dose fields to expose their semantics while retaining explicit
  backward-compatible mappings:
  - `learned_ee_pattern_dose -> legacy g_EE`
  - `learned_e_to_i_pattern_dose -> legacy g_EtoI`
- Record that `Both/BOS` is a derived joint-dose arm.
- Add a forbidden-input guard for Z/M, patient ictal artifacts and Fig.5 metrics.
- Freeze the fixed-topology parameter domain and the separate out-of-scope list: total EE
  strength, connection length scale and rebuilt topology.

**Deliverables:** rev22 config, provenance manifest, frozen-input hashes and a parameter
semantics audit.

## Task 1: offline objective reconstruction and qualification

- Locate and hash all 232 rev20 worker artifacts: 124 screen and 108 confirmation runs.
- Verify every included artifact stores complete contact-onset arrays, returned-family masks,
  contact order and candidate/seed provenance.
- Extend the event representation with physical relative onset in milliseconds while
  retaining contact identity, missingness and normalized order.
- Implement `D_cloud`, patient-to-model `D_cover` and shaft-balanced physical `D_lag` as pure
  functions with no model labels or held-out input.
- Compute patient split-block/count-matched floors and within-candidate seed noise for each
  component.
- Run the four synthetic manipulations from spec section 6 and the rev20
  aspect-3-versus-Joint-1.25 failure control.
- Write a locked objective-qualification JSON containing each identifiability ratio and the
  resulting `lambda_lag` decision.

**Stop rule:** if minority removal or SCL censoring does not worsen coverage, or aspect 3.0
still beats Joint 1.25 by losing minority support, stop at `OBJECTIVE_MODE_COVERAGE_BLIND`.
Do not launch an optimizer or adjust search bounds to compensate.

**Tests:** label-invariance, missing-contact retention, millisecond time-stretch selectivity,
pair-state normalization, count-matched floor determinism, low-model-event behavior and
known-control ordering.

## Task 2: split topology and dynamics seeds without changing legacy output

- Add `topology_seed` and `dynamics_seed` at the highest common runner boundary.
- Route topology seed only into neuron placement/connectivity construction.
- Route dynamics seed only into OU/background/simulation RNG streams.
- Preserve `seed` as the legacy default; explicit equal seeds must use the same RNG call
  order as rev20.
- Store both seeds and all derived RNG stream identifiers in every worker JSON/NPZ.

**Parity test:** for at least one rev20 candidate and one archived seed, setting
`topology_seed=dynamics_seed=legacy_seed` must reproduce substrate hashes, family boundaries,
contact onsets, ranks and endpoint arrays byte for byte.

**Cross test:** changing only dynamics seed must leave topology/delay hashes fixed; changing
only topology seed must change the graph while preserving the frozen parameter contract.

## Task 3: structure-only admissibility audit

- Evaluate a dense no-simulation grid over `theta_FT in [22.5,67.5]` and
  `AR_FT in [1,3]` on each fit topology.
- Verify exact reference no-op, unchanged topology/delays/GABA, AMPA cache rebuilding and
  per-target incoming E-to-E conservation.
- Report edge-ratio distribution, effective source count, weighted distance/orientation
  support and targets with poor numerical support.
- Keep values inside the predeclared domain as exploratory even if weak, but mark low-support
  regions so the response surface cannot silently extrapolate through them.

This audit does not reinterpret fixed-topology angle as anatomical direction.

## Task 4: freeze the 96-point response design

- Generate one deterministic augmented maximin design using a frozen design seed:
  - one reference;
  - 47 full four-dimensional points;
  - 32 leave-one-locked points, eight per plane;
  - eight learned-dose-pair points;
  - eight geometry-pair points.
- Reject duplicates after canonical rounding and regenerate before freezing.
- Attach all four physical parameter values, model-mask membership and structure-audit
  summaries to each point.
- Freeze two topology and two dynamics fit seeds as a crossed 2x2 design shared by all
  candidates.
- Keep rev20 one-dimensional artifacts as a separate historical batch; do not rerun them.

**Deliverables:** candidate manifest, seed manifest, manifest SHA256 and a coverage figure
showing every 1D/2D projection of the design.

## Task 5: implement the response and conditional-optimum analysis

- Aggregate raw seed-level `J_fit` and all component scores with equal weight per crossed
  simulation unit. Add rescored rev20 one-dimensional points as flagged axis anchors with
  their own heteroskedastic noise; never relabel them as 2x2 crossed observations.
- Fit the preregistered heteroskedastic Matérn-5/2 GP to valid candidate-level means and
  observed seed variance.
- Fit the tree-ensemble sensitivity model without changing candidate selection rules.
- Run leave-one-candidate-out prediction checks and report RMSE, Spearman rank correlation
  and interval coverage.
- Optimize the response conditionally for the 12 model masks in the spec. Locked coordinates
  remain exactly at reference values.
- Freeze one GP proposal per family. When GP and tree proposals occupy disjoint regions,
  freeze both before viewing validation metrics.

Runaway/nonfinite candidates are excluded from continuous response fitting and modeled in a
separate feasibility surface. No arbitrary score of 10 is inserted into `J_fit`.

## Task 6: engineering canary

- Run the reference, one interior four-dimensional point, one geometry-boundary point and
  one joint-dose point on one crossed seed unit.
- Verify Z/M off, complete causal-family output, physical onset arrays, topology/dynamics
  provenance, late-runaway handling and endpoint reproducibility.
- Measure peak RSS and wall time, then set worker concurrency. Numerical threads remain 1.

**Stop only for:** parity failure, provenance drift, nonfinite output, insufficient disk or an
OOM-risk estimate that cannot retain at least 32 GiB free RAM.

## Task 7: formal 96 x 4 fit run

- Run 384 trajectories of 20 s each using common crossed fit seeds.
- Use `systemd-run --user` plus `nohup`; no foreground long run.
- Maximum concurrency is the lesser of 16 workers and the measured RSS-based safe count.
- Monitor at 600 s intervals; the monitor exits and notifies on completion, provenance drift,
  memory pressure, low disk or failed workers. Do not continuously poll.
- Zero/low event yield is a valid model result. Record its uncertainty; do not impose a
  20-event biological gate.
- Aggregate training components only. The producer must fail if it can import KMeans/OOD,
  patient held-out or patient ictal inputs during this stage.

## Task 8: response fit and conditional candidate freeze

- Fit the GP and tree sensitivity model only after all 384 worker artifacts are immutable.
- Generate the 12 conditional optima and any predeclared model-disagreement duplicates.
- Write a freeze file containing coordinates, source surface hash and optimization trace.
- Freeze qualification and confirmation seed pairs before launching either stage.
- Do not read KMeans, OOD or held-out arrays while changing a parameter value.

## Task 9: new-seed qualification

- Run every frozen family candidate on three new topology seeds crossed with two dynamics
  seeds: six units per candidate.
- Compare observed training `J_fit` with the response-surface prediction and interval.
- Do not drop a family because the first seeds point in an unfavorable direction.
- Parameter values cannot be revised after this stage. A badly predicted optimum is retained
  as evidence of surrogate error.

Estimated maximum for 12 proposals: 72 trajectories, plus predeclared GP/tree disagreement
duplicates if any.

## Task 10: confirmation

- Run all 12 family candidates on four fresh topology seeds crossed with three fresh dynamics
  seeds: 12 units per candidate and 144 trajectories total.
- Preserve common random numbers across candidates.
- Complete all candidates regardless of interim direction unless an engineering safety stop
  is triggered.
- Freeze the worker aggregate and hashes before opening selection-blind validation.

## Task 11: open validation once

- Compute recording-block held-out complete-distribution distance.
- Run natural KMeans K=2 on readable events and align clusters to frozen patient templates;
  report the full 2x2 matrix, balanced alignment and minority proportion.
- Compute OOD using all returned families as denominator, with unreadable shown separately.
- Compute mode-specific recruitment, physical lag and rank-profile diagnostics.
- Report pooled proportions, equal-network-weighted proportions and how many individual
  networks express both modes.
- Use topology-first hierarchical bootstrap for paired contrasts.
- Report raw distance, a single fixed-floor `F_closed`, candidate-specific count-matched
  floors and returned-event yield side by side.

The validation report must call KMeans/OOD **selection-blind**, not independent. Only the
recording-block held-out distance is data-held-out, and the revision remains development-only.

## Task 12: figures and scientific closeout

Generate:

1. objective-qualification controls showing minority removal, SCL censoring and time stretch;
2. model-family matrix with free/locked coordinates and paired values for training distance,
   held-out distance, KMeans alignment, OOD and yield;
3. response atlas and the two primary two-dimensional slices (`g_LEE x g_LEI`,
   `theta_FT x AR_FT`);
4. Pareto plot: held-out distance versus KMeans alignment, color=OOD, size=event yield;
5. Fig.4-style direct readout/GIF and KMeans panel for the final nondominated candidate;
6. Chinese `figures/README.md`, PNG/PDF, metadata and visual QA record.

The report leads with:

- the safest claim;
- the largest unresolved gap to the patient floor;
- which coordinates are useful, conditionally useful or non-identifiable;
- whether improvement came from distribution coverage rather than event suppression;
- the explicit boundary that no total synaptic strength, rebuilt tract geometry, Z/M or
  ictal claim was tested.

## Budget and completion definition

Expected new long-run budget, excluding canaries and any GP/tree disagreement duplicate:

| Stage | Trajectories |
|---|---:|
| 96-point fit x 4 crossed units | 384 |
| 12 candidates x 6 qualification units | 72 |
| 12 candidates x 12 confirmation units | 144 |
| Total | 600 |

At the rev20 measured rate of approximately 42 min per trajectory and 16 effective workers,
the ideal compute floor is about 26 wall-clock hours; scheduling contention and invalid
retries make a two-day reservation more realistic. This is an order of magnitude smaller
than separate multi-restart CMA-ES searches for every nested family.

rev22 is complete only when:

- the objective passes all offline controls;
- topology/dynamics seed parity and separation tests pass;
- all frozen fit, qualification and confirmation artifacts are accounted for;
- validation is opened only after candidate freeze;
- the two Fig.4-style acceptance panels and response atlas are visually inspected;
- code, config, manifests, reports and figure metadata have clean provenance;
- no Fig.5, Z/M or patient ictal optimization has been introduced.

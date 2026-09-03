# Topic 4 rev22-DCI implementation plan

**Depends on:** `docs/topic4_rev22_dci_interictal_parameter_identifiability_spec.md`

**Execution state:** v4 in execution since 2026-09-03. Goals are executed in ladder order;
each goal has a verification check that must pass before the next starts.

## Goal ladder

| Goal | Tasks | Verification |
|---|---|---|
| G0 | spec/plan v4 committed | both documents reference the vector endpoint, per-component surfaces, 4/6/12 topology seed plan, section 14 nulls |
| G1 | Task 0 + Task 1 | `objective_qualification.json` frozen with four identifiability ratios, four passing controls with exact-invariance clauses, and unit tests green |
| G2 | Task 1b | rev20 validation-atlas prototype figure inspected; display-identifiability decisions frozen |
| G3 | Task 2 | substrate-hash parity for equal seeds and a full 20 s archived-seed byte parity run; cross tests green |
| G4 | Task 3 + Task 4 | domain hash and design manifest frozen; coverage figure inspected |
| G5 | Task 5 | component GP + feasibility + minimax proposal code green on synthetic surfaces |
| G6 | Task 6 | canary provenance clean; RSS/wall time recorded; concurrency set |
| G7 | Task 7 + Task 8 | all fit artifacts immutable; proposals frozen before any validation field is loaded |
| G8 | Tasks 9-12 | qualification, confirmation, null block, validation opened once, figures and closeout |

Any stop rule in Task 1 or Task 3 ends the ladder at that goal with the named status.

## Task 0: freeze the revision boundary

- Create a rev22 config with the exact rev20 `dualcore_s39`, Node, OU, detector,
  causal-family producer and patient input hashes.
- Rename machine-facing dose fields to expose their semantics while retaining explicit
  backward-compatible mappings:
  - `learned_ee_pattern_dose -> legacy g_EE`
  - `learned_e_to_i_pattern_dose -> legacy g_EtoI`
- Record that `Both/BOS` is a derived joint-dose arm.
- Add a forbidden-input guard for Z/M, patient ictal artifacts and Fig.5 metrics.
- Record `[22.5,67.5] deg x [1,3]` as the provisional geometry audit range, not the final fit
  domain. Freeze the final fixed-topology domain only after Task 3.
- Freeze the separate out-of-scope list: total EE strength, connection length scale and
  rebuilt topology.

**Deliverables:** rev22 config, provenance manifest, frozen-input hashes and a parameter
semantics audit.

## Task 1: offline objective reconstruction and qualification

- Locate and hash all 232 rev20 worker artifacts: 124 screen and 108 confirmation runs.
- Verify every included artifact stores complete contact-onset arrays, returned-family masks,
  contact order and candidate/seed provenance.
- Extend the embedding vector with the `1 +` recruited-offset order block and the physical
  relative-onset block while retaining contact identity and missingness.
- Implement the three conditional views of spec section 5.1 and the four training
  components `D_support`, `D_order`, `D_lag`, `D_cover` as pure functions with no model
  labels or held-out input. Keep the composite `D_cloud` as a reported continuity value
  that is never a training component.
- Use the full patient training set as the `D_cover` query set. Build block-split
  count-matched floors for all four components: pseudo-model sample from a random half of
  training recording blocks against all events of the other half.
- Pool events over each candidate's topology units for the formal candidate-level value;
  keep per-unit values as sidecars; estimate uncertainty by leave-one-topology-out
  jackknife (spec section 5.3).
- Enforce pair eligibility `n_pair_min = 5` on the pooled model side; emit
  `NOT_ESTIMABLE_LOW_JOINT_SUPPORT` when any pair class has pooled eligible fraction below
  0.5.
- Build recruitment-thinned, count-matched floors for `D_order` and `D_lag` per candidate
  and count-matched floors for `D_support` and `D_cover` per pooled size.
- Compute the jackknife noise and the between-candidate identifiability ratio for each
  component on the standardized excess of all rev20 candidates (31 screen x 4 seeds,
  9 confirmation x 12 seeds).
- Report 180 ms clipping fractions by event-contact onset and by event.
- Run the four synthetic manipulations from spec section 6 as the only objective controls,
  each with its exact-invariance clause checked to floating-point tolerance.
- After the objective is frozen, describe the rev20 aspect-3-versus-Joint-1.25 ordering
  per component without using it to accept, reject or modify the objective.
- Write a locked objective-qualification JSON containing each identifiability ratio, the
  identifiable set `A`, the minimax proposal definition and the control outcomes.

**Stop rule:** stop at `OBJECTIVE_MODE_COVERAGE_BLIND` if any synthetic control 1-4 fails.
Stop at `TRAINING_OBJECTIVE_UNIDENTIFIABLE` if no component exceeds its seed-noise
identifiability threshold. Do not launch a response search or adjust search bounds
to compensate.

**Tests:** label-invariance, missing-contact retention, exact invariance of untouched views
under each control, pair-eligibility handling, block-split floor determinism, low-model-event
and low-joint-support behavior, and descriptive known-case reporting that cannot alter
objective qualification.

## Task 1b: freeze the validation-atlas grammar on rev20 artifacts

Only after the Task 1 objective JSON is immutable, run a separate descriptive producer over
all 232 rev20 trajectories. It may read historical validation inputs but must not write to
the objective-qualification or parameter-domain artifacts.

- Compute the six primary endpoints on held-out data (support, order and timing views,
  fixed-budget recall, natural KMeans balanced alignment, OOD) plus the composite distance.
- Compute event yield and classifier two-sample AUC as secondary outputs.
- Freeze prototype-only `n_cov_rev20` from the rev20 reference and calibrate
  `r_cov_rev20` using patient-training data only; use all held-out patient events as recall
  queries. These prototype values do not replace the formal rev22 freeze in Task 8.
- Apply the same between-candidate/within-seed identifiability ratio to each proposed
  continuous-response row. Freeze which rows remain in the main atlas before any rev22
  simulation; retain non-identifiable metrics in the sidecar.
- Produce the first two-layer prototype: continuous parameter slices and the historical
  family matrix. This is a layout and assay-sensitivity check, not candidate selection.

**Tests:** fixed support-budget determinism, no patient-query count matching, low-yield
recall status, grouped C2ST folds, label-permutation behavior and a guard proving the
descriptive producer cannot modify Task 1 or Task 3 freeze files.

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

- Evaluate a dense no-simulation grid over the provisional
  `theta_FT in [22.5,67.5]`, `AR_FT in [1,3]` range on each fit topology.
- Verify exact reference no-op, unchanged topology/delays/GABA, AMPA cache rebuilding and
  per-target incoming E-to-E conservation.
- Report edge-ratio distribution, effective source count, weighted distance/orientation
  support and targets with poor numerical support.
- Freeze the largest connected axis-aligned rectangle containing `(45 deg,2)` for which, on
  every fit topology: incoming-budget error is at most `1e-9`; no denominator is zero;
  edge-ratio p01/p99 remains within `[0.25,4]`; median effective-source-count ratio is at
  least 0.75; and p05 effective-source-count ratio is at least 0.50.
- Hash the final domain before generating Task 4. If only the reference survives, mark the
  geometry families structurally non-estimable and continue with dose-only families.

This audit does not reinterpret fixed-topology angle as anatomical direction.

## Task 4: freeze the branch-specific response design

- If Task 3 freezes a nontrivial geometry rectangle, generate one deterministic augmented
  96-point maximin design using a frozen design seed:
  - one reference;
  - 47 full four-dimensional points;
  - 32 leave-one-locked points, eight per plane;
  - eight learned-dose-pair points;
  - eight geometry-pair points.
- If Task 3 marks geometry structurally non-estimable, freeze the geometry coordinates at
  their reference values and generate one deterministic 32-point maximin design over
  `g_LEE x g_LEI`, including the exact reference. Only `M0000`, `M1000`, `M0100` and
  `M1100` remain branch-eligible.
- Reject duplicates after canonical rounding and regenerate before freezing.
- Attach all four physical parameter values, branch-eligible model-mask membership and
  structure-audit summaries to each point.
- Freeze four fit topology seeds, each paired with its own dynamics seed, shared by all
  candidates as common random numbers.
- Freeze the variance-decomposition block: `M0000` and one predeclared interior design
  point on the four fit topologies x three dynamics seeds (16 additional trajectories).
- Keep rev20 one-dimensional artifacts as a separate historical batch; do not rerun them.

**Deliverables:** candidate manifest, seed manifest, manifest SHA256 and a coverage figure
showing every 1D/2D projection of the design.

## Task 5: implement the response and conditional-optimum analysis

- Aggregate raw seed-level component scores with equal weight per topology unit. Add
  rescored rev20 one-dimensional points as flagged axis anchors; never relabel them as
  rev22 fit observations.
- Exclude a point from a component's continuous surface unless all four fit units have at
  least 12 complete returned families, are finite/non-runaway and, for order and lag, are
  joint-support estimable. Fit a separate feasibility classifier to every completed point.
- Fit one heteroskedastic Matérn-5/2 GP per training component using the pooled-shrinkage
  noise formula in spec section 7; never fit a surface to `J_fit` or to any maximum.
- Fit the per-component tree-ensemble sensitivity model without changing proposal rules.
- Run leave-one-candidate-out prediction checks per component and report RMSE, Spearman
  rank correlation and interval coverage.
- For every branch-eligible mask, propose the minimax point of predicted normalized
  excesses over the identifiable components with locked coordinates at reference, inside
  the domain and the region of predicted joint feasibility at least 0.80. Store the
  predicted Pareto set with the proposal.
- Freeze one GP proposal per family. When GP and tree proposals occupy disjoint regions,
  freeze both before viewing validation metrics.

Conditional optimization requires predicted joint feasibility at least 0.80.
Runaway/nonfinite/low-yield candidates receive no arbitrary `J_fit` penalty and remain visible
in the feasibility atlas.

## Task 6: engineering canary

- In the primary branch, run the reference, one interior four-dimensional point, one
  geometry-boundary point and one joint-dose point on one fit topology unit. In the
  dose-only fallback, run the reference and two interior joint-dose points; do not simulate
  a geometry point that already failed the structure-only audit.
- Verify Z/M off, complete causal-family output, physical onset arrays, topology/dynamics
  provenance, late-runaway handling and endpoint reproducibility.
- Measure peak RSS and wall time, then set worker concurrency. Numerical threads remain 1.

**Stop only for:** parity failure, provenance drift, nonfinite output, insufficient disk or an
OOM-risk estimate that cannot retain at least 32 GiB free RAM.

## Task 7: formal response-design fit run

- Run either 384 primary-branch trajectories (`96 x 4`) or 128 dose-only fallback
  trajectories (`32 x 4`) of 20 s each on the four common fit topology units, plus the
  16-trajectory variance-decomposition block.
- Use `systemd-run --user` plus `nohup`; no foreground long run.
- Maximum concurrency is the lesser of 16 workers and the measured RSS-based safe count.
- Monitor at 600 s intervals; the monitor exits and notifies on completion, provenance drift,
  memory pressure, low disk or failed workers. Do not continuously poll.
- Zero/low event yield is a valid model result, but a trajectory with fewer than 12 complete
  returned families enters the feasibility surface rather than the continuous response
  surface. This is an estimability rule, not a 20-event biological gate.
- Aggregate training components only. The producer must fail if it can import KMeans/OOD,
  patient held-out or patient ictal inputs during this stage.

## Task 8: response fit and conditional candidate freeze

- Fit the per-component GPs, feasibility classifier and tree sensitivity models only after
  all expected branch-specific worker artifacts are immutable: 384 in the primary branch or
  128 in the fallback branch, plus the decomposition block.
- Report the topology and dynamics variance fractions from the decomposition block before
  any interval is interpreted.
- Generate all branch-eligible conditional optima and any predeclared model-disagreement
  duplicates.
- Retain any family whose optimum equals the reference as a `REFERENCE_RETURN` row. Verify an
  exact same-seed duplicate is byte-identical, and freeze a topology-matched independent
  dynamics replicate for the stochastic reference floor.
- Freeze `n_cov` from the four `M0000` fit units and calibrate `r_cov` from 1,000 frozen
  patient-training support draws using the exact formula in spec section 9. Hash both before
  any held-out metric is opened. If `n_cov < 12`, emit
  `REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE` and prohibit a six-endpoint Pareto support claim.
- Write a freeze file containing coordinates, source surface hash and optimization trace.
- Freeze qualification and confirmation seed pairs before launching either stage.
- Do not read KMeans, OOD or held-out arrays while changing a parameter value.

## Task 9: new-seed qualification

- Run every frozen family candidate on six new topology seeds, one dynamics seed each.
- Compare each observed training component with its surface prediction and interval.
- Do not drop a family because the first seeds point in an unfavorable direction.
- Parameter values cannot be revised after this stage. A badly predicted optimum is retained
  as evidence of surrogate error.

Estimated maximum is 72 trajectories for 12 primary-branch proposals or 24 for four
fallback-branch proposals, plus predeclared GP/tree disagreement duplicates if any.

## Task 10: confirmation

- Run all branch-eligible family candidates on twelve fresh topology seeds, one dynamics
  seed each: 12 units per candidate, totaling 144 trajectories in the primary branch or 48
  in the fallback branch, structurally identical to the rev20 confirmation.
- Preserve common random numbers across candidates.
- Complete all candidates regardless of interim direction unless an engineering safety stop
  is triggered.
- Freeze the worker aggregate and hashes before opening selection-blind validation.

## Task 10b: matched structural nulls and Node blocking factor

Launched after the Task 8 freeze, on the first six confirmation topology seeds, one
dynamics seed each, for `M0000` and the frozen full-model proposal:

- re-registration nulls `r180` and `r90` through the existing `field_transform` path;
- two frozen matched-norm random learned-pattern rows at the reference doses;
- two placement nulls with the same 1,499-neuron budget: merged midpoint core, and frozen
  random two-core centers;
- one isotropic-graph arm (`rho_EE = 0`, unchanged in-degree) built through a separate
  network cache; it is reported unpaired.
- two previously frozen Node candidates named in the Task 0 config, loaded through the
  existing override path, as the Node blocking factor.

Budget: 4 nulls x 2 arms x 2 conditions x 6 = 96 paired trajectories, 12 isotropic
trajectories, 24 Node-factor trajectories. Nulls are opened with validation and never
select a parameter.

## Task 11: open validation once

- Score every immutable response-design trajectory as a descriptive second pass, in addition
  to scoring qualification and confirmation trajectories. Never refit `J_fit`, revise an
  optimum or change family membership after this pass opens.
- Fit metric-specific descriptive response surfaces solely to draw the frozen conditional
  slices. Do not optimize, rank candidates or update the training surrogate with these
  surfaces.
- Compute the held-out support, order and timing views and the composite distance.
- Run natural KMeans K=2 on readable events and align clusters to frozen patient templates;
  report the full 2x2 matrix, balanced alignment and minority proportion.
- Compute OOD using all returned families as denominator, with unreadable counted as OOD and
  shown separately; also report `1-OOD` as precision-like support agreement.
- Compute fixed-budget held-out recall with frozen `n_cov` and `r_cov`; keep all-event recall
  as a sidecar and mark candidates below `n_cov` as `NOT_ESTIMABLE_LOW_YIELD`.
- Report timing in raw milliseconds and floor-normalized units, plus mode-specific
  recruitment and rank-profile diagnostics.
- Run the K=2 versus K=1 held-out likelihood, label-shuffle alignment null and
  patient-matched KMeans benchmark on every confirmed candidate.
- Score the null block and Node blocking factor with the same paired network bootstrap;
  write `NULL_NOT_SEPARATED` or the separated status per null and the Node-factor
  direction-preservation table.
- Run the secondary class-balanced, group-separated classifier two-sample test and its label
  permutation reference.
- Report both 180 ms clipping sidecars: event-contact fraction and any-clipped-event fraction.
- Report pooled proportions, equal-network-weighted proportions and how many individual
  networks express both modes.
- Use the paired network bootstrap over topology seeds for contrasts; topology-first
  hierarchical resampling only inside the decomposition block.
- Apply the six-endpoint paired Pareto rule to `M1111` versus each leave-one-locked family
  in the primary branch. In the fallback, compare `M1100` with `M1000` and `M0100`, while
  retaining `M0000` as the paired reference. Write `PARETO_SUPPORTED`, `TRADEOFF` or
  `NON_IDENTIFIABLE_AT_CURRENT_SEEDS` directly from point estimates and paired 90% intervals.
- Report raw distance, a single fixed-floor `F_closed`, candidate-specific count-matched
  floors and returned-event yield side by side.

The validation report must call KMeans/OOD **selection-blind**, not independent. Held-out
views, fixed-budget recall, composite distance and C2ST use the held-out recording block; KMeans templates, OOD support and `r_cov` originate from patient training
data. The revision remains development-only.

## Task 12: figures and scientific closeout

Generate:

1. objective-qualification controls showing minority removal, SCL censoring and time stretch;
2. conditional continuous-response atlas with parameters as columns and, subject to the
   pre-frozen display-identifiability rule, six rows: held-out support view, held-out order
   view, held-out physical timing, fixed-budget recall, KMeans alignment and OOD. Overlay pale design
   points, open rev20 anchors and valid patient self-comparison bands; encode yield by size;
3. nested-family matrix with free/locked parameter cells, paired six-endpoint differences
   and intervals, plus secondary composite-distance, yield and classifier-AUC columns;
4. branch-specific training-response atlas: `g_LEE x g_LEI` in both branches and
   `theta_FT x AR_FT` only in the primary branch;
5. Pareto plot: held-out order view versus KMeans alignment, color=OOD, size=event yield;
5b. matched structural-null figure: six endpoints for intact versus each null, paired by
   topology, for the reference and the full-model proposal;
6. Fig.4-style direct readout/GIF and KMeans panel for the final nondominated candidate;
7. Chinese `figures/README.md`, PNG/PDF, metadata and visual QA record.

The report leads with:

- the safest claim;
- the largest unresolved gap to the patient floor;
- which coordinates are useful, conditionally useful or non-identifiable;
- whether improvement came from distribution coverage rather than event suppression;
- the explicit boundary that no total synaptic strength, rebuilt tract geometry, Z/M or
  ictal claim was tested.

## Budget and completion definition

Expected new long-run budget, excluding canaries, GP/tree disagreement duplicates and any
explicit `REFERENCE_RETURN` stochastic-floor replicate:

| Stage | Primary 4D branch | Dose-only fallback |
|---|---:|---:|
| response-design fit x 4 topology units | 384 | 128 |
| variance-decomposition block | 16 | 16 |
| candidates x 6 qualification topologies | 72 | 24 |
| candidates x 12 confirmation topologies | 144 | 48 |
| null block + isotropic arm + Node factor | 132 | 132 |
| Total | 748 | 348 |

Any excluded add-on is listed as a separate manifest row before launch; it cannot be hidden
inside retries or the nominal branch budget.

At the rev20 measured rate of approximately 42 min per trajectory and 16 effective workers,
the ideal compute floor is about 33 wall-clock hours for the primary branch and 15 hours for
the dose-only fallback. Scheduling contention and invalid retries make roughly two and a half
days or one day, respectively, more realistic. Both remain substantially smaller than separate
multi-restart CMA-ES searches for every nested family.

rev22 is complete only when:

- the objective passes all offline controls;
- topology/dynamics seed parity and separation tests pass;
- all frozen fit, qualification and confirmation artifacts are accounted for;
- validation is opened only after candidate freeze;
- the two Fig.4-style acceptance panels and response atlas are visually inspected;
- code, config, manifests, reports and figure metadata have clean provenance;
- no Fig.5, Z/M or patient ictal optimization has been introduced.

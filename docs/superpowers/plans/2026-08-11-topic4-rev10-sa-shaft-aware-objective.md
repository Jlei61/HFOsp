# Topic 4 rev10-SA execution plan

## Execution discipline

- Work on `codex/topic4-rev10-sa-shaft-aware`.
- Do not touch unrelated Topic 5 or paper-figure worktree changes.
- Patient held-out is forbidden; all current results are development-only.
- Zero-simulation contracts and controls precede SNN canaries.
- Missing per-event data are reported as unavailable, never reconstructed from
  aggregate curves.
- New figure directories receive a Chinese `README.md` after rendering.
- Long jobs use managed nohup execution, bounded workers, 180-second waits, and
  completion notification.

## Previous-round acceptance

- [x] rev9-L L3b completed `1026/1026` workers with no runaway or pre-trigger mismatch.
- [x] count-matched `n=2/n=3` floors, default-path bit parity, contact-order checks,
  and derived capacity-audit facts are reproducible.
- [x] optimizer was not tested because no known-good shared solution exists.
- [x] scientific claim downgraded to the frozen ICL-biased field, shaft-blind
  objective, and finite static-edge library.
- [x] `beta`, topology expansion, and optimizer comparison remain closed.

## SA0: contact and shaft contract

- [x] Freeze name, index, shaft, numeric contact id, within-shaft order, sheet xy,
  shared-axis coordinate, and readout parameters for all 15 contacts.
- [x] Freeze unordered pair classes `55/6/44` and canonical hashes.
- [x] Verify the canonical patient/model contact order and explicitly distinguish
  event readability from multishaft coverage.

## SA1: patient target

- [x] Reconstruct training-only `m`, normalized contact onset `u`, shaft fractions,
  first-shaft offset, and validity without reading held-out scores.
- [x] Preserve old A/B labels as primary.
- [x] Fit patient-only standardization/PCA and K=2 consensus KMeans; report AMI,
  block stability, proportions, and K=3 exploratory stability.
- [x] Stop before model optimization when K=2 AMI to old labels was below 0.8.
- [x] Resolve the low-AMI stop with a separate training-only factorization audit:
  old A/B remain the direction factor, while shaft-aware K=2 is an event-extent
  factor and is rejected as a replacement patient-mode label.

## SA2: objective

- [x] Implement equal-shaft recruitment distance.
- [x] Implement unordered-pair four-state precedence and II/SS/IS class distances.
- [x] Implement separate ICL/SCL/cross profile terms.
- [x] Implement patient-only shaft-aware event-cloud embedding and distance.
- [x] Build matched-count training floors and nested smooth-worst objective.
- [x] Record full-timing versus ordinal-compatible evaluation semantics.

## SA3: zero-simulation controls

- [x] Patient SCL censoring.
- [x] Cross-shaft timing shift with unchanged masks and within-shaft order.
- [x] All-combination progressive 0/4 to 4/4 SCL restoration.
- [x] Shared-axis collapse with retained contact identity.
- [x] Produce JSON/NPZ, diagnostic figures, and figure README.

## SA4: historical artifact audit

- [x] Inventory rev8.1 fit/final, Node, Node+Edge, L2/L3 Sobol, hand dual-core,
  and Stage 2 filament artifacts.
- [x] Assign each artifact to full-timing, ordinal-compatible, or not-rescorable.
- [x] Recompute all supported shaft-aware recruitment/precedence/coverage endpoints.
- [x] Determine whether any retained historical candidate supports an old-objective
  selection miss; unavailable metrics remain explicit.
- [x] Rebuild target, factorization, and SA4 from commit `c6bde4b4`; all three
  runtime provenance records are clean and reproduce the exploratory values.

## Conditional next phase

- [x] Design and implement SA5 contact detectability after freezing SA0-SA4.
- [x] Run six paired SA5 network workers from clean commit `226338e9` and adjudicate
  observation versus local-network limitations.
- [x] Freeze the conditional SA6 matched SCL relocation and fixed-budget field
  canary design; SA5 cleared launch with current ratio `0.961` and neural ratio
  `0.953`.
- [x] Implement the deterministic 21-candidate SA6 manifest, paired worker,
  bounded launcher, three-event shaft-aware aggregation, and diagnostic figure.
- [x] Launch SA6 from clean commit `b066026f` with 12 workers and 120-second waits; then
  adjudicate fixed-budget dual-shaft capacity before any optimizer run.
- [x] Aggregate all 63 workers: no candidate met the exploratory capacity
  reference; strongest SCL field still had zero ICL-to-SCL recruitment.
- [x] Do not launch formal field optimization, Edge recalibration, `beta`, or
  optimizer comparison in this phase.

## SA6 decision

- [x] Freeze status as
  `DUAL_SHAFT_FIELD_CAPACITY_NOT_FOUND_IN_TESTED_GRID_CANARY`, not a general
  K=3 family failure.
- [x] Keep SA7 field refit closed because there is no known-good dual-shaft
  initialization.
- [ ] In a separate next phase, test packet-amplitude and SCL mass/total-budget
  curves before opening directional route support.

## SA6F representation correction

- [x] Reclassify SA6 as a constrained fixed-K3 component-3 relocation canary,
  not a continuous-field capacity test.
- [x] Remove `K`, component identity, and peak count from the primary field.
- [x] Implement a continuous tensor-product cubic B-spline surface followed by
  the existing exact field-mass projection.
- [x] Use `4x4` controls as the matched-DoF primary (15 effective versus 17 for
  K=3) and `6x6` only as a resolution sensitivity; controls are not cores.
- [x] Build patient-training recruitment initializations with shaft-balanced
  fitting weights and no shaft-assigned basis functions.
- [x] Launch 37 coefficient-unique fields on the same three paired SA6
  networks using the common
  detector and forced-source/readout contract.
- [x] Adjudicate the run as a low-resolution initialization negative, not a
  continuous-family negative: all 111 workers completed, but maximum mean SCL
  `h` was only 0.128 and no candidate produced ICL-to-SCL recruitment.
- [x] Note the mode-A ICL precedence lead (excess approximately 0.393 versus
  approximately 2.99 for the frozen K3 benchmark) without calling it a patient
  solution.

## SA6G no-K continuous support capacity control

- [x] Define a smooth distance-to-path field with no component or peak count.
- [x] Use observed dual-shaft polylines with and without the shortest direct
  cross-shaft bridge; path segments are geometry controls, not cores.
- [x] Freeze widths 0.10/0.20/0.35/0.50 mm at the same exact field-mass budget.
- [x] Verify dense-sheet projected path strength before simulation: mean
  `h=0.535-1.000` within 0.25 mm of the path.
- [x] Launch eight fields x three paired networks through systemd-run -> nohup,
  using at most 12 workers and 120-second waits.
- [x] Aggregate actual projected path/bridge `h`, directional forced response,
  spontaneous events, return, and safety.
- [x] Freeze the result as no cross-shaft support at fixed packet and budget:
  bridge mean `h=0.528-0.907`, all 24 workers clean, zero ICL-to-SCL contacts,
  zero runaway.
- [ ] Next phase: run a small paired packet-amplitude curve on connected fields;
  only then run a total-budget curve if needed.
- [ ] Open directional route support only after packet/budget ambiguity is
  removed; keep Edge and `beta` closed.

## SA6H observation-invariant field correction

- [x] Reclassify SA6F as observation-conditioned because patient contact
  targets directly set its coefficients.
- [x] Reclassify SA6G as an observation-conditioned reachability control because
  observed shaft paths directly define its support.
- [x] Freeze a whole-sheet real Fourier representation with free sine/cosine
  phase, isotropic frequency support, no component count, and no peak-count
  constraint.
- [x] Add a regression test that the field builder accepts no contact, shaft,
  onset, or patient-label arguments.
- [x] Project the selected Stage 3 field over a uniform sheet as a warm start;
  keep the exact K3 field only as a historical benchmark.
- [x] Freeze V0/V1/V2 candidates: exact/projected/uniform controls, three
  antithetic low-frequency pairs, and six antithetic multiscale pairs.
- [x] Implement one-candidate/one-network 8-s spontaneous workers using the
  common detector and no patient-derived kick source.
- [x] Implement shaft-aware aggregation, OOD accounting, KMeans stability/AMI,
  field landscapes, direct readout, and patient/model prototype figures.
- [x] Commit and launch `21 x 3` workers through `systemd-run -> nohup`, at most
  10 concurrent workers, with 180-s waits.
- [x] Complete all 63 workers with zero failures/runaway and generate the field
  landscape plus direct-readout/KMeans figure.
- [x] Adjudicate the initial library: the spectral projection preserves old
  dynamics, KMeans is stable, but no eligible candidate changes SCL recruitment
  and the apparent V1 winner does not improve the raw weakest-mode score.
- [x] Build V3 as 16 identical allocation directions on a 4x4 uniform sheet grid,
  plus warm attenuation and initial-reference controls; these are Fourier
  optimizer probes, not cores.
- [x] Launch the `21 x 3` V3 refinement on the paired development networks;
  do not tune Edge or `beta`.
- [x] Complete `21 x 3` V3 workers from `c933986b` with zero failure/runaway.
- [x] Re-audit V3 without simulation: retain all detected events, separate old
  A/B direction from shaft participation, and show that SCL-rich fields mostly
  switch shafts rather than produce patient-like joint events.
- [x] Identify and close the old shared-axis entry-filter bug and the ill-
  conditioned half-period Fourier optimization coordinates.
- [x] Implement V4 stable `14 x 14` uniform B-spline fields, block-validated
  supervised A/B assignment, all-event joint-shaft scoring, and Pareto outputs.
- [x] Launch the 50-candidate V4 screen on common seed 1031 with 16 bounded
  workers; all 50 completed with zero failure/runaway.
- [x] Freeze V4 as `REV10SA_V4_NO_JOINT_SHAFT_CANDIDATE`: six candidates had
  SCL-only activity but no candidate had a joint ICL+SCL event, so the scalar
  minimum is a diagnostic and not a winner.
- [x] Identify the V4 search-radius miss: V4 did not include the V3
  `0.5 x warm + amplitude 4, width 2.5 mm` capacity probes.
- [x] Add fail-closed aggregation: `selected_candidate_id=null` whenever the
  library has no joint event.
- [x] Freeze V4.1 as a complete 21-field V3-to-`18 x 18` spline bridge with no
  score-based source selection; preflight maximum `h` RMSE is `0.00316`.
- [x] Run all 21 V4.1 bridge candidates on paired seed 1031; all workers were
  clean and stable, and uniform 09/12 retained joint-event capacity.
- [x] Freeze V5 with four score-selected but spatially unconditioned anchors and
  all pairwise latent-linear/density-mixture interpolation paths, for 40 unique
  continuous fields.
- [x] Run the 40-candidate V5 fit screen on seed 1031 with bounded workers;
  all workers completed cleanly, and only density-mixture interpolation
  increased joint-shaft support (`3/6` for the sparse fit winner).
- [x] Freeze a diverse eight-field V5.1 subset, not only the scalar minimum,
  for seeds 1032/1033 without contact-conditioned basis functions. Require at
  least one joint-shaft event in both networks before selection.
- [x] Run the 16 paired V5.1 workers and freeze a fail-closed cross-network
  verdict without adapting candidates to the selection outcomes. Two fields
  had joint events in both networks, but joint fraction and A/B occupancy
  remained far from the patient target and varied by network.
- [x] Freeze V5.2 with the V5.1 score winner, the highest-joint eligible anchor,
  and the Stage 3 reference before reading fresh seeds 1041-1043.
- [x] Run nine V5.2 workers; all completed cleanly, but eventwise audit showed
  that pooled joint support was entirely mode B while pseudo-mode A was OOD and
  SCL-only. Downgrade the automatic pooled PASS.
- [x] Run the zero-simulation mode-conditioned joint-support audit over V4.1,
  V5, V5.1, and V5.2 artifacts; all four returned no eligible field.
- [x] Freeze V6 as 11 complete-field density mixtures along the continuous
  uniform12-to-uniform06 path over `t=[0,0.25]`.
- [x] Run the V6 fit workers and require joint in-distribution A and B support
  separately before any candidate enters network-seed selection.
- [x] Run V6 fit: all 11 workers completed, but A support appeared only after B
  support disappeared; no candidate was selected.
- [x] Freeze all 11 V6 fields unchanged for V6.1 on seeds 1032/1033.
- [x] Run V6.1: three adjacent fields produced one supported joint A event in
  seed 1033 while retaining supported joint B; seed 1032 had no supported A.
- [x] Freeze all three adjacent fields for V6.2 on seeds 1041-1043.
- [x] Run V6.2: all nine workers completed without failure/runaway; all three
  fields retained shared patient-supported joint B events, but only `t=0.05`
  produced one joint+ID A event and only in one of three fresh networks.
- [x] Close the coefficient-only Node-field path as
  `REV10SA_V62_FRESH_NETWORK_MODE_COEXISTENCE_NOT_CONFIRMED`.
- [x] Freeze the interpretation: the `18 x 18` spline coefficients form one
  continuous field and are not cores; do not add K, components, or peak-count
  constraints.
- [x] Keep optimizer comparison and `beta` closed because no known-good shared
  mode-A solution exists and the residual is directional route support rather
  than radial width.
- [x] Hand the next exploratory mechanism to a separate contact-density-
  invariant graph edge-flow spec; do not reuse the failed Gaussian
  component-pair residual or claim that the inherited rank scaffold is
  observation-free.

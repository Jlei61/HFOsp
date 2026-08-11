# Topic 4 rev10-SA execution plan

## Execution discipline

- Work on `codex/topic4-rev10-sa-shaft-aware`.
- Do not touch unrelated Topic 5 or paper-figure worktree changes.
- Patient held-out is forbidden; all current results are development-only.
- Zero-simulation contracts and controls precede SNN canaries.
- Missing per-event data are reported as unavailable, never reconstructed from
  aggregate curves.
- New figure directories receive a Chinese `README.md` after rendering.
- Long jobs use managed nohup execution, bounded workers, 120-second waits, and
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
- [ ] Launch SA6 from a clean commit with 12 workers and 120-second waits; then
  adjudicate fixed-budget dual-shaft capacity before any optimizer run.
- [ ] Do not launch formal field optimization, Edge recalibration, `beta`, or
  optimizer comparison in this phase.

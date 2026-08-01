# Fast inhibitory phase map — stage acceptance

**Date:** 2026-08-02  
**Accepted status:** `ACCEPT_EXPLORATORY_FAST_INHIBITORY_PHASE_MAP`  
**Dynamic status:** `INCOMPLETE_DYNAMIC_LIFECYCLE_EXPLORATION`

## Accepted scientific result

On the unchanged anisotropic E→E scaffold, local fast inhibitory state changes
the macroscopic branch identity.  The 21-cell frozen-state race separates a
near-synchronous relaxation-burst region from a localized tonic-patch/spreading
plateau region, with burst-to-patch transition phenotypes between them.

This supports fast inhibitory state as a branch-control coordinate.  It does
not establish a data-consistent ictal carrier, native termination, recovery or
control.

## Corrections locked at acceptance

1. The historical scalar `J` is retained unchanged but is not used for further
   candidate selection.  It can reward a continuous, very weak signal.
2. `mechanism_race_ranking_v2.json` is a separate multi-objective reanalysis.
   Frozen artifacts lack a mechanism-matched event-free baseline, so baseline
   gain fields are explicitly null rather than borrowed from another arm.
3. A small `eta_m*m` relative to a single-cell voltage gap is only a scale
   diagnostic.  It cannot rule out an M-driven collective transition.  M must
   be tested by matched-noise gain interventions.
4. Full lifecycle development is no longer serialised behind a frozen-carrier
   gate.  Fast phenotype, M-mediated exit and finite control are explored on
   the same fully dynamic trajectories.

## Engineering closure before the next batch

- I→E depression is applied at inhibitory-spike emission, before the weighted
  event is enqueued in the delay ring; a deterministic two-spike delayed test
  now locks this semantic.
- `i2e_resource` and I-threshold adaptation are classified as conditionally
  current-affecting when their feature gates are enabled.
- lifecycle workers now write a running heartbeat and a durable terminal
  receipt.
- new runs save excitatory and inhibitory virtual-SEEG current contributions
  alongside the legacy total proxy.

## Claim boundary

The only full-dynamic 30-s artifact presently demonstrates spontaneous entry
and numerical containment, but persists to the end and spreads from core to
surround.  Two of the three previously scheduled dynamic workers have no
durable artifact or terminal cause and remain historical missing runs.  They
are not counted as scientific results.


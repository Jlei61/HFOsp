# Topic 4 Z/M fast-inhibition × M × control lifecycle sprint

**Date:** 2026-08-02  
**Status:** active seed-1 development sprint  
**Supersedes:** serial carrier-before-exit execution in the 2026-08-01 D1 spec

## Goal

Without changing the pathological E→E graph, weights, anisotropy, orientation,
STD/plasticity, Z equation, S_G implementation, noise law or virtual-electrode
geometry, search for a fully dynamic development trajectory

\[
\mathcal I\rightarrow\mathcal S
\xrightarrow{u(t)\ \mathrm{or\ native\ offset}}
\mathcal P\rightarrow\mathcal I.
\]

The first priority is spontaneous entry, finite controlled exit and a returning
interictal event.  Native offset, multiple seeds and publication validation are
not required in this sprint.

## Fixed coordinates

- original seed-1 current-based Z/M SNN and pre-entry checkpoint;
- weakest effective E-only dynamic threshold setting:
  `tau_phi=60 ms`, `fraction=0.15`;
- all fast and slow states active throughout each trajectory;
- paired future noise for parameter comparisons.

## Development coordinates

- I→E depression: `tau_D in [300,850] ms`, `d_star in [0.55,0.85]`;
- I-threshold adaptation: `tau_I in [60,350] ms`, `f_I in [0,0.12]`;
- M intervention: `g_M in {0,1,3,10,30}` and
  `tau_M in {500,2000} ms`;
- Z speed remains one unless a credible phenotype becomes unreachable, when
  `g_Z in {0.8,1,1.25}` is allowed;
- finite control is an E-threshold uplift after sustained high activity, with
  no state reset or parameter switch.

## Execution order

1. Audit delay/resource semantics, conditional state inventory, worker receipt
   and readout decomposition; preserve historical J and write v2 separately.
2. Run 36 paired-noise, 12-s fully dynamic trajectories: 16 depression-only
   Latin-hypercube points, 16 combined points and four old anchors.
3. Update the phenotype map every eight completed runs and refine the observed
   burst-patch boundary; do not require a scalar carrier gate.
4. Promote four diverse phenotypes to the M panel and estimate episode duration,
   post-offset state and return as functions of `g_M,tau_M`.
5. Apply finite threshold-uplift control to promising persistent episodes.
6. Long-run the best native and controlled conditions for 45–60 s.

## Candidate description

Every run retains five separate dimensions rather than one total score:

- intensity: event-free-baseline energy gain, integrated energy, +6 dB occupancy;
- continuity: deep gaps and longest return-to-baseline gap;
- temporal structure: modulation, spectral entropy and tonic penalty;
- within-episode spatial dynamics: post-entry centroid motion, effective rank,
  PC1 common mode and repeated relay;
- lifecycle progress: entry, native/control offset, recovery and returning event.

Frozen legacy artifacts with no matched event-free baseline must report those
fields as unavailable.  They cannot borrow another mechanism's baseline.

## Resource contract

- up to eight OMP=1 workers;
- do not launch new workers below 90 GB MemAvailable;
- every worker writes start/config/checkpoint, heartbeat, peak RSS and terminal
  status;
- no paper-ready figure, multi-seed confirmation or per-phenotype archive until
  a complete development lifecycle or final sprint adjudication exists.


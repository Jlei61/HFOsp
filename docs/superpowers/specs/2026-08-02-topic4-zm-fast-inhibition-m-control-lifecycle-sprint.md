# Topic 4 Z/M fast-inhibition × M × control lifecycle sprint

**Date:** 2026-08-02  
**Status:** completed bounded-negative seed-1 development sprint
**Supersedes:** serial carrier-before-exit execution in the 2026-08-01 D1 spec

**Final adjudication:** `NO_GO_FULL_ICTAL_LIFECYCLE`; the requested fallback
deliverables (fast phase map, 36-cell M surface, finite-control dose response
and unified auditable ledger) are complete.  See
`docs/archive/topic4/sef_hfo/zm_fast_inhibition_m_control_lifecycle_sprint_2026-08-02.md`.

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

### Finite-control calibration lock

This correction was locked before the first controlled trajectory completed.
The control time is the first continuous 50 ms core-active window after
measured onset plus 1500 ms; it is not placed in a burst trough.  Calibration
uses threshold uplifts `0.25, 0.5, 1, 2, 4 mV` for 50 ms.

The calibration response is the core spike-count reduction during the pulse
window relative to the **paired no-control continuation at the identical
time**, checkpoint and future-noise stream.  The controlled and no-control
traces must be exactly identical before the pulse.  Comparing only against a
pre-pulse mean is forbidden because a burst may naturally terminate after the
pulse time.  The smallest uplift producing a 50--70% paired reduction without
an all-sheet zero-rate dwell longer than 100 ms is `u_ref`; otherwise the
nearest non-silencing dose is retained with an explicit outside-target label.

If every non-silencing calibration arm has zero or negative
paired reduction, calibration is `uncalibrated_no_paired_drop` and no weaker
dose panel is manufactured from that null response.

The dose panel is `0.5, 1, 1.5 × u_ref` crossed with `50, 200 ms`.  A controlled
offset counts only when it occurs after the pulse and either the paired
no-control episode is right-censored or the controlled duration is at least
1000 ms shorter.  Returning-event recovery is adjudicated separately and is
never implied by an activity drop alone.

### Control-clock correction

This correction was locked after auditing the first five calibration artifacts
and before any corrected control trajectory was launched.  Manifest `t0_ms` is
time **after the pre-entry checkpoint**, but the resumed SNN engine advances on
the checkpoint's absolute clock.  The runner must therefore apply the pulse at
`source_t_ms + t0_ms`, store both the relative and absolute window, and require
their relation during analysis.  The first five unversioned artifacts compared
absolute engine time directly with relative `t0_ms`; their pulses never fired.
They are invalid engineering artifacts, not a zero-effect control result.  All
corrected artifacts use clock version
`relative_to_pre_entry_checkpoint_v2` and a separate `__clkrel2` stem.

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

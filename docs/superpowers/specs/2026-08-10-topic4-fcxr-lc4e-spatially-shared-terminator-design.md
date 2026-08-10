# FCXR-LC4e — spatially shared execution of the locked cooperative terminator

Date: 2026-08-10

Status: **DESIGN LOCK — one causal architecture comparison authorised**

## 1. Core scientific question

LC4d retained the accepted no-kick D/Z entry but failed to terminate within five seconds.  Its local
actuator suppressed the original cores while high load and the local H carrier persisted off-axis.  The
next question is therefore architectural rather than another strength search:

> With the same cell-autonomous load, exact dead zone, Hill activation, time constants and population-
> mean executed dose as LC4d, does spatially sharing the actuator across all E cells remove the off-axis
> escape and produce a 1--5 s autonomous offset without cancelling D/Z entry?

This sprint tests one new spatial execution mode.  It does not add a seizure classifier, new sensor,
new edge, kick, reset, state fork or parameter step.

## 2. Locked equations

The per-cell load and gate remain exactly those of LC4d:

```text
dm_i/dt = -m_i/tau_m + sum_k delta(t-t_i^k)
u_i = max(m_i-m_dead, 0)
a_inf,i = (u_i/K)^n / (1+(u_i/K)^n)
da_i/dt = (a_inf,i-a_i)/tau_on   if a_inf,i>a_i
          (a_inf,i-a_i)/tau_off  otherwise.
```

The archived local executor is:

```text
I_M,i^local = g_max a_i.
```

LC4e changes only the spatial readout of the already computed gate:

```text
I_M,i^shared = g_max mean_E(a),  for every E cell i.
```

Hence, at every identical state:

```text
mean_E(I_M^shared) = mean_E(I_M^local) = g_max mean_E(a).
```

The comparison changes spatial allocation, not population-mean dose.  It is a rank-one spatial
executor driven by a population average of a local biophysical load, not by a classifier label.

## 3. Frozen candidate

Copy every LC4d candidate value unchanged:

- `deadzone=46.83235549926758`;
- `K=19.869522094726562`, `n=4`;
- `tau_adp=1000 ms`, `tau_a_on=100 ms`, `tau_a_off=10000 ms`;
- `g_m_max=734.1686843528613`;
- `theta_h_lc2=1.7317735254764568`;
- all RC1, Z, H, X, E/I, geometry, connection seed 1 and noise 401 settings.

Add only:

```text
m_hill_spatial_mode = shared
```

No neighbouring gain, mixture fraction, spatial kernel or regional mask is authorised.

## 4. E0 engineering and causal invariants

Before 40k execution, tests must prove:

1. omitted mode and `local` are byte-identical to the executed LC4d path;
2. `shared` emits a uniform E-cell current and leaves I cells untouched;
3. local and shared currents have exactly equal E-population means for arbitrary finite `a_i`;
4. exact-dead-zone zero gives exactly zero current in both modes;
5. invalid modes fail loudly;
6. the setting reaches the conductance membrane path and is snapshot-safe;
7. all older configs remain valid and the six blessed engine files are unchanged.

The L0 lock must hash the LC4d candidate, result, trace and source files.  It must verify from the
archived LC4d trace that onset was 11 s and executed current first became nonzero at 11.83 s.  Thus the
new and old arms must be identical through the pre-current prefix; any earlier divergence is an
implementation failure.

## 5. E1 18 s matched spatial-execution screen

Run one fresh-from-rest shared-executor trajectory for 18 s.  Reuse the LC4d local trajectory as the
locked matched control; do not spend another 40k run reproducing it.  Both use the same substrate,
connection seed, noise stream and candidate.

Required causal checks:

- shared and archived local onset are both 11 s;
- at least 3 returning events precede shared onset;
- shared executed current is exactly zero through the archived first-nonzero boundary;
- the pre-current event/rate prefix matches the local arm exactly at the stored resolution;
- local reproduces the archived failure: carrier runs beyond 5 s with core suppression and off-axis
  persistence;
- shared produces a bounded, non-refractory high bout lasting 1--5 s and an autonomous offset;
- at least 2 s of no-ictal relapse follow offset and post-offset rate is below pre-onset rate;
- finite, zero clip and refractory fraction `<=0.01`.

The positive label is:

```text
SPATIALLY_SHARED_OFFSET_CANDIDATE
```

It is a one-seed causal architecture candidate, not a lifecycle claim.  If shared does not terminate,
report `SHARED_EXECUTOR_OFFSET_NEGATIVE`.  If it terminates before a 1 s carrier, report
`SHARED_EXECUTOR_OVERFAST`.  If it changes entry before any current is delivered, report
`CAUSAL_PREFIX_MISMATCH` and stop as an engineering failure.

## 6. E2 conditional 70 s lifecycle and exact-D confirmation

Only E1 positive launches one fresh 70 s nominal trajectory with the identical candidate.  Reuse the
locked LC4 nominal gates:

- at least 8 s and 3 returning IEDs before spontaneous onset;
- bounded non-refractory 1--5 s high bout;
- autonomous offset and at least 2 s relapse protection;
- final 8 s wholly after protection, interictal, with at least 3 returning events inside the frozen
  event-rate, duration and participation bands.

Only nominal eligibility launches the unchanged 12 s exact-final-state confirmation with the actual
spatial D field frozen.  After 2 s burn-in, the remaining 10 s must stay low/interictal and reference-
like.

## 7. Interpretation and route after this sprint

- Shared positive, local negative: spatial coordination is causally required at matched mean dose.
  The next mechanism mapping must realize this through the X/recruited-area/non-local inhibitory path;
  this abstract shared executor is not itself the final physiological claim.
- Both negative: stop the LC4 cell-load actuator family; do not escalate `g_m_max` again.  Return to the
  already demonstrated X relay offset surface or a recruited-area inhibitory coordinate.
- Shared overfast: a spatially coordinated path has authority but needs a slower rise or bounded spatial
  fraction in a separately registered sprint; do not weaken the carrier minimum post hoc.
- Full E2 pass: report a single-seed lifecycle candidate only; morphology and confirmation seeds remain
  separate gates.

## 8. Resources and detached execution

- One 40k worker and one pinned BLAS/OpenMP thread for every stage.
- Every simulation is a distinct `setsid nohup` session with PID/SID, source lock, stage flock and
  RUNNING/DONE/FAILED/STOP sentinels.
- Do not submit below 128 GiB MemAvailable.  Swap +256 MiB blocks later stages; +512 MiB terminates only
  the newest task-owned process.
- E1 wall guard 4 h; E2 nominal 7 h; exact-D 3 h.
- Never touch sibling processes/worktrees.
- Result root:
  `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/`.


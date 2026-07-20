# Topic 4 Stage 0D: local-basin and neighbouring-parameter replication (locked design)

**Version:** 1.0 (locked before implementation and execution)
**Date:** 2026-07-20
**Role:** post-discovery prospective replication; it does not amend or reinterpret Stage 0C transfer-support v1.1.

## 1. Question and claim boundary

Stage 0C transfer-support v1.1 found one numerically credible trajectory at
`z=0.85, alpha_G=16, root_0_plus` (`mean=5.969 Hz`, `frequency=1.665 Hz`), but
only one of 17 histories survived and no parameter point passed the two-history
basin gate. Stage 0D asks one narrow question:

> Does that isolated trajectory belong to an open local basin, and does the same
> finite oscillatory object persist at at least one pre-specified neighbouring
> frozen parameter point?

This is a prospective replication because the state and parameter neighbourhood,
integration schedule, numerical checks, and acceptance rules below are fixed
before Stage 0D code or results exist. The discovery point and waveform are known,
so Stage 0D is not a blind discovery or preregistered confirmation of the original
Stage 0C scan. It cannot be written back into the v1.1 result.

The following remain out of scope: changing `w_ee_mult`, external drive, transfer
function, pool time constants, recruitment nonlinearity, noise, slow `Z/M/r`
dynamics, spatial coupling, stimulation, dynamic threshold, and SNN simulation.
No Stage 1 simulation may be launched by this runner. A positive Stage 0D result
only makes the fast object eligible for an independent Stage 1 review.

## 2. Frozen implementation and provenance inputs

Stage 0D must reuse without modification:

- the nine-dimensional Stage 0C state and dynamic divisive-pool equations in
  `src/topic4_spatial_slowfast_stage0c.py`;
- the no-clip/no-extrapolation extra-fine Siegert transfer and every-Euler audit
  in `src/topic4_spatial_slowfast_stage0c_transfer.py`;
- `w_ee_mult=1.1`, external-drive ratio `1.0`, and all existing Stage 0C constants;
- the `root_0_plus` initial state reconstructed from the preserved root artifact.

Locked upstream files at design time:

| Input | SHA256 |
|---|---|
| `stage0c_dynamic_divisive_pool/root_continuation.json` | `eb89514f62e127367774e1c849ec16f340ef2b4ac1713b3f57b6c9696ed4999f` |
| `stage0c_transfer_support_audit_v1_1/extended_transfer_extra_fine.npz` | `dd40a7b82e1ca5ca4a6fcf514b1e0c721242502e3806133295e4c4411bd4e703` |
| `stage0c_transfer_support_audit_v1_1/state_fork_confirm_extra_fine_traces.npz` | `8151891d26dffefc9b87dc1fcf107269e515090277e8bf996c7e5a5e212e1579` |
| `src/topic4_spatial_slowfast_stage0c_transfer.py` | `48ab839f6039134bfab22968d6deaad25011fbc654c206363341eef1bb1bc7ed` |

The result must record hashes of these inputs and of the Stage 0D spec, config,
module, and runner. Any mismatch is an engineering/provenance failure, not a
scientific result.

## 3. Fixed phase source

1. Reconstruct the centre-point `root_0_plus` state exactly as Stage 0C did.
2. Integrate it at `z=0.85, alpha_G=16` for 12,000 ms with `dt=0.125 ms` and
   save every 5 ms using the preserved extra-fine transfer.
3. The independent full-state tracer must be checked against the authoritative
   `simulate_extended_forks` output for all shared traces. Maximum absolute
   disagreement is `1e-7 kHz` for rates and `1e-7` for pool coordinates.
4. The authoritative trace must pass all support/state-bound audits, direct-exact
   transfer error (`max <=0.25 Hz`, meaningful-rate `p99 <=2%`), and retain the
   `bounded_oscillatory_candidate` class.
5. In the interval `t>=7200 ms`, find E-rate peaks with height at least 20 Hz,
   prominence at least 10 Hz, and separation at least 300 ms. Use the final
   complete peak-to-peak cycle. Select the saved full states nearest phases
   `0, 0.25, 0.50, 0.75` of that cycle.

Failure of any source condition yields `STAGE0D_PHASE_SOURCE_INVALID` and stops
before the 3x3 replay.

## 4. Fixed parameter and initial-state battery

The frozen 3x3 parameter neighbourhood is the Cartesian product:

```text
z       = [0.84, 0.85, 0.86]
alpha_G = [15, 16, 17]
```

The four Manhattan neighbours of the centre are the only points that can satisfy
the neighbouring-parameter gate. The four diagonal points are fixed sensitivity
readouts and cannot rescue a failed neighbour gate.

At each of the four phase states, create exactly five histories:

1. `phase_anchor`: unchanged orbit state (phase control; not open-basin evidence);
2. `fast_plus`: multiply coordinates `rE,rI,sEE,sEI,sIE,sII` by `1.03`;
3. `fast_minus`: multiply those six coordinates by `0.97`;
4. `pool_plus`: multiply `rE_fast,mu_G,S_G` by `1.03`;
5. `pool_minus`: multiply those three coordinates by `0.97`.

Thus every parameter point has 20 histories and the full screen has 180. No
random perturbation, replacement state, coordinate clipping, or adaptive radius
is allowed. Every perturbed state must naturally remain finite, non-negative,
inside rate/refractory and pool bounds. Otherwise the run fails closed.

## 5. Cheap-first integration and numerical gates

All runs use the extra-fine transfer, forward Euler, one process, and one BLAS
thread. Peak RSS must be below 4 GiB.

| Phase | dt | duration | save stride | Who runs |
|---|---:|---:|---:|---|
| screen | 0.25 ms | 6,000 ms | 20 | all 180 histories |
| confirm | 0.125 ms | 24,000 ms | 40 | screen survivors only |
| dt/2 | 0.0625 ms | 24,000 ms | 80 | confirm survivors only |

The existing Stage 0C classifier is unchanged (`tail_fraction=0.40`, low mean
5 Hz, finite upper bound 100 Hz, at least three cycles, spectral ratio at least
0.20, and the existing drift/slope/refractory rules). At every phase, each fork
must pass:

- no transfer-support violation, clipping, extrapolation, NaN, negative rate,
  state-bound violation, or synaptic-bound violation at any Euler state;
- no E or I refractory-tail occupancy above 5%;
- no E rate above 100 Hz in the classification tail;
- direct exact-vs-LUT audit (`max absolute <=0.25 Hz`, meaningful-rate p99
  relative error `<=2%`);
- identical candidate class under confirm and dt/2;
- confirm-vs-dt/2 mean-rate difference `<=max(1 Hz, 10% of pair mean)`;
- confirm-vs-dt/2 frequency difference `<=max(0.25 Hz, 10% of pair mean)`;
- confirm-vs-dt/2 peak-to-trough amplitude difference
  `<=max(5 Hz, 10% of pair mean amplitude)`.

`bounded_indeterminate`, `indeterminate_long_transient`, class disagreement, or
an unfinished low/high transition is `numerical_unresolved`; it is never counted
as a survivor. The runner must preserve those rows rather than extend selected
trajectories or tune thresholds after inspection.

## 6. Open-basin and neighbour replication gates

For a parameter point to have **open-local-basin support**, its final survivors
must include all of the following:

1. at least two off-orbit histories (`phase_anchor` never counts);
2. both independent perturbation families (`fast` and `pool`);
3. at least two distinct phase IDs;
4. across those off-orbit survivors, mean-rate range
   `<=max(1 Hz, 10% of their mean)`, frequency range
   `<=max(0.25 Hz, 10% of their mean)`, and peak-to-trough amplitude range
   `<=max(5 Hz, 10% of their mean amplitude)`.

The full Stage 0D replication passes only if:

- the centre point `(0.85,16)` has open-local-basin support; and
- at least one of `(0.84,16)`, `(0.86,16)`, `(0.85,15)`, `(0.85,17)` also has
  open-local-basin support; and
- that neighbour's basin-centroid mean, frequency, and amplitude agree with the
  centre under the same pairwise tolerances above.

Locked verdicts:

- `STAGE0D_REPLICATED_OPEN_BASIN_AND_LOCAL_PARAMETER_SUPPORT`;
- `STAGE0D_CENTER_BASIN_ONLY_NO_NEIGHBOR_REPLICATION`;
- `STAGE0D_NO_REPLICATION_WITH_UNRESOLVED_TRAJECTORIES`;
- `STAGE0D_CLEAN_NO_LOCAL_BASIN_REPLICATION`;
- `STAGE0D_PHASE_SOURCE_INVALID`;
- `STAGE0D_ENGINEERING_OR_PROVENANCE_FAIL`.

No qualitative visual judgement can override these gates.

## 7. Required outputs

Independent root:

`results/topic4_sef_hfo/spatial_slowfast_topology/stage0d_local_basin_replication/`

Required artifacts are:

- `stage0d_local_basin_summary.json`;
- `fork_outcomes.json` and `fork_outcomes.csv`;
- `parameter_point_outcomes.json` and `parameter_point_outcomes.csv`;
- `phase_source.json` and compact trace NPZ;
- screen/confirm/dt-half compact trace NPZ files;
- `figures/stage0d_local_basin_replication.png` plus Chinese `figures/README.md`;
- `STATUS.md` with verdict, counts, numerical audits, resource use, and explicit
  statement that slow/spatial Stage 1 remains outside this run.

## 8. Stop rules

- Do not alter upstream Stage 0C/v1.1 files or their interpretation.
- Do not change this spec, config, battery, thresholds, duration, or gate after
  inspecting Stage 0D outcomes.
- Stop after the locked verdict. Do not launch slow or spatial simulation.
- A negative or unresolved result closes this specific local dynamic-divisor
  neighbourhood; it is not proof against every current-based or conductance-based
  model.

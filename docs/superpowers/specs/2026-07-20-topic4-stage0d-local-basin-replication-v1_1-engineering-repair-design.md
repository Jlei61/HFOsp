# Topic 4 Stage 0D v1.1: engineering repair (locked design)

**Version:** 1.1
**Date:** 2026-07-20
**Role:** independent engineering repair and authoritative rerun; Stage 0D v1 remains immutable historical evidence.

## 1. Why v1.1 exists

Stage 0D v1 correctly preserved the 180-history prospective battery and returned
`STAGE0D_NO_REPLICATION_WITH_UNRESOLVED_TRAJECTORIES`. Its implementation has two
engineering deviations that require a separate result lineage:

1. The v1 spec required confirm-vs-dt/2 frequency agreement
   `<=max(0.25 Hz, 10% of pair mean)`. The v1 helper first called the older
   Stage-0C temporal gate, whose frequency tolerance is
   `<=max(0.5 Hz, 15% of pair mean)`, and did not subsequently apply the stricter
   Stage 0D frequency tolerance. The implemented gate was therefore more
   permissive than the written spec.
2. The numeric v1 runner left Figure B as an empty default axis when the centre
   had no dt/2 survivor. A later visualization-only derivative showed diagnostic
   traces, but that did not repair the authoritative runner. v1.1 must explicitly
   write `none passed locked gate` in Figure B whenever no centre trajectory
   passes the final locked gate.

The repair cannot be written into v1 artifacts. It gets a new spec, config,
module, runner, tests, and result root.

## 2. Immutable v1 lineage

The following v1 files are locked read-only inputs:

| v1 file | SHA256 |
|---|---|
| `docs/superpowers/specs/2026-07-20-topic4-stage0d-local-basin-replication-design.md` | `21f45fb5d1f980a7001e605cc9f6f53c8aad2289996f5d898a08a1c615e55d89` |
| `config/topic4_spatial_slowfast_stage0d.yaml` | `2413909ce3c5c51418ccdfb95685ac7d3e76b3aae7a051fc26ae966489258793` |
| `src/topic4_spatial_slowfast_stage0d.py` | `7ded4464f77ac96c37c9669c9fb16b5b2c13eb66c75d0acce563ed58199952d7` |
| `scripts/run_topic4_spatial_slowfast_stage0d.py` | `5bf87bc5a6d574f19072b4c3fae5dbc1d10a6205647b6c12afa90549b32388cb` |
| `tests/test_topic4_spatial_slowfast_stage0d.py` | `6bac873d6100e0aa287884af363a2044ba774fc60c85e08acb34859f067826ed` |
| `stage0d_local_basin_replication/stage0d_local_basin_summary.json` | `0ac3084102e56085ddf6c4e3f8dfd5ace1e16214efb2a22bf7ad25739f4dfccd` |
| `stage0d_local_basin_replication/fork_outcomes.json` | `dcba29655fd673503b0d462debf23dbfc456cb8f485477aa114e1c2854e26fee` |
| `stage0d_local_basin_replication/phase_source.json` | `f676c6f331cbaa1401ff348e014ded724aba22d4e6000c3ff50ffa9ea1e3cec9` |

Any mismatch stops v1.1 as `STAGE0D_V1_1_ENGINEERING_OR_PROVENANCE_FAIL`.

## 3. Scientific contract frozen to v1

The following must be byte-for-byte equivalent in meaning to v1:

- Stage-0C nine-state dynamic-divisor equations and extra-fine no-clip transfer;
- centre `z=.85, alpha_G=16, root_0_plus` and phase-source construction;
- parameter Cartesian product `z=[.84,.85,.86]`, `alpha_G=[15,16,17]`;
- four phases `[0,.25,.50,.75]` and five histories
  `[phase_anchor,fast_plus,fast_minus,pool_plus,pool_minus]`;
- exactly 180 screen histories and fixed 3% perturbation radius;
- screen `6 s / .25 ms`, confirm `24 s / .125 ms`, dt/2
  `24 s / .0625 ms`, with unchanged save strides;
- all classifier thresholds, direct-exact gates, state/support audits, open-basin
  requirements, Manhattan-neighbour rule, verdict logic, and `<4 GiB` contract;
- no Stage 0E, slow-variable, spatial, noise, or SNN simulation.

The config validator must compare all frozen scientific sections against the v1
config and reject drift.

## 4. The only numerical repair

For every confirm survivor, dt/2 acceptance must require:

- identical `bounded_oscillatory_candidate` class;
- rate difference `<=max(1 Hz, 10% of pair mean)`;
- **frequency difference `<=max(0.25 Hz, 10% of pair mean)`**;
- peak-to-trough amplitude difference `<=max(5 Hz, 10% of pair mean)`;
- all unchanged finite, support, state-bound, refractory, over-100-Hz, and
  direct-exact requirements.

The v1.1 helper may use the older gate only as an initial fail-closed screen. It
must always apply and record the explicit `.25 Hz / 10%` frequency calculation
before returning `candidate_survives`. Unit tests must include a frequency pair
that passes `.5 Hz / 15%` but fails `.25 Hz / 10%`.

## 5. Figure repair

Figure B is titled `Centre dt/2 survivors` and is driven only by v1.1
`final_status == candidate_survives` rows at `(z=.85, alpha_G=16)`.

- If such rows exist, plot them and state their count.
- If none exist, render the exact visible phrase **`none passed locked gate`**;
  do not substitute a screen/confirm trajectory or a neighbouring parameter.

The Chinese `figures/README.md` must describe what Figure B actually contains.

## 6. Pre-execution implementation lock

Before any phase-source or battery integration, the authoritative runner must:

1. validate the v1 immutable hashes;
2. validate that all scientific config sections match v1;
3. hash the v1.1 spec, config, module, runner, and tests;
4. write those hashes to the new result root as `EXECUTION_LOCK.json`;
5. re-hash them after simulation and fail closed if any changed.

The final summary embeds the same lock and verifies it. Thus source hashes are
fixed before numerical execution without creating a self-referential config hash.

## 7. Result and comparison contract

New root:

`results/topic4_sef_hfo/spatial_slowfast_topology/stage0d_local_basin_replication_v1_1/`

It must contain the same numeric artifact classes as v1, plus:

- `EXECUTION_LOCK.json`;
- `v1_vs_v1_1_comparison.json` with per-fork status changes, verdict/count
  changes, and an explicit `scientific_result_changed` boolean;
- summary fields describing the v1 deviation, repaired tolerance, and whether
  the scientific result changed.

Scientific verdict labels remain the v1 labels for direct comparison. An
engineering/provenance failure uses
`STAGE0D_V1_1_ENGINEERING_OR_PROVENANCE_FAIL`.

## 8. Stop rules

- Never edit, overwrite, delete, or reinterpret any v1 file or artifact.
- Never change the scientific contract after seeing v1.1 output.
- Stop after v1.1 comparison and QA; do not start Stage 0E/slow/space.
- If v1.1 agrees with v1, report that the repair closes implementation drift but
  does not strengthen the scientific evidence.

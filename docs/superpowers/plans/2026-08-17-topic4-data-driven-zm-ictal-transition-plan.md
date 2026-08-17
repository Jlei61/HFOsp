# Topic 4 ZM-ITX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the frozen patient-constrained Node + E→E + E→I substrate for `epilepsiae_1146`, switch on the per-neuron Z/M slow variables, capture the interictal-to-sustained-high-activity transition, and measure whether local perturbation susceptibility rises before the transition and is spatially organized along the data-driven substrate.

**Architecture:** Four off-by-default parameters are added to the existing LIF engine (`post_runaway_record_ms`, `checkpoint_steps`/`checkpoint_sink`, `resume_state`, `time_offset_ms`) with byte-parity gates. A new substrate-rebuild module reconstructs the frozen rev11-NLC candidates without touching the producer script. A primary worker runs the 20 s Z/M trajectories and drops checkpoints; a perturbation worker resumes from those checkpoints and branches into paired sham/probe continuations that share the RNG stream. All statistics are paired over network seeds.

**Tech Stack:** Python 3.11.5, numpy 1.26.4, scipy sparse, pytest, matplotlib. Interpreter `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`. Long runs via `systemd-run --user` + `nohup`.

**Spec:** `docs/superpowers/specs/2026-08-17-topic4-data-driven-zm-ictal-transition-design.md`

## Global Constraints

- Work only inside `.worktrees/topic4-data-driven-zm-ictal-transition` on branch `codex/topic4-data-driven-zm-ictal-transition`. The dirty main worktree at `/home/honglab/leijiaxin/HFOsp` is never modified.
- Base commit: `7393745c6777adaf88fbf0c5bc087e4c2f1c0a9e`.
- `scripts/run_topic4_rev10_r_edge_flow_worker.py` is **never modified** — it is the producer of record for the frozen substrate.
- Frozen manifest sha256 `545b029d2d7947de5a27979e7166f6bc55ea2b01c64a67a9b736977ec960fcbb`; Z/M reference config sha256 `2b9586d274b85d9e3663557b5f4dfab7ac64292817667020503d144579ff8a91`. Verified before every run.
- Z/M parameters, verbatim, in every formal arm: `use_z=true`, `use_m=true`, `I_th_EI=95.19851312666987`, `tau_z=5000.0`, `tau_adp=500.0`, `eta_m=0.007451594355587098`, `trace_stride_steps=10`. No other slow protocol (`q_I`, `g_K`, `h_G`, EE-STD, adaptation, inhibitory resource) is ever enabled.
- Engine geometry, frozen: `L=20.0` mm, `density=100.0`, `g=3.6`, `dt=0.1` ms, `AR=2.0`, `theta_EE=-22.805383965058470` deg, `nu_ext_ratio` from `cmrun.DRIVE`, `n_E=32000`, `n_I=8000`, `N_core_manual=1129`, `quantile_seed=20260806`.
- Spatial OU, frozen: `mode="local"`, `sigma_rate_per_ms=0.1`, `tau_ms=20.0`, `ell_mm=0.38`, `update_interval_ms=1.0`, `grid_spacing_mm=0.4`, `seed_offset=500000`.
- Common detector threshold, frozen: `population_active_fraction_threshold = 0.0195703125`.
- Contact readout, frozen: 15 contacts (`ICL1..ICL11`, `SCL6..SCL9`), `participation_margin_fraction=0.1`, `timing_fraction=0.5`, `k_dir=2`, `part_min=5`.
- Every new off-by-default engine parameter must leave the default path byte-identical. Any task that adds one ends with an explicit parity test.
- Network seeds: `1801-1803` (canary), `1811-1822` (formal and null). Seed `1561` is used only for the parity gate.
- Numeric threads pinned to 1 per worker (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`).
- Tests run from the worktree root: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest <path> -v`.
- Commit after every task. Never `git add -A`; always name files.

## Facts established before planning (do not re-derive)

- Under Z/M **off**, 48 runs (4 arms × seeds 1561-1572, 20 s) produced **0 transitions** and ~105 detected / ~87 returned events per run. This is the control this round cites.
- One 20 s run of `joint_04_control` at seed 1561 took **1890.7 s wall** (~31.5 min), single-threaded, peak RSS **14.6 GiB**.
- The network cache directory `results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache` **no longer exists**. Every seed must be rebuilt. The rebuild guard in `scripts/run_topic4_rev9_node_kick_canary.py::_load_network` has been verified to pass at the base commit: `src/snn_engine/{params,connectivity,connectivity_rot}.py` all match the hashes in `node_kick_canary.json`, and numpy is 1.26.4 as required.
- The archived seed-1561 network pickle hash is recorded as `cache_sha256 = dba81d32d6c542bda4d1cfa0de196551c16f811a88c0864c7572a8db60852828` inside `.../workers/joint_04_control_seed_1561.json` → `network.cache_source`.
- `results/` is gitignored except three legacy files; the frozen manifest is untracked, so the worktree checkout does **not** contain it. Task 1 wires it in.
- Machine: 80 cores, 251 GiB RAM (229 GiB available), 187 GiB free on `/`.

## File Structure

**Create:**

| Path | Responsibility |
|---|---|
| `config/topic4_data_driven_zm_ictal_transition_v1.json` | round config: frozen input hashes, arms, phases, seeds, sites, dose ladder, D4 assignment |
| `src/snn_engine/checkpoint.py` | single enumeration point for simulator state capture/restore/serialise |
| `src/topic4_zm_ictal_transition.py` | rebuild the frozen substrate (node field, local-connectivity mapper, montage, slow object, OU drive, pathway gains) without touching the rev10-R producer |
| `src/topic4_zm_d4.py` | covariant D4 transform of node field queries and directed flow coefficients |
| `src/topic4_zm_state_characterization.py` | what the sustained high-activity state actually is |
| `src/topic4_zm_perturbation.py` | frozen probe sites, packet selection, sham-subtracted response metrics, susceptibility maps, hotspot compactness |
| `src/topic4_zm_statistics.py` | paired network bootstrap, restricted ictal-free time, spatial correlation with a spatial null |
| `scripts/run_topic4_zm_ictal_transition_worker.py` | one primary 20 s run: substrate → Z/M → trajectory → checkpoints → readout |
| `scripts/run_topic4_zm_perturbation_worker.py` | one (network, checkpoint) job: load once, loop all sites, sham/probe pairs |
| `scripts/launch_topic4_zm_ictal_transition.py` | memory sentinel, worker pool, systemd units, 600 s monitor |
| `scripts/aggregate_topic4_zm_ictal_transition.py` | cohort tables and the pre-registered statistics |
| `scripts/audit_topic4_zm_ictal_transition.py` | re-derives every reported number from artifacts |
| `scripts/freeze_topic4_zm_ictal_transition.py` | manifest + provenance freeze before any run |
| `scripts/paper_figures/plot_topic4_zm_ictal_transition_panels.py` | the ten panel producers |
| `scripts/paper_figures/build_main_figure_5.py` | both assemblies from one set of panels |
| `tests/test_snn_checkpoint.py`, `tests/test_zm_ictal_transition_substrate.py`, `tests/test_zm_d4.py`, `tests/test_zm_state_characterization.py`, `tests/test_zm_perturbation.py`, `tests/test_zm_statistics.py`, `tests/test_kick_probe_zm_itx_parity.py` | tests |

**Modify:**

| Path | Change |
|---|---|
| `src/snn_engine/kick_probe.py` | add `post_runaway_record_ms`, `checkpoint_steps`, `checkpoint_sink`, `resume_state`, `time_offset_ms` — all off by default |
| `src/snn_engine/mz_slow_vars.py` | add an off-by-default per-neuron slow-current accumulator |
| `docs/topic4_sef_hfo.md` | round entry after the run completes |
| `results/FIGURE_INDEX.md` | Fig5 candidate entry after the figure exists |

---

### Task 1: Results-tree wiring, round config, and the substrate rebuild module

**Files:**
- Create: `config/topic4_data_driven_zm_ictal_transition_v1.json`
- Create: `src/topic4_zm_ictal_transition.py`
- Test: `tests/test_zm_ictal_transition_substrate.py`

**Interfaces:**
- Consumes: the frozen rev11-NLC manifest and the rev10-R construction sequence at `scripts/run_topic4_rev10_r_edge_flow_worker.py:240-330` (read, never imported or modified).
- Produces:
  ```python
  @dataclass
  class Substrate:
      params: Any               # snn_engine params.Params
      net: dict                 # mapped network (post local-connectivity flow)
      raw_net: dict             # pre-mapping network, same object identity rules
      n_e: int                  # 32000
      n_i: int                  # 8000
      positions_e: np.ndarray   # (n_e, 2) float64
      positions_i: np.ndarray   # (n_i, 2) float64
      h_e: np.ndarray           # (n_e,) float64, sums to 1129
      h_i: np.ndarray           # (n_i,) float64
      vtheta: np.ndarray        # (n_e + n_i,) float64, node["vtheta"]
      delta_vtheta: np.ndarray  # (n_e,) float64, node["delta_vtheta"]
      montage: Any              # VirtualMontage over 15 contacts
      contact_names: list[str]
      contact_xy: np.ndarray    # (15, 2)
      shaft_ids: np.ndarray     # (15,) dtype "U8"
      valid_contacts: np.ndarray  # (15,) bool, all True by contract
      edge_audit: dict
      edge_coefficients: np.ndarray  # (2, 6) float64, post-transform if any
      ee_out_gain: np.ndarray   # (n_e,) post/pre outgoing E->E weight
      etoi_out_gain: np.ndarray # (n_e,) post/pre outgoing E->I weight
      axis_unit: np.ndarray     # (2,) = (0.92182673, -0.38760221)
      axis_source_xy: np.ndarray  # (4.19921432, 9.12890135)
      axis_sink_xy: np.ndarray    # (16.47920304, 3.96551153)
      detector_threshold: float   # 0.0195703125
      engine: dict
      network_cache: dict         # key, path, sha256, hit

  def load_round_config(path) -> dict
  def verify_frozen_inputs(config: dict) -> dict          # raises RuntimeError on drift
  def build_substrate(config, candidate_id, seed, *, cache_dir,
                      field_transform=None) -> Substrate
  def make_slow(substrate: Substrate, zm_cfg: dict)       # -> MZSlowVars | None
  def make_external_drive(substrate: Substrate, ou_cfg: dict, seed: int)  # -> SpatialOUDrive | None
  ```

- [ ] **Step 1: Wire the results tree**

The worktree has no `results/topic4_sef_hfo`. Create it and symlink the read-only inputs from the main tree; keep this round's outputs and the network cache local so the worktree stays deletable.

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-data-driven-zm-ictal-transition
MAIN=/home/honglab/leijiaxin/HFOsp
mkdir -p results/topic4_sef_hfo
for d in data_driven_core_field data_driven_core_field_rev9 \
         data_driven_core_field_rev10_sa data_driven_core_field_rev10_d \
         data_driven_local_connectivity_rev11_nlc; do
  ln -sfn "$MAIN/results/topic4_sef_hfo/$d" "results/topic4_sef_hfo/$d"
done
mkdir -p results/topic4_sef_hfo/data_driven_zm_ictal_transition/{workers,checkpoints,perturbation,network_cache,run_logs}
mkdir -p results/paper-ready-figure/fig5/figures
ls -l results/topic4_sef_hfo | head
```

Expected: five symlinks resolving into `$MAIN`, plus one real directory. Verify the manifest is now readable:

```bash
sha256sum results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/candidate_manifest.json
```

Expected: `545b029d2d7947de5a27979e7166f6bc55ea2b01c64a67a9b736977ec960fcbb`.

- [ ] **Step 2: Write the round config**

Create `config/topic4_data_driven_zm_ictal_transition_v1.json`. Copy the `inputs` hash records verbatim from `config/topic4_rev11_nlc_frozen_substrate_confirmation.json`, then add this round's own blocks:

```json
{
  "schema_version": "topic4_data_driven_zm_ictal_transition_v1",
  "scientific_role": "development_only_data_driven_zm_interictal_to_ictal_transition",
  "output_root": "results/topic4_sef_hfo/data_driven_zm_ictal_transition",
  "inputs": {
    "frozen_substrate_manifest": {
      "path": "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/candidate_manifest.json",
      "sha256": "545b029d2d7947de5a27979e7166f6bc55ea2b01c64a67a9b736977ec960fcbb"
    },
    "zm_baseline": {
      "path": "config/topic4_data_driven_snn_baseline_zm_v1.json",
      "sha256": "2b9586d274b85d9e3663557b5f4dfab7ac64292817667020503d144579ff8a91"
    },
    "rev11_config": {
      "path": "config/topic4_rev11_nlc_frozen_substrate_confirmation.json",
      "sha256": "7e3ff4786a96ed5fc5b27f7b6b292eb5d7cd51fedb847f669c0e648165835df7"
    },
    "spec": {
      "path": "docs/superpowers/specs/2026-08-17-topic4-data-driven-zm-ictal-transition-design.md",
      "sha256": "88309d38cf56231a62df000bdfac46d332479d4afd549e5c6e1724ed23748955"
    }
  },
  "arms": {
    "Node": "node_baseline",
    "Node+EE": "joint_04_ee_only",
    "Node+EtoI": "joint_04_etoi_only",
    "Joint": "joint_04_control"
  },
  "zm": {
    "mode": "z_plus_m", "use_z": true, "use_m": true,
    "I_th_EI": 95.19851312666987, "tau_z": 5000.0,
    "tau_adp": 500.0, "eta_m": 0.007451594355587098,
    "trace_stride_steps": 10
  },
  "simulation": {
    "duration_ms": 20000.0,
    "early_stop_runaway": true,
    "es_thresh_hz": 120.0,
    "es_dur_ms": 100.0,
    "post_runaway_record_ms": 500.0
  },
  "seeds": {"canary": [1801, 1802, 1803], "formal": [1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822]},
  "checkpoints": {
    "baseline_ms": 2000.0,
    "pre_ictal_offset_ms": 500.0,
    "sensitivity_offset_ms": 1000.0,
    "minimum_onset_for_perturbation_ms": 2500.0,
    "minimum_onset_for_sensitivity_ms": 3500.0
  },
  "interictal_baseline_gate": {
    "minimum_onset_ms": 2500.0,
    "minimum_returned_events_before_onset": 3,
    "baseline_window_ms": [1500.0, 2000.0],
    "zm_off_reference_workers": "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/workers",
    "baseline_rate_percentile": 95.0
  },
  "perturbation": {
    "dose_ladder_cells": [16, 32, 64, 128, 256],
    "dose_minimum_median_excess_spikes": 200.0,
    "dose_maximum_event_triggering_sites": 1,
    "packet_radius_mm": 1.0,
    "response_window_ms": 200.0,
    "response_split_ms": 50.0,
    "grid_seeds": [1811, 1812, 1813],
    "grid_extent_mm": [3.0, 17.0],
    "grid_n": 7,
    "baseline_onset_search_cap_ms": 20000.0
  },
  "substrate_null": {
    "elements": ["r90", "r180", "r270", "mx", "my", "md1", "md2"],
    "assignment": {"1811": "r180", "1812": "r90", "1813": "r270", "1814": "mx",
                   "1815": "my", "1816": "md1", "1817": "md2", "1818": "r180",
                   "1819": "r90", "1820": "r270", "1821": "mx", "1822": "my"},
    "arm": "Joint",
    "minimum_transitioned_null_networks": 6
  },
  "statistics": {"bootstrap_draws": 4096, "bootstrap_seed": 20260817, "spatial_null_draws": 2000},
  "execution": {
    "max_workers": 8,
    "minimum_available_memory_gib": 32.0,
    "numeric_threads_per_worker": 1,
    "monitor_interval_seconds": 600
  }
}
```

- [ ] **Step 3: Write the failing substrate test**

Create `tests/test_zm_ictal_transition_substrate.py`. The decisive test rebuilds seed 1561's network and compares against the recorded cache hash and the archived field statistics.

```python
"""Substrate rebuild must reproduce the frozen rev11-NLC substrate exactly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, verify_frozen_inputs)

CONFIG = ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"
ARCHIVE = ROOT / ("results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
                  "/frozen_substrate_confirmation/workers")


def test_frozen_inputs_verify():
    config = load_round_config(CONFIG)
    report = verify_frozen_inputs(config)
    assert report["all_match"] is True


@pytest.mark.slow
@pytest.mark.integration
def test_seed_1561_substrate_matches_archive(tmp_path):
    config = load_round_config(CONFIG)
    sub = build_substrate(config, "joint_04_control", 1561,
                          cache_dir=str(tmp_path))
    archived = json.loads((ARCHIVE / "joint_04_control_seed_1561.json").read_text())

    assert sub.n_e == 32000 and sub.n_i == 8000
    assert sub.network_cache["sha256"] == archived["network"]["cache_source"]["cache_sha256"]

    with np.load(ARCHIVE / "joint_04_control_seed_1561.npz", allow_pickle=False) as z:
        assert np.array_equal(sub.h_e.astype(np.float32), z["h"])
        assert np.array_equal(sub.positions_e.astype(np.float32), z["positions_E"])
        assert np.array_equal(sub.delta_vtheta.astype(np.float32), z["delta_vtheta"])
        assert np.array_equal(sub.h_i.astype(np.float32), z["h_I_for_edge"])
        assert np.array_equal(sub.edge_coefficients, z["edge_coefficients"])
        assert list(sub.contact_names) == list(z["contact_names"])
        assert np.array_equal(sub.contact_xy, z["contact_xy_mm"])
    assert sub.edge_audit["coefficients_sha256"] == archived["edge_audit"]["coefficients_sha256"]
    assert np.isclose(sub.h_e.sum(), 1129.0, atol=1e-8)
    assert np.allclose(sub.axis_unit, [0.92182673, -0.38760221], atol=1e-8)
```

- [ ] **Step 4: Run it to confirm it fails**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_ictal_transition_substrate.py -v`
Expected: `ModuleNotFoundError: No module named 'src.topic4_zm_ictal_transition'`.

- [ ] **Step 5: Implement `src/topic4_zm_ictal_transition.py`**

Mirror the construction sequence at `scripts/run_topic4_rev10_r_edge_flow_worker.py:240-330` exactly, in this order — any reordering changes RNG consumption and breaks the parity gate:

1. `params = Params(g, L, density, T=duration_ms, dt, nu_ext_ratio=cmrun.DRIVE, seed=seed)`
2. `net, n_e, n_i, hit, source = _load_network(params, stage, _placement(stage), seed, base, cache_dir)` — import `_load_network` from `scripts.run_topic4_rev9_node_kick_canary`, `_placement` from `src.topic4_core_field_runner`.
3. `positions = net["pos"][:n_e]`
4. `node = _candidate_node(candidate["node_field"], positions, n_total=n_e+n_i, stage=stage, config=anchor_config)`; assert `node["h"].sum() ≈ 1129`.
5. `coefficients = np.asarray(candidate["coefficients"], float)`; assert `array_sha256(coefficients) == candidate["coefficients_sha256"]`.
6. `net["rng"] = np.random.default_rng(seed)`
7. `h_e, h_i, field_audit = continuous_field_h_with_queries(node_field["coefficients"], query_e, query_i, n_basis, degree, target_count=1129, L=20.0)` where `query_e`/`query_i` are the E/I positions, or their `field_transform`-inverse images when a transform is supplied (Task 10). Assert `np.array_equal(h_e, node["h"])` when no transform is given.
8. `mapped_net, edge_audit = continuous_local_e_source_flow(net, net["pos"], concat(h_e, h_i), coefficients_eff, l_ee=0.38, l_e_to_i=0.25, raw_logit_clip=candidate.get("raw_logit_clip"))` where `coefficients_eff` is `coefficients` unless a transform is supplied.
9. Build the `VirtualMontage` from the frozen contact contract; assert `cmrun.valid_mask(...)` is all-True.
10. Pathway gains, computed from the pre- and post-mapping AMPA bins:

```python
def _outgoing_by_pathway(matrices, n_e, n_i, pathway):
    total = np.zeros(n_e, float)
    for matrix in matrices:
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        cols = np.asarray(coo.col, np.int64)
        mask = rows < n_e if pathway == "E_to_E" else rows >= n_e
        total += np.bincount(cols[mask], weights=np.asarray(coo.data[mask], float),
                             minlength=n_e)
    return total

pre_ee = _outgoing_by_pathway(raw_net["ampa_by_delay"], n_e, n_i, "E_to_E")
post_ee = _outgoing_by_pathway(mapped_net["ampa_by_delay"], n_e, n_i, "E_to_E")
ee_out_gain = np.where(pre_ee > 0.0, post_ee / np.maximum(pre_ee, 1e-300), np.nan)
```

and identically for `E_to_I`. `raw_net` must be a deep copy of the AMPA bins taken **before** `continuous_local_e_source_flow` mutates or replaces them; verify by asserting `pre_ee.sum()` is unchanged after mapping.

`verify_frozen_inputs` hashes every entry of `config["inputs"]` and returns `{"all_match": bool, "records": {...}}`, raising `RuntimeError` on any mismatch. The recorded spec hash `88309d38cf5623...` is the spec as it stands alongside this plan; if the spec is edited during review, recompute it with `sha256sum` and update the config before Task 13's freeze, which re-verifies it and fails on drift.

`make_slow` returns `MZSlowVars(n_e + n_i, params.V_th, MZSlowVarsConfig(**zm_cfg_subset), NE=n_e, core_mask_E=(h_e >= 0.5))` when `zm_cfg["mode"] != "off"`, else `None` — `core_mask_E` matches the rev10-R worker exactly.

`make_external_drive` returns `SpatialOUDrive(positions_e, L, dt, SpatialOUConfig(**ou_cfg, seed=seed + 500000))`.

- [ ] **Step 6: Run the fast test**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_ictal_transition_substrate.py::test_frozen_inputs_verify -v`
Expected: PASS.

- [ ] **Step 7: Run the slow substrate test and record build cost**

```bash
/usr/bin/time -v /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest \
  tests/test_zm_ictal_transition_substrate.py::test_seed_1561_substrate_matches_archive -v \
  2>&1 | tail -30
du -sh results/topic4_sef_hfo/data_driven_zm_ictal_transition/network_cache 2>/dev/null
```

Expected: PASS, including `cache_sha256 == dba81d32d6c542bda4d1cfa0de196551c16f811a88c0864c7572a8db60852828`.

Record the wall time, peak RSS and the pickle size in `results/topic4_sef_hfo/data_driven_zm_ictal_transition/network_build_sentinel.json`.

**Decision point, write the answer into that file:** if one network pickle exceeds 2.0 GiB, the launcher (Task 11) must cache at most 3 seeds at a time and delete a seed's pickle once its four arms finish; otherwise all 16 networks may be cached for the whole round. 16 × pickle size must stay under 60 GiB.

If `cache_sha256` does **not** match, stop. Do not proceed by comparing "close enough" arrays — report the mismatch, since it means the frozen substrate cannot be rebuilt at this commit and the entire round is blocked.

- [ ] **Step 8: Commit**

```bash
git add config/topic4_data_driven_zm_ictal_transition_v1.json \
        src/topic4_zm_ictal_transition.py \
        tests/test_zm_ictal_transition_substrate.py \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/network_build_sentinel.json
git commit -m "topic4 zm-itx: rebuild the frozen rev11-NLC substrate outside the producer script"
```

---

### Task 2: Checkpoint state module

**Files:**
- Create: `src/snn_engine/checkpoint.py`
- Test: `tests/test_snn_checkpoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  ```python
  CHECKPOINT_SCHEMA = "topic4_snn_checkpoint_v1"

  REQUIRED_KEYS = (
      "schema", "step", "absolute_time_ms", "V", "ref", "s_E", "I_E", "s_I", "I_I",
      "ring_sE", "ring_sI", "xi", "rng_state", "ras_keep", "es_ema", "es_run",
      "track_rec", "s_E_rec", "I_E_rec", "slow", "external_drive",
  )

  def capture(*, step, absolute_time_ms, V, ref, s_E, I_E, s_I, I_I,
              ring_sE, ring_sI, xi, rng, ras_keep, es_ema, es_run,
              track_rec, s_E_rec, I_E_rec, slow, external_drive) -> dict
  ```
  `absolute_time_ms` is the simulation clock **at** the captured step, so a
  resumed segment is started with `time_offset_ms = state["absolute_time_ms"]`.
  Storing the absolute time rather than a segment origin removes the only
  arithmetic that could silently desynchronise the spatial OU process.
  ```python
  def restore_slow(state: dict, slow) -> None
  def restore_external_drive(state: dict, drive) -> None
  def save(state: dict, path) -> str        # returns sha256 of the written file
  def load(path) -> dict
  def digest(state: dict) -> str            # sha256 over canonical bytes
  ```

- [ ] **Step 1: Write the failing test**

```python
"""Checkpoint capture must be complete and round-trip exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.snn_engine.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA, REQUIRED_KEYS, capture, digest, load,
    restore_external_drive, restore_slow, save)
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive  # noqa: E402


def _state(n=12, ne=8, m=5):
    rng = np.random.default_rng(7)
    slow = MZSlowVars(n, 18.0, MZSlowVarsConfig(use_z=True, use_m=True,
                                                I_th_EI=1.0, eta_m=0.01), NE=ne)
    slow.z[:] = rng.random(n)
    slow.m[:] = rng.random(n)
    slow._step_index = 41
    drive = SpatialOUDrive(rng.random((ne, 2)) * 4.0, 4.0, 0.1,
                           SpatialOUConfig(mode="local", sigma_rate_per_ms=0.1,
                                           tau_ms=20.0, ell_mm=0.4, seed=3))
    drive.step(5.0)
    return capture(
        step=137, absolute_time_ms=13.7,
        V=rng.random(n), ref=rng.integers(0, 5, n).astype(np.int32),
        s_E=rng.random(n), I_E=rng.random(n), s_I=rng.random(n), I_I=rng.random(n),
        ring_sE=rng.random((m, n)), ring_sI=rng.random((m, n)),
        xi=0.31, rng=rng, ras_keep=np.array([0, 3, 5]),
        es_ema=12.5, es_run=3, track_rec=False, s_E_rec=None, I_E_rec=None,
        slow=slow, external_drive=drive), slow, drive


def test_capture_has_every_required_key():
    state, _, _ = _state()
    assert state["schema"] == CHECKPOINT_SCHEMA
    assert set(REQUIRED_KEYS) <= set(state)
    for key in ("z", "m", "I_I_last", "step_index"):
        assert key in state["slow"]
    for key in ("field_state", "cached", "next_step", "last_step", "rng_state"):
        assert key in state["external_drive"]


def test_capture_copies_and_does_not_alias():
    state, slow, _ = _state()
    before = state["slow"]["z"].copy()
    slow.z[:] = 0.0
    assert np.array_equal(state["slow"]["z"], before)


def test_round_trip_is_exact(tmp_path):
    state, _, _ = _state()
    path = tmp_path / "ckpt.npz"
    written = save(state, path)
    assert len(written) == 64
    back = load(path)
    assert digest(back) == digest(state)
    assert np.array_equal(back["ring_sE"], state["ring_sE"])
    assert back["rng_state"] == state["rng_state"]
    assert back["step"] == 137


def test_restore_puts_slow_and_drive_back(tmp_path):
    state, slow, drive = _state()
    z_before, m_before = slow.z.copy(), slow.m.copy()
    field_before = drive._state.copy()
    slow.z[:] = 0.0
    slow.m[:] = 0.0
    drive._state[:] = 0.0
    restore_slow(state, slow)
    restore_external_drive(state, drive)
    assert np.array_equal(slow.z, z_before)
    assert np.array_equal(slow.m, m_before)
    assert np.array_equal(drive._state, field_before)
    assert drive._rng.bit_generator.state == state["external_drive"]["rng_state"]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_snn_checkpoint.py -v`
Expected: `ModuleNotFoundError: No module named 'src.snn_engine.checkpoint'`.

- [ ] **Step 3: Implement the module**

```python
"""Single enumeration point for LIF simulator state.

Every mutable quantity the integration loop reads is captured here. Adding a new
mutable engine variable without adding it to REQUIRED_KEYS is the failure mode
this module exists to prevent.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

CHECKPOINT_SCHEMA = "topic4_snn_checkpoint_v1"

REQUIRED_KEYS = (
    "schema", "step", "absolute_time_ms", "V", "ref", "s_E", "I_E", "s_I", "I_I",
    "ring_sE", "ring_sI", "xi", "rng_state", "ras_keep", "es_ema", "es_run",
    "track_rec", "s_E_rec", "I_E_rec", "slow", "external_drive",
)


def capture(*, step, absolute_time_ms, V, ref, s_E, I_E, s_I, I_I, ring_sE,
            ring_sI, xi, rng, ras_keep, es_ema, es_run, track_rec,
            s_E_rec, I_E_rec, slow, external_drive):
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "absolute_time_ms": float(absolute_time_ms),
        "V": np.array(V, copy=True),
        "ref": np.array(ref, copy=True),
        "s_E": np.array(s_E, copy=True),
        "I_E": np.array(I_E, copy=True),
        "s_I": np.array(s_I, copy=True),
        "I_I": np.array(I_I, copy=True),
        "ring_sE": np.array(ring_sE, copy=True),
        "ring_sI": np.array(ring_sI, copy=True),
        "xi": float(xi),
        "rng_state": rng.bit_generator.state,
        "ras_keep": np.array(ras_keep, copy=True),
        "es_ema": float(es_ema),
        "es_run": int(es_run),
        "track_rec": bool(track_rec),
        "s_E_rec": None if s_E_rec is None else np.array(s_E_rec, copy=True),
        "I_E_rec": None if I_E_rec is None else np.array(I_E_rec, copy=True),
        "slow": None,
        "external_drive": None,
    }
    if slow is not None:
        state["slow"] = {
            "kind": type(slow).__name__,
            "z": np.array(slow.z, copy=True),
            "m": np.array(slow.m, copy=True),
            "I_I_last": np.array(slow._I_I_last, copy=True),
            "step_index": int(slow._step_index),
        }
    if external_drive is not None:
        state["external_drive"] = {
            "field_state": np.array(external_drive._state, copy=True),
            "cached": np.array(external_drive._cached, copy=True),
            "next_step": int(external_drive._next_step),
            "last_step": int(external_drive._last_step),
            "rng_state": external_drive._rng.bit_generator.state,
        }
    return state


def restore_slow(state, slow):
    payload = state["slow"]
    if payload is None or slow is None:
        if (payload is None) != (slow is None):
            raise ValueError("checkpoint slow payload and slow object disagree")
        return
    if payload["kind"] != type(slow).__name__:
        raise ValueError("checkpoint slow protocol differs from the live object")
    slow.z[:] = payload["z"]
    slow.m[:] = payload["m"]
    slow._I_I_last = np.array(payload["I_I_last"], copy=True)
    slow._step_index = int(payload["step_index"])


def restore_external_drive(state, drive):
    payload = state["external_drive"]
    if payload is None or drive is None:
        if (payload is None) != (drive is None):
            raise ValueError("checkpoint drive payload and drive object disagree")
        return
    drive._state = np.array(payload["field_state"], copy=True)
    drive._cached = np.array(payload["cached"], copy=True)
    drive._next_step = int(payload["next_step"])
    drive._last_step = int(payload["last_step"])
    drive._rng.bit_generator.state = payload["rng_state"]
```

`save` flattens the nested dicts into a single `np.savez` with `slow__z`, `external_drive__field_state`-style keys plus a `meta` JSON string holding the scalars and both RNG states; it writes to a temporary file in the same directory and `os.replace`s it, then returns the file's sha256. `load` inverts that flattening. `digest` serialises every array with `arr.tobytes()` in sorted-key order together with `json.dumps(scalars, sort_keys=True)` and returns the sha256.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_snn_checkpoint.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/snn_engine/checkpoint.py tests/test_snn_checkpoint.py
git commit -m "topic4 zm-itx: add the LIF checkpoint state module"
```

---

### Task 3: Engine — `post_runaway_record_ms`

**Files:**
- Modify: `src/snn_engine/kick_probe.py:108-120` (signature), `:281-289` (early-stop state), `:444-451` (break block), `:506-523` (truncation and result)
- Test: `tests/test_kick_probe_zm_itx_parity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `simulate_kick(..., post_runaway_record_ms=0.0)`; `res["runaway_early_stop_ms"]` keeps its existing meaning (detection time, `None` if never detected) and `res["post_runaway_recorded_ms"]` is new (`0.0` when the feature is off).

- [ ] **Step 1: Write the failing test**

```python
"""Off-by-default engine additions must not move the default path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from kick_probe import simulate_kick  # noqa: E402
from model import build_network  # noqa: E402
from params import Params  # noqa: E402


def _tiny(T=400.0, seed=11):
    p = Params(g=3.6, L=1.0, density=4000.0, T=T, dt=0.1, nu_ext_ratio=0.9, seed=seed)
    net = build_network(p, verbose=False)
    return p, net


def _run(p, net, **kwargs):
    net["rng"] = np.random.default_rng(p.seed)
    return simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, **kwargs)


def test_post_record_zero_is_byte_identical():
    p, net = _tiny()
    a = _run(p, net, early_stop_runaway=True)
    b = _run(p, net, early_stop_runaway=True, post_runaway_record_ms=0.0)
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert a["runaway_early_stop_ms"] == b["runaway_early_stop_ms"]
    assert b["post_runaway_recorded_ms"] == 0.0


def test_post_record_extends_only_the_tail():
    p, net = _tiny()
    a = _run(p, net, early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=10.0)
    b = _run(p, net, early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=10.0,
             post_runaway_record_ms=20.0)
    assert a["runaway_early_stop_ms"] == b["runaway_early_stop_ms"]
    n = len(a["rate_E"])
    assert len(b["rate_E"]) == min(n + 200, int(round(p.T / p.dt)))
    assert np.array_equal(a["rate_E"], b["rate_E"][:n])
    assert b["post_runaway_recorded_ms"] == (len(b["rate_E"]) - n) * p.dt


def test_post_record_never_exceeds_the_duration_cap():
    p, net = _tiny(T=60.0)
    b = _run(p, net, early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=10.0,
             post_runaway_record_ms=100000.0)
    assert len(b["rate_E"]) <= int(round(p.T / p.dt))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_kick_probe_zm_itx_parity.py -v`
Expected: `TypeError: simulate_kick() got an unexpected keyword argument 'post_runaway_record_ms'`.

- [ ] **Step 3: Implement**

Add `post_runaway_record_ms=0.0` to the signature. Near the early-stop state initialisation, add:

```python
    _post_steps = int(round(float(post_runaway_record_ms) / dt))
    if _post_steps < 0:
        raise ValueError("post_runaway_record_ms must be non-negative")
    _detect_t = None
```

Replace the break block so detection records the time and only stops once the tail is recorded:

```python
        if early_stop_runaway:
            _es_ema += _es_alpha * (rate_E[t] / NE / dt * 1e3 - _es_ema)
            _es_run = _es_run + 1 if _es_ema >= es_thresh_hz else 0
            if _detect_t is None and _es_run >= _es_dur:
                _detect_t = t + 1
            if _detect_t is not None and t + 1 >= _detect_t + _post_steps:
                _stop_t = t + 1
                if dump_ee_std_trace:
                    _rec_xdep(t)
                break
```

With `_post_steps == 0` this fires on the same step as before, so the default path is untouched. `runaway_early_stop_ms` keeps using `_detect_t`:

```python
    _nsteps_full = int(round(p.T / dt))
    res["runaway_early_stop_ms"] = (
        None if _detect_t is None else round(_detect_t * dt, 1))
    res["post_runaway_recorded_ms"] = (
        0.0 if _detect_t is None else round((nsteps - _detect_t) * dt, 1))
```

Note the pre-existing definition returned `None` when `_stop_t >= nsteps_full`; with a post-record tail that comparison would misreport a late detection, so it is replaced by the explicit `_detect_t` above. Keep `_stop_t` for array truncation only.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_kick_probe_zm_itx_parity.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the pre-existing engine tests for regressions**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/ -k "snn or kick or mz or engine" -v`
Expected: all pass, no new failures.

- [ ] **Step 6: Commit**

```bash
git add src/snn_engine/kick_probe.py tests/test_kick_probe_zm_itx_parity.py
git commit -m "topic4 zm-itx: record a bounded tail after runaway detection"
```

---

### Task 4: Engine — checkpoint emission, resume, and absolute time

**Files:**
- Modify: `src/snn_engine/kick_probe.py` (signature, state init, loop head, result)
- Test: `tests/test_kick_probe_zm_itx_parity.py` (append)

**Interfaces:**
- Consumes: `src.snn_engine.checkpoint.capture` from Task 2.
- Produces: `simulate_kick(..., checkpoint_steps=None, checkpoint_sink=None, resume_state=None, time_offset_ms=0.0)`. `checkpoint_sink` is `Callable[[int, dict], None]` invoked as `sink(absolute_step, state)`. `res["times"]` is absolute: `time_offset_ms + arange(nsteps) * dt`.

- [ ] **Step 1: Write the failing tests**

```python
def _tail_params(seed, T=200.0):
    return Params(g=3.6, L=1.0, density=4000.0, T=T, dt=0.1,
                  nu_ext_ratio=0.9, seed=seed)


def _full_with_checkpoint(p, net, steps, **kwargs):
    """Run the WHOLE trajectory while capturing. The checkpoint step must be
    inside range(nsteps): a head run shorter than the checkpoint step never
    reaches it and silently captures nothing."""
    captured = {}
    net["rng"] = np.random.default_rng(p.seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9,
                        checkpoint_steps=steps,
                        checkpoint_sink=lambda step, state: captured.setdefault(step, state),
                        **kwargs)
    return res, captured


def test_checkpoint_off_is_byte_identical():
    p, net = _tiny()
    a = _run(p, net)
    b = _run(p, net, checkpoint_steps=None, resume_state=None, time_offset_ms=0.0)
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])


def test_capturing_does_not_perturb_the_trajectory():
    p, net = _tiny(T=400.0)
    plain = _run(p, net)
    withck, captured = _full_with_checkpoint(p, net, [2000])
    assert 2000 in captured
    assert np.array_equal(plain["rate_E"], withck["rate_E"])
    assert np.array_equal(plain["E_spk_bool"], withck["E_spk_bool"])


def test_checkpoint_and_resume_are_byte_identical():
    p, net = _tiny(T=400.0)
    full, captured = _full_with_checkpoint(p, net, [2000])
    assert captured[2000]["absolute_time_ms"] == 200.0

    tail = simulate_kick(_tail_params(p.seed), net, KICK_BOOST=0.0, t_kick=1e9,
                         resume_state=captured[2000], time_offset_ms=200.0)
    assert np.array_equal(tail["rate_E"], full["rate_E"][2000:])
    assert np.array_equal(tail["E_spk_bool"], full["E_spk_bool"][2000:])
    assert np.isclose(tail["times"][0], 200.0)


def test_resume_rejects_a_mismatched_clock():
    p, net = _tiny(T=400.0)
    _, captured = _full_with_checkpoint(p, net, [2000])
    try:
        simulate_kick(_tail_params(p.seed), net, KICK_BOOST=0.0, t_kick=1e9,
                      resume_state=captured[2000], time_offset_ms=0.0)
    except ValueError as exc:
        assert "clock" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on a mismatched resume clock")
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_kick_probe_zm_itx_parity.py -k checkpoint -v`
Expected: `TypeError: unexpected keyword argument 'checkpoint_steps'`.

- [ ] **Step 3: Implement**

Add the four parameters. After the existing state initialisation (`V`, `ref`, `s_E`, ... `ring_sI`) and after `ras_keep` is drawn, insert the resume block — placing it after `ras_keep` means the setup's RNG draws are overwritten by the restored bit-generator state, which is what makes the continuation exact:

```python
    _resume_step = 0
    if resume_state is not None:
        from checkpoint import restore_external_drive, restore_slow
        V[:] = resume_state["V"]; ref[:] = resume_state["ref"]
        s_E[:] = resume_state["s_E"]; I_E[:] = resume_state["I_E"]
        s_I[:] = resume_state["s_I"]; I_I[:] = resume_state["I_I"]
        if resume_state["ring_sE"].shape != ring_sE.shape:
            raise ValueError("checkpoint delay-ring shape differs from this network")
        ring_sE[:] = resume_state["ring_sE"]; ring_sI[:] = resume_state["ring_sI"]
        xi = float(resume_state["xi"])
        rng.bit_generator.state = resume_state["rng_state"]
        ras_keep = np.array(resume_state["ras_keep"], copy=True)
        ras_mask = np.zeros(N, dtype=bool); ras_mask[ras_keep] = True
        _es_ema = float(resume_state["es_ema"]); _es_run = int(resume_state["es_run"])
        if bool(resume_state["track_rec"]) != bool(track_rec):
            raise ValueError("checkpoint track_rec differs from this run")
        if track_rec:
            s_E_rec[:] = resume_state["s_E_rec"]; I_E_rec[:] = resume_state["I_E_rec"]
        restore_slow(resume_state, slow)
        restore_external_drive(resume_state, external_e_rate_drive)
        _resume_step = int(resume_state["step"])
        if not np.isclose(float(resume_state["absolute_time_ms"]),
                          float(time_offset_ms), atol=1e-9):
            raise ValueError(
                "time_offset_ms does not continue the checkpoint clock")
```

Change the time expression at the top of the loop from `tm = t * dt` to:

```python
        tm = time_offset_ms + t * dt
```

Emit checkpoints at the top of the loop, before any RNG draw:

```python
    _ckpt_steps = set() if checkpoint_steps is None else {int(v) for v in checkpoint_steps}
    ...
    for t in range(nsteps):
        _abs_step = _resume_step + t
        if _abs_step in _ckpt_steps:
            if checkpoint_sink is None:
                raise ValueError("checkpoint_steps requires a checkpoint_sink")
            from checkpoint import capture
            checkpoint_sink(_abs_step, capture(
                step=_abs_step, absolute_time_ms=time_offset_ms + t * dt,
                V=V, ref=ref, s_E=s_E, I_E=I_E, s_I=s_I, I_I=I_I,
                ring_sE=ring_sE, ring_sI=ring_sI, xi=xi, rng=rng,
                ras_keep=ras_keep, es_ema=_es_ema, es_run=_es_run,
                track_rec=track_rec,
                s_E_rec=(s_E_rec if track_rec else None),
                I_E_rec=(I_E_rec if track_rec else None),
                slow=slow, external_drive=external_e_rate_drive))
        tm = time_offset_ms + t * dt
```

The keyword is `external_drive=` — that is `capture`'s parameter name from Task 2; the engine's local variable happens to be called `external_e_rate_drive`. Set `res["times"] = time_offset_ms + np.arange(nsteps) * dt` and report `runaway_early_stop_ms` in absolute time as `time_offset_ms + _detect_t * dt`.

**Forced-spike time must be resolved on the same clock.** The pre-existing line

```python
        forced_spike_step = int(round(float(forced_spike_ms) / dt))
```

becomes

```python
        forced_spike_step = int(round((float(forced_spike_ms) - time_offset_ms) / dt))
```

and the `abs(forced_spike_step * dt - forced_spike_ms) > 1e-9` grid check becomes
`abs(time_offset_ms + forced_spike_step * dt - float(forced_spike_ms)) > 1e-9`.
With `time_offset_ms == 0.0` both are literally the old expressions, so parity holds.
Without this change, a probe resumed at 2000 ms and injected at 2000 ms would compute
`forced_spike_step = 20000`, fail the `< nsteps` bound of the 200 ms continuation, and
raise "forced spike time lies outside the simulation".

The ring buffer is indexed by `slot = t % M`. On resume this must continue the original phase, so replace `slot = t % M` with `slot = (_resume_step + t) % M`, and likewise `((t + a_dly[idx]) % M)` with `((_resume_step + t + a_dly[idx]) % M)`. With `_resume_step == 0` these are literally the old expressions, preserving parity.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_kick_probe_zm_itx_parity.py -v`
Expected: 7 passed (3 from Task 3 plus the 4 added here).

- [ ] **Step 5: Run the regression sweep**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/ -k "snn or kick or mz or engine" -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/snn_engine/kick_probe.py tests/test_kick_probe_zm_itx_parity.py
git commit -m "topic4 zm-itx: emit and resume from simulator checkpoints on an absolute clock"
```

---

### Task 5: Gate C — perturbed reload equals a full rerun

**Files:**
- Test: `tests/test_kick_probe_zm_itx_parity.py` (append)

**Interfaces:**
- Consumes: `simulate_kick` with `resume_state`, `forced_spike_mask`, `forced_spike_ms` (pre-existing).
- Produces: no new code — this task's deliverable is the gate itself, plus any engine fix it forces.

- [ ] **Step 1: Write the gate**

```python
def _packet(net):
    NE, NI = net["NE"], net["NI"]
    mask = np.zeros(NE + NI, bool)
    mask[np.arange(0, NE, max(1, NE // 40))] = True
    return mask


def test_perturbed_resume_equals_full_rerun_with_the_same_packet():
    p, net = _tiny(T=400.0)
    packet = _packet(net)

    net["rng"] = np.random.default_rng(p.seed)
    full = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9,
                         forced_spike_mask=packet, forced_spike_ms=250.0)

    # Capture during an UNPERTURBED full run. The packet fires at step 2500,
    # after the step-2000 checkpoint, so the two trajectories are identical up
    # to the capture point -- assert that rather than assume it.
    sham_full, captured = _full_with_checkpoint(p, net, [2000])
    assert np.array_equal(sham_full["E_spk_bool"][:2500], full["E_spk_bool"][:2500])

    probe = simulate_kick(_tail_params(p.seed), net, KICK_BOOST=0.0, t_kick=1e9,
                          resume_state=captured[2000], time_offset_ms=200.0,
                          forced_spike_mask=packet, forced_spike_ms=250.0)
    assert np.array_equal(probe["E_spk_bool"], full["E_spk_bool"][2000:])
    assert np.array_equal(probe["rate_E"], full["rate_E"][2000:])
    assert probe["forced_spike_collision_count"] == full["forced_spike_collision_count"]


def test_sham_and_probe_from_one_checkpoint_diverge_only_after_injection():
    p, net = _tiny(T=400.0)
    packet = _packet(net)
    _, captured = _full_with_checkpoint(p, net, [2000])

    common = dict(KICK_BOOST=0.0, t_kick=1e9, time_offset_ms=200.0)
    sham = simulate_kick(_tail_params(p.seed), net,
                         resume_state=captured[2000], **common)
    probe = simulate_kick(_tail_params(p.seed), net,
                          resume_state=captured[2000],
                          forced_spike_mask=packet, forced_spike_ms=250.0, **common)
    inject = int(round((250.0 - 200.0) / 0.1))
    assert np.array_equal(sham["E_spk_bool"][:inject], probe["E_spk_bool"][:inject])
    assert not np.array_equal(sham["E_spk_bool"], probe["E_spk_bool"])
```

Both `simulate_kick` calls consume the same `captured[2000]` object, so `restore_slow` / `restore_external_drive` must copy out of the checkpoint rather than alias it — if the first call mutated the stored arrays in place, the second would resume from a different state and `test_sham_and_probe...` would show divergence before `inject`. Task 2's `test_capture_copies_and_does_not_alias` covers the capture side; these two calls cover the restore side.

- [ ] **Step 2: Run and fix whatever it exposes**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_kick_probe_zm_itx_parity.py -k "perturbed_resume or diverge" -v`

If the first assertion fails, the missing state is almost certainly one of: the delay-ring slot phase (Task 4 Step 3), `ras_keep`, or a slow/drive field not listed in `REQUIRED_KEYS`. Add the missing field to `capture`/`restore`, add an assertion for it in `tests/test_snn_checkpoint.py::test_capture_has_every_required_key`, and re-run both files. Do not weaken the assertion from `array_equal` to `allclose`.

Expected when done: 2 passed.

- [ ] **Step 3: Run the full engine suite**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/ -k "snn or kick or mz or engine or checkpoint" -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_kick_probe_zm_itx_parity.py tests/test_snn_checkpoint.py src/snn_engine/
git commit -m "topic4 zm-itx: gate perturbed checkpoint resume against a full rerun"
```

---

### Task 6: `MZSlowVars` per-neuron slow-current accumulator

**Files:**
- Modify: `src/snn_engine/mz_slow_vars.py:67-100`
- Test: `tests/test_mz_slow_vars.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  MZSlowVars.enable_field_accumulator(n_steps: int) -> None
  MZSlowVars.field_accumulator_result() -> dict
      {"n_steps": int, "disinhibition_D": (NE,) float64,
       "adaptation_A": (NE,) float64, "net_slow_current": (NE,) float64}
  ```
  `D_i = mean_t[(1 - z_i(t)) * I_I,i(t)]`, `A_i = mean_t[eta_m * m_i(t)]`, both averaged over the first `n_steps` calls to `apply_currents` after enabling. Off by default.

- [ ] **Step 1: Write the failing test**

```python
def test_field_accumulator_is_off_by_default_and_exact_when_on():
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    assert slow.field_accumulator_result() is None

    slow.enable_field_accumulator(3)
    excitatory = np.zeros(10)
    d_expected = np.zeros(8)
    a_expected = np.zeros(8)
    for k in range(3):
        inhibitory = np.full(10, float(k) + 1.0)
        slow.z[:8] = 0.25 * (k + 1)
        slow.m[:8] = float(k)
        d_expected += (1.0 - slow.z[:8]) * inhibitory[:8]
        a_expected += 0.5 * slow.m[:8]
        slow.apply_currents(excitatory, inhibitory)
    out = slow.field_accumulator_result()
    assert out["n_steps"] == 3
    assert np.allclose(out["disinhibition_D"], d_expected / 3.0, rtol=0, atol=0)
    assert np.allclose(out["adaptation_A"], a_expected / 3.0, rtol=0, atol=0)
    assert np.allclose(out["net_slow_current"],
                       out["disinhibition_D"] - out["adaptation_A"])


def test_field_accumulator_stops_after_n_steps():
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    slow.enable_field_accumulator(2)
    for _ in range(5):
        slow.apply_currents(np.zeros(10), np.ones(10))
    assert slow.field_accumulator_result()["n_steps"] == 2


def test_field_accumulator_does_not_change_returned_current():
    a = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    b = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    b.enable_field_accumulator(4)
    excitatory, inhibitory = np.arange(10, dtype=float), np.arange(10, dtype=float) * 0.3
    assert np.array_equal(a.apply_currents(excitatory, inhibitory),
                          b.apply_currents(excitatory, inhibitory))
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_mz_slow_vars.py -k accumulator -v`
Expected: `AttributeError: 'MZSlowVars' object has no attribute 'field_accumulator_result'`.

- [ ] **Step 3: Implement**

In `__init__`, add `self._acc_n = 0`, `self._acc_seen = 0`, `self._acc_D = None`, `self._acc_A = None`. Then:

```python
    def enable_field_accumulator(self, n_steps):
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("field accumulator needs at least one step")
        self._acc_n = n_steps
        self._acc_seen = 0
        self._acc_D = np.zeros(self.NE, dtype=float)
        self._acc_A = np.zeros(self.NE, dtype=float)

    def field_accumulator_result(self):
        if self._acc_D is None or self._acc_seen == 0:
            return None
        scale = 1.0 / float(self._acc_seen)
        disinhibition = self._acc_D * scale
        adaptation = self._acc_A * scale
        return {
            "n_steps": int(self._acc_seen),
            "disinhibition_D": disinhibition,
            "adaptation_A": adaptation,
            "net_slow_current": disinhibition - adaptation,
        }
```

At the **top** of `apply_currents`, before `self._I_I_last = I_I`, add:

```python
        if self._acc_D is not None and self._acc_seen < self._acc_n:
            self._acc_D += (1.0 - self.z[:self.NE]) * np.asarray(I_I)[:self.NE]
            self._acc_A += self.cfg.eta_m * self.m[:self.NE]
            self._acc_seen += 1
```

This reads exactly the `z` and `m` the membrane equation consumes on this step, so the product average is over the physically-applied pairing. With the accumulator disabled `self._acc_D is None`, so no float is touched and byte parity holds.

Also extend `src/snn_engine/checkpoint.py::capture` to store `acc_n`, `acc_seen`, `acc_D`, `acc_A` under `state["slow"]`, and `restore_slow` to put them back, and add them to the Task 2 key assertion.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_mz_slow_vars.py tests/test_snn_checkpoint.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/snn_engine/mz_slow_vars.py src/snn_engine/checkpoint.py \
        tests/test_mz_slow_vars.py tests/test_snn_checkpoint.py
git commit -m "topic4 zm-itx: accumulate per-neuron slow-current products inside the Z/M protocol"
```

---

### Task 7: Covariant D4 substrate transform

**Files:**
- Create: `src/topic4_zm_d4.py`
- Modify: `src/topic4_zm_ictal_transition.py` (honour `field_transform`)
- Test: `tests/test_zm_d4.py`

**Interfaces:**
- Consumes: `build_substrate(..., field_transform=...)` from Task 1.
- Produces:
  ```python
  D4_ELEMENTS = ("r90", "r180", "r270", "mx", "my", "md1", "md2")

  def d4_matrix(element: str) -> np.ndarray                    # (2, 2)
  def inverse_query_positions(positions, element, *, L) -> np.ndarray
  def transform_flow_coefficients(coefficients, element) -> np.ndarray  # (2, 6)
  def transform_report(element, coefficients) -> dict
  ```
  Field convention: `h'(x) = h(R^{-1}(x - c) + c)` with `c = (L/2, L/2)`, so `inverse_query_positions` returns `R^{-1}(x - c) + c`. Flow convention: the last two coefficients of each pathway row are rotated by the **same** `R`, i.e. `c' = R @ c`, which makes the transformed substrate an exact isometric copy.

- [ ] **Step 1: Write the failing tests**

```python
"""Covariant D4 transform must be an exact isometry of the substrate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_d4 import (  # noqa: E402
    D4_ELEMENTS, d4_matrix, inverse_query_positions, transform_flow_coefficients)
from src.topic4_local_connectivity import local_pair_features  # noqa: E402


def test_elements_are_orthogonal_with_unit_determinant_magnitude():
    for element in D4_ELEMENTS:
        R = d4_matrix(element)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
        assert np.isclose(abs(np.linalg.det(R)), 1.0)


def test_r90_has_order_four():
    R = d4_matrix("r90")
    assert np.allclose(np.linalg.matrix_power(R, 4), np.eye(2), atol=1e-12)


def test_inverse_query_round_trips_about_the_sheet_centre():
    rng = np.random.default_rng(0)
    xy = rng.random((50, 2)) * 20.0
    for element in D4_ELEMENTS:
        once = inverse_query_positions(xy, element, L=20.0)
        R = d4_matrix(element)
        back = (R @ (once - 10.0).T).T + 10.0
        assert np.allclose(back, xy, atol=1e-10)


def test_flow_coefficients_only_swap_and_negate_so_bounds_are_preserved():
    coefficients = np.array([[0.5, -0.5, 0.15, -0.15, 0.15, -0.15],
                             [-0.4, 0.3, 0.1, 0.05, -0.12, 0.09]])
    bounds = np.array([0.5, 0.5, 0.15, 0.15, 0.15, 0.15])
    for element in D4_ELEMENTS:
        out = transform_flow_coefficients(coefficients, element)
        assert out.shape == coefficients.shape
        assert np.all(np.abs(out) <= bounds + 1e-12)
        assert np.array_equal(np.abs(np.sort(out[:, 4:], axis=1)),
                              np.abs(np.sort(coefficients[:, 4:], axis=1)))
        assert np.array_equal(out[:, :4], coefficients[:, :4])


def test_covariant_transform_reproduces_the_feature_contribution():
    """The whole point: rotating the field and the flow together leaves the
    edge contribution invariant when the edge is rotated with them."""
    rng = np.random.default_rng(3)
    target = rng.random((200, 2)) * 20.0
    source = rng.random((200, 2)) * 20.0
    h_t, h_s = rng.random(200), rng.random(200)
    coefficients = np.array([[0.3, -0.2, 0.1, -0.05, 0.12, -0.07],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    base = local_pair_features(target, source, h_t, h_s, length_scale=0.38)
    contribution = base @ coefficients[0]

    for element in D4_ELEMENTS:
        R = d4_matrix(element)
        rot_t = (R @ (target - 10.0).T).T + 10.0
        rot_s = (R @ (source - 10.0).T).T + 10.0
        rot_c = transform_flow_coefficients(coefficients, element)
        rotated = local_pair_features(rot_t, rot_s, h_t, h_s, length_scale=0.38)
        assert np.allclose(rotated @ rot_c[0], contribution, atol=1e-10)
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_d4.py -v`
Expected: `ModuleNotFoundError: No module named 'src.topic4_zm_d4'`.

- [ ] **Step 3: Implement**

```python
"""Covariant square-symmetry transforms of the data-driven substrate.

Rotating the node field alone is NOT an isometry of this substrate: the local
connectivity mapper's last two features are signed and linear in the
source-minus-target displacement (src/topic4_local_connectivity.py:50-62), so a
field-only rotation reverses the correspondence between field structure and the
flow it drives. Rotating the two flow coefficients by the same matrix restores
it, and because the group elements only swap and negate components, the frozen
coefficient bounds survive element-wise with no re-clipping.
"""
from __future__ import annotations

import numpy as np

D4_ELEMENTS = ("r90", "r180", "r270", "mx", "my", "md1", "md2")

_MATRICES = {
    "r90": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "r180": np.array([[-1.0, 0.0], [0.0, -1.0]]),
    "r270": np.array([[0.0, 1.0], [-1.0, 0.0]]),
    "mx": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "my": np.array([[-1.0, 0.0], [0.0, 1.0]]),
    "md1": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "md2": np.array([[0.0, -1.0], [-1.0, 0.0]]),
}


def d4_matrix(element):
    if element not in _MATRICES:
        raise ValueError(f"unknown D4 element {element!r}")
    return _MATRICES[element].copy()


def inverse_query_positions(positions, element, *, L):
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    centre = float(L) / 2.0
    inverse = np.linalg.inv(d4_matrix(element))
    return (inverse @ (positions - centre).T).T + centre


def transform_flow_coefficients(coefficients, element):
    coefficients = np.asarray(coefficients, float)
    if coefficients.ndim != 2 or coefficients.shape[1] != 6:
        raise ValueError("coefficients must have shape (pathways, 6)")
    R = d4_matrix(element)
    out = coefficients.copy()
    out[:, 4:] = (R @ coefficients[:, 4:].T).T
    return out
```

`transform_report(element, coefficients)` returns the element name, the matrix, whether the element preserves the undirected patient axis (`|R @ axis_unit · axis_unit| == 1`), whether it preserves the directed axis (`R @ axis_unit · axis_unit == 1`), and the pre/post coefficient rows, for the manifest.

Then extend `build_substrate` so that when `field_transform` is a D4 element name, the field queries use `inverse_query_positions(positions, element, L=engine["L"])` for both E and I, the `np.array_equal(h_e, node["h"])` assertion is skipped, `coefficients_eff = transform_flow_coefficients(coefficients, element)`, `vtheta` is rebuilt from the transformed `h_e`, and the montage, contacts, network and axis are left untouched.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_d4.py -v`
Expected: 5 passed.

- [ ] **Step 5: Add and run the substrate-level isometry check**

Append to `tests/test_zm_ictal_transition_substrate.py`:

```python
@pytest.mark.slow
@pytest.mark.integration
def test_transformed_substrate_preserves_field_mass_and_bounds(tmp_path):
    config = load_round_config(CONFIG)
    plain = build_substrate(config, "joint_04_control", 1811, cache_dir=str(tmp_path))
    rotated = build_substrate(config, "joint_04_control", 1811, cache_dir=str(tmp_path),
                              field_transform="r180")
    assert np.isclose(rotated.h_e.sum(), 1129.0, atol=1e-8)
    assert not np.allclose(rotated.h_e, plain.h_e)
    assert np.array_equal(rotated.contact_xy, plain.contact_xy)
    assert np.allclose(rotated.axis_unit, plain.axis_unit)
    bounds = np.array([0.5, 0.5, 0.15, 0.15, 0.15, 0.15])
    assert np.all(np.abs(rotated.edge_audit["coefficients"]["E_to_E"]) <= bounds + 1e-12)
```

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_ictal_transition_substrate.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/topic4_zm_d4.py src/topic4_zm_ictal_transition.py \
        tests/test_zm_d4.py tests/test_zm_ictal_transition_substrate.py
git commit -m "topic4 zm-itx: covariant D4 substrate transform for the spatial null"
```

---

### Task 8: State characterization

**Files:**
- Create: `src/topic4_zm_state_characterization.py`
- Test: `tests/test_zm_state_characterization.py`

**Interfaces:**
- Consumes: `res["rate_E"]`, `res["E_spk_bool"]`, contact envelopes from the worker.
- Produces:
  ```python
  def characterize_state(rate_E_hz, *, dt_ms, window_ms, silence_threshold_hz,
                         zero_window_ms=20.0) -> dict
  def interictal_reference(rate_E_hz, *, dt_ms, window_ms) -> dict
  def band_proxy(rate_E_hz, *, dt_ms, band_hz=(30.0, 80.0)) -> dict
  def spatial_recruitment(E_spk_bool, *, positions_e, grid_mm=1.0) -> dict
  def contact_recruitment(envelope, *, envelope_dt_ms, window_ms, floor) -> dict
  ```
  `characterize_state` returns `active_durations_ms`, `silent_durations_ms`,
  `burst_interval_ms`, `reignition_rate_hz`, `zero_spike_window_fraction`,
  `peak_rate_hz`, `median_rate_hz`, `mean_rate_hz`.

- [ ] **Step 1: Write the failing test**

```python
"""State characterization must recover a synthetic burst train exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_state_characterization import (  # noqa: E402
    band_proxy, characterize_state)


def _burst_train(dt=0.1, period_ms=86.0, active_ms=22.0, total_ms=500.0,
                 high=300.0, low=0.0):
    n = int(round(total_ms / dt))
    t = np.arange(n) * dt
    phase = np.mod(t, period_ms)
    return np.where(phase < active_ms, high, low)


def test_recovers_burst_geometry():
    rate = _burst_train()
    out = characterize_state(rate, dt_ms=0.1, window_ms=(0.0, 500.0),
                             silence_threshold_hz=1.0)
    assert np.isclose(np.median(out["active_durations_ms"]), 22.0, atol=0.2)
    assert np.isclose(np.median(out["silent_durations_ms"]), 64.0, atol=0.2)
    assert np.isclose(out["burst_interval_ms"], 86.0, atol=0.5)
    assert np.isclose(out["reignition_rate_hz"], 1000.0 / 86.0, atol=0.2)
    assert np.isclose(out["peak_rate_hz"], 300.0)


def test_zero_spike_window_fraction_matches_the_silent_duty_cycle():
    rate = _burst_train()
    out = characterize_state(rate, dt_ms=0.1, window_ms=(0.0, 500.0),
                             silence_threshold_hz=1.0, zero_window_ms=20.0)
    assert 0.3 <= out["zero_spike_window_fraction"] <= 0.55


def test_band_proxy_finds_a_planted_frequency():
    dt, total = 0.1, 500.0
    t = np.arange(int(total / dt)) * dt
    rate = 50.0 + 20.0 * np.sin(2 * np.pi * 45.0 * t / 1000.0)
    out = band_proxy(rate, dt_ms=dt, band_hz=(30.0, 80.0))
    assert np.isclose(out["peak_frequency_hz"], 45.0, atol=2.5)
    assert out["frequency_resolution_hz"] <= 2.5
    assert out["n_cycles_at_band_low"] >= 10.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_state_characterization.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`characterize_state` thresholds the rate at `silence_threshold_hz`, runs a run-length encoding over the boolean, and reports the two duration distributions; `burst_interval_ms` is the median onset-to-onset spacing of the active runs; `reignition_rate_hz` is `1000 / burst_interval_ms`; `zero_spike_window_fraction` is the fraction of non-overlapping `zero_window_ms` windows whose summed spike count is zero. `interictal_reference` runs the same code over a length-matched pre-onset window and additionally returns the 95th percentile of the rate, which supplies `silence_threshold_hz` for the ictal window and the comparison band for the honest "is the instantaneous rate above or below the interictal band" statement. `band_proxy` applies a Hann window, takes `np.fft.rfft`, reports in-band power, `peak_frequency_hz`, total power ratio versus the length-matched interictal window, `frequency_resolution_hz = 1000 / window_ms`, and `n_cycles_at_band_low = band_hz[0] * window_ms / 1000`. `spatial_recruitment` bins first-spike positions on a `grid_mm` grid and returns the fraction of occupied bins and the recruited-area centroid. `contact_recruitment` counts contacts whose envelope exceeds `floor` inside the window.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_state_characterization.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_state_characterization.py tests/test_zm_state_characterization.py
git commit -m "topic4 zm-itx: characterize what the sustained high-activity state actually is"
```

---

### Task 9: Perturbation sites, packets, and response metrics

**Files:**
- Create: `src/topic4_zm_perturbation.py`
- Test: `tests/test_zm_perturbation.py`

**Interfaces:**
- Consumes: `Substrate` from Task 1.
- Produces:
  ```python
  def frozen_sites(substrate, config, *, kind) -> list[dict]
      # kind in {"grid", "representative"}; each dict has
      # {"site_id": str, "xy_mm": (2,) float, "kind": str}
  def select_packet(positions_e, site_xy, *, n_cells, radius_mm) -> np.ndarray  # (n_e,) bool
  def response_metrics(probe, sham, *, dt_ms, positions_e, packet_xy,
                       envelope_probe, envelope_sham, envelope_dt_ms,
                       inject_ms, split_ms, window_ms) -> dict
  def susceptibility_map(rows, *, sites) -> dict
  def hotspot_compactness(sites_xy, values, *, quantile, n_null, seed) -> dict
  ```
  `response_metrics` returns `excess_spikes_early`, `excess_spikes_late`,
  `susceptibility` (the 0–window_ms total, the canonical scalar), `r90_mm`,
  `contact_excess_energy`, `excess_per_neuron` (n_e float32).

- [ ] **Step 1: Write the failing test**

```python
"""Perturbation geometry and sham-subtracted metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_perturbation import (  # noqa: E402
    hotspot_compactness, response_metrics, select_packet)


def test_packet_is_the_nearest_cells_and_respects_the_radius():
    rng = np.random.default_rng(0)
    positions = rng.random((5000, 2)) * 20.0
    mask = select_packet(positions, np.array([10.0, 10.0]), n_cells=64,
                         radius_mm=1.0)
    assert mask.sum() == 64
    chosen = positions[mask]
    assert np.all(np.linalg.norm(chosen - 10.0, axis=1) <= 1.0)
    others = positions[~mask]
    inside = others[np.linalg.norm(others - 10.0, axis=1) <= 1.0]
    assert np.all(np.linalg.norm(chosen - 10.0, axis=1).max()
                  <= np.linalg.norm(inside - 10.0, axis=1).min() + 1e-12)


def test_packet_raises_when_the_disk_is_too_sparse():
    positions = np.array([[10.0, 10.0], [10.1, 10.0]])
    try:
        select_packet(positions, np.array([10.0, 10.0]), n_cells=64, radius_mm=1.0)
    except ValueError as exc:
        assert "insufficient" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_identical_probe_and_sham_give_exactly_zero_susceptibility():
    n_steps, n_e = 2000, 500
    spikes = np.zeros((n_steps, n_e), bool)
    spikes[::7, ::3] = True
    envelope = np.zeros((15, 200), np.float32)
    out = response_metrics({"E_spk_bool": spikes}, {"E_spk_bool": spikes},
                           dt_ms=0.1, positions_e=np.zeros((n_e, 2)),
                           packet_xy=np.zeros(2),
                           envelope_probe=envelope, envelope_sham=envelope,
                           envelope_dt_ms=2.0, inject_ms=0.0, split_ms=50.0,
                           window_ms=200.0)
    assert out["susceptibility"] == 0.0
    assert out["excess_spikes_early"] == 0.0
    assert out["excess_spikes_late"] == 0.0
    assert out["contact_excess_energy"] == 0.0


def test_susceptibility_is_the_sum_of_its_two_parts():
    rng = np.random.default_rng(2)
    n_steps, n_e = 2000, 300
    sham = rng.random((n_steps, n_e)) < 0.001
    probe = sham | (rng.random((n_steps, n_e)) < 0.002)
    envelope = np.zeros((15, 200), np.float32)
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           dt_ms=0.1, positions_e=rng.random((n_e, 2)) * 5.0,
                           packet_xy=np.array([2.5, 2.5]),
                           envelope_probe=envelope, envelope_sham=envelope,
                           envelope_dt_ms=2.0, inject_ms=0.0, split_ms=50.0,
                           window_ms=200.0)
    assert np.isclose(out["susceptibility"],
                      out["excess_spikes_early"] + out["excess_spikes_late"])
    assert 0.0 < out["r90_mm"] <= np.sqrt(2) * 5.0


def test_hotspot_compactness_detects_a_planted_cluster():
    xy = np.stack(np.meshgrid(np.linspace(3, 17, 7), np.linspace(3, 17, 7)),
                  axis=-1).reshape(-1, 2)
    values = np.zeros(len(xy))
    corner = np.linalg.norm(xy - np.array([4.0, 4.0]), axis=1)
    values[np.argsort(corner)[:10]] = 100.0
    out = hotspot_compactness(xy, values, quantile=0.8, n_null=500, seed=1)
    assert out["p_value"] < 0.01
    assert out["observed_mean_pairwise_mm"] < out["null_mean_pairwise_mm"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_perturbation.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`frozen_sites` builds the two frozen site sets from geometry only:

```python
def frozen_sites(substrate, config, *, kind):
    perturbation = config["perturbation"]
    if kind == "grid":
        lo, hi = perturbation["grid_extent_mm"]
        n = int(perturbation["grid_n"])
        axis = np.linspace(lo, hi, n)
        xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        return [{"site_id": f"g{i:02d}", "xy_mm": p, "kind": "grid"}
                for i, p in enumerate(xy)]
    if kind == "representative":
        src, snk = substrate.axis_source_xy, substrate.axis_sink_xy
        mid = 0.5 * (src + snk)
        normal = np.array([-substrate.axis_unit[1], substrate.axis_unit[0]])
        points = [src, snk, mid, mid + 4.0 * normal, mid - 4.0 * normal,
                  np.array([substrate.engine["L"] / 2.0, substrate.engine["L"] / 2.0])]
        names = ["source", "sink", "midpoint", "normal_plus", "normal_minus", "centre"]
        return [{"site_id": n_, "xy_mm": np.asarray(p, float), "kind": "representative"}
                for n_, p in zip(names, points)]
    raise ValueError(f"unknown site kind {kind!r}")
```

`select_packet` takes the `n_cells` nearest E neurons within `radius_mm`, raising `ValueError("insufficient E neurons within the packet radius")` when fewer than `n_cells` qualify. `response_metrics` computes `excess = probe["E_spk_bool"].sum(axis=0) - sham["E_spk_bool"].sum(axis=0)` restricted to the window after `inject_ms`, splits at `split_ms`, sums for the scalar, computes `r90_mm` as the radius about `packet_xy` containing 90 % of the positive excess, and integrates `clip(envelope_probe - envelope_sham, 0, None)` for `contact_excess_energy`. `hotspot_compactness` takes sites above the `quantile` of `values`, computes their mean pairwise distance, and compares against `n_null` random equal-size subsets of the same site set, returning a one-sided empirical p-value.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_perturbation.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_perturbation.py tests/test_zm_perturbation.py
git commit -m "topic4 zm-itx: frozen probe geometry and sham-subtracted response metrics"
```

---

### Task 10: Statistics

**Files:**
- Create: `src/topic4_zm_statistics.py`
- Test: `tests/test_zm_statistics.py`

**Interfaces:**
- Produces:
  ```python
  def paired_bootstrap(a, b, *, draws, seed) -> dict
      # {"status", "n", "mean_difference", "q05", "q50", "q95", "n_positive"}
  def restricted_ictal_free_time(onset_ms, *, cap_ms) -> dict
      # onset_ms may contain None/NaN for censored networks
  def paired_onset_difference(onset_a, onset_b) -> dict     # both-entered subset only
  def spatial_correlation(values, covariate, positions, *, draws, seed,
                          block_mm=2.0) -> dict
  ```

- [ ] **Step 1: Write the failing test**

```python
"""Pre-registered statistics under censoring and spatial autocorrelation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_statistics import (  # noqa: E402
    paired_bootstrap, paired_onset_difference, restricted_ictal_free_time,
    spatial_correlation)


def test_paired_bootstrap_is_paired_and_deterministic():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a + 0.5
    one = paired_bootstrap(a, b, draws=1000, seed=5)
    two = paired_bootstrap(a, b, draws=1000, seed=5)
    assert one == two
    assert np.isclose(one["mean_difference"], -0.5)
    assert one["q95"] < 0.0
    assert one["n"] == 4


def test_paired_bootstrap_rejects_unequal_lengths():
    try:
        paired_bootstrap(np.zeros(3), np.zeros(4), draws=10, seed=1)
    except ValueError as exc:
        assert "align" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_restricted_ictal_free_time_treats_none_as_censored_at_the_cap():
    out = restricted_ictal_free_time([5000.0, None, 15000.0], cap_ms=20000.0)
    assert np.isclose(out["restricted_mean_ms"], (5000.0 + 20000.0 + 15000.0) / 3.0)
    assert out["n_censored"] == 1
    assert out["n"] == 3
    assert np.isclose(out["entered_fraction"], 2.0 / 3.0)


def test_paired_onset_difference_uses_only_networks_where_both_entered():
    out = paired_onset_difference([1000.0, None, 3000.0], [2000.0, 4000.0, None])
    assert out["n"] == 1
    assert np.isclose(out["mean_difference_ms"], -1000.0)
    assert out["n_dropped"] == 2


def test_spatial_correlation_block_null_is_not_fooled_by_smooth_noise():
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(np.linspace(0, 20, 30), np.linspace(0, 20, 30)),
                    axis=-1).reshape(-1, 2)
    smooth = np.sin(grid[:, 0] / 3.0) + np.cos(grid[:, 1] / 3.0)
    unrelated = np.sin(grid[:, 1] / 3.1) + np.cos(grid[:, 0] / 2.9)
    out = spatial_correlation(smooth, unrelated, grid, draws=500, seed=1, block_mm=2.0)
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["n_null"] == 500
    same = spatial_correlation(smooth, smooth, grid, draws=500, seed=1, block_mm=2.0)
    assert np.isclose(same["spearman_r"], 1.0)
    assert same["p_value"] < 0.05
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_statistics.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`paired_bootstrap` asserts equal length, resamples **network indices** (not values independently), and reports quantiles of the mean paired difference `a - b`. `restricted_ictal_free_time` maps `None`/`NaN` to `cap_ms` and averages — this is the restricted mean ictal-free time and is the only latency number that may be reported across arms with censoring. `paired_onset_difference` keeps only indices where both entries are finite. `spatial_correlation` computes the Spearman correlation between `values` and `covariate` and builds the null by rigid **block circular shifts** of the covariate field on a `block_mm` grid, which preserves the covariate's spatial autocorrelation; a plain permutation null would be anticonservative here and must not be used.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_statistics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_statistics.py tests/test_zm_statistics.py
git commit -m "topic4 zm-itx: censored latency, paired bootstrap and spatially-aware correlation null"
```

---

### Task 11: Primary worker and Gate A

**Files:**
- Create: `scripts/run_topic4_zm_ictal_transition_worker.py`
- Create: `scripts/audit_topic4_zm_ictal_transition.py` (parity mode only for now)

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces: per run, `<output_root>/workers/<arm>_seed_<seed>[_null_<element>].json` and `.npz`, plus `<output_root>/checkpoints/<arm>_seed_<seed>_<label>.npz`. The npz keys mirror the rev10-R worker's (`contact_names`, `shaft_ids`, `contact_xy_mm`, `onsets`, `ranks`, `event_t_on_ms`, `event_t_off_ms`, `event_returned`, `active_fraction`, `active_fraction_bin_ms`, `contact_envelope`, `contact_envelope_dt_ms`, `positions_E`, `h`, `delta_vtheta`, `edge_coefficients`, `spatial_ou_*`, `mz_*`) and add `ee_out_gain`, `etoi_out_gain`, `state_characterization_*`, `first_spike_step` (n_e int32, the step index of each E neuron's first spike inside the 100 ms window ending at detection, `-1` if absent).

- [ ] **Step 1: Write the worker**

CLI: `--config --candidate-id --seed --expected-commit --zm-mode {z_plus_m,off} --field-transform {none,r90,...} --out-json --out-npz --cache-dir --checkpoint-dir`.

Order of operations, which must not change:

1. `verify_frozen_inputs(config)`.
2. Provenance: reuse `_runtime_provenance` from `scripts/run_topic4_rev10_r_edge_flow_worker.py` by import, extended with this round's module list — `src/topic4_zm_ictal_transition.py`, `src/topic4_zm_d4.py`, `src/topic4_zm_state_characterization.py`, `src/topic4_zm_perturbation.py`, `src/topic4_zm_statistics.py`, `src/snn_engine/checkpoint.py`, `src/snn_engine/kick_probe.py`, `src/snn_engine/mz_slow_vars.py`, `src/topic4_spatial_ou_drive.py`, `src/topic4_local_connectivity.py`, `src/topic4_continuous_field.py`, and this script. Raise if dirty or if it differs from `--expected-commit`.
3. `substrate = build_substrate(config, candidate_id, seed, cache_dir=..., field_transform=...)`.
4. `slow = make_slow(substrate, config["zm"] if zm_mode == "z_plus_m" else {"mode": "off"})`.
5. `drive = make_external_drive(substrate, frozen spatial OU block, seed)`.
6. `net["rng"] = np.random.default_rng(seed)`.
7. `simulate_kick(params, substrate.net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=substrate.vtheta, slow=slow, early_stop_runaway=True, es_thresh_hz=120.0, es_dur_ms=100.0, external_e_rate_drive=drive, post_runaway_record_ms=<config>, checkpoint_steps=[20000], checkpoint_sink=...)` — step 20000 is the 2000 ms baseline checkpoint.
8. The pre-ictal and sensitivity checkpoints cannot be scheduled in advance because they are defined relative to an onset that is not yet known. Handle this with a **two-pass** run:
   - **Pass 1** runs the whole trajectory with `checkpoint_steps=[20000]` only, and records `onset_step`.
   - **Pass 2** resumes from the baseline checkpoint with `time_offset_ms=2000.0`, `T = (onset_ms - 500.0) - 2000.0`, and `checkpoint_steps=[onset_step - 10000, onset_step - 5000]`, then stops. It does **not** re-run past `onset - 500 ms`, so it costs `(onset_ms - 2500) / onset_ms` of a pass-1 run rather than a second full run.
   - Assert `np.array_equal(pass2["rate_E"], pass1["rate_E"][20000:20000 + len(pass2["rate_E"])])`. By Gate B this is guaranteed; if it fails, the checkpoint is incomplete and the round stops.
   - Skip pass 2 entirely when `onset_ms < 2500` (no perturbation analysis for that network) and emit only the `onset - 500` checkpoint when `2500 <= onset_ms < 3500`.

   A ring-buffer alternative (keep the last N checkpoints in memory and pick after detection) was rejected: at ~130 MB per checkpoint a ring fine enough to hit `onset - 500` exactly would cost tens of GB per worker, and a coarser ring would make the pre-ictal lead time vary per network, which weakens the paired primary contrast.
9. Readout: `active_fraction` → `detect_events` at the frozen detector → per-event contact onsets and ranks via the rev10-R `_contact_onsets` helper (import it from that module — it is read-only reuse, not modification).
10. State characterization over the post-detection recording, with the length-matched interictal reference taken from the 500 ms ending at `onset - 1000 ms`.
11. Write json + npz atomically.

- [ ] **Step 2: Write the Gate A audit script and run it**

`scripts/audit_topic4_zm_ictal_transition.py --gate parity` runs the worker with `--candidate-id joint_04_control --seed 1561 --zm-mode off --field-transform none` and `post_runaway_record_ms=0`, then compares the produced npz against the archived one:

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate parity \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json \
  --expected-commit "$(git rev-parse HEAD)"
```

Compared bit-for-bit: `onsets`, `ranks`, `event_t_on_ms`, `event_t_off_ms`, `event_returned`, `active_fraction`, `contact_envelope`, `spatial_ou_time_ms`, `spatial_ou_mean_rate_per_ms`, `spatial_ou_argmax_x_mm`, `spatial_ou_argmax_y_mm`, `h`, `positions_E`, `contact_xy_mm`. Also assert `n_common_detector_events == 105` and `n_returned_events == 87` against the archived json.

Expected: `{"gate": "parity", "status": "PASS", "mismatched_keys": []}` written to `<output_root>/gate_a_parity.json`. Wall time ~32 min.

If any key mismatches, **stop the round**. Report the mismatching keys and their first differing index. Do not relax `array_equal` to `allclose`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic4_zm_ictal_transition_worker.py \
        scripts/audit_topic4_zm_ictal_transition.py \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/gate_a_parity.json
git commit -m "topic4 zm-itx: primary worker and the default-path parity gate"
```

---

### Task 12: Perturbation worker

**Files:**
- Create: `scripts/run_topic4_zm_perturbation_worker.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5, 6, 9; the checkpoints written by Task 11.
- Produces: `<output_root>/perturbation/<arm>_seed_<seed>[_null_<element>]_<label>.json` and `.npz`, where `label ∈ {baseline, pre_ictal, sensitivity}`. The npz holds `site_id` (U8), `site_xy_mm`, `susceptibility`, `excess_spikes_early`, `excess_spikes_late`, `r90_mm`, `contact_excess_energy`, `onset_advance_ms`, `onset_censored` (bool), and `excess_per_neuron` (n_sites × n_e float32) — plus `slow_field_D`, `slow_field_A`, `slow_field_net` (n_e float64) from the sham run's accumulator.

- [ ] **Step 1: Write the worker**

CLI: `--config --candidate-id --seed --checkpoint --label --sites {grid,representative} --dose-cells --measure-onset-advance --onset-cap-ms --expected-commit --out-json --out-npz`.

The worker loads the network and the checkpoint **once**, then for each site:

1. `restore` a fresh copy of the checkpoint (the loaded dict is deep-copied per branch so the two branches cannot alias).
2. Run the sham continuation for `window_ms` with the slow-current accumulator enabled for the first 1000 steps (100 ms). Cache it — the sham is identical for every site at this checkpoint, so it is computed **once** per (network, label) and reused, which halves the cost.
3. Run the probe continuation with `forced_spike_mask` from `select_packet` and `forced_spike_ms = checkpoint_time + 0.0` (injection at the first step of the continuation).
4. `response_metrics(...)`.
5. If `--measure-onset-advance`, run both branches again with `early_stop_runaway=True` and duration `onset_cap_ms - checkpoint_time`, recording `runaway_early_stop_ms` for each; `onset_advance_ms = sham_onset - probe_onset`, with `onset_censored=True` and `onset_advance_ms=nan` if either branch fails to transition inside the cap. The sham long run is also computed once per (network, label).

For the `baseline` label `onset_cap_ms = 20000.0` (right-censored at the frozen duration cap); for `pre_ictal` the cap is `sham_onset + 1500.0`.

Peak memory per continuation is bounded by `E_spk_bool` over the continuation only: 200 ms → 2000 × 32000 bytes = 64 MB; a baseline onset-advance continuation of 18 s → 5.8 GB. Set `--measure-onset-advance` runs to their own worker slot and cap concurrency accordingly in Task 13.

- [ ] **Step 2: Add the smoke test**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/run_topic4_zm_perturbation_worker.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json \
  --candidate-id joint_04_control --seed 1801 --label baseline \
  --checkpoint results/topic4_sef_hfo/data_driven_zm_ictal_transition/checkpoints/joint_04_control_seed_1801_baseline.npz \
  --sites representative --dose-cells 64 --expected-commit "$(git rev-parse HEAD)"
```

Expected: six rows, `susceptibility > 0` at every site, `excess_per_neuron` shape `(6, 32000)`, and `slow_field_net` finite everywhere. This depends on Task 14 having produced the canary checkpoints, so run it there.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic4_zm_perturbation_worker.py
git commit -m "topic4 zm-itx: paired sham/probe perturbation worker resuming from checkpoints"
```

---

### Task 13: Launcher, memory sentinel, monitor

**Files:**
- Create: `scripts/launch_topic4_zm_ictal_transition.py`
- Create: `scripts/freeze_topic4_zm_ictal_transition.py`

**Interfaces:**
- Produces: `<output_root>/{candidate_manifest.json, screen_memory_audit.json, controller.log, controller.status, DONE.json}` and `<output_root>/run_logs/*.log`.

- [ ] **Step 1: Write the freeze script and run it**

`freeze_topic4_zm_ictal_transition.py` writes `candidate_manifest.json` containing: the four arm ids, the seeds, the frozen Z/M block, the frozen spatial OU block, the checkpoint offsets, the frozen site definitions (resolved to explicit coordinates), the dose ladder, the D4 assignment with `transform_report` for each element, the endpoint tier table, the interictal-baseline gate, the git commit, the runtime module hashes, and the spec hash. It refuses to run if `<output_root>/workers` already contains any json.

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/freeze_topic4_zm_ictal_transition.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

- [ ] **Step 2: Write the launcher**

Behaviour, copied from the rev11 controller pattern in `results/.../frozen_substrate_confirmation/controller.log`:

1. Run one sentinel job under `/usr/bin/time -v`, record `peak_rss_kib` and `mem_available_kib_after_sentinel` into `screen_memory_audit.json`.
2. `max_workers = min(8, floor((available_gib - 32) / (peak_rss_gib * 1.2)))`, at least 1.
3. Launch each job as `systemd-run --user --unit topic4-zmitx-<phase>-<arm>-s<seed>[-<element>]-<commit8> --setenv OMP_NUM_THREADS=1 ... /usr/bin/nohup /usr/bin/time -v <python> <worker> ...`, redirecting to `run_logs/`.
4. Poll every 600 s: worker states, `MemAvailable`, `df` on the output filesystem, and the runtime module hashes. Append one JSON line per poll to `controller.log`; write the latest to `controller.status`.
5. Stop launching new jobs on OOM, a non-finite result, hash drift, or a checkpoint-replay mismatch. Never signal processes outside this round's unit prefix.
6. On completion write `DONE.json` and fire a desktop notification via `notify-send`.

- [ ] **Step 3: Commit**

```bash
git add scripts/launch_topic4_zm_ictal_transition.py \
        scripts/freeze_topic4_zm_ictal_transition.py \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/candidate_manifest.json
git commit -m "topic4 zm-itx: freeze the round manifest and add the systemd launcher"
```

---

### Task 14: Phase 1 — canary and the interictal-baseline gate

**Files:**
- Modify: `scripts/audit_topic4_zm_ictal_transition.py` (add `--gate interictal-baseline` and `--gate dose`)

- [ ] **Step 1: Run the three canary networks**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase canary \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Seeds 1801–1803, `joint_04_control`, Z/M on, 20 s, 500 ms post-detection recording, baseline checkpoint at 2000 ms plus the two onset-relative checkpoints from pass 2.

- [ ] **Step 2: Adjudicate the gate**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate interictal-baseline \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

The audit must evaluate **all three clauses for every network and report every failing clause**, not only the first:

```
onset_ms >= 2500
n_returned_events_before_onset >= 3
median 20 ms-EMA E rate over [1500, 2000] ms
    <= percentile_95 of the same statistic across the 48 Z/M-off reference runs
```

The reference percentile is computed from `active_fraction` in the 48 archived npz files, converted to the same 20 ms-EMA rate units, and cached into `<output_root>/zm_off_reference_baseline.json` with its own hash.

Expected: `interictal_baseline_gate.json` with per-network clause results and an overall verdict.

**If the gate fails on any network, stop.** Write the finding — the current Z/M work point lacks an interpretable interictal residence segment — into the report and ask before continuing. Do not move the baseline checkpoint earlier.

- [ ] **Step 3: Freeze the perturbation dose**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate dose \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Runs the perturbation worker at the **baseline** checkpoint only, on the 6 representative sites, for each packet size in `[16, 32, 64, 128, 256]`, across seeds 1801–1803. Selects the smallest size with median `susceptibility >= 200` and at most 1 of 6 sites triggering a detector-qualified event. Writes `dose_freeze.json` including every ladder rung's numbers, and patches `candidate_manifest.json` with `perturbation.frozen_dose_cells`.

This audit must not read any pre-ictal checkpoint or any patient-derived score; assert that in the script by refusing any `--label` other than `baseline`.

- [ ] **Step 4: Smoke-test the perturbation worker**

Run the command from Task 12 Step 2. Expected: six rows with positive susceptibility.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_topic4_zm_ictal_transition.py \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/{interictal_baseline_gate.json,dose_freeze.json,zm_off_reference_baseline.json,candidate_manifest.json}
git commit -m "topic4 zm-itx: canary networks, interictal-baseline gate, frozen perturbation dose"
```

---

### Task 15: Phase 2 — the four-arm formal runs

- [ ] **Step 1: Launch**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase formal \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

48 jobs: 4 arms × seeds 1811–1822, Z/M on. Expected wall clock 2–3 h at 8 workers (shorter than the Z/M-off reference because runs stop at detection plus 500 ms).

- [ ] **Step 2: Monitor**

Check `controller.status` every 600 s. Expected fields: `active`, `pending`, `mem_available_gib >= 32`, `module_hash_drift: false`.

- [ ] **Step 3: Verify completeness before moving on**

```bash
ls results/topic4_sef_hfo/data_driven_zm_ictal_transition/workers/*.json | wc -l
ls results/topic4_sef_hfo/data_driven_zm_ictal_transition/checkpoints/*.npz | wc -l
```

Expected: 48 worker jsons (plus the canary and parity ones), and up to 3 checkpoints per transitioned network. Any missing job is re-run individually before proceeding; do not aggregate over a partial set.

- [ ] **Step 4: Commit the artifacts index**

```bash
git add results/topic4_sef_hfo/data_driven_zm_ictal_transition/controller.log
git commit -m "topic4 zm-itx: complete the four-arm formal runs"
```

---

### Task 16: Phase 3 — perturbation, and the substrate null

- [ ] **Step 1: Launch the main perturbation sweep**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase perturbation \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Jobs: for each transitioned Joint-arm network, one job per `(seed, label)` with `label ∈ {baseline, pre_ictal, sensitivity}`; `--sites grid` for seeds 1811–1813 and `--sites representative` otherwise; `--measure-onset-advance` for `baseline` and `pre_ictal`, not for `sensitivity`.

Networks with `onset < 2500 ms` are skipped for all three labels and listed in `perturbation_exclusions.json`; networks with `onset < 3500 ms` are skipped for `sensitivity` only.

Because a baseline onset-advance continuation can allocate ~5.8 GB of spike buffer, the launcher must run `--measure-onset-advance --label baseline` jobs at a reduced concurrency computed from the measured sentinel, not at the flat cap.

- [ ] **Step 2: Launch the substrate null**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase null \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

12 primary runs (`joint_04_control`, seeds 1811–1822, `--field-transform` from the frozen assignment), then the representative-site perturbation protocol at `baseline` and `pre_ictal` for every null network that transitioned.

- [ ] **Step 3: Build the observation null (no simulation)**

Add `--gate observation-null` to the audit script: for each of the three grid networks, re-read the stored `E_spk_bool`-derived contact envelope with the montage rotated by each D4 element about the sheet centre, recompute the contact-level endpoints, and write `observation_null.json`. Assert that every rotated contact still passes `cmrun.valid_mask`; list any that do not and exclude them, reporting the count.

This gate answers readout dependence only and is labelled as such in its output.

- [ ] **Step 4: Verify completeness**

```bash
ls results/topic4_sef_hfo/data_driven_zm_ictal_transition/perturbation/*.json | wc -l
cat results/topic4_sef_hfo/data_driven_zm_ictal_transition/perturbation_exclusions.json
```

- [ ] **Step 5: Commit**

```bash
git add results/topic4_sef_hfo/data_driven_zm_ictal_transition/{controller.log,perturbation_exclusions.json,observation_null.json}
git commit -m "topic4 zm-itx: perturbation sweep, substrate null and observation null"
```

---

### Task 17: Aggregation and the pre-registered statistics

**Files:**
- Create: `scripts/aggregate_topic4_zm_ictal_transition.py`

**Interfaces:**
- Produces: `<output_root>/cohort_summary.json`, `cohort_summary.csv`, `primary_endpoint.json`, `spatial_endpoint.json`, `latency_endpoint.json`, `null_comparison.json`, `state_characterization.json`, `mode_evolution.json`.

- [ ] **Step 1: Implement the primary endpoint**

For every network: `susceptibility_pre - susceptibility_baseline`, each being the mean over that network's retained sites. Report `paired_bootstrap(pre, baseline, draws=4096, seed=20260817)` with `n`, `mean_difference`, `q05/q50/q95` and `n_positive`. Report the sensitivity checkpoint the same way as a robustness row, clearly labelled as such.

- [ ] **Step 2: Implement the primary spatial endpoint**

For each network, `spatial_correlation(susceptibility_field, covariate, site_xy, draws=2000, seed=20260817, block_mm=2.0)` for covariates `h`, `ee_out_gain`, `etoi_out_gain` (each averaged over the E neurons within 1.0 mm of each site so it lives on the site grid), and the neuron-level ictal onset density. Report per-network Spearman r and the cohort-level `paired_bootstrap` of the r values against zero, plus `hotspot_compactness` at baseline and pre-ictal.

- [ ] **Step 3: Implement the secondary latency endpoint**

`restricted_ictal_free_time` per arm over `[0, 20000]` ms; `paired_onset_difference` of each non-Node arm against Node on the both-entered subset; the entered fraction per arm. The output must state explicitly that 20 s is a censoring cap, not an onset.

- [ ] **Step 4: Implement the null comparison**

Paired, per network seed, data-driven Joint versus its assigned D4 image, on the **contact-independent** endpoints only for any mechanism statement: primary susceptibility change, spatial correlations, hotspot compactness, restricted ictal-free time, neuron-level onset density. Contact-dependent endpoints are reported in a separate block explicitly marked as readout-dependent. Per-element values are listed descriptively, with `r180` first and labelled `axis-preserving, flow-consistent spatial transform`. If more than half the null runs did not transition, the paired susceptibility contrast is emitted as `{"status": "NOT_EVALUABLE", "n_transitioned": k}` rather than computed.

- [ ] **Step 5: Implement the descriptive blocks**

State characterization aggregated across networks, always alongside the length-matched interictal reference and the `frequency_resolution_hz` / `n_cycles_at_band_low` caveats. Mode 1 / Mode 2 share, KMeans match and OOD fraction at baseline versus the last 2 s before onset, over returned events only, using `formal_mode_assignments` from `src/topic4_nlc_pathway_mechanism.py` with the frozen classifier from the manifest. Report the count of returned events inside the last 2 s next to every mode number.

- [ ] **Step 6: Run and re-derive**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/aggregate_topic4_zm_ictal_transition.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate reported-numbers \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

The `reported-numbers` gate recomputes every scalar in `cohort_summary.json` from the per-run artifacts and asserts equality to 1e-12; it fails loudly rather than warning.

- [ ] **Step 7: Commit**

```bash
git add scripts/aggregate_topic4_zm_ictal_transition.py \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/*.json \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/cohort_summary.csv
git commit -m "topic4 zm-itx: aggregate the pre-registered endpoints"
```

---

### Task 18: Figure 5 candidate, both assemblies

**Files:**
- Create: `scripts/paper_figures/plot_topic4_zm_ictal_transition_panels.py`
- Create: `scripts/paper_figures/build_main_figure_5.py`
- Create: `results/paper-ready-figure/fig5/figures/README.md`

- [ ] **Step 1: Read the style contract before writing any plotting code**

Read `docs/figure_style_guide.md` lines 166-206 (the Topic 4 section) and `docs/topic4_data_driven_snn_figure_spec.md`. This round's figure is a new type — an ictal-transition figure, not the four-column `mechanism + tempA source + tempB source + electrode readout` mechanism figure — so it does not inherit that layout, but it does inherit the colour locks: mechanism substrate in `plasma`; propagation events in `viridis` early→late; contact colours fixed, axis/A shaft orange and cross/B shaft cyan; Fig4's red/blue for the two interictal modes. The model ictal state uses a separate dark grey that is used for nothing else.

- [ ] **Step 2: Write the ten panel producers**

Each is a function `panel_<letter>(ax_or_axes, artifacts) -> dict` returning the metadata it wants recorded. No panel re-runs a simulation; all read from `<output_root>`.

```
A  substrate: h in plasma, the 15 contacts, the patient axis, and a schematic of the Z/M loop
B  the continuous 15-contact readout as one unbroken trace plus the population rate,
   with baseline / pre-ictal / model ictal onset marked
C  x(t) = 1 - mean_E[z], y(t) = eta_m * mean_E[m], coloured by time,
   title exactly "Projected Z/M trajectory"
D  baseline and pre-ictal D - A fields on one shared colour scale, static h as contours
E  baseline and pre-ictal response fields for one fixed representative site,
   shared grid, dose and colour scale
F  baseline and pre-ictal susceptibility maps plus the pre-minus-baseline difference
G  per-arm restricted ictal-free time with the censored fraction annotated
H  per-network susceptibility growth against onset advance, censored points marked
I  data-driven versus covariant transform on the contact-independent endpoints,
   r180 drawn first and separately labelled
J  Mode 1 / Mode 2 share, KMeans match and OOD fraction, baseline versus the last 2 s
```

No PASS/FAIL text, no internal status codes, no long explanatory text inside the axes. The readout panel's y-label says "virtual contact activity (firing-density envelope)", never voltage.

- [ ] **Step 3: Build both assemblies**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/paper_figures/build_main_figure_5.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Outputs:

```
results/paper-ready-figure/fig5/figures/
  fig5-data-driven-zm-transition-candidate.{png,pdf}      A-J, one sheet
  fig5-data-driven-zm-transition-candidate.gif            the transition, animated
  fig5-data-driven-zm-transition-main.{png,pdf}           A-F
  fig5-data-driven-zm-transition-supplement.{png,pdf}     G-J
  metadata.json
```

The GIF animates the virtual readout and the projected trajectory together from 0 ms to onset + 500 ms, at 20 fps, under 8 MB.

- [ ] **Step 4: Check panel non-redundancy, then write the README**

Open E and F side by side. E shows where activity goes when one fixed site is perturbed; F shows which site is more sensitive. If the two renders carry the same information — for example if the response field is essentially the susceptibility map re-scaled — replace E with the single-site response *time course* at the two checkpoints instead, and record the substitution in `metadata.json`.

Then write `results/paper-ready-figure/fig5/figures/README.md` in Chinese, `### filename` per file, 2–4 sentences of body, and a final `**关注点**：` line, per the repository standard. Write it only after the figures exist.

- [ ] **Step 5: Commit**

```bash
git add -f results/paper-ready-figure/fig5/figures/
git add scripts/paper_figures/plot_topic4_zm_ictal_transition_panels.py \
        scripts/paper_figures/build_main_figure_5.py
git commit -m "topic4 zm-itx: Figure 5 candidate in both a single-sheet and a main+supplement layout"
```

---

### Task 19: Report, docs, provenance

**Files:**
- Create: `docs/archive/topic4/zm_ictal_transition/zm_itx_report_2026-08-17.md`
- Modify: `docs/topic4_sef_hfo.md`, `results/FIGURE_INDEX.md`

- [ ] **Step 1: Write the scientific report**

Archive prose may use code names. The **first paragraph** answers, in this order: can this continue, what is safe to conclude now, what is the largest gap, what is next.

Required content: the primary endpoint with its interval and `n`; the primary spatial endpoint per covariate; the secondary latency endpoint stated as a restricted ictal-free time with the censoring cap named; the null comparison split into contact-independent and contact-dependent blocks; the state characterization recomputed on this round's trajectories with the length-matched interictal reference; every exclusion count.

Copy the claim-boundary section from the spec verbatim, including the prohibition on writing the null result as "patient spatial structure is unnecessary" and the four forbidden sentences.

- [ ] **Step 2: Update the topic doc**

Add one section to `docs/topic4_sef_hfo.md` with a summary and a link to the archive report. The section's opening sentence follows CLAUDE.md §8 — plain first-principles language, code names only in a trailing parenthetical. Invoke the `hfosp-plain-language-recap` skill before writing it.

- [ ] **Step 3: Update the figure index**

Add the Fig5 candidate row to `results/FIGURE_INDEX.md` with its producer script, its inputs, and the explicit note that the state label is operational.

- [ ] **Step 4: Write DONE.json and notify**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --finalize \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Writes `DONE.json` with the phase table, wall times, peak RSS, disk used, the git commit, and every gate verdict; fires `notify-send`.

- [ ] **Step 5: Run the whole test suite**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/ -m "not slow" -q`
Expected: all pass, no new failures versus the base commit.

- [ ] **Step 6: Commit**

```bash
git add docs/archive/topic4/zm_ictal_transition/zm_itx_report_2026-08-17.md \
        docs/topic4_sef_hfo.md results/FIGURE_INDEX.md \
        results/topic4_sef_hfo/data_driven_zm_ictal_transition/DONE.json
git commit -m "topic4 zm-itx: report the data-driven Z/M interictal-to-ictal transition round"
```

---

## Cost model

The measured unit is **94.5 s of wall clock per simulated second**, from the archived
seed-1561 run (1890.7 s for 20 s, single-threaded). Everything below scales from it, and
everything after Phase 2 depends on the median onset time, which **Phase 1 measures** — so
Phase 1's report must recompute this table before Phase 3 is launched.

| Phase | Work | onset ≈ 10 s | onset ≈ 18 s |
|---|---|---|---|
| Task 1 rebuild + Gate A parity | 1 network build + 1 full 20 s run | ~1.2 h | ~1.2 h |
| Phase 1 canary + dose freeze | 3 runs (2-pass) + 90 short probes | ~1 h | ~1.5 h |
| Phase 2 formal | 48 runs, 2-pass (pass 2 ≈ 75 % / 86 % of pass 1) | ~2.9 h | ~5.3 h |
| Phase 3 perturbation | ~400 probes; the ~200 baseline onset-advance continuations dominate at `onset - 2 s` each | ~5.3 h | ~10.6 h |
| Substrate null | 12 runs + ~144 probes | ~2.5 h | ~4.5 h |
| Observation null | 0 simulations | minutes | minutes |
| Aggregation + figures | — | ~1 h | ~1 h |
| **Total** | | **~14 h** | **~24 h** |

If Phase 1 lands at the slow end, the launcher reports the projection and **asks before
starting Phase 3**; the cheapest reduction is dropping the 7×7 grid from three seeds to one,
which removes ~86 of the ~200 long continuations.

Memory: 14.6 GiB per 20 s run. A 200 ms probe continuation holds only 64 MB of spike buffer;
an 18 s baseline onset-advance continuation holds ~5.8 GB, so those jobs get their own
concurrency budget rather than the flat cap.

Disk: network cache (16 pickles, size measured in Task 1 Step 7), checkpoints ≈ 130 MB × up
to 36 ≈ 4.7 GB, per-probe aggregate fields ≈ 51 MB, exemplar raw traces ≤ 400 MB. Against
187 GiB free.

## Self-review

**Spec coverage.** Engine changes → Tasks 3, 4, 6. Checkpoint contents → Task 2. Gates A/B/C → Tasks 11, 4, 5. Code layout → Tasks 1, 11, 12, 13. Phase 1 and the interictal-baseline gate → Task 14. Dose calibration → Task 14 Step 3. Phase 2 → Task 15. Phase 3, sites, susceptibility scalar, hotspot, onset density → Tasks 9, 12, 16, 17. Observation null → Task 16 Step 3. Covariant substrate null → Tasks 7, 16, 17. Endpoint tiers and the two orthogonal axes → Task 17. Censored latency → Tasks 10, 17. Slow-current product averages → Task 6. Projected trajectory → Task 18 panel C. State characterization → Tasks 8, 17. Interictal-mode analysis → Task 17 Step 5. Figure contract → Task 18. Claim boundary → Task 19. Execution discipline → Task 13.

**Known gap accepted deliberately:** the spec's `results/paper-ready-figure/fig5/README.md` is written as `figures/README.md` in Task 18, matching the repository standard that the README lives beside the figures.

**Type consistency.** `build_substrate` returns `Substrate` in Tasks 1, 7, 9, 11, 12. `capture`/`restore_slow`/`restore_external_drive` keep their Task 2 signatures in Tasks 4, 5, 6, 12. `susceptibility` is the same scalar in Tasks 9, 16, 17, 18. `field_transform` takes a D4 element name in Tasks 1, 7, 11, 16. `frozen_sites(kind=...)` takes `"grid"` or `"representative"` in Tasks 9, 12, 16.

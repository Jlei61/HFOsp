# Topic 4 ZM-ITX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the frozen patient-constrained Node + E→E + E→I substrate for `epilepsiae_1146`, switch on the per-neuron Z/M slow variables and answer whether the substrate **organizes** where pre-transition local susceptibility rises and where the transition ignites — or merely makes an already-unstable system destabilize sooner. Latency is secondary throughout.

**Architecture:** Four off-by-default parameters are added to the existing LIF engine (`post_runaway_record_ms`, `checkpoint_steps`/`checkpoint_sink`, `resume_state`, `time_offset_ms`) with byte-parity gates. A new substrate-rebuild module reconstructs the frozen rev11-NLC candidates without touching the producer script. A primary worker runs the Z/M trajectories and drops checkpoints; a perturbation worker resumes from those checkpoints and branches into paired sham/probe continuations that share the RNG stream, including **counterfactual splices** that swap `z` and `m` between checkpoints to separate slow-state accumulation from fast-state proximity. Phases are staged so a dead end costs three canary networks.

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
- Commit after every task. Never `git add -A`; always name files. **`results/` is gitignored except three legacy files** — every commit of a result artifact uses `git add -f`, and only small decision artifacts (gate verdicts, manifests, cohort summaries, figure files) are committed. Bulk `.npz` is never committed.
- **Phase gating is part of the contract, not an optimisation.** Phase 2 runs the Joint arm only; the four-arm latency runs, the spatial control and every long onset-advance continuation happen in Phase 3 and **only when Phase 2's paired E1 bootstrap gives `q05 > 0`**. A significantly negative result (`q95 < 0`) and a straddling interval both stop the round. An executor must not "get ahead" by launching Phase 3 work early.
- Two endpoints are kept separate and never pooled: **E1** sub-event finite response (primary) and **E2** ignition / onset advance (secondary, nonlinear). Every E1 site nevertheless carries its own in-window ignition flags, because those flags decide whether the site is an E1 measurement at all.

## Facts established before planning (do not re-derive)

- Under Z/M **off**, 48 runs (4 arms × seeds 1561-1572, 20 s) produced **0 transitions** and ~105 detected / ~87 returned events per run. This is the incidence control this round cites.
- Under Z/M **on**, `results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_active_zm_d7_canary/d7_canary_verdict.json` records **98/98 workers ran away** across 49 continuous node fields × 2 networks, `n_nonrunaway_workers: 0`, `safe_candidate_ids: []`. Onset: median **7989 ms**, min 5834, max 10291, q05 6458, q95 9981; per-candidate mean onset spans only 6362–9887 ms across 49 fields. **Boundary:** D7 used a different node-field family, an exact no-op edge mapper, and seeds 1421/1422 — a strong prior on onset timing, not a guarantee for the rev11-NLC mapper. It is why latency is secondary and why ~8 s is the working prior in the cost model.
- The node field's mass is `N_core_manual = 1129` of 32000 E neurons = **3.53 %**, so any unweighted population-mean Z/M statistic mostly reports background. Panel C and every headline Z/M trajectory are `h`-weighted.
- In the Z/M-off reference the network is above the common detector **41.2 %** of the time (`fraction_time_above_common_detector = 0.4115`). "The probe triggered an event" is meaningless here; every ignition test must be **probe-attributable** — present in the probe branch, absent from the paired sham branch.
- `src.topic4_forced_source_capacity.exclude_injected_packet_frame(forced, sham, packet_mask, trigger_step=...)` already exists and does exactly what the descendant-spike metric needs: it replaces the injection frame's packet-neuron entries with the sham's values. Reuse it; do not write a second one.
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
| `src/topic4_zm_recruitment.py` | bin-wise local recruitment time, 10→90 % spatial spread duration, axial vs off-axial lag |
| `src/topic4_zm_perturbation.py` | frozen probe sites, packet selection, **descendant-only** sham-subtracted response metrics, probe-attributable ignition, susceptibility maps, hotspot compactness, counterfactual state splicing |
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
      "sha256": "b97c442c1ca5bccc87a4d30ee73616c7703c14c9da146e3e96b4c1b91ad9249d"
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
  "phases": {
    "canary_arms": ["Joint"],
    "canary_zm_off_paired": true,
    "phase2_arms": ["Joint"],
    "phase3_latency_arms": ["Node", "Node+EE", "Node+EtoI"],
    "onset_relative_checkpoints_for_arms": ["Joint"],
    "phase2_continue_rule": "see phase2_stop_rule: continue only when q05 > 0",
    "canary_gate_minimum_passing_networks": 2
  },
  "observation_control": {
    "montage_transforms": ["identity", "r90", "r180", "r270", "mx", "my", "md1", "md2"],
    "record_in_primary_run": true
  },
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
    "reference": "same-seed Z/M-off canary runs at seeds 1801-1803",
    "baseline_rate_percentile": 95.0,
    "minimum_passing_canary_networks": 2
  },
  "repertoire_claim_gate": {
    "name": "INTERICTAL_REPERTOIRE_RETAINED",
    "is_run_blocker": false,
    "rule": "conjunctive: all four clauses must hold",
    "scope": "all returned events before onset",
    "reference_workers": "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/workers",
    "minimum_returned_events_before_onset": 20,
    "maximum_ood_fraction": "q95 across the 48 reference runs",
    "minimum_events_per_mode": 3,
    "minimum_kmeans_balanced_alignment": "q05 across the 48 reference runs",
    "kmeans_metric": "src.topic4_d6_natural_kmeans.best_binary_alignment -> balanced_alignment"
  },
  "phase2_stop_rule": {
    "q05_gt_0": "continue to Phase 3",
    "q95_lt_0": "stop, report the opposite direction",
    "straddles_zero": "stop, report unresolved at n=12"
  },
  "recruitment": {
    "bin_mm": 1.0,
    "rate_kernel_ms": 5.0,
    "bin_baseline_quantile": 0.99,
    "minimum_persistence_ms": 15.0,
    "reference_window_ms": [1000.0, 2000.0],
    "reference_window_rationale": "early interictal, NOT pre-onset: tau_z=5000 ms means a window 1 s before onset is already inside the buildup",
    "search_window_relative_to_onset_ms": [-300.0, 200.0],
    "offaxial_uses_absolute_perpendicular_distance": true
  },
  "perturbation": {
    "dose_ladder_cells": [16, 32, 64, 128, 256],
    "dose_selection": "smallest rung satisfying all four clauses",
    "dose_minimum_median_descendant_excess_spikes": 50.0,
    "dose_maximum_probe_attributable_event_units": 0,
    "dose_maximum_model_ictal_units": 0,
    "dose_linearity_ratio_range": [1.2, 3.0],
    "dose_calibration_units": "3 canary seeds x 6 representative sites = 18",
    "exclude_injected_frame_from_response": true,
    "packet_radius_mm": 1.0,
    "response_window_ms": 200.0,
    "response_split_ms": 50.0,
    "grid_seeds": "all formal seeds",
    "grid_extent_mm": [3.0, 17.0],
    "grid_n": 7,
    "grid_measures": ["E1", "in_window_ignition_flags"],
    "onset_advance_sites": "representative",
    "regime_limited_ignition_fraction": 0.25,
    "baseline_onset_search_cap_ms": 20000.0
  },
  "counterfactual_splices": [
    "native_baseline", "native_pre_ictal", "reset_z", "reset_m", "reset_zm", "slow_only"
  ],
  "spatial_reregistration_control": {
    "formal_element": "r180",
    "formal_seeds": "all formal seeds",
    "descriptive_elements": ["r90", "mx"],
    "descriptive_seeds": [1801, 1802, 1803],
    "arm": "Joint",
    "minimum_transitioned_control_networks": 6,
    "name": "matched spatial re-registration control"
  },
  "statistics": {"bootstrap_draws": 4096, "bootstrap_seed": 20260817,
                 "spatial_null_draws": 2000, "collinearity_report_threshold": 0.7},
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

`verify_frozen_inputs` hashes every entry of `config["inputs"]` and returns `{"all_match": bool, "records": {...}}`, raising `RuntimeError` on any mismatch. The recorded spec hash `b97c442c1ca5bc...` is the spec as it stands alongside this plan; if the spec is edited during review, recompute it with `sha256sum` and update the config before Task 13's freeze, which re-verifies it and fails on drift.

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
        tests/test_zm_ictal_transition_substrate.py
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/network_build_sentinel.json
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
  Field convention: `h'(x) = h(R^{-1}(x - c) + c)` with `c = (L/2, L/2)`, so `inverse_query_positions` returns `R^{-1}(x - c) + c`. Flow convention: the last two coefficients of each pathway row are rotated by the **same** `R`, i.e. `c' = R @ c`.

  **Scope of the covariance claim — the review caught an overclaim here.** Rotating `(c_x, c_y)` with the field makes the *field-and-flow rule* a rigid image of the original, so the internal field↔flow correspondence is preserved. It does **not** make the transformed substrate an isometric copy: the realized random graph, its patient-derived anisotropic topology (`theta_EE = -22.81 deg`) and the 15 contacts are all held fixed. The construct is therefore a **matched spatial re-registration control** — it asks whether the node field must be co-registered with the patient axis, the realized graph and the electrodes, not whether patient structure matters at all. The test below proves the rule's covariance by rotating the edge endpoints too, which the actual application does not do; it is named accordingly.

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


def test_flow_rule_is_covariant_when_the_edge_is_rotated_with_it():
    """Proves the field-and-flow RULE is a rigid image under the transform.
    Note the edge endpoints are rotated here; the actual control does not rotate
    the graph, which is why it is a re-registration control and not an isometry."""
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

Scope: this makes the field-and-flow RULE a rigid image of the original. It does
NOT make the substrate an isometric copy -- the realized random graph, its
patient-derived anisotropic topology and the contacts stay fixed. The construct
is a matched spatial re-registration control, not an isometry.
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
git commit -m "topic4 zm-itx: covariant field+flow transform for the spatial re-registration control"
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

### Task 8b: Local recruitment, replacing first-spike onset density

**Files:**
- Create: `src/topic4_zm_recruitment.py`
- Test: `tests/test_zm_recruitment.py`

**Why this replaces the earlier design.** The first plan built "ictal onset density" from each E
neuron's first spike inside the 100 ms before detection. In this network the unperturbed
population sits above the common detector 41 % of the time, so essentially every E neuron fires
at least once in any 100 ms window and that statistic is close to uniform noise. It is replaced
by a bin-wise threshold-crossing measure that can actually distinguish **sequential local
spread** from **near-simultaneous whole-field ignition** — the distinction the earlier
`q_I`/`g_K` line failed to make.

**Interfaces:**
- Consumes: `res["E_spk_bool"]`, `positions_e`, and the pre-onset reference window.
- Produces:
  ```python
  def spatial_bins(positions_e, *, bin_mm, sheet_l_mm) -> dict
      # {"bin_index": (n_e,) int, "bin_xy_mm": (n_bins, 2), "bin_counts": (n_bins,)}
  def bin_rate_traces(E_spk_bool, bin_index, n_bins, *, dt_ms, kernel_ms) -> np.ndarray
      # (n_bins, n_steps) float32, per-neuron rate in Hz within each bin
  def bin_baseline(rate_traces, *, dt_ms, window_steps, quantile) -> np.ndarray  # (n_bins,)
  def local_recruitment(rate_traces, thresholds, *, dt_ms, search_window_steps,
                        minimum_persistence_ms) -> dict
      # {"recruitment_step": (n_bins,) float (nan = never),
      #  "recruited_fraction": float,
      #  "spread_10_90_ms": float,
      #  "first_recruited_bin": int}
  def axial_lag(recruitment_step, bin_xy_mm, *, dt_ms, axis_unit, origin_xy) -> dict
      # {"axial_slope_ms_per_mm", "offaxial_slope_ms_per_mm", "axial_r", "offaxial_r"}
  ```

  **Two frozen windows, both fixed here rather than derived at run time:**

  - **Reference window `[1000, 2000] ms`, same seed**, for each bin's own q99 threshold. An
    earlier draft took it from the second ending at `onset − 1000 ms`. With `tau_z = 5000 ms`
    the slow drift operates on a five-second scale, so a window one second before onset is
    already inside the buildup; thresholding against it would inflate every bin's q99 by exactly
    the rise the measurement is meant to detect and mask real spread.
  - **Search window `[onset − 300 ms, onset + 200 ms]`**, fixed relative to the operational
    onset. `runaway_early_stop_ms` fires 100 ms after the EMA first crosses, so ignition starts
    before the reported onset and the window is placed to contain it.

  **`axial_lag` regresses on the SIGNED along-axis coordinate and the ABSOLUTE perpendicular
  distance `|d_perp|`.** With a signed perpendicular coordinate, spread that is symmetric about
  the axis cancels to a slope near zero and would be misread as "no off-axis propagation".

- [ ] **Step 1: Write the failing test**

```python
"""Local recruitment must separate sequential spread from simultaneous ignition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_recruitment import (  # noqa: E402
    axial_lag, bin_baseline, bin_rate_traces, local_recruitment, spatial_bins)


def _traveling(n_bins=20, n_steps=4000, dt=0.1, speed_bins_per_ms=0.1, base=10.0, hi=200.0):
    """Bin b crosses threshold at t = b / speed, then stays high."""
    traces = np.full((n_bins, n_steps), base, np.float32)
    for b in range(n_bins):
        onset = int(round((b / speed_bins_per_ms) / dt))
        if onset < n_steps:
            traces[b, onset:] = hi
    return traces


def _simultaneous(n_bins=20, n_steps=4000, base=10.0, hi=200.0, onset_step=1000):
    traces = np.full((n_bins, n_steps), base, np.float32)
    traces[:, onset_step:] = hi
    return traces


def test_bins_partition_every_neuron_exactly_once():
    rng = np.random.default_rng(0)
    positions = rng.random((5000, 2)) * 20.0
    out = spatial_bins(positions, bin_mm=1.0, sheet_l_mm=20.0)
    assert out["bin_index"].shape == (5000,)
    assert out["bin_index"].min() >= 0
    assert out["bin_counts"].sum() == 5000
    assert out["bin_xy_mm"].shape == (out["bin_counts"].size, 2)


def test_traveling_wave_gives_a_long_spread_duration():
    traces = _traveling()
    thresholds = np.full(traces.shape[0], 100.0)
    out = local_recruitment(traces, thresholds, dt_ms=0.1,
                            search_window_steps=4000, minimum_persistence_ms=15.0)
    assert out["recruited_fraction"] == 1.0
    # bins 0..19 at 10 ms apart -> 10 %-90 % spans about 160 ms
    assert 120.0 <= out["spread_10_90_ms"] <= 200.0
    assert out["first_recruited_bin"] == 0


def test_simultaneous_ignition_gives_a_near_zero_spread_duration():
    traces = _simultaneous()
    thresholds = np.full(traces.shape[0], 100.0)
    out = local_recruitment(traces, thresholds, dt_ms=0.1,
                            search_window_steps=4000, minimum_persistence_ms=15.0)
    assert out["recruited_fraction"] == 1.0
    assert out["spread_10_90_ms"] <= 1.0


def test_a_brief_blip_shorter_than_the_persistence_floor_is_not_recruitment():
    traces = np.full((5, 4000), 10.0, np.float32)
    traces[:, 1000:1050] = 200.0          # 5 ms, below the 15 ms floor
    thresholds = np.full(5, 100.0)
    out = local_recruitment(traces, thresholds, dt_ms=0.1,
                            search_window_steps=4000, minimum_persistence_ms=15.0)
    assert out["recruited_fraction"] == 0.0
    assert np.all(np.isnan(out["recruitment_step"]))


def test_threshold_is_each_bin_s_own_baseline_not_a_global_one():
    traces = np.zeros((2, 2000), np.float32)
    traces[0] = 5.0
    traces[1] = 500.0                     # a permanently busy bin
    thresholds = bin_baseline(traces, dt_ms=0.1, window_steps=2000, quantile=0.99)
    assert thresholds[1] > thresholds[0] * 10.0


def test_axial_lag_recovers_a_planted_axial_gradient():
    xy = np.stack([np.linspace(0, 19, 20), np.zeros(20)], axis=-1)
    recruitment_step = np.arange(20, dtype=float) * 100.0     # 10 ms per mm at dt=0.1
    out = axial_lag(recruitment_step, xy, dt_ms=0.1,
                    axis_unit=np.array([1.0, 0.0]), origin_xy=np.zeros(2))
    assert np.isclose(out["axial_slope_ms_per_mm"], 10.0, atol=0.5)
    assert abs(out["axial_r"]) > 0.99


def test_symmetric_offaxis_spread_is_not_cancelled_by_a_signed_coordinate():
    """The decisive off-axis regression: bins spread symmetrically on BOTH sides
    of the axis. A signed perpendicular coordinate averages the two sides to a
    slope near zero; the absolute distance recovers the real 8 ms/mm."""
    offsets = np.concatenate([np.arange(1.0, 11.0), -np.arange(1.0, 11.0)])
    xy = np.stack([np.zeros(20), offsets], axis=-1)
    recruitment_step = np.abs(offsets) * 80.0                 # 8 ms per mm at dt=0.1
    out = axial_lag(recruitment_step, xy, dt_ms=0.1,
                    axis_unit=np.array([1.0, 0.0]), origin_xy=np.zeros(2))
    assert np.isclose(out["offaxial_slope_ms_per_mm"], 8.0, atol=0.5)
    assert abs(out["offaxial_r"]) > 0.99
    assert abs(out["axial_slope_ms_per_mm"]) < 1e-6           # nothing along the axis


def test_offaxis_slope_is_not_confounded_by_the_axial_gradient():
    rng = np.random.default_rng(9)
    along = rng.uniform(-8.0, 8.0, 60)
    perp = rng.uniform(-6.0, 6.0, 60)
    xy = np.stack([along, perp], axis=-1)
    recruitment_step = along * 50.0 + np.abs(perp) * 20.0     # 5 and 2 ms/mm
    out = axial_lag(recruitment_step, xy, dt_ms=0.1,
                    axis_unit=np.array([1.0, 0.0]), origin_xy=np.zeros(2))
    assert np.isclose(out["axial_slope_ms_per_mm"], 5.0, atol=0.3)
    assert np.isclose(out["offaxial_slope_ms_per_mm"], 2.0, atol=0.3)
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_recruitment.py -v`
Expected: `ModuleNotFoundError: No module named 'src.topic4_zm_recruitment'`.

- [ ] **Step 3: Implement**

`spatial_bins` floors `positions / bin_mm` into a regular lattice over `[0, sheet_l_mm)` and
returns only the occupied bins, with `bin_xy_mm` at each bin's centre. `bin_rate_traces` sums
spikes per bin per step, divides by that bin's neuron count and by `dt_ms * 1e-3` to get Hz, then
convolves with a normalised Gaussian of `kernel_ms` standard deviation. `bin_baseline` returns
the `quantile` of each bin's own rate over the reference window — **per bin, never a single
global threshold**, because bin occupancy and background rate vary several-fold across the
sheet. `local_recruitment` finds, per bin, the first step inside the search window where the
rate exceeds that bin's threshold **and stays above it for at least `minimum_persistence_ms`**;
`spread_10_90_ms` is the time between the 10th and 90th percentile of the finite recruitment
times, which is the number that separates a travelling front from a simultaneous flash.

`axial_lag` builds the design matrix from `d_along = (xy - origin) @ axis_unit` and
`d_perp_abs = |(xy - origin) @ normal|`, fits recruitment time jointly on both columns, and
returns each slope with its partial correlation. Fitting them **jointly** is what makes
`test_offaxis_slope_is_not_confounded_by_the_axial_gradient` pass; two separate univariate fits
would let an axial gradient leak into the off-axis slope.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_recruitment.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_recruitment.py tests/test_zm_recruitment.py
git commit -m "topic4 zm-itx: bin-wise local recruitment replaces the noise-dominated first-spike density"
```

---

### Task 9: Perturbation sites, packets, descendant-only response, and counterfactual splices

**Files:**
- Create: `src/topic4_zm_perturbation.py`
- Test: `tests/test_zm_perturbation.py`

**Interfaces:**
- Consumes: `Substrate` from Task 1; `src.topic4_forced_source_capacity.exclude_injected_packet_frame`; `src.snn_engine.checkpoint`.
- Produces:
  ```python
  def frozen_sites(substrate, config, *, kind) -> list[dict]
      # kind in {"grid", "representative"}; each dict has
      # {"site_id": str, "xy_mm": (2,) float, "kind": str}
  def select_packet(positions_e, site_xy, *, n_cells, radius_mm) -> np.ndarray  # (n_e,) bool
  def response_metrics(probe, sham, *, dt_ms, positions_e, packet_mask, packet_xy,
                       envelope_probe, envelope_sham, envelope_dt_ms,
                       inject_step, split_ms, window_ms) -> dict          # E1
  def in_window_ignition(probe, sham, *, dt_ms, detector_threshold,
                         inject_step, window_ms) -> dict                  # ALWAYS, free
  def ignition_metrics(probe, sham, *, dt_ms, detector_threshold, window_ms,
                       probe_onset_ms, sham_onset_ms) -> dict             # E2 long arm
  def splice_checkpoint(pre_ictal_state, baseline_state, *, mode) -> dict
  def susceptibility_map(rows, *, sites) -> dict
  def hotspot_compactness(sites_xy, values, *, quantile, n_null, seed) -> dict
  ```

  **E1** (`response_metrics`) returns `susceptibility` — the canonical scalar, the total
  **descendant** probe-minus-sham excess E spikes over `0..window_ms` after injection — plus its
  `excess_spikes_early` / `excess_spikes_late` decomposition, `r90_mm`,
  `contact_excess_energy`, and `excess_per_neuron` (n_e float32). "Descendant" means the probe
  spike array is first passed through `exclude_injected_packet_frame`, which replaces the
  injection frame's packet-neuron entries with the sham's. **Without that step a 256-cell packet
  contributes 256 excess spikes with zero recursive amplification.**

  **`in_window_ignition` runs on EVERY E1 site, grid included**, and returns
  `probe_attributable_event_200ms` and `reached_model_ictal_200ms`. It is computed from arrays
  the run already holds and costs nothing. **Freezing the dose on baseline checkpoints
  guarantees nothing at the pre-ictal checkpoint**, which is exactly where excitability is
  hypothesised to be higher; without these flags a pre-ictal probe that ignites would have its
  escape-dominated spike count recorded as "susceptibility grew".

  A site with either flag true gets `e1_evaluable = False` and is excluded from the E1 mean, but
  **is never deleted from the site set** — deleting igniting sites would strip the most excitable
  locations out of the pre-ictal map. It is handed to E2 instead.

  **E2** (`ignition_metrics`) additionally returns `onset_advance_ms` from the long
  continuation, run only at the representative sites. An event counts only if it is present in
  the probe branch and **absent from the paired sham branch** — the unperturbed network is above
  the common detector 41 % of the time, so "an event occurred" is not evidence of anything.

  `splice_checkpoint` builds the counterfactual states. `mode` is one of
  `native_baseline`, `native_pre_ictal`, `reset_z`, `reset_m`, `reset_zm`, `slow_only`;
  it deep-copies the donor state and overwrites only `state["slow"]["z"]` and/or
  `state["slow"]["m"]`, never the fast state, RNG or OU payloads of the host.

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
    hotspot_compactness, ignition_metrics, in_window_ignition, response_metrics,
    select_packet, splice_checkpoint)


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


def _flat(n_steps=2000, n_e=500):
    envelope = np.zeros((15, 200), np.float32)
    return dict(dt_ms=0.1, envelope_probe=envelope, envelope_sham=envelope,
                envelope_dt_ms=2.0, inject_step=0, split_ms=50.0, window_ms=200.0)


def test_identical_probe_and_sham_give_exactly_zero_susceptibility():
    n_steps, n_e = 2000, 500
    spikes = np.zeros((n_steps, n_e), bool)
    spikes[::7, ::3] = True
    packet = np.zeros(n_e, bool); packet[:32] = True
    out = response_metrics({"E_spk_bool": spikes}, {"E_spk_bool": spikes},
                           positions_e=np.zeros((n_e, 2)), packet_mask=packet,
                           packet_xy=np.zeros(2), **_flat())
    assert out["susceptibility"] == 0.0
    assert out["excess_spikes_early"] == 0.0
    assert out["excess_spikes_late"] == 0.0
    assert out["contact_excess_energy"] == 0.0


def test_injected_spikes_alone_produce_exactly_zero_susceptibility():
    """The decisive regression: a packet that fires once and propagates to
    nothing must score 0, not `n_cells`."""
    n_steps, n_e, n_cells = 2000, 500, 256
    sham = np.zeros((n_steps, n_e), bool)
    probe = sham.copy()
    packet = np.zeros(n_e, bool)
    packet[:n_cells] = True
    probe[0, packet] = True                     # the injection, and nothing else
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           positions_e=np.zeros((n_e, 2)), packet_mask=packet,
                           packet_xy=np.zeros(2), **_flat())
    assert out["susceptibility"] == 0.0
    assert out["excess_spikes_early"] == 0.0


def test_descendant_spikes_are_still_counted():
    n_steps, n_e, n_cells = 2000, 500, 32
    sham = np.zeros((n_steps, n_e), bool)
    probe = sham.copy()
    packet = np.zeros(n_e, bool)
    packet[:n_cells] = True
    probe[0, packet] = True                     # injection, removed
    probe[40, 400:410] = True                   # 10 descendants at 4 ms
    probe[900, 400:405] = True                  # 5 descendants at 90 ms
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           positions_e=np.zeros((n_e, 2)), packet_mask=packet,
                           packet_xy=np.zeros(2), **_flat())
    assert out["susceptibility"] == 15.0
    assert out["excess_spikes_early"] == 15.0   # both inside 0-50 ms
    assert out["excess_spikes_late"] == 0.0


def test_susceptibility_is_the_sum_of_its_two_parts():
    rng = np.random.default_rng(2)
    n_steps, n_e = 2000, 300
    sham = rng.random((n_steps, n_e)) < 0.001
    probe = sham | (rng.random((n_steps, n_e)) < 0.002)
    packet = np.zeros(n_e, bool); packet[:16] = True
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           positions_e=rng.random((n_e, 2)) * 5.0,
                           packet_mask=packet, packet_xy=np.array([2.5, 2.5]),
                           **_flat())
    assert np.isclose(out["susceptibility"],
                      out["excess_spikes_early"] + out["excess_spikes_late"])
    assert 0.0 < out["r90_mm"] <= np.sqrt(2) * 5.0


def test_ignition_requires_the_event_to_be_absent_from_the_sham():
    """The network is above the common detector 41 % of the time, so an event
    in the probe branch is only evidence if the sham branch lacks it."""
    n_steps, n_e = 2000, 500
    both = np.zeros((n_steps, n_e), bool)
    both[500:700, :200] = True                  # a large event in BOTH branches
    shared = ignition_metrics({"E_spk_bool": both}, {"E_spk_bool": both},
                              dt_ms=0.1, detector_threshold=0.02, window_ms=200.0,
                              probe_onset_ms=None, sham_onset_ms=None)
    assert shared["probe_attributable_event"] is False

    probe_only = np.zeros((n_steps, n_e), bool)
    probe_only[500:700, :200] = True
    only = ignition_metrics({"E_spk_bool": probe_only},
                            {"E_spk_bool": np.zeros((n_steps, n_e), bool)},
                            dt_ms=0.1, detector_threshold=0.02, window_ms=200.0,
                            probe_onset_ms=None, sham_onset_ms=None)
    assert only["probe_attributable_event"] is True


def test_onset_advance_is_nan_when_either_branch_is_censored():
    out = ignition_metrics({"E_spk_bool": np.zeros((10, 5), bool)},
                           {"E_spk_bool": np.zeros((10, 5), bool)},
                           dt_ms=0.1, detector_threshold=0.02, window_ms=1.0,
                           probe_onset_ms=None, sham_onset_ms=8000.0)
    assert np.isnan(out["onset_advance_ms"])
    assert out["onset_censored"] is True


def test_splice_leaves_every_non_slow_field_bit_identical():
    """Splice integrity is a bit-level property, not an assumption: apart from
    the named z/m arrays, the host's fast state, OU state, both RNG states, the
    delay rings and the time index must be untouched."""
    import copy

    from src.snn_engine.checkpoint import digest

    rng = np.random.default_rng(11)
    def _state(tag):
        return {
            "schema": "topic4_snn_checkpoint_v1", "step": 20000 + tag,
            "absolute_time_ms": 2000.0 + tag,
            "V": rng.random(8), "ref": rng.integers(0, 4, 8).astype(np.int32),
            "s_E": rng.random(8), "I_E": rng.random(8),
            "s_I": rng.random(8), "I_I": rng.random(8),
            "ring_sE": rng.random((5, 8)), "ring_sI": rng.random((5, 8)),
            "xi": float(rng.random()), "rng_state": {"bit_generator": "PCG64", "n": tag},
            "ras_keep": np.array([0, 2, 4]), "es_ema": 3.0 + tag, "es_run": tag,
            "track_rec": False, "s_E_rec": None, "I_E_rec": None,
            "slow": {"kind": "MZSlowVars", "z": rng.random(8), "m": rng.random(8),
                     "I_I_last": rng.random(8), "step_index": 100 + tag},
            "external_drive": {"field_state": rng.random((4, 4)), "cached": rng.random(8),
                               "next_step": 30 + tag, "last_step": 29 + tag,
                               "rng_state": {"bit_generator": "PCG64", "n": 99 + tag}},
        }

    pre, base = _state(1), _state(2)
    for mode, host in (("reset_z", pre), ("reset_m", pre), ("reset_zm", pre),
                       ("slow_only", base)):
        out = splice_checkpoint(pre, base, mode=mode)
        host_no_slow = copy.deepcopy(host); host_no_slow["slow"] = None
        out_no_slow = copy.deepcopy(out); out_no_slow["slow"] = None
        assert digest(out_no_slow) == digest(host_no_slow), mode
        assert out["slow"]["step_index"] == host["slow"]["step_index"], mode
        assert np.array_equal(out["slow"]["I_I_last"], host["slow"]["I_I_last"]), mode


def test_splice_only_touches_the_named_slow_variable():
    pre = {"slow": {"z": np.full(4, 0.2), "m": np.full(4, 5.0)},
           "V": np.arange(4.0), "external_drive": {"next_step": 7}}
    base = {"slow": {"z": np.full(4, 0.9), "m": np.full(4, 0.1)},
            "V": np.zeros(4), "external_drive": {"next_step": 1}}
    out = splice_checkpoint(pre, base, mode="reset_z")
    assert np.allclose(out["slow"]["z"], 0.9)     # taken from baseline
    assert np.allclose(out["slow"]["m"], 5.0)     # kept from pre-ictal
    assert np.allclose(out["V"], np.arange(4.0))  # fast state untouched
    assert out["external_drive"]["next_step"] == 7
    assert np.allclose(pre["slow"]["z"], 0.2)     # donor not mutated

    both = splice_checkpoint(pre, base, mode="reset_zm")
    assert np.allclose(both["slow"]["z"], 0.9) and np.allclose(both["slow"]["m"], 0.1)

    sufficiency = splice_checkpoint(pre, base, mode="slow_only")
    assert np.allclose(sufficiency["V"], np.zeros(4))        # baseline fast state
    assert np.allclose(sufficiency["slow"]["z"], 0.2)        # pre-ictal slow state
    assert sufficiency["external_drive"]["next_step"] == 1


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

`select_packet` takes the `n_cells` nearest E neurons within `radius_mm`, raising `ValueError("insufficient E neurons within the packet radius")` when fewer than `n_cells` qualify.

`response_metrics` **first** strips the injection:

```python
from src.topic4_forced_source_capacity import exclude_injected_packet_frame

probe_descendant = exclude_injected_packet_frame(
    probe["E_spk_bool"], sham["E_spk_bool"], packet_mask, trigger_step=inject_step)
excess = probe_descendant.sum(axis=0) - sham["E_spk_bool"].sum(axis=0)
```

then restricts to `inject_step .. inject_step + window_ms/dt`, splits at `split_ms`, sums for
the scalar, computes `r90_mm` as the radius about `packet_xy` containing 90 % of the positive
excess, and integrates `clip(envelope_probe - envelope_sham, 0, None)` for
`contact_excess_energy`.

`in_window_ignition` computes the population active fraction of each branch in 1 ms bins over
`inject_step .. inject_step + window_ms/dt`, applies the frozen `detector_threshold`, and sets
`probe_attributable_event_200ms` only when the probe branch has a supra-threshold run that the
sham branch does not have overlapping it. `reached_model_ictal_200ms` applies the 120 Hz /
100 ms criterion to the probe branch inside the same window. It is called on **every** E1 site.

`ignition_metrics` wraps `in_window_ignition` and adds the long arm: `onset_advance_ms =
sham_onset_ms - probe_onset_ms`, or `nan` with `onset_censored=True` if either is `None`.

`splice_checkpoint` deep-copies the **host** state and overwrites only the named slow arrays:

```python
_SPLICE = {                      # (host, z source, m source)
    "native_baseline":  ("baseline",  "baseline",  "baseline"),
    "native_pre_ictal": ("pre_ictal", "pre_ictal", "pre_ictal"),
    "reset_z":          ("pre_ictal", "baseline",  "pre_ictal"),
    "reset_m":          ("pre_ictal", "pre_ictal", "baseline"),
    "reset_zm":         ("pre_ictal", "baseline",  "baseline"),
    "slow_only":        ("baseline",  "pre_ictal", "pre_ictal"),
}
```

It must `copy.deepcopy` before writing so neither donor is mutated — the same state object is
reused across every site at a checkpoint. A spliced state is **off-manifold**: the dynamics
never visit "pre-ictal fast state with baseline `z`". Every consumer stamps
`"off_manifold": mode not in ("native_baseline", "native_pre_ictal")` into its output so the
report cannot silently present a splice as a trajectory.

`hotspot_compactness` takes sites above the `quantile` of `values`, computes their mean pairwise
distance, and compares against `n_null` random equal-size subsets of the same site set,
returning a one-sided empirical p-value.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_perturbation.py -v`
Expected: 10 passed. `test_injected_spikes_alone_produce_exactly_zero_susceptibility` is the
regression that must never be weakened — it is the whole reason the descendant metric exists —
and `test_splice_leaves_every_non_slow_field_bit_identical` is what turns splice integrity from
an assumption into a checked property.

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
  def exact_toroidal_shifts(grid_n) -> np.ndarray
      # (grid_n**2, 2) int, every (dx, dy) on the torus INCLUDING (0, 0)
  def spatial_correlation_exact_shift(values, covariate, *, grid_n) -> dict
      # {"spearman_r", "p_value", "n_distinct_shifts", "p_floor", "null_r"}
  def covariate_collinearity(covariates: dict) -> dict
      # {"pairwise_spearman": {...}, "max_abs_r": float}
  ```

  **The spatial null is enumerated, not sampled.** On the frozen 7×7 grid the shift group has
  exactly `49` elements including the identity, so `spatial_correlation_exact_shift` takes no
  `draws` argument at all: it computes all 49 null correlations, returns
  `n_distinct_shifts == 49` and `p_floor == 1/49`, and `p_value = mean(|null_r| >= |r_obs|)`
  over that complete set. An earlier draft carried a generic `draws=2000` sampler here, which
  contradicted the spec's "exactly 49 shifts" and would have implied a precision the design does
  not have.

  **`covariate_collinearity` reports, it does not decide.** `h` is the primary spatial covariate
  and local recruitment time is the second primary; the outgoing E→E and E→I gains are
  descriptive companions. There is no data-dependent merge rule and no composite: an earlier
  draft collapsed all three into one "family" whenever any pair correlated above `0.7`, which
  both left the composite undefined and made the reported quantity depend on the data.

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
    covariate_collinearity, exact_toroidal_shifts, paired_bootstrap,
    paired_onset_difference, restricted_ictal_free_time,
    spatial_correlation_exact_shift)


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


def test_the_shift_group_is_enumerated_in_full():
    shifts = exact_toroidal_shifts(7)
    assert shifts.shape == (49, 2)
    assert len({tuple(s) for s in shifts}) == 49
    assert (0, 0) in {tuple(s) for s in shifts}


def test_exact_shift_null_reports_49_shifts_and_a_1_over_49_floor():
    rng = np.random.default_rng(0)
    values = rng.random(49)
    covariate = rng.random(49)
    out = spatial_correlation_exact_shift(values, covariate, grid_n=7)
    assert out["n_distinct_shifts"] == 49
    assert np.isclose(out["p_floor"], 1.0 / 49.0)
    assert out["p_value"] >= out["p_floor"] - 1e-12
    assert len(out["null_r"]) == 49


def test_exact_shift_null_hits_its_floor_on_a_perfect_match():
    grid = np.stack(np.meshgrid(np.arange(7), np.arange(7), indexing="ij"),
                    axis=-1).reshape(-1, 2)
    field = np.sin(grid[:, 0] * 0.9) + np.cos(grid[:, 1] * 0.7)
    out = spatial_correlation_exact_shift(field, field, grid_n=7)
    assert np.isclose(out["spearman_r"], 1.0)
    assert np.isclose(out["p_value"], 1.0 / 49.0)


def test_exact_shift_null_is_deterministic():
    rng = np.random.default_rng(3)
    values, covariate = rng.random(49), rng.random(49)
    assert (spatial_correlation_exact_shift(values, covariate, grid_n=7)
            == spatial_correlation_exact_shift(values, covariate, grid_n=7))


def test_collinearity_reports_every_pair_and_decides_nothing():
    rng = np.random.default_rng(4)
    h = rng.random(49)
    covariates = {"h": h, "ee_gain": h * 2.0 + 0.01 * rng.random(49),
                  "etoi_gain": h * -1.5 + 0.01 * rng.random(49)}
    out = covariate_collinearity(covariates)
    assert out["max_abs_r"] > 0.9
    assert set(out["pairwise_spearman"]) == {("ee_gain", "etoi_gain"),
                                             ("h", "ee_gain"), ("h", "etoi_gain")}
    assert "report_as_single_family" not in out    # h is primary by design, not by data
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_statistics.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`paired_bootstrap` asserts equal length, resamples **network indices** (not values independently), and reports quantiles of the mean paired difference `a - b`. `restricted_ictal_free_time` maps `None`/`NaN` to `cap_ms` and averages — this is the restricted mean ictal-free time and is the only latency number that may be reported across arms with censoring. `paired_onset_difference` keeps only indices where both entries are finite.

`exact_toroidal_shifts(grid_n)` returns all `grid_n**2` integer shifts including `(0, 0)`.
`spatial_correlation_exact_shift` reshapes the covariate to `(grid_n, grid_n)`, applies each
shift with `np.roll` on both axes, flattens, and computes the Spearman correlation against
`values`; rigid shifts preserve the covariate's spatial autocorrelation, which a plain
permutation null would destroy, making it anticonservative. Because the group is small the null
is **enumerated in full** — no `draws`, no seed, deterministic — and the function returns
`n_distinct_shifts = grid_n**2`, `p_floor = 1/grid_n**2` and the complete `null_r` vector so the
report can state the floor rather than imply arbitrary precision. The load-bearing test remains
the cohort-level paired bootstrap over the 12 per-network r values, not the per-network p.

`covariate_collinearity` returns every pairwise Spearman r among the supplied covariate fields
and the maximum absolute value — and nothing else. It deliberately has **no** decision output:
`h` and local recruitment time are the primary spatial covariates by design, fixed before any
run, and the outgoing pathway gains are descriptive companions. A data-dependent merge rule
would be a degree of freedom, and the composite it implied was never defined.

- [ ] **Step 4: Run the tests**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest tests/test_zm_statistics.py -v`
Expected: 9 passed.

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
- Produces: per run, `<output_root>/workers/<arm>_seed_<seed>[_ctl_<element>].json` and `.npz`, plus `<output_root>/checkpoints/<arm>_seed_<seed>_<label>.npz`. The npz keys mirror the rev10-R worker's (`contact_names`, `shaft_ids`, `contact_xy_mm`, `onsets`, `ranks`, `event_t_on_ms`, `event_t_off_ms`, `event_returned`, `active_fraction`, `active_fraction_bin_ms`, `contact_envelope`, `contact_envelope_dt_ms`, `positions_E`, `h`, `h_I_for_edge`, `delta_vtheta`, `edge_coefficients`, `spatial_ou_*`, `mz_*`) and add:
  - `ee_out_gain`, `etoi_out_gain` — (n_e,) float32 pathway gains
  - `zm_h_weighted_z_mean`, `zm_h_weighted_m_mean` — (n_trace,) float32, the `h`-weighted trajectory that Panel C plots; the unweighted `mz_z_mean` / `mz_m_mean` stay as the grey reference
  - `state_characterization_*` — the post-detection block
  - `recruitment_bin_xy_mm`, `recruitment_step`, `recruitment_spread_10_90_ms`, `recruitment_axial_slope_ms_per_mm`, `recruitment_offaxial_slope_ms_per_mm` — from Task 8b
  - `contact_envelope_ctl_<element>` — one extra (15, n_frames) float32 per pre-frozen transformed montage, **eight in total including identity**

  **The transformed-montage envelopes must be recorded here.** The 20 s per-neuron spike array is 6.4 GB and is never written to disk, so the observation control cannot be reconstructed offline from the original envelope alone. Sampling the spikes that are already in memory through seven extra montages costs one `snn_event_envelope` call each (~600 KB output) and **zero extra simulation**.

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
   - **Pass 2 runs for the Joint arm only.** Node, Node+EE and Node+EtoI carry the latency endpoint and nothing else, so they need no onset-relative checkpoints and stop after pass 1. The worker refuses `--emit-onset-checkpoints` for any arm outside `config["phases"]["onset_relative_checkpoints_for_arms"]`.

   A ring-buffer alternative (keep the last N checkpoints in memory and pick after detection) was rejected: at ~130 MB per checkpoint a ring fine enough to hit `onset - 500` exactly would cost tens of GB per worker, and a coarser ring would make the pre-ictal lead time vary per network, which weakens the paired primary contrast.
9. Readout: `active_fraction` → `detect_events` at the frozen detector → per-event contact onsets and ranks via the rev10-R `_contact_onsets` helper (import it from that module — it is read-only reuse, not modification).
10. **Observation-control montages**: while `spikes` is still in memory, call `snn_event_envelope(spikes, positions, montage_k, dt)` once for each of the eight pre-frozen montages (identity plus the seven square-symmetry images of the contact set about the sheet centre). Assert each transformed montage passes `cmrun.valid_mask`; list and exclude any contact that does not, and record the excluded count. Store as `contact_envelope_ctl_<element>`.
11. **`h`-weighted Z/M trajectory**: from the `MZSlowVars` trace stride, additionally accumulate `sum_i h_i z_i / sum_i h_i` and `eta_m * sum_i h_i m_i / sum_i h_i`. The node field is only 3.53 % of the E population, so the unweighted population mean mostly reports background and may not be the headline trajectory.
12. State characterization over the post-detection recording, with the length-matched interictal reference taken from the 500 ms ending at `onset - 1000 ms`.
13. Local recruitment (Task 8b) over the **frozen search window `[onset − 300 ms, onset + 200 ms]`**, with each bin's threshold taken from its own rate over the **frozen reference window `[1000, 2000] ms`** — an early interictal segment, *not* a pre-onset one. With `tau_z = 5000 ms` the slow drift runs on a five-second scale, so a threshold window one second before onset already sits inside the buildup and would be inflated by the very rise the measurement is meant to detect.
14. Write json + npz atomically.

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
- Produces: `<output_root>/perturbation/<arm>_seed_<seed>[_ctl_<element>]_<label>[_<splice>].json` and `.npz`, where `label ∈ {baseline, pre_ictal, sensitivity}` and `splice` defaults to `native`. The npz holds `site_id` (U8), `site_xy_mm`, the **E1** block — `susceptibility`, `excess_spikes_early`, `excess_spikes_late`, `r90_mm`, `contact_excess_energy`, `excess_per_neuron` (n_sites × n_e float32) — and, **always**, the in-window regime block `probe_attributable_event_200ms`, `reached_model_ictal_200ms`, `e1_evaluable` (all n_sites bool). Only when `--measure-onset-advance` is set it additionally holds `onset_advance_ms`, `onset_censored`. Every record carries `off_manifold` (bool) from `splice_checkpoint`. `slow_field_D`, `slow_field_A`, `slow_field_net` (n_e float64) come from the sham run's accumulator.

  **E1 and E2 are never written into the same column and never averaged together.** The in-window regime flags travel with every E1 row because they say whether that row is an E1 measurement at all; the ignition *endpoint* is assembled from them separately by the aggregation, and the long onset-advance arm lives in its own files.

- [ ] **Step 1: Write the worker**

CLI: `--config --candidate-id --seed --checkpoint --baseline-checkpoint --label --splice {native,reset_z,reset_m,reset_zm,slow_only} --sites {grid,representative} --dose-cells --measure-onset-advance --onset-cap-ms --expected-commit --out-json --out-npz`.

The worker loads the network and the checkpoint **once**, then:

1. If `--splice` is not `native`, load `--baseline-checkpoint` as well and build the host state with `splice_checkpoint(pre_ictal_state, baseline_state, mode=splice)`. Refuse a splice when `--label` is not `pre_ictal` (except `slow_only`, whose host is the baseline state), and stamp `off_manifold=True` into every record. Splices are only ever run with `--sites representative`.
2. Run the **sham** continuation for `window_ms` with the slow-current accumulator enabled for the first 1000 steps (100 ms). The sham is identical for every site at this (checkpoint, splice), so compute it **once** and reuse — this halves the cost.
3. Per site: deep-copy the host state, restore, run the **probe** continuation with `forced_spike_mask` from `select_packet` and `forced_spike_ms = checkpoint_absolute_time_ms` (injection on the first step of the continuation; the Task 4 clock fix is what makes this legal).
4. `response_metrics(..., packet_mask=..., inject_step=0)` → E1, **and always**
   `in_window_ignition(..., inject_step=0)` → `probe_attributable_event_200ms`,
   `reached_model_ictal_200ms`, `e1_evaluable = not (either flag)`.
5. Only when `--measure-onset-advance`: run both branches again with `early_stop_runaway=True` and duration `onset_cap_ms - checkpoint_time`, then `ignition_metrics(...)` → the long E2 arm. For `baseline`, `onset_cap_ms = 20000.0` (right-censored at the frozen duration cap); for `pre_ictal`, `sham_onset + 1500.0`.

**The in-window flags run on grid jobs too — that is the point.** They come free from arrays the
run already holds, and without them a pre-ictal grid probe that ignites would silently record an
escape-dominated spike count as susceptibility. A site whose flag fires keeps its row with
`e1_evaluable = False`; **the worker never drops the site**, because deleting igniting sites
would strip the most excitable locations out of the pre-ictal map.

What grid jobs still may not do is the **long** continuation: `--measure-onset-advance` is never
combined with `--sites grid`, and the worker errors if both are given.

Peak memory per continuation is bounded by `E_spk_bool` over the continuation only: 200 ms → 2000 × 32000 bytes = 64 MB; a baseline ignition continuation of ~6 s → 1.9 GB. Ignition jobs get their own concurrency budget in Task 13.

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
        scripts/freeze_topic4_zm_ictal_transition.py
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/candidate_manifest.json
git commit -m "topic4 zm-itx: freeze the round manifest and add the systemd launcher"
```

---

### Task 14: Phase 1A — canary, and the one science gate

**Files:**
- Modify: `scripts/audit_topic4_zm_ictal_transition.py` (add `--gate interictal-baseline`)

- [ ] **Step 1: Run six canary runs**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase canary \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Seeds 1801–1803, `joint_04_control`, in **two** variants at each seed:
- Z/M **on** — 20 s cap, 500 ms post-detection recording, two-pass checkpointing.
- Z/M **off** — 20 s, no checkpoints. These three runs exist to supply the *same-seed* interictal reference the gate compares against. They do **not** reopen the earlier decision to skip a formal Z/M-off arm; the incidence control remains the archived 48-run 0/48 reference, and that is stated wherever it is cited.

- [ ] **Step 2: Adjudicate the gate**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate interictal-baseline \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Evaluate **all three clauses for every network and report every failing clause**, not only the first:

```
onset_ms >= 2500
n_returned_events_before_onset >= 3
median 20 ms-EMA E rate over [1500, 2000] ms
    <= percentile_95 of the same statistic across the SAME-SEED Z/M-off canary runs
```

The reference percentile is computed from `active_fraction` in the three Z/M-off canary npz files, converted to the same 20 ms-EMA rate units, and cached into `<output_root>/zm_off_reference_baseline.json` with its own hash.

**Continue when ≥ 2 of 3 networks pass all three clauses.** A single network failing is a draw of the network seed, not a property of the work point; the formal phase excludes such a network individually. Passing fewer than 2 of 3 means the work point itself has no interpretable interictal residence segment — write that finding and stop. **The baseline checkpoint is never moved earlier to rescue a failing network.**

Per D7's 5834 ms minimum onset across 98 runs, clause 1 is expected to pass comfortably; if it does not, that discrepancy against the D7 prior is itself worth reporting, since D7 used an exact no-op edge mapper and this round does not.

- [ ] **Step 3: Recompute the cost projection**

The audit writes `cost_projection.json` from the observed onset times and the measured 94.5 s per simulated second: projected wall clock for Phase 2, Phase 3 and the spatial control. If Phase 2 alone projects beyond 4 h at the resolved worker count, the launcher prints the projection and **asks before starting Phase 2**.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit_topic4_zm_ictal_transition.py
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/{interictal_baseline_gate.json,zm_off_reference_baseline.json,cost_projection.json}
git commit -m "topic4 zm-itx: canary networks and the interictal-baseline gate"
```

---

### Task 15: Phase 1B — dose freeze, counterfactual attribution, repertoire gate, recruitment audit

**Files:**
- Modify: `scripts/audit_topic4_zm_ictal_transition.py` (add `--gate dose`, `--gate counterfactual`, `--gate repertoire`, `--gate recruitment`)

- [ ] **Step 1: Freeze the perturbation dose**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate dose \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Runs the perturbation worker at the **baseline** checkpoint only, 6 representative sites, 3 canary seeds — 18 units per rung — for each packet size in `[16, 32, 64, 128, 256]`. The in-window regime flags are produced automatically on every unit.

Selection: the **smallest** rung satisfying **all four**:

```
0 / 18 units with a probe-attributable detector-qualified event
0 / 18 units reaching the model ictal criterion
median descendant susceptibility over the 18 units >= 50 excess spikes
median response ratio to the next larger rung lies in [1.2, 3.0]
```

**Smallest, not largest.** The previous draft picked the largest baseline-safe rung on the argument that a bigger probe is better conditioned. That argument does not apply: this is not an inversion needing conditioning, and the largest baseline-safe rung is precisely the one most likely to leave the sub-event regime once the network becomes more excitable at the pre-ictal checkpoint — which would recycle the contamination the descendant metric was introduced to remove.

The ratio clause is the linearity check. The packet doubles between rungs, so a linear regime gives a ratio near 2; below 1.2 the probe is saturating and above 3.0 it is sitting near a threshold. Compute it against the next rung up even when that rung is itself rejected for igniting — the ratio is a property of the response curve, not a candidate.

The 50-spike floor is on **descendant** spikes and is therefore independent of the packet size; the original "≥ 200 total excess spikes" rule was satisfied outright by a 256-cell packet with zero recursive amplification.

If no rung satisfies all four, write `{"verdict": "NO_SUBEVENT_PROBE_REGIME"}` and **stop the round at Phase 1**. Do not loosen the ignition criterion, do not shrink the response window, do not drop the linearity clause, do not proceed with an igniting probe. That verdict is itself the finding: this work point admits no sub-ignition probe, so a finite-response susceptibility question cannot be posed here.

Writes `dose_freeze.json` with every rung's numbers and patches `candidate_manifest.json` with `perturbation.frozen_dose_cells`. The script refuses any `--label` other than `baseline`, so the dose can never be tuned on a pre-ictal or patient-derived quantity.

- [ ] **Step 2: Run the counterfactual attribution block**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate counterfactual \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Six branches × 6 representative sites × 3 canary seeds, 200 ms each, at the frozen dose:

| id | fast state | `z` | `m` |
|---|---|---|---|
| `native_baseline` | baseline | baseline | baseline |
| `native_pre_ictal` | pre-ictal | pre-ictal | pre-ictal |
| `reset_z` | pre-ictal | baseline | pre-ictal |
| `reset_m` | pre-ictal | pre-ictal | baseline |
| `reset_zm` | pre-ictal | baseline | baseline |
| `slow_only` | baseline | pre-ictal | pre-ictal |

Writes `counterfactual_attribution.json` with, per branch, the mean descendant susceptibility and its paired difference against `native_baseline`, plus the fraction of `native_pre_ictal − native_baseline` recovered by each reset. Every record carries `off_manifold: true` for the four spliced branches.

**Report language is fixed here:** a spliced state is a counterfactual attribution test, not a trajectory — the dynamics never visit "pre-ictal fast state with baseline `z`". **If this block is skipped or inconclusive, the permitted claim for the whole round degrades to "pre-ictal susceptibility on a Z/M-active trajectory" and may not name Z/M as the carrier.**

- [ ] **Step 3: Adjudicate the repertoire claim gate**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate repertoire \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Over **all returned events before onset**, with the frozen patient direction classifier from the manifest via `formal_mode_assignments` in `src/topic4_nlc_pathway_mechanism.py`, and `best_binary_alignment` from `src/topic4_d6_natural_kmeans.py` for the clustering match.

The decision rule is **conjunctive and fully specified**, because a list of measures without thresholds leaves the interpretation open once the figure exists:

```
INTERICTAL_REPERTOIRE_RETAINED = ALL of

  n_returned_events_before_onset       >= 20
  ood_fraction_returned                <= q95 over the 48 Z/M-off reference runs
  min(TA_like_count, TB_like_count)    >= 3
  balanced_alignment                   >= q05 over the 48 Z/M-off reference runs
```

Calibration, so none of these is arbitrary or unreachable:

- Z/M-off on this substrate yields **4.4 returned events per second** — median 88 per 20 s over the 12 `joint_04_control` seeds, range 78–97. With D7's ~8 s onset prior a healthy pre-onset window holds roughly 35 returned events, so 20 is a real but clearable bar.
- `>= 3` per mode is the repository's existing `fallback_events_per_mode` convention from the rev11 confirmation config, not a new number.
- `balanced_alignment` is the mean of the two per-mode recalls from `best_binary_alignment`; it is used rather than raw `purity` because purity can be inflated by a single dominant mode.
- Both reference quantiles are computed once from the 48 archived Z/M-off runs and cached to `zm_off_reference_repertoire.json` with its own hash; the fact that those runs use seeds 1561-1572 is printed wherever the gate is cited.

Conjunctive because "the repertoire is retained" means all of it. Per-clause results are always written, including which clause failed.

This is a **claim gate, not a run blocker**. It writes `INTERICTAL_REPERTOIRE_RETAINED: true|false` plus the four clause results into `repertoire_gate.json`, and the report's wording follows it: retained → *data-driven interictal modes → model ictal state*; not retained → *low-activity background → high-activity state*, with every mode statement dropped.

- [ ] **Step 4: Run the local-recruitment audit**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate recruitment \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Reads the canary runs' recruitment arrays and writes `recruitment_audit.json`: per network the recruited bin fraction, the 10 %→90 % spread duration, and the axial versus off-axial slopes. Reported descriptively — this is the measurement that distinguishes sequential local spread from near-simultaneous whole-field ignition, and it must appear in the report whichever way it comes out.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_topic4_zm_ictal_transition.py
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/{dose_freeze.json,counterfactual_attribution.json,repertoire_gate.json,recruitment_audit.json,candidate_manifest.json}
git commit -m "topic4 zm-itx: freeze a sub-ignition dose, attribute the change, gate the repertoire claim"
```

---

### Task 16: Phase 2 — the Joint arm and the primary endpoint

Only the Joint arm runs here. The four-arm latency comparison, the spatial control and every ignition continuation wait for the Phase 2 gate.

- [ ] **Step 1: Launch the 12 Joint trajectories**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase joint \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

12 jobs: `joint_04_control` × seeds 1811–1822, Z/M on, two-pass. Expected ~0.6 h at 8 workers with the D7 onset prior.

- [ ] **Step 2: Launch the uniform response maps**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase response-maps \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

For **every** transitioned Joint network, `--sites grid` (the frozen 7×7) at `--label baseline` and `--label pre_ictal`, 200 ms, **without** `--measure-onset-advance`. The in-window ignition flags are produced on every grid site regardless — they cost nothing and they are what stops a pre-ictal probe that ignites from being recorded as a larger finite response. All 12 networks get the same grid: a 6-point irregular site set cannot support a per-network spatial correlation or its 49-shift null, and cannot be pooled with 49-point maps as equal-precision samples.

Networks with `onset < 2500 ms` are excluded and listed in `perturbation_exclusions.json`. The `sensitivity` label (`onset − 1000 ms`) runs only where `onset >= 3500 ms`, as a robustness row.

- [ ] **Step 3: Verify completeness before aggregating**

```bash
ls results/topic4_sef_hfo/data_driven_zm_ictal_transition/workers/joint_04_control_seed_18*.json | wc -l
ls results/topic4_sef_hfo/data_driven_zm_ictal_transition/perturbation/*_grid_*.json | wc -l
```

Expected: 12 worker jsons and 2 grid jobs per transitioned network. Re-run any missing job individually; never aggregate over a partial set.

- [ ] **Step 4: Adjudicate the Phase 2 gate**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/aggregate_topic4_zm_ictal_transition.py --stage primary \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Computes E1's paired pre-minus-baseline difference over the retained networks with a 4096-draw network bootstrap and writes `primary_endpoint.json`. Only grid sites with `e1_evaluable = True` at **both** checkpoints enter the network mean; the excluded count, the pre-ictal ignition fraction and the `REGIME_LIMITED` flag are written alongside.

The rule is **three-way and directional**:

```
q05 >  0           pre-ictal susceptibility is higher  -> continue to Phase 3
q95 <  0           pre-ictal susceptibility is LOWER   -> STOP, report the opposite direction
q05 <= 0 <= q95    unresolved at n = 12                -> STOP, report as unresolved
```

The earlier draft said "the 90 % CI excludes 0 → continue", which would have launched five more hours of mechanism experiments on a **significantly negative** result, since a negative interval also excludes 0.

On either stop, the four-arm latency runs, the onset-advance continuations and the spatial re-registration control are **not** launched. Everything spent so far is ~3.3 h; everything saved is ~5 h. Neither stop may be written as "no effect": `q95 < 0` says the effect runs the other way, and a straddling interval says n = 12 could not tell.

If `regime_limited` is true — more than 25 % of pre-ictal grid sites ignited — the same three-way rule is applied, but the headline susceptibility statement in the report becomes the pre-ictal ignition fraction rather than the E1 difference, per the pre-registered switch.

- [ ] **Step 5: Commit**

```bash
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/{controller.log,perturbation_exclusions.json,primary_endpoint.json}
git commit -m "topic4 zm-itx: Joint-arm trajectories, uniform response maps, primary endpoint"
```

---

### Task 16b: Phase 3 — latency arms, ignition, and the re-registration control

**Run only if Task 16 Step 4 resolved the primary.**

- [ ] **Step 1: Launch the three latency arms**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase latency-arms \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

36 jobs: `node_baseline`, `joint_04_ee_only`, `joint_04_etoi_only` × seeds 1811–1822, Z/M on, **pass 1 only** — these arms carry latency and nothing else, so they need no onset-relative checkpoints.

- [ ] **Step 2: Launch the ignition measurements**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase ignition \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Representative sites only, `--measure-onset-advance`, at `baseline` and `pre_ictal`, on every transitioned Joint network. Because the dose was frozen to give 0/18 baseline ignitions, the informative reading is whether the same sub-threshold probe ignites at pre-ictal — and that reading comes from the in-window flags already collected everywhere, including on the grid. What this phase adds is the **long** arm: how much earlier the transition arrives. A baseline continuation allocates ~1.9 GB, so the launcher gives these jobs their own concurrency budget computed from the measured sentinel rather than the flat cap.

- [ ] **Step 3: Launch the spatial re-registration control**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/launch_topic4_zm_ictal_transition.py --phase reregistration \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

12 primary runs, `joint_04_control` with `--field-transform r180`, seeds 1811–1822 — **one transform on all seeds**, so the pooled comparison has a single interpretation. Then the frozen 7×7 grid at `baseline` and `pre_ictal` for every control network that transitioned. `r90` and `mx` are additionally run on the 3 canary seeds and reported descriptively only; **no claim is made that the seven non-identity elements were surveyed at power.**

- [ ] **Step 4: Build the observation control (no simulation)**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate observation-control \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

Reads the `contact_envelope_ctl_<element>` arrays already recorded inside each primary run (Task 11 Step 10) and recomputes the contact-level endpoints for each transformed montage. Writes `observation_control.json` including the per-element count of contacts that failed `cmrun.valid_mask` and were excluded. Labelled in its own output as answering **readout dependence only**; it may never support a mechanism conclusion.

- [ ] **Step 5: Verify completeness and commit**

```bash
ls results/topic4_sef_hfo/data_driven_zm_ictal_transition/perturbation/*.json | wc -l
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/{controller.log,observation_control.json}
git commit -m "topic4 zm-itx: latency arms, ignition endpoint, spatial re-registration and observation controls"
```

---

### Task 17: Aggregation and the pre-registered statistics

**Files:**
- Create: `scripts/aggregate_topic4_zm_ictal_transition.py`

**Interfaces:**
- Produces: `<output_root>/cohort_summary.json`, `cohort_summary.csv`, `primary_endpoint.json`, `spatial_endpoint.json`, `latency_endpoint.json`, `ignition_endpoint.json`, `control_comparison.json`, `state_characterization.json`, `mode_evolution.json`. Supports `--stage primary` for the Phase 2 gate and a full run afterwards.

- [ ] **Step 1: Implement the primary endpoint (E1)**

For every network: `susceptibility_pre - susceptibility_baseline`, each the mean **descendant** susceptibility over the 7×7 grid sites with `e1_evaluable = True` at **both** checkpoints. Report `paired_bootstrap(pre, baseline, draws=4096, seed=20260817)` with `n`, `mean_difference`, `q05/q50/q95` and `n_positive`, plus per network: the number of sites excluded, which checkpoint excluded them, and the pre-ictal ignition fraction. Set `regime_limited = pre_ictal_ignition_fraction > 0.25`.

The report must carry the bias direction verbatim: excluded sites are the ones that ignited, i.e. the largest responses, so the complete-case difference is **conservative for a positive claim and unsafe for a negative one**.

Report the `sensitivity` checkpoint the same way as a robustness row, clearly labelled. E2 numbers never enter this block.

- [ ] **Step 2: Report collinearity BEFORE any spatial interpretation**

```python
collinearity = covariate_collinearity(
    {"h": h_site, "ee_out_gain": ee_site, "etoi_out_gain": etoi_site})
```

`h`, the outgoing E→E gain and the outgoing E→I gain are all functions of the same field and are expected to be strongly collinear, which is exactly why **`h` is the primary spatial covariate by design** and the two gains are descriptive companions. The table is printed so a reader can see how little independent information the companions carry; it does **not** select anything. Three separate correlations are never written as three independent mechanisms, and there is no composite.

- [ ] **Step 3: Implement the primary spatial endpoint**

For each network, `spatial_correlation_exact_shift(susceptibility_field, covariate, grid_n=7)` for the two primary covariates — `h`, and **local recruitment time** from Task 8b — and for the two descriptive gains. Each covariate is averaged over the E neurons within 1.0 mm of each grid site so it lives on the same grid.

The null is enumerated, not sampled: all 49 toroidal shifts including the identity, so every per-network row carries `n_distinct_shifts = 49` and `p_floor = 1/49 ≈ 0.0204`, and the report states that floor rather than implying arbitrary precision. The load-bearing test is the cohort-level `paired_bootstrap` of the 12 per-network r values against zero. Also report `hotspot_compactness` at baseline and pre-ictal.

- [ ] **Step 4: Implement the attribution block**

From `counterfactual_attribution.json`: per branch, the mean descendant susceptibility and its paired difference against `native_baseline`, and the fraction of `native_pre_ictal − native_baseline` recovered by `reset_z`, `reset_m`, `reset_zm` and `slow_only`. Every spliced row carries `off_manifold: true`, and the block's header states that these are counterfactual attributions, not trajectories.

**The output field is named `carrier_candidate`, never `carrier`.** The block runs on three canary networks, so the permitted wording is "identifies a counterfactual carrier candidate consistent with the pre-ictal rise". Naming a carrier outright would require running the block on the full formal cohort, which this round does not do. If the block is absent, the aggregator writes `"zm_attribution": {"status": "NOT_ESTABLISHED"}` and the report may not name any variable at all.

- [ ] **Step 5: Implement the secondary endpoints**

**E2 ignition**: at the representative sites, the fraction of units with a probe-attributable event and with `reached_model_ictal`, at baseline versus pre-ictal, plus `onset_advance_ms` on the uncensored subset with the censored count printed alongside. Because the dose was frozen at 0/18 baseline ignitions, the headline is the pre-ictal ignition fraction of a probe that never ignited at baseline.

**Latency**: `restricted_ictal_free_time` per arm over `[0, 20000]` ms; `paired_onset_difference` of each non-Node arm against Node on the both-entered subset; the entered fraction per arm. The output states explicitly that 20 s is a censoring cap, not an onset, and that incidence is expected at ceiling given the 98/98 D7 prior.

- [ ] **Step 6: Implement the control comparison**

Paired, per network seed, data-driven Joint versus its `r180` image — **one transform across all 12 seeds**, so the pooled number has a single interpretation. Mechanism statements use **contact-independent** endpoints only: E1 change, the spatial correlations, hotspot compactness, restricted ictal-free time, local recruitment time and spread duration. Contact-dependent endpoints go in a separate block marked readout-dependent. `r90` and `mx` on the canary seeds are listed descriptively with an explicit "not surveyed at power" note.

The control is named **`matched spatial re-registration control`** everywhere. It is **not** an isometric copy of the substrate: the field-and-flow rule is transformed as a rigid unit, but the realized random graph, its patient-derived anisotropic topology and the contacts are not. `spatial_endpoint.json` and the report both carry that sentence.

If more than half the control runs did not transition, the paired susceptibility contrast is emitted as `{"status": "NOT_EVALUABLE", "n_transitioned": k}` rather than computed on the survivors.

- [ ] **Step 7: Implement the descriptive blocks**

State characterization aggregated across networks, always alongside the length-matched interictal reference and the `frequency_resolution_hz` / `n_cycles_at_band_low` caveats. Local recruitment: recruited bin fraction, 10 %→90 % spread duration, axial versus off-axial slopes — the numbers that separate sequential spread from simultaneous ignition. Mode 1 / Mode 2 share, KMeans match and OOD fraction at baseline versus the last 2 s before onset, over returned events only, via `formal_mode_assignments`; the count of returned events inside the last 2 s is printed next to every mode number, and the whole block is prefixed by the `INTERICTAL_REPERTOIRE_RETAINED` verdict that governs how it may be worded.

- [ ] **Step 8: Run and re-derive**

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/aggregate_topic4_zm_ictal_transition.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/audit_topic4_zm_ictal_transition.py --gate reported-numbers \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json
```

The `reported-numbers` gate recomputes every scalar in `cohort_summary.json` from the per-run artifacts and asserts equality to 1e-12; it fails loudly rather than warning. It additionally asserts that no E1 and E2 quantity has been averaged into a shared field, and that every `off_manifold` record is confined to the attribution block.

- [ ] **Step 9: Commit**

```bash
git add scripts/aggregate_topic4_zm_ictal_transition.py
git add -f results/topic4_sef_hfo/data_driven_zm_ictal_transition/{cohort_summary.json,cohort_summary.csv,primary_endpoint.json,spatial_endpoint.json,latency_endpoint.json,ignition_endpoint.json,control_comparison.json,state_characterization.json,mode_evolution.json}
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
C  h-WEIGHTED projected Z/M trajectory:
       x(t) = 1 - (sum_i h_i z_i) / (sum_i h_i)
       y(t) = eta_m * (sum_i h_i m_i) / (sum_i h_i)
   coloured by time; the unweighted population mean is a thin grey reference line on the
   same axes. Title exactly "Projected Z/M trajectory".
D  baseline and pre-ictal D - A fields on one shared colour scale, static h as contours
E  baseline and pre-ictal response fields for one fixed representative site,
   shared grid, dose and colour scale
F  baseline and pre-ictal descendant-susceptibility maps (uniform 7x7) plus the
   pre-minus-baseline difference
G  counterfactual attribution: mean descendant susceptibility for
   native_baseline / native_pre_ictal / reset_z / reset_m / reset_zm / slow_only,
   with the four spliced bars hatched and a legend entry reading
   "counterfactual state, not a trajectory"
H  local recruitment map and the 10-90 % spatial recruitment duration, with the axial
   versus off-axial slopes inset
I  data-driven substrate versus the r180 re-registration control, contact-independent
   endpoints only
J  Mode 1 / Mode 2 share, KMeans match and OOD fraction, baseline versus the last 2 s
```

The four-arm latency comparison and the observation control go to the supplement, not into
A–J: latency is secondary and the observation control answers readout dependence only.

Panel C must not use an unweighted population mean as the headline — the node field is 3.53 %
of the E population, so that curve mostly reports background.

No PASS/FAIL text, no internal status codes, no long explanatory text inside the axes. The readout panel's y-label says "virtual contact activity (firing-density envelope)", never voltage. Panel G's caption states that a spliced state is off-manifold. Panel B's caption states that the model ictal state is defined operationally by the runaway threshold.

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

Required content, in this order:

1. **E1**, the primary: the paired pre-minus-baseline **descendant** susceptibility with its 90 % interval and `n`, plus the statement that injected spikes are excluded from the count.
2. **Attribution**: what `reset_z` / `reset_m` / `reset_zm` / `slow_only` recovered, with the off-manifold caveat. If this block is `NOT_ESTABLISHED`, the report says only "pre-ictal susceptibility on a Z/M-active trajectory" and never names Z/M as the carrier.
3. **Primary spatial**: the collinearity table first, then one headline number for the substrate-structure family (or three, only if the covariates are not collinear), then local recruitment time separately; per-network `p_floor = 1/49` stated.
4. **Local recruitment**: recruited fraction, 10 %→90 % spread duration, axial vs off-axial slopes — the sequential-spread versus simultaneous-ignition reading, reported whichever way it comes out.
5. **Repertoire gate** verdict, which governs the wording of everything about modes.
6. **E2 ignition**, kept visually and numerically separate from E1.
7. **Latency**, as a restricted ictal-free time with the censoring cap named, alongside the D7 98/98 prior so a ceiling incidence is not read as a finding.
8. **Re-registration control**, split into contact-independent and contact-dependent blocks, named `matched spatial re-registration control`, with the explicit sentence that it is not an isometric copy of the substrate.
9. **State characterization** recomputed on this round's trajectories with the length-matched interictal reference and the resolution caveat.
10. Every exclusion count, and the dose-freeze table including the rungs that were rejected.

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
git add docs/archive/topic4/zm_ictal_transition/zm_itx_report_2026-08-17.md docs/topic4_sef_hfo.md
git add -f results/FIGURE_INDEX.md results/topic4_sef_hfo/data_driven_zm_ictal_transition/DONE.json
git commit -m "topic4 zm-itx: report the data-driven Z/M interictal-to-ictal transition round"
```

---

## Cost model

The measured unit is **94.5 s of wall clock per simulated second**, from the archived
seed-1561 run (1890.7 s for 20 s, single-threaded). The working onset prior is D7's median
**~8 s** (98/98 runaway, q05 6.5 s, q95 10.0 s); Task 14 Step 3 recomputes the table from the
observed canary onsets before Phase 2 launches.

| Phase | Work | Wall clock at 8 workers | Cumulative |
|---|---|---|---|
| Tasks 1–13 gates | 1 network build + Gate A parity run + unit tests | ~1.2 h | 1.2 h |
| 1A canary | 3 Joint (2-pass) + 3 Z/M-off | ~0.4 h | 1.6 h |
| 1B dose + attribution | 90 dose probes + 108 counterfactual probes, all 200 ms | ~0.3 h | 1.9 h |
| 2 Joint trajectories | 12 runs, 2-pass | ~0.6 h | 2.5 h |
| 2 response maps | 12 networks × 2 labels × 49 sites = 1176 probes at 200 ms | ~0.8 h | **3.3 h ← the gate** |
| 3 latency arms | 36 runs, pass 1 only | ~1.0 h | 4.3 h |
| 3 ignition | 144 continuations at ~6 s / ~1.5 s | ~1.8 h | 6.1 h |
| 3 re-registration control | 12 runs + 1176 grid probes | ~1.4 h | 7.5 h |
| Observation control | 0 simulations | minutes | 7.5 h |
| Aggregation + figures | — | ~1 h | **~8.5 h** |

Two structural savings against the first draft, both from the review: the 7×7 grid now carries
E1 only — no onset-advance continuation at any grid point — and the three latency arms run pass 1
only. Together those removed roughly 6 h. **Only the first ~3.3 h runs before the Phase 2 gate
decides whether the remaining ~5 h happens at all.**

Memory: 14.6 GiB per 20 s run. A 200 ms probe continuation holds only 64 MB of spike buffer; a
~6 s baseline ignition continuation holds ~1.9 GB, so those jobs get their own concurrency
budget rather than the flat cap.

Disk: network cache (16 pickles, size measured in Task 1 Step 7), checkpoints ≈ 130 MB × up to
36 ≈ 4.7 GB, per-probe aggregate fields ≈ 51 MB (E1 grid) plus the counterfactual and ignition
sets, exemplar raw traces ≤ 400 MB. Against 187 GiB free.

## Self-review

**Spec coverage.** Engine changes → Tasks 3, 4, 6. Checkpoint contents → Task 2. Gates A/B/C →
Tasks 11, 4, 5. Code layout → Tasks 1, 11, 12, 13. E1 descendant metric and the injected-frame
regression → Task 9. E2 probe-attributable ignition → Task 9. Dose freeze and
`NO_SUBEVENT_PROBE_REGIME` → Task 15 Step 1. Counterfactual attribution → Tasks 9, 12, 15 Step 2,
17 Step 4, panel G. Uniform 7×7 sampling on all 12 networks → Tasks 9, 16 Step 2. Collinearity →
Tasks 10, 17 Step 2. Local recruitment replacing first-spike density → Task 8b, panels H.
`INTERICTAL_REPERTOIRE_RETAINED` → Task 15 Step 3. Interictal-baseline gate with the ≥2/3 rule →
Task 14. Phase gating with the Phase 2 stop rule → Tasks 16, 16b. `r180`-only re-registration
control and its naming → Tasks 7, 16b Step 3, 17 Step 6. Observation control recorded in-run →
Tasks 11 Step 10, 16b Step 4. `h`-weighted trajectory → Tasks 11 Step 11, 18 panel C. Endpoint
tiers and the two orthogonal axes → Task 17. Censored latency → Tasks 10, 17 Step 5.
Slow-current product averages → Task 6. State characterization → Tasks 8, 17 Step 7. Figure
contract → Task 18. Claim boundary → Task 19. Execution discipline and `git add -f` → Task 13
and the Global Constraints.

**Deliberate deviations from the spec, recorded rather than silently absorbed:**
- The spec's `results/paper-ready-figure/fig5/README.md` is written as `figures/README.md` in
  Task 18, matching the repository standard that the README lives beside the figures.
- The spec lists six counterfactual branches; Task 15 Step 2 runs all six but only four are
  spliced, since `native_baseline` and `native_pre_ictal` are the unmodified checkpoints
  already produced by Task 16 and are re-used rather than re-simulated.

**Type consistency.** `build_substrate` returns `Substrate` in Tasks 1, 7, 9, 11, 12.
`capture` / `restore_slow` / `restore_external_drive` keep their Task 2 signatures in Tasks 4, 5,
6, 12, and `capture`'s drive keyword is `external_drive=` everywhere. `absolute_time_ms` is the
checkpoint clock field in Tasks 2, 4, 12. `susceptibility` means the same descendant scalar in
Tasks 9, 15, 16, 17, 18 and never appears in an E2 context. `field_transform` takes a D4 element
name in Tasks 1, 7, 11, 16b. `frozen_sites(kind=...)` takes `"grid"` or `"representative"` in
Tasks 9, 12, 15, 16, 16b. `splice_checkpoint(mode=...)` takes the same six mode strings in Tasks
9, 12, 15, 17.

**Contamination guards that must not be relaxed.** These tests encode findings that cost review
rounds to surface; weakening any of them re-opens the corresponding contamination:

| Test | Guards against |
|---|---|
| `test_injected_spikes_alone_produce_exactly_zero_susceptibility` (Task 9) | the packet's own spikes being scored as response |
| `test_ignition_requires_the_event_to_be_absent_from_the_sham` (Task 9) | reading one of the network's own 41 %-of-the-time events as a probe effect |
| `test_splice_leaves_every_non_slow_field_bit_identical` (Task 9) | a counterfactual that silently perturbs more than `z`/`m` |
| `test_a_brief_blip_shorter_than_the_persistence_floor_is_not_recruitment` (Task 8b) | background fluctuation counted as recruitment |
| `test_symmetric_offaxis_spread_is_not_cancelled_by_a_signed_coordinate` (Task 8b) | symmetric off-axis spread averaging to "no spread" |
| `test_exact_shift_null_reports_49_shifts_and_a_1_over_49_floor` (Task 10) | a sampled null implying precision the 7×7 grid does not have |

**Pre-registered decision rules, all fixed before any run.** Dose: smallest rung, 0/18 on both
ignition tests, ≥ 50 descendant spikes, ratio in `[1.2, 3.0]`. Phase 1A: ≥ 2 of 3 networks.
Phase 2: `q05 > 0` continue, `q95 < 0` stop-opposite, straddling stop-unresolved. Regime switch:
pre-ictal grid ignition fraction > 0.25 → `REGIME_LIMITED`. Repertoire: four conjunctive clauses
with the reference quantiles cached. Primary spatial covariates: `h` and local recruitment time,
chosen by design and not by any data-dependent rule.

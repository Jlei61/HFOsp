"""Integration tests for scripts/run_sef_itp_phase1.py:load_subject_for_phase1.

Mocks the two upstream loaders (load_subject_propagation_events,
load_subject_coords) and writes small JSON fixtures to tmp_path, then verifies
the contract clauses listed in load_subject_for_phase1 docstring:

  - channel alignment between Phase 0a and lagPat loader
  - PR-6 endpoint name → index conversion
  - coord_units mm assertion (voxel rejected when require_coords=True)
  - valid_indices = mapped non-endpoint (v1.0.7; was PR-6 valid_mask intersection)
  - endpoint S/K drop when mapped_mask=False, with n_dropped audit
  - H2 swap_check ingest from PR-6 (v1.0.8; was self-implemented set/spatial reversal)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_sef_itp_phase1 as runner
from src.seeg_coord_loader import BipolarRes, CoordResult


CHANNELS = ["A1-A2", "A2-A3", "A3-A4", "B1-B2", "B2-B3", "B3-B4"]


def _make_phase0a_json(channels, clusters_meta, pairs):
    return {
        "dataset": "epilepsiae",
        "subject": "1073",
        "n_channels": len(channels),
        "n_events_total": 1000,
        "n_blocks_used": 1,
        "channel_names": list(channels),
        "adaptive_cluster": {
            "stable_k": 2,
            "clusters": clusters_meta,
            "candidate_forward_reverse_pairs": pairs,
        },
    }


def _make_pr6_json(channels, templates, h2_swap=None):
    return {
        "subject_id": "1073",
        "dataset": "epilepsiae",
        "per_template": templates,
        "h2_swap_check": h2_swap if h2_swap is not None else {
            "swap_score": 0.8,
            "null_p": 0.01,
            "null_95th": 0.5,
            "null_median": 0.3,
            "n_perm": 1000,
            "exit_reason": None,
            "jaccard_t0src_t1snk": 0.8,
            "jaccard_t0snk_t1src": 0.8,
        },
    }


def _make_fake_loader_dict(channels, n_events=100):
    rng = np.random.default_rng(0)
    n_ch = len(channels)
    bools = rng.uniform(0, 1, size=(n_ch, n_events)) < 0.5
    return {
        "bools": bools,
        "channel_names": list(channels),
        "ranks": np.zeros((n_ch, n_events)),
        "lag_raw": np.zeros((n_ch, n_events)),
        "event_abs_times": np.arange(n_events, dtype=float),
        "event_rel_times": np.arange(n_events, dtype=float),
        "block_ids": np.zeros(n_events, dtype=int),
        "record_names": ["fake_rec"],
        "block_boundaries": [],
        "block_start_times": np.array([0.0]),
        "block_time_ranges": [(0.0, float(n_events))],
        "n_blocks_total": 1,
        "n_blocks_used": 1,
    }


def _make_coord_result(channels, mapped_mask=None, units="mm", space="mni152_1mm"):
    n = len(channels)
    if mapped_mask is None:
        mapped_mask = np.ones(n, dtype=bool)
    coords = np.full((n, 3), np.nan, dtype=float)
    for i, mapped in enumerate(mapped_mask):
        if mapped:
            coords[i] = [float(i), 0.0, 0.0]
    return CoordResult(
        schema_version="coord_loader_v3",
        dataset="epilepsiae",
        subject_id="107302",
        channel_names_requested=list(channels),
        coords_array_in_requested_order=coords,
        mapped_mask_in_requested_order=np.asarray(mapped_mask, dtype=bool),
        coord_space=space,
        coord_units=units,
        provenance={
            "source_path": "<fake>",
            "affine_path": "<fake>",
            "loader_version": "test",
            "canonical_subject_id": "107302",
            "input_subject_id": "1073",
        },
        missing=[],
        bipolar_resolution={},
        source_coord_type="sql_voxel_ijk",
        normalization_certainty="grid_confirmed_warp_type_unverified",
    )


@pytest.fixture
def phase0a_path(tmp_path):
    clusters = [
        {"cluster_id": 0, "n_events": 500, "fraction": 0.5,
         "template_rank": list(range(len(CHANNELS)))},
        {"cluster_id": 1, "n_events": 500, "fraction": 0.5,
         "template_rank": list(reversed(range(len(CHANNELS))))},
    ]
    pairs = [{"cluster_a": 0, "cluster_b": 1, "spearman_r": -0.95,
              "label": "candidate_forward_reverse"}]
    js = _make_phase0a_json(CHANNELS, clusters, pairs)
    p = tmp_path / "phase0a" / "epilepsiae_1073.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(js))
    return p


@pytest.fixture
def pr6_root(tmp_path):
    templates = [
        {"cluster_id": 0,
         "source": ["A1-A2", "A2-A3"],
         "sink": ["B2-B3", "B3-B4"],
         "endpoint": ["A1-A2", "A2-A3", "B2-B3", "B3-B4"],
         "middle": ["A3-A4", "B1-B2"],
         "valid_mask": [True] * len(CHANNELS),
         "n_valid_channels": len(CHANNELS),
         "n_valid": len(CHANNELS)},
        {"cluster_id": 1,
         "source": ["B2-B3", "B3-B4"],
         "sink": ["A1-A2", "A2-A3"],
         "endpoint": ["A1-A2", "A2-A3", "B2-B3", "B3-B4"],
         "middle": ["A3-A4", "B1-B2"],
         "valid_mask": [True] * len(CHANNELS),
         "n_valid_channels": len(CHANNELS),
         "n_valid": len(CHANNELS)},
    ]
    js = _make_pr6_json(CHANNELS, templates)
    p = tmp_path / "pr6"
    p.mkdir()
    (p / "epilepsiae_1073.json").write_text(json.dumps(js))
    return p


def test_load_subject_for_phase1_happy_path(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """Happy path: all 4 sources align, mm coords, unified namespace with no extras.

    v1.0.7: valid_indices_per_cluster is now "all mapped SEEG MINUS endpoints"
    (not PR-6 valid_mask intersection).
    """
    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))
    # All-SEEG enumerator returns no extra channels — unified == CHANNELS
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )

    assert subj.subject_id == "1073"
    assert subj.dataset == "epilepsiae"
    assert subj.channel_names == CHANNELS
    assert subj.n_lagpat_channels == len(CHANNELS)
    assert subj.coord_units == "mm"
    assert subj.coord_space == "mni152_1mm"
    assert subj.events_bool.shape[0] == len(CHANNELS)

    # Endpoint name → index conversion
    assert subj.cluster_endpoints[0]["S"] == [0, 1]       # A1-A2, A2-A3
    assert subj.cluster_endpoints[0]["K"] == [4, 5]       # B2-B3, B3-B4
    assert subj.cluster_endpoints[1]["S"] == [4, 5]       # swap for reverse
    assert subj.cluster_endpoints[1]["K"] == [0, 1]

    # v1.0.7: valid_indices = mapped channels MINUS endpoints
    # endpoint cluster 0 = {0,1,4,5}; mapped = all → valid = [2, 3]
    assert subj.valid_indices_per_cluster[0] == [2, 3]
    assert subj.valid_indices_per_cluster[1] == [2, 3]

    # v1.0.8: H2 swap_check ingested from PR-6, not recomputed
    assert subj.h2_swap_check is not None
    assert subj.h2_swap_check["swap_score"] == 0.8
    assert subj.h2_swap_check["null_p"] == 0.01
    assert subj.h2_swap_check["exceeds_null_95th"] is True
    assert subj.h2_swap_check["source_contract"] == "pr6_h2_swap_check"

    # No drops
    assert subj.n_dropped_endpoints_no_coords_per_cluster == {0: 0, 1: 0}


def test_load_subject_for_phase1_unified_namespace_extends_pool(
    monkeypatch, phase0a_path, pr6_root, tmp_path,
):
    """v1.0.7: when all-SEEG enumerator returns extra channels, valid_indices
    expands to include those extras (with coords); endpoint indices stay in
    the lagPat front-portion of the unified namespace."""
    EXTRAS = ["X1-X2", "X2-X3", "Y1-Y2"]
    unified = list(CHANNELS) + EXTRAS

    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(unified))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: EXTRAS)

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )

    assert subj.channel_names == unified
    assert subj.n_lagpat_channels == len(CHANNELS)
    assert subj.events_bool.shape[0] == len(unified)
    # Non-lagPat rows are all False
    assert subj.events_bool[len(CHANNELS):].sum() == 0
    # Endpoint indices stay in front (lagPat region)
    assert subj.cluster_endpoints[0]["S"] == [0, 1]
    assert subj.cluster_endpoints[0]["K"] == [4, 5]
    # valid_indices_per_cluster expands to include the all-SEEG extras
    # endpoint c0 = {0,1,4,5}; mapped = all 9; valid = [2,3,6,7,8]
    assert subj.valid_indices_per_cluster[0] == [2, 3, 6, 7, 8]


def test_load_subject_for_phase1_rejects_channel_mismatch(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """lagPat returns different channel ordering → raises."""
    shuffled = CHANNELS[::-1]
    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(shuffled))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="Channel-name mismatch"):
        runner.load_subject_for_phase1(
            phase0a_path,
            pr6_anchoring_root=pr6_root,
            epilepsiae_lagpat_root=tmp_path / "lagpat_root",
        )


def test_load_subject_for_phase1_rejects_voxel_coords(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """coord_units=voxel + require_coords=True → mm assertion raises."""
    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS, units="voxel",
                                                       space="mri_native_voxel_ijk"))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="coord_units='mm'"):
        runner.load_subject_for_phase1(
            phase0a_path,
            pr6_anchoring_root=pr6_root,
            epilepsiae_lagpat_root=tmp_path / "lagpat_root",
            require_coords=True,
        )


def test_load_subject_for_phase1_drops_unmapped_endpoints(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """When coord mapped_mask=False on some endpoint channels, they are dropped + audited."""
    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))

    # mask out A2-A3 (idx 1) and B3-B4 (idx 5) → each endpoint set loses one channel
    mapped_mask = np.array([True, False, True, True, True, False])
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS, mapped_mask=mapped_mask))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )

    # cluster 0: S=[A1-A2(0), A2-A3(1)] → drop idx 1 → [0]
    #            K=[B2-B3(4), B3-B4(5)] → drop idx 5 → [4]
    assert subj.cluster_endpoints[0]["S"] == [0]
    assert subj.cluster_endpoints[0]["K"] == [4]
    # cluster 1 (swap): S=[4,5]→[4], K=[0,1]→[0]
    assert subj.cluster_endpoints[1]["S"] == [4]
    assert subj.cluster_endpoints[1]["K"] == [0]

    # n_dropped audit (1 dropped from S + 1 dropped from K per cluster)
    assert subj.n_dropped_endpoints_no_coords_per_cluster == {0: 2, 1: 2}

    # v1.0.7: valid_indices = mapped channels MINUS endpoints
    # mapped = [0,2,3,4]; endpoint c0 = {0,4} → valid = [2,3]
    assert subj.valid_indices_per_cluster[0] == [2, 3]
    assert subj.valid_indices_per_cluster[1] == [2, 3]


def test_load_subject_for_phase1_rejects_unknown_endpoint_name(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """PR-6 endpoint name not in unified channel_names → raises (no silent index swap)."""
    bad_templates = [
        {"cluster_id": 0,
         "source": ["A1-A2", "Z9-Z10"],  # Z9-Z10 not in CHANNELS or all-SEEG enumerator
         "sink": ["B2-B3"],
         "endpoint": [], "middle": [],
         "valid_mask": [True] * len(CHANNELS),
         "n_valid_channels": len(CHANNELS), "n_valid": len(CHANNELS)},
    ]
    (pr6_root / "epilepsiae_1073.json").write_text(
        json.dumps(_make_pr6_json(CHANNELS, bad_templates))
    )

    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="Z9-Z10"):
        runner.load_subject_for_phase1(
            phase0a_path,
            pr6_anchoring_root=pr6_root,
            epilepsiae_lagpat_root=tmp_path / "lagpat_root",
        )


def test_load_subject_for_phase1_pr6_valid_mask_no_longer_constrains_pool(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """v1.0.7: PR-6 valid_mask is no longer used to constrain valid_indices.

    Previously v1.0.6 took intersection (PR-6 valid_mask ∩ mapped_mask).
    v1.0.7 dropped this because PR-6 valid_mask = "lagPat-participating",
    which created a circular null pool. valid_indices is now strictly
    "all mapped SEEG channels MINUS endpoint" — independent of PR-6 mask."""
    # PR-6 reports only idx 0..3 valid, but with all 6 channels mapped,
    # valid_indices should ignore the PR-6 mask and use mapped non-endpoint.
    partial_templates = [
        {"cluster_id": 0,
         "source": ["A1-A2", "A2-A3"], "sink": ["A3-A4"],
         "endpoint": [], "middle": [],
         "valid_mask": [True, True, True, True, False, False],
         "n_valid_channels": 4, "n_valid": 4},
        {"cluster_id": 1,
         "source": ["A3-A4"], "sink": ["A1-A2", "A2-A3"],
         "endpoint": [], "middle": [],
         "valid_mask": [True, True, True, True, False, False],
         "n_valid_channels": 4, "n_valid": 4},
    ]
    (pr6_root / "epilepsiae_1073.json").write_text(
        json.dumps(_make_pr6_json(CHANNELS, partial_templates))
    )

    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )
    # c0 endpoint = {0,1,2}; mapped = all → valid = [3,4,5] (PR-6 valid_mask IGNORED)
    assert subj.valid_indices_per_cluster[0] == [3, 4, 5]
    # c1 endpoint = {0,1,2} (S=[2], K=[0,1]); mapped = all → valid = [3,4,5]
    assert subj.valid_indices_per_cluster[1] == [3, 4, 5]


def test_load_subject_for_phase1_h2_swap_check_unavailable_when_pr6_missing(
    monkeypatch, phase0a_path, pr6_root, tmp_path
):
    """When PR-6 h2_swap_check.exit_reason is non-None, h2_swap_check should be None.

    v1.0.8: H2 is mechanism sanity ingested from PR-6. If PR-6 couldn't compute
    it (exit_reason set), Phase 1 propagates as h2_swap_check=None — caller
    treats subject as "H2 not testable" for cohort sign-test."""
    single = [
        {"cluster_id": 0,
         "source": ["A1-A2"], "sink": ["B2-B3"],
         "endpoint": [], "middle": [],
         "valid_mask": [True] * len(CHANNELS),
         "n_valid_channels": len(CHANNELS), "n_valid": len(CHANNELS)},
    ]
    (pr6_root / "epilepsiae_1073.json").write_text(
        json.dumps(_make_pr6_json(CHANNELS, single, h2_swap={
            "swap_score": None, "null_p": None, "null_95th": None,
            "exit_reason": "n_valid_too_small", "n_perm": 0,
        }))
    )

    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))
    monkeypatch.setattr(runner, "enumerate_subject_all_channels",
                        lambda ds, sid: list(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )
    assert subj.h2_swap_check is None
    # cluster endpoints still populated
    assert 0 in subj.cluster_endpoints


def test_load_subject_for_phase1_subject_filename_mismatch_raises(tmp_path, pr6_root):
    """Phase 0a JSON content subject != filename subject → raises."""
    bad = {
        "dataset": "epilepsiae", "subject": "9999",  # mismatch
        "n_channels": len(CHANNELS), "n_events_total": 100,
        "channel_names": CHANNELS,
        "adaptive_cluster": {
            "clusters": [], "candidate_forward_reverse_pairs": []},
    }
    p = tmp_path / "phase0a" / "epilepsiae_1073.json"
    p.parent.mkdir()
    p.write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="subject"):
        runner.load_subject_for_phase1(p, pr6_anchoring_root=pr6_root)

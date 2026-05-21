"""Integration tests for scripts/run_sef_itp_phase1.py:load_subject_for_phase1.

Mocks the two upstream loaders (load_subject_propagation_events,
load_subject_coords) and writes small JSON fixtures to tmp_path, then verifies
the contract clauses listed in load_subject_for_phase1 docstring:

  - channel alignment between Phase 0a and lagPat loader
  - PR-6 endpoint name → index conversion
  - coord_units mm assertion (voxel rejected when require_coords=True)
  - valid_indices = PR-6 valid_mask ∩ coord mapped_mask
  - endpoint S/K drop when mapped_mask=False, with n_dropped audit
  - forward_reverse_pairs schema translation (cluster_a → cluster_A_id)
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


def _make_pr6_json(channels, templates):
    return {
        "subject_id": "1073",
        "dataset": "epilepsiae",
        "per_template": templates,
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
    """Happy path: all 4 sources align, mm coords, full valid pool."""
    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))

    # fake lagpat dir
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
    assert subj.coord_units == "mm"
    assert subj.coord_space == "mni152_1mm"
    assert subj.events_bool.shape[0] == len(CHANNELS)

    # Endpoint name → index conversion
    assert subj.cluster_endpoints[0]["S"] == [0, 1]       # A1-A2, A2-A3
    assert subj.cluster_endpoints[0]["K"] == [4, 5]       # B2-B3, B3-B4
    assert subj.cluster_endpoints[1]["S"] == [4, 5]       # swap for reverse
    assert subj.cluster_endpoints[1]["K"] == [0, 1]

    # Valid indices = full pool (all mapped, all valid)
    assert subj.valid_indices_per_cluster[0] == list(range(len(CHANNELS)))
    assert subj.valid_indices_per_cluster[1] == list(range(len(CHANNELS)))

    # Forward-reverse pair schema translation
    assert len(subj.forward_reverse_pairs) == 1
    pair = subj.forward_reverse_pairs[0]
    assert pair["cluster_A_id"] == 0
    assert pair["cluster_B_id"] == 1
    assert pair["reproducibility_source"] == "candidate_forward_reverse"

    # No drops
    assert subj.n_dropped_endpoints_no_coords_per_cluster == {0: 0, 1: 0}


def test_load_subject_for_phase1_rejects_channel_mismatch(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """lagPat returns different channel ordering → raises."""
    shuffled = CHANNELS[::-1]
    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(shuffled))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))

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

    # valid_indices = mask ∩ mapped = idx where both True
    assert subj.valid_indices_per_cluster[0] == [0, 2, 3, 4]
    assert subj.valid_indices_per_cluster[1] == [0, 2, 3, 4]


def test_load_subject_for_phase1_rejects_unknown_endpoint_name(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """PR-6 endpoint name not in Phase 0a channel_names → raises (no silent index swap)."""
    bad_templates = [
        {"cluster_id": 0,
         "source": ["A1-A2", "Z9-Z10"],  # Z9-Z10 not in CHANNELS
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

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="Z9-Z10"):
        runner.load_subject_for_phase1(
            phase0a_path,
            pr6_anchoring_root=pr6_root,
            epilepsiae_lagpat_root=tmp_path / "lagpat_root",
        )


def test_load_subject_for_phase1_pr6_valid_mask_intersection(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """When PR-6 valid_mask is partial, valid_indices reflects intersection with mapped."""
    # PR-6 says only idx 0..3 participate; coord loader maps all
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

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )
    assert subj.valid_indices_per_cluster[0] == [0, 1, 2, 3]
    assert subj.valid_indices_per_cluster[1] == [0, 1, 2, 3]


def test_load_subject_for_phase1_no_pr6_pair_skipped(monkeypatch, phase0a_path, pr6_root, tmp_path):
    """Forward-reverse pair with cluster missing from PR-6 → pair filtered out."""
    # PR-6 has only cluster 0
    single = [
        {"cluster_id": 0,
         "source": ["A1-A2"], "sink": ["B2-B3"],
         "endpoint": [], "middle": [],
         "valid_mask": [True] * len(CHANNELS),
         "n_valid_channels": len(CHANNELS), "n_valid": len(CHANNELS)},
    ]
    (pr6_root / "epilepsiae_1073.json").write_text(
        json.dumps(_make_pr6_json(CHANNELS, single))
    )

    monkeypatch.setattr(runner, "load_subject_propagation_events",
                        lambda d: _make_fake_loader_dict(CHANNELS))
    monkeypatch.setattr(runner, "load_subject_coords",
                        lambda **kw: _make_coord_result(CHANNELS))

    fake_dir = tmp_path / "lagpat_root" / "1073" / "all_recs"
    fake_dir.mkdir(parents=True)

    subj = runner.load_subject_for_phase1(
        phase0a_path,
        pr6_anchoring_root=pr6_root,
        epilepsiae_lagpat_root=tmp_path / "lagpat_root",
    )
    # Phase 0a says pair (0,1) but PR-6 has no cluster 1 → pair filtered
    assert subj.forward_reverse_pairs == []
    assert 0 in subj.cluster_endpoints
    assert 1 not in subj.cluster_endpoints


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

"""Tests for scripts/augment_topic5_r2b_3d.py — R2b native-3D + R2_nm (2D no-mirror)
on the common coord-mapped subset (Task 3 of the R2b-3D sensitivity plan).

Fixture-based where possible (monkeypatch _ctx + load_subject_coords); one real-subject
smoke gated on data presence. See docs/superpowers/plans/2026-07-01-topic5-r2b-3d-sensitivity.md.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.augment_topic5_r2b_3d as aug
from src.seeg_coord_loader import CoordResult
from src.topic5_contact_similarity import median_nn_spacing

MAIN = "/home/honglab/leijiaxin/HFOsp/results"


# --------------------------------------------------------------------------- helpers

def _make_ctx(names, source_pts, sigma, sz_vals, *, rank_b=True, seed=0):
    """Minimal synthetic _ctx dict (only the keys augment_subject reads)."""
    n = len(names)
    rng = np.random.default_rng(seed)
    rank_a = list(np.arange(n, dtype=float))
    rb = list(np.arange(n, dtype=float)[::-1]) if rank_b else None
    return {
        "names_m": list(names),
        "rank_a": rank_a,
        "rank_b": rb,
        "source_pts": np.asarray(source_pts, float),
        "support": np.ones(n, float),
        "sigma": float(sigma),
        "sz_vals": sz_vals,
    }


def _make_cr(names, coords_array, *, coord_units="mm", coord_space="mni152_1mm"):
    """Minimal CoordResult aligned to `names`. mask = all-finite rows."""
    coords = np.asarray(coords_array, float)
    mask = np.isfinite(coords).all(axis=1)
    return CoordResult(
        schema_version="coord_loader_v3",
        dataset="epilepsiae",
        subject_id="synthetic",
        channel_names_requested=list(names),
        coords_array_in_requested_order=coords,
        mapped_mask_in_requested_order=mask,
        coord_space=coord_space,
        coord_units=coord_units,
        provenance={},
    )


# --------------------------------------------------------------------------- (a) NA_insufficient

def test_common_subset_na_insufficient(monkeypatch):
    """Dropping no-coord channels leaves n_common<6 -> r2b_status='NA_insufficient'."""
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]  # 8 matched, 2 shafts
    rng = np.random.default_rng(1)
    src = rng.random((8, 2))
    sz = {i: rng.random(8) for i in range(3)}
    ctx = _make_ctx(names, src, median_nn_spacing(src), sz)

    # coords for only 5 channels (rest NaN) -> n_common = 5 < 6
    coords = np.full((8, 3), np.nan)
    coords[:5] = rng.random((5, 3)) * 10.0
    cr = _make_cr(names, coords)

    monkeypatch.setattr(aug, "_ctx", lambda *a, **k: ctx)
    monkeypatch.setattr(aug, "load_subject_coords", lambda *a, **k: cr)

    out = aug.augment_subject("epilepsiae_synthetic", activation="broadband",
                              B=20, input_results_root="/nonexistent")
    assert out["r2b_status"] == "NA_insufficient"
    assert out["n_common"] == 5
    assert out["n_matched_2d"] == 8


# --------------------------------------------------------------------------- (b) NA_units

def test_units_gate_na_units(monkeypatch):
    """load_subject_coords returns coord_units='voxel' -> assert raises -> NA_units, no crash."""
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    rng = np.random.default_rng(2)
    src = rng.random((8, 2))
    sz = {i: rng.random(8) for i in range(3)}
    ctx = _make_ctx(names, src, median_nn_spacing(src), sz)

    coords = rng.random((8, 3)) * 10.0
    cr = _make_cr(names, coords, coord_units="voxel", coord_space="mri_native_voxel_ijk")

    monkeypatch.setattr(aug, "_ctx", lambda *a, **k: ctx)
    monkeypatch.setattr(aug, "load_subject_coords", lambda *a, **k: cr)

    out = aug.augment_subject("epilepsiae_synthetic", activation="broadband",
                              B=20, input_results_root="/nonexistent")
    assert out["r2b_status"] == "NA_units"


# --------------------------------------------------------------------------- (c) degenerate: R2b==R2_nm

def test_degenerate_r2b_equals_r2nm(monkeypatch):
    """3D coords = 2D plane embedded at z=0 with sigma matched -> R2b == R2_nm exactly.

    sigma_xy (frozen) == median 2D NN spacing of plane pts; median_nn_spacing of
    [x,y,0] == that same value, and z-column adds 0 to every squared distance.
    So the only difference (coordinate space) collapses -> identical stat -> delta 0.
    """
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    rng = np.random.default_rng(3)
    src = rng.random((8, 2))
    sigma = median_nn_spacing(src)     # matches what median_nn_spacing([x,y,0]) returns
    sz = {i: rng.random(8) for i in range(4)}
    ctx = _make_ctx(names, src, sigma, sz)

    coords3d = np.column_stack([src, np.zeros(8)])   # plane at z=0
    cr = _make_cr(names, coords3d)

    monkeypatch.setattr(aug, "_ctx", lambda *a, **k: ctx)
    monkeypatch.setattr(aug, "load_subject_coords", lambda *a, **k: cr)

    out = aug.augment_subject("epilepsiae_synthetic", activation="broadband",
                              B=30, input_results_root="/nonexistent")
    assert out["r2b_status"] == "ok"
    assert np.isfinite(out["r2b_minus_r2nm"])
    assert np.isclose(out["R2b"]["obs_subject"], out["R2_nm"]["obs_subject"], atol=1e-9)
    assert np.isclose(out["r2b_minus_r2nm"], 0.0, atol=1e-9)


# --------------------------------------------------------------------------- (d) real-subject smoke

@pytest.mark.skipif(
    not Path(MAIN, "topic5_ictal_recruitment", "t0_feature_cache",
             "epilepsiae_1146.npz").exists(),
    reason="real T0 cache / axis records absent",
)
def test_real_subject_smoke_1146():
    out = aug.augment_subject("epilepsiae_1146", activation="broadband",
                              B=20, input_results_root=MAIN)
    assert out["r2b_status"] == "ok"
    assert out["coord_units"] == "mm"
    assert np.isfinite(out["r2b_minus_r2nm"])
    assert out["n_common"] >= 6
    assert out["n_shafts_common"] >= 2

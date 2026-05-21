"""Unit tests for SEEG coord loader v2 strict schema.

Plan: docs/archive/topic4/sef_itp_phase1/coord_loader_plan_2026-05-21.md
Module: src/seeg_coord_loader.py

All tests use synthetic data (in-memory or tmp_path fixtures). NO real cohort
data — real-data integration tests are gated on user wiring it up.

Coverage map (each v2 invariant has ≥ 1 test):
  Inv #1 NULL → explicit missing + NaN row + mask False    → test_sql_null_*
  Inv #2 Bipolar midpoint both-required                    → test_bipolar_*
  Inv #3 Three-state match                                 → test_*_not_found
  Inv #4 Provenance absolute paths                         → test_provenance_*
  Inv #5 Order anchor (the most critical)                  → test_channel_order_*
  Inv #6 Space/units explicit, no voxel-as-mm              → test_coord_space_*
  Inv #7 No cross-subject registration                     → (out of scope; per-subject)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.seeg_coord_loader import (
    BipolarRes,
    CoordResult,
    LOADER_VERSION,
    MNI152_1MM_AFFINE,
    MNI152_1MM_SHAPE,
    MissingEntry,
    _apply_affine,
    _build_result,
    _find_epilepsiae_mri,
    _parse_channel_to_query_keys,
    _resolve_bipolar,
    _validate_coord_result,
    assert_coord_result_is_mm_for_main_analysis,
    load_subject_coords,
)


SEED = 20260521


# =============================================================================
# Synthetic data fixtures
# =============================================================================


@pytest.fixture
def fake_yuquan_root(tmp_path):
    """Create a fake Yuquan dir with chnXyzDict.npy for two synthetic subjects."""
    root = tmp_path / "patients_elecs_reGen"
    root.mkdir(parents=True)

    # Subject "synth1": shafts A, B, C with 5 contacts each
    s1 = root / "synth1"
    s1.mkdir()
    chn_dict_s1 = {
        "A": np.array([[i * 1.0, 0.0, 0.0] for i in range(5)]),  # along x
        "B": np.array([[0.0, i * 1.0, 0.0] for i in range(5)]),  # along y
        "C": np.array([[0.0, 0.0, i * 1.0] for i in range(5)]),  # along z
    }
    np.save(s1 / "chnXyzDict.npy", chn_dict_s1, allow_pickle=True)

    # Subject "synth2": single shaft only
    s2 = root / "synth2"
    s2.mkdir()
    np.save(s2 / "chnXyzDict.npy", {"A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}, allow_pickle=True)

    return root


@pytest.fixture
def fake_epilepsiae_root(tmp_path):
    """Create a fake Epilepsiae SQL dir with synthetic electrode INSERTs.

    Uses numeric subject "999" → canonical "99902" → SQL file pat_99902_*.sql.
    Tests pass subject_id="999" which canonicalizes to "99902".
    """
    root = tmp_path / "all_data_sqls"
    root.mkdir()

    sql_content = """
-- pat_99902_2026-05-21.sql
INSERT INTO patient (id, patientcode) VALUES (99902, 'FR_999');
INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 100, 'HLA1', 'HLA1', NULL, 'i', TRUE, 'AD', 10.0, 20.0, 30.0, NULL);
INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (2, 100, 'HLA2', 'HLA2', NULL, 'i', TRUE, 'AD', 12.0, 22.0, 32.0, NULL);
INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (3, 100, 'HLA3', 'HLA3', NULL, 'i', TRUE, 'AD', NULL, NULL, NULL, 'nicht abgrenzbar!');
INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (4, 100, 'HLA4', 'HLA4', NULL, 'i', TRUE, 'AD', 16.0, 26.0, 36.0, 'Mikrokontakt');
INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (5, 100, 'GC1', 'GC1', NULL, 'e', TRUE, 'AD', 100.0, 110.0, 120.0, NULL);
INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (6, 100, 'GC2', 'GC2', NULL, 'e', TRUE, 'AD', 102.0, 112.0, 122.0, NULL);
"""
    (root / "pat_99902_2026-05-21.sql").write_text(sql_content)
    return root


# =============================================================================
# Channel parsing
# =============================================================================


def test_parse_monopolar_channel():
    q = _parse_channel_to_query_keys("HLA1")
    assert q.is_bipolar is False
    assert q.left == ("HLA", 1)
    assert q.right is None


def test_parse_bipolar_channel():
    q = _parse_channel_to_query_keys("HLA1-HLA2")
    assert q.is_bipolar is True
    assert q.left == ("HLA", 1)
    assert q.right == ("HLA", 2)


def test_parse_primed_shaft():
    q = _parse_channel_to_query_keys("A'1-A'2")
    assert q.is_bipolar is True
    assert q.left == ("A'", 1)
    assert q.right == ("A'", 2)


def test_parse_unparseable_returns_none():
    q = _parse_channel_to_query_keys("garbage")
    assert q.left is None


# =============================================================================
# Bipolar midpoint (invariant #2)
# =============================================================================


def test_bipolar_midpoint_both_endpoints_required():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 4.0, 6.0])
    out = _resolve_bipolar(a, b)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_bipolar_partial_endpoint_returns_none():
    a = np.array([0.0, 0.0, 0.0])
    assert _resolve_bipolar(a, None) is None
    assert _resolve_bipolar(None, a) is None
    assert _resolve_bipolar(None, None) is None


# =============================================================================
# Yuquan loader (invariants #5, #6)
# =============================================================================


def test_yuquan_coord_space_is_fs_native_ras_mm(fake_yuquan_root):
    result = load_subject_coords(
        dataset="yuquan",
        subject_id="synth1",
        channel_names_requested=["A1-A2"],
        yuquan_root=fake_yuquan_root,
    )
    assert result.coord_space == "fs_native_ras_mm"
    assert result.coord_units == "mm"
    assert result.dataset == "yuquan"
    assert result.schema_version == "coord_loader_v3"


def test_yuquan_rejects_mri_affine(fake_yuquan_root):
    """Yuquan coords are already mm; affine arg must raise."""
    with pytest.raises(ValueError, match="already fs_native_ras_mm"):
        load_subject_coords(
            dataset="yuquan",
            subject_id="synth1",
            channel_names_requested=["A1-A2"],
            mri_affine=np.eye(4),
            yuquan_root=fake_yuquan_root,
        )


def test_yuquan_channel_order_preserved_under_dict_shuffle(fake_yuquan_root):
    """v2 invariant #5: output array MUST match input channel order, not internal sort."""
    requested = ["C1-C2", "A1-A2", "B1-B2"]  # alphabetically backwards
    result = load_subject_coords(
        dataset="yuquan",
        subject_id="synth1",
        channel_names_requested=requested,
        yuquan_root=fake_yuquan_root,
    )
    assert result.channel_names_requested == requested
    # synth1 layout: A along x, B along y, C along z
    # C1-C2 midpoint = (0, 0, 0.5); A1-A2 = (0.5, 0, 0); B1-B2 = (0, 0.5, 0)
    np.testing.assert_allclose(result.coords_array_in_requested_order[0], [0, 0, 0.5])
    np.testing.assert_allclose(result.coords_array_in_requested_order[1], [0.5, 0, 0])
    np.testing.assert_allclose(result.coords_array_in_requested_order[2], [0, 0.5, 0])


def test_yuquan_unmapped_channel_marked_missing(fake_yuquan_root):
    """If channel not in chnXyzDict, must be missing + NaN row + mask False."""
    result = load_subject_coords(
        dataset="yuquan",
        subject_id="synth1",
        channel_names_requested=["A1-A2", "Z9-Z10"],  # Z doesn't exist
        yuquan_root=fake_yuquan_root,
    )
    assert result.mapped_mask_in_requested_order[0] == True
    assert result.mapped_mask_in_requested_order[1] == False
    assert np.all(np.isnan(result.coords_array_in_requested_order[1]))
    z_missing = [m for m in result.missing if m.channel == "Z9-Z10"]
    assert len(z_missing) == 1
    assert z_missing[0].reason == "bipolar_partial_endpoint"
    assert z_missing[0].index_in_requested == 1


def test_yuquan_provenance_records_absolute_path(fake_yuquan_root):
    result = load_subject_coords(
        dataset="yuquan",
        subject_id="synth1",
        channel_names_requested=["A1-A2"],
        yuquan_root=fake_yuquan_root,
    )
    assert "source_path" in result.provenance
    assert Path(result.provenance["source_path"]).is_absolute()
    assert "chnXyzDict.npy" in result.provenance["source_path"]
    assert result.provenance["loader_version"] == LOADER_VERSION


def test_yuquan_monopolar_channel(fake_yuquan_root):
    """Test monopolar 'A1' lookup (not bipolar)."""
    result = load_subject_coords(
        dataset="yuquan",
        subject_id="synth1",
        channel_names_requested=["A1"],
        yuquan_root=fake_yuquan_root,
    )
    assert result.mapped_mask_in_requested_order[0] == True
    np.testing.assert_allclose(result.coords_array_in_requested_order[0], [0, 0, 0])


# =============================================================================
# Epilepsiae loader (invariants #1, #2, #6)
# =============================================================================


def test_epilepsiae_voxel_path_requires_explicit_optin(fake_epilepsiae_root):
    """v3.1: default behavior with no MRI must RAISE, not silently return voxel.

    To get voxel sensitivity mode, caller must either:
      - pass allow_voxel_fallback=True, OR
      - pass epilepsiae_mri_search_roots=[] (disable auto-discovery)
    """
    # Default: no MRI found in real mounts (subject 999 doesn't exist in /mnt)
    # → must raise (v3.1 lock)
    with pytest.raises(FileNotFoundError, match="MRI not found"):
        load_subject_coords(
            dataset="epilepsiae",
            subject_id="999",
            channel_names_requested=["HLA1-HLA2"],
            epilepsiae_root=fake_epilepsiae_root,
            # No epilepsiae_mri_search_roots override; default has no fake 999 MRI
        )

    # Explicit opt-in: allow_voxel_fallback=True
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],  # disable real-mount auto-discovery
        allow_voxel_fallback=True,
    )
    assert result.coord_space == "mri_native_voxel_ijk"
    assert result.coord_units == "voxel"


def test_epilepsiae_with_affine_becomes_ras_mm(fake_epilepsiae_root):
    """Identity-scaled affine → voxel * 2 = mm."""
    affine = np.eye(4)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        mri_affine=affine,
        epilepsiae_root=fake_epilepsiae_root,
    )
    assert result.coord_space == "ras_mm_via_affine"
    assert result.coord_units == "mm"
    # HLA1 voxel = (10, 20, 30), HLA2 voxel = (12, 22, 32)
    # midpoint voxel = (11, 21, 31); after affine *2 = (22, 42, 62)
    np.testing.assert_allclose(result.coords_array_in_requested_order[0], [22, 42, 62])


def test_epilepsiae_sql_null_marks_missing(fake_epilepsiae_root):
    """HLA3 has coord NULL + 'nicht abgrenzbar!' commentary → HLA3-HLA4 pair missing."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA3-HLA4"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,  # SQL-focused test; no MRI needed
    )
    assert result.mapped_mask_in_requested_order[0] == False
    assert np.all(np.isnan(result.coords_array_in_requested_order[0]))
    entries = [m for m in result.missing if m.channel == "HLA3-HLA4"]
    assert len(entries) == 1
    assert entries[0].reason == "commentary_nicht_abgrenzbar"
    assert "nicht abgrenzbar" in (entries[0].commentary or "").lower()


def test_epilepsiae_mikrokontakt_commentary_captured(fake_epilepsiae_root):
    """HLA4 has Mikrokontakt commentary but valid coord — should still be found.

    Mikrokontakt is informational, not invalid; only when coord is NULL it counts as missing.
    """
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA4"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    # HLA4 has valid coords (16, 26, 36) with Mikrokontakt commentary
    assert result.mapped_mask_in_requested_order[0] == True
    np.testing.assert_allclose(result.coords_array_in_requested_order[0], [16, 26, 36])


def test_epilepsiae_bipolar_one_end_null_marks_pair_missing(fake_epilepsiae_root):
    """Bipolar 'HLA2-HLA3': HLA2 valid, HLA3 NULL → pair missing (invariant #2)."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA2-HLA3"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    assert result.mapped_mask_in_requested_order[0] == False
    assert "HLA2-HLA3" in result.bipolar_resolution
    bp = result.bipolar_resolution["HLA2-HLA3"]
    assert bp.left_coord is not None  # HLA2 found
    assert bp.right_coord is None     # HLA3 NULL
    assert bp.midpoint_strategy == "mean_both_required"


def test_epilepsiae_channel_order_preserved(fake_epilepsiae_root):
    """Invariant #5: output array index matches requested index, not SQL order."""
    requested = ["GC1-GC2", "HLA1-HLA2"]  # GC after HLA in SQL file (id 5/6 vs 1/2)
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=requested,
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    assert result.channel_names_requested == requested
    # GC1-GC2 midpoint = (101, 111, 121); HLA1-HLA2 midpoint = (11, 21, 31)
    np.testing.assert_allclose(result.coords_array_in_requested_order[0], [101, 111, 121])
    np.testing.assert_allclose(result.coords_array_in_requested_order[1], [11, 21, 31])


# =============================================================================
# Affine
# =============================================================================


def test_apply_affine_1d():
    aff = np.eye(4)
    aff[0, 0] = 2.0
    coord = np.array([1.0, 2.0, 3.0])
    out = _apply_affine(coord, aff)
    np.testing.assert_allclose(out, [2.0, 2.0, 3.0])


def test_apply_affine_2d():
    aff = np.eye(4)
    aff[:3, 3] = [10.0, 20.0, 30.0]  # translation
    coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    out = _apply_affine(coords, aff)
    np.testing.assert_allclose(out, [[11, 22, 33], [14, 25, 36]])


# =============================================================================
# Validation / error paths
# =============================================================================


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_subject_coords(
            dataset="invalid",
            subject_id="anything",
            channel_names_requested=["A1-A2"],
        )


def test_yuquan_subject_not_found(fake_yuquan_root):
    with pytest.raises(FileNotFoundError, match="Yuquan coord file not found"):
        load_subject_coords(
            dataset="yuquan",
            subject_id="nonexistent",
            channel_names_requested=["A1-A2"],
            yuquan_root=fake_yuquan_root,
        )


def test_epilepsiae_subject_not_found(fake_epilepsiae_root):
    with pytest.raises(FileNotFoundError, match="Epilepsiae SQL not found"):
        load_subject_coords(
            dataset="epilepsiae",
            subject_id="999999999",
            channel_names_requested=["HLA1-HLA2"],
            epilepsiae_root=fake_epilepsiae_root,
        )


def test_validation_catches_mask_coords_mismatch():
    """Internal _validate_coord_result raises if mask and NaN-rows disagree."""
    # Build a corrupt CoordResult manually
    bad = CoordResult(
        schema_version="coord_loader_v2",
        dataset="yuquan",
        subject_id="bad",
        channel_names_requested=["A1-A2"],
        coords_array_in_requested_order=np.array([[1.0, 2.0, 3.0]]),  # finite
        mapped_mask_in_requested_order=np.array([False]),  # but mask says missing!
        coord_space="fs_native_ras_mm",
        coord_units="mm",
        provenance={"source_path": "/tmp/fake"},
    )
    with pytest.raises(ValueError, match="mapped_mask must match"):
        _validate_coord_result(bad)


def test_validation_catches_unmapped_without_missing_entry():
    """v1.0.5 advisor #2 fix: every mask=False index must have missing[] entry.

    A NaN row with no missing[] entry silently loses the reason — caller can't
    tell why the channel wasn't found.
    """
    bad = CoordResult(
        schema_version="coord_loader_v2",
        dataset="yuquan",
        subject_id="bad",
        channel_names_requested=["A1-A2", "Z9-Z10"],
        coords_array_in_requested_order=np.array(
            [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]
        ),
        mapped_mask_in_requested_order=np.array([True, False]),
        coord_space="fs_native_ras_mm",
        coord_units="mm",
        provenance={"source_path": "/tmp/fake"},
        missing=[],  # EMPTY — but index 1 has mask=False; invariant violated
    )
    with pytest.raises(ValueError, match="no missing\\[\\] entry"):
        _validate_coord_result(bad)


def test_validation_catches_space_units_mismatch():
    bad = CoordResult(
        schema_version="coord_loader_v2",
        dataset="yuquan",
        subject_id="bad",
        channel_names_requested=["A1-A2"],
        coords_array_in_requested_order=np.array([[1.0, 2.0, 3.0]]),
        mapped_mask_in_requested_order=np.array([True]),
        coord_space="fs_native_ras_mm",
        coord_units="voxel",  # WRONG — should be mm
        provenance={"source_path": "/tmp/fake"},
    )
    with pytest.raises(ValueError, match="fs_native_ras_mm must have coord_units='mm'"):
        _validate_coord_result(bad)


# =============================================================================
# Cross-cutting contract for Phase 1 consumers (invariant #6 enforcement)
# =============================================================================


def test_assert_mm_passes_for_yuquan(fake_yuquan_root):
    result = load_subject_coords(
        dataset="yuquan",
        subject_id="synth1",
        channel_names_requested=["A1-A2"],
        yuquan_root=fake_yuquan_root,
    )
    # Yuquan is mm → should not raise
    assert_coord_result_is_mm_for_main_analysis(result)


def test_assert_mm_rejects_voxel_for_main_analysis(fake_epilepsiae_root):
    """Critical cross-cutting: Phase 1 main analysis must reject voxel coords."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    # Default Epilepsiae is voxel → must raise
    with pytest.raises(ValueError, match="coord_units='mm'"):
        assert_coord_result_is_mm_for_main_analysis(result)


def test_assert_mm_passes_for_epilepsiae_with_affine(fake_epilepsiae_root):
    """Epilepsiae with affine becomes mm → assertion passes."""
    aff = np.eye(4) * 2.0
    aff[3, 3] = 1.0
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        mri_affine=aff,
        epilepsiae_root=fake_epilepsiae_root,
    )
    # Affine applied → mm → should not raise
    assert_coord_result_is_mm_for_main_analysis(result)


# =============================================================================
# v3 (2026-05-21) MNI152 auto-discovery
# =============================================================================


@pytest.fixture
def fake_epilepsiae_mri_root(tmp_path):
    """Build a fake Epilepsiae MRI mount for canonical subject pat_99902.

    Uses Nifti1Pair (preserves full affine in .hdr+.img split format;
    AnalyzeImage strips affine to just origin/pixdim and fails to round-trip).
    """
    import nibabel as nib

    root = tmp_path / "inv_fake"
    pat_dir = root / "pat_99902"
    mri_dir = pat_dir / "adm_999102" / "MRI"
    mri_dir.mkdir(parents=True)

    data = np.zeros(MNI152_1MM_SHAPE, dtype=np.uint8)
    img = nib.Nifti1Pair(data, affine=MNI152_1MM_AFFINE)
    nib.save(img, str(mri_dir / "mri_999102.img"))

    return root


def test_find_epilepsiae_mri_with_canonical_id(fake_epilepsiae_mri_root):
    """_find_epilepsiae_mri (v3.1) takes canonical_id only — no short-form expansion."""
    path = _find_epilepsiae_mri("99902", [fake_epilepsiae_mri_root])
    assert path is not None
    assert path.name == "mri_999102.img"


def test_find_epilepsiae_mri_short_id_does_not_match(fake_epilepsiae_mri_root):
    """v3.1: passing short-form '999' (uncanonicalized) must NOT match pat_99902.

    Caller is expected to canonicalize via _canonicalize_epilepsiae_subject_id
    before passing. _find_epilepsiae_mri is strict exact-match.
    """
    path = _find_epilepsiae_mri("999", [fake_epilepsiae_mri_root])
    assert path is None  # pat_999/ does not exist; pat_99902/ does (canonical)


def test_find_epilepsiae_mri_not_found(tmp_path):
    """No MRI file → None."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert _find_epilepsiae_mri("999999", [empty_root]) is None


def test_epilepsiae_auto_discovery_yields_mni152_1mm(
    fake_epilepsiae_root, fake_epilepsiae_mri_root
):
    """v3: when MRI auto-discovered, coord_space becomes mni152_1mm + mm units."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[fake_epilepsiae_mri_root],
    )
    assert result.coord_space == "mni152_1mm"
    assert result.coord_units == "mm"
    assert result.source_coord_type == "sql_voxel_ijk"
    assert result.normalization_certainty == "grid_confirmed_warp_type_unverified"
    assert "mri_999102.img" in result.provenance["affine_path"]


def test_epilepsiae_auto_discovery_applies_mni_affine(
    fake_epilepsiae_root, fake_epilepsiae_mri_root
):
    """Coords after auto-discovery should match the formula x_mm = 90 - voxel_x etc."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[fake_epilepsiae_mri_root],
    )
    # HLA1 voxel = (10, 20, 30), HLA2 = (12, 22, 32); midpoint voxel = (11, 21, 31)
    # MNI mm: (90 - 11, 21 - 126, 31 - 72) = (79, -105, -41)
    np.testing.assert_allclose(
        result.coords_array_in_requested_order[0], [79.0, -105.0, -41.0]
    )


def test_epilepsiae_voxel_fallback_when_disabled(fake_epilepsiae_root):
    """When epilepsiae_mri_search_roots=[] AND allow_voxel_fallback=True, returns voxel."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[],  # explicitly disable auto-discovery
        allow_voxel_fallback=True,  # v3.1: required for voxel mode
    )
    assert result.coord_space == "mri_native_voxel_ijk"
    assert result.coord_units == "voxel"
    assert result.source_coord_type == "sql_voxel_ijk"
    assert result.normalization_certainty == "subject_native_voxel_no_affine"


def test_epilepsiae_explicit_affine_overrides_auto_discovery(
    fake_epilepsiae_root, fake_epilepsiae_mri_root
):
    """User-provided mri_affine wins over auto-discovery."""
    custom = np.eye(4) * 2.0
    custom[3, 3] = 1.0
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        mri_affine=custom,  # explicit override
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[fake_epilepsiae_mri_root],
    )
    assert result.coord_space == "ras_mm_via_affine"  # not mni152_1mm
    assert result.normalization_certainty == "external_affine_provided"


def test_epilepsiae_raises_on_non_mni_affine(fake_epilepsiae_root, tmp_path):
    """Defensive: if a subject's MRI affine doesn't match canonical MNI152, raise.

    Cohort assumption (all 27 verified subjects share MNI152 affine) is broken
    if any subject's MRI shows a different affine — must be surfaced loudly.
    """
    import nibabel as nib

    bad_root = tmp_path / "inv_bad"
    pat_dir = bad_root / "pat_99902" / "adm_999102" / "MRI"
    pat_dir.mkdir(parents=True)
    # Build MRI with NON-MNI affine
    bad_affine = np.eye(4)
    bad_affine[0, 0] = 0.5  # different voxel size
    img = nib.Nifti1Pair(
        np.zeros(MNI152_1MM_SHAPE, dtype=np.uint8), affine=bad_affine
    )
    nib.save(img, str(pat_dir / "mri_999102.img"))

    with pytest.raises(ValueError, match="does not match canonical MNI152"):
        load_subject_coords(
            dataset="epilepsiae",
            subject_id="999",
            channel_names_requested=["HLA1-HLA2"],
            epilepsiae_root=fake_epilepsiae_root,
            epilepsiae_mri_search_roots=[bad_root],
        )


def test_assert_mm_passes_for_mni152_1mm(
    fake_epilepsiae_root, fake_epilepsiae_mri_root
):
    """Phase 1 mm assertion passes for v3 mni152_1mm coord space."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[fake_epilepsiae_mri_root],
    )
    # No raise expected
    assert_coord_result_is_mm_for_main_analysis(result)


# =============================================================================
# v3.1 (2026-05-21) Cross-patient pollution regression tests
# =============================================================================


def test_canonicalize_subject_id_appends_02():
    """v3.1 canonicalization rule: '<id>' → '<id>02' if not already ending in 02."""
    from src.seeg_coord_loader import _canonicalize_epilepsiae_subject_id as canon
    assert canon("115") == "11502"      # subject 115 → pat_11502
    assert canon("1150") == "115002"    # subject 1150 → pat_115002
    assert canon("1084") == "108402"    # subject 1084 → pat_108402
    assert canon("620") == "62002"      # subject 620 → pat_62002
    assert canon("862") == "86202"      # subject 862 → pat_86202


def test_canonicalize_subject_id_keeps_canonical():
    """If already ending in '02', return as-is."""
    from src.seeg_coord_loader import _canonicalize_epilepsiae_subject_id as canon
    assert canon("108402") == "108402"
    assert canon("11502") == "11502"
    assert canon("115002") == "115002"


def test_canonicalize_rejects_non_numeric():
    from src.seeg_coord_loader import _canonicalize_epilepsiae_subject_id as canon
    with pytest.raises(ValueError, match="numeric"):
        canon("pat_108402")
    with pytest.raises(ValueError, match="numeric"):
        canon("108402_2012-12-20")


def test_canonicalize_rejects_empty():
    from src.seeg_coord_loader import _canonicalize_epilepsiae_subject_id as canon
    with pytest.raises(ValueError, match="non-empty"):
        canon("")


def test_short_id_115_does_not_match_115002_sql(tmp_path):
    """Critical regression (user 2026-05-21 audit): subject_id='115' must NOT
    silently match pat_115002's SQL — they are DIFFERENT patients.

    v3.1 fix: canonical '11502' is searched exactly via pat_11502_*.sql glob.
    Even if pat_115002 SQL exists in the same dir, it must not be picked.
    """
    sql_root = tmp_path / "sqls"
    sql_root.mkdir()
    # Create BOTH pat_11502 (subject 115) and pat_115002 (subject 1150) SQLs
    (sql_root / "pat_11502_2010-01-01.sql").write_text(
        """INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 1, 'A1', 'A1', NULL, 'i', TRUE, 'AD', 10.0, 20.0, 30.0, NULL);
"""
    )
    (sql_root / "pat_115002_2011-01-01.sql").write_text(
        """INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 1, 'B1', 'B1', NULL, 'i', TRUE, 'AD', 99.0, 99.0, 99.0, NULL);
"""
    )

    # Caller passes "115" → should canonicalize to "11502", read pat_11502 SQL only
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="115",
        channel_names_requested=["A1"],
        epilepsiae_root=sql_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    # Should resolve channel A1 (in pat_11502 SQL) at (10, 20, 30) — NOT B1 from pat_115002
    assert result.mapped_mask_in_requested_order[0] == True
    np.testing.assert_allclose(
        result.coords_array_in_requested_order[0], [10.0, 20.0, 30.0]
    )
    assert "pat_11502" in result.provenance["source_path"]
    assert "pat_115002" not in result.provenance["source_path"]
    assert result.provenance["canonical_subject_id"] == "11502"
    assert result.provenance["input_subject_id"] == "115"


def test_short_id_1150_canonicalizes_to_115002(tmp_path):
    """Mirror of above: subject_id='1150' must match pat_115002, NOT pat_11502."""
    sql_root = tmp_path / "sqls"
    sql_root.mkdir()
    (sql_root / "pat_11502_2010-01-01.sql").write_text(
        """INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 1, 'A1', 'A1', NULL, 'i', TRUE, 'AD', 10.0, 20.0, 30.0, NULL);
"""
    )
    (sql_root / "pat_115002_2011-01-01.sql").write_text(
        """INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 1, 'B1', 'B1', NULL, 'i', TRUE, 'AD', 99.0, 99.0, 99.0, NULL);
"""
    )

    # Caller passes "1150" → canonicalize to "115002" → finds pat_115002 SQL
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="1150",
        channel_names_requested=["B1"],
        epilepsiae_root=sql_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    assert result.mapped_mask_in_requested_order[0] == True
    np.testing.assert_allclose(
        result.coords_array_in_requested_order[0], [99.0, 99.0, 99.0]
    )
    assert "pat_115002" in result.provenance["source_path"]
    assert result.provenance["canonical_subject_id"] == "115002"


def test_short_id_620_does_not_match_86202(tmp_path):
    """Regression: '620' must canonicalize to '62002', NOT match pat_86202.

    Old fuzzy glob *620*.sql would have matched both pat_62002 and pat_86202
    (since "620" appears in "86202"). v3.1 exact-match prevents this.
    """
    sql_root = tmp_path / "sqls"
    sql_root.mkdir()
    (sql_root / "pat_62002_2010-01-01.sql").write_text(
        """INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 1, 'A1', 'A1', NULL, 'i', TRUE, 'AD', 1.0, 2.0, 3.0, NULL);
"""
    )
    (sql_root / "pat_86202_2010-01-01.sql").write_text(
        """INSERT INTO electrode (id, "array", name, moniker, artifact, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary) VALUES (1, 1, 'X1', 'X1', NULL, 'i', TRUE, 'AD', 99.0, 99.0, 99.0, NULL);
"""
    )

    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="620",
        channel_names_requested=["A1"],
        epilepsiae_root=sql_root,
        epilepsiae_mri_search_roots=[],
        allow_voxel_fallback=True,
    )
    assert result.mapped_mask_in_requested_order[0] == True
    np.testing.assert_allclose(
        result.coords_array_in_requested_order[0], [1.0, 2.0, 3.0]
    )
    assert "pat_62002" in result.provenance["source_path"]


def test_no_sql_match_raises(tmp_path):
    """0 SQL matches → FileNotFoundError (not silent default)."""
    sql_root = tmp_path / "sqls"
    sql_root.mkdir()
    # No SQL file for this canonical
    with pytest.raises(FileNotFoundError, match="SQL not found"):
        load_subject_coords(
            dataset="epilepsiae",
            subject_id="777",  # canonical "77702", no SQL exists
            channel_names_requested=["A1"],
            epilepsiae_root=sql_root,
            epilepsiae_mri_search_roots=[],
            allow_voxel_fallback=True,
        )


def test_ambiguous_sql_match_raises(tmp_path):
    """>1 SQL matches for one canonical → ValueError (data integrity issue)."""
    sql_root = tmp_path / "sqls"
    sql_root.mkdir()
    # Two SQL files with identical pat_<canonical>_ prefix but different dates
    # (shouldn't happen in practice but loader must defensive-fail)
    (sql_root / "pat_99902_2010-01-01.sql").write_text("")
    (sql_root / "pat_99902_2011-01-01.sql").write_text("")

    with pytest.raises(ValueError, match="Ambiguous"):
        load_subject_coords(
            dataset="epilepsiae",
            subject_id="999",
            channel_names_requested=["A1"],
            epilepsiae_root=sql_root,
            epilepsiae_mri_search_roots=[],
            allow_voxel_fallback=True,
        )


def test_mri_miss_default_raises(fake_epilepsiae_root):
    """v3.1 lock: MRI miss with no allow_voxel_fallback → FileNotFoundError."""
    # Default: search real /mnt roots, no fake "999" MRI there
    with pytest.raises(FileNotFoundError, match="MRI not found"):
        load_subject_coords(
            dataset="epilepsiae",
            subject_id="999",
            channel_names_requested=["HLA1-HLA2"],
            epilepsiae_root=fake_epilepsiae_root,
            # allow_voxel_fallback defaults to False
        )


def test_provenance_records_canonical_and_input_id(
    fake_epilepsiae_root, fake_epilepsiae_mri_root
):
    """v3.1 provenance: canonical_subject_id + input_subject_id both recorded."""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="999",  # short form
        channel_names_requested=["HLA1-HLA2"],
        epilepsiae_root=fake_epilepsiae_root,
        epilepsiae_mri_search_roots=[fake_epilepsiae_mri_root],
    )
    assert result.provenance["canonical_subject_id"] == "99902"
    assert result.provenance["input_subject_id"] == "999"
    # subject_id field also stores canonical for downstream consistency
    assert result.subject_id == "99902"

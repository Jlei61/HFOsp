import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.topic4_rev22_response_design import (
    FALLBACK_MASKS, MASKS, PARAMS, REFERENCE, STATUS_FALLBACK, STATUS_PRIMARY,
    assert_seed_disjointness, build_seed_manifest, canonical_round, design_rows_to_manifest,
    domain_from_geometry, family_membership, generate_design, nearest_to_centre,
    point_table_sha256, synthetic_domain,
)

ROOT = Path(__file__).resolve().parents[1]
NODE_FIELD = {"field_type": "manual_dual_core_budget_matched", "target_count": 1499,
              "field_sha256": "abc", "centers_mm": [[1.0, 1.0], [18.0, 2.0]]}
NODE_MAPPING = {"node_gain": 1.0, "signed_depth_shrinkage": 1.0}


def _primary_domain():
    return domain_from_geometry(synthetic_domain((30.0, 60.0), (1.25, 2.75)))


def _rows(domain, seed=20260905):
    design = generate_design(domain, seed=seed)
    return design, design_rows_to_manifest(design, node_field=NODE_FIELD, node_mapping=NODE_MAPPING)


def test_primary_block_counts_and_single_reference():
    design, rows = _rows(_primary_domain())
    assert design["branch"] == STATUS_PRIMARY
    counts = {}
    for r in rows:
        counts[r["block"]] = counts.get(r["block"], 0) + 1
    assert counts == {"reference": 1, "full4d": 47, "lock_g_LEE": 8, "lock_g_LEI": 8,
                      "lock_theta": 8, "lock_AR": 8, "dose_plane": 8, "geometry_plane": 8}
    assert len(rows) == 96
    ref = [r for r in rows if r["is_reference"]]
    assert len(ref) == 1 and [ref[0]["physical"][p] for p in PARAMS] == list(REFERENCE)
    assert ref[0]["family_membership"] == list(MASKS)


def test_points_inside_domain_and_locked_coordinates_exact():
    domain = _primary_domain()
    _, rows = _rows(domain)
    bounds = domain["bounds"]
    locked = {"lock_g_LEE": 0, "lock_g_LEI": 1, "lock_theta": 2, "lock_AR": 3}
    for r in rows:
        for d, p in enumerate(PARAMS):
            assert bounds[d][0] - 1e-9 <= r["physical"][p] <= bounds[d][1] + 1e-9
        if r["block"] in locked:
            d = locked[r["block"]]
            assert r["physical"][PARAMS[d]] == REFERENCE[d]
            assert PARAMS[d] not in r["free_dimensions"]
        if r["block"] == "dose_plane":
            assert r["physical"]["theta_FT_deg"] == REFERENCE[2] and r["physical"]["AR_FT"] == REFERENCE[3]
            assert set(r["family_membership"]) >= {"M1100", "M1111", "M1101", "M1110"}
            assert "M0011" not in r["family_membership"]
        if r["block"] == "geometry_plane":
            assert r["physical"]["g_LEE"] == REFERENCE[0] and r["physical"]["g_LEI"] == REFERENCE[1]
            assert set(r["family_membership"]) >= {"M0011", "M1111", "M0111", "M1011"}
        assert r["mechanisms"]["Z_M"] == "off"
        assert r["mechanisms"]["g_EE"] == r["physical"]["g_LEE"]
        assert r["mechanisms"]["ellipse_angle_deg"] == r["physical"]["theta_FT_deg"]


def test_determinism_and_seed_sensitivity():
    domain = _primary_domain()
    _, a = _rows(domain, seed=1)
    _, b = _rows(domain, seed=1)
    _, c = _rows(domain, seed=2)
    assert point_table_sha256(a) == point_table_sha256(b)
    assert point_table_sha256(a) != point_table_sha256(c)


def test_augmented_design_has_nonzero_separation():
    design, _ = _rows(_primary_domain(), seed=9)
    unit = np.asarray([row["unit"] for row in design["rows"]], float)
    delta = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=2)
    delta[np.eye(len(unit), dtype=bool)] = np.inf
    assert float(delta.min()) > 0.01


def test_duplicate_rejection_after_rounding():
    domain = _primary_domain()
    design, rows = _rows(domain)
    keys = [tuple(canonical_round(np.asarray([r["physical"][p] for p in PARAMS])).tolist()) for r in rows]
    assert len(set(keys)) == len(keys)
    # rounding is canonical: angle 2 decimals, others 4
    for r in rows:
        assert r["physical"]["theta_FT_deg"] == round(r["physical"]["theta_FT_deg"], 2)
        assert r["physical"]["g_LEE"] == round(r["physical"]["g_LEE"], 4)


def test_fallback_branch_from_non_estimable_or_degenerate_domain():
    for payload in (
        {"status": "GEOMETRY_STRUCTURALLY_NON_ESTIMABLE", "admissible_rectangle": None},
        {"status": "GEOMETRY_DOMAIN_FROZEN", "admissible_rectangle": {"theta_deg": [45.0, 45.0],
                                                                       "aspect_ratio": [2.0, 2.0]}},
        {"status": "GEOMETRY_DOMAIN_FROZEN_ONE_DIMENSION_DEGENERATE",
         "admissible_rectangle": {"theta_deg": [45.0, 45.0], "aspect_ratio": [1.5, 2.5]}},
    ):
        domain = domain_from_geometry(payload)
        assert domain["branch"] == STATUS_FALLBACK
        design, rows = _rows(domain)
        assert len(rows) == 32
        assert sum(r["is_reference"] for r in rows) == 1
        for r in rows:
            assert r["physical"]["theta_FT_deg"] == REFERENCE[2] and r["physical"]["AR_FT"] == REFERENCE[3]
            assert set(r["family_membership"]) <= set(FALLBACK_MASKS)
        assert nearest_to_centre(rows) is not None


def test_domain_must_contain_reference():
    with pytest.raises(ValueError):
        domain_from_geometry({"status": "GEOMETRY_DOMAIN_FROZEN",
                              "admissible_rectangle": {"theta_deg": [50.0, 60.0], "aspect_ratio": [1.0, 3.0]}})


def test_family_membership_rules():
    assert family_membership(np.asarray(REFERENCE), STATUS_PRIMARY) == list(MASKS)
    assert family_membership(np.asarray([0.7, 1.0, 45.0, 2.0]), STATUS_PRIMARY) == [
        "M1000", "M1100", "M1111", "M1011", "M1101", "M1110"]
    assert family_membership(np.asarray([0.7, 1.2, 50.0, 2.5]), STATUS_PRIMARY) == ["M1111"]


def test_seed_manifest_non_overlap_and_decomposition_block():
    _, rows = _rows(_primary_domain())
    manifest = build_seed_manifest("dci_p000", nearest_to_centre(rows))
    assert_seed_disjointness(manifest)
    assert [u["topology_seed"] for u in manifest["fit"]["units"]] == [2511, 2512, 2513, 2514]
    assert all(u["dynamics_seed"] == u["topology_seed"] for u in manifest["fit"]["units"])
    assert manifest["variance_decomposition_block"]["new_trajectories"] == 16
    assert len(manifest["qualification"]["units"]) == 6 and len(manifest["confirmation"]["units"]) == 12
    assert manifest["null_and_node_block"]["topology_seeds"] == list(range(2621, 2627))
    bad = json.loads(json.dumps(manifest))
    bad["qualification"]["units"][0]["topology_seed"] = 2521
    with pytest.raises(ValueError):
        assert_seed_disjointness(bad)


def test_cli_synthetic_run_refuses_results_and_writes_outputs(tmp_path):
    script = ROOT / "scripts/freeze_topic4_rev22_response_design.py"
    rev20_manifest = tmp_path / "rev20_manifest.json"
    rev20_manifest.write_text(json.dumps({"candidates": [{"candidate_id": "ref", "is_reference": True,
                                                          "node_field": NODE_FIELD}]}))
    out = tmp_path / "design"
    subprocess.run([sys.executable, str(script), "--theta-range", "30", "60", "--ar-range", "1.25", "2.75",
                    "--out-dir", str(out), "--rev20-manifest", str(rev20_manifest)], cwd=ROOT, check=True,
                   capture_output=True)
    manifest = json.loads((out / "response_design_manifest.json").read_text())
    assert manifest["candidate_count"] == 96 and manifest["branch"] == STATUS_PRIMARY
    assert (out / "response_design_manifest.json.sha256").is_file()
    assert (out / "seed_manifest.json").is_file()
    assert (out / "figures/response_design_coverage.png").is_file()
    assert (out / "figures/README.md").read_text().startswith("### response_design_coverage.png")
    refused = subprocess.run([sys.executable, str(script), "--theta-range", "30", "60", "--ar-range", "1.25", "2.75",
                              "--out-dir", "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/x",
                              "--rev20-manifest", str(rev20_manifest)], cwd=ROOT, capture_output=True)
    assert refused.returncode != 0

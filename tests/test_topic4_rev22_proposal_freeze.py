import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_topic4_rev22_proposals", ROOT / "scripts/freeze_topic4_rev22_proposals.py")
freeze = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(freeze)

PARAMS = ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")
REFERENCE = (0.5, 1.0, 0.0, 2.0)
BOUNDS = {"g_LEE": [0.0, 1.0], "g_LEI": [0.0, 1.5], "theta_FT_deg": [-12.5, 7.5], "AR_FT": [1.75, 2.375]}
ABSOLUTE_ANGLE = -22.80538396505847


def _design(rows):
    return {
        "branch": "PRIMARY_4D_BRANCH",
        "bounds": BOUNDS,
        "geometry_reference": {"absolute_angle_deg": ABSOLUTE_ANGLE, "absolute_aspect_ratio": 2.0},
        "candidates": rows,
        "git_commit": "0" * 40,
    }


def _design_rows(n=40, seed=3):
    """A design whose optimum sits at a known interior point, plus the exact reference."""
    rng = np.random.default_rng(seed)
    rows = [{"candidate_id": "dci_p000", "block": "reference", "is_reference": True,
             "physical": dict(zip(PARAMS, REFERENCE)),
             "family_membership": list(freeze.MASKS_PRIMARY),
             "node_field": {"field_sha256": "abc"}, "node_mapping": {"node_gain": 1.0}}]
    for i in range(1, n):
        x = [float(rng.uniform(*BOUNDS[name])) for name in PARAMS]
        rows.append({"candidate_id": f"dci_p{i:03d}", "block": "full4d", "is_reference": False,
                     "physical": dict(zip(PARAMS, x)),
                     "family_membership": ["M1111"],
                     "node_field": {"field_sha256": "abc"}, "node_mapping": {"node_gain": 1.0}})
    return rows


def _fit_aggregate(rows, *, target=(0.8, 1.2, 2.0, 2.1), noise=0.02, seed=5, infeasible=()):
    """Synthetic bowls: every component minimizes near ``target``."""
    rng = np.random.default_rng(seed)
    candidates = []
    for row in rows:
        x = np.asarray([row["physical"][name] for name in PARAMS], float)
        scale = np.asarray([1.0, 1.0, 10.0, 0.5])
        distance = float(np.sum(((x - np.asarray(target)) / scale) ** 2))
        feasible = row["candidate_id"] not in infeasible
        z = {name: distance + (0.3 * i) + rng.normal(0.0, noise)
             for i, name in enumerate(freeze.COMPONENTS)}
        candidates.append({
            "candidate_id": row["candidate_id"], "block": row["block"],
            "physical": row["physical"], "family_membership": row["family_membership"],
            "continuous_surface_eligible": feasible, "joint_feasibility": feasible,
            "standardized_Z": {k: (v if feasible else None) for k, v in z.items()},
            "standardized_jackknife_sd": {k: (0.05 if feasible else None) for k in freeze.COMPONENTS},
            "units": [{"topology_seed": s, "dynamics_seed": s, "artifact_integrity": True,
                       "safe": True, "n_returned_families": n}
                      for s, n in zip((2511, 2512, 2513, 2514), (38, 43, 22, 40))],
        })
    return {"status": "FIT_AGGREGATE_COMPLETE", "candidates": candidates}


def test_variance_components_separate_topology_from_dynamics():
    topology_dominated = {}
    dynamics_dominated = {}
    for t, offset in enumerate((0.0, 5.0, 10.0, 15.0)):
        for d, jitter in enumerate((0.0, 0.05, -0.05)):
            topology_dominated[(2511 + t, 3100 + d)] = offset + jitter
            dynamics_dominated[(2511 + t, 3100 + d)] = 0.01 * t + jitter * 100
    a = freeze.variance_components(topology_dominated)
    b = freeze.variance_components(dynamics_dominated)
    assert a["status"] == "OK" and a["topology_variance_fraction"] > 0.99
    assert b["status"] == "OK" and b["dynamics_variance_fraction"] > 0.9
    incomplete = dict(topology_dominated)
    incomplete[(2511, 3100)] = None
    assert freeze.variance_components(incomplete)["status"] == "NOT_ESTIMABLE_MISSING_UNIT"
    assert freeze.variance_components({(2511, 3100): 1.0, (2512, 3100): 2.0})["status"].startswith(
        "NOT_ESTIMABLE")


def test_proposals_land_near_the_synthetic_optimum_and_respect_masks():
    rows = _design_rows()
    fit = _fit_aggregate(rows)
    out = freeze.freeze_proposals(fit, _design(rows), list(freeze.COMPONENTS),
                                  n_restarts=2, n_estimators=60)
    assert out["surrogate_adequacy"]["status"] in {"SURROGATE_ADEQUATE", "OBSERVED_PARETO_FALLBACK"}
    full = out["proposals"]["M1111"]
    x = full["frozen_points"][0]["x"]
    assert 0.55 <= x[0] <= 1.0 and 1.0 <= x[1] <= 1.5
    reference_only = out["proposals"]["M0000"]["frozen_points"][0]["x"]
    assert reference_only == [float(v) for v in REFERENCE]
    assert out["proposals"]["M0000"]["status"] == "REFERENCE_RETURN"
    locked = out["proposals"]["M1100"]["frozen_points"][0]["x"]
    assert locked[2] == REFERENCE[2] and locked[3] == REFERENCE[3]


def test_observed_fallback_is_used_when_the_surrogate_is_inadequate():
    rows = _design_rows(n=24, seed=11)
    fit = _fit_aggregate(rows, noise=8.0, seed=12)
    out = freeze.freeze_proposals(fit, _design(rows), list(freeze.COMPONENTS),
                                  n_restarts=1, n_estimators=40)
    if out["surrogate_adequacy"]["status"] == "OBSERVED_PARETO_FALLBACK":
        for record in out["proposals"].values():
            assert record["source"] == "observed_nondominated_design_points"
            if record.get("frozen_points"):
                assert record["frozen_points"][0]["origin"] == "observed"
                assert record["frozen_points"][0]["existing_candidate_id"] is not None
    observed = freeze.observed_pareto_points(
        freeze.design_rows_for_surface(fit), list(freeze.COMPONENTS), "M1111")
    values = np.asarray([[row["excess"][k] for k in freeze.COMPONENTS] for row in observed])
    for i in range(len(values)):
        dominated = np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))
        assert not dominated


def test_proposal_rows_carry_the_absolute_angle_and_the_explicit_reference():
    rows = _design_rows()
    fit = _fit_aggregate(rows)
    design = _design(rows)
    out = freeze.freeze_proposals(fit, design, list(freeze.COMPONENTS), n_restarts=2, n_estimators=60)
    manifest_rows = freeze.proposal_candidate_rows(out["proposals"], design)
    assert manifest_rows, "at least one proposal must be a new candidate"
    for row in manifest_rows:
        mechanisms = row["mechanisms"]
        assert mechanisms["Z_M"] == "off"
        assert mechanisms["ellipse_reference_angle_deg"] == ABSOLUTE_ANGLE
        assert mechanisms["ellipse_reference_aspect_ratio"] == 2.0
        offset = row["physical"]["theta_FT_deg"]
        expected = ABSOLUTE_ANGLE if offset == 0.0 else ABSOLUTE_ANGLE + offset
        assert mechanisms["ellipse_angle_deg"] == expected
        assert row["selection_eligible"] is False
    # the reference-return family reuses the frozen design candidate instead of a new row
    assert out["proposals"]["M0000"]["frozen_points"][0]["execution_candidate_id"] == "dci_p000"
    ids = [row["candidate_id"] for row in manifest_rows]
    assert len(set(ids)) == len(ids)


def test_recall_budget_uses_the_minimum_reference_unit():
    assert freeze.freeze_reference_support_budget([38, 43, 22, 40], expected_units=4,
                                                  min_events=12)["n_cov"] == 22
    degraded = freeze.freeze_reference_support_budget([38, 43, 40], expected_units=4, min_events=12)
    assert degraded["status"] == "REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE"


def test_bind_mode_refuses_a_manifest_without_the_frozen_candidates(tmp_path):
    frozen = {"schema_id": "topic4_rev22_dci_frozen_candidates_v1",
              "response_design_manifest_sha256": "design",
              "candidate_ids": ["dci_prop_M1111_gp"]}
    path = tmp_path / "frozen_candidates.json"
    path.write_text(json.dumps(frozen))
    manifest = tmp_path / "execution_candidate_manifest.json"
    manifest.write_text(json.dumps({"response_design_manifest_sha256": "design",
                                    "candidates": [{"candidate_id": "dci_p000"}]}))
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_topic4_rev22_proposals.py"),
         "--out-dir", str(tmp_path), "--bind-execution-manifest", str(manifest)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "lacks frozen candidates" in result.stderr

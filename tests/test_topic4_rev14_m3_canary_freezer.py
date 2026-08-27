from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import freeze_topic4_rev14_m3_canary as freezer
from src.topic4_core_field import (
    core_thresholds,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_rev14_fourier_field import (
    array_sha256,
    evaluate,
    mode_inventory,
    uniform_quadrature,
    weighted_centered_surface_rms,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev14_m3_canary.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_m3_freezer_builds_exactly_32_selectable_and_two_benchmarks():
    candidates, audit = freezer.build_candidates(_config())

    assert len(candidates) == 34
    assert sum(row["selection_eligible"] for row in candidates) == 32
    assert [row["candidate_id"] for row in candidates[:2]] == [
        "exact_off", "uniform_node",
    ]
    assert all(not row["selection_eligible"] for row in candidates[:2])
    assert candidates[0]["fourier_coordinate"] is None
    assert candidates[1]["field_kind"] == "zero_fourier_uniform_benchmark"
    assert audit["observation_geometry_used"] is False
    assert audit["predeclared_object_basis_used"] is False


def test_sobol_directions_are_orthogonal_and_observation_free():
    design = _config()["m3_design"]
    directions, audit = freezer.orthogonal_sobol_directions(design)

    assert directions.shape == (8, 28)
    assert np.allclose(directions @ directions.T, np.eye(8), atol=1e-13)
    assert audit["maximum_gram_error"] < 1e-12
    generation = json.dumps({
        "inputs": audit["candidate_generation_inputs"],
        "geometry": audit["observation_geometry_used"],
        "objects": audit["predeclared_object_basis_used"],
    }).lower()
    assert not any(token in generation for token in freezer.FORBIDDEN_GENERATION_TOKENS)


def test_every_selectable_coordinate_has_frozen_centered_surface_rms():
    config = _config()
    candidates, _ = freezer.build_candidates(config)
    modes = mode_inventory(3)
    positions, weights = uniform_quadrature(128, L=20.0)

    for candidate in candidates:
        if not candidate["selection_eligible"]:
            continue
        coordinate = candidate["fourier_coordinate"]
        coefficients = np.asarray(coordinate["coefficients"], dtype=np.float64)
        observed = weighted_centered_surface_rms(
            evaluate(coefficients, positions, modes, L=20.0), weights,
        )
        assert observed == pytest.approx(
            coordinate["target_centered_surface_rms"], abs=2e-12,
        )
        assert coordinate["absolute_field_not_exact_off_residual"] is True
        assert coordinate["observation_geometry_used"] is False


def test_sign_pairs_are_exact_negatives_at_each_direction_and_rms():
    candidates, _ = freezer.build_candidates(_config())
    indexed = {
        (
            row["fourier_coordinate"]["direction_index"],
            row["fourier_coordinate"]["target_centered_surface_rms"],
            row["fourier_coordinate"]["sign"],
        ): np.asarray(row["fourier_coordinate"]["coefficients"], dtype=np.float64)
        for row in candidates if row["selection_eligible"]
    }
    for direction in range(8):
        for rms in (0.8, 1.4):
            assert np.array_equal(
                indexed[(direction, rms, -1)], -indexed[(direction, rms, 1)],
            )


def test_candidate_coordinates_are_bitwise_stable_across_blas_thread_counts():
    command = [
        sys.executable, "-c",
        (
            "import json; "
            "from scripts.freeze_topic4_rev14_m3_canary import build_candidates; "
            "config=json.load(open('config/topic4_rev14_m3_canary.json')); "
            "rows,_=build_candidates(config); "
            "print(json.dumps([r.get('fourier_coordinate', {}) and "
            "r['fourier_coordinate'].get('coefficients_sha256') for r in rows]))"
        ),
    ]
    outputs = []
    for threads in (None, "1", "4"):
        environment = os.environ.copy()
        for name in (
                "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            if threads is None:
                environment.pop(name, None)
            else:
                environment[name] = threads
        outputs.append(subprocess.check_output(
            command, cwd=ROOT, env=environment, text=True,
        ).strip())
    assert outputs[0] == outputs[1] == outputs[2]


def test_original_signed_depth_formula_and_array_hash_are_frozen():
    config = _config()
    contract = config["node_mapping"]["signed_depth_contract"]
    values = signed_depth(core_thresholds(
        sample_core_quantiles(
            contract["expected_n_e"], contract["quantile_seed"],
        ),
        contract["core_mean_mV"],
        contract["core_std_mV"],
    ), contract["v_base_mV"])
    audit = freezer.frozen_signed_depth_audit(config)

    assert array_sha256(values) == contract["sha256"]
    assert audit["sha256"] == contract["sha256"]
    assert audit["position_or_exact_off_geometry_used"] is False
    assert audit["source"] == "original_frozen_quantile_draw"


def test_prepare_only_builds_contract_without_writing_manifest(monkeypatch, capsys):
    def forbidden_write(*args, **kwargs):
        raise AssertionError("prepare-only must not write the candidate manifest")

    monkeypatch.setattr(freezer, "_atomic_json", forbidden_write)
    freezer.main([
        "--config", str(CONFIG),
        "--artifact-root", str(ARTIFACT_ROOT),
        "--prepare-only",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == freezer.PREPARE_STATUS
    assert payload["n_candidates"] == 34
    assert payload["n_selectable"] == 32
    assert payload["output_written"] is False


def test_formal_provenance_fails_closed_when_runtime_path_is_dirty(monkeypatch):
    monkeypatch.setattr(freezer, "_git", lambda arguments, text=True: (
        "a" * 40 + "\n" if arguments[:2] == ["git", "rev-parse"] else ""
    ))
    monkeypatch.setattr(freezer, "_path_provenance", lambda relative, expected: {
        "observed_sha256": "b" * 64,
        "tracked": True,
        "dirty": relative.endswith("m3_canary.json"),
        "expected_sha256": "b" * 64,
        "matches_expected_commit": True,
    })
    with pytest.raises(RuntimeError, match="not clean and frozen"):
        freezer.runtime_provenance(
            CONFIG, expected_commit="HEAD", require_clean=True,
        )


def test_real_frozen_inputs_rebuild_exact_off_without_observation_target():
    provenance = {
        "git_commit": "prepare-only", "formal_ready": False,
    }
    payload = freezer.build_manifest_payload(
        CONFIG, artifact_root=ARTIFACT_ROOT,
        provenance=provenance, status=freezer.PREPARE_STATUS,
    )
    reconstruction = payload["exact_off_reconstruction"]
    assert reconstruction["primary_substrate_id"] == (
        "stage_ak_mean_g10_p_disp_g08_m"
    )
    assert payload["pathways"] == freezer.EXPECTED_PATHWAYS
    assert payload["signed_depth_audit"]["sha256"] == (
        _config()["node_mapping"]["signed_depth_contract"]["sha256"]
    )
    assert "patient" not in json.dumps(payload["inputs"]).lower()

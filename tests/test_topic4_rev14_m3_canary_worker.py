from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import freeze_topic4_rev14_m3_canary as freezer
from scripts import run_topic4_rev14_m3_canary_worker as worker
from src.topic4_core_field import (
    core_thresholds,
    project_to_budget,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_rev14_fourier_field import array_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev14_m3_canary.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _candidates() -> dict[str, dict]:
    rows, _ = freezer.build_candidates(_config())
    return {row["candidate_id"]: row for row in rows}


def _substrate(*, reverse_geometry: bool = False):
    rng = np.random.default_rng(991)
    n_e, n_i = 96, 24
    positions = rng.uniform(0.0, 20.0, size=(n_e, 2))
    q = np.exp(0.8 * np.sin(positions[:, 0] / 3.0))
    h, _ = project_to_budget(q, target_count=31.0)
    if reverse_geometry:
        h = h[::-1].copy()
    quantile_seed = 77
    core_mean, core_std = 17.5, 1.0
    signed_depth_values = signed_depth(core_thresholds(
        sample_core_quantiles(n_e, quantile_seed), core_mean, core_std,
    ))
    v_base = 18.0
    vtheta = np.full(n_e + n_i, v_base, dtype=np.float64)
    vtheta[:n_e] = v_base - h * signed_depth_values
    substrate = SimpleNamespace(
        h_e=h, h_i=np.zeros(n_i), vtheta=vtheta,
        delta_vtheta=vtheta[:n_e] - v_base,
        n_e=n_e, n_i=n_i, positions_e=positions,
        engine={
            "v_base": v_base, "L": 20.0,
            "core_mean": core_mean, "core_std": core_std,
        },
        stage={"N_core_manual": 31.0, "quantile_seed": quantile_seed},
        edge_coefficients=np.zeros((4, 4), dtype=np.float64),
        extras={},
    )
    config = _config()
    config["node_mapping"]["expected_target_h_mass"] = 31.0
    config["node_mapping"]["signed_depth_contract"] = {
        "expected_n_e": n_e,
        "quantile_seed": quantile_seed,
        "core_mean_mV": core_mean,
        "core_std_mV": core_std,
        "v_base_mV": v_base,
        "sha256": array_sha256(signed_depth_values),
    }
    return substrate, np.asarray(signed_depth_values), config


def test_zero_fourier_coefficients_produce_uniform_h_not_exact_off():
    substrate, _, config = _substrate()
    exact_h = substrate.h_e.copy()
    projection = worker._project_candidate(
        _candidates()["uniform_node"], substrate, config,
    )

    expected = substrate.stage["N_core_manual"] / substrate.n_e
    assert np.allclose(projection["h"], expected, atol=1e-14)
    assert not np.allclose(projection["h"], exact_h)
    assert projection["audit"]["mapping"] == (
        "absolute_zero_fourier_uniform_mass_projection"
    )
    assert projection["audit"]["candidate_h_inherits_exact_off_geometry"] is False


def test_m3_h_is_independent_of_exact_off_h_geometry():
    first, signed_depth_values, config = _substrate(reverse_geometry=False)
    second, _, _ = _substrate(reverse_geometry=True)
    candidate = _candidates()["m3_d03_m_r14"]
    left = worker._project_candidate(candidate, first, config)
    right = worker._project_candidate(candidate, second, config)

    assert np.array_equal(left["h"], right["h"])
    assert np.array_equal(left["signed_depth"], signed_depth_values)
    assert np.array_equal(left["signed_depth"], right["signed_depth"])
    assert left["hashes"]["h_sha256"] == right["hashes"]["h_sha256"]
    assert np.array_equal(left["vtheta"], right["vtheta"])


def test_exact_off_is_separate_benchmark_that_bypasses_fourier():
    substrate, _, config = _substrate()
    h = substrate.h_e.copy()
    vtheta = substrate.vtheta.copy()
    projection = worker._project_candidate(
        _candidates()["exact_off"], substrate, config,
    )

    assert np.array_equal(projection["h"], h)
    assert np.array_equal(projection["vtheta"], vtheta)
    assert projection["audit"]["exact_off_bypasses_fourier"] is True
    assert projection["audit"]["candidate_h_inherits_exact_off_geometry"] is True
    assert projection["audit"]["mapping"] == "stage_ak_exact_off_bypass_fourier"
    assert projection["audit"]["exact_off_bypass_h_max_abs_error"] == 0.0
    assert projection["audit"]["exact_off_bypass_vtheta_max_abs_mV"] == 0.0
    assert projection["audit"]["exact_off_mapping_contract"] == (
        "historical_dual_continuous_mean_dispersion_bypass"
    )
    assert np.isfinite(projection["audit"][
        "exact_off_vs_original_signed_depth_mapping_max_abs_mV"
    ])


def test_absolute_m3_projection_preserves_mass_and_frozen_signed_depth():
    substrate, signed_depth_values, config = _substrate()
    candidate = _candidates()["m3_d00_p_r08"]
    projection = worker._project_candidate(
        candidate, substrate, config,
    )

    assert projection["h"].sum() == pytest.approx(31.0, abs=1e-8)
    assert np.all((projection["h"] > 0.0) & (projection["h"] < 1.0))
    assert np.array_equal(projection["signed_depth"], signed_depth_values)
    assert np.allclose(
        projection["vtheta"][:substrate.n_e],
        18.0 - projection["h"] * signed_depth_values,
        atol=1e-14,
    )
    assert projection["audit"]["centered_latent_surface_rms"] == pytest.approx(
        0.8, abs=2e-12,
    )


def test_npz_augmentation_preserves_exact_off_rescorer_event_arrays():
    substrate, _, config = _substrate()
    candidate = _candidates()["m3_d00_p_r08"]
    projection = worker._project_candidate(
        candidate, substrate, config,
    )
    state = SimpleNamespace(
        projection=projection, candidate=candidate,
        config=config,
    )
    original = {
        "onsets": np.ones((2, 15), dtype=np.float32),
        "ranks": np.ones((2, 15), dtype=np.float32),
        "event_t_on_ms": np.asarray([1.0, 2.0], dtype=np.float32),
        "event_returned": np.asarray([True, False]),
        "source_onset_maps_ms": np.ones((2, 3, 3), dtype=np.float32),
    }
    augmented = worker._augment_npz_arrays(original, state)

    for key, value in original.items():
        assert np.array_equal(augmented[key], value)
    assert augmented["rev14_fourier_coefficients"].shape == (14, 2)
    assert np.array_equal(
        augmented["rev14_frozen_signed_depth"], projection["signed_depth"]
    )
    assert str(augmented["rev14_projection_sha256"]) == (
        projection["hashes"]["projection_sha256"]
    )


def test_json_augmentation_keeps_pathways_off_and_records_projection():
    substrate, _, config = _substrate()
    candidate = _candidates()["uniform_node"]
    projection = worker._project_candidate(
        candidate, substrate, config,
    )
    state = SimpleNamespace(
        projection=projection, candidate=candidate, config=config,
        provenance={"formal_ready": True}, manifest_audit={"match": True},
        compatibility_audit={"patient_target_input_loaded": False},
    )
    payload = {
        "status": "REV12ND_NODE_WORKER_COMPLETE",
        "mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": True,
        },
        "provenance": {},
    }
    output = worker._augment_json_payload(payload, state)

    assert output["status"] == worker.WORKER_STATUS
    assert output["base_worker_status"] == "REV12ND_NODE_WORKER_COMPLETE"
    assert output["mechanism_freeze"]["static_node_field"] == (
        "zero_fourier_uniform_benchmark"
    )
    assert output["node_mapping"]["candidate_h_inherits_exact_off_geometry"] is False
    assert output["field_sha256"] == projection["hashes"]["h_sha256"]


def test_prepare_only_never_calls_composed_worker(monkeypatch, capsys):
    monkeypatch.setattr(
        worker, "_run_rev12_composed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare-only must not enter the SNN worker")
        ),
    )
    worker.main([
        "--config", str(CONFIG),
        "--candidate-id", "m3_d00_p_r08",
        "--seed", "2321",
        "--artifact-root", str(ARTIFACT_ROOT),
        "--prepare-only",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == worker.PREPARE_STATUS
    assert payload["snn_started"] is False
    assert payload["duration_ms"] == 20000.0


def test_prepare_only_rejects_seed_outside_frozen_pool():
    with pytest.raises(SystemExit):
        worker._preflight([
            "--config", str(CONFIG),
            "--candidate-id", "m3_d00_p_r08",
            "--seed", "2322",
            "--artifact-root", str(ARTIFACT_ROOT),
            "--prepare-only",
        ])


def test_composed_interface_calls_frozen_rev12_path_once():
    substrate, _, config = _substrate()
    candidate = _candidates()["m3_d00_p_r08"]
    captured = {"simulate_calls": 0}
    fake = SimpleNamespace()
    fake.json = json

    def original_build(*args, **kwargs):
        return copy.deepcopy(substrate)

    def original_simulate(*args, **kwargs):
        captured["simulate_calls"] += 1
        assert kwargs["slow"] is None
        return {"E_spk_bool": np.zeros((1, substrate.n_e), dtype=bool)}

    def original_npz(path, **arrays):
        captured["npz"] = arrays

    def original_json(payload, path):
        captured["json"] = payload

    fake.build_substrate = original_build
    fake.simulate_kick = original_simulate
    fake._atomic_npz = original_npz
    fake.atomic_write_json = original_json

    def fake_main():
        built = fake.build_substrate(
            {}, "node_baseline", 2321, ee_dose=0.0, etoi_dose=0.0,
        )
        fake.simulate_kick(None, None, None, slow=None, node_accessibility=None)
        fake._atomic_npz(
            Path("unused.npz"),
            onsets=np.zeros((0, 15), dtype=np.float32),
            ranks=np.zeros((0, 15), dtype=np.float32),
            h=np.asarray(built.h_e),
            delta_vtheta=np.asarray(built.delta_vtheta),
        )
        fake.atomic_write_json({
            "status": "REV12ND_NODE_WORKER_COMPLETE",
            "mechanism_freeze": {
                "EE": "off", "E_to_I": "off", "Z_M": "off",
                "edge_coefficients_all_zero": True,
            },
            "provenance": {},
        }, Path("unused.json"))

    fake.main = fake_main
    state = worker._RunState(
        config=config, candidate=candidate, manifest={},
        provenance={"formal_ready": True}, manifest_audit={"match": True},
        compatibility_config={}, compatibility_manifest={},
        compatibility_audit={"patient_target_input_loaded": False},
        source_config_text="CONFIG", source_manifest_text="MANIFEST",
    )
    worker._run_rev12_composed(state, base_module=fake)

    assert captured["simulate_calls"] == 1
    assert captured["json"]["status"] == worker.WORKER_STATUS
    assert captured["npz"]["onsets"].shape == (0, 15)
    assert captured["npz"]["rev14_fourier_coefficients"].shape == (14, 2)
    assert not np.array_equal(captured["npz"]["h"], substrate.h_e)


def test_physical_dose_table_covers_all_32_selectable_candidates():
    substrate, signed_depth_values, config = _substrate()
    candidates, audit = freezer.build_candidates(_config())
    manifest = {"candidates": candidates, "direction_audit": audit}

    table = worker.build_physical_dose_table(
        manifest, substrate, config, seed=2321,
    )

    assert table["simulation_run"] is False
    assert table["n_selectable"] == 32
    assert table["n_benchmarks"] == 2
    assert len(table["selectable_candidates"]) == 32
    assert {row["candidate_id"] for row in table["nonselectable_benchmarks"]} == {
        "exact_off", "uniform_node",
    }
    assert {row["frozen_signed_depth_sha256"] for row in (
        table["selectable_candidates"] + table["nonselectable_benchmarks"]
    )} == {array_sha256(signed_depth_values)}
    assert all(row["target_h_mass"] == pytest.approx(31.0) for row in (
        table["selectable_candidates"]
    ))
    assert len(table["table_sha256"]) == 64


def test_worker_composes_over_rev12_and_does_not_import_a_simulator():
    source = (ROOT / "scripts/run_topic4_rev14_m3_canary_worker.py").read_text()
    assert "from scripts import run_topic4_rev12_node_worker as rev12" in source
    assert "from kick_probe import" not in source
    assert "project_fourier_to_frozen_node" in source
    assert "def _direct_absolute_h" not in source
    assert "(v_base - exact_vtheta[:n_e]) / exact_h" not in source
    assert "sample_core_quantiles" in source
    assert "candidate_h_inherits_exact_off_geometry" in source

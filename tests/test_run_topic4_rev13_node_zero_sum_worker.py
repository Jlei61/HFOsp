import json
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_topic4_rev13_node_zero_sum_worker as worker


def _candidate(mode="exact_off", **controller):
    if mode == "exact_off" and not controller:
        return {"candidate_id": "c0", "node_accessibility": None}
    payload = {"mode": mode, **controller}
    return {"candidate_id": "c0", "node_accessibility": payload}


def test_scientific_role_is_rev13_only_and_patient_inputs_are_rejected():
    worker._validate_scientific_role(worker.SCIENTIFIC_ROLE)
    with pytest.raises(RuntimeError, match="scientific role"):
        worker._validate_scientific_role("development_only_node_dualmode_refit")
    with pytest.raises(RuntimeError, match="patient target"):
        worker._assert_no_patient_runtime_inputs({
            "inputs": {"patient_prototype": {"path": "results/target.npz"}}
        })
    with pytest.raises(RuntimeError, match="forbidden runtime key"):
        worker._normalized_controller_spec(_candidate(
            "zero_sum", patient_mode_label="TA"
        ))
    with pytest.raises(RuntimeError, match="forbidden patient"):
        worker._assert_candidate_is_patient_free({
            "candidate_id": "c0", "source": "patient_prototype_library",
        })


def test_controller_spec_accepts_exact_off_and_nested_active_payload():
    off = worker._normalized_controller_spec(_candidate())
    assert off == {"mode": "exact_off", "enabled": False}
    active = worker._normalized_controller_spec({
        "node_accessibility": {
            "enabled": True,
            "controller": {
                "kind": "field_gated_bounded_node_recovery",
                "mode": "zero_sum", "tau_ms": 250.0,
                "a_max_multiplier": 2.0,
                "draws_random_numbers_at_runtime": False,
            },
        }
    })
    assert active["enabled"] is True
    assert active["mode"] == "zero_sum"


def test_real_freezer_candidate_schema_is_accepted_without_patient_fields():
    import scripts.freeze_topic4_rev13_node_zero_sum_recovery as freezer

    config = json.loads((worker.ROOT / "config/topic4_rev13_node_zero_sum_recovery.json").read_text())
    stage_config = json.loads((
        worker.ROOT / config["inputs"]["stage_ak_config"]["path"]
    ).read_text())
    artifact = Path("/home/honglab/leijiaxin/HFOsp")
    stage_ak = json.loads((
        artifact / config["inputs"]["stage_ak_manifest"]["path"]
    ).read_text())
    stage_al = json.loads((
        artifact / config["inputs"]["stage_al_manifest"]["path"]
    ).read_text())
    candidates, _ = freezer.build_candidates(
        stage_ak, stage_al, stage_config, config
    )
    normalized = {
        row["candidate_id"]: worker._normalized_controller_spec(row)
        for row in candidates
    }
    assert normalized["exact_off"] == {"mode": "exact_off", "enabled": False}
    assert normalized["zero_sum_c020"]["c"] == pytest.approx(0.2)
    assert normalized["spatial_shift_c020"]["mode"] == "spatial_shift"
    assert normalized["spatial_shift_c020"]["spatial_shift"][
        "mapping_method"
    ] == "rank_matched_equal_width_spatial_blocks_toroidal_shift"
    worker._assert_no_patient_runtime_inputs(config)


def test_compatibility_view_adds_only_transition_and_contact_readout():
    config = json.loads((worker.ROOT / "config/topic4_rev13_node_zero_sum_recovery.json").read_text())
    compatibility, audit = worker._compatibility_view(
        config, artifact_root=Path("/home/honglab/leijiaxin/HFOsp")
    )
    assert set(compatibility["inputs"]) == {"transition_config"}
    assert "contact_readout" in compatibility["search"]
    assert "classifier_config" in audit["excluded_stage_ak_inputs"]
    assert audit["patient_label_or_prototype_input_loaded"] is False


def test_transition_verifier_does_not_open_target_or_classifier_inputs(monkeypatch):
    transition = json.loads((
        worker.ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"
    ).read_text())
    state = worker._RunState(
        candidate=_candidate(),
        controller_spec={"mode": "exact_off", "enabled": False},
    )
    opened = []
    original = worker.transition_module._sha256_file

    def tracked(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(worker.transition_module, "_sha256_file", tracked)
    audit = worker._verify_rev13_substrate_inputs(
        transition,
        artifact_root=Path("/home/honglab/leijiaxin/HFOsp"),
        state=state,
    )
    assert set(audit["records"]) == worker.SUBSTRATE_INPUT_KEYS
    assert not any("shaft_aware_patient_training_target.npz" in path for path in opened)
    assert not any("direction_classifier" in path for path in opened)
    assert "shaft_aware_target_npz" in state.compatibility_audit[
        "excluded_transition_inputs"
    ]


def test_stratified_permutation_is_deterministic_complete_and_hash_checked():
    support = np.linspace(0.0, 1.0, 24)
    static = np.linspace(-1.0, 1.0, 24)
    contract = {
        "seed": 17,
        "support_quantile_bins": 3,
        "signed_depth_quantile_bins": 4,
    }
    first = worker._stratified_permutation(support, static, contract)
    second = worker._stratified_permutation(support, static, contract)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(24))
    contract["permutation_sha256"] = worker._sha256_array(first)
    assert np.array_equal(
        worker._stratified_permutation(support, static, contract), first
    )
    contract["permutation_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash changed"):
        worker._stratified_permutation(support, static, contract)


def test_spatial_shift_mapping_is_config_defined_complete_and_amplitude_matched():
    positions = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
    contract = {
        "mapping_method": "rank_matched_equal_width_spatial_blocks_toroidal_shift",
        "block_shape": [4, 1],
        "shift_blocks": [2, 0],
        "within_block_order": "y_then_x_then_original_index",
        "permutation_scope": "all_E_neurons",
        "dynamic_scale_match": (
            "support_weighted_centered_SD_each_step_to_unshifted_zero_sum"
        ),
    }
    permutation = worker._spatial_shift_permutation(
        positions, support_g=np.ones(8), L=8.0, contract=contract
    )
    assert np.array_equal(np.sort(permutation), np.arange(8))
    assert np.array_equal(permutation, np.array([4, 5, 6, 7, 0, 1, 2, 3]))
    contract["permutation_sha256"] = worker._sha256_array(permutation)
    assert np.array_equal(
        worker._spatial_shift_permutation(
            positions, support_g=np.ones(8), L=8.0, contract=contract
        ),
        permutation,
    )

    support = np.ones(8)
    shifted = worker.FieldGatedZeroSumNodeRecovery(
        support, dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.2,
        mode="spatial_shift", spatial_shift_permutation=permutation,
    )
    reference = worker.FieldGatedZeroSumNodeRecovery(
        support, dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.2,
        mode="zero_sum",
    )
    state = np.arange(8, dtype=float) / 10.0
    shifted.state_mV[:] = state
    reference.state_mV[:] = state
    shifted_delta = shifted.delta_theta()
    reference_delta = reference.delta_theta()
    assert np.sum(shifted_delta) == pytest.approx(0.0, abs=1e-12)
    assert worker._support_weighted_centered_sd(
        support, shifted_delta
    ) == pytest.approx(
        worker._support_weighted_centered_sd(support, reference_delta)
    )
    assert not np.array_equal(shifted_delta, reference_delta)


def test_threshold_recorder_saves_bounds_zero_sum_saturation_and_sign_flips():
    controller = worker.FieldGatedZeroSumNodeRecovery(
        np.ones(4), dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.2,
        mode="zero_sum", trace_dt_ms=0.1,
    )
    controller.state_mV[:] = [0.4, 0.0, 0.0, 0.0]
    recorder = worker._ThresholdRecorder(
        controller=controller,
        static_modulation=np.array([-0.05, 0.05, 0.05, 0.05]),
        reset_mV=11.0,
    )
    effective = recorder.threshold(np.full(5, 18.0))
    arrays = recorder.trace_arrays()
    assert effective.shape == (5,)
    assert arrays["threshold_min_mV"].size == 1
    assert arrays["zero_sum_error_mV"][0] < 1e-12
    assert arrays["saturation_fraction"][0] == pytest.approx(0.25)
    assert arrays["static_modulation_sign_flip_fraction"][0] > 0.0
    assert arrays["legacy_unweighted_saturation_fraction"][0] == pytest.approx(0.25)
    diagnostics = recorder.diagnostics()
    assert diagnostics["threshold_min_mV_observed"] > 12.0
    assert diagnostics["maximum_zero_sum_error_mV"] < 1e-12
    assert diagnostics["maximum_support_weighted_saturation_fraction"] == pytest.approx(0.25)
    assert diagnostics[
        "maximum_support_weighted_static_modulation_sign_flip_fraction"
    ] > 0.0
    assert diagnostics["static_amplitude_definition"] == "support_weighted_centered_sd"
    assert diagnostics["saturation_fraction"] == pytest.approx(0.25)
    assert diagnostics["legacy_unweighted_saturation_fraction"] == pytest.approx(0.25)


def test_primary_diagnostics_are_support_weighted_and_keep_legacy_values():
    support = np.array([0.01, 0.10, 0.30, 0.59])
    controller = worker.FieldGatedZeroSumNodeRecovery(
        support, dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.2,
        mode="zero_sum", trace_dt_ms=0.1,
    )
    recorder = worker._ThresholdRecorder(
        controller=controller,
        static_modulation=np.array([0.001, -1.0, 1.0, 1.0]),
        reset_mV=11.0,
    )
    assert np.array_equal(
        recorder.primary_diagnostic_mask, np.array([False, True, True, True])
    )
    indicator = np.array([True, True, False, False])
    assert recorder._support_weighted_fraction(indicator) == pytest.approx(
        0.10 / (0.10 + 0.30 + 0.59)
    )
    controller.state_mV[:] = [controller.a_max_mV, 0.0, 0.0, 0.0]
    recorder.threshold(np.full(5, 18.0))
    arrays = recorder.trace_arrays()
    assert arrays["saturation_fraction"][0] == 0.0
    assert arrays["legacy_unweighted_saturation_fraction"][0] == pytest.approx(0.25)
    diagnostics = recorder.diagnostics()
    assert diagnostics["saturation_fraction"] == 0.0
    assert diagnostics["legacy_unweighted_saturation_fraction"] == pytest.approx(0.25)


def test_threshold_recorder_fails_instead_of_clipping_at_reset_margin():
    controller = worker.FieldGatedZeroSumNodeRecovery(
        np.ones(2), dt_ms=0.1, tau_ms=250.0, a_ref_mV=2.0,
        mode="zero_sum", trace_dt_ms=0.1,
    )
    controller.state_mV[:] = [0.0, 4.0]
    recorder = worker._ThresholdRecorder(
        controller=controller,
        static_modulation=np.array([-1.0, 1.0]),
        reset_mV=11.0,
    )
    with pytest.raises(FloatingPointError, match="reset margin"):
        recorder.threshold(np.array([13.0, 13.0]))


class _FakeBase:
    """Small producer with the same four hook points as the rev12 worker."""

    def __init__(self, root: Path):
        self.root = root
        self._validate_scientific_role = lambda role: None
        self.build_substrate = self._build
        self.simulate_kick = self._simulate
        self._atomic_npz = self._npz
        self.atomic_write_json = self._json
        self.last_simulation_kwargs = None

    @staticmethod
    def _build(*args, **kwargs):
        return SimpleNamespace(
            edge_coefficients=np.zeros(3),
            n_e=2,
            vtheta=np.array([17.9, 18.1]),
            delta_vtheta=np.array([-0.1, 0.1]),
            params=SimpleNamespace(V_reset=11.0),
        )

    def _simulate(self, **kwargs):
        self.last_simulation_kwargs = kwargs
        return {"event_ids": np.array([3, 7])}

    @staticmethod
    def _npz(path, **arrays):
        np.savez(path, **arrays)

    @staticmethod
    def _json(payload, path):
        Path(path).write_text(json.dumps(payload))

    def main(self):
        self._validate_scientific_role(worker.SCIENTIFIC_ROLE)
        substrate = self.build_substrate(
            None, "node_baseline", ee_dose=0.0, etoi_dose=0.0
        )
        simulation = self.simulate_kick(slow=None)
        self._atomic_npz(
            self.root / "worker.npz",
            event_ids=simulation["event_ids"],
            event_t_on_ms=np.array([10.0, 20.0]),
            ranks=np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.atomic_write_json({
            "status": "REV12ND_NODE_WORKER_COMPLETE",
            "scientific_role": "old",
            "events": [{"event_index": 0}, {"event_index": 1}],
            "mechanism_freeze": {
                "EE": "off", "E_to_I": "off", "Z_M": "off",
                "edge_coefficients_all_zero": True,
            },
            "provenance": {},
        }, self.root / "worker.json")


def test_off_composition_is_event_identical_and_adds_empty_dynamic_fields(tmp_path):
    direct_root = tmp_path / "direct"
    composed_root = tmp_path / "composed"
    direct_root.mkdir()
    composed_root.mkdir()
    direct = _FakeBase(direct_root)
    direct.main()
    base = _FakeBase(composed_root)
    state = worker._RunState(
        candidate=_candidate(),
        controller_spec={"mode": "exact_off", "enabled": False},
    )
    worker._run_rev12_composed(state, base_module=base)

    with np.load(direct_root / "worker.npz") as expected, np.load(
        composed_root / "worker.npz"
    ) as actual:
        for key in expected.files:
            assert np.array_equal(actual[key], expected[key])
        assert actual["node_accessibility_enabled"].item() is False
        assert actual["node_accessibility_trace_time_ms"].size == 0
    expected_json = json.loads((direct_root / "worker.json").read_text())
    actual_json = json.loads((composed_root / "worker.json").read_text())
    assert actual_json["events"] == expected_json["events"]
    assert actual_json["status"] == "REV13_NODE_ZERO_SUM_WORKER_COMPLETE"
    assert actual_json["base_worker_status"] == "REV12ND_NODE_WORKER_COMPLETE"
    assert actual_json["node_accessibility"]["enabled"] is False
    assert actual_json["node_accessibility"]["diagnostics"] == {
        "threshold_min_mV_observed": 17.9,
        "threshold_max_mV_observed": 18.1,
        "maximum_zero_sum_error_mV": 0.0,
        "maximum_saturation_fraction": 0.0,
        "maximum_support_weighted_saturation_fraction": 0.0,
        "maximum_static_modulation_sign_flip_fraction": 0.0,
        "maximum_support_weighted_static_modulation_sign_flip_fraction": 0.0,
    }
    assert base.last_simulation_kwargs["node_accessibility"] is None


def test_active_npz_and_json_augmentation_cannot_drop_controller_fields():
    controller = worker.FieldGatedZeroSumNodeRecovery(
        np.ones(2), dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.1,
        mode="zero_sum", trace_dt_ms=0.1,
    )
    recorder = worker._ThresholdRecorder(
        controller=controller,
        static_modulation=np.array([-0.2, 0.2]),
        reset_mV=11.0,
    )
    recorder.threshold(np.array([18.0, 18.0]))
    state = worker._RunState(
        candidate=_candidate("zero_sum"),
        controller_spec={"mode": "zero_sum", "enabled": True},
        controller=controller,
        recorder=recorder,
        support_audit={"support_sha256": "abc"},
        static_centered_sd_mV=0.2,
        substrate=SimpleNamespace(
            n_e=2,
            delta_vtheta=np.array([-0.2, 0.2]),
            vtheta=np.array([17.8, 18.2]),
        ),
    )
    arrays = worker._augment_npz_arrays({"event_t_on_ms": np.array([1.0])}, state)
    assert np.array_equal(arrays["event_t_on_ms"], np.array([1.0]))
    assert arrays["node_accessibility_threshold_zero_sum_error_mV"].size == 1
    assert "node_accessibility_trace_saturation_fraction" in arrays
    assert arrays["node_accessibility_support_g"].shape == (2,)
    assert arrays["node_accessibility_state_final_mV"].shape == (2,)
    assert arrays[
        "node_accessibility_static_modulation_sign_flip_final"
    ].dtype == np.bool_
    payload = worker._augment_json_payload({
        "events": [{"event_index": 0}],
        "mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": True,
        },
    }, state)
    assert payload["events"] == [{"event_index": 0}]
    assert payload["node_accessibility"]["diagnostics"][
        "maximum_zero_sum_error_mV"
    ] < 1e-12
    assert "node_accessibility_threshold_static_modulation_sign_flip_fraction" in (
        payload["node_accessibility"]["trace_fields"]
    )
    assert payload["node_accessibility"]["static_amplitude_definition"] == (
        "support_weighted_centered_sd"
    )


def test_json_augmentation_rejects_any_active_edge_or_zm_mechanism():
    state = worker._RunState(
        candidate=_candidate(),
        controller_spec={"mode": "exact_off", "enabled": False},
    )
    with pytest.raises(RuntimeError, match="active learned edge"):
        worker._augment_json_payload({"mechanism_freeze": {
            "EE": "on", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": False,
        }}, state)
    with pytest.raises(RuntimeError, match="Z/M"):
        worker._augment_json_payload({"mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "on",
            "edge_coefficients_all_zero": True,
        }}, state)


def test_zero_sum_trace_must_stay_inside_the_frozen_tolerance():
    controller = worker.FieldGatedZeroSumNodeRecovery(
        np.ones(2), dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.1,
        mode="zero_sum", trace_dt_ms=0.1,
    )
    recorder = worker._ThresholdRecorder(
        controller=controller,
        static_modulation=np.array([-0.2, 0.2]),
        reset_mV=11.0,
    )
    recorder.threshold(np.array([18.0, 18.0]))
    state = worker._RunState(
        candidate=_candidate("zero_sum"),
        controller_spec={
            "mode": "zero_sum", "enabled": True,
            "zero_sum_atol_mV": 1e-9,
        },
        controller=controller, recorder=recorder,
    )
    recorder.zero_sum_error_mV[0] = 1e-4
    with pytest.raises(RuntimeError, match="zero-sum"):
        worker._augment_npz_arrays({}, state)


def test_manifest_is_rebuilt_from_hashed_inputs_and_compared_field_by_field(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    input_payloads = {
        "stage_ak_config": {"kind": "ak_config"},
        "stage_ak_manifest": {"kind": "ak_manifest"},
        "stage_al_manifest": {"kind": "al_manifest"},
    }
    inputs = {}
    for name, payload in input_payloads.items():
        path = artifact / f"{name}.json"
        path.write_text(json.dumps(payload))
        inputs[name] = {
            "path": path.name,
            "sha256": worker._sha256_file(path),
        }
    config = {
        "candidate_manifest": "manifest.json",
        "inputs": inputs,
        "event_unit": {"name": "event"},
        "source_topology": {"bin_mm": 1.0},
        "search": {"simulation": {"duration_ms": 20000.0}},
        "pathways": {"Z_M": "off"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    candidates = [{"candidate_id": "c0", "nested": {"gain": 0.2}}]
    substrate_audit = {"n_unique_substrates": 1}

    def fake_build(stage_ak, stage_al, stage_config, received_config):
        assert stage_ak == input_payloads["stage_ak_manifest"]
        assert stage_al == input_payloads["stage_al_manifest"]
        assert stage_config == input_payloads["stage_ak_config"]
        assert received_config == config
        return candidates, substrate_audit

    monkeypatch.setattr(worker.rev13_freezer, "build_candidates", fake_build)
    manifest = {
        "status": worker.MANIFEST_STATUS,
        "config_sha256": worker._sha256_file(config_path),
        "candidates": copy.deepcopy(candidates),
        "substrate_audit": substrate_audit,
        "event_unit": config["event_unit"],
        "source_topology": config["source_topology"],
        "search": config["search"],
        "pathways": config["pathways"],
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    _, rebuilt, audit = worker._rebuild_and_validate_manifest(
        config, config_path=config_path, artifact_root=artifact
    )
    assert rebuilt == candidates
    assert audit["candidate_exact_compare"] is True

    manifest["candidates"][0]["nested"]["gain"] = 0.2000000001
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=r"\$\.candidates\[0\]\.nested\.gain"):
        worker._rebuild_and_validate_manifest(
            config, config_path=config_path, artifact_root=artifact
        )


def test_explicit_runtime_provenance_hashes_each_required_path_and_fails_dirty(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    root.mkdir()
    for relative in worker.EXPLICIT_RUNTIME_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {relative}\n")
    config_path = root / "config/rev13.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=root, check=True)
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    monkeypatch.setattr(worker, "ROOT", root)
    provenance = worker._explicit_runtime_provenance(config_path, expected)
    assert provenance["all_explicit_paths_clean"] is True
    assert set(provenance["explicit_runtime_files"]) == {
        *worker.EXPLICIT_RUNTIME_PATHS,
        "config/rev13.json",
    }
    assert all(
        row["matches_expected_commit"] and not row["dirty"]
        for row in provenance["explicit_runtime_files"].values()
    )

    (root / "src/snn_engine/kick_probe.py").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="explicit runtime freeze failed"):
        worker._explicit_runtime_provenance(config_path, expected)


def test_duration_override_is_engineering_only_and_never_exceeds_frozen_run():
    assert worker._validate_duration_override(
        2000.0, "sentinel", frozen_duration_ms=20000.0
    ) == 2000.0
    assert worker._validate_duration_override(
        20000.0, "parity", frozen_duration_ms=20000.0
    ) == 20000.0
    assert worker._validate_duration_override(
        None, None, frozen_duration_ms=20000.0
    ) is None
    with pytest.raises(RuntimeError, match="restricted"):
        worker._validate_duration_override(
            2000.0, None, frozen_duration_ms=20000.0
        )
    with pytest.raises(RuntimeError, match=r"\(0, 20000\]"):
        worker._validate_duration_override(
            20000.1, "sentinel", frozen_duration_ms=20000.0
        )
    with pytest.raises(RuntimeError, match=r"\(0, 20000\]"):
        worker._preflight([
            "--config", str(worker.ROOT / "config/topic4_rev13_node_zero_sum_recovery.json"),
            "--candidate-id", "exact_off",
            "--seed", "2311",
            "--expected-commit", "HEAD",
            "--duration-ms", "20001",
            "--engineering-run-kind", "sentinel",
        ])


def test_preflight_applies_two_second_sentinel_only_to_compatibility_config(
    tmp_path, monkeypatch
):
    config = {
        "scientific_role": worker.SCIENTIFIC_ROLE,
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off",
            "Z_M": "off",
        },
        "inputs": {},
        "candidate_manifest": "unused.json",
        "search": {
            "simulation": {"duration_ms": 20000.0},
            "canary_network_seeds": [2311],
            "engineering_parity_network_seeds": [2291],
        },
    }
    config_path = tmp_path / "rev13.json"
    config_path.write_text(json.dumps(config))
    candidate = {
        "candidate_id": "exact_off",
        "node_accessibility": None,
        "pathways": config["pathways"],
    }
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worker.ROOT, text=True
    ).strip()
    monkeypatch.setattr(
        worker,
        "_rebuild_and_validate_manifest",
        lambda *args, **kwargs: (
            {"provenance": {"git_commit": expected_commit}},
            [candidate],
            {"candidate_exact_compare": True},
        ),
    )
    monkeypatch.setattr(
        worker,
        "_explicit_runtime_provenance",
        lambda *args, **kwargs: {"all_explicit_paths_clean": True},
    )
    monkeypatch.setattr(
        worker,
        "_compatibility_view",
        lambda *args, **kwargs: (copy.deepcopy(config), {}),
    )
    _, state = worker._preflight([
        "--config", str(config_path),
        "--candidate-id", "exact_off",
        "--seed", "2311",
        "--expected-commit", expected_commit,
        "--artifact-root", str(tmp_path),
        "--duration-ms", "2000",
        "--engineering-run-kind", "sentinel",
    ])
    assert config["search"]["simulation"]["duration_ms"] == 20000.0
    assert state.compatibility_config["search"]["simulation"]["duration_ms"] == 2000.0
    assert state.requested_duration_ms == 2000.0
    assert state.engineering_run_kind == "sentinel"
    assert state.rev13_provenance["manifest_rebuild"] == {
        "candidate_exact_compare": True,
        "manifest_git_commit": expected_commit,
    }


def test_duration_override_is_written_to_json_and_provenance():
    state = worker._RunState(
        candidate=_candidate(),
        controller_spec={"mode": "exact_off", "enabled": False},
        rev13_provenance={"all_explicit_paths_clean": True},
        engineering_run_kind="sentinel",
        requested_duration_ms=2000.0,
        frozen_duration_ms=20000.0,
    )
    payload = worker._augment_json_payload({
        "mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": True,
        },
        "provenance": {},
    }, state)
    assert payload["execution_duration"] == {
        "frozen_duration_ms": 20000.0,
        "requested_duration_ms": 2000.0,
        "engineering_run_kind": "sentinel",
        "duration_override_used": True,
    }
    assert payload["provenance"]["rev13_execution_duration"] == (
        payload["execution_duration"]
    )
    assert payload["provenance"]["rev13_explicit_runtime_freeze"] == {
        "all_explicit_paths_clean": True
    }


def test_base_worker_argv_strips_rev13_only_engineering_arguments():
    args = SimpleNamespace(
        config=Path("config/rev13.json"),
        candidate_id="exact_off",
        seed=2291,
        expected_commit="a" * 40,
        artifact_root=Path("/artifact"),
        out_json=Path("/output/worker.json"),
        out_npz=Path("/output/worker.npz"),
        duration_ms=20000.0,
        engineering_run_kind="parity",
    )
    argv = worker._base_worker_argv(args)
    assert argv[argv.index("--candidate-id") + 1] == "exact_off"
    assert argv[argv.index("--seed") + 1] == "2291"
    assert "--duration-ms" not in argv
    assert "--engineering-run-kind" not in argv
    assert argv[argv.index("--out-json") + 1] == "/output/worker.json"
    assert argv[argv.index("--out-npz") + 1] == "/output/worker.npz"

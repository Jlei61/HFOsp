import json
from pathlib import Path
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
    assert normalized["stratified_shuffle_c020"]["mode"] == "stratified_shuffle"
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
    diagnostics = recorder.diagnostics()
    assert diagnostics["threshold_min_mV_observed"] > 12.0
    assert diagnostics["maximum_zero_sum_error_mV"] < 1e-12


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
        "maximum_static_modulation_sign_flip_fraction": 0.0,
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
        static_rms_mV=0.2,
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

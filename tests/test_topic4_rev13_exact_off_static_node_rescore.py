import json
from pathlib import Path

import numpy as np
import pytest

from scripts import rescore_topic4_rev13_exact_off_static_node as rescore


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
CONFIG_PATH = ROOT / "config/topic4_rev14_static_node_causal_family_diagnostic.json"


def test_source_has_no_classifier_refit_or_cohort_target_dependency():
    source = (ROOT / "scripts/rescore_topic4_rev13_exact_off_static_node.py").read_text()
    for forbidden in (
        "LogisticRegression", "LedoitWolf", "_fit_training_classifier",
        "patient_training_calibration", "_patient_data",
    ):
        assert forbidden not in source
    config = json.loads(CONFIG_PATH.read_text())
    target = config["inputs"]["patient_training_target"]
    assert target["path"].endswith("shaft_aware_patient_training_target.npz")
    assert target["sha256"] == (
        "7d11100c0f431479f8decb17b6de97575a91dcb889887ce9e2cedbbdcd54140b"
    )
    assert "cohort" not in target["path"].lower()
    for record_name in (
        "patient_training_target", "frozen_direction_classifier_manifest",
        "contact_contract",
    ):
        record = config["inputs"][record_name]
        assert len(record["sha256"]) == 64
        path = ARTIFACT_ROOT / record["path"]
        assert rescore._sha256(path) == record["sha256"]


def test_training_loader_accesses_only_six_training_arrays(tmp_path):
    path = tmp_path / "training_only.npz"
    ranks = np.asarray([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 2.0, 1.0],
        [1.0, 2.0, 0.0],
    ])
    np.savez(
        path,
        contact_names=np.asarray(["ICL1", "ICL2", "SCL1"]),
        patient_train_block_ids=np.asarray([0, 0, 1, 1]),
        patient_train_old_labels=np.asarray([1, 1, 0, 0]),
        patient_train_ranks=ranks,
        patient_train_shaft_aware_k2_labels=np.asarray([0, 0, 1, 1]),
        feature_center=np.zeros(11),
        feature_scale=np.ones(11),
        pca_components=np.zeros((2, 11)),
        patient_heldout_poison=np.asarray([999]),
    )
    loaded = rescore.load_patient_training_target(
        path, events_per_mode=2, seed=7,
    )
    assert loaded["loaded_training_data_keys"] == list(rescore.TRAINING_TARGET_KEYS)
    assert loaded["loaded_frozen_embedding_keys"] == list(
        rescore.FROZEN_EMBEDDING_KEYS
    )
    accessed = (
        loaded["loaded_training_data_keys"]
        + loaded["loaded_frozen_embedding_keys"]
    )
    assert all("heldout" not in key.lower() for key in accessed)
    assert loaded["patient_heldout_loaded"] is False
    np.testing.assert_array_equal(
        loaded["diagnostic_old_to_shaft_extent_mapping"], [1, 0],
    )
    np.testing.assert_array_equal(loaded["all_labels"], [1, 1, 0, 0])
    assert loaded["primary_mode_definition"] == "frozen_old_A_B_direction_labels"
    np.testing.assert_array_equal(
        loaded["old_by_shaft_aware_k2_contingency"], [[0, 2], [2, 0]],
    )


def test_frozen_manifest_decodes_only_inference_allowlist(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "FROZEN",
        "direction_classifier": {
            "coef": [1.0, 2.0],
            "intercept": 0.5,
            "heldout_poison": "DO_NOT_DECODE",
            "training_roc_auc": 0.99,
        },
        "patient_heldout_path": "DO_NOT_DECODE_PATH",
    }))
    decoded_values = []
    original_loads = json.loads

    def guarded_loads(value, *args, **kwargs):
        decoded_values.append(value)
        assert "DO_NOT_DECODE" not in value
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(rescore.json, "loads", guarded_loads)
    selected = rescore._load_frozen_classifier_keys(
        manifest, ("coef", "intercept"),
    )
    assert selected == {"coef": [1.0, 2.0], "intercept": 0.5}
    assert len(decoded_values) == 2


def test_old_classifier_labels_remain_primary_without_refit(monkeypatch):
    monkeypatch.setattr(rescore, "assign_direction_modes", lambda *args, **kwargs: {
        "labels": np.asarray([0, 1], dtype=int),
        "probability_B": np.asarray([0.2, 0.8]),
        "ood_distance": np.asarray([1.0, 2.0]),
        "ood": np.asarray([False, True]),
    })
    assigned = rescore._assign_training_modes(
        np.zeros((2, 3)), {"embedding": {}, "classifier": {}}, {},
    )
    np.testing.assert_array_equal(assigned["raw_old_labels"], [0, 1])
    np.testing.assert_array_equal(assigned["labels"], [0, 1])
    np.testing.assert_allclose(assigned["probability_B"], [0.2, 0.8])
    np.testing.assert_array_equal(assigned["ood"], [False, True])


def test_primary_selection_excludes_entire_overlap_component_and_keeps_single_shaft(
        monkeypatch):
    n_events = 7
    ranks = np.full((n_events, 5), np.nan)
    ranks[0, :3] = [0, 1, 2]
    ranks[1, :3] = [1, 2, 0]
    ranks[2, :3] = [2, 0, 1]
    ranks[3, :3] = [0, 2, 1]  # ICL-only/missing SCL remains readable.
    ranks[4, :3] = [0, 1, 2]
    ranks[5, :3] = [0, 1, 2]
    ranks[6, :3] = [0, 1, 2]
    arrays = {
        "event_returned": np.asarray([1, 1, 1, 1, 0, 1, 1], bool),
        "source_onset_evaluable": np.asarray([1, 1, 1, 1, 1, 0, 1], bool),
        "event_t_on_ms": np.asarray([0, 8, 19, 40, 50, 60, 70], float),
        "event_trigger_t_on_ms": np.asarray([0, 8, 19, 40, 50, 60, 70], float),
        "event_t_off_ms": np.asarray([10, 20, 30, 45, 55, 65, np.nan]),
        "event_fragment_count": np.ones(n_events, int),
        "event_directed_root_id": np.arange(n_events),
        "event_root_count": np.ones(n_events, int),
        "onsets": ranks.copy(),
        "ranks": ranks,
        "source_onset_maps_ms": np.zeros((n_events, 2, 2)),
        "source_bin_mm": np.asarray(1.0),
        "positions_E": np.zeros((4, 2)),
        "delta_vtheta": np.ones(4),
    }
    monkeypatch.setattr(rescore, "substrate_pca_axis", lambda *args: np.asarray([1., 0.]))
    monkeypatch.setattr(
        rescore, "event_axis_displacements",
        lambda maps, **kwargs: np.arange(1, len(maps) + 1, dtype=float),
    )
    selected = rescore.primary_family_selection(
        arrays, minimum_readable_contacts=3,
    )
    np.testing.assert_array_equal(selected["contact_primary_indices"], [3, 5])
    np.testing.assert_array_equal(selected["topology_primary_indices"], [3])
    np.testing.assert_array_equal(selected["fig4_readable_indices"], [3, 5])
    assert selected["overlap_audit"]["formal_action"] == (
        "EXCLUDE_ALL_MEMBERS_OF_OVERLAP_CONNECTED_EPISODES"
    )
    assert selected["n_primary_isolated"] == 2
    assert selected["n_topology_primary"] == 1
    assert selected["n_fig4_readable"] == 2


def test_equal_network_mean_does_not_event_weight():
    result = rescore.equal_network_mean([{"objective": 1.0}, {"objective": 9.0}])
    assert result == {"objective": 5.0}


def test_runtime_provenance_tracks_config_producer_and_dependencies():
    paths = {
        str(path.relative_to(ROOT))
        for path in rescore._runtime_paths(CONFIG_PATH)
    }
    assert "config/topic4_rev14_static_node_causal_family_diagnostic.json" in paths
    assert "scripts/rescore_topic4_rev13_exact_off_static_node.py" in paths
    assert "src/topic4_node_dualmode.py" in paths
    provenance = rescore.runtime_provenance(CONFIG_PATH)
    assert len(provenance["git_commit_at_analysis"]) == 40
    assert provenance["runtime_paths_dirty"] == bool(
        provenance["runtime_dirty_porcelain"]
    )
    assert set(provenance["runtime_path_sha256"]) == {
        str(path) for path in rescore._runtime_paths(CONFIG_PATH)
    }


@pytest.mark.parametrize(
    "seed,expected_primary,expected_readable",
    [(2311, 26, 24), (2312, 31, 26), (2313, 29, 22)],
)
def test_real_exact_off_primary_counts(seed, expected_primary, expected_readable):
    config = json.loads(CONFIG_PATH.read_text())
    worker = next(row for row in config["inputs"]["workers"] if row["seed"] == seed)
    path = ARTIFACT_ROOT / worker["npz"]["path"]
    if not path.exists():
        pytest.skip("canonical rev13 worker artifact is not mounted")
    keys = (
        "event_returned", "source_onset_evaluable", "event_t_on_ms",
        "event_trigger_t_on_ms", "event_t_off_ms", "event_fragment_count",
        "event_directed_root_id", "event_root_count", "onsets", "ranks",
        "source_onset_maps_ms", "source_bin_mm", "positions_E", "delta_vtheta",
    )
    arrays = rescore._load_npz_keys(path, keys)
    selected = rescore.primary_family_selection(
        arrays, minimum_readable_contacts=3,
    )
    assert selected["n_primary_isolated"] == expected_primary
    assert selected["n_fig4_readable"] == expected_readable

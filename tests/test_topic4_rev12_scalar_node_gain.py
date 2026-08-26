import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_rev12_scalar_node_gain_canary import build_candidates
from scripts.analyze_topic4_rev12_scalar_node_gain_canary import gain_audit
from src.topic4_core_field_rev9 import reconstruct_node_from_h


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _load(relative):
    local = ROOT / relative
    return json.loads((local if local.exists() else ARTIFACT_ROOT / relative).read_text())


def test_gain_one_preserves_historical_node_arrays_bitwise():
    h = np.linspace(0.01, 0.99, 100)
    kwargs = dict(
        n_total=125, quantile_seed=7, core_mean=17.5,
        core_std=1.0, v_base=18.0,
    )
    historical = reconstruct_node_from_h(h, **kwargs)
    explicit = reconstruct_node_from_h(h, node_gain=1.0, **kwargs)
    for key in ("h", "d", "d_shrunk", "d_effective", "vtheta", "delta_vtheta"):
        assert np.array_equal(historical[key], explicit[key])
    assert historical["hashes"] == explicit["hashes"]


@pytest.mark.parametrize("gain", [0.0, 0.5, 1.5])
def test_gain_scales_only_frozen_node_modulation(gain):
    h = np.linspace(0.01, 0.99, 100)
    reference = reconstruct_node_from_h(
        h, n_total=125, quantile_seed=7, core_mean=17.5,
        core_std=1.0, v_base=18.0,
    )
    changed = reconstruct_node_from_h(
        h, n_total=125, quantile_seed=7, core_mean=17.5,
        core_std=1.0, v_base=18.0, node_gain=gain,
    )
    assert np.array_equal(changed["h"], reference["h"])
    assert np.array_equal(changed["d"], reference["d"])
    assert np.allclose(changed["delta_vtheta"], gain * reference["delta_vtheta"])
    assert changed["mapping_audit"]["gain_application_error"] == pytest.approx(0.0, abs=1e-12)
    assert np.all(changed["vtheta"][len(h):] == 18.0)


def test_gain_manifest_is_bounded_and_does_not_duplicate_gain_one():
    config = _load("config/topic4_rev12_nd_scalar_node_gain_canary.json")
    manifest = _load(config["inputs"]["stage_ag_manifest"]["path"])
    audit = _load(config["inputs"]["stage_ag_paired_audit"]["path"])
    inventory = _load(config["inputs"]["stage_ai_inventory"]["path"])
    candidates, mapping = build_candidates(
        manifest, audit, inventory, config["scalar_node_gain"],
    )
    assert len(candidates) == 9
    assert all(row["selection_eligible"] is False for row in candidates)
    gains = [row["node_mapping"]["node_gain"] for row in candidates]
    assert gains.count(0.0) == 1
    assert 1.0 not in gains
    assert set(gains) == {0.0, 0.5, 0.75, 1.25, 1.5}
    assert mapping["patient_heldout_used"] is False


def test_gain_builder_requires_inventory_gap():
    config = _load("config/topic4_rev12_nd_scalar_node_gain_canary.json")
    manifest = _load(config["inputs"]["stage_ag_manifest"]["path"])
    audit = _load(config["inputs"]["stage_ag_paired_audit"]["path"])
    inventory = copy.deepcopy(_load(config["inputs"]["stage_ai_inventory"]["path"]))
    inventory["status"] = "SCALAR_NODE_GAIN_ALREADY_TESTED"
    with pytest.raises(RuntimeError, match="does not justify"):
        build_candidates(manifest, audit, inventory, config["scalar_node_gain"])


def _record(seed, objective, mode_0, mode_1, direction_0, direction_1):
    return {
        "seed": seed,
        "soft_objective": {
            "objective": objective,
            "modes": {"0": {"mean": mode_0}, "1": {"mean": mode_1}},
        },
        "soft_causal_direction": {"modes": {
            "0": {"alignment_score": direction_0},
            "1": {"alignment_score": direction_1},
        }},
        "soft_causal_monotonicity": {"modes": {
            "0": {"alignment_score": 0.2},
            "1": {"alignment_score": 0.2},
        }},
    }


def test_gain_audit_requires_balanced_absolute_improvement():
    anchor = [_record(s, 2, 2, 2, 0.2, 0.2) for s in range(6)]
    source = [_record(s, 1.8, 1.8, 1.8, 0.3, 0.1) for s in range(6)]
    resolved = [_record(s, 1.7, 1.7, 1.7, 0.4, 0.4) for s in range(6)]
    unresolved = [_record(s, 1.6, 1.6, 1.6, 0.5, 0.1) for s in range(6)]
    stage_ag = {"rows": [
        {"candidate_id": "stage_ag_anchor", "per_network": anchor},
        {"candidate_id": "stage_ag_g10_p", "per_network": source},
    ]}
    stage_aj = {"rows": [
        {"candidate_id": "resolved", "per_network": resolved,
         "mean_events": 40, "fit_valid": True, "invalid_reasons": []},
        {"candidate_id": "unresolved", "per_network": unresolved,
         "mean_events": 40, "fit_valid": True, "invalid_reasons": []},
    ]}
    manifest = {"candidates": [
        {"candidate_id": "resolved", "source_candidate_ids": ["stage_ag_g10_p"],
         "node_field": {"field_sha256": "f"},
         "node_mapping": {"node_gain": 1.25, "mapping_sha256": "a"}},
        {"candidate_id": "unresolved", "source_candidate_ids": ["stage_ag_g10_p"],
         "node_field": {"field_sha256": "f"},
         "node_mapping": {"node_gain": 1.5, "mapping_sha256": "b"}},
    ]}
    result = gain_audit(
        stage_aj, stage_ag, manifest,
        primary_endpoints=[
            "soft_objective", "mode_0", "mode_1",
            "mode_0_direction", "mode_1_direction",
        ],
        diagnostic_endpoints=["mode_0_monotonicity", "mode_1_monotonicity"],
        minimum_positive_networks=4,
        bootstrap_draws=1000, bootstrap_confidence=0.90, bootstrap_seed=3,
    )
    assert result["status"] == "SCALAR_NODE_GAIN_BALANCED_CANDIDATE_FOUND"
    assert result["balanced_candidate_ids"] == ["resolved"]

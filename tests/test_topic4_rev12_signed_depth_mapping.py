import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_rev12_signed_depth_mapping_audit import build_candidates
from src.topic4_core_field_rev9 import reconstruct_node_from_h


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _load(relative):
    local = ROOT / relative
    return json.loads((local if local.exists() else ARTIFACT_ROOT / relative).read_text())


def test_rho_one_is_bitwise_identical_to_default_mapping():
    h = np.linspace(0.01, 0.99, 100)
    kwargs = dict(
        n_total=125, quantile_seed=7, core_mean=17.5,
        core_std=1.0, v_base=18.0,
    )
    default = reconstruct_node_from_h(h, **kwargs)
    explicit = reconstruct_node_from_h(h, depth_shrinkage=1.0, **kwargs)
    for key in ("h", "d", "d_effective", "vtheta", "delta_vtheta"):
        assert np.array_equal(default[key], explicit[key])
    assert default["hashes"] == explicit["hashes"]
    assert np.array_equal(default["d"], default["d_effective"])


@pytest.mark.parametrize("rho", [1.0, 0.5, 0.0])
def test_depth_shrinkage_preserves_h_weighted_modulation(rho):
    h = np.linspace(0.01, 0.99, 100)
    node = reconstruct_node_from_h(
        h, n_total=125, quantile_seed=7, core_mean=17.5,
        core_std=1.0, v_base=18.0, depth_shrinkage=rho,
    )
    audit = node["mapping_audit"]
    assert audit["budget_error"] == pytest.approx(0.0, abs=1e-12)
    assert np.sum(h * node["d_effective"]) == pytest.approx(
        np.sum(h * node["d"]), abs=1e-12,
    )
    assert audit["h_weighted_depth_sd_effective"] == pytest.approx(
        rho * audit["h_weighted_depth_sd_original"], abs=1e-12,
    )
    assert np.all(node["vtheta"][len(h):] == 18.0)


def test_mapping_manifest_keeps_field_identity_separate_from_rho():
    config = _load("config/topic4_rev12_nd_signed_depth_mapping_audit.json")
    manifest = _load(config["inputs"]["stage_ag_manifest"]["path"])
    audit = _load(config["inputs"]["stage_ag_paired_audit"]["path"])
    candidates, mapping = build_candidates(
        manifest, audit, config["signed_depth_mapping"],
    )
    assert len(candidates) == 4
    assert all(row["selection_eligible"] is False for row in candidates)
    by_source = {}
    for row in candidates:
        by_source.setdefault(row["source_candidate_ids"][0], []).append(row)
    assert set(by_source) == {"stage_ag_anchor", "stage_ag_g10_p"}
    for rows in by_source.values():
        assert len({row["node_field"]["field_sha256"] for row in rows}) == 1
        assert {row["node_mapping"]["signed_depth_shrinkage"] for row in rows} == {0.0, 0.5}
        assert len({row["node_mapping"]["mapping_sha256"] for row in rows}) == 2
    assert mapping["patient_heldout_used"] is False
    assert mapping["manual_field_used"] is False


def test_mapping_builder_rejects_balanced_stage_ag_status():
    config = _load("config/topic4_rev12_nd_signed_depth_mapping_audit.json")
    manifest = _load(config["inputs"]["stage_ag_manifest"]["path"])
    audit = copy.deepcopy(_load(config["inputs"]["stage_ag_paired_audit"]["path"]))
    audit["status"] = "BROAD_FIELD_BALANCED_FIT_DIRECTION_CANDIDATE_FOUND"
    with pytest.raises(RuntimeError, match="does not justify"):
        build_candidates(manifest, audit, config["signed_depth_mapping"])

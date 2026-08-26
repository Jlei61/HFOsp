import json
from pathlib import Path

from scripts.analyze_topic4_rev12_dual_node_channel_canary import dual_channel_audit
from scripts.freeze_topic4_rev12_dual_node_channel_canary import build_candidates


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _load(relative):
    local = ROOT / relative
    return json.loads((local if local.exists() else ARTIFACT_ROOT / relative).read_text())


def test_dual_channel_manifest_has_coupled_controls_and_cross_pairs():
    config = _load("config/topic4_rev12_nd_dual_node_channel_canary.json")
    inputs = {name: _load(record["path"]) for name, record in config["inputs"].items()}
    candidates, audit = build_candidates(
        inputs["stage_ag_manifest"], inputs["stage_ag_paired_audit"],
        inputs["stage_ah_result"], inputs["stage_aj_result"],
        inputs["stage_aj_crossing_audit"], config["dual_node_channel"],
    )
    assert len(candidates) == 7
    assert sum(row["coupled_control"] for row in audit["combinations"]) == 3
    assert all(row["selection_eligible"] is False for row in candidates)
    assert all("node_dispersion_field" in row for row in candidates)


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


def _summary_row(candidate_id, values):
    return {
        "candidate_id": candidate_id, "per_network": values,
        "mean_events": 30, "fit_valid": True, "invalid_reasons": [],
    }


def test_dual_channel_audit_only_nominates_balanced_cross_mapping():
    base = [_record(s, 2, 2, 2, 0.2, 0.2) for s in range(6)]
    other = [_record(s, 1.9, 1.9, 1.9, 0.25, 0.25) for s in range(6)]
    cross = [_record(s, 1.7, 1.7, 1.7, 0.4, 0.4) for s in range(6)]
    summary = {"rows": [
        _summary_row("anchor", base),
        _summary_row("other", other),
        _summary_row("cross", cross),
    ]}
    manifest = {"candidates": [
        {"candidate_id": "anchor", "source_candidate_ids": {"mean": "a", "dispersion": "a"}},
        {"candidate_id": "other", "source_candidate_ids": {"mean": "b", "dispersion": "b"}},
        {"candidate_id": "cross", "source_candidate_ids": {"mean": "a", "dispersion": "b"}},
    ]}
    result = dual_channel_audit(
        summary, manifest, anchor_id="anchor",
        primary_endpoints=[
            "soft_objective", "mode_0", "mode_1",
            "mode_0_direction", "mode_1_direction",
        ],
        diagnostic_endpoints=["mode_0_monotonicity", "mode_1_monotonicity"],
        minimum_positive_networks=4, bootstrap_draws=1000,
        bootstrap_confidence=0.9, bootstrap_seed=4,
    )
    assert result["status"] == "DUAL_CONTINUOUS_NODE_CHANNEL_BALANCED_CANDIDATE_FOUND"
    assert result["balanced_candidate_ids"] == ["cross"]

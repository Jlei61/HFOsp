import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_fcxr_lc6a_confirmation.py"
SPEC = importlib.util.spec_from_file_location("lc6a_confirmation", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _phenotype(condition, margin, bounded=True):
    return {
        "condition": condition,
        "boundedness": {"bounded_candidate": bounded, "boundedness_margin": margin},
    }


def _gain(condition, *, responsive=True, global_sat=False, local_sat=.01):
    return {
        "condition": condition,
        "checkpoints": [{
            "checkpoint": "onset_plus_2s",
            "response_detected_nonzero": responsive,
            "probe": {
                "registered_global_saturation": global_sat,
                "local_saturation": {"max_near_refractory_fraction": local_sat},
                "diverged": False,
            },
        }],
    }


def test_confirmation_selects_largest_margin_bounded_nonsaturating_response():
    phenotype = {"rows": [_phenotype("Q1", .1), _phenotype("Q2", .4), _phenotype("Q3", .8)]}
    gains = {"rows": [
        _gain("Q1"), _gain("Q2"), _gain("Q3", global_sat=True),
    ]}
    selected = RUNNER.select_candidate(phenotype, gains)
    assert selected["condition"] == "Q2"
    assert selected["responsive_checkpoints"] == ["onset_plus_2s"]


def test_confirmation_is_not_triggered_by_unbounded_or_inert_candidate():
    phenotype = {"rows": [_phenotype("Q1", .1, bounded=False), _phenotype("Q2", .2)]}
    gains = {"rows": [_gain("Q1"), _gain("Q2", responsive=False)]}
    assert RUNNER.select_candidate(phenotype, gains) is None


def test_confirmation_prelock_freezes_graph_seed_and_forbids_outcome_q_tuning():
    lock = RUNNER._prelock()
    assert lock["graph"]["same_q_target"] is True
    assert lock["graph"]["trajectory_outcome_may_adjust_q_or_width"] is False
    assert lock["runtime"]["noise_seed"] == 401


def test_natural_confirmation_lock_checks_parent_output_and_graph_hash(tmp_path):
    graph = tmp_path / "graph.npz"
    graph.write_bytes(b"graph")
    payload = {
        "status": "LOCKED", "authorized": True,
        "parent_condition": "Q2", "output_condition": "CONF_Q2_B",
        "graph_artifact": str(graph), "graph_artifact_sha256": RUNNER._sha(graph),
    }
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(payload))
    path, got = RUNNER.NAT._validate_confirmation(
        lock, parent_condition="Q2", output_condition="CONF_Q2_B", graph_path=graph,
    )
    assert path == lock.resolve() and got["authorized"] is True


def test_graph_builder_exposes_separate_seed_and_output_without_changing_primary_defaults():
    source = (ROOT / "scripts/build_topic4_fcxr_lc6a_graph_condition.py").read_text()
    assert "graph_seed_override=None" in source
    assert "output_condition=None" in source
    assert 'output_id = condition if output_condition is None' in source

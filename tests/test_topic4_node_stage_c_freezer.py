import json
from pathlib import Path

from scripts.freeze_topic4_rev12_node_stage_c import build_stage_c_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_stage_c_library_is_continuous_observation_invariant_and_unique():
    source = json.loads((
        Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/")
        / "data_driven_node_dualmode_rev12/node_stage_b_selection/candidate_manifest.json"
    ).read_text())
    config = json.loads((ROOT / "config/topic4_rev12_nd_node_stage_c_fit.json").read_text())
    rows = build_stage_c_candidates(source, config["field_search"])
    assert len(rows) == 18
    assert len({row["node_field"]["field_sha256"] for row in rows}) == 18
    assert all(row["node_field"]["field_type"] == "spline_continuous" for row in rows)
    assert all(row["node_field"]["component_count"] is None for row in rows)
    assert all(row["node_field"]["peak_count_constraint"] is None for row in rows)

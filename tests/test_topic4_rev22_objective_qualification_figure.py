import json
from pathlib import Path

from scripts.plot_topic4_rev22_objective_qualification import build_cells, render


def _payload():
    return {
        "schema_id": "topic4_rev22_dci_objective_qualification_v2",
        "status": "OBJECTIVE_QUALIFIED",
        "controls": {
            "1_minority_removed": {
                "pass": True,
                "target": {"D_order": {"fraction_worse": 1.0}},
                "reported": {"D_cover": {"fraction_worse": 0.75}},
            },
            "2_scl_censored": {
                "pass": True,
                "target": {"D_support": {"fraction_worse": 1.0}},
                "invariance": {"D_order.ICL-ICL": {"max_abs_difference": 0.0}},
            },
            "3_stretched": {
                "pass": True,
                "target": {"D_lag": {"fraction_worse": 1.0}},
                "invariance": {
                    "D_support": {"max_abs_difference": 0.0},
                    "D_order": {"max_abs_difference": 0.0},
                },
            },
            "4_permuted_within_shaft": {
                "pass": True,
                "target": {"D_order": {"fraction_worse": 1.0}},
                "invariance": {"D_support": {"max_abs_difference": 0.0}},
            },
            "all_pass": True,
        },
    }


def test_cells_preserve_target_invariant_and_descriptive_semantics():
    cells = build_cells(_payload())
    assert cells[0][1] == {"kind": "target", "value": 1.0, "label": "100%"}
    assert cells[0][3] == {"kind": "descriptive", "value": 0.75, "label": "75%"}
    assert cells[1][1]["kind"] == "invariant"
    assert "ICL only" in cells[1][1]["label"]
    assert cells[2][0]["kind"] == "invariant"
    assert cells[2][2]["kind"] == "target"


def test_render_writes_hashed_outputs_and_readme(tmp_path: Path):
    metadata = render(_payload(), tmp_path)
    assert set(metadata["output_sha256"]) == {
        "rev22_dci_objective_qualification_controls.png",
        "rev22_dci_objective_qualification_controls.pdf",
        "rev22_dci_objective_qualification_controls.svg",
    }
    for name in metadata["output_sha256"]:
        assert (tmp_path / name).is_file()
    saved = json.loads(
        (tmp_path / "rev22_dci_objective_qualification_controls_metadata.json").read_text()
    )
    assert saved["claim_boundary"] == metadata["claim_boundary"]
    assert (tmp_path / "README.md").is_file()

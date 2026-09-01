import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((
        ROOT / "config/topic4_rev10_d4_1_packet_dose_confirmation.json"
    ).read_text())


def test_d4_1_uses_fresh_networks_and_frozen_mirror_sources():
    config = _config()
    assert config["network_seeds"] == [1231, 1232, 1233, 1234, 1235, 1236]
    assert config["sources"] == [
        {
            "source_id": "route_A_x18_y06", "xy_mm": [18.0, 6.0],
            "expected_mode": "A",
            "selection_role": "D4 outcome-selected then frozen before fresh networks",
        },
        {
            "source_id": "route_B_x02_y14", "xy_mm": [2.0, 14.0],
            "expected_mode": "B",
            "selection_role": "geometric mirror of A around sheet center and D4 positive control",
        },
    ]
    assert config["claim_boundary"]["source_A_is_patient_outcome_selected"]


def test_d4_1_scans_nested_doses_and_requires_five_of_six_for_both_modes():
    config = _config()
    assert config["packet_fractions_of_E"] == [
        0.000625, 0.00125, 0.0025, 0.005,
    ]
    assert config["decision"]["minimum_networks_per_source_at_same_dose"] == 5
    assert config["execution"]["wait_seconds"] >= 180
    assert config["claim_boundary"]["Fig4_replacement_forbidden"]

from __future__ import annotations

import json

from scripts import plot_sef_itp_phase1_cohort
from scripts.summarize_sef_itp_phase1 import aggregate


def _subject_json(subject_id: str, h2: dict | None) -> dict:
    return {
        "subject_id": subject_id,
        "dataset": "epilepsiae",
        "n_channels": 6,
        "n_coord_mapped": 6,
        "coord_space": "mni152_1mm",
        "coord_provenance": {"normalization_certainty": "test"},
        "n_dropped_endpoints_no_coords_per_cluster": {},
        "h6": {"verdict": "NULL"},
        "h1": {"per_cluster": {}},
        "h2": h2 or {"available": False, "reason": "rank-displacement missing"},
    }


def test_aggregate_h2_rank_displacement_swap_class_summary(tmp_path):
    per_subject = tmp_path / "per_subject"
    per_subject.mkdir()
    strict_h2 = {
        "available": True,
        "source_contract": "rank_displacement_swap_sweep_v1",
        "swap_class": "strict",
        "decision_k": 2,
        "swap_endpoint_channels": ["A1-A2", "A2-A3", "B2-B3", "B3-B4"],
        "swap_score": 1.0,
        "p_fw": 0.001,
        "null_p": 0.001,
        "n_valid": 6,
        "spatial_compactness": {
            "available": True,
            "source_side": {"verdict": "PASS"},
            "sink_side": {"verdict": "NULL"},
            "combined_endpoint": {"verdict": "INSUFFICIENT_NULL"},
        },
    }
    candidate_h2 = {
        "available": True,
        "source_contract": "rank_displacement_swap_sweep_v1",
        "swap_class": "candidate",
        "decision_k": 2,
        "swap_endpoint_channels": ["A1-A2", "A2-A3", "B2-B3", "B3-B4"],
        "swap_score": 0.75,
        "p_fw": 0.08,
        "null_p": 0.08,
        "n_valid": 6,
        "spatial_compactness": {
            "available": True,
            "source_side": {"verdict": "NULL"},
            "sink_side": {"verdict": "PASS"},
            "combined_endpoint": {"verdict": "NULL"},
        },
    }
    rows = [
        _subject_json("1073", strict_h2),
        _subject_json("139", candidate_h2),
        _subject_json("253", None),
    ]
    for row in rows:
        (per_subject / f"epilepsiae_{row['subject_id']}.json").write_text(
            json.dumps(row)
        )

    summary, csv_rows = aggregate(per_subject)

    assert summary["h2"]["source_contract"] == "rank_displacement_swap_sweep"
    assert summary["h2"]["n_testable_total"] == 2
    assert summary["h2"]["n_not_testable"] == 1
    assert summary["h2"]["swap_class_distribution"] == {
        "strict": 1,
        "candidate": 1,
        "none": 0,
        "unknown": 0,
    }
    assert summary["h2"]["spatial_compactness_verdict_distribution"]["source_side"] == {
        "PASS": 1,
        "NULL": 1,
    }
    assert summary["h2"]["spatial_compactness_verdict_distribution"]["sink_side"] == {
        "NULL": 1,
        "PASS": 1,
    }
    assert summary["subjects"][0]["h2_swap_check"]["swap_endpoint_channels"] == [
        "A1-A2", "A2-A3", "B2-B3", "B3-B4"
    ]
    assert csv_rows[0]["h2_swap_class"] == "strict"
    assert csv_rows[0]["h2_decision_k"] == 2
    assert csv_rows[0]["h2_source_spatial_verdict"] == "PASS"
    assert csv_rows[0]["h2_sink_spatial_verdict"] == "NULL"


def test_plot_cohort_accepts_rank_displacement_h2_schema(tmp_path):
    summary = {
        "n_subjects": 3,
        "h1": {
            "verdict_distribution_per_cluster": {"PASS": 1, "NULL": 1},
            "n_clusters_total": 2,
        },
        "h2": {
            "n_testable_total": 3,
            "n_not_testable": 0,
            "swap_class_distribution": {
                "strict": 1,
                "candidate": 1,
                "none": 0,
                "unknown": 1,
            },
            "n_strict_or_candidate": 2,
            "swap_score_median": 0.875,
            "p_fw_median": 0.0405,
        },
        "h6": {
            "verdict_distribution": {"NULL": 2},
            "n_subjects": 2,
        },
        "coord_coverage": {
            "coord_space_distribution": {"mni152_1mm": 2},
            "normalization_certainty_distribution": {"test": 2},
        },
        "subjects": [
            {
                "dataset": "epilepsiae",
                "subject_id": "1073",
                "h2_swap_check": {
                    "swap_score": 1.0,
                    "p_fw": 0.001,
                    "swap_class": "strict",
                    "decision_k": 2,
                },
            },
            {
                "dataset": "epilepsiae",
                "subject_id": "139",
                "h2_swap_check": {
                    "swap_score": 0.75,
                    "p_fw": 0.08,
                    "swap_class": "candidate",
                    "decision_k": 2,
                },
            },
            {
                "dataset": "epilepsiae",
                "subject_id": "253",
                "h2_swap_check": {
                    "swap_score": 0.50,
                    "p_fw": None,
                    "swap_class": "unknown",
                    "decision_k": None,
                },
            },
        ],
    }
    summary_path = tmp_path / "cohort_summary.json"
    out_dir = tmp_path / "figures"
    summary_path.write_text(json.dumps(summary))

    plot_sef_itp_phase1_cohort.main([
        "--summary", str(summary_path),
        "--output-dir", str(out_dir),
    ])

    assert (out_dir / "cohort_h2_rank_displacement_swap_class.png").exists()

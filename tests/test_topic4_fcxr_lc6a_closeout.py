import pytest

import scripts.finalize_topic4_fcxr_lc6a as closeout


def _phenotype_row(condition, *, q, onset_ms, lead, area, width=None):
    n = len(lead)
    return {
        "condition": condition,
        "construction_q": q,
        "effective_onset_ms": onset_ms,
        "baseline_metrics": {"n_events": 30},
        "global_rate_100ms_peak_hz": 400.0,
        "local_rate_q99_peak_hz": 430.0,
        "spatial_slow_flow": {
            "max_active_area_mm2": 400.0,
            "active_area_mm2": area,
            "D_halo_lead_mm": lead,
            "D_halo_width_q05_q95_mm": [18.1] * n if width is None else width,
        },
        "boundedness": {"boundedness_margin": -1.5},
        "baseline_tradeoff": {"tradeoff": False},
        "headline": "SATURATED_HIGH_STATE",
    }


def test_graph_rows_uses_weighted_two_hop_readout_not_construction_q():
    graph = {
        "audits": {
            "C0": {"construction_q": 0.7, "graph_sha256": "c0"},
            "C1": {"construction_q": 0.8, "graph_sha256": "c1"},
            "Q1": {"construction_q": 1.0, "graph_sha256": "q1"},
            "Q2": {"construction_q": 1.25, "graph_sha256": "q2"},
            "Q3": {"construction_q": 1.5, "graph_sha256": "q3"},
        }
    }
    two_hop = {
        "audits": {
            condition: {
                "operator": {
                    "q_parallel_two_hop": index + 0.01,
                    "surround_center_ratio": index + 0.02,
                },
                "latency": {"q95_ms": index + 10.0},
            }
            for index, condition in enumerate(("C0", "C1", "Q1", "Q2", "Q3"))
        }
    }

    rows = closeout._graph_rows(graph, two_hop)

    assert rows[0]["construction_q"] == 0.7
    assert rows[0]["two_hop_q"] == 0.01
    assert rows[-1]["two_hop_q"] == 4.01
    assert rows[-1]["graph_sha256"] == "q3"


def test_trajectory_rows_keeps_baseline_tradeoff_separate_from_headline():
    anchor = _phenotype_row("C0", q=0.93, onset_ms=11000.0, lead=[None, 0.01], area=[0.0, 156.0])
    row = _phenotype_row("Q2", q=1.29, onset_ms=12000.0, lead=[None, 1.2], area=[0.0, 30.0])
    row["baseline_tradeoff"] = {"tradeoff": True}
    phenotype = {"rows": [anchor, row]}

    result = closeout._trajectory_rows(phenotype, natural_summaries={
        "C0": {"n_returning_pre_onset": 29}, "Q2": {"n_returning_pre_onset": 37},
    })[1]

    assert result["onset_s"] == 12.0
    assert result["n_returning_pre_onset"] == 37
    assert result["baseline_tradeoff"] is True
    assert result["headline"] == "SATURATED_HIGH_STATE"


def test_trajectory_rows_flags_the_q_matched_microstate_band():
    """C0/C1/Q1 realized inside one same-q tolerance must not read as three ladder rungs."""

    phenotype = {"rows": [
        _phenotype_row("C0", q=0.9335, onset_ms=11000.0, lead=[None, 0.01], area=[0.0, 156.0]),
        _phenotype_row("C1", q=0.9583, onset_ms=10000.0, lead=[None, 1.24], area=[0.0, 40.0]),
        _phenotype_row("Q1", q=0.9792, onset_ms=13000.0, lead=[None, 1.19], area=[0.0, 68.0]),
        _phenotype_row("Q2", q=1.2913, onset_ms=12000.0, lead=[None, 1.21], area=[0.0, 29.0]),
        _phenotype_row("Q3", q=1.4994, onset_ms=6000.0, lead=[None, 0.57], area=[0.0, 280.0]),
    ]}
    summaries = {
        condition: {"n_returning_pre_onset": 30}
        for condition in ("C0", "C1", "Q1", "Q2", "Q3")
    }

    rows = closeout._trajectory_rows(phenotype, natural_summaries=summaries)
    inside = {row["condition"] for row in rows if row["in_q_matched_control_band"]}

    assert inside == {"C0", "C1", "Q1"}


def test_trajectory_rows_marks_degenerate_axial_front_readouts():
    """The halo lead exists only in the first supra-threshold bin, so it carries no signal."""

    phenotype = {"rows": [_phenotype_row(
        "C0", q=0.9583, onset_ms=10000.0,
        lead=[None, 1.236, 0.02, -0.004, 0.01],
        area=[0.0, 39.8, 193.4, 286.3, 400.0],
    )]}

    row = closeout._trajectory_rows(
        phenotype, natural_summaries={"C0": {"n_returning_pre_onset": 28}},
    )[0]
    degeneracy = row["front_readout_degeneracy"]

    assert degeneracy["degenerate"] is True
    assert degeneracy["first_supra_threshold_area_mm2"] == pytest.approx(39.8)
    assert degeneracy["later_bin_max_abs_lead_mm"] < closeout.AXIAL_BIN_MM


def test_trajectory_rows_keeps_a_real_moving_halo():
    """A lead that survives past the first bin by more than a grid bin is not degenerate."""

    phenotype = {"rows": [_phenotype_row(
        "C0", q=1.5, onset_ms=5000.0,
        lead=[None, 1.2, 2.4, 3.6],
        area=[0.0, 40.0, 120.0, 260.0],
        width=[8.0, 9.0, 12.0, 16.0],
    )]}

    row = closeout._trajectory_rows(
        phenotype, natural_summaries={"C0": {"n_returning_pre_onset": 20}},
    )[0]

    assert row["front_readout_degeneracy"]["degenerate"] is False


def test_closeout_table_reports_the_control_band_and_no_halo_column():
    rows = [{
        "condition": "Q3", "two_hop_q": 1.49, "onset_s": 6.0,
        "in_q_matched_control_band": False,
        "n_returning_pre_onset": 21, "peak_global_rate_100ms_hz": 395.7,
        "baseline_tradeoff": True,
    }]

    text = closeout._fmt_table(rows)

    assert "saturation" in text
    assert "carrier" not in text
    assert "同 q 对照带内" in text
    assert "halo" not in text.lower()


def test_escalation_after_entry_compares_against_the_q_matched_spread():
    summaries = {
        "C0": {"onset_ms": 2000.0, "per_second_mean_rate_hz": [4.0, 4.0, 10.0, 100.0]},
        "C1": {"onset_ms": 1000.0, "per_second_mean_rate_hz": [4.0, 20.0, 200.0, 300.0]},
        "Q3": {"onset_ms": 1000.0, "per_second_mean_rate_hz": [4.0, 15.0, 150.0, 400.0]},
    }

    result = closeout._escalation_after_entry(["C0", "C1"], summaries)

    assert result["compared_depth_s"] == 2
    assert result["seconds_from_entry_to_arm_end"] == {"C0": 1, "C1": 2, "Q3": 2}
    assert result["per_second"][0]["q_matched_spread_ratio"] == pytest.approx(2.0)
    assert result["max_all_arm_spread_ratio"] >= result["per_second"][0]["all_arm_spread_ratio"]

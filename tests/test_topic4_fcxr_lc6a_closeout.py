import scripts.finalize_topic4_fcxr_lc6a as closeout


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
    phenotype = {
        "rows": [{
            "condition": "Q2",
            "effective_onset_ms": 12000.0,
            "baseline_metrics": {"n_events": 33},
            "global_rate_100ms_peak_hz": 398.0,
            "local_rate_q99_peak_hz": 429.0,
            "spatial_slow_flow": {
                "max_D_halo_lead_mm": 1.2,
                "max_active_area_mm2": 400.0,
            },
            "boundedness": {"boundedness_margin": -1.5},
            "baseline_tradeoff": {"tradeoff": True},
            "headline": "SATURATED_HIGH_STATE",
        }]
    }

    row = closeout._trajectory_rows(phenotype)[0]

    assert row["onset_s"] == 12.0
    assert row["n_returning_pre_onset"] == 33
    assert row["baseline_tradeoff"] is True
    assert row["headline"] == "SATURATED_HIGH_STATE"


def test_closeout_table_does_not_call_saturation_a_carrier():
    row = {
        "condition": "Q3", "two_hop_q": 1.49, "onset_s": 6.0,
        "n_returning_pre_onset": 21, "peak_global_rate_hz": 395.7,
        "D_halo_lead_mm": 0.57, "baseline_tradeoff": True,
    }

    text = closeout._fmt_table([row])

    assert "saturation" in text
    assert "carrier" not in text

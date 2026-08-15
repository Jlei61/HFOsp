from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _bandpass_contact_activity,
    _calibration_caption,
    _calibration_columns,
    _nlc_readme_text,
    _figure2a_context_geometry,
    _equal_network_row,
    _map_kmeans_clusters_to_modes,
    _same_network_pair,
    _science_status,
    _status_banner,
    _write_readme,
    formal_clean_mask,
    normalize_event_ranks,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_clean_mask_requires_both_shafts_and_patient_support():
    onsets = np.asarray([
        [0.0, np.nan, 1.0],
        [0.0, 1.0, np.nan],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])
    clean = formal_clean_mask(
        onsets, np.asarray([0, 0, 1, 1]), np.asarray([False, False, False, True]),
        {"ICL": np.asarray([0]), "SCL": np.asarray([1, 2])},
    )
    assert clean.tolist() == [True, True, True, False]


def test_formal_clean_mask_excludes_nonreturned_event():
    clean = formal_clean_mask(
        np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        np.asarray([0, 1]), np.asarray([False, False]),
        {"ICL": np.asarray([0]), "SCL": np.asarray([1])},
        event_returned=np.asarray([True, False]),
    )
    assert clean.tolist() == [True, False]


def test_rank_normalization_preserves_missing_contacts():
    values = normalize_event_ranks(np.asarray([[4.0, np.nan, 2.0, 3.0]]))
    np.testing.assert_allclose(values[0, [0, 2, 3]], [1.0, 0.0, 0.5])
    assert np.isnan(values[0, 1])


def test_same_network_pair_never_crosses_networks():
    bundle = {
        "config": {"search": {"fit_network_seeds": [1, 2]}},
        "clean": np.asarray([True, True]),
        "labels": np.asarray([0, 1]),
        "records": [
            {"seed": 1, "local_index": 0},
            {"seed": 2, "local_index": 0},
        ],
        "blocks": [
            {"seed": 1, "event_t_on_ms": np.asarray([0.0]),
             "event_t_off_ms": np.asarray([10.0])},
            {"seed": 2, "event_t_on_ms": np.asarray([0.0]),
             "event_t_off_ms": np.asarray([10.0])},
        ],
    }
    assert _same_network_pair(bundle) is None
    bundle["records"][1]["seed"] = 1
    pair = _same_network_pair(bundle)
    assert pair[0:2] == (1, 0)
    assert pair[2]["seed"] == 1


def test_confirmation_readme_preserves_pre_network_freeze_semantics(tmp_path):
    _write_readme(tmp_path, {
        "candidate_id": "edge_spatial_02_pos",
        "phase": "confirmation",
    })
    text = (tmp_path / "README.md").read_text()
    assert "selection 阶段在读取确认网络前冻结的非零候选" in text
    assert "fit screen 冻结的 diagnostic best" not in text


def test_figure_consumer_has_no_simulation_or_candidate_selection():
    path = ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
    tree = ast.parse(path.read_text())
    calls = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "simulate_kick" not in calls
    source = path.read_text()
    assert 'summary["diagnostic_best_candidate_id"]' in source
    assert "fit_network_seeds" in source
    assert "same-network" in source
    assert "a.u." in source
    assert 'manifest["selection_freeze"]["selected_nonzero_candidate_id"]' in source
    assert 'verdict["diagnostic_display_candidate_id"]' in source
    assert "post-run preregistered diagnostic display rule" in source
    assert "candidate_is_phase_diagnostic_best" in source


def test_d62_explicit_audit_candidate_does_not_require_verdict():
    source = (
        ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
    ).read_text()
    assert "candidate_id is None or not allow_exploratory_candidate" in source
    assert "explicit frozen-library audit candidate" in source


def test_d63_figure_reads_two_arm_replication_schema():
    source = (
        ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
    ).read_text()
    assert 'd61_row = verdict["candidate_metrics"]' in source
    assert "D6.3 figure candidate differs from verdict" in source


def test_kmeans_display_mapping_uses_best_two_mode_permutation():
    labels = np.array([0, 0, 0, 1, 1])
    contingency = np.array([[1, 2], [2, 0]])

    mapped, cluster_to_mode = _map_kmeans_clusters_to_modes(labels, contingency)

    assert np.array_equal(cluster_to_mode, [1, 0])
    assert np.array_equal(mapped, [1, 1, 1, 0, 0])


def test_same_network_pair_prefers_supported_separated_opposite_modes():
    bundle = {
        "network_seed_key": "fit_network_seeds",
        "config": {"search": {"fit_network_seeds": [11]}},
        "clean": np.array([True, True, True, True]),
        "labels": np.array([0, 1, 0, 1]),
        "onsets": np.array([
            [0.0] * 12 + [np.nan] * 3,
            [0.0] * 11 + [np.nan] * 4,
            [0.0] * 4 + [np.nan] * 11,
            [0.0] * 4 + [np.nan] * 11,
        ]),
        "records": [
            {"seed": 11, "local_index": 0},
            {"seed": 11, "local_index": 1},
            {"seed": 11, "local_index": 2},
            {"seed": 11, "local_index": 3},
        ],
        "blocks": [{
            "seed": 11,
            "event_t_on_ms": np.array([100.0, 900.0, 500.0, 540.0]),
            "event_t_off_ms": np.array([120.0, 920.0, 520.0, 560.0]),
        }],
    }

    ta_index, tb_index, block = _same_network_pair(bundle)

    assert (ta_index, tb_index) == (1, 0)
    assert block["seed"] == 11


def test_figure2a_context_includes_gray_scl_contacts_outside_readout_contract():
    contract = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev10_sa/"
        "shaft_aware_target/contact_shaft_contract.json"
    )
    import json

    rows = json.loads(contract.read_text())["contacts"]
    bundle = {"static": {
        "contact_names": np.asarray([row["contact_name"] for row in rows]),
        "contact_xy_mm": np.asarray([row["sheet_xy_mm"] for row in rows]),
    }}

    context = _figure2a_context_geometry(bundle)

    assert context["n_context_contacts"] == 20
    assert context["n_selected_contacts"] == 15
    assert context["context_not_selected"] == [
        "SCL1", "SCL2", "SCL3", "SCL4", "SCL5",
    ]
    assert context["registration_max_abs_error_mm"] < 1e-9


def test_contact_activity_bandpass_is_signed_and_input_preserving():
    dt_ms = 2.0
    time_s = np.arange(0.0, 2.0, dt_ms / 1000.0)
    envelope = np.vstack([
        2.0 + np.sin(2.0 * np.pi * 50.0 * time_s),
        3.0 + 0.5 * np.sin(2.0 * np.pi * 12.0 * time_s),
    ])
    original = envelope.copy()

    filtered = _bandpass_contact_activity(envelope, dt_ms)

    assert np.array_equal(envelope, original)
    assert filtered.shape == envelope.shape
    assert np.min(filtered[0]) < -0.5
    assert np.max(filtered[0]) > 0.5
    assert np.std(filtered[0]) > 8.0 * np.std(filtered[1])


def _verdict_bundle(tmp_path, *, seeds, replication_pass=False):
    (tmp_path / "confirmation_verdict.json").write_text(json.dumps({
        "status": "REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED",
        "fig4_acceptance": "DIAGNOSTIC_ONLY",
        "replication_pass": replication_pass,
        "replication_rule": "both paired lower bounds above zero",
        "network_seed_is_the_independent_unit": True,
    }))
    return {
        "output_root": tmp_path,
        "candidate_id": "d62_a0p5_b0p5",
        "phase": "confirmation",
        "candidate": {"spatial_ou": {"mode": "local"}},
        "network_seed_key": "confirmation_network_seeds",
        "config": {
            "scientific_role": (
                "development_only_continuous_field_joint_direction_replication"
            ),
            "search": {"confirmation_network_seeds": list(seeds)},
        },
    }


def test_science_status_travels_with_the_figure(tmp_path):
    status = _science_status(_verdict_bundle(tmp_path, seeds=range(1401, 1413)))
    assert status["fig4_acceptance"] == "DIAGNOSTIC_ONLY"
    assert status["verdict_status"].endswith("NOT_REPLICATED")
    assert status["replication_pass"] is False
    assert _science_status({"output_root": tmp_path / "missing"}) is None


def test_nlc_science_status_and_equal_network_metrics_follow_nested_verdict(tmp_path):
    (tmp_path / "confirmation_verdict.json").write_text(json.dumps({
        "status": "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PASS",
        "figure_eligible": True,
        "candidate_rows": [{
            "candidate_id": "joint_04_control",
            "crossfit_and_natural_metrics": {
                "candidate_id": "joint_04_control",
                "natural_kmeans_by_network": {},
                "natural_balanced_alignment_equal_network": {
                    "n_networks": 12,
                },
                "crossfit_margin_equal_network": {"n_networks": 12},
            },
        }],
    }))
    bundle = {
        "output_root": tmp_path,
        "candidate_id": "joint_04_control",
        "config": {"scientific_role": (
            "development_only_data_driven_node_local_connectivity_"
            "frozen_confirmation"
        )},
    }

    status = _science_status(bundle)
    row = _equal_network_row(bundle)

    assert status["figure_eligible"] is True
    assert row["candidate_id"] == "joint_04_control"
    assert row["natural_balanced_alignment_equal_network"]["n_networks"] == 12


def test_status_banner_states_the_accepted_verdict():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    text = _status_banner(fig, {
        "fig4_acceptance": "DIAGNOSTIC_ONLY",
        "verdict_status": "REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED",
        "replication_pass": False,
    })
    plt.close(fig)
    assert "DIAGNOSTIC_ONLY" in text
    assert "NOT_REPLICATED" in text
    assert "replication rule NOT met" in text
    assert _status_banner(fig, None) is None


def test_status_banner_maps_nlc_eligibility_without_inventing_replication():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    text = _status_banner(fig, {
        "figure_eligible": True,
        "verdict_status": "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PASS",
        "replication_pass": None,
    })
    plt.close(fig)
    assert "FIGURE_ELIGIBLE" in text
    assert "CONFIRMATION_PASS" in text
    assert "replication rule" not in text


def test_readme_reports_the_real_network_count_and_verdict(tmp_path):
    _write_readme(tmp_path, _verdict_bundle(tmp_path, seeds=range(1401, 1413)))
    text = (tmp_path / "README.md").read_text()
    assert "12 张网络等权" in text
    assert "6 张网络" not in text
    assert "DIAGNOSTIC_ONLY" in text
    assert "不得替换主文 Fig.4" in text


def test_figure_b_keeps_the_required_qualifiers_on_canvas():
    source = (
        ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
    ).read_text()
    # The spec requires purity, stability, within-cluster consistency and the
    # supervised matrix to be reported on the figure, not only in metadata.
    assert "visible_qualifier_removed" not in source
    assert "_kmeans_qualifier_caption" in source
    for fragment in ("natural KMeans: purity", "seed AMI", "within-cluster tau",
                     "supervised MTA-TA", "pooled descriptive MTA-TA"):
        assert fragment in source
    # A replication-failed diagnostic run must not print a validation status.
    assert "REV10D6_3_FIG4_DIAGNOSTIC_COMPLETE" in source


def test_landscape_colorbar_stays_inside_its_own_cell():
    source = (
        ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
    ).read_text()
    assert "plt.colorbar(surface, ax=ax" not in source
    assert "figure.colorbar(surface, cax=colorbar_ax)" in source


def _calibration_fixture():
    def arm(alignment, margin, purity, seeds):
        return {
            "per_network": {
                str(seed): {
                    "observed_balanced_alignment": alignment + 0.01 * index,
                    "observed_crossfit_margin": margin + 0.01 * index,
                }
                for index, seed in enumerate(seeds)
            },
            "balanced_alignment_vs_label_permutation_null": {
                "observed_equal_network_mean": alignment,
                "null_median": 0.53, "null_q95": 0.57, "one_sided_p": 0.0005,
            },
            "crossfit_margin_vs_contact_permutation_null": {
                "observed_equal_network_mean": margin,
                "null_median": -0.25, "null_q95": -0.12, "one_sided_p": 0.001,
            },
            "equal_network_alignment_bootstrap": {"equal_network_mean": alignment},
            "equal_network_margin_bootstrap": {"equal_network_mean": margin},
            "pooled_diagnostics": {
                "pooled_kmeans_direction_purity": purity,
                "patient_matched_kmeans_direction_purity": {"q05": 0.898},
            },
        }

    return {
        "status": "REV11NLC_NULL_CALIBRATION_COMPLETE",
        "gates_do_not_separate_arms": True,
        "arms_passing_both_uncalibrated_gates": [
            "node_baseline", "joint_04_control",
        ],
        "arms": {
            "node_baseline": arm(0.662, 0.423, 0.674, (1561, 1562, 1563)),
            "joint_04_ee_only": arm(0.765, 0.481, 0.700, (1561, 1562, 1563)),
            "joint_04_etoi_only": arm(0.732, 0.398, 0.743, (1561, 1562)),
            "joint_04_control": arm(0.705, 0.484, 0.663, (1561, 1562, 1563)),
        },
    }


def test_calibration_columns_keep_arm_order_and_shared_networks():
    columns, seeds, values = _calibration_columns(
        _calibration_fixture(), "observed_balanced_alignment",
    )
    assert [candidate for candidate, _ in columns] == [
        "node_baseline", "joint_04_ee_only", "joint_04_etoi_only",
        "joint_04_control",
    ]
    # Only networks every arm evaluated may enter the paired comparison.
    assert seeds == ["1561", "1562"]
    assert values.shape == (2, 4)


def test_calibration_caption_names_the_control_and_the_patient_benchmark():
    text = _calibration_caption(_calibration_fixture())
    assert "Node-only control also clears both fixed thresholds" in text
    assert "0.662" in text and "0.423" in text
    assert "patient-matched benchmark q05 0.898" in text
    assert "label-permutation chance 0.530" in text
    assert "contact-permutation chance -0.250" in text
    assert "unsupervised KMeans clusters" in text


def test_status_banner_leads_with_plain_language():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    text = _status_banner(fig, {
        "figure_eligible": True,
        "verdict_status": "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PASS",
    }, plain_prefix="development check: static substrate")
    plt.close(fig)
    assert text.startswith("development check: static substrate")
    assert "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PASS" in text


def test_nlc_readme_states_the_control_arm_when_calibration_exists(tmp_path):
    (tmp_path / "null_calibration.json").write_text(
        json.dumps(_calibration_fixture())
    )
    bundle = {"output_root": tmp_path}
    text = _nlc_readme_text(bundle, 12)
    assert "只有地形" in text
    assert "0.898" in text
    assert "不是临床 SEEG" in text
    assert "unavailable" not in text


def test_nlc_readme_says_when_no_calibration_is_available(tmp_path):
    text = _nlc_readme_text({"output_root": tmp_path}, 12)
    assert "null_calibration.json" in text
    assert "unavailable" in text

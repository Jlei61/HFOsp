from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _bandpass_contact_activity,
    _figure2a_context_geometry,
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

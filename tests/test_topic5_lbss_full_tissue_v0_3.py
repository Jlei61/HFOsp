from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.audit_topic5_lbss_full_tissue_early_ictal_metadata_v0_3 import (
    EVENT_COLUMNS,
    target_metadata_path,
)
from scripts.score_topic5_lbss_full_tissue_early_ictal_v0_3 import (
    aggregate_subject,
    exact_align,
)
from scripts.score_topic5_lbss_early_ictal_v0_2 import (
    ARMS as SCORER_ARMS,
    REQUIRED_ATTENUATION_TARGETS,
    load_candidates,
)
from scripts.prepare_topic5_lbss_selected_primary_root_v0_4 import prepare
from scripts.run_topic5_lbss_spatial_decision_watcher_v0_4 import (
    current_contract_is_selective,
)
from scripts.run_topic5_lbss_spatial_search_v0_4 import load_rows
from scripts.summarize_topic5_lbss_model_field_recovery_v0_3 import (
    ARMS as FIELD_ARMS,
    METRICS as FIELD_METRICS,
    SHUFFLE as FIELD_SHUFFLE,
    summarize as summarize_model_field_recovery,
)
from scripts.summarize_topic5_lbss_topology_plateau_v0_3 import (
    ARMS as TOPOLOGY_ARMS,
    summarize as summarize_topology_plateau,
)


def test_v03_closeout_requires_no_recurrence_equivalence_audit():
    source = (
        __import__("pathlib").Path(
            "scripts/audit_topic5_lbss_full_tissue_closeout_v0_3.py"
        ).read_text()
    )
    assert "NO_REC_EQUIVALENCE_AUDIT.json" in source
    assert "EQUIVALENT_ENOUGH_FOR_MATCHED_CONTRAST" in source


def test_formal_selected_loader_accepts_full_fit_inventory(tmp_path):
    search_name = "search"
    config = "chosen"
    fits = ("fit_a", "fit_b")
    arms = ("L0", "L3")
    for fit in fits:
        for arm in arms:
            for seed in (0, 1, 2):
                unit = (
                    tmp_path / search_name / "units" / "formal_selected" / config
                    / fit / arm / f"seed{seed}"
                )
                unit.mkdir(parents=True)
                (unit / "metrics.json").write_text(json.dumps({
                    "subject": fit,
                    "test": {"contact_nll": 1.0},
                    "distance_bins": {"distal": {"contact_nll": 1.1}},
                    "rollout": {"seed_removed_spearman_median": 0.5},
                    "converged": True,
                    "best_checkpoint_eligible": True,
                    "hit_ceiling": False,
                    "target_values_read": False,
                }))
    rows = load_rows(
        tmp_path, search_name, "formal_selected", [config], arms, fits=fits
    )
    assert len(rows) == len(fits) * len(arms) * 3
    assert set(rows.fit_id) == set(fits)


def test_figure6_spatial_stars_use_within_panel_holm():
    source = __import__("pathlib").Path(
        "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py"
    ).read_text()
    assert "spatial_q = holm(spatial_p)" in source
    assert "spatial_holm_q_vs_channel_shuffle" in source


def test_figure6_pathway_stars_use_within_panel_holm():
    source = __import__("pathlib").Path(
        "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py"
    ).read_text()
    assert "q_values = holm(p_values)" in source
    assert '"endpoint_holm_q": q_values[0]' in source


def test_figure6_formal_mechanism_panels_use_two_sided_tests_and_eligible_controls():
    source = __import__("pathlib").Path(
        "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py"
    ).read_text()
    assert 'raw = [paired_test(values, "two-sided")' in source
    assert 'p_values = [paired_test(values, "two-sided")' in source
    assert 'auc.inferential_eligible.astype(bool)' in source
    assert 'dose_p = paired_test(dose_auc, "two-sided")' in source


def test_figure6_distal_panel_uses_actual_contrast_table_keys_and_rejects_empty_groups():
    source = __import__("pathlib").Path(
        "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py"
    ).read_text()
    assert 'L0: "L3_vs_L0"' in source
    assert 'SHUFFLE: "L3_vs_shuffle"' in source
    assert "panel G contrast inventory is missing or nonfinite" in source


def test_closeout_rejects_nonfinite_figure_metadata():
    source = __import__("pathlib").Path(
        "scripts/audit_topic5_lbss_full_tissue_closeout_v0_3.py"
    ).read_text()
    assert "def nonfinite_json_paths" in source
    assert "Figure 6 metadata contains nonfinite values" in source


def test_closeout_audits_target_free_spatial_selection_and_selected_root():
    source = __import__("pathlib").Path(
        "scripts/audit_topic5_lbss_full_tissue_closeout_v0_3.py"
    ).read_text()
    assert '"SPATIAL_DECISION_COMPLETE.json"' in source
    assert '"PRIMARY_ARTIFACT_POINTER.json"' in source
    assert '"FORMAL_SELECTED_DECISION.json"' in source
    assert '"MULTISTATE_REPAIR_PROVENANCE.json"' in source
    assert '"SELECTED_SPATIAL_CONFIG.json"' in source
    assert '"all_465_units_match_overrides"' in source
    assert 'expected_unit_caches = 31 * 3 * 4' in source
    assert 'attenuation unit caches incomplete' in source


def test_metadata_audit_never_deserializes_target_values(tmp_path):
    forbidden = {"observed", "null_median", "null_p95", "a_abs", "b_abs"}
    assert forbidden.isdisjoint(EVENT_COLUMNS)
    broadband = target_metadata_path(tmp_path, "epilepsiae_1146", "strict_broadband")
    gamma = target_metadata_path(tmp_path, "epilepsiae_1146", "gamma_nonbroadband")
    assert broadband.name == "epilepsiae_1146.json"
    assert "t0_feature_cache_bb150_1_150" in broadband.parts
    assert "v2_band_scan" in gamma.parts


def test_exact_align_preserves_missing_target_support():
    aligned = exact_align(["A", "C"], np.asarray([1.0, 3.0]), ["A", "B", "C"])
    assert np.allclose(aligned[[0, 2]], [1.0, 3.0])
    assert np.isnan(aligned[1])


def _write_candidate(path, contacts=("A", "B", "C")):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray([0.2, 0.5, 0.8], float)
    np.savez_compressed(
        path,
        contacts=np.asarray(contacts),
        A_canonical_full=values,
        B_canonical_full=values[::-1],
        A_seed_removed=values,
        B_seed_removed=values[::-1],
    )


def test_early_ictal_candidate_inventory_allows_absent_matched_local(tmp_path):
    subject = "p"
    for arm in SCORER_ARMS:
        _write_candidate(
            tmp_path / "model_fields/intact/per_patient" / subject / f"{arm}.npz"
        )
    for target in REQUIRED_ATTENUATION_TARGETS:
        for alpha in (0.25, 0.50, 0.75, 1.00):
            _write_candidate(
                tmp_path / "attenuation/fields/per_patient" / subject / target
                / f"alpha{alpha:.2f}.npz"
            )
    result = load_candidates(
        tmp_path, subject, "canonical_full", ["A", "B", "C"]
    )
    assert len(result) == len(SCORER_ARMS) + 3 * 4
    assert not any("L3_MATCHED_LOCAL" in key for key in result)


def test_early_ictal_candidate_inventory_rejects_partial_matched_local(tmp_path):
    subject = "p"
    for arm in SCORER_ARMS:
        _write_candidate(
            tmp_path / "model_fields/intact/per_patient" / subject / f"{arm}.npz"
        )
    for target in REQUIRED_ATTENUATION_TARGETS:
        for alpha in (0.25, 0.50, 0.75, 1.00):
            _write_candidate(
                tmp_path / "attenuation/fields/per_patient" / subject / target
                / f"alpha{alpha:.2f}.npz"
            )
    _write_candidate(
        tmp_path / "attenuation/fields/per_patient" / subject
        / "L3_MATCHED_LOCAL/alpha0.25.npz"
    )
    import pytest
    with pytest.raises(RuntimeError, match="only partially frozen"):
        load_candidates(tmp_path, subject, "canonical_full", ["A", "B", "C"])


def test_patient_aggregation_folds_seizures_before_cohort():
    rows = []
    nulls = {}
    for seizure in (0, 1):
        for condition in ("x", "y"):
            prefix = f"s|{seizure}|{condition}|canonical_full"
            for suffix in ("all", "shaft", "common"):
                nulls[f"{prefix}|{suffix}"] = np.asarray([0.1, 0.2, 0.3])
            rows.append({
                "subject": "s",
                "condition": condition,
                "endpoint": "canonical_full",
                "family": "intact",
                "arm": condition,
                "target": "",
                "alpha": 0.0,
                "n_contacts": 8,
                "observed": 0.4,
                "common_observed": 0.3,
                "within_shaft_permutable_contacts": 6,
                "null_key_all": f"{prefix}|all",
                "null_key_shaft": f"{prefix}|shaft",
                "null_key_common": f"{prefix}|common",
            })
    aggregated = aggregate_subject(
        pd.DataFrame(rows), nulls, group_label="all_phenotype_matched"
    )
    assert len(aggregated) == 2
    assert all(row["n_seizures"] == 2 for row in aggregated)
    assert all(np.isclose(row["all_contact_margin"], 0.2) for row in aggregated)


def test_confirmed_spatial_config_gets_isolated_primary_root(tmp_path):
    source = tmp_path / "formal"
    selected = tmp_path / "selected"
    search_name = "development_spatial_search_v0_4"
    search = source / search_name
    units = search / "units" / "formal_selected" / "radius_1p5"
    (source / "cache").mkdir(parents=True)
    search.mkdir(parents=True)
    (search / "SPATIAL_MODEL_DECISION.json").write_text(json.dumps({
        "selected_config_id": "radius_1p5", "target_values_read": False,
    }))
    (search / "FORMAL_SELECTED_DECISION.json").write_text(json.dumps({
        "config_id": "radius_1p5",
        "verdict": "FULL_COHORT_SELECTIVE_NONLOCAL_CONFIRMED",
        "target_values_read": False,
    }))
    (search / "configs").mkdir()
    (search / "configs/radius_1p5.json").write_text(json.dumps({
        "r_local_multiplier": 1.5,
    }))
    for name in (
        "INPUT_CACHE_MANIFEST.json", "FULL_TISSUE_CACHE_COMPLETE.json",
        "RUN_CONTRACT.json",
        "EARLY_ICTAL_METADATA_INVENTORY.json",
        "EARLY_ICTAL_METADATA_AUDIT_COMPLETE.json",
    ):
        (source / name).write_text("{}\n")
    (source / "LATENT_DOMAIN_AUDIT.csv").write_text("fit_id,version\np,v0.3\n")
    (source / "EARLY_ICTAL_METADATA_INVENTORY.csv").write_text("subject\np\n")
    arms = ("L0", "L1", "L2", "L3", "shuffle")
    for patient in range(31):
        for arm in arms:
            for seed in range(3):
                unit = units / f"p{patient:02d}" / arm / f"seed{seed}"
                unit.mkdir(parents=True)
                (unit / "metrics.json").write_text(json.dumps({
                    "target_values_read": False,
                    "best_checkpoint_eligible": True,
                    "hit_ceiling": False,
                    "config": {"r_local_multiplier": 1.5},
                }))
                (unit / "DONE.json").write_text(json.dumps({
                    "ok": True, "converged": True,
                }))
    payload = prepare(source, selected, search_name)
    assert payload["config_id"] == "radius_1p5"
    assert payload["unit_audit"]["n_metrics"] == 465
    assert (selected / "per_fit").is_symlink()
    assert (selected / "per_fit").resolve() == units.resolve()
    assert (selected / "cache").resolve() == (source / "cache").resolve()
    assert (selected / "LATENT_DOMAIN_AUDIT.csv").resolve() == (
        source / "LATENT_DOMAIN_AUDIT.csv"
    ).resolve()
    marker = json.loads((selected / "FORMAL_TRAINING_COMPLETE.json").read_text())
    assert marker["complete"] == 465
    assert marker["target_values_read"] is False
    selected_config = json.loads((selected / "SELECTED_SPATIAL_CONFIG.json").read_text())
    assert selected_config["overrides"] == {"r_local_multiplier": 1.5}
    assert selected_config["all_465_units_match_overrides"] is True


def test_current_spatial_contract_requires_distal_overall_and_rollout(tmp_path):
    arms = (
        "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
        "C_L3_ORDER_SHUFFLED",
    )
    rows = []
    for subject in ("p1", "p2", "p3"):
        for arm in arms:
            rows.append({
                "subject": subject,
                "arm": arm,
                "test_contact_nll": 1.00 if arm != "L3_LOCAL_PLUS_LEARNED_LR" else 1.02,
                "distal_contact_nll": 1.10 if arm != "L3_LOCAL_PLUS_LEARNED_LR" else 1.00,
                "rollout_spearman": 0.50 if arm != "L3_LOCAL_PLUS_LEARNED_LR" else 0.40,
            })
    pd.DataFrame(rows).to_csv(tmp_path / "interictal_per_patient.csv", index=False)
    retain, comparisons = current_contract_is_selective(tmp_path)
    assert comparisons["L0_LOCAL_ONLY"]["median_distal_gain"] > 0
    assert comparisons["L0_LOCAL_ONLY"]["median_overall_gain"] < -0.01
    assert comparisons["L0_LOCAL_ONLY"]["median_rollout_gain"] < -0.02
    assert retain is False


def test_model_field_recovery_summary_is_patient_first_and_target_free():
    rows = []
    for patient in range(21):
        subject = f"p{patient:02d}"
        for arm in (*FIELD_ARMS, FIELD_SHUFFLE):
            row = {"subject": subject, "arm": arm}
            for metric in FIELD_METRICS:
                row[metric] = 0.6 if arm != FIELD_SHUFFLE else 0.2
            rows.append(row)
    result = summarize_model_field_recovery(pd.DataFrame(rows))
    assert result["n_patients"] == 21
    assert result["patient_first"] is True
    assert result["target_values_read"] is False
    for metric in FIELD_METRICS:
        for arm in FIELD_ARMS:
            values = result["metrics"][metric][arm]
            assert values["field_n_positive"] == 21
            assert np.isclose(values["vs_order_shuffle"]["median"], 0.4)
            assert values["vs_order_shuffle"]["holm_q_within_metric"] < 0.05


def test_topology_plateau_summary_is_patient_first_and_target_free(tmp_path):
    rows = []
    fields = []
    for patient in range(21):
        subject = f"p{patient:02d}"
        for arm_index, arm in enumerate(TOPOLOGY_ARMS):
            rows.append({
                "subject": subject,
                "arm": arm,
                "test_contact_nll": 1.0 + 0.001 * arm_index,
                "distal_contact_nll": 1.2 + 0.001 * arm_index,
                "rollout_spearman": 0.5 - 0.001 * arm_index,
            })
            fields.append({
                "subject": subject,
                "arm": arm,
                "canonical_empirical_r": 0.4 - 0.001 * arm_index,
                "seed_removed_empirical_r": 0.3 - 0.001 * arm_index,
                "canonical_contrast_empirical_r": 0.7 - 0.001 * arm_index,
                "seed_removed_contrast_empirical_r": 0.6 - 0.001 * arm_index,
            })
    pd.DataFrame(rows).to_csv(tmp_path / "interictal_per_patient.csv", index=False)
    pd.DataFrame(fields).to_csv(tmp_path / "model_field_patient_metrics.csv", index=False)
    result = summarize_topology_plateau(tmp_path)
    assert result["n_patients"] == 21
    assert result["target_values_read"] is False
    assert result["early_ictal_values_used"] is False
    assert set(result["endpoints"]) == {
        "overall_contact_nll", "distal_contact_nll", "free_rollout_spearman",
        "canonical_interictal_field_r", "seed_removed_interictal_field_r",
        "canonical_ab_contrast_r", "seed_removed_ab_contrast_r",
    }


def test_closeout_requires_three_unit_cpu_gpu_attenuation_equivalence():
    source = __import__("pathlib").Path(
        "scripts/audit_topic5_lbss_full_tissue_closeout_v0_3.py"
    ).read_text()
    assert "ATTENUATION_DEVICE_EQUIVALENCE_AUDIT.json" in source
    assert 'int(device_audit.get("n_units", 0)) < 3' in source
    assert 'device_audit.get("all_units_pass") is not True' in source


def test_figure6_pretarget_contract_uses_frozen_orientation_and_explicit_input_support():
    source = __import__("pathlib").Path(
        "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py"
    ).read_text()
    assert "_align_tissue_plane_to_frozen_display" in source
    assert "_canonical_transverse_sign" in source
    assert "input_support" in source
    assert 'orientation="vertical"' in source
    assert 'parser.add_argument("--pretarget-preview"' in source

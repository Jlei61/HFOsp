from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts import audit_topic4_rev15_node_postselection as audit
from scripts import prepare_topic4_rev15_node_postselection_config as prepare
from scripts.paper_figures import (
    plot_topic4_rev15_node_postselection_fig4 as figure,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return {"path": None, "sha256": _sha(path)}


def _synthetic_preparation_inputs(tmp_path):
    repository = tmp_path / "repository"
    artifact = tmp_path / "artifact"
    manifest_path = artifact / "results/robust/candidate_manifest.json"
    aggregate_path = artifact / "results/robust/aggregate.json"
    robust_path = repository / "config/robust.json"
    j14_path = repository / "config/j14.json"
    patient_paths = {}
    for name in (
        "patient_training_target", "frozen_direction_classifier_manifest",
        "contact_contract",
    ):
        path = artifact / f"inputs/{name}.json"
        _write(path, {"name": name})
        patient_paths[name] = {
            "path": str(path.relative_to(artifact)), "sha256": _sha(path),
        }
    _write(j14_path, {"inputs": patient_paths})
    robust = {
        "candidate_manifest": str(manifest_path.relative_to(artifact)),
        "inputs": {
            "j14_config": {
                "path": str(j14_path.relative_to(repository)),
                "sha256": _sha(j14_path),
            },
        },
    }
    _write(robust_path, robust)
    _write(manifest_path, {"candidates": [{
        "candidate_id": "robust", "selection_eligible": True,
        "fourier_coordinate": {"coefficients_sha256": "c" * 64},
    }]})
    _write(aggregate_path, {
        "status": "COMPLETE",
        "inventory": {"complete_cartesian_product": True},
        "ranking_contract": {
            "natural_kmeans_used": False,
            "patient_heldout_used": False,
            "ictal_data_used": False,
            "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "best_usable_anchor": "robust",
        "usable_two_mode_anchor_ids": ["robust"],
    })
    figure2 = artifact / "inputs/figure2.json"
    _write(figure2, {"interictal_field": {}})
    return repository, artifact, robust_path, aggregate_path, figure2


def test_prepare_config_hashes_training_inputs_and_forbids_other_mechanisms(tmp_path):
    repository, artifact, robust, aggregate, figure2 = (
        _synthetic_preparation_inputs(tmp_path)
    )
    payload = prepare.build_config(
        robust_config_path=robust,
        robust_aggregate_path=aggregate,
        artifact_root=artifact,
        repository_root=repository,
        figure2_field_path=figure2,
    )
    assert payload["selected_candidate"]["candidate_id"] == "robust"
    assert payload["network_seeds"] == [2341, 2342, 2343]
    assert payload["event_contract"]["both_shafts_required"] is True
    assert payload["inputs"]["figure2_template_field"]["sha256"] == _sha(figure2)
    assert payload["boundaries"]["patient_heldout_used"] is False
    assert payload["boundaries"]["EE_EtoI_ZM"] == "off"


def test_prepare_config_fails_if_natural_kmeans_selected_field(tmp_path):
    repository, artifact, robust, aggregate, figure2 = (
        _synthetic_preparation_inputs(tmp_path)
    )
    payload = json.loads(aggregate.read_text())
    payload["ranking_contract"]["natural_kmeans_used"] = True
    aggregate.write_text(json.dumps(payload) + "\n")
    try:
        prepare.build_config(
            robust_config_path=robust,
            robust_aggregate_path=aggregate,
            artifact_root=artifact,
            repository_root=repository,
            figure2_field_path=figure2,
        )
    except RuntimeError as error:
        assert "forbidden boundary" in str(error)
    else:
        raise AssertionError("KMeans-informed Node selection was accepted")


def test_clean_event_mask_requires_readability_two_shafts_and_patient_support():
    onsets = np.asarray([
        [0.0, 1.0, np.nan, 2.0],
        [0.0, 1.0, np.nan, np.nan],
        [np.nan, np.nan, 1.0, np.nan],
        [0.0, 1.0, np.nan, 2.0],
        [0.0, 1.0, np.nan, 2.0],
    ])
    clean, summary = audit.clean_event_mask(
        readable=np.asarray([1, 1, 1, 1, 0], bool),
        onsets=onsets,
        ood=np.asarray([0, 0, 0, 1, 0], bool),
        groups={"ICL": np.asarray([0, 1]), "SCL": np.asarray([2, 3])},
    )
    assert clean.tolist() == [True, False, False, False, False]
    assert summary["n_formal_clean"] == 1


def test_cluster_mapping_is_posthoc_and_label_invariant():
    supervised = np.asarray([1, 1, 1, 0, 0, 0])
    raw_cluster = np.asarray([0, 0, 0, 1, 1, 1])
    mapped, summary = audit._map_clusters(raw_cluster, supervised)
    assert mapped.tolist() == supervised.tolist()
    assert summary["direction_purity"] == 1.0
    assert summary["ami_with_supervised_direction"] == 1.0


def test_semantic_audit_recovers_numeric_one_as_ta():
    ranks = np.asarray([
        [0, 1, 2, 3], [0, 1, 2, 3],
        [3, 2, 1, 0], [3, 2, 1, 0],
    ], float)
    patient = {
        "contact_names": np.asarray(["a", "b", "c", "d"]),
        "all_ranks": ranks,
        "all_labels": np.asarray([1, 1, 0, 0]),
    }
    figure2 = {"interictal_field": {
        "contact_order": ["a", "b", "c", "d"],
        "rank_a": [0, 1, 2, 3],
        "rank_b": [3, 2, 1, 0],
    }}
    result = audit.patient_semantic_audit(
        patient=patient, figure2_field=figure2,
    )
    assert result["status"] == "PASS"
    assert result["numeric_label_to_semantic_model_mode"] == {
        "0": "MTB", "1": "MTA",
    }


def _network_row(ami=0.9, supervised=(5, 5), kmeans=(5, 5)):
    return {
        "supervised_counts_MTA_MTB": list(supervised),
        "kmeans_counts_MTA_MTB": list(kmeans),
        "kmeans_ami_with_supervised_direction": ami,
    }


def _acceptance():
    return {
        "minimum_supervised_events_per_mode_per_network": 3,
        "minimum_kmeans_events_per_cluster_per_network": 3,
        "same_networks_with_both_modes_required": 3,
        "minimum_per_network_kmeans_ami_with_supervised_direction": 0.8,
        "networks_meeting_kmeans_ami_required": 3,
    }


def test_acceptance_requires_all_three_networks_and_matrix_signs():
    matrix = np.asarray([[0.8, -0.5], [-0.6, 0.7]])
    result = audit.acceptance_decision(
        network_rows=[_network_row(), _network_row(), _network_row()],
        pooled_matrix=matrix, acceptance=_acceptance(),
    )
    assert result["accepted"] is True
    failed = audit.acceptance_decision(
        network_rows=[_network_row(), _network_row(ami=0.7), _network_row()],
        pooled_matrix=matrix, acceptance=_acceptance(),
    )
    assert failed["accepted"] is False
    assert failed["clauses"]["same_network_natural_kmeans_alignment"][
        "passed_networks"
    ] == 2


def test_positive_crossed_cell_rejects_patient_profile_match():
    result = audit.acceptance_decision(
        network_rows=[_network_row(), _network_row(), _network_row()],
        pooled_matrix=np.asarray([[0.8, 0.1], [-0.6, 0.7]]),
        acceptance=_acceptance(),
    )
    assert result["accepted"] is False
    assert result["clauses"]["pooled_patient_profile_matrix_signs"]["pass"] is False


def test_final_figure_fails_closed_on_rejected_postselection(tmp_path):
    config = tmp_path / "config.json"
    result = tmp_path / "audit.json"
    config.write_text(json.dumps({
        "selected_candidate": {"candidate_id": "field"}, "inputs": {},
    }))
    result.write_text(json.dumps({
        "status": "NODE_POSTSELECTION_REJECTED", "candidate_id": "field",
    }))
    try:
        figure._load_bundle(
            config_path=config, audit_path=result, artifact_root=tmp_path,
            allow_rejected_diagnostic=False,
        )
    except RuntimeError as error:
        assert "not eligible" in str(error)
    else:
        raise AssertionError("rejected Node field produced a final Fig.4")


def test_figure_adapter_relocates_canonical_outputs_and_records_boundary(
    tmp_path, monkeypatch,
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output_root": "results/postselection"}))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}")
    bundle = {
        "candidate_id": "robust",
        "config_path": config,
        "postselection_audit": {
            "status": "NODE_POSTSELECTION_ACCEPTED",
            "acceptance": {"accepted": True},
            "pooled": {"kmeans_ami_with_supervised_direction": 0.91},
            "network_results": [{
                "network_seed": seed,
                "kmeans_ami_with_supervised_direction": 0.9,
                "supervised_counts_MTA_MTB": [5, 6],
            } for seed in (2341, 2342, 2343)],
        },
    }
    monkeypatch.setattr(figure, "_load_bundle", lambda **kwargs: bundle)

    def fake_renderer(_bundle, output_dir):
        stem = output_dir / "canonical_temporary"
        stem.with_suffix(".png").write_bytes(b"png")
        stem.with_suffix(".pdf").write_bytes(b"pdf")
        (output_dir / "canonical_temporary_metadata.json").write_text(
            json.dumps({"figure": "old", "files": {}})
        )
        return stem

    monkeypatch.setattr(figure.canonical, "_render_direct", fake_renderer)
    monkeypatch.setattr(figure.canonical, "_render_kmeans", fake_renderer)
    payload = figure.render(
        config_path=config, audit_path=audit_path, artifact_root=tmp_path,
    )
    output = tmp_path / "results/postselection/figures"
    assert (output / "fig4a_rev15_node_direct_readout.png").is_file()
    assert (output / "fig4b_rev15_node_kmeans_consistency.pdf").is_file()
    metadata = json.loads((
        output / "fig4a_rev15_node_direct_readout_metadata.json"
    ).read_text())
    assert metadata["patient_heldout_loaded"] is False
    assert metadata["EE_EtoI_ZM"] == "off"
    assert payload["SNN_simulation_run"] is False

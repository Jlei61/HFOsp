import json
from pathlib import Path

from scripts.paper_figures.validate_topic4_fig45_integration import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/topic4_fig45_data_driven_zm_integration.json"


def _contract():
    return json.loads(CONTRACT_PATH.read_text())


def test_committed_contract_is_valid_but_artifacts_are_pending():
    report = validate_contract(_contract(), repo_root=ROOT)
    assert report["status"] == "CONTRACT_VALID_ARTIFACTS_PENDING"
    assert report["errors"] == []


def test_fig5_is_exact_fig4_substrate_plus_zm_only():
    contract = _contract()
    fig4 = contract["figures"]["fig4"]
    fig5 = contract["figures"]["fig5"]
    assert fig5["inherits_frozen_substrate_from"] == fig4["model_id"]
    assert set(fig5["added_state_variables"]) == {"Z", "M"}


def test_artifacts_inside_repository_are_rejected():
    report = validate_contract(
        _contract(), repo_root=ROOT, artifact_root=ROOT / "results/topic4_fig45"
    )
    assert report["status"] == "INVALID"
    assert "artifact_root must be outside the Git repository" in report["errors"]


def test_required_external_artifact_directories_can_be_locked(tmp_path):
    contract = _contract()
    for figure in contract["figures"].values():
        (tmp_path / figure["artifact_subdir"]).mkdir(parents=True)
    report = validate_contract(
        contract,
        repo_root=ROOT,
        artifact_root=tmp_path,
        require_artifacts=True,
    )
    assert report["status"] == "CONTRACT_VALID_ARTIFACTS_READY"
    assert report["errors"] == []

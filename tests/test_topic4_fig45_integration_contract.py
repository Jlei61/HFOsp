import hashlib
import json
from pathlib import Path

from scripts.paper_figures.validate_topic4_fig45_integration import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/topic4_fig45_data_driven_zm_integration.json"


def _contract():
    return json.loads(CONTRACT_PATH.read_text())


def test_committed_contract_has_only_fig5a_locked():
    report = validate_contract(_contract(), repo_root=ROOT)
    assert report["status"] == "CONTRACT_VALID_PARTIALLY_LOCKED"
    assert report["locked_panels"] == ["fig5.fig5a_tonic_global_high"]
    assert report["figure_artifacts_ready"] == {"fig4": False, "fig5": False}
    assert report["artifacts_ready"] is False
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


def test_locked_panel_hashes_are_verified(tmp_path):
    contract = _contract()
    panel = contract["figures"]["fig5"]["locked_panels"]["fig5a_tonic_global_high"]
    panel_root = tmp_path / panel["artifact_subdir"]
    panel_root.mkdir(parents=True)
    expected = {}
    for index, relative_name in enumerate(panel["required_artifacts"]):
        payload = f"artifact-{index}".encode()
        (panel_root / relative_name).write_bytes(payload)
        expected[relative_name] = hashlib.sha256(payload).hexdigest()
    panel["required_artifacts"] = expected
    report = validate_contract(
        contract,
        repo_root=ROOT,
        artifact_root=tmp_path,
    )
    assert report["status"] == "CONTRACT_VALID_PARTIALLY_LOCKED"
    assert report["errors"] == []

    first_artifact = next(iter(expected))
    (panel_root / first_artifact).write_text("mutated")
    report = validate_contract(contract, repo_root=ROOT, artifact_root=tmp_path)
    assert report["status"] == "INVALID"
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_full_artifact_requirement_still_fails_for_partial_lock():
    report = validate_contract(_contract(), repo_root=ROOT, require_artifacts=True)
    assert report["status"] == "INVALID"
    assert "full Fig4/Fig5 artifact set is not locked" in report["errors"]

import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_topic4_rev10_d5_4_spatial_ou_kmeans_selection import adjudicate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_d5_4_spatial_ou_kmeans_selection.json"


def _row(*, evaluable, purity, margin):
    return {
        "evaluable": evaluable,
        "balanced_kmeans": {
            "purity_median": purity,
        },
        "signed_patient_margin": margin,
    }


def test_d5_4_selection_contract_is_frozen_and_minimal():
    config = json.loads(CONFIG.read_text())
    assert config["search"]["phase"] == "selection"
    assert config["search"]["selection_network_seeds"] == [1311, 1312, 1313]
    assert config["search"]["beta"] == "closed"
    assert config["search"]["network_topology"] == "frozen"
    assert config["search"]["kmeans_selection"]["minimum_fresh_purity"] == (
        0.6736842105263158
    )
    assert config["claim_boundary"]["selection_is_not_final_Fig4_confirmation"]


def test_d5_4_adjudication_does_not_use_permutation_as_hidden_gate():
    local = _row(evaluable=True, purity=0.8, margin=0.2)
    permuted = _row(evaluable=True, purity=0.9, margin=0.3)
    off = _row(evaluable=False, purity=None, margin=-1.0)
    verdict = adjudicate(local, permuted, off, minimum_purity=0.67)
    assert verdict["status"].endswith("CONFIRMS_KMEANS_CANDIDATE")
    assert verdict["local_minus_permuted_balanced_purity"] < 0


def test_d5_4_adjudication_separates_support_purity_and_geometry():
    off = _row(evaluable=False, purity=None, margin=-1.0)
    assert adjudicate(
        _row(evaluable=False, purity=None, margin=-1.0), off, off,
        minimum_purity=0.67,
    )["status"].endswith("SUPPORT_FAIL")
    assert adjudicate(
        _row(evaluable=True, purity=0.6, margin=0.2), off, off,
        minimum_purity=0.67,
    )["status"].endswith("KMEANS_NOT_REPLICATED")
    assert adjudicate(
        _row(evaluable=True, purity=0.8, margin=-0.1), off, off,
        minimum_purity=0.67,
    )["status"].endswith("PATIENT_GEOMETRY_FAIL")


def test_d5_4_entrypoints_are_directly_executable():
    for script in (
        "freeze_topic4_rev10_d5_4_spatial_ou_kmeans_selection.py",
        "audit_topic4_rev10_d5_4_spatial_ou_kmeans_selection.py",
    ):
        completed = subprocess.run([
            sys.executable, str(ROOT / "scripts" / script), "--help",
        ], cwd=ROOT, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr

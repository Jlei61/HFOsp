import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_topic4_rev10_d6_1_natural_kmeans_closeout import (
    paired_contrast,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_d6_1_natural_kmeans_closeout.json"


def _row(values):
    by_seed = {}
    for seed, (natural, margin, recruitment) in values.items():
        by_seed[str(seed)] = {
            "natural_kmeans": {
                "status": "OK" if natural is not None else "INSUFFICIENT_EVENTS",
                "direction_balanced_alignment": natural,
            },
            "crossfit_patient_readout": {"signed_margin": margin},
            "recruitment": {
                "A": {"absolute_error_fraction_of_15": recruitment},
                "B": {"absolute_error_fraction_of_15": recruitment},
            },
        }
    return {"natural_kmeans_by_network": by_seed}


def test_d61_contract_uses_fixed_duration_six_fresh_networks_and_closed_edge():
    config = json.loads(CONFIG.read_text())
    assert config["search"]["phase"] == "confirmation"
    assert config["search"]["confirmation_network_seeds"] == list(range(1341, 1347))
    assert config["search"]["simulation"]["duration_ms"] == 16000.0
    assert config["search"]["beta"] == "closed"
    assert config["spatial_edge_basis"]["role"] == "exact no-op edge adapter only"
    assert config["field_search"]["candidate_count"] == 5
    assert "secondary" in config["search"]["kmeans_selection"][
        "balanced_supervised_mode_purity"
    ]


def test_paired_contrast_excludes_missing_network_metric_instead_of_zero_filling():
    candidate = _row({
        1: (0.8, 0.4, 0.1),
        2: (None, 0.3, 0.2),
        3: (0.7, None, 0.1),
    })
    baseline = _row({
        1: (0.6, 0.2, 0.2),
        2: (0.5, 0.1, 0.2),
        3: (0.8, 0.1, 0.2),
    })
    result = paired_contrast(candidate, baseline, [1, 2, 3])
    assert result["n_natural_paired_networks"] == 2
    assert result["networks_with_positive_natural_delta"] == 1
    assert result["networks_with_positive_crossfit_delta"] == 2
    assert result["networks_with_positive_recruitment_improvement"] == 2


def test_d61_entrypoints_are_directly_executable():
    for script in (
        "freeze_topic4_rev10_d6_1_natural_kmeans_closeout.py",
        "audit_topic4_rev10_d6_1_natural_kmeans_closeout.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

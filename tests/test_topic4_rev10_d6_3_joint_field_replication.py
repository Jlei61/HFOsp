import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_topic4_rev10_d6_3_joint_field_replication import (
    replication_pass,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_d6_3_joint_field_replication.json"


def _contrast(natural_q05, crossfit_q05):
    return {
        "natural_alignment_delta": {"network_bootstrap_q05": natural_q05},
        "crossfit_margin_delta": {"network_bootstrap_q05": crossfit_q05},
    }


def test_d63_contract_is_two_arm_twelve_network_fixed_replication():
    config = json.loads(CONFIG.read_text())
    assert config["field_replication"]["candidate_count"] == 2
    assert config["field_replication"]["component_count"] is None
    assert config["search"]["confirmation_network_seeds"] == list(range(1401, 1413))
    assert config["search"]["simulation"]["duration_ms"] == 16000.0
    assert config["search"]["beta"] == "closed"
    assert "new field directions" in config["forbidden"]
    assert "slow variables" in config["forbidden"]


def test_d63_replication_rule_uses_both_metrics_density_and_safety():
    assert replication_pass(_contrast(0.01, 0.02), density_count=8, n_runaway=0)
    assert not replication_pass(_contrast(-0.01, 0.02), density_count=12, n_runaway=0)
    assert not replication_pass(_contrast(0.01, 0.02), density_count=7, n_runaway=0)
    assert not replication_pass(_contrast(0.01, 0.02), density_count=12, n_runaway=1)


def test_d63_entrypoints_are_directly_executable():
    for script in (
        "freeze_topic4_rev10_d6_3_joint_field_replication.py",
        "audit_topic4_rev10_d6_3_joint_field_replication.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

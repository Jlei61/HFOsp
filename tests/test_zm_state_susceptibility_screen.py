"""The state screen must not quietly become a cohort claim.

Three networks is the whole sample. These tests pin the two ways that fact could
be lost: escalating on fewer than three seeds, and counting sites where the probe
itself ignited the network as if they were sub-event responses.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_topic4_zm_state_susceptibility.py"
PY = sys.executable


def _row(site, susceptibility, *, event=False, ictal=False):
    return {"site_id": site, "susceptibility": susceptibility,
            "excess_spikes_early": susceptibility * 0.6,
            "excess_spikes_late": susceptibility * 0.4,
            "r90_mm": 2.0, "contact_excess_energy": 1.0,
            "probe_attributable_event_200ms": event,
            "reached_model_ictal_200ms": ictal}


def _write(root, joint, seed, label, dose, rows, t_ms):
    path = root / f"{joint}_seed_{seed}_{label}_n{dose}.json"
    path.write_text(json.dumps({"rows": rows, "checkpoint_absolute_time_ms": t_ms}))


@pytest.fixture
def sandbox(tmp_path):
    out = tmp_path / "out"
    (out / "perturbation").mkdir(parents=True)
    config = {"output_root": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT)
              else str(out),
              "arms": {"Joint": "joint_04_control"}}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    return cfg, out


def _run(cfg, seeds, dose=64):
    proc = subprocess.run(
        [PY, str(SCRIPT), "--config", str(cfg), "--dose", str(dose),
         "--seeds", *[str(s) for s in seeds]],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    return proc


def test_three_consistent_seeds_escalate(sandbox, monkeypatch):
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row(f"s{i}", 180.0) for i in range(6)], 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["screen"]["seeds_with_both_states"] == 3
    assert report["screen"]["all_seed_medians_same_sign"] is True
    assert report["state_difference_established_for_escalation"] is True


def test_two_seeds_never_escalate(sandbox):
    cfg, out = sandbox
    for seed in (1801, 1802):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row(f"s{i}", 180.0) for i in range(6)], 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["state_difference_established_for_escalation"] is False
    assert report["missing"], "the absent seed must be recorded, not silently dropped"


def test_disagreeing_seeds_do_not_escalate(sandbox):
    cfg, out = sandbox
    for seed, pre in ((1801, 180.0), (1802, 180.0), (1803, 40.0)):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row(f"s{i}", pre) for i in range(6)], 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["screen"]["all_seed_medians_same_sign"] is False
    assert report["state_difference_established_for_escalation"] is False


def test_probe_attributable_sites_are_excluded_not_counted(sandbox):
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        rows = [_row(f"s{i}", 180.0) for i in range(4)]
        rows += [_row(f"s{i}", 9000.0, event=True, ictal=True) for i in (4, 5)]
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               rows, 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["screen"]["site_units_excluded_probe_attributable"] == 6
    assert report["screen"]["site_units_comparable"] == 12
    assert report["screen"]["site_units_positive"] == 12


def test_claim_boundary_names_the_sample_size(sandbox):
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row("s0", 100.0)], 1000.0)
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row("s0", 180.0)], 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    text = " ".join(report["claim_boundary"])
    assert "n = 3" in text and "not a cohort claim" in text

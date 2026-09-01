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


def test_an_ignited_site_cannot_inflate_the_seed_median(sandbox):
    """The bug this pins: a probe that ignites the network at the pre-ictal state
    answers with ~10^4 excess spikes. If that site stays in the median, three
    seeds all 'agree' and the screen escalates on contamination alone."""
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        # two comparable sites go DOWN, four ignited sites go hugely up. A median
        # tolerates a minority of outliers, so the contamination only shows once
        # the ignited sites are the majority -- which is exactly the situation
        # the pre-ictal state makes likely.
        rows = [_row(f"s{i}", 60.0) for i in range(2)]
        rows += [_row(f"s{i}", 90000.0, event=True, ictal=True) for i in range(2, 6)]
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               rows, 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    for seed in ("1801", "1802", "1803"):
        entry = report["per_seed"][seed]
        assert entry["n_comparable_sites"] == 2
        assert entry["median_delta_susceptibility"] == -40.0
        assert entry["median_delta_susceptibility_including_ignited"] > 1000.0
    # the reported screen must follow the comparable-only median
    assert all(v < 0 for v in report["screen"]["seed_median_deltas"])
    assert report["screen"]["site_units_excluded_probe_attributable"] == 12


def test_a_seed_whose_every_site_ignited_is_named_not_dropped(sandbox):
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        ignite = seed == 1803
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row(f"s{i}", 180.0, event=ignite, ictal=ignite) for i in range(6)],
               3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["screen"]["seeds_with_no_comparable_site"] == [1803]
    assert report["state_difference_established_for_escalation"] is False


def test_a_dose_that_ignites_only_at_pre_ictal_still_establishes_a_difference(sandbox):
    """The false negative this guards: the dose is calibrated to be sub-event at
    the low-activity state, so igniting at the pre-ictal state IS the difference.
    Reporting only the graded endpoint would call this 'no comparable units'."""
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row(f"s{i}", 90000.0, event=True, ictal=True) for i in range(6)],
               3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["screen"]["site_units_comparable"] == 0
    assert report["established_by"]["graded"] is False
    assert report["established_by"]["ignition"] is True
    assert report["state_difference_established_for_escalation"] is True
    assert report["ignition_endpoint"]["pre_ictal"] == 18
    assert report["ignition_endpoint"]["low_activity"] == 0


def test_equal_ignition_at_both_states_does_not_establish_anything(sandbox):
    cfg, out = sandbox
    for seed in (1801, 1802, 1803):
        rows = [_row(f"s{i}", 100.0) for i in range(4)]
        rows += [_row(f"s{i}", 90000.0, event=True, ictal=True) for i in (4, 5)]
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               rows, 1000.0)
        rows2 = [_row(f"s{i}", 100.0) for i in range(4)]
        rows2 += [_row(f"s{i}", 90000.0, event=True, ictal=True) for i in (4, 5)]
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               rows2, 3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["ignition_endpoint"]["difference"] == 0
    assert report["established_by"]["ignition"] is False
    assert report["state_difference_established_for_escalation"] is False


def test_ignition_endpoint_needs_all_three_seeds(sandbox):
    cfg, out = sandbox
    for seed in (1801, 1802):
        _write(out / "perturbation", "joint_04_control", seed, "low_activity", 64,
               [_row(f"s{i}", 100.0) for i in range(6)], 1000.0)
        _write(out / "perturbation", "joint_04_control", seed, "pre_ictal", 64,
               [_row(f"s{i}", 90000.0, event=True, ictal=True) for i in range(6)],
               3600.0)
    _run(cfg, [1801, 1802, 1803])
    report = json.loads((out / "state_susceptibility_screen.json").read_text())
    assert report["established_by"]["ignition"] is False
    assert report["state_difference_established_for_escalation"] is False

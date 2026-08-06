import numpy as np
import os
import subprocess
import sys
import pytest

from src.topic4_mz_conductance import oscillation_metrics, staircase_metrics


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "scripts", "run_topic4_mz_conductance.py")


def test_staircase_metrics_detect_event_locked_positive_steps():
    dt = 1.0
    D = np.zeros(1000)
    events = []
    for i, on in enumerate((100, 250, 400, 550, 700)):
        D[on:] += 0.01
        events.append(dict(t_on=float(on), t_off=float(on + 20), returned=True))
    out = staircase_metrics(events, 1.0 - D, dt)
    assert out["n_events"] == 5
    assert out["positive_delta_fraction"] == 1.0
    assert out["event_index_Dpre_spearman"] > 0.9
    assert out["event_locked_positive_increment_fraction"] > 0.9


def test_oscillation_metrics_separates_bursts_from_plateau():
    dt = 1.0
    t = np.arange(8000) * dt / 1000.0
    osc = 30.0 + 25.0 * np.maximum(np.sin(2 * np.pi * 2.0 * t), 0.0)
    af = 0.2 + 0.1 * np.maximum(np.sin(2 * np.pi * 2.0 * t), 0.0)
    out = oscillation_metrics(osc, dt, analysis_start_ms=500.0, baseline_rate=2.0, baseline_sigma=1.0,
                              active_fraction=af, af_bin_ms=dt, baseline_af_q95=0.1)
    assert out["oscillatory_candidate"] is True
    assert 1.0 < out["dominant_hz"] < 3.0
    flat = oscillation_metrics(np.full_like(osc, 40.0), dt, analysis_start_ms=500.0,
                               baseline_rate=2.0, baseline_sigma=1.0,
                               active_fraction=np.full_like(af, 0.2), af_bin_ms=dt,
                               baseline_af_q95=0.1)
    assert flat["oscillatory_candidate"] is False


def test_recovery_requires_full_two_second_tail_in_band():
    dt = 1.0
    rate = np.full(6000, 2.0)
    rate[500:3000] = 50.0
    out = oscillation_metrics(rate, dt, analysis_start_ms=500.0, baseline_rate=2.0, baseline_sigma=1.0)
    assert out["tail_rate_band"] is True
    rate[-500:] = 20.0
    out2 = oscillation_metrics(rate, dt, analysis_start_ms=500.0, baseline_rate=2.0, baseline_sigma=1.0)
    assert out2["tail_rate_band"] is False


def test_terminal_runaway_event_is_excluded_from_staircase():
    D = np.zeros(1000)
    events = []
    for on in (100, 250, 400, 550, 700):
        D[on:] += 0.01
        events.append(dict(t_on=float(on), t_off=float(on + 20), returned=True))
    clean = staircase_metrics(events, 1.0 - D, 1.0, transition_ms=900.0)
    D[900:] += 0.5
    contaminated = events + [dict(t_on=900.0, t_off=999.0, returned=False)]
    got = staircase_metrics(contaminated, 1.0 - D, 1.0, transition_ms=900.0)
    assert got["n_events"] == clean["n_events"]
    assert got["median_delta_D"] == clean["median_delta_D"]
    assert got["excluded_nonreturning"] == 1


def test_interictal_train_without_sustained_high_recruitment_is_not_ictal():
    rate = np.zeros(8000)
    af = np.zeros(8000)
    for on in (1000, 2000, 3000, 4000, 5000, 6000):
        rate[on:on + 30] = 50.0
        af[on:on + 30] = 0.2
    out = oscillation_metrics(rate, 1.0, analysis_start_ms=500.0, baseline_rate=1.0, baseline_sigma=1.0,
                              active_fraction=af, af_bin_ms=1.0, baseline_af_q95=0.1)
    assert out["high_duration_ms"] < 1000.0
    assert out["oscillatory_candidate"] is False


def test_runner_import_is_side_effect_free_and_exposes_cap():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mzcond_runner_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MAX_WORKERS == 4
    assert mod._base_cfg(1.0)["membrane_mode"] == "conductance"
    assert mod._base_cfg(1.0)["fail_on_clip"] is True
    assert mod._base_cfg(1.0)["e_gaba"] == 0.0
    assert mod._base_cfg(1.0, e_gaba=11.0)["e_gaba"] == 11.0


def test_figure_capture_onset_rank_is_windowed_and_normalized():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mzcond_runner_onset_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raster = np.zeros((20, 4), bool)
    raster[4, 0] = True          # outside the selected window
    raster[7, 2] = True
    raster[9, 1] = True
    onset, rank = mod._onset_rank(raster, 1.0, 5.0, 12.0)
    assert np.isnan(onset[0]) and np.isnan(rank[0])
    assert onset[2] == 7.0 and onset[1] == 9.0
    assert rank[2] == 0.0 and rank[1] == 1.0
    assert np.isnan(onset[3]) and np.isnan(rank[3])


def test_runner_task_contract_rejects_bad_pairs_and_duplicate_labels():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mzcond_runner_contract_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod._base_cfg(0.25)
    with pytest.raises(SystemExit, match="unique"):
        mod._validate_tasks([dict(label="x", cfg=cfg), dict(label="x", cfg=cfg)])
    with pytest.raises(SystemExit, match="paired control"):
        mod._validate_tasks([
            dict(label="response", cfg=cfg, kick_boost=1.0, analysis_start_ms=1200.0,
                 paired_control_label="missing")
        ])
    with pytest.raises(SystemExit, match="same explicit"):
        mod._validate_tasks([
            dict(label="control", cfg=cfg, kick_boost=0.0),
            dict(label="response", cfg=cfg, kick_boost=1.0, analysis_start_ms=1200.0,
                 paired_control_label="control"),
        ])


def test_runner_caps_long_retained_raster_at_two_workers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mzcond_runner_resource_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit, match="capped at 2 workers"):
        mod._resource_preflight(20000.0, 3)


def test_runner_refuses_sim_without_confirm_run():
    proc = subprocess.run(
        [sys.executable, RUNNER, "baseline-screen", "--seed", "1", "--T", "10", "--workers", "1"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "confirm-run" in (proc.stdout + proc.stderr)

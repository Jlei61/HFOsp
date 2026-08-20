import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_zm_discovery_boundary import load_audit_config
from src.topic4_fig5_ictal_bridge import (
    ELIGIBLE, NOT_ELIGIBLE, NOT_EVALUABLE, NotEvaluableError,
    joint_recruitment_duty, occupancy_rule_is_inert, qualification_sensitivities,
    qualify_model_ictal_v2, require_model_ictal_eligible, sheet_bin_occupancy,
    time_landmarks)

ROOT = Path(__file__).resolve().parents[1]
ZM = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition"
DT = 0.1
ONSET = 5000.0
DURATION = 7000.0


@pytest.fixture(scope="module")
def config():
    return load_audit_config()


def _dense_positions(n=32000, l_mm=20.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, l_mm, size=(n, 2))


def _occupancy(bin_mm=1.0):
    return sheet_bin_occupancy(_dense_positions(), bin_mm=bin_mm, sheet_l_mm=20.0)


def _provenance(bin_mm=1.0, minimum=1.0):
    return {"bin_mm": bin_mm, "recruited_bin_fraction": 0.5,
            "minimum_bin_occupancy_applied": minimum}


def _recruitment(duty, *, sheet_duty=None, stride=5.0, window=20.0,
                 duration=DURATION, early=(5000.0, 6000.0), seed=1):
    """Traces whose W_early duty is exactly ``duty`` by construction."""
    time = np.arange(window, duration + stride, stride)
    f_e = np.full(time.shape, 0.05)
    f_sheet = np.full(time.shape, 0.05)
    inside = (time - window >= early[0]) & (time <= early[1])
    index = np.flatnonzero(inside)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(index))
    n_high = int(round(duty * len(index)))
    f_e[index[order[:n_high]]] = 0.9
    sheet_high = n_high if sheet_duty is None else int(round(sheet_duty * len(index)))
    f_sheet[index[order[:sheet_high]]] = 0.9
    return time, f_e, f_sheet


def _rate(base_hz=30.0, early_hz=300.0, duration=DURATION, dt=DT,
          early=(5000.0, 6000.0), freq=(5400.0, 5900.0)):
    time = np.arange(0.0, duration, dt)
    values = np.full(time.shape, base_hz)
    values[(time >= early[0]) & (time < early[1])] = early_hz
    values[(time >= freq[0]) & (time < freq[1])] = early_hz
    return values


def _contact(base_f=10.0, early_f=60.0, duration=DURATION, dt=DT, n=15,
             base=(500.0, 1000.0), freq=(5400.0, 5900.0), seed=3):
    time = np.arange(0.0, duration, dt)
    rng = np.random.default_rng(seed)
    trace = 0.01 * rng.standard_normal((len(time), n))
    in_base = (time >= base[0]) & (time < base[1])
    in_freq = (time >= freq[0]) & (time < freq[1])
    trace[in_base] += np.sin(2 * np.pi * base_f * time[in_base] / 1000.0)[:, None]
    trace[in_freq] += np.sin(2 * np.pi * early_f * time[in_freq] / 1000.0)[:, None]
    return trace


def _qualify(config, *, duty=0.87, sheet_duty=None, base_f=10.0, early_f=60.0,
             base_hz=30.0, early_hz=300.0, duration=DURATION,
             provenance=None, occupancy=None):
    time, f_e, f_sheet = _recruitment(duty, sheet_duty=sheet_duty,
                                      duration=duration)
    return qualify_model_ictal_v2(
        operational_onset_ms=ONSET, recruitment_time_ms=time, f_e=f_e,
        f_sheet=f_sheet, f_sheet_provenance=provenance or _provenance(),
        occupancy_audit=occupancy or _occupancy(),
        rate_hz=_rate(base_hz, early_hz, duration=duration), rate_dt_ms=DT,
        contact_trace=_contact(base_f, early_f, duration=duration),
        contact_dt_ms=DT, config=config)


def test_time_landmarks_follow_the_spec_offsets(config):
    marks = time_landmarks(5000.0, config)
    assert marks["t_ictal_ms"] == 4900.0
    assert marks["w_pre_ms"] == [4400.0, 4900.0]
    assert marks["w_early_ms"] == [5000.0, 6000.0]
    assert marks["w_freq_ms"] == [5400.0, 5900.0]
    assert marks["t_base_ms"] == [500.0, 1000.0]


def test_sustained_broad_state_passes_primary_morphology(config):
    verdict = _qualify(config, duty=0.87)
    assert verdict["status"] == ELIGIBLE, verdict.get("failing_clauses")
    assert verdict["recruitment"]["joint_duty"] == pytest.approx(0.87, abs=0.01)
    assert require_model_ictal_eligible(verdict) is verdict


def test_short_returned_bursts_fail_the_duty_clause(config):
    verdict = _qualify(config, duty=0.15)
    assert verdict["status"] == NOT_ELIGIBLE
    assert "joint_broad_recruitment_duty" in verdict["failing_clauses"]


def test_local_activity_without_sheet_recruitment_fails_the_joint_clause(config):
    """F_E alone clears 0.5 in 95% of windows; the sheet does not."""
    verdict = _qualify(config, duty=0.95, sheet_duty=0.20)
    assert verdict["recruitment"]["f_e_duty"] == pytest.approx(0.95, abs=0.01)
    assert verdict["recruitment"]["joint_duty"] == pytest.approx(0.20, abs=0.01)
    assert verdict["status"] == NOT_ELIGIBLE
    assert verdict["failing_clauses"] == ["joint_broad_recruitment_duty"]


def test_broad_high_rate_without_frequency_rise_fails_but_stays_visible(config):
    verdict = _qualify(config, duty=0.95, base_f=60.0, early_f=60.0)
    assert verdict["status"] == NOT_ELIGIBLE
    assert verdict["failing_clauses"] == ["contact_frequency_increased"]
    assert verdict["clauses"]["joint_broad_recruitment_duty"] is True
    assert verdict["clauses"]["population_rate_ratio"] is True
    diagnostics = verdict["contact_frequency"]
    assert np.isfinite(diagnostics["median_centroid_base_hz"])
    assert np.isfinite(diagnostics["median_centroid_early_hz"])
    assert diagnostics["primary_ratio"] < 1.25


def test_rate_reference_is_t_base_not_the_pre_onset_window(config):
    verdict = _qualify(config, duty=0.95, base_hz=30.0, early_hz=300.0)
    rate = verdict["population_rate"]
    assert rate["median_rate_base_hz"] == pytest.approx(30.0)
    assert rate["ratio_early_over_base"] == pytest.approx(10.0, rel=1e-6)
    assert verdict["clauses"]["population_rate_ratio"] is True


def test_sparse_bins_cannot_inflate_f_sheet(config):
    """A trace built on possibly-sparse bins is refused, never passed."""
    sparse = _occupancy(bin_mm=1.0)
    sparse = dict(sparse, minimum_occupancy=5.0)
    verdict = _qualify(config, duty=0.95, occupancy=sparse,
                       provenance=_provenance(minimum=1.0))
    assert verdict["status"] == NOT_EVALUABLE
    assert verdict["sheet_rule"]["occupancy_admissible"] is False
    with pytest.raises(NotEvaluableError):
        require_model_ictal_eligible(verdict)


def test_occupancy_rule_inert_when_every_bin_is_dense(config):
    audit = _occupancy(bin_mm=1.0)
    assert audit["minimum_occupancy"] >= 20.0
    assert occupancy_rule_is_inert(audit, 20) is True
    assert occupancy_rule_is_inert({"minimum_occupancy": 7.0}, 20) is False


def test_incomplete_post_window_is_not_evaluable(config):
    verdict = _qualify(config, duty=0.95, duration=5500.0)
    assert verdict["status"] == NOT_EVALUABLE
    assert verdict["eligible"] is None
    assert verdict["missing_evidence"]
    with pytest.raises(NotEvaluableError):
        require_model_ictal_eligible(verdict)


def test_detector_never_reached_is_not_eligible(config):
    verdict = qualify_model_ictal_v2(
        operational_onset_ms=None, recruitment_time_ms=np.zeros(0),
        f_e=np.zeros(0), f_sheet=np.zeros(0),
        f_sheet_provenance=_provenance(), occupancy_audit=_occupancy(),
        rate_hz=np.zeros(0), rate_dt_ms=DT, contact_trace=np.zeros((0, 15)),
        contact_dt_ms=DT, config=config)
    assert verdict["status"] == NOT_ELIGIBLE
    assert verdict["clauses"]["operational_detector_reached"] is False


def test_joint_duty_uses_fully_contained_windows_only():
    """Stamps are window END times, so a 20 ms window ending at 5010 starts at
    4990 and lies partly before W_early; only fully contained windows count."""
    time = np.array([5010.0, 5020.0, 5990.0, 6010.0])
    f_e = np.full(time.shape, 0.9)
    row = joint_recruitment_duty(f_e, f_e, time, (5000.0, 6000.0),
                                 activity_threshold=0.5)
    assert row["n_windows"] == 2


def test_sensitivities_never_replace_the_primary_verdict(config):
    time, f_e, f_sheet = _recruitment(0.75)
    kwargs = dict(operational_onset_ms=ONSET, recruitment_time_ms=time, f_e=f_e,
                  f_sheet=f_sheet, f_sheet_provenance=_provenance(),
                  occupancy_audit=_occupancy(), rate_hz=_rate(), rate_dt_ms=DT,
                  contact_trace=_contact(), contact_dt_ms=DT, config=config)
    verdict = qualify_model_ictal_v2(**kwargs)
    sensitivity = qualification_sensitivities(**kwargs)
    assert verdict["status"] == NOT_ELIGIBLE
    assert sensitivity["activity_and_duty"]["0.5"]["passes_duty"]["0.7"] is True
    assert sensitivity["activity_and_duty"]["0.5"]["passes_duty"]["0.8"] is False
    assert set(sensitivity["onset_shift"]) == {"-100ms", "+0ms", "+100ms"}
    assert set(sensitivity["bin_and_occupancy"]) == {"0.5mm", "1mm", "2mm"}
    assert sensitivity["bin_and_occupancy"]["1mm"]["f_sheet_recomputable"] is True
    assert sensitivity["bin_and_occupancy"]["0.5mm"]["status"] == NOT_EVALUABLE


@pytest.mark.skipif(not (ZM / "zm_joint_morphology_calibration_v4_etoi_refine"
                         / "ith080_ei005.npz").exists(),
                    reason="calibration artifacts absent in this checkout")
@pytest.mark.parametrize("tag,directory", [
    ("ith080_ei002", "zm_joint_morphology_calibration_v5_etoi_boundary"),
    ("ith080_ei005", "zm_joint_morphology_calibration_v4_etoi_refine"),
])
def test_real_fixtures_are_classified_mechanically_and_raw_metrics_survive(
        config, tag, directory):
    payload = json.loads((ZM / directory / f"{tag}.json").read_text())
    with np.load(ZM / directory / f"{tag}.npz", allow_pickle=False) as handle:
        time = np.asarray(handle["full_field_time_ms"], float)
        f_e = np.asarray(handle["active_neuron_fraction_20ms"], float)
        f_sheet = np.asarray(handle["recruited_spatial_fraction_1mm"], float)
        rate = np.asarray(handle["rate_E_hz"], float)
        lfp = np.asarray(handle["lfp_trace"], float)
        dt = float(handle["lfp_dt_ms"])

    stored = payload["runaway_morphology"]["full_field_recruitment"]
    onset = float(payload["runaway_morphology"]["onset_ms"])
    post = (time >= onset) & (time < onset + 1500.0)
    assert float(np.mean(f_e[post] >= 0.5)) == pytest.approx(
        stored["fraction_windows_majority_E_active"], abs=1e-12)
    # the traces are stored as float32, so the float64 summaries in the JSON are
    # reproduced to float32 precision, not bit-for-bit
    assert float(np.median(f_sheet[post])) == pytest.approx(
        stored["median_recruited_spatial_fraction_1mm"], rel=1e-6)
    assert float(np.median(f_e[post])) == pytest.approx(
        stored["median_active_neuron_fraction_20ms"], rel=1e-6)
    assert float(np.mean(f_sheet[post] >= 0.5)) == pytest.approx(
        stored["fraction_windows_majority_sheet_recruited"], abs=1e-12)

    verdict = qualify_model_ictal_v2(
        operational_onset_ms=payload["operational_onset_ms"],
        recruitment_time_ms=time, f_e=f_e, f_sheet=f_sheet,
        f_sheet_provenance={"bin_mm": 1.0, "recruited_bin_fraction": 0.5,
                            "minimum_bin_occupancy_applied": 1.0},
        occupancy_audit={"bin_mm": 1.0, "minimum_occupancy": 53.0,
                         "n_bins": 400, "n_occupied": 400,
                         "median_occupancy": 80.0},
        rate_hz=rate, rate_dt_ms=dt, contact_trace=lfp, contact_dt_ms=dt,
        config=config)
    assert verdict["status"] in (ELIGIBLE, NOT_ELIGIBLE)
    assert verdict["landmarks"]["t_op_ms"] == payload["operational_onset_ms"]
    assert 0.0 <= verdict["recruitment"]["joint_duty"] <= 1.0


def test_require_model_ictal_eligible_refuses_every_non_eligible_status():
    for status in (NOT_ELIGIBLE, NOT_EVALUABLE):
        with pytest.raises(NotEvaluableError):
            require_model_ictal_eligible({"status": status, "eligible": False})
    with pytest.raises(NotEvaluableError):
        require_model_ictal_eligible({"status": ELIGIBLE, "eligible": None})

import numpy as np
from types import SimpleNamespace

from src.topic4_rev21_zm_transition import (
    ELIGIBLE,
    NOT_ELIGIBLE,
    bin_population_rate,
    build_transition_readout,
    formal_interictal_stop_step,
    qualify_model_ictal_rev21,
)
from src.topic4_runaway_morphology import rolling_full_field_recruitment


def _config(minimum=2000.0):
    return {"model_ictal_v2": {
        "t_ictal_offset_from_operational_onset_ms": -100.0,
        "t_base_ms": [500.0, 1000.0],
        "w_pre_ms_relative_to_t_ictal": [-500.0, 0.0],
        "w_early_ms_relative_to_t_ictal": [100.0, 1100.0],
        "w_freq_ms_relative_to_t_ictal": [500.0, 1000.0],
        "minimum_ictal_onset_ms": minimum,
        "activity_threshold": 0.5, "duty_threshold": 0.8,
        "population_rate_ratio_min": 2.0,
        "contact_centroid_band_hz": [10.0, 250.0],
        "contact_centroid_shift_min_hz": 5.0,
        "contact_centroid_ratio_min": 1.25,
        "sheet_bin_mm": 1.0, "sheet_recruited_bin_fraction": 0.5,
        "sensitivity_bin_mm": [0.5, 1.0, 2.0],
        "sheet_minimum_bin_occupancy": 20,
    }}


def test_rate_binning_replaces_the_sparse_integration_step_median():
    raw = np.zeros(400)
    raw[::20] = 200.0
    binned, dt = bin_population_rate(raw, 0.1, 20.0)
    assert dt == 20.0
    assert np.all(binned == 10.0)


def test_interictal_cutoff_uses_detector_adjusted_onset():
    assert formal_interictal_stop_step(
        5000.0, _config(), dt_ms=0.1, total_steps=60000,
    ) == 49000
    assert formal_interictal_stop_step(
        None, _config(), dt_ms=0.1, total_steps=60000,
    ) == 60000


def _qualified_inputs(operational_onset):
    config = _config()
    duration, dt = 7000.0, 1.0
    time = np.arange(20.0, duration + 5.0, 5.0)
    f_e = np.full(len(time), 0.05)
    f_sheet = np.full(len(time), 0.05)
    early = (time - 20.0 >= operational_onset) & (time <= operational_onset + 1000.0)
    f_e[early] = 0.9
    f_sheet[early] = 0.9
    rate = np.full(int(duration / dt), 20.0)
    rate[int(operational_onset / dt):int((operational_onset + 1000.0) / dt)] = 200.0
    trace = np.zeros((len(rate), 2))
    base_t = np.arange(500, 1000)
    early_t = np.arange(int(operational_onset + 400), int(operational_onset + 900))
    trace[base_t] = np.sin(2 * np.pi * 10 * base_t / 1000.0)[:, None]
    trace[early_t] = np.sin(2 * np.pi * 60 * early_t / 1000.0)[:, None]
    return dict(
        operational_onset_ms=operational_onset,
        recruitment_time_ms=time, f_e=f_e, f_sheet=f_sheet,
        f_sheet_provenance={"bin_mm": 1.0,
                            "recruited_bin_fraction": 0.5,
                            "minimum_bin_occupancy_applied": 20.0},
        occupancy_audit={"minimum_occupancy": 40.0},
        rate_hz=rate, rate_dt_ms=dt, contact_trace=trace,
        contact_dt_ms=dt, config=config,
    )


def test_rev21_requires_a_real_low_activity_dwell_before_transition():
    early = qualify_model_ictal_rev21(**_qualified_inputs(1500.0))
    assert early["status"] == NOT_ELIGIBLE
    assert "transition_after_minimum_dwell" in early["failing_clauses"]


def test_rev21_preserves_an_eligible_v2_state_after_the_dwell():
    verdict = qualify_model_ictal_rev21(**_qualified_inputs(3000.0))
    assert verdict["status"] == ELIGIBLE, verdict.get("failing_clauses")


def test_sheet_recruitment_excludes_sparse_bins_when_requested():
    positions = np.asarray([[0.1, 0.1], [1.1, 0.1], [1.2, 0.1]])
    spikes = np.zeros((20, 3), bool)
    spikes[5, 0] = True
    trace = rolling_full_field_recruitment(
        spikes, positions, dt_ms=1.0, sheet_l_mm=2.0,
        window_ms=10.0, stride_ms=10.0, spatial_bin_mm=1.0,
        recruited_bin_fraction=0.5, minimum_bin_occupancy=2,
    )
    assert trace["eligible_spatial_bins"] == 1
    assert trace["recruited_spatial_fraction"][0] == 0.0


def test_one_active_worker_artifact_contains_all_evidence_families():
    x, y = np.meshgrid(np.linspace(0.01, 1.99, 40),
                       np.linspace(0.01, 1.99, 20))
    substrate = SimpleNamespace(
        positions_e=np.column_stack([x.ravel(), y.ravel()]),
        engine={"dt": 1.0, "L": 2.0},
    )

    class Slow:
        @staticmethod
        def trace_arrays():
            return {"time_ms": np.arange(10), "z_mean": np.ones(10)}

        @staticmethod
        def weighted_trace_arrays():
            return {"time_ms": np.arange(10), "z_weighted_mean": np.ones(10)}

        @staticmethod
        def field_frames():
            return None

        @staticmethod
        def summary():
            return {"trace_samples": 10}

    result = {
        "E_spk_bool": np.zeros((100, len(substrate.positions_e)), bool),
        "rate_E": np.zeros(100),
        "lfp_trace": np.zeros((100, 15)),
        "runaway_early_stop_ms": None,
    }
    payload, arrays = build_transition_readout(
        result, substrate, Slow(), _config(),
    )
    assert payload["model_ictal_rev21"]["status"] == NOT_ELIGIBLE
    assert "transition_lfp_trace" in arrays
    assert "transition_sheet_fraction_1mm" in arrays
    assert "transition_sheet_fraction_0p5mm" in arrays
    assert "transition_spatial_spike_count_20ms" in arrays
    assert "slow_z_mean" in arrays
    assert "slow_weighted_z_weighted_mean" in arrays

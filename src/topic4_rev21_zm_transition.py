"""Model-internal readout contract for the rev21 dual-core Z/M screen."""
from __future__ import annotations

import copy

import numpy as np

from src.topic4_fig5_ictal_bridge import (
    ELIGIBLE as V2_ELIGIBLE,
    NOT_ELIGIBLE as V2_NOT_ELIGIBLE,
    NOT_EVALUABLE as V2_NOT_EVALUABLE,
    joint_recruitment_duty,
    qualify_model_ictal_v2,
    sheet_bin_occupancy,
)
from src.topic4_runaway_morphology import rolling_full_field_recruitment


ELIGIBLE = "MODEL_ICTAL_ELIGIBLE_REV21"
NOT_ELIGIBLE = "MODEL_ICTAL_NOT_ELIGIBLE_REV21"
NOT_EVALUABLE = "MODEL_ICTAL_NOT_EVALUABLE_REV21"


def bin_population_rate(rate_hz, dt_ms: float, bin_ms: float = 20.0):
    """Average the integration-step rate into fixed, non-overlapping bins."""
    values = np.asarray(rate_hz, float)
    dt_ms, bin_ms = float(dt_ms), float(bin_ms)
    if values.ndim != 1 or dt_ms <= 0.0 or bin_ms <= 0.0:
        raise ValueError("rate must be one-dimensional and time steps positive")
    steps = int(round(bin_ms / dt_ms))
    if steps < 1 or not np.isclose(steps * dt_ms, bin_ms, atol=1e-9):
        raise ValueError("bin_ms must be an integer multiple of dt_ms")
    count = len(values) // steps
    if count == 0:
        return np.zeros(0, dtype=float), bin_ms
    return values[:count * steps].reshape(count, steps).mean(axis=1), bin_ms


def scientific_onset_ms(operational_onset_ms, config):
    if operational_onset_ms is None:
        return None
    return (float(operational_onset_ms)
            + float(config["model_ictal_v2"][
                "t_ictal_offset_from_operational_onset_ms"]))


def formal_interictal_stop_step(operational_onset_ms, config, *, dt_ms,
                                total_steps):
    """End the event observation before the candidate ictal state begins."""
    onset = scientific_onset_ms(operational_onset_ms, config)
    if onset is None:
        return int(total_steps)
    stop = int(np.floor(max(0.0, onset) / float(dt_ms) + 1e-12))
    return min(int(total_steps), stop)


def qualify_model_ictal_rev21(**kwargs):
    """Apply V2 morphology plus rev21's explicit minimum-onset clause."""
    config = kwargs["config"]
    verdict = copy.deepcopy(qualify_model_ictal_v2(**kwargs))
    onset = scientific_onset_ms(kwargs["operational_onset_ms"], config)
    if onset is not None:
        onset += float(kwargs.get("onset_shift_ms", 0.0))
    minimum = float(config["model_ictal_v2"]["minimum_ictal_onset_ms"])
    onset_ok = onset is not None and onset >= minimum
    clauses = verdict.setdefault("clauses", {})
    clauses["transition_after_minimum_dwell"] = bool(onset_ok)
    verdict["scientific_onset_ms"] = onset
    verdict.setdefault("thresholds", {})["minimum_ictal_onset_ms"] = minimum

    failing = [name for name, value in clauses.items() if value is False]
    unresolved = [name for name, value in clauses.items() if value is None]
    verdict["failing_clauses"] = failing
    verdict["unresolved_clauses"] = unresolved
    if failing:
        verdict["status"] = NOT_ELIGIBLE
        verdict["eligible"] = False
    elif unresolved or verdict.get("status") == V2_NOT_EVALUABLE:
        verdict["status"] = NOT_EVALUABLE
        verdict["eligible"] = None
    elif verdict.get("status") == V2_ELIGIBLE:
        verdict["status"] = ELIGIBLE
        verdict["eligible"] = True
    elif verdict.get("status") == V2_NOT_ELIGIBLE:
        verdict["status"] = NOT_ELIGIBLE
        verdict["eligible"] = False
    else:
        raise RuntimeError("unexpected V2 model-ictal status")
    verdict["schema_id"] = "model_ictal_eligibility_rev21_v1"
    return verdict


def build_transition_readout(result, substrate, slow, config):
    """Produce all model-state evidence from one uninterrupted active-Z/M run."""
    if slow is None:
        raise ValueError("active transition readout requires a Z/M slow layer")
    contact_trace = result.get("lfp_trace")
    if contact_trace is None:
        raise ValueError("active transition readout requires the 15-contact trace")
    spikes = np.asarray(result["E_spk_bool"], bool)
    dt_ms = float(substrate.engine["dt"])
    spec = config["model_ictal_v2"]
    minimum = int(spec["sheet_minimum_bin_occupancy"])
    bins = [float(v) for v in spec.get("sensitivity_bin_mm", [0.5, 1.0, 2.0])]
    primary_bin = float(spec["sheet_bin_mm"])
    if primary_bin not in bins:
        bins.append(primary_bin)
    recruitments = {
        value: rolling_full_field_recruitment(
            spikes, substrate.positions_e, dt_ms=dt_ms,
            sheet_l_mm=float(substrate.engine["L"]),
            spatial_bin_mm=value,
            recruited_bin_fraction=float(spec["sheet_recruited_bin_fraction"]),
            minimum_bin_occupancy=minimum,
        )
        for value in sorted(set(bins))
    }
    primary = recruitments[primary_bin]
    occupancy = sheet_bin_occupancy(
        substrate.positions_e, bin_mm=primary_bin,
        sheet_l_mm=float(substrate.engine["L"]),
    )
    rate_20ms, rate_dt_ms = bin_population_rate(
        result["rate_E"], dt_ms, bin_ms=20.0,
    )
    common = dict(
        operational_onset_ms=result.get("runaway_early_stop_ms"),
        recruitment_time_ms=primary["time_ms"],
        f_e=primary["active_neuron_fraction"],
        f_sheet=primary["recruited_spatial_fraction"],
        f_sheet_provenance={
            "bin_mm": primary_bin,
            "recruited_bin_fraction": float(
                spec["sheet_recruited_bin_fraction"]),
            "minimum_bin_occupancy_applied": float(minimum),
        },
        occupancy_audit=occupancy,
        rate_hz=rate_20ms, rate_dt_ms=rate_dt_ms,
        contact_trace=np.asarray(contact_trace, float),
        contact_dt_ms=dt_ms, config=config,
    )
    verdict = qualify_model_ictal_rev21(**common)
    if result.get("runaway_early_stop_ms") is None:
        sensitivity = {
            "status": NOT_ELIGIBLE,
            "reason": "operational detector was not reached",
        }
    else:
        sensitivity = {"activity_and_duty": {}}
        landmarks = verdict["landmarks"]
        for activity in (0.4, 0.5, 0.6):
            duty = joint_recruitment_duty(
                primary["active_neuron_fraction"],
                primary["recruited_spatial_fraction"], primary["time_ms"],
                landmarks["w_early_ms"], activity_threshold=activity,
            )
            duty["passes_duty"] = {
                f"{threshold:g}": bool(
                    duty["joint_duty"] >= threshold
                )
                for threshold in (0.7, 0.8, 0.9)
            }
            sensitivity["activity_and_duty"][f"{activity:g}"] = duty
        bin_duty = {}
        for value, trace in recruitments.items():
            try:
                duty = joint_recruitment_duty(
                    trace["active_neuron_fraction"],
                    trace["recruited_spatial_fraction"], trace["time_ms"],
                    landmarks["w_early_ms"],
                    activity_threshold=float(spec["activity_threshold"]),
                )
                duty["passes_primary_duty"] = bool(
                    duty["joint_duty"] >= float(spec["duty_threshold"])
                )
                bin_duty[f"{value:g}mm"] = duty
            except Exception as error:
                bin_duty[f"{value:g}mm"] = {
                    "status": NOT_EVALUABLE, "reason": str(error),
                }
        sensitivity["bin_size_recruitment_duty"] = bin_duty
        sensitivity["onset_shift_rev21"] = {
            f"{shift:+g}ms": qualify_model_ictal_rev21(
                **common, onset_shift_ms=float(shift),
            )["status"]
            for shift in (-100.0, 0.0, 100.0)
        }

    arrays = {
        "transition_rate_E_hz_raw": np.asarray(result["rate_E"], np.float32),
        "transition_rate_E_hz_20ms": np.asarray(rate_20ms, np.float32),
        "transition_rate_E_20ms_dt_ms": np.asarray(rate_dt_ms, float),
        "transition_lfp_trace": np.asarray(contact_trace, np.float32),
        "transition_lfp_dt_ms": np.asarray(dt_ms, float),
    }
    frame_ms = 20.0
    frame_steps = max(1, int(round(frame_ms / dt_ms)))
    grid_n = max(1, int(round(float(substrate.engine["L"]) / 0.5)))
    positions = np.asarray(substrate.positions_e, float)
    ix = np.clip(
        (positions[:, 0] / float(substrate.engine["L"]) * grid_n).astype(int),
        0, grid_n - 1,
    )
    iy = np.clip(
        (positions[:, 1] / float(substrate.engine["L"]) * grid_n).astype(int),
        0, grid_n - 1,
    )
    flat = iy * grid_n + ix
    n_frames = len(spikes) // frame_steps
    spatial = np.empty((n_frames, grid_n, grid_n), np.float32)
    for frame in range(n_frames):
        counts = np.sum(
            spikes[frame * frame_steps:(frame + 1) * frame_steps], axis=0,
        )
        spatial[frame] = np.bincount(
            flat, weights=counts, minlength=grid_n * grid_n,
        ).reshape(grid_n, grid_n)
    arrays.update({
        "transition_spatial_frame_time_ms": (
            np.arange(n_frames, dtype=np.float32) + 0.5
        ) * frame_ms,
        "transition_spatial_spike_count_20ms": spatial,
        "transition_spatial_bin_mm": np.asarray(0.5, float),
    })
    for value, trace in recruitments.items():
        token = f"{value:g}".replace(".", "p")
        arrays[f"transition_recruitment_time_{token}mm_ms"] = np.asarray(
            trace["time_ms"], np.float32,
        )
        arrays[f"transition_active_E_fraction_{token}mm"] = np.asarray(
            trace["active_neuron_fraction"], np.float32,
        )
        arrays[f"transition_sheet_fraction_{token}mm"] = np.asarray(
            trace["recruited_spatial_fraction"], np.float32,
        )
    for name, values in slow.trace_arrays().items():
        arrays[f"slow_{name}"] = np.asarray(values, np.float32)
    weighted = slow.weighted_trace_arrays()
    if weighted is not None:
        for name, values in weighted.items():
            arrays[f"slow_weighted_{name}"] = np.asarray(values, np.float32)
    fields = slow.field_frames()
    if fields is not None:
        arrays["slow_field_net_current"] = np.asarray(
            fields["net_slow_current"], np.float32,
        )
        arrays["slow_field_time_ms"] = np.asarray(
            fields["call_index"], np.float64,
        ) * dt_ms
    payload = {
        "model_ictal_rev21": verdict,
        "model_ictal_sensitivity": sensitivity,
        "slow_state_summary": slow.summary(),
        "recruitment_geometry": {
            f"{value:g}mm": {
                key: trace[key] for key in (
                    "minimum_bin_occupancy", "eligible_spatial_bins",
                    "minimum_eligible_bin_occupancy",
                    "recruited_bin_fraction",
                )
            }
            for value, trace in recruitments.items()
        },
    }
    return payload, arrays

#!/usr/bin/env python3
"""Freeze an early-runaway snapshot by patient TA/TB field similarity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_fig5 import (  # noqa: E402
    select_positive_identity_candidate,
    sustained_fraction_around,
)
from src.topic5_template_axis_field import (  # noqa: E402
    align_activation_to_interictal_field,
    score_field,
    scorers_from_interictal_record,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _record_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sample_grid_at_contacts(grid: np.ndarray, contacts: np.ndarray,
                             sheet_size_mm: float) -> np.ndarray:
    n = int(grid.shape[0])
    if grid.shape != (n, n):
        raise ValueError("activity energy must be a square sheet grid")
    indices = np.clip(
        np.floor(np.asarray(contacts, float) / float(sheet_size_mm) * n).astype(int),
        0, n - 1,
    )
    return grid[indices[:, 0], indices[:, 1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--field-record", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-after-onset-ms", type=float, default=300.0)
    parser.add_argument("--sustained-window-ms", type=float, default=100.0)
    parser.add_argument("--sustained-fraction-min", type=float, default=0.95)
    parser.add_argument("--global-recruitment-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.15)
    parser.add_argument("--sheet-size-mm", type=float, default=20.0)
    args = parser.parse_args()

    replay_path = Path(args.replay).resolve()
    replay = _load_npz(replay_path)
    replay_meta = json.loads(replay_path.with_suffix(".json").read_text())
    field_path = Path(args.field_record).resolve()
    field_record = json.loads(field_path.read_text())
    scorers = scorers_from_interictal_record(field_record)
    if not {"shared_a", "shared_b"}.issubset(scorers):
        raise RuntimeError("frozen field record has no shared TA/TB scorers")

    frame_time = np.asarray(replay["frame_time_ms"], float)
    spike_counts = np.asarray(replay["activity_spike_counts"], float)
    occupancy = np.asarray(replay["activity_cell_occupancy"], float)
    contacts = np.asarray(replay["contact_xy_mm"], float)
    contact_names = np.asarray(replay["contact_names"]).astype(str)
    full_time = np.asarray(replay["full_field_time_ms"], float)
    active_fraction = np.asarray(replay["active_neuron_fraction_20ms"], float)
    spatial_fraction = np.asarray(replay["recruited_spatial_fraction_1mm"], float)
    activity_window_ms = float(replay_meta["activity_window_ms"])
    onset_ms = float(replay_meta["morphology_onset_ms"])
    search_stop_ms = onset_ms + float(args.search_after_onset_ms)

    rows: list[dict[str, object]] = []
    grids: dict[float, np.ndarray] = {}
    contact_values: dict[float, np.ndarray] = {}
    for index, time_ms in enumerate(frame_time):
        time_ms = float(time_ms)
        if time_ms < onset_ms or time_ms > search_stop_ms + 1e-9:
            continue
        sustained = sustained_fraction_around(
            full_time, active_fraction, spatial_fraction,
            center_ms=time_ms, window_ms=float(args.sustained_window_ms),
            threshold=float(args.global_recruitment_threshold),
        )
        if (not np.isfinite(sustained)
                or sustained < float(args.sustained_fraction_min)):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            local_rate_hz = (
                spike_counts[index] / occupancy / (activity_window_ms * 1e-3)
            )
        energy = np.square(np.nan_to_num(local_rate_hz)) / 1e3
        smooth = gaussian_filter(
            energy, sigma=float(args.smoothing_sigma_bins))
        sampled = _sample_grid_at_contacts(
            smooth, contacts, float(args.sheet_size_mm))
        aligned = align_activation_to_interictal_field(
            field_record, contact_names, sampled)
        if int(aligned["n_finite"]) != int(aligned["n_target"]):
            raise RuntimeError(
                "model snapshot does not cover every frozen patient-field contact")
        ta = score_field(scorers["shared_a"], aligned["values"])
        tb = score_field(scorers["shared_b"], aligned["values"])
        nearest = int(np.argmin(np.abs(full_time - time_ms)))
        row = {
            "time_ms": time_ms,
            "sustained_global_fraction": float(sustained),
            "active_E_fraction": float(active_fraction[nearest]),
            "recruited_sheet_fraction": float(spatial_fraction[nearest]),
            "ta_identity_r": float(ta["r_identity"]),
            "ta_mirror_r": float(ta["r_mirror"]),
            "tb_identity_r": float(tb["r_identity"]),
            "tb_mirror_r": float(tb["r_mirror"]),
        }
        rows.append(row)
        grids[time_ms] = np.asarray(smooth, np.float32)
        contact_values[time_ms] = np.asarray(sampled, np.float32)

    selected = select_positive_identity_candidate(rows)
    selected_time = float(selected["time_ms"])
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        selected_time_ms=np.asarray(selected_time, np.float64),
        selected_energy_grid=grids[selected_time],
        selected_contact_energy=contact_values[selected_time],
        contact_xy_mm=np.asarray(contacts, np.float32),
        contact_names=np.asarray(contact_names),
        candidate_time_ms=np.asarray([float(row["time_ms"]) for row in rows]),
        candidate_ta_identity_r=np.asarray(
            [float(row["ta_identity_r"]) for row in rows], np.float32),
        candidate_tb_identity_r=np.asarray(
            [float(row["tb_identity_r"]) for row in rows], np.float32),
        candidate_sustained_fraction=np.asarray(
            [float(row["sustained_global_fraction"]) for row in rows], np.float32),
    )
    summary = {
        "status": "ZM_FIG5_EARLY_RUNAWAY_MODE_SNAPSHOT_FROZEN",
        "replay": _record_path(replay_path),
        "field_record": _record_path(field_path),
        "field_fingerprint_sha256": field_record["interictal_field"][
            "fingerprint_sha256"],
        "seed": int(replay_meta["seed"]),
        "morphology_onset_ms": onset_ms,
        "search_interval_ms": [onset_ms, search_stop_ms],
        "candidate_contract": {
            "activity_snapshot": (
                f"one causal {activity_window_ms:g}-ms local E-spike-count frame"),
            "sustained_window_ms": float(args.sustained_window_ms),
            "minimum_joint_sustained_fraction": float(
                args.sustained_fraction_min),
            "active_E_fraction_threshold": float(
                args.global_recruitment_threshold),
            "recruited_sheet_fraction_threshold": float(
                args.global_recruitment_threshold),
        },
        "selection_contract": (
            "maximize max(positive shared-TA identity r, positive shared-TB "
            "identity r); mirror and absolute correlations are forbidden; ties "
            "choose the earliest candidate"),
        "n_eligible_candidates": len(rows),
        "selected": selected,
        "spatial_energy": {
            "measure": "square of local E-neuron firing rate, in 1e3 Hz^2",
            "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
            "sheet_size_mm": float(args.sheet_size_mm),
        },
        "npz": _record_path(out),
        "candidate_rows": rows,
        "claim_boundary": (
            "single-trajectory early-runaway snapshot selected in a frozen "
            "patient TA/TB contact-field space; descriptive continuity, not "
            "patient ictal-waveform reproduction"),
    }
    atomic_write_json(summary, str(out.with_suffix(".json")))
    print(json.dumps({"out": str(out), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()

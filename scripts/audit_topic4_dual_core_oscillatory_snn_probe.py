#!/usr/bin/env python3
"""Audit spatial coverage of the fast-GABA full-SNN contact oscillation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_dual_core_oscillation_phase import contact_oscillation_assay  # noqa: E402


DEFAULT_PROBE = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/oscillatory_snn_probe/"
    "rev21_si_0p7_sm_0p5_t2542_d2641_tauGABA8.json")
DEFAULT_BASELINE = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_zm_transition/coarse/workers/"
    "rev21_si_0p7_sm_0p5_topology_2542_dynamics_2641.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _windows(time_ms: np.ndarray, onset_ms: float) -> dict[str, np.ndarray]:
    baseline = (time_ms >= 500.0) & (time_ms < 1000.0)
    early = ((time_ms >= onset_ms + 100.0)
             & (time_ms < min(float(time_ms[-1]), onset_ms + 1100.0)))
    terminal = time_ms >= float(time_ms[-1]) - 1000.0
    if any(np.sum(window) < 64 for window in (baseline, early, terminal)):
        raise RuntimeError("baseline, early or terminal contact window is too short")
    return {"baseline": baseline, "early_recruitment": early,
            "terminal_high_state": terminal}


def _decorate(assay: dict, names: np.ndarray, shafts: np.ndarray,
              xy: np.ndarray) -> dict:
    passing = np.asarray(assay["persistent_contact_mask"], bool)
    records = []
    for index, name in enumerate(names):
        records.append({
            "contact": str(name),
            "shaft": str(shafts[index]),
            "xy_mm": np.asarray(xy[index], float).tolist(),
            "dominant_frequency_hz": assay["dominant_frequency_hz"][index],
            "target_band_rms_ratio": assay["target_band_rms_ratio"][index],
            "plv_to_max_rms_contact": assay["plv_to_max_rms_contact"][index],
            "whole_window_passes_frequency_and_rms": bool(
                assay["passing_contact_mask"][index]),
            "passing_windows": int(assay["passing_windows_per_contact"][index]),
            "fine_passing_windows": int(
                assay["fine_passing_windows_per_contact"][index]),
            "persistent_frequency_and_rms": bool(passing[index]),
        })
    shaft_summary = {}
    for shaft in np.unique(shafts):
        selected = shafts == shaft
        shaft_summary[str(shaft)] = {
            "n_contacts": int(np.sum(selected)),
            "n_passing": int(np.sum(passing[selected])),
            "passing_fraction": float(np.mean(passing[selected])),
        }
    overall_fraction = float(np.mean(passing))
    every_shaft_majority = bool(all(
        record["passing_fraction"] >= 0.5
        for record in shaft_summary.values()))
    assay = dict(assay)
    assay.update({
        "contacts": records,
        "shaft_summary": shaft_summary,
        "spatial_coverage_gate": {
            "overall_passing_fraction": overall_fraction,
            "minimum_overall_fraction": 0.8,
            "every_shaft_at_least_half": every_shaft_majority,
            "passes": bool(overall_fraction >= 0.8 and every_shaft_majority),
            "role": (
                "development-only operationalization of the author's global "
                "and sustained high-frequency Figure 5A target; not preregistered"),
        },
    })
    return assay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    probe_json = args.probe.resolve()
    baseline_json = args.baseline.resolve()
    probe_meta = json.loads(probe_json.read_text(encoding="utf-8"))
    baseline_meta = json.loads(baseline_json.read_text(encoding="utf-8"))
    probe_npz = Path(probe_meta["arrays"]["path"])
    baseline_npz = baseline_json.with_suffix(".npz")
    probe = np.load(probe_npz, allow_pickle=True)
    baseline = np.load(baseline_npz, allow_pickle=True)

    names = np.asarray(
        probe["contact_names"] if "contact_names" in probe.files
        else baseline["contact_names"]).astype(str)
    shafts = np.asarray(
        probe["contact_shaft_ids"] if "contact_shaft_ids" in probe.files
        else baseline["shaft_ids"]).astype(str)
    xy = np.asarray(probe["contact_xy_mm"], float)
    if names.shape != shafts.shape or names.size != xy.shape[0]:
        raise RuntimeError("contact identity arrays do not align")
    if not np.allclose(xy, np.asarray(baseline["contact_xy_mm"], float)):
        raise RuntimeError("probe and baseline contact geometry differ")

    probe_time = np.asarray(probe["time_ms"], float)
    probe_lfp = np.asarray(probe["virtual_seeg"], float)
    probe_windows = _windows(
        probe_time, float(probe_meta["operational_runaway_onset_ms"]))
    probe_assays = {
        role: _decorate(contact_oscillation_assay(
            probe_lfp[probe_windows["baseline"]], probe_lfp[window],
            dt_ms=float(np.median(np.diff(probe_time)))), names, shafts, xy)
        for role, window in probe_windows.items() if role != "baseline"
    }

    baseline_lfp = np.asarray(baseline["transition_lfp_trace"], float)
    baseline_dt = float(baseline["transition_lfp_dt_ms"])
    baseline_time = np.arange(baseline_lfp.shape[0], dtype=float) * baseline_dt
    baseline_onset = float(
        baseline_meta["model_ictal_rev21"]["landmarks"]["t_op_ms"])
    baseline_windows = _windows(baseline_time, baseline_onset)
    baseline_assays = {
        role: _decorate(contact_oscillation_assay(
            baseline_lfp[baseline_windows["baseline"]], baseline_lfp[window],
            dt_ms=baseline_dt), names, shafts, xy)
        for role, window in baseline_windows.items() if role != "baseline"
    }

    payload = {
        "status": "FULL_SNN_CONTACT_OSCILLATION_SPATIAL_AUDIT_COMPLETE",
        "probe_tau_d_GABA_ms": float(probe_meta["tau_d_GABA_ms"]),
        "baseline_tau_d_GABA_ms": float(
            probe_meta["baseline_tau_d_GABA_ms"]),
        "probe": probe_assays["terminal_high_state"],
        "baseline": baseline_assays["terminal_high_state"],
        "probe_by_role": probe_assays,
        "baseline_by_role": baseline_assays,
        "windows": {
            "baseline_ms": [500.0, 1000.0],
            "early_recruitment_relative_to_operational_onset_ms": [100.0, 1100.0],
            "terminal_high_state": "last 1000 ms of the recorded trajectory",
        },
        "sources": {
            "probe_json": {"path": str(probe_json), "sha256": _sha256(probe_json)},
            "probe_npz": {"path": str(probe_npz), "sha256": _sha256(probe_npz)},
            "baseline_json": {
                "path": str(baseline_json), "sha256": _sha256(baseline_json)},
            "baseline_npz": {
                "path": str(baseline_npz), "sha256": _sha256(baseline_npz)},
        },
        "claim_boundary": (
            "Single topology and dynamics seed. Early-recruitment and terminal "
            "persistence are reported separately so spatial recruitment delay is "
            "not mislabelled as absence of a sustained high state. This remains "
            "a sensitivity result, not a multi-seed phase or patient mechanism."),
    }
    output = (args.output.resolve() if args.output else
              probe_json.parent / "contact_spatial_oscillation_audit_tauGABA8_vs18.json")
    _atomic_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "probe_early_passing": probe_assays[
            "early_recruitment"]["n_persistent_contacts"],
        "probe_terminal_passing": probe_assays[
            "terminal_high_state"]["n_persistent_contacts"],
        "baseline_terminal_passing": baseline_assays[
            "terminal_high_state"]["n_persistent_contacts"],
        "probe_global_contact_gate": payload["probe"]["spatial_coverage_gate"]["passes"],
    }, indent=2))


if __name__ == "__main__":
    main()

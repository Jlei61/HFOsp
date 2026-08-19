#!/usr/bin/env python3
"""Rebuild the frozen Qi/gK Figure 5 trajectory as a morphology positive control."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig_m3a_v2_1_qigk_runaway_transition_gif import (  # noqa: E402
    ProtocolConfig,
    run_one,
)
from src.topic4_runaway_morphology import (  # noqa: E402
    classify_sustained_runaway,
    contact_oscillation_metrics,
    population_rate_frequency_metrics,
    rolling_full_field_recruitment,
    summarize_runaway_morphology,
)


OUT_DIR = (
    ROOT / "results" / "topic4_sef_hfo" / "data_driven_zm_ictal_transition"
    / "runaway_morphology"
)


def _git(command):
    return subprocess.check_output(
        ["git", *command], cwd=ROOT, text=True).strip()


def main():
    started = time.time()
    config = ProtocolConfig(
        seed=1,
        T=2200.0,
        layout="subject1146",
        k_q=0.10,
        q_min=0.05,
        kick_boost=5.0,
        r_kick=0.6,
        fig_name="fig5_qigk_morphology_reference",
    )
    substrate, result, legacy_metrics = run_one(config, record_gif=True)
    onset_ms = legacy_metrics["runaway_start_ms"]
    if onset_ms is None:
        raise RuntimeError("frozen Qi/gK positive control did not enter runaway")
    dt_ms = float(substrate["p"].dt)
    recruitment = rolling_full_field_recruitment(
        result["E_spk_bool"], substrate["posE"], dt_ms=dt_ms,
        sheet_l_mm=float(substrate["L"]))
    oscillation = contact_oscillation_metrics(
        result["lfp_trace"], dt_ms=dt_ms, onset_ms=float(onset_ms))
    population_frequency = population_rate_frequency_metrics(
        result["rate_E"], dt_ms=dt_ms, onset_ms=float(onset_ms))
    morphology = summarize_runaway_morphology(
        recruitment, oscillation, onset_ms=float(onset_ms),
        population_frequency=population_frequency)
    morphology["classification"] = classify_sustained_runaway(morphology)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "qigk_e1146_reference"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        time_ms=np.asarray(result["times"], np.float32),
        rate_E_hz=np.asarray(result["rate_E"], np.float32),
        lfp_trace=np.asarray(result["lfp_trace"], np.float32),
        lfp_dt_ms=np.asarray(dt_ms, float),
        contact_names=np.asarray(result["names"], dtype="U16"),
        contact_xy_mm=np.asarray(result["contacts"], np.float32),
        full_field_time_ms=np.asarray(recruitment["time_ms"], np.float32),
        active_neuron_fraction_20ms=np.asarray(
            recruitment["active_neuron_fraction"], np.float32),
        recruited_spatial_fraction_1mm=np.asarray(
            recruitment["recruited_spatial_fraction"], np.float32),
    )
    payload = {
        "status": "QIGK_FIG5_MORPHOLOGY_REFERENCE_COMPLETE",
        "role": "positive_control_for_model_state_morphology_instrument",
        "frozen_command_contract": (
            "subject1146, k_q=0.10, q_min=0.05, kick_boost=5.0, "
            "r_kick=0.6, seed=1; the frozen T=1500 ms figure trajectory is "
            "extended deterministically to T=2200 ms only to provide a complete "
            "500 ms post-onset morphology window"),
        "config": asdict(config),
        "legacy_rate_detector": legacy_metrics,
        "runaway_morphology": morphology,
        "provenance": {
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git(["status", "--porcelain"])),
            "producer": str(Path(__file__).resolve().relative_to(ROOT)),
        },
        "wall_seconds": time.time() - started,
        "npz": str(stem.with_suffix(".npz").relative_to(ROOT)),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "onset_ms": onset_ms,
        "classification": morphology["classification"]["status"],
        "wall_s": round(time.time() - started, 1),
    }), flush=True)


if __name__ == "__main__":
    main()

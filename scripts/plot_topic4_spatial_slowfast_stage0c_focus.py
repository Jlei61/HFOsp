#!/usr/bin/env python3
"""Reproduce the strongest Stage-0C LUT-blocked oscillatory diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_m3b_spectral_phase import _LUT_MU, _LUT_SIG  # noqa: E402
from src.topic4_spatial_slowfast_stage0c import (  # noqa: E402
    ForkClassifierThresholds,
    PoolParameters,
    classify_fork_batch,
    equilibrium_state,
    simulate_forks,
)


RESULT = ROOT / "results/topic4_sef_hfo/spatial_slowfast_topology/stage0c_dynamic_divisive_pool"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run() -> tuple[Path, Path, Path]:
    params = PoolParameters(z=0.80, alpha_g=12.0, w_ee_mult=1.1, ratio=1.0)
    initial = equilibrium_state((0.0005, 0.002))[None, :]
    simulation = simulate_forks(
        initial,
        [params],
        dt_ms=0.25,
        duration_ms=6000.0,
        save_stride=20,
        audit_tail_fraction=0.40,
    )
    metadata = [
        {
            "z": params.z,
            "alpha_G": params.alpha_g,
            "w_ee_mult": params.w_ee_mult,
            "ratio": params.ratio,
            "initial_kind": "on_manifold_probe",
            "initial_label": "probe_rest",
            "root_index": None,
            "initial_rE_hz": 0.5,
            "initial_rI_hz": 2.0,
        }
    ]
    row = classify_fork_batch(metadata, simulation, ForkClassifierThresholds())[0]
    screen_rows = json.loads((RESULT / "state_fork_screen.json").read_text(encoding="utf-8"))
    cohort_rows = [
        item
        for item in screen_rows
        if item["initial_kind"] != "exact_root"
        and item["classification"] == "audit_invalid_candidate"
        and np.isclose(float(item["z"]), 0.80)
        and np.isclose(float(item["alpha_G"]), 12.0)
    ]
    cohort_context = {
        "n_distinct_nonexact_histories": len({item["initial_label"] for item in cohort_rows}),
        "tail_mean_hz_median": float(np.median([item["tail_mean_hz"] for item in cohort_rows])),
        "tail_peak_hz_max": float(np.max([item["stepwise_tail_peak_rE_hz"] for item in cohort_rows])),
        "dominant_frequency_hz_consensus": float(np.median([item["dominant_frequency_hz"] for item in cohort_rows])),
        "tail_lut_clip_occupancy_median": float(np.median([item["lut_clip_tail_occupancy_stepwise"] for item in cohort_rows])),
    }
    fields = ("muE_mV", "sigmaE_mV", "muI_mV", "sigmaI_mV")
    ranges = {
        field: {
            "min": float(np.min(simulation[field][:, 0])),
            "max": float(np.max(simulation[field][:, 0])),
        }
        for field in fields
    }
    summary = {
        "schema_version": "topic4_spatial_slowfast_stage0c.focus.v1",
        "scope": "diagnostic_replay_of_existing_locked_grid_point_not_confirm_or_new_parameter_search",
        "point": {"z": 0.80, "alpha_G": 12.0, "w_ee_mult": 1.1, "ratio": 1.0},
        "initial_condition": "probe_rest_on_manifold",
        "screen_contract": {"dt_ms": 0.25, "duration_ms": 6000.0, "save_stride": 20},
        "classification": row,
        "locked_screen_cohort_context": cohort_context,
        "moment_ranges_saved": ranges,
        "lut_support": {"mu_mV": list(_LUT_MU[:2]), "sigma_mV": list(_LUT_SIG[:2])},
        "interpretation_cn": (
            "该锁定点在LUT实现中呈重复有界振荡，但振荡低谷的muE/muI越过LUT下界；"
            "因此它是需要扩展/精确transfer复核的信号，不是已确认limit cycle。"
        ),
        "implementation_sha256": _sha(ROOT / "src/topic4_spatial_slowfast_stage0c.py"),
        "producer_sha256": _sha(Path(__file__).resolve()),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    json_path = RESULT / "focused_clipped_orbit_diagnostic.json"
    npz_path = RESULT / "focused_clipped_orbit_trace.npz"
    _atomic_json(json_path, summary)
    np.savez_compressed(
        npz_path,
        time_ms=simulation["time_ms"],
        rE_khz=simulation["rE_khz"][:, 0],
        rI_khz=simulation["rI_khz"][:, 0],
        rE_fast_khz=simulation["rE_fast_khz"][:, 0],
        mu_G=simulation["mu_G"][:, 0],
        S_G=simulation["S_G"][:, 0],
        divisor=simulation["divisor"][:, 0],
        muE_mV=simulation["muE_mV"][:, 0],
        sigmaE_mV=simulation["sigmaE_mV"][:, 0],
        muI_mV=simulation["muI_mV"][:, 0],
        sigmaI_mV=simulation["sigmaI_mV"][:, 0],
    )

    time_s = simulation["time_ms"] / 1000.0
    r_e_hz = 1000.0 * simulation["rE_khz"][:, 0]
    figures = RESULT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_path = figures / "stage0c_clipped_orbit_diagnostic.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    axes[0, 0].plot(time_s, r_e_hz, color="#b33b4b", lw=1.25)
    axes[0, 0].axhline(100.0, color="0.45", ls="--", lw=0.8)
    axes[0, 0].axvspan(3.6, 6.0, color="#d9d9d9", alpha=0.35, lw=0)
    axes[0, 0].set(xlabel="time (s)", ylabel="E rate (Hz)", title="Repeated bounded activity in the LUT implementation")

    axes[0, 1].plot(time_s, simulation["S_G"][:, 0], label=r"$S_G$", color="#4c78a8")
    axes[0, 1].plot(time_s, simulation["divisor"][:, 0], label="D", color="#f28e2b")
    axes[0, 1].set(xlabel="time (s)", ylabel="pool state / divisor", title="Delayed recurrent-gain feedback")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(time_s, simulation["muE_mV"][:, 0], label=r"$\mu_E$", color="#59a14f", lw=1.0)
    axes[1, 0].plot(time_s, simulation["muI_mV"][:, 0], label=r"$\mu_I$", color="#af7aa1", lw=1.0)
    axes[1, 0].axhline(_LUT_MU[0], color="black", ls="--", lw=0.9, label="LUT lower bound")
    axes[1, 0].fill_between(time_s, np.minimum(simulation["muE_mV"][:, 0], _LUT_MU[0]), _LUT_MU[0], color="#e15759", alpha=0.18)
    axes[1, 0].set(xlabel="time (s)", ylabel="input mean (mV)", title="The apparent orbit crosses transfer support")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=8)

    points = axes[1, 1].scatter(simulation["S_G"][:, 0], r_e_hz, c=time_s, s=7, cmap="viridis", rasterized=True)
    axes[1, 1].set(xlabel=r"$S_G$", ylabel="E rate (Hz)", title="Projected loop (diagnostic only)")
    colorbar = fig.colorbar(points, ax=axes[1, 1], pad=0.02)
    colorbar.set_label("time (s)")
    axes[1, 1].text(
        0.02,
        0.98,
        f"pre-audit: {row.get('pre_audit_classification')}\n"
        f"{cohort_context['n_distinct_nonexact_histories']} non-exact histories; {cohort_context['dominant_frequency_hz_consensus']:.2f} Hz\n"
        f"tail mean {cohort_context['tail_mean_hz_median']:.2f} Hz; peak {cohort_context['tail_peak_hz_max']:.2f} Hz\n"
        f"tail LUT clip {100*cohort_context['tail_lut_clip_occupancy_median']:.1f}%",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.suptitle(r"Stage 0C unresolved signal: $z=0.80$, $\alpha_G=12$", fontsize=14)
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    return figure_path, json_path, npz_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to reproduce the locked focused diagnostic")
    paths = run()
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

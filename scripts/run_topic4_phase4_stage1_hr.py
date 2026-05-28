#!/usr/bin/env python3
"""SEF-ITP Phase 4 Stage 1 CLI: HR single-node parameter sweep.

Modes:
  --mode smoke      Small sweep, ~5 min single-threaded; default.
  --mode full       Full sweep, ~25 min single-threaded; lock-in baseline.
  --mode synthetic-allsilent
                    Artificial test mode: tiny grid guaranteed to find no
                    excitable regime. Used by test_topic4_modeling_hr_cli
                    to verify exit-code 1 contract.

Outputs (JSON, in --output-dir):
  - sweep_results.json    : list[dict] of all sweep rows
  - regime_summary.json   : config, regime counts, baseline, exit-contract flag
  - baseline.json         : (I*, r*, sigma*) — only if baseline found
  - regime_map.png        : 3-axis heatmap
  - phase_portraits/      : baseline + 2 neighbor phase portraits

Exit codes:
  0  Stage 1 exit contract met (baseline found, noise-robust)
  1  Exit contract failed (no excitable baseline)
  2  Argparse/usage error (default argparse behavior)

Spec:  docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
Plan:  docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.topic4_modeling.hr_core import HRParams
from src.topic4_modeling.hr_dynamics import simulate_trajectory
from src.topic4_modeling.hr_sweep import (
    pick_excitable_baseline,
    sweep_hr_parameters,
)
from src.topic4_modeling.hr_viz import plot_phase_portrait, plot_regime_map

DEFAULT_OUTPUT = Path("results/topic4_sef_itp/phase4_modeling/stage1_hr_single")

# Parameter ranges recalibrated 2026-05-28 to cover HR's actual excitable
# regime. Diagnostic (post detect_bursts recalibration, commit 5f9450c):
# the excitable band — silent at sigma=0, noise-triggered events at sigma>0,
# no runaway — lives at I in [-1, 1] with sigma ~ 0.2-0.6. The original
# guessed ranges (I up to 0.0, sigma <= 0.15) sat entirely in the silent
# region → all-silent sweep → no baseline. These ranges make the grid
# COVER the regime that exists; the picker still selects objectively
# (silent at sigma=0 + excitable + both-side noise-robust). Confirmed: a
# regime-covering sweep returns baseline (I*=1.0, r*=0.006, σ*=0.4).
# Note the sigma grids include enough density (0.5x / 1x / 1.5x spacings)
# for the picker's lower/upper [0.4-0.6 sigma] / [1.4-1.6 sigma] neighbor
# checks to have in-grid candidates.
CONFIGS: dict[str, dict] = {
    "smoke": {
        "I_grid": np.linspace(-1.5, 1.0, 6).tolist(),
        "r_grid": [0.004, 0.006, 0.008],
        "sigma_grid": [0.0, 0.2, 0.4, 0.6],
        "seeds": [0, 1, 2],
        "T": 500.0, "dt": 0.05,
    },
    "full": {
        "I_grid": np.arange(-1.5, 1.05, 0.25).tolist(),
        "r_grid": np.arange(0.003, 0.013, 0.001).tolist(),
        "sigma_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "seeds": [0, 1, 2, 3, 4],
        "T": 1000.0, "dt": 0.05,
    },
    # Test-only: deep-subthreshold + zero noise = guaranteed all silent
    "synthetic-allsilent": {
        "I_grid": [-3.5, -3.0],
        "r_grid": [0.006],
        "sigma_grid": [0.0, 0.05],
        "seeds": [0, 1],
        "T": 50.0, "dt": 0.05,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=list(CONFIGS), default="smoke")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cfg = CONFIGS[args.mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_portraits").mkdir(exist_ok=True)

    n_cells = (len(cfg["I_grid"]) * len(cfg["r_grid"])
                * len(cfg["sigma_grid"]) * len(cfg["seeds"]))
    print(f"[stage1] mode={args.mode}  n_jobs={args.n_jobs}  n_cells={n_cells}")

    df = sweep_hr_parameters(
        I_grid=cfg["I_grid"], r_grid=cfg["r_grid"],
        sigma_grid=cfg["sigma_grid"], seeds=cfg["seeds"],
        T=cfg["T"], dt=cfg["dt"], n_jobs=args.n_jobs,
    )
    print(f"[stage1] sweep done, n_rows={len(df)}")
    print(f"[stage1] regimes:\n{df['regime'].value_counts().to_string()}")

    # JSON sweep results (no parquet)
    (args.output_dir / "sweep_results.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, default=str)
    )

    # Regime map. Plot failure handling (v3 fix):
    #   - synthetic-allsilent: tiny grid may legitimately not plot → log + continue
    #   - smoke / full: plot failure is an exit-contract violation → flag + exit 1
    plot_failed = False
    plot_error: str | None = None
    try:
        fig = plot_regime_map(df)
        fig.savefig(args.output_dir / "regime_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[stage1] regime_map.png saved")
    except Exception as e:  # noqa: BLE001 — intentional broad catch + exit-1 escalation
        if args.mode == "synthetic-allsilent":
            print(f"[stage1] regime_map skipped (synthetic mode): {e}")
        else:
            plot_failed = True
            plot_error = repr(e)
            print(f"[stage1] regime_map FAILED ({args.mode} mode): {e}")

    baseline = pick_excitable_baseline(df)
    contract_passed = (baseline is not None) and (not plot_failed)
    summary = {
        "mode": args.mode,
        "sweep_config": {k: v if not hasattr(v, "tolist") else v.tolist()
                          for k, v in cfg.items()},
        "regime_counts": df["regime"].value_counts().to_dict(),
        "baseline": baseline,
        "regime_map_failed": plot_failed,
        "regime_map_error": plot_error,
        "stage1_exit_contract_passed": contract_passed,
    }
    (args.output_dir / "regime_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("[stage1] regime_summary.json saved")

    if baseline is None:
        print("[stage1] EXIT CONTRACT FAILED: no excitable baseline found")
        print("[stage1] per spec §8 stage 1 fallback: try FHN-with-adaptation")
        return 1
    if plot_failed:
        print("[stage1] EXIT CONTRACT FAILED: regime_map plot failed in non-synthetic mode")
        print(f"[stage1] error: {plot_error}")
        return 1

    # Baseline + phase portraits
    (args.output_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2)
    )
    print(f"[stage1] baseline (I*, r*, σ*) = "
           f"({baseline['I_star']:.3f}, {baseline['r_star']:.4f}, "
           f"{baseline['sigma_star']:.3f})  [hand-off to Stage 2]")

    p = HRParams()
    for label, (I, r, sigma) in [
        ("baseline", (baseline["I_star"], baseline["r_star"], baseline["sigma_star"])),
        ("baseline_zero_noise", (baseline["I_star"], baseline["r_star"], 0.0)),
        ("baseline_high_noise",
         (baseline["I_star"], baseline["r_star"], baseline["sigma_star"] * 2.0)),
    ]:
        p_cell = replace(p, r=r)
        _, traj = simulate_trajectory(p_cell, I=I, T=300.0, dt=0.05,
                                       sigma_ou=sigma, tau_ou=10.0, seed=0)
        fig = plot_phase_portrait(traj, p_cell, I=I)
        fig.savefig(args.output_dir / "phase_portraits" / f"{label}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("[stage1] phase_portraits/ written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

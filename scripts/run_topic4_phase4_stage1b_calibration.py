#!/usr/bin/env python3
"""SEF-ITP Phase 4 Stage 1b CLI: burst-envelope observation-unit calibration.

User-ratified decision (2026-05-28): the primary event is a node's BURST
ENVELOPE (onset of a cluster of spikes), not a single spike excursion. This
script runs the minimal single-node calibration that validates that unit with
data at the Stage 1 baseline (I*=1.0, r*=0.006), comparing spike-level vs
burst-envelope event statistics across sigma in {0, 0.2, 0.4, 0.6}.

It does NOT route envelopes through classify_regime (RegimeConfig is spike-unit
tuned; reusing it on longer envelope spans would mislabel — Stage 1b plan
contract clause #6). Instead it reports RAW envelope statistics; "excitable-like"
is judged from those (sigma=0 silent + sigma>0 sparse, brief events, IBI >>
duration).

Outputs (in --output-dir):
  - comparison.json            : per-sigma spike-vs-envelope table + pass flags
  - figures/spike_vs_envelope.png
  - figures/README.md          : Chinese per-figure note (AGENTS.md spec)

Exit codes:
  0  Stage 1b acceptance met (sigma=0 silent + sigma>0 noise-triggered envelopes)
  1  Acceptance failed
  2  Argparse/usage error

Spec:  docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §5.6
Plan:  docs/superpowers/plans/2026-05-28-topic4-phase4-stage1b-burst-envelope-calibration.md
Stage1: docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-28.md
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

from src.topic4_modeling.hr_config import BurstConfig
from src.topic4_modeling.hr_core import HRParams
from src.topic4_modeling.hr_dynamics import (
    detect_burst_envelopes,
    detect_bursts,
    simulate_trajectory,
)

DEFAULT_OUTPUT = Path(
    "results/topic4_sef_itp/phase4_modeling/stage1b_envelope_calibration"
)

# Stage 1 baseline hand-off (stage1_results_2026-05-28.md): per-node dynamics.
BASELINE_I = 1.0
BASELINE_R = 0.006
SIGMAS = [0.0, 0.2, 0.4, 0.6]
T = 1000.0
DT = 0.05
BURN_IN = 100.0
SEEDS = [0, 1, 2, 3, 4]
# sigma=0 silent gate: mean envelope count must be at/under this (tolerance for
# a rare burn-in edge spike). Spec acceptance #2: "sigma=0 -> 0 or very low".
SIGMA0_TOL = 0.5
# Excitable-like gate: events must be sparse + brief, i.e. mean inter-event
# interval >> mean event duration, for EVERY sigma>0 (incl the tightest, 0.6).
# This is the property that actually supports the Stage 1b conclusion — without
# it a future "repetitive" regime (IBI ~ duration) would false-PASS the weaker
# count-only gate. k=3 ⇒ active duty cycle < 25%. Margin check: measured ratios
# are 22.4 / 9.8 / 6.0 at sigma 0.2/0.4/0.6, so k=3 passes with ≥2× headroom on
# the worst point yet rejects repetitive regimes (ratio → 1-2). MEASURED bound.
IBI_DURATION_RATIO_MIN = 3.0


def _spike_stats(spikes: list[tuple[float, float]]) -> dict:
    n = len(spikes)
    durs = [e - s for s, e in spikes]
    ibis = [spikes[i + 1][0] - spikes[i][1] for i in range(n - 1)]
    return {
        "count": n,
        "mean_duration": float(np.mean(durs)) if durs else 0.0,
        "mean_ibi": float(np.mean(ibis)) if ibis else float("nan"),
    }


def _envelope_stats(envs) -> dict:
    n = len(envs)
    durs = [e.duration for e in envs]
    nsp = [e.n_spikes for e in envs]
    ibis = [envs[i + 1].onset - envs[i].offset for i in range(n - 1)]
    return {
        "count": n,
        "mean_duration": float(np.mean(durs)) if durs else 0.0,
        "mean_n_spikes_per_burst": float(np.mean(nsp)) if nsp else 0.0,
        "mean_ibi": float(np.mean(ibis)) if ibis else float("nan"),
    }


def evaluate_acceptance(rows: list[dict]) -> dict:
    """Stage 1b acceptance from per-sigma rows (raw envelope stats only).

    Pass requires ALL of (no classify_regime on envelopes — plan clause #6):
      1. sigma=0 envelope count <= SIGMA0_TOL                (silent gate)
      2. every sigma>0 has >= 1 envelope event              (noise-triggered)
      3. every sigma>0 is excitable-like: mean_ibi > k * mean_duration
         (sparse + brief, NOT repetitive). The tightest point (sigma=0.6) must
         satisfy this too — it is included in "every sigma>0". A row with < 2
         envelopes has no defined IBI but a single event in T is trivially
         sparse, so it passes condition 3.
    """
    k = IBI_DURATION_RATIO_MIN
    sigma0 = next(r for r in rows if r["sigma"] == 0.0)
    sigma0_silent = sigma0["envelope"]["count"] <= SIGMA0_TOL
    noise_rows = [r for r in rows if r["sigma"] > 0.0]
    noise_triggered = all(r["envelope"]["count"] >= 1.0 for r in noise_rows)

    def _ratio(r: dict) -> float:
        e = r["envelope"]
        dur = e["mean_duration"]
        ibi = e["mean_ibi"]
        if e["count"] < 2 or np.isnan(ibi):
            return float("inf")  # single event = trivially sparse
        return ibi / dur if dur > 0 else float("inf")

    ratios = {r["sigma"]: _ratio(r) for r in noise_rows}
    excitable_like = all(ratios[s] > k for s in ratios)
    return {
        "sigma0_silent": bool(sigma0_silent),
        "sigma0_envelope_count": sigma0["envelope"]["count"],
        "noise_triggered": bool(noise_triggered),
        "ibi_duration_ratio_min": k,
        "ibi_duration_ratios": {f"{s:.1f}": ratios[s] for s in ratios},
        "excitable_like": bool(excitable_like),
        "stage1b_pass": bool(sigma0_silent and noise_triggered and excitable_like),
    }


def calibrate(output_dir: Path) -> dict:
    cfg = BurstConfig()  # production defaults: spike fields + envelope_gap=30
    p = replace(HRParams(), r=BASELINE_R)
    rows = []
    for sigma in SIGMAS:
        spike_runs, env_runs = [], []
        for seed in SEEDS:
            t, traj = simulate_trajectory(
                p, I=BASELINE_I, T=T, dt=DT,
                sigma_ou=sigma, tau_ou=10.0, seed=seed, burn_in=BURN_IN,
            )
            x = traj[:, 0]
            spike_runs.append(_spike_stats(detect_bursts(x, t, cfg)))
            env_runs.append(_envelope_stats(detect_burst_envelopes(x, t, cfg)))

        def _avg(runs, key):
            vals = [r[key] for r in runs if not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        rows.append({
            "sigma": sigma,
            "spike": {
                "count": _avg(spike_runs, "count"),
                "mean_duration": _avg(spike_runs, "mean_duration"),
                "mean_ibi": _avg(spike_runs, "mean_ibi"),
            },
            "envelope": {
                "count": _avg(env_runs, "count"),
                "mean_duration": _avg(env_runs, "mean_duration"),
                "mean_n_spikes_per_burst": _avg(env_runs, "mean_n_spikes_per_burst"),
                "mean_ibi": _avg(env_runs, "mean_ibi"),
            },
        })

    return {
        "baseline": {"I": BASELINE_I, "r": BASELINE_R, "T": T, "dt": DT,
                     "burn_in": BURN_IN, "seeds": SEEDS},
        "envelope_gap": cfg.envelope_gap,
        "burst_config": {"x_threshold": cfg.x_threshold,
                         "min_burst_duration": cfg.min_burst_duration,
                         "bridge_gap": cfg.bridge_gap,
                         "envelope_gap": cfg.envelope_gap},
        "rows": rows,
        "acceptance": evaluate_acceptance(rows),  # raw-stats only — clause #6
    }


def plot_comparison(result: dict) -> "plt.Figure":
    rows = result["rows"]
    sig = [r["sigma"] for r in rows]
    spike_n = [r["spike"]["count"] for r in rows]
    env_n = [r["envelope"]["count"] for r in rows]
    env_dur = [r["envelope"]["mean_duration"] for r in rows]
    env_ibi = [r["envelope"]["mean_ibi"] for r in rows]
    nspb = [r["envelope"]["mean_n_spikes_per_burst"] for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: event count under each unit (Q: does unit change the count?).
    axA.plot(sig, spike_n, "o-", color="#888888", label="spike-level")
    axA.plot(sig, env_n, "s-", color="#1f77b4", label="burst envelope")
    for xs, ne, k in zip(sig, env_n, nspb):
        if ne >= 1:
            axA.annotate(f"{k:.1f} sp/burst", (xs, ne),
                         textcoords="offset points", xytext=(4, 6), fontsize=8)
    axA.set_xlabel("OU noise amplitude σ")
    axA.set_ylabel("events per 1000 HR time units")
    axA.set_title("Event count: spike vs burst-envelope unit")
    axA.legend(frameon=False)
    axA.set_ylim(bottom=0)
    axA.set_xlim(-0.03, 0.70)  # room for the rightmost sp/burst annotation

    # Panel B: envelope sparsity/brevity (Q: are envelopes excitable-like?).
    axB.plot(sig, env_dur, "s-", color="#d62728", label="mean duration")
    axB.plot(sig, env_ibi, "^-", color="#2ca02c", label="mean inter-event interval")
    axB.set_xlabel("OU noise amplitude σ")
    axB.set_ylabel("HR time units")
    axB.set_title("Burst-envelope brevity vs spacing (excitable-like if IBI ≫ dur)")
    axB.legend(frameon=False)
    axB.set_ylim(bottom=0)

    fig.suptitle(
        f"Stage 1b burst-envelope calibration  "
        f"(baseline I={BASELINE_I}, r={BASELINE_R}, envelope_gap="
        f"{result['envelope_gap']:g})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _write_figures_readme(fig_dir: Path, result: dict) -> None:
    acc = result["acceptance"]
    verdict = "PASS" if acc["stage1b_pass"] else "FAIL"
    fig_dir.joinpath("README.md").write_text(
        "# Stage 1b 图说明\n\n"
        "### spike_vs_envelope.png\n\n"
        "两面板对照"
        "“把单次越阈（spike）当事件”与“把一簇 spike 合并成一个 burst envelope "
        "当事件”两种观测单位。左图：每种单位在 1000 时间单位里数出多少事件；点上"
        "标注的是每个 envelope 平均含几个 spike。右图：envelope 的平均时长与平均"
        "事件间隔——间隔远大于时长说明事件稀疏且短暂（可激样）。"
        f"baseline I={BASELINE_I}, r={BASELINE_R}, envelope_gap="
        f"{result['envelope_gap']:g}；当前判定 **{verdict}**。\n\n"
        "**关注点**：σ=0 时 envelope 计数应为 0（静息）；σ>0 时 envelope 数明显少于 "
        "spike 数且每 burst 含 >1 spike（两单位确实不同），右图 IBI 应远大于 duration。\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage1b] baseline I={BASELINE_I} r={BASELINE_R}  σ={SIGMAS}  "
          f"seeds={SEEDS}  T={T}")
    result = calibrate(args.output_dir)

    print("[stage1b] spike vs envelope (averaged over seeds):")
    for r in result["rows"]:
        e = r["envelope"]
        s = r["spike"]
        print(f"  σ={r['sigma']:.1f}  spike_n={s['count']:.1f}  "
              f"env_n={e['count']:.1f}  n_sp/burst={e['mean_n_spikes_per_burst']:.2f}  "
              f"env_dur={e['mean_duration']:.1f}  env_ibi={e['mean_ibi']:.1f}")

    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    fig = plot_comparison(result)
    fig.savefig(fig_dir / "spike_vs_envelope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_figures_readme(fig_dir, result)
    print("[stage1b] comparison.json + figures/ written")

    acc = result["acceptance"]
    if not acc["stage1b_pass"]:
        print(f"[stage1b] ACCEPTANCE FAILED: sigma0_silent="
              f"{acc['sigma0_silent']} (env_count={acc['sigma0_envelope_count']:.2f}), "
              f"noise_triggered={acc['noise_triggered']}, "
              f"excitable_like={acc['excitable_like']} "
              f"(IBI/dur ratios {acc['ibi_duration_ratios']}, "
              f"need > {acc['ibi_duration_ratio_min']})")
        return 1
    print("[stage1b] ACCEPTANCE MET: σ=0 silent + σ>0 noise-triggered + "
          "excitable-like (IBI ≫ duration for all σ>0 incl 0.6)")
    print(f"[stage1b] hand-off to Stage 2: unit = burst envelope "
          f"(onset = first spike start), envelope_gap={result['envelope_gap']:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

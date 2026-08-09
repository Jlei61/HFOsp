#!/usr/bin/env python3
"""Plot the furthest scientifically eligible FCXR-LC4 result: the paired F0 gate.

The sprint stopped before the frozen-D onset surface and lifecycle stages.  This diagnostic
therefore plots only measured F0 quantities; it never fabricates empty F1/F2 panels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
          / "lc4_lifecycle_gate")
OUT = RESULT / "figures"

COLORS = {
    "control": "#7f7f7f",
    "n6": "#b2182b",
    "n8": "#2166ac",
    "pass": "#3a923a",
    "fail": "#c33c2f",
}


def _smooth(x: np.ndarray, bins: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(bins) / bins, mode="same")


def _load() -> tuple[dict, dict[str, np.lib.npyio.NpzFile]]:
    verdict = json.loads((RESULT / "baseline_verdict.json").read_text())
    traces = {
        key: np.load(RESULT / "runs" / f"baseline_{key}_traces.npz")
        for key in ("control", "n6", "n8")
    }
    return verdict, traces


def main() -> None:
    verdict, traces = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = {int(row["candidate"]["n"]): row for row in verdict["candidates"]}

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.15), facecolor="white")

    # A — the paired population traces -------------------------------------------------------
    for key, label in (("control", "actuator off"), ("n6", "Hill n=6"),
                       ("n8", "Hill n=8")):
        z = traces[key]
        dt_s = float(z["rate_dt_ms"][0]) / 1000.0
        t = np.arange(z["rate_E"].size) * dt_s
        axes[0].plot(t, _smooth(z["rate_E"]), lw=1.25, color=COLORS[key], label=label)
    axes[0].axvspan(0, 2, color="0.92", lw=0)
    axes[0].text(1.0, 0.96, "burn-in", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", color="0.35", fontsize=7.5)
    axes[0].set(xlim=(0, 12), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="paired interictal trajectories")
    axes[0].legend(frameon=False, fontsize=7.7, loc="upper right")

    # B — every functional ratio, with its own locked acceptance interval ------------------
    metrics = [
        ("event_rate", "event\nrate"),
        ("iei_cv", "IEI CV"),
        ("duration", "event\nduration"),
        ("participation", "spatial\nparticipation"),
    ]
    x = np.arange(len(metrics), dtype=float)
    for i, (key, _label) in enumerate(metrics):
        lo, hi = candidates[6]["gate"]["ratio_bands"][key]
        axes[1].vlines(i, lo, hi, color="#b7d8b7", lw=12, alpha=0.75, zorder=0)
        axes[1].plot(i, 1.0, "_", color="#246b24", ms=11, mew=1.2, zorder=1)
    for n, dx in ((6, -0.09), (8, 0.09)):
        values = [candidates[n]["gate"]["ratios"][key] for key, _ in metrics]
        axes[1].scatter(x + dx, values, s=38, color=COLORS[f"n{n}"], edgecolor="white",
                        linewidth=0.5, label=f"Hill n={n}", zorder=3)
    axes[1].axhline(1, color="0.35", ls=":", lw=0.8)
    axes[1].set_xticks(x, [label for _, label in metrics])
    axes[1].set_ylim(0.45, 1.85)
    axes[1].set(ylabel="candidate / paired control", title="functional baseline gate")
    axes[1].legend(frameon=False, fontsize=7.7, loc="upper right")

    # C — leakage is not the failing clause -------------------------------------------------
    ns = [6, 8]
    leakage_pct = [100.0 * candidates[n]["gate"]["current_fraction"] for n in ns]
    axes[2].bar([0, 1], leakage_pct, color=[COLORS["n6"], COLORS["n8"]], width=0.62)
    axes[2].axhline(0.1, color="k", ls="--", lw=1.0)
    for i, value in enumerate(leakage_pct):
        axes[2].text(i, value * 1.45, f"{value:.4f}%", ha="center", va="bottom", fontsize=8)
    axes[2].text(1.45, 0.1, "locked ceiling 0.1%", ha="right", va="bottom", fontsize=7.5)
    axes[2].set_yscale("log")
    axes[2].set_ylim(5e-4, 0.2)
    axes[2].set_xticks([0, 1], ["Hill n=6", "Hill n=8"])
    axes[2].set(ylabel="max outward current / recurrent scale", title="leakage is numerically tiny")

    # D — explicit stop reason; F1/F2 were never measured ----------------------------------
    clause_order = [
        ("numerical_safe", "numerically safe"),
        ("no_sustained_bout", "no high bout"),
        ("at_least_three_returning", "at least 3 IEDs"),
        ("current_leakage", "current below 0.1%"),
        ("event_rate_ratio", "event-rate ratio"),
        ("iei_cv_ratio", "IEI-CV ratio"),
        ("duration_ratio", "duration ratio"),
        ("participation_ratio", "participation ratio"),
    ]
    matrix = np.asarray([[bool(candidates[n]["gate"]["clauses"][key])
                          for key, _ in clause_order] for n in ns], dtype=int)
    rgba = np.empty((2, len(clause_order), 4), dtype=float)
    rgba[matrix == 1] = matplotlib.colors.to_rgba(COLORS["pass"])
    rgba[matrix == 0] = matplotlib.colors.to_rgba(COLORS["fail"])
    axes[3].imshow(rgba, aspect="auto", interpolation="nearest")
    axes[3].set_yticks([0, 1], ["Hill n=6", "Hill n=8"])
    axes[3].set_xticks(np.arange(len(clause_order)), [lab for _, lab in clause_order],
                       rotation=57, ha="right", fontsize=7.2)
    for i in range(2):
        for j in range(len(clause_order)):
            axes[3].text(j, i, "PASS" if matrix[i, j] else "FAIL", ha="center", va="center",
                         fontsize=6.2, color="white", fontweight="bold")
    axes[3].set_title("F0 stop: no eligible candidate\nF1 onset / F2 lifecycle not authorised")

    for letter, ax in zip("ABCD", axes):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("FCXR-LC4: steep cooperative activation reduces leakage but does not preserve\n"
                 "the paired interictal event statistics", fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lc4_functional_baseline_gate.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "lc4_functional_baseline_gate",
        "kind": "F0 diagnostic; not a lifecycle or paper-claim figure",
        "verdict": verdict["verdict"],
        "panels": {
            "A": "100 ms-smoothed paired E-population rates; grey region is excluded burn-in",
            "B": "candidate/control functional ratios and metric-specific locked bands",
            "C": "executed maximum outward-current fraction and the locked leakage ceiling",
            "D": "all eight F0 clauses; F1/F2 explicitly not run",
        },
        "claim_boundary": (
            "Both candidates were numerically safe and below the leakage ceiling.  n=6 failed "
            "event rate and IEI-CV; n=8 failed event rate.  F1/F2 were gate-blocked, so this "
            "artifact contains no onset-surface or lifecycle evidence."
        ),
        "source": str(RESULT.relative_to(ROOT)),
    }
    (OUT / "lc4_functional_baseline_gate_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(OUT / "lc4_functional_baseline_gate.png")


if __name__ == "__main__":
    main()

"""Topic5 R2b native-3D sensitivity — cohort figure.

Reads  results/topic5_ictal_recruitment/contact_similarity/r2b_summary_{activation}.json
Writes results/topic5_ictal_recruitment/contact_similarity/figures/r2b_sensitivity_{activation}.png

Question this figure answers: does swapping the normalized 2D contact-plane
projection for native 3D Euclidean distance (mm) change the contact-similarity
readout? Two independent panels (CLAUDE.md §7):

  A  per-subject R2b - R2_nm (both no-mirror, SAME coord-mapped common subset)
     Q: for each subject, does native-3D geometry move the observed similarity
        away from the 2D-plane value, and by how much? NA subjects (no coord
        coverage / units failure / too few common channels) are greyed and
        labeled with their r2b_status reason, not silently dropped.
  B  the ladder extended: R1 (stored, mirror, full-channel) -> R2_nm (2D
     no-mirror, common subset) -> R2b (native-3D no-mirror, common subset)
     Q: where does R2b sit relative to the published ladder's R1, for readers
        who only know the original R1/R2/R3 figure?

Tier: sensitivity/robustness (docs/superpowers/plans/2026-07-01-topic5-r2b-3d-
sensitivity.md). Language stays narrow -- "2D-plane vs native-3D geometry",
never "characterizes pathological network".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

_DEF_OUT_DIR = "results/topic5_ictal_recruitment/contact_similarity"

# same progression semantics as plot_topic5_contact_similarity.RUNG_COLORS
# (raw/blue -> in-plane-smoothed/orange -> more-geometry/red), reused here for
# R1 (stored) -> R2_nm (2D no-mirror) -> R2b (native-3D no-mirror).
RUNG_COLORS = {"R1": "#4575b4", "R2_nm": "#fdae61", "R2b": "#d73027"}
OK_COLOR = "#2b8cbe"
NA_COLOR = "#bdbdbd"


def _short(subject_id):
    return subject_id.replace("epilepsiae_", "E").replace("yuquan_", "Y")


def _rung_obs(p, rung):
    d = p.get(rung)
    if not isinstance(d, dict):
        return np.nan
    v = d.get("obs_subject")
    return float(v) if v is not None else np.nan


def _rung_insufficient(p, rung):
    d = p.get(rung)
    return isinstance(d, dict) and d.get("status") == "INSUFFICIENT_NULL"


def load_summary(path: Path):
    return json.load(open(path))


# ---------------------------------------------------------------------------
# Panel A — per-subject R2b - R2_nm
# ---------------------------------------------------------------------------
def panel_A(ax, per_subject):
    """Bar per subject: R2b - R2_nm (no-mirror, common coord-mapped subset).
    NA subjects: grey, height 0, labeled with their r2b_status reason."""
    n = len(per_subject)
    x = np.arange(n)
    heights, colors, hatches, na_reasons = [], [], [], []
    for p in per_subject:
        if p["r2b_status"] != "ok":
            heights.append(0.0)
            colors.append(NA_COLOR)
            hatches.append(None)
            na_reasons.append(p["r2b_status"])
        else:
            heights.append(p.get("r2b_minus_r2nm") or 0.0)
            colors.append(OK_COLOR)
            insuff = _rung_insufficient(p, "R2_nm") or _rung_insufficient(p, "R2b")
            hatches.append("////" if insuff else None)
            na_reasons.append(None)

    bars = ax.bar(x, heights, color=colors, width=0.6, zorder=3)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("black")
    ax.axhline(0, color="black", linewidth=0.8, zorder=4)

    for xi, reason in zip(x, na_reasons):
        if reason is not None:
            ax.text(xi, 0.0, reason, rotation=90, ha="center", va="bottom",
                    fontsize=6, color="#636363")

    labels = [_short(p["subject_id"]) for p in per_subject]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    for tick, p in zip(ax.get_xticklabels(), per_subject):
        if p["r2b_status"] != "ok":
            tick.set_color("#636363")
            tick.set_fontstyle("italic")

    ax.set_ylabel("R2b − R2_nm  (native-3D − 2D-plane, no-mirror, common subset)", fontsize=9)
    ax.set_title("A  Per-subject 3D-vs-2D contact-kernel delta", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)

    legend_handles = [
        mpatches.Patch(facecolor=OK_COLOR, label="ok"),
        mpatches.Patch(facecolor=OK_COLOR, hatch="////", edgecolor="black",
                       label="ok, null under-powered (INSUFFICIENT_NULL)"),
        mpatches.Patch(facecolor=NA_COLOR, label="NA (reason labeled on bar)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right", framealpha=0.7)

    finite = [h for h, p in zip(heights, per_subject) if p["r2b_status"] == "ok"]
    if finite:
        lo, hi = min(finite + [0.0]), max(finite + [0.0])
        pad = max((hi - lo) * 0.2, 0.02)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.7, n - 0.3)


# ---------------------------------------------------------------------------
# Panel B — ladder extended: R1 (stored) -> R2_nm -> R2b
# ---------------------------------------------------------------------------
def panel_B(ax, per_subject):
    """Per (ok) subject vertical connector through R1 (stored, mirror,
    full-channel) -> R2_nm (2D no-mirror, common subset) -> R2b (native-3D
    no-mirror, common subset), so a reader anchored on the published R1/R2/R3
    figure sees where R2b sits."""
    ok = [p for p in per_subject if p["r2b_status"] == "ok"]
    if not ok:
        ax.text(0.5, 0.5, "No ok subjects", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(ok))
    rung_vals = {
        "R1": [p.get("r1_obs_stored") if p.get("r1_obs_stored") is not None else np.nan for p in ok],
        "R2_nm": [_rung_obs(p, "R2_nm") for p in ok],
        "R2b": [_rung_obs(p, "R2b") for p in ok],
    }

    for xi in x:
        ys = [rung_vals[r][xi] for r in ("R1", "R2_nm", "R2b")]
        ys = [y for y in ys if np.isfinite(y)]
        if len(ys) >= 2:
            ax.plot([xi] * len(ys), ys, "-", color="grey", linewidth=0.8, zorder=1)

    for rung, marker in (("R1", "o"), ("R2_nm", "s"), ("R2b", "^")):
        ax.scatter(x, rung_vals[rung], color=RUNG_COLORS[rung], marker=marker,
                  s=45, zorder=3, label=rung)

    labels = [_short(p["subject_id"]) for p in ok]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Median|corr| (within-shaft null, obs)", fontsize=9)
    ax.set_title("B  Geometry ladder extended: R1 (stored) → R2_nm (2D) → R2b (native-3D)",
                 fontsize=9, fontweight="bold")
    legend_labels = {"R1": "R1 (stored, mirror, full-channel)",
                     "R2_nm": "R2_nm (2D-plane, no-mirror, common subset)",
                     "R2b": "R2b (native-3D, no-mirror, common subset)"}
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_labels[r] for r in ("R1", "R2_nm", "R2b")],
              fontsize=6.5, loc="lower right", framealpha=0.7)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    all_vals = [v for r in rung_vals.values() for v in r if np.isfinite(v)]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.12 + 0.02
        ax.set_ylim(max(0, lo - pad), min(1.05, hi + pad))
    ax.set_xlim(-0.7, len(ok) - 0.3)


# ---------------------------------------------------------------------------
# main figure
# ---------------------------------------------------------------------------
def make_figure(summary: dict, out_dir: Path, activation: str):
    per_subject = sorted(summary.get("per_subject", []), key=lambda p: p["subject_id"])
    if not per_subject:
        print(f"[plot] r2b_summary_{activation}.json has no per_subject entries; nothing to plot.")
        return None

    n_ok = summary.get("n_ok", sum(1 for p in per_subject if p["r2b_status"] == "ok"))
    act_label = activation.replace("broadband", "broadband (80–500 Hz)")

    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(15, 6))
    panel_A(ax_A, per_subject)
    panel_B(ax_B, per_subject)

    verdict = summary.get("r2b_minus_r2nm_negligible")
    verdict_txt = "n/a" if verdict is None else ("negligible" if verdict else "not negligible")
    median = summary.get("r2b_minus_r2nm_median")
    median_txt = "n/a" if median is None else f"{median:+.3f}"
    n_insuff = summary.get("n_ok_insufficient_null", 0)

    fig.suptitle(
        f"R2b native-3D sensitivity · {act_label} · n_ok={n_ok}/{len(per_subject)}\n"
        f"cohort R2b−R2_nm median={median_txt}, SESOI(0.05) verdict={verdict_txt} "
        f"({n_insuff} ok subject(s) with under-powered null)",
        fontsize=10, y=1.00
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"r2b_sensitivity_{activation}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", choices=["broadband", "hfa"], default=None,
                    help="render one band; default renders both broadband + hfa "
                         "(skipping any band whose r2b_summary json is absent)")
    ap.add_argument("--out-dir", default=_DEF_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    activations = [args.activation] if args.activation else ["broadband", "hfa"]

    for activation in activations:
        summary_path = out_dir / f"r2b_summary_{activation}.json"
        if not summary_path.exists():
            print(f"[plot] {summary_path} not found — skipping {activation} "
                  "(run scripts/augment_topic5_r2b_3d.py first)")
            continue
        summary = load_summary(summary_path)
        make_figure(summary, out_dir / "figures", activation)


if __name__ == "__main__":
    main()

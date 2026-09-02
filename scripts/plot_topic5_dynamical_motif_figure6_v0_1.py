#!/usr/bin/env python3
"""Figure 6 for the Topic 5.2 dynamical motif run.

Panel contract -- one independent question each:

    a  what the model is asked to do, and which four rules are compared
    b  one patient: real events keep a template while contact order varies
    c  stretching the spread along a direction: does it help predict the next contact?
    d  pushing along the early movement: does it help?
    e  the point the calibration split picked, scored on held-out events
    f  the hand-forward chain against its three alternative mechanisms
    g  directional persistence: data against the isotropic model's own events
    h  early seizure field: what the interictal pattern adds over a static picture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANEL_LABEL = dict(fontsize=11, fontweight="bold", va="top", ha="left")
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
})
INK = "#22252a"
ACCENT = "#1f6f8b"
WARM = "#c1553a"
GREY = "#9aa0a6"
LIGHT = "#c8ccd0"


def unavailable(axis, message: str = "not available") -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes,
              color=GREY, fontsize=7.5)
    axis.set_axis_off()


def schematic_panel(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.add_patch(plt.Rectangle((0.02, 0.60), 0.24, 0.30, fill=False, ec=INK, lw=0.9))
    axis.text(0.14, 0.94, "seen so far", ha="center", fontsize=7.5, color=INK)
    for index, height in enumerate((0.64, 0.72, 0.80)):
        for column in range(4):
            axis.add_patch(plt.Rectangle((0.05 + 0.048 * column, height), 0.034, 0.05,
                                         fc=ACCENT if column == index else "white",
                                         ec=INK, lw=0.5))
    axis.annotate("", xy=(0.345, 0.75), xytext=(0.285, 0.75),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    axis.add_patch(plt.Circle((0.50, 0.75), 0.105, fill=False, ec=INK, lw=0.9))
    axis.text(0.50, 0.75, "tissue\nsheet", ha="center", va="center", fontsize=7.5)
    axis.annotate("", xy=(0.715, 0.75), xytext=(0.62, 0.75),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    axis.add_patch(plt.Rectangle((0.73, 0.60), 0.24, 0.30, fill=False, ec=INK, lw=0.9))
    axis.text(0.85, 0.94, "one continuation", ha="center", fontsize=7.5, color=INK)
    rng = np.random.default_rng(3)
    for height in (0.64, 0.72, 0.80):
        pick = int(rng.integers(0, 4))
        for column in range(4):
            axis.add_patch(plt.Rectangle((0.76 + 0.048 * column, height), 0.034, 0.05,
                                         fc=WARM if column == pick else "white",
                                         ec=INK, lw=0.5))
    labels = ["spread evenly",
              "spread stretched along one direction",
              "pushed the way the event already moved",
              "handed forward along that direction"]
    for index, text in enumerate(labels):
        y = 0.44 - 0.10 * index
        axis.add_patch(plt.Circle((0.07, y + 0.012), 0.014,
                                  fc=[LIGHT, "#8fb8c8", ACCENT, WARM][index], ec=INK, lw=0.5))
        axis.text(0.12, y, text, fontsize=7.4, color=INK, va="center")
    axis.text(0.07, -0.10, "same sheet and same local neighbourhood in all four;\n"
                           "only how activity moves across it changes",
              fontsize=7.0, color=GREY, va="bottom")


def case_panel(axis_a, axis_b, root: Path, frame: str, subject: str | None) -> str | None:
    cache = root / "frame_cache" / frame
    census = pd.read_csv(root / "GEOMETRY_ONLY_FIT_CENSUS.csv")
    two_d = census[census.geometry_class == "TWO_DIMENSIONAL"]
    if subject is None:
        # A case panel needs a genuinely two-dimensional layout and enough
        # held-out events for both templates to be estimated.
        candidates = two_d[(two_d.n_contacts >= 9) & (two_d.n_contacts <= 20)]
        candidates = candidates.sort_values("n_model_unseen", ascending=False)
        subject = str(candidates.subject.iloc[0]) if len(candidates) else str(two_d.subject.iloc[0])
    directory = cache / subject
    if not directory.exists():
        unavailable(axis_a)
        axis_b.set_axis_off()
        return None
    plane = np.load(directory / "plane.npz", allow_pickle=False)
    events = np.load(directory / "events.npz", allow_pickle=True)
    xy = np.asarray(plane["contacts_xy_mm"], float)
    ranks = np.asarray(events["ranks"])
    split = np.asarray(events["split"])
    mode = np.asarray(events["prefix_mode"])
    unseen = np.flatnonzero(split == -1)
    scatter = None
    for axis, template in ((axis_a, 0), (axis_b, 1)):
        members = unseen[mode[unseen] == template]
        if members.size == 0:
            axis.set_axis_off()
            continue
        block = np.where(ranks[members] >= 0, ranks[members], np.nan).astype(float)
        with np.errstate(invalid="ignore"):
            normalised = block / np.nanmax(block, axis=1, keepdims=True)
            colour = np.nanmean(normalised, axis=0)
            spread = np.nanstd(normalised, axis=0)
        size = 26 + 150 * np.nan_to_num(spread) / max(float(np.nanmax(spread)), 1e-9)
        scatter = axis.scatter(xy[:, 0], xy[:, 1], c=colour, s=size, cmap="viridis",
                               vmin=0, vmax=1, edgecolor=INK, linewidth=0.4)
        axis.set_title(f"template {'A' if template == 0 else 'B'}  (n = {members.size})",
                       loc="left", pad=12)
        axis.set_xlabel("along the implantation axis (mm)")
        if template == 0:
            axis.set_ylabel("across it (mm)")
        else:
            axis.set_yticklabels([])
    if scatter is not None:
        bar = plt.colorbar(scatter, ax=axis_b, fraction=0.06, pad=0.04)
        bar.set_label("mean order (0 first, 1 last);\nmarker size = order variability")
    return subject


def dose_panel(axis, profile: pd.DataFrame, sweep: str, xlabel: str, title: str) -> None:
    block = profile[profile.sweep == sweep]
    if block.empty:
        unavailable(axis)
        return
    # At each dose keep the best angle: the strongest case the motif can make.
    best = (block.groupby(["subject", "value"]).calibration_contact_nll.min().reset_index())
    zero = best[best.value == 0.0].set_index("subject").calibration_contact_nll
    best["delta"] = [row.calibration_contact_nll - zero.get(row.subject, np.nan)
                     for row in best.itertuples()]
    for _, group in best.groupby("subject"):
        group = group.sort_values("value")
        axis.plot(group.value, group.delta, color=LIGHT, lw=0.7, zorder=1)
    cohort = best.groupby("value").delta.median().reset_index().sort_values("value")
    axis.plot(cohort.value, cohort.delta, color=WARM, lw=2.0, zorder=3,
              label=f"cohort median (n = {best.subject.nunique()})")
    axis.axhline(0.0, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    axis.axvline(0.0, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("change in next-contact error (nats)\nbelow zero = better")
    axis.set_title(title, loc="left")
    axis.legend(frameon=False, loc="upper center")
    limit = float(np.nanquantile(np.abs(best.delta), 0.90)) or 1e-3
    axis.set_ylim(-0.45 * limit, 1.15 * limit)
    hidden = int((best.delta.abs() > 1.15 * limit).sum())
    if hidden:
        axis.text(0.99, 0.02, f"{hidden} of {len(best)} points outside", fontsize=6.5,
                  color=GREY, ha="right", va="bottom", transform=axis.transAxes)


def held_out_panel(axis, selected: pd.DataFrame) -> None:
    if selected.empty:
        unavailable(axis)
        return
    order = [("eta", "stretch the spread"), ("beta", "push along early movement"),
             ("gamma", "hand forward along it")]
    present = [(key, label) for key, label in order if (selected.sweep == key).any()]
    rng = np.random.default_rng(5)
    counts: list[str] = []
    for position, (key, _) in enumerate(present):
        values = selected[selected.sweep == key].unseen_contact_nll_gain.to_numpy()
        jitter = (rng.random(values.size) - 0.5) * 0.26
        axis.scatter(values, position + jitter, s=16, color=GREY, alpha=0.8,
                     edgecolor="none", zorder=1)
        axis.scatter([np.median(values)], [position], s=48, color=WARM, zorder=3,
                     edgecolor=INK, linewidth=0.6)
        counts.append(f"{int((values > 0).sum())}/{values.size}")
    axis.axvline(0.0, color=INK, lw=0.9, ls=(0, (4, 3)))
    axis.set_yticks(np.arange(len(present)))
    axis.set_yticklabels([f"{label}\n({count} better)"
                          for (_, label), count in zip(present, counts)])
    axis.set_ylim(-0.6, len(present) - 0.4)
    axis.invert_yaxis()
    axis.set_xlabel("held-out gain at the point the other half chose (nats)\n"
                    "right = better; label = patients better")
    axis.set_title("does the chosen amount generalise?", loc="left")


def paired_panel(axis, evidence: dict, goal: str, keys: list[tuple[str, str]],
                 metric: str, title: str, xlabel: str) -> None:
    block = (evidence.get("goals", {}).get(goal) or {})
    entries = []
    for key, label in keys:
        effect = (block.get(key) or {}).get(metric) or {}
        if effect.get("per_subject"):
            entries.append((label, effect, np.asarray(list(effect["per_subject"].values()), float)))
    if not entries:
        unavailable(axis)
        return
    rng = np.random.default_rng(7)
    for position, (_, effect, sample) in enumerate(entries):
        jitter = (rng.random(sample.size) - 0.5) * 0.26
        axis.scatter(sample, position + jitter, s=14, color=GREY, alpha=0.8,
                     edgecolor="none", zorder=1)
        axis.plot([effect["ci_low"], effect["ci_high"]], [position, position],
                  color=INK, lw=1.5, solid_capstyle="butt", zorder=2)
        axis.scatter([effect["median"]], [position], s=44, color=WARM, zorder=3,
                     edgecolor=INK, linewidth=0.6)

    axis.axvline(0.0, color=INK, lw=0.9, ls=(0, (4, 3)))
    axis.set_yticks(np.arange(len(entries)))
    axis.set_yticklabels([f"{label}\n({effect['n_positive']}/{effect['n']} better)"
                          for label, effect, _ in entries])
    axis.set_ylim(-0.6, len(entries) - 0.4)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left")


def persistence_panel(axis, root: Path) -> None:
    path = root / "PERSISTENCE_MODEL_GAP_PER_PATIENT.csv"
    if not path.exists():
        unavailable(axis)
        return
    table = pd.read_csv(path).dropna(subset=["DM0_ISOTROPIC_cosine_mean"])
    if table.empty:
        unavailable(axis)
        return
    x = table.DM0_ISOTROPIC_cosine_mean.to_numpy()
    y = table.observed_cosine.to_numpy()
    low = float(min(x.min(), y.min())) - 0.02
    high = float(max(x.max(), y.max())) + 0.02
    axis.plot([low, high], [low, high], color=GREY, lw=0.9, ls=(0, (4, 3)))
    axis.scatter(x, y, s=26, color=ACCENT, edgecolor=INK, linewidth=0.5, zorder=3)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal")
    axis.set_xlabel("events the even-spreading model generates")
    axis.set_ylabel("real events")
    axis.set_title("how much events keep going the same way", loc="left")
    above = int((y > x).sum())
    axis.text(0.03, 0.97, f"{above}/{len(table)} above the line",
              transform=axis.transAxes, fontsize=7, va="top", color=INK,
              bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))


def seizure_panel(axis, root: Path) -> None:
    path = root / "SEIZURE_INCREMENTAL_REUSE_PER_PATIENT.csv"
    if not path.exists():
        unavailable(axis, "seizure branch not available")
        return
    table = pd.read_csv(path)
    if table.empty:
        unavailable(axis, "seizure branch not available")
        return
    real = table.delta_error_real_median.to_numpy()
    pseudo = table.delta_error_pseudo_median.to_numpy()
    finite = np.isfinite(real) & np.isfinite(pseudo)
    real, pseudo = real[finite], pseudo[finite]
    limit = float(np.quantile(np.abs(np.concatenate([real, pseudo])), 0.9)) * 1.6 + 1e-6
    axis.plot([-limit, limit], [-limit, limit], color=GREY, lw=0.9, ls=(0, (4, 3)))
    axis.axhline(0, color=INK, lw=0.7)
    axis.axvline(0, color=INK, lw=0.7)
    inside = (np.abs(real) <= limit) & (np.abs(pseudo) <= limit)
    axis.scatter(pseudo[inside], real[inside], s=26, color=ACCENT,
                 edgecolor=INK, linewidth=0.5, zorder=3)
    outside = int((~inside).sum())
    if outside:
        axis.text(0.97, 0.03, f"{outside} patient off scale", transform=axis.transAxes,
                  fontsize=6.8, color=GREY, ha="right",
                  bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.85))
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.set_xlabel("at a matched non-seizure moment")
    axis.set_ylabel("at the real seizure onset")
    axis.set_title("what the interictal pattern adds\nover a static picture", loc="left")
    axis.text(0.03, 0.97, f"{int((real > 0).sum())}/{len(real)} above zero",
              transform=axis.transAxes, fontsize=7, va="top", color=INK,
              bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))


def sensitivity_panel(axis, root: Path) -> None:
    """How big a gain the same pipeline reports when the answer is known.

    Three rows on one axis: synthetic cells built with no motif at all,
    synthetic cells built with a real motif, and the real patients.  Without
    this the eight panels above are eight zeros with no scale to read them
    against.
    """
    grid_path = root / "toy_identifiability/IDENTIFIABILITY_GRID.csv"
    selected_path = root / "DOSE_RESPONSE_PER_PATIENT.csv"
    if not grid_path.exists() or not selected_path.exists():
        unavailable(axis, "sensitivity map not available")
        return
    grid = pd.read_csv(grid_path)
    grid = grid[grid.sweep.notna()].copy()
    for column in ("truth_value", "unseen_gain"):
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    grid = grid[np.isfinite(grid.unseen_gain)]
    # The synthetic map only sweeps eta and beta, so restrict the real patients to
    # the same two knobs; both tables define gain as zero-NLL minus best-NLL.
    real_patients = pd.read_csv(selected_path)
    real_patients = real_patients[real_patients.sweep.isin(sorted(set(grid.sweep)))]
    real_gain = pd.to_numeric(real_patients.unseen_contact_nll_gain,
                              errors="coerce").dropna().to_numpy()

    rows = [
        ("made-up data\nwith no motif", grid[grid.truth_value == 0].unseen_gain.to_numpy(), WARM),
        ("made-up data\nwith a real motif", grid[grid.truth_value != 0].unseen_gain.to_numpy(),
         ACCENT),
        ("real patients", real_gain, INK),
    ]
    axis.axvline(0, color=INK, lw=0.7, zorder=1)
    rng = np.random.default_rng(20260816)
    limit = 0.06
    for position, (label, values, colour) in enumerate(rows):
        y = len(rows) - 1 - position
        inside = np.abs(values) <= limit
        jitter = rng.uniform(-0.16, 0.16, size=int(inside.sum()))
        axis.scatter(values[inside], np.full(int(inside.sum()), y) + jitter,
                     s=17, color=colour, edgecolor="white", linewidth=0.35, zorder=3)
        for outlier in values[~inside]:
            axis.annotate(f"{outlier:+.2f}", xy=(limit * 0.98, y), xytext=(limit * 0.72, y + 0.30),
                          fontsize=6.6, color=colour, ha="center",
                          arrowprops=dict(arrowstyle="->", color=colour, lw=0.7))
        axis.text(-limit * 0.98, y + 0.34, label, fontsize=7, color=INK, va="center")
    axis.set_yticks([])
    axis.set_ylim(-0.6, len(rows) - 0.2)
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("held-out gain from adding the motif (nats)\nright = the motif helped")
    axis.set_title("could we have seen it if it were there?", loc="left")
    with_motif = rows[1][1]
    axis.text(0.97, 0.03,
              f"motif really present: {int((with_motif > 0).sum())}/{len(with_motif)} positive",
              transform=axis.transAxes, fontsize=6.8, color=GREY, ha="right",
              bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.85))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--case-subject", default=None)
    args = parser.parse_args()

    root = args.out_root
    evidence_path = root / "EVIDENCE_MATRIX.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    profile_path = root / "DOSE_RESPONSE_PROFILE.csv"
    selected_path = root / "DOSE_RESPONSE_PER_PATIENT.csv"
    profile = pd.read_csv(profile_path) if profile_path.exists() else pd.DataFrame()
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()

    figure = plt.figure(figsize=(13.0, 12.4))
    grid = GridSpec(4, 12, figure=figure, hspace=0.78, wspace=3.0,
                    height_ratios=[1.05, 1.0, 1.0, 1.0])
    axes = {
        "a": figure.add_subplot(grid[0, 0:6]),
        "b1": figure.add_subplot(grid[0, 6:9]),
        "b2": figure.add_subplot(grid[0, 9:12]),
        "c": figure.add_subplot(grid[1, 0:6]),
        "d": figure.add_subplot(grid[1, 6:12]),
        "e": figure.add_subplot(grid[2, 0:6]),
        "f": figure.add_subplot(grid[2, 6:12]),
        "g": figure.add_subplot(grid[3, 0:4]),
        "h": figure.add_subplot(grid[3, 4:8]),
        "i": figure.add_subplot(grid[3, 8:12]),
    }
    schematic_panel(axes["a"])
    case_subject = case_panel(axes["b1"], axes["b2"], root, args.frame, args.case_subject)
    dose_panel(axes["c"], profile, "eta",
               "how much the spread is stretched along its best direction",
               "stretching the spread into a corridor")
    dose_panel(axes["d"], profile, "beta",
               "push along the direction the event already moved\n(left = pushed backwards)",
               "pushing the way the event already moved")
    held_out_panel(axes["e"], selected)
    paired_panel(axes["f"], evidence, "G3",
                 [("DM3_AXIS_FEEDFORWARD_TRANSIENT__vs__DM2_LOCAL_DIRECTIONAL",
                   "vs. push alone"),
                  ("DM3_AXIS_FEEDFORWARD_TRANSIENT__vs__DM3_GAIN_MEMORY",
                   "vs. stronger and slower"),
                  ("DM3_AXIS_FEEDFORWARD_TRANSIENT__vs__DM3_SYMMETRIC_MATCHED",
                   "vs. same links, both ways"),
                  ("DM3_AXIS_FEEDFORWARD_TRANSIENT__vs__DM3_AXIS_SHUFFLED_TRIANGULAR",
                   "vs. same links, shuffled order")],
                 "prediction", "the hand-forward chain against its alternatives",
                 "held-out gain per patient (nats)\nright = chain is better; label = patients better")
    persistence_panel(axes["g"], root)
    seizure_panel(axes["h"], root)
    sensitivity_panel(axes["i"], root)

    for key, axis in axes.items():
        if key == "b2":
            continue
        axis.text(-0.22, 1.20, {"b1": "b"}.get(key, key), transform=axis.transAxes,
                  **PANEL_LABEL)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / "figure6_dynamical_motif"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(f"{stem}.{extension}", bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "contract": "topic5_dynamical_motif_figure6_v0_1",
        "frame": args.frame, "case_subject": case_subject,
        "n_patients_dose_profile": int(profile.subject.nunique()) if not profile.empty else 0,
        "sources": {
            "dose_profile": str(profile_path), "dose_selected": str(selected_path),
            "evidence": str(evidence_path),
            "persistence": str(root / "PERSISTENCE_MODEL_GAP_PER_PATIENT.csv"),
            "seizure": str(root / "SEIZURE_INCREMENTAL_REUSE_PER_PATIENT.csv"),
            "sensitivity": str(root / "toy_identifiability/IDENTIFIABILITY_GRID.csv"),
        },
    }
    (figures / "figure6_dynamical_motif_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    source = figures / "FIGURE6_SOURCE_DATA"
    source.mkdir(exist_ok=True)
    for name, frame_data in (("panel_cd_dose_profile.csv", profile),
                             ("panel_e_dose_selected.csv", selected)):
        if not frame_data.empty:
            frame_data.to_csv(source / name, index=False)
    for name, path in (("panel_g_persistence.csv", root / "PERSISTENCE_MODEL_GAP_PER_PATIENT.csv"),
                       ("panel_h_seizure.csv", root / "SEIZURE_INCREMENTAL_REUSE_PER_PATIENT.csv"),
                       ("panel_i_sensitivity.csv",
                        root / "toy_identifiability/IDENTIFIABILITY_GRID.csv")):
        if path.exists():
            pd.read_csv(path).to_csv(source / name, index=False)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()

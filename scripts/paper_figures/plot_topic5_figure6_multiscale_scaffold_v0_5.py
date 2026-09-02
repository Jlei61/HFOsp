#!/usr/bin/env python3
"""Paper-ready Figure 6 candidate for the locked v0.5 multiscale scaffold."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_multiscale_fields_v0_5 import train_mode_to_ab  # noqa: E402
from paper_figures.plot_topic5_figure6_lbss_full_tissue_v0_3 import (  # noqa: E402
    BLUE, DARK, ENERGY_CMAP, GRAY, RED, TIMING_CMAP,
    draw_field, draw_full_tissue_graph, field_geometry, grid_letter,
    paired_axis, paired_test, patient_strip, stars,
)


SUBJECT = "epilepsiae_1146"
FIT_ID = f"{SUBJECT}__shared"
L0 = "L0_LOCAL_ONLY"
L1 = "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"
L2M = "L2M_MACRO_MATCHED_RANDOM_LR"
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
SUFFIX = "C_L3_ORDER_SHUFFLED"
COLORS = {L0: "#7c858b", L1: "#6592a2", L2M: "#b98b44", L3: RED, SUFFIX: "#b9b9b9"}


def generated_rank(sequence: list[list[int]], n_contacts: int) -> np.ndarray:
    result = np.full(n_contacts, -1, dtype=int)
    for rank, contacts in enumerate(sequence):
        result[np.asarray(contacts, int)] = rank
    return result


def normalized_event_matrix(rows: list[np.ndarray], n_contacts: int) -> np.ndarray:
    matrix = np.full((n_contacts, len(rows)), np.nan)
    for column, rank in enumerate(rows):
        finite = rank >= 0
        if finite.any():
            matrix[finite, column] = rank[finite] / max(1.0, float(rank[finite].max()))
    return matrix


def draw_event_reproduction(fig: plt.Figure, spec, out: Path, old: Path,
                            canonical: Path) -> None:
    sub = spec.subgridspec(2, 3, width_ratios=(1, 1, 0.035), wspace=0.10, hspace=0.13)
    axes = np.asarray([[fig.add_subplot(sub[i, j]) for j in range(2)] for i in range(2)])
    events = np.load(out / "cache" / FIT_ID / "events.npz", allow_pickle=False)
    provenance = json.loads((out / "cache" / FIT_ID / "provenance.json").read_text())
    mapping = train_mode_to_ab(out / "cache" / FIT_ID, SUBJECT,
                               np.asarray(provenance["joint_contacts"]),
                               canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject")
    with gzip.open(old / "per_fit" / FIT_ID / L3 / "seed0/heldout_rollouts.json.gz", "rt") as stream:
        rollout_rows = json.load(stream)
    by_source = {int(row["event_source_index"]): row for row in rollout_rows}
    empirical = json.loads((canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json").read_text())["interictal_field"]
    order = [str(value) for value in empirical["contact_order"]]
    contacts = list(map(str, provenance["joint_contacts"]))
    take = np.asarray([order.index(name) for name in contacts])
    common_order = np.argsort(np.asarray(empirical["rank_a"], float)[take], kind="stable")
    image = None
    for row_index, (template, color) in enumerate((("A", RED), ("B", BLUE))):
        candidates = [index for index in np.flatnonzero(events["split"] == 2)
                      if mapping[int(events["mode"][index])] == template
                      and int(events["event_source_index"][index]) in by_source]
        candidates = candidates[:30]
        observed = [events["ranks"][index] for index in candidates]
        generated = [generated_rank(
            by_source[int(events["event_source_index"][index])]["generated_rank_sets"], len(contacts)
        ) for index in candidates]
        for column, rows in enumerate((observed, generated)):
            matrix = normalized_event_matrix(rows, len(contacts))[common_order]
            cmap = mpl.colormaps[TIMING_CMAP].copy(); cmap.set_bad("#e7e7e7")
            image = axes[row_index, column].imshow(matrix, aspect="auto", interpolation="nearest",
                                                   cmap=cmap, vmin=0, vmax=1)
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values(): spine.set_visible(False)
        axes[row_index, 0].set_ylabel(f"T{template}", color=color, rotation=0, labelpad=12,
                                     fontsize=11.5, fontweight="bold", va="center")
    axes[0, 0].set_title("Data", fontsize=11, pad=4)
    axes[0, 1].set_title("Generated", fontsize=11, pad=4)
    axes[0, 0].text(0, 1.20, "E1146", transform=axes[0, 0].transAxes,
                    fontsize=10.5, fontweight="bold", va="bottom")
    cax = fig.add_subplot(sub[:, 2])
    bar = fig.colorbar(image, cax=cax, orientation="vertical")
    bar.set_ticks([0, 1], labels=["First", "Last"])
    bar.ax.set_title("Rank", fontsize=8.5, pad=3)
    bar.ax.tick_params(labelsize=8, length=2, pad=2)


def draw_interictal_cohort(ax: plt.Axes, contact_analysis: Path) -> dict:
    frame = pd.read_csv(contact_analysis / "interictal_patient_statistics.csv").sort_values("subject")
    p = paired_test(frame.native_model.to_numpy() - frame.static_only.to_numpy(), "greater")
    paired_axis(ax, frame.native_model, frame.static_only, ("RNN", "Static"),
                (RED, GRAY), "Propagation correlation", p)
    ax.set_title(f"Interictal · n={frame.subject.nunique()}", fontsize=11.5,
                 fontweight="bold", pad=5)
    return {"n": int(frame.subject.nunique()), "p": p}


def draw_cross_state_fields(fig: plt.Figure, spec, out: Path, canonical: Path) -> dict:
    sub = spec.subgridspec(1, 6, width_ratios=(1, .075, 1, .075, 1, .055), wspace=.24)
    axes = [fig.add_subplot(sub[0, index]) for index in (0, 2, 4)]
    bars = [fig.add_subplot(sub[0, index]) for index in (1, 3, 5)]
    field = json.loads((canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json").read_text())["interictal_field"]
    order = list(map(str, field["contact_order"]))
    with np.load(out / "model_fields/intact/per_patient" / SUBJECT / f"{L3}.npz", allow_pickle=False) as model:
        names = model["contacts"].astype(str).tolist()
        take = np.asarray([names.index(name) for name in order])
        rank_a = 1.0 - np.asarray(model["A_canonical_full"], float)[take]
        rank_b = 1.0 - np.asarray(model["B_canonical_full"], float)[take]
        support_a = np.asarray(model["A_participation"], float)[take]
        support_b = np.asarray(model["B_participation"], float)[take]
    with np.load(out / "early_ictal/per_patient_targets" / f"{SUBJECT}.npz", allow_pickle=False) as target:
        names = target["contacts"].astype(str).tolist()
        lookup = dict(zip(names, np.asarray(target["median_broadband_energy"], float)))
        energy = np.asarray([lookup[name] for name in order])
        n_seizures = int(target["n_seizures"])
    points, xlim, ylim = field_geometry(field)
    images = [
        draw_field(axes[0], points, rank_a, support_a, xlim, ylim, cmap=TIMING_CMAP,
                   vmin=0, vmax=1, title="RNN TA", title_color=RED, show_y=True),
        draw_field(axes[1], points, rank_b, support_b, xlim, ylim, cmap=TIMING_CMAP,
                   vmin=0, vmax=1, title="RNN TB", title_color=BLUE, show_y=False),
    ]
    energy_image = draw_field(axes[2], points, energy, np.ones_like(energy), xlim, ylim,
                              cmap=ENERGY_CMAP, vmin=float(np.nanmin(energy)),
                              vmax=float(np.nanmax(energy)), title="Early-ictal broadband",
                              title_color=DARK, show_y=False)
    for image, cax in zip(images, bars[:2]):
        bar = fig.colorbar(image, cax=cax, orientation="vertical")
        bar.set_ticks([0, 1], labels=["Early", "Late"]); bar.ax.tick_params(labelsize=8, pad=1)
    bar = fig.colorbar(energy_image, cax=bars[2], orientation="vertical")
    bar.ax.set_title("z", fontsize=8, pad=2); bar.ax.tick_params(labelsize=8, pad=1)
    return {"subject": SUBJECT, "n_seizures": n_seizures}


def draw_early_cohort(fig: plt.Figure, spec, out: Path) -> dict:
    sub = spec.subgridspec(1, 2, wspace=.70)
    ax_null, ax_j = (fig.add_subplot(sub[0, index]) for index in range(2))
    frame = pd.read_csv(out / "early_ictal/EARLY_ICTAL_PER_PATIENT.csv")
    l3 = frame[(frame.condition == f"INTACT|{L3}") & (frame.endpoint == "canonical_full")].sort_values("subject")
    p = paired_test(l3.observed.to_numpy() - l3.all_contact_null_median.to_numpy(), "greater")
    paired_axis(ax_null, l3.observed, l3.all_contact_null_median,
                ("RNN", "Shuffle"), (RED, GRAY), "Signed field correlation", p)
    ax_null.set_title(f"Early ictal · n={l3.subject.nunique()}", fontsize=10.8,
                      fontweight="bold", pad=4)
    patient = frame[frame.endpoint == "canonical_full"].pivot(
        index="subject", columns="condition", values="observed"
    )
    delta = patient[f"INTACT|{L3}"] - patient[f"INTACT|{L2M}"]
    J = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv").set_index("subject")
    common = delta.index.intersection(J.index)
    ax_j.scatter(J.loc[common, "J_lat_exceedance_burden"], delta.loc[common],
                 s=25, color=RED, edgecolor="white", lw=.35)
    ax_j.axhline(0, color="#858b8e", lw=.75, ls="--")
    ax_j.set_xscale("symlog", linthresh=1e-4)
    ax_j.set_xlabel("Cross-fitted nonlocality J")
    ax_j.set_ylabel("Selected − matched\nsigned field correlation")
    summary = json.loads((out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json").read_text())
    interaction = summary["primary_interaction"]
    if interaction.get("status") != "NOT_IDENTIFIABLE" and stars(interaction["permutation_p_greater"]):
        ax_j.text(.98, .98, stars(interaction["permutation_p_greater"]), transform=ax_j.transAxes,
                  ha="right", va="top", fontsize=12, fontweight="bold")
    ax_j.spines[["top", "right"]].set_visible(False)
    return {"n": int(l3.subject.nunique()), "p_vs_null": p, "interaction": interaction}


def draw_mechanism_row(fig: plt.Figure, spec, out: Path) -> dict:
    sub = spec.subgridspec(1, 4, wspace=.70)
    axes = [fig.add_subplot(sub[0, index]) for index in range(4)]
    contrasts = pd.read_csv(out / "INTERICTAL_PATIENT_CONTRASTS.csv")
    summary = json.loads((out / "INTERICTAL_V0_5_SUMMARY.json").read_text())
    primary = pd.DataFrame(summary["primary_rows"])
    axes[0].scatter(primary.J_lat_exceedance_burden, primary.gain_nats, s=24,
                    color=RED, edgecolor="white", lw=.35)
    axes[0].axhline(0, color="#858b8e", lw=.75, ls="--")
    axes[0].set_xscale("symlog", linthresh=1e-4)
    axes[0].set(xlabel="Cross-fitted nonlocality J", ylabel="Selected − matched\n distal gain (nats)")
    p_primary = summary["comparisons"]["primary_nonlocality_interaction_all"]["permutation_p_greater"]
    if stars(p_primary):
        axes[0].text(.98, .98, stars(p_primary), transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=12, fontweight="bold")

    labels = ("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2m_distal")
    names = ("Local", "Nearby", "Matched")
    p_values = []
    for index, label in enumerate(labels):
        values = contrasts.loc[contrasts.contrast == label, "gain_nats"].to_numpy()
        axes[1].scatter(index + np.linspace(-.11, .11, len(values)), values, s=14,
                        color="#a4a9ac", alpha=.78)
        axes[1].plot([index-.18, index+.18], [np.nanmedian(values)]*2, color=RED, lw=2)
        p_values.append(summary["comparisons"][label].get("holm_p_greater_within_claim", 1.0))
        if stars(p_values[-1]):
            axes[1].text(index, axes[1].get_ylim()[1] if axes[1].has_data() else 1,
                         stars(p_values[-1]), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].axhline(0, color="#858b8e", lw=.75, ls="--")
    axes[1].set_xticks(range(3), names); axes[1].set_ylabel("Distal gain (nats)")

    attenuation = pd.read_csv(out / "ATTENUATION_PER_PATIENT_DOSE.csv")
    for target, label, color in (("L1_ADDED", "Nearby", BLUE), ("L2M_ADDED", "Matched", "#b98b44"),
                                 ("L3_ADDED", "Selected", RED), ("L3_MATCHED_LOCAL", "Local", GRAY)):
        eligible = attenuation[
            (attenuation.target == target) & attenuation.inferential_eligible.astype(bool)
        ]
        data = eligible.groupby("alpha").distal_selectivity.median()
        n_patients = int(eligible.subject.nunique())
        axes[2].plot(data.index, data.values, marker="o", ms=3.5, lw=1.5,
                     color=color, label=f"{label} (n={n_patients})")
    axes[2].axhline(0, color="#858b8e", lw=.75, ls="--")
    axes[2].set(xlabel="Edge attenuation", ylabel="Distal-selective damage")
    axes[2].legend(frameon=False, fontsize=8, ncol=2, handlelength=1.3,
                   loc="upper left", bbox_to_anchor=(-.03, 1.03))

    flow = pd.read_csv(out / "mechanism/MODE_FLOW_ATTENUATION_PER_PATIENT.csv")
    flow = flow.loc[(flow.condition != "MATCHED_RANDOM") | flow.random_match_eligible.astype(bool)]
    flow = flow.groupby(["subject", "condition"], as_index=False).distal_selectivity.mean()
    order = ("SAME_MODE", "CROSS_MODE", "MATCHED_RANDOM")
    for index, (condition, color) in enumerate(zip(order, (RED, BLUE, GRAY))):
        values = flow.loc[flow.condition == condition, "distal_selectivity"].to_numpy()
        axes[3].scatter(index + np.linspace(-.11, .11, len(values)), values, s=14,
                        color="#a4a9ac", alpha=.78)
        axes[3].plot([index-.18, index+.18], [np.nanmedian(values)]*2, color=color, lw=2)
    axes[3].axhline(0, color="#858b8e", lw=.75, ls="--")
    matched_n = int(flow.loc[flow.condition == "MATCHED_RANDOM", "subject"].nunique())
    axes[3].set_xticks(range(3), ("Same", "Cross", f"Matched\nrandom\n(n={matched_n})"))
    axes[3].set_ylabel("Mode-selective damage")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return {"primary_interaction_p": p_primary, "distal_holm_p": dict(zip(labels, p_values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_multiscale_effective_scaffold_v0_5")
    parser.add_argument("--old-root", type=Path,
                        default=ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3")
    parser.add_argument("--contact-analysis", type=Path,
                        default=ROOT / "results/topic5_rnn_full_cohort_field_transfer_v0_1")
    parser.add_argument("--canonical-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures")
    args = parser.parse_args()
    out, old, canonical = args.out_root.resolve(), args.old_root.resolve(), args.canonical_root.resolve()
    if not (out / "EARLY_ICTAL_SCORING_COMPLETE.json").exists():
        raise RuntimeError("Figure 6 requires completed locked early-ictal scoring")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.7, "axes.labelsize": 11.5,
        "axes.titlesize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(15.4, 11.1), facecolor="white")
    grid = fig.add_gridspec(3, 12, height_ratios=(.93, .90, .88), left=.045,
                           right=.985, bottom=.065, top=.97, wspace=.78, hspace=.42)
    ax_a = fig.add_subplot(grid[0, 0:3]); draw_full_tissue_graph(ax_a, old, FIT_ID, canonical)
    draw_event_reproduction(fig, grid[0, 3:10], out, old, canonical)
    ax_c = fig.add_subplot(grid[0, 10:12]); c_stats = draw_interictal_cohort(ax_c, args.contact_analysis.resolve())
    d_stats = draw_cross_state_fields(fig, grid[1, 0:8], out, canonical)
    e_stats = draw_early_cohort(fig, grid[1, 8:12], out)
    f_stats = draw_mechanism_row(fig, grid[2, 0:12], out)
    cells = (grid[0, 0:3], grid[0, 3:10], grid[0, 10:12], grid[1, 0:8],
             grid[1, 8:12], grid[2, 0:3], grid[2, 3:6], grid[2, 6:9], grid[2, 9:12])
    for label, cell in zip("ABCDEFGHI", cells): grid_letter(fig, cell, label)
    destination = args.out_dir.resolve(); destination.mkdir(parents=True, exist_ok=True)
    stem = destination / "topic5_figure6_multiscale_scaffold_v0_5"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {
        "contract": "topic5_figure6_multiscale_scaffold_v0_5_candidate",
        "representative": SUBJECT, "panel_c": c_stats, "panel_d": d_stats,
        "panel_e": e_stats, "panels_f_i": f_stats,
        "early_ictal_target": "clinical onset 0-10 s mean baseline-robust-z 1-150 Hz broadband energy",
        "target_role": "locked internal cross-state benchmark; never training or selection",
    }
    (destination / "FIGURE6_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assets = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]}
    (destination / "FIGURE6_COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE", "assets_sha256": assets,
    }, indent=2) + "\n")
    (destination / "README.md").write_text(
        "### topic5_figure6_multiscale_scaffold_v0_5.png / .pdf / .svg\n\n"
        "A 在 E1146 真实组织平面上叠加 full-tissue recurrent nodes、局部 backbone、task-selected nonlocal shortcuts 与 SEEG contact readout。B 比较同患者 TA/TB 留出事件和只给第一 rank 后的自由生成。C 是34位患者的间期生成统计。D 为 E1146 冻结 RNN TA/TB fields 与同患者 clinical onset 后0–10 s、1–150 Hz broadband energy field。E 在17位患者/167次发作上显示 signed field correlation 相对同步全通道 shuffle，以及 cross-fitted nonlocality 对 selected-vs-matched cross-state增量的调节。F–I 分别给出 target-free nonlocality interaction、distal controls、arm-specific attenuation 和 TA/TB mode-flow attenuation。\n\n"
        "**关注点**：间期生成、nonlocal specificity、mode-specific flow 与 early-ictal field correspondence 是四层不同结论；只有对应患者级统计通过时才升级该层主张。\n"
    )
    print(json.dumps({"figure": str(stem.with_suffix('.png')), **metadata}, indent=2))


if __name__ == "__main__":
    main()

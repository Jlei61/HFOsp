#!/usr/bin/env python3
"""Figure 6: source-conditioned structured RNN, interictal to early ictal.

Panel geometry and colour follow the established paper-ready idioms rather
than being reinvented here:

* b reuses the Fig-1 rank-heatmap idiom (contacts x events, viridis First to
  Last, grey for a contact that did not participate);
* d reuses the frozen shared contact plane and the Fig-3b field renderer, so
  the model fields and the early-ictal field sit on the same real geometry.

Nothing is recomputed from the ictal target and the representative patient
was fixed before any target value was unsealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_shared_scaffold_rnn_unit_v0_2 import (  # noqa: E402
    load_one_patient_record,
)
from src.topic5_patient_specific_rnn_bridge import (  # noqa: E402
    chronological_60_20_20,
)

REPRESENTATIVE = "epilepsiae_1146"
MODEL_ORDER = ("static", "ordinary_gru", "structured", "structured_rank_shuffle")
MODEL_LABEL = {
    "static": "Static",
    "ordinary_gru": "Dense GRU",
    "structured": "Structured",
    "structured_rank_shuffle": "Shuffled order",
}
MODEL_COLOR = {
    "static": "#9E9E9E",
    "ordinary_gru": "#4D908E",
    "structured": "#B2182B",
    "structured_rank_shuffle": "#E8B4B8",
}
MODE_COLOR = {"minus": "#B2182B", "plus": "#2166AC"}
MODE_LABEL = {"minus": "Mode 1", "plus": "Mode 2"}
N_DISPLAY_EVENTS = 220

plt.rcParams.update(
    {
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "savefig.bbox": "tight",
    }
)


# ------------------------------------------------------------------ panel a
def panel_a(ax):
    """Input column, recurrent contact network, output column."""

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- input column: which contacts fired at this rank step
    active = {1, 3, 4}
    ys = 0.80 - np.arange(6) * 0.125
    ax.scatter(np.full(6, 0.06), ys, s=24, marker="o",
               c=["#444444" if i in active else "white" for i in range(6)],
               edgecolors="#444444", linewidths=0.6, zorder=3)
    ax.text(0.06, 0.90, "Contacts\nat rank $t$", ha="center", va="bottom", fontsize=6)

    # --- recurrent network: neurons ordered along the learned signed axis
    n_units = 7
    xs = np.linspace(0.32, 0.66, n_units)
    offsets = np.array([0.11, -0.06, 0.09, -0.11, 0.07, -0.07, 0.10])
    positions = np.column_stack([xs, 0.47 + offsets])

    # symmetric scaffold: undirected, weight falls with distance along the axis
    for i in range(n_units):
        for j in range(i + 1, n_units):
            weight = float(np.exp(-((xs[j] - xs[i]) / 0.18) ** 2))
            if weight < 0.12:
                continue
            ax.plot(positions[[i, j], 0], positions[[i, j], 1], color="#B0B0B0",
                    linewidth=0.3 + 1.4 * weight, alpha=0.7, zorder=1)
    # signed flow: the single extra structured term, between the two ends only
    for i, j in ((0, 6), (1, 5)):
        ax.add_patch(
            FancyArrowPatch(positions[i], positions[j], arrowstyle="-|>",
                            mutation_scale=6, connectionstyle="arc3,rad=-0.28",
                            linewidth=0.9, color=MODE_COLOR["minus"], zorder=2)
        )
    ax.scatter(positions[:, 0], positions[:, 1], s=52, marker="o",
               c=np.linspace(0, 1, n_units), cmap="coolwarm",
               edgecolors="#333333", linewidths=0.6, zorder=4)
    ax.text(0.49, 0.955, "Recurrent contact network", ha="center", va="bottom", fontsize=6.5)
    ax.annotate("", xy=(0.70, 0.15), xytext=(0.28, 0.15),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.5))
    ax.text(0.49, 0.10, "one learned signed axis", ha="center", va="top",
            fontsize=5.6, color="#777777")
    ax.text(0.485, 0.275, "grey: shared symmetric scaffold", ha="right",
            va="center", fontsize=5.5, color="#777777")
    ax.text(0.505, 0.275, "red: signed flow, sign set by the first rank set",
            ha="left", va="center", fontsize=5.5, color=MODE_COLOR["minus"])

    # --- output column
    labels = ("next contact", "stop", "set size")
    out_ys = 0.66 - np.arange(3) * 0.16
    ax.scatter(np.full(3, 0.86), out_ys, s=24, marker="o", c="white",
               edgecolors="#444444", linewidths=0.6, zorder=3)
    for y, label in zip(out_ys, labels):
        ax.text(0.885, y, label, ha="left", va="center", fontsize=5.8)
    ax.text(0.86, 0.90, "Rank $t\\!+\\!1$", ha="center", va="bottom", fontsize=6)

    for x0, x1 in ((0.10, 0.27), (0.71, 0.83)):
        ax.add_patch(
            FancyArrowPatch((x0, 0.47), (x1, 0.47), arrowstyle="-|>",
                            mutation_scale=7, linewidth=0.7, color="#333333", zorder=3)
        )


# ------------------------------------------------------------------ panel b
def _rank_image(groups: np.ndarray, order: np.ndarray, n_events: int, seed: int):
    """Contacts x events image of within-event rank, grey where absent."""

    if len(groups) > n_events:
        rng = np.random.default_rng(seed)
        groups = groups[np.sort(rng.choice(len(groups), size=n_events, replace=False))]
    image = np.asarray(groups, dtype=float)[:, order].T
    return np.where(image < 0, np.nan, image)


def panel_b(axes, profile_axes, observed, rollout, order, cbar_ax):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    vmax = max(
        np.nanmax(image)
        for side in ("minus", "plus")
        for image in (observed[side]["image"], rollout[side]["image"])
    )
    handle = None
    for row, side in enumerate(("minus", "plus")):
        for column, (source, title) in enumerate(
            ((observed, "Observed held-out events"), (rollout, "Model rollout"))
        ):
            ax = axes[row][column]
            image = source[side]["image"]
            handle = ax.imshow(
                image, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            if column == 0:
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels(source[side]["names"], fontsize=4.4)
                ax.set_ylabel(MODE_LABEL[side], color=MODE_COLOR[side], fontsize=6.5)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=7)
            ax.text(0.99, 0.02, f"n={source[side]['n_events']:,}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2,
                    color="#111111",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))
        # mean rank profile: does the model reproduce this mode's ordering?
        ax = profile_axes[row]
        for source, style, label in (
            (observed, "-", "observed"), (rollout, "--", "model"),
        ):
            profile = source[side]["profile"]
            ax.plot(profile, np.arange(len(profile)), style, color=MODE_COLOR[side],
                    linewidth=0.9, label=label)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=5)
        ax.set_xlabel("mean rank", fontsize=5.8)
        if row == 0:
            ax.legend(frameon=False, fontsize=5, handlelength=1.2, loc="lower right")
    axes[1][0].set_xlabel("Events", fontsize=6.5)
    axes[1][1].set_xlabel("Events", fontsize=6.5)
    bar = plt.colorbar(handle, cax=cbar_ax)
    bar.set_label("First $\\rightarrow$ Last", fontsize=6)
    bar.ax.tick_params(labelsize=5)


# ------------------------------------------------------------------ panel c/e
def _paired(ax, wide, models, ylabel, seed):
    positions = np.arange(len(models), dtype=float)
    rng = np.random.default_rng(seed)
    for _, row in wide.iterrows():
        ax.plot(positions, [row[m] for m in models], color="#CCCCCC",
                linewidth=0.3, alpha=0.7, zorder=1)
    for index, model in enumerate(models):
        values = wide[model].to_numpy(float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(positions[index] + jitter, values, s=5,
                   color=MODEL_COLOR[model], edgecolor="none", zorder=3)
        ax.hlines(np.median(values), positions[index] - 0.24, positions[index] + 0.24,
                  color="#111111", linewidth=1.0, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.set_xlim(-0.5, len(models) - 0.5)


def panel_c(ax, patient: pd.DataFrame, stats: dict):
    models = [m for m in MODEL_ORDER if m in set(patient.model)]
    wide = patient.pivot(index="subject", columns="model", values="contact_nll").dropna(
        subset=models
    )
    _paired(ax, wide, models, "Held-out next-contact NLL", 4)
    ax.set_title(
        f"Interictal prediction, {len(wide)} patients", loc="left", fontweight="bold",
        fontsize=7.5, pad=3,
    )


def panel_e(ax, patient: pd.DataFrame, cohort: dict, supportive: str):
    models = [m for m in ("static", "ordinary_gru", "structured") if m in set(patient.model)]
    wide = patient.pivot(index="subject", columns="model", values="all_contact_margin")
    wide = wide.dropna(subset=models)
    primary = wide.drop(index=supportive, errors="ignore")
    _paired(ax, primary, models, "Correspondence above\nits own shuffled null", 5)
    if supportive in wide.index:
        ax.scatter(np.arange(len(models)), [wide.loc[supportive, m] for m in models],
                   s=12, facecolor="none", edgecolor="#B2182B", linewidth=0.7, zorder=5)
    ax.axhline(0.0, color="#777777", linewidth=0.5, linestyle=":")
    n_primary = cohort["model_statistics"]["structured"]["n_primary_patients"]
    ax.set_title(f"Cross-state correspondence, {n_primary} patients", loc="left",
                 fontweight="bold", fontsize=7.5, pad=3)


# ------------------------------------------------------------------ panel d
def panel_d(axes, cbar_axes, frozen_plane, fields, event_field):
    """Model fields and the early-ictal field on the frozen contact plane."""

    points = np.asarray(frozen_plane["points_mm"], dtype=float)
    support = np.asarray(frozen_plane["support_a"], dtype=float)
    handles = []
    for ax, (title, values, cmap, colour) in zip(axes, fields):
        display = (values - np.nanmin(values)) / max(
            np.nanmax(values) - np.nanmin(values), 1e-12
        )
        x_grid, y_grid, field, _, _ = event_field(frozen_plane, display, support)
        image = ax.imshow(
            field, origin="lower",
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear",
        )
        handles.append(image)
        ax.scatter(points[:, 0], points[:, 1], c=display, cmap=cmap, vmin=0.0, vmax=1.0,
                   s=11, edgecolors="white", linewidths=0.45, zorder=3)
        ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
        ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
        ax.set_title(title, fontsize=6.8, color=colour, fontweight="bold", pad=3)
        ax.set_xlabel("shared axis (mm)", fontsize=6)
        ax.tick_params(labelsize=5)
    axes[0].set_ylabel("transverse (mm)", fontsize=6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    first = plt.colorbar(handles[0], cax=cbar_axes[0])
    first.set_label("model arrival\nlate $\\rightarrow$ early", fontsize=5.6)
    first.set_ticks([])
    second = plt.colorbar(handles[2], cax=cbar_axes[1])
    second.set_label("ictal power\nlow $\\rightarrow$ high", fontsize=5.6)
    second.set_ticks([])


# --------------------------------------------------------------------- data
def build_panel_b_inputs(output: Path, dataset_root: Path, order_by: np.ndarray):
    record = load_one_patient_record(dataset_root, REPRESENTATIVE)
    _, _, test20 = chronological_60_20_20(record)
    freeze = output / "field_freeze" / "per_subject" / REPRESENTATIVE
    with np.load(freeze / "structured_fields.npz", allow_pickle=False) as data:
        pools = {
            side: np.asarray(data[f"source_{side}_indices"], dtype=int)
            for side in ("minus", "plus")
        }
        names = np.asarray(data["contact_names"]).astype(str)
    order = np.argsort(order_by)
    observed_groups = np.asarray(record.group_ids, dtype=np.int64)[np.asarray(test20)]

    def profile(groups):
        values = np.where(groups < 0, np.nan, groups).astype(float)
        with np.errstate(invalid="ignore"):
            return np.nanmean(values[:, order], axis=0)

    observed, rollout = {}, {}
    for side, own in pools.items():
        other = pools["plus" if side == "minus" else "minus"]
        first = observed_groups == 0
        picked = observed_groups[
            first[:, own].sum(axis=1) > first[:, other].sum(axis=1)
        ]
        observed[side] = {
            "image": _rank_image(picked, order, N_DISPLAY_EVENTS, 11),
            "profile": profile(picked),
            "n_events": int(len(picked)),
            "names": names[order],
        }
        stacked = []
        for seed_dir in sorted((freeze / "per_seed" / "structured").glob("seed_*")):
            with np.load(seed_dir / f"{side}.npz", allow_pickle=False) as data:
                stacked.append(np.asarray(data["event_group_ids"], dtype=np.int64))
        generated = np.concatenate(stacked, axis=0)
        rollout[side] = {
            "image": _rank_image(generated, order, N_DISPLAY_EVENTS, 12),
            "profile": profile(generated),
            "n_events": int(len(generated)),
            "names": names[order],
        }
    return observed, rollout, order, names


def median_early_ictal_field(readout: dict, names: np.ndarray):
    target_root = Path(readout["target_cache_root"]).resolve()
    files = sorted((target_root / f"outer_{REPRESENTATIVE}").glob(f"{REPRESENTATIVE}__*.npz"))
    stacked = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            target_names = np.asarray(data["contact_names"]).astype(str)
            values = np.asarray(data[str(readout["target_key"])], dtype=float)
        lookup = dict(zip(target_names, values))
        stacked.append([lookup.get(name, np.nan) for name in names])
    return np.nanmedian(np.asarray(stacked, dtype=float), axis=0)


def directional_field_opposition(output: Path):
    from scipy.stats import spearmanr

    rows = []
    for path in sorted((output / "field_freeze" / "per_subject").glob("*/structured_fields.npz")):
        with np.load(path, allow_pickle=False) as data:
            minus = np.asarray(data["field_minus"], dtype=float)
            plus = np.asarray(data["field_plus"], dtype=float)
        rows.append({"subject": path.parent.name,
                     "rho_minus_plus": float(spearmanr(minus, plus).statistic)})
    values = np.asarray([row["rho_minus_plus"] for row in rows], dtype=float)
    return rows, {
        "n_patients": int(len(rows)),
        "median_rho": float(np.median(values)),
        "n_opposite_below_minus_0p5": int(np.count_nonzero(values < -0.5)),
    }


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from scripts.plot_topic5_interictal_event_envelope_field import (
        _event_field,
        load_frozen,
    )

    training = yaml.safe_load((ROOT / args.training_config).read_text())
    readout = yaml.safe_load((ROOT / args.readout_config).read_text())
    output = (ROOT / args.output_root).resolve()
    dataset_root = Path(training["dataset_artifact_root"]).resolve() / training["dataset_root"]

    patient_interictal = pd.read_csv(output / "interictal_patient_metrics.csv")
    interictal_stats = json.loads((output / "interictal_cohort_statistics.json").read_text())
    patient_ictal = pd.read_csv(output / "early_ictal" / "patient_scores.csv")
    ictal_stats = json.loads((output / "early_ictal" / "cohort_statistics.json").read_text())

    freeze = output / "field_freeze" / "per_subject" / REPRESENTATIVE
    with np.load(freeze / "structured_fields.npz", allow_pickle=False) as data:
        coordinate = np.asarray(data["diffusion_coordinate"], dtype=float)
        field_minus = np.asarray(data["field_minus"], dtype=float)
        field_plus = np.asarray(data["field_plus"], dtype=float)

    observed, rollout, order, names = build_panel_b_inputs(output, dataset_root, coordinate)
    frozen_plane = load_frozen(REPRESENTATIVE)
    if list(map(str, frozen_plane["names"])) != list(names):
        raise RuntimeError("frozen plane and model contact order differ")
    ictal_field = median_early_ictal_field(readout, names)
    opposition = directional_field_opposition(output)

    figure = plt.figure(figsize=(7.09, 8.7))

    a_ax = figure.add_axes([0.055, 0.855, 0.905, 0.125])
    panel_a(a_ax)

    b_left, b_wide, b_gap = 0.105, 0.300, 0.017
    b_rows = ((0.610, 0.185), (0.395, 0.185))
    b_axes, b_profiles = [], []
    for bottom, height in b_rows:
        b_axes.append([
            figure.add_axes([b_left, bottom, b_wide, height]),
            figure.add_axes([b_left + b_wide + b_gap, bottom, b_wide, height]),
        ])
        b_profiles.append(
            figure.add_axes([b_left + 2 * (b_wide + b_gap), bottom, 0.098, height])
        )
    b_cbar = figure.add_axes([0.930, 0.470, 0.011, 0.180])
    panel_b(b_axes, b_profiles, observed, rollout, order, b_cbar)
    b_axes[0][0].annotate(
        f"{REPRESENTATIVE.replace('epilepsiae_', 'E')}   two observed propagation modes",
        xy=(0.0, 1.30), xycoords="axes fraction", fontsize=7.5, fontweight="bold",
        annotation_clip=False,
    )

    d_axes = [figure.add_axes([0.105 + i * 0.205, 0.235, 0.175, 0.115]) for i in range(3)]
    d_cbars = [figure.add_axes([0.735, 0.235, 0.010, 0.115]),
               figure.add_axes([0.805, 0.235, 0.010, 0.115])]
    panel_d(
        d_axes, d_cbars, frozen_plane,
        [
            ("Model field, start 1", field_minus, "viridis_r", MODE_COLOR["minus"]),
            ("Model field, start 2", field_plus, "viridis_r", MODE_COLOR["plus"]),
            ("Early-ictal power", ictal_field, "Blues", "#111111"),
        ],
        _event_field,
    )
    d_axes[0].annotate("Frozen model fields and the early-ictal field",
                       xy=(0.0, 1.13), xycoords="axes fraction", fontsize=7.5,
                       fontweight="bold", annotation_clip=False)
    d_axes[1].text(
        0.5, -0.34,
        f"the two model fields are not opposites: $\\rho$={opposition[1]['median_rho']:+.2f} "
        f"median, below $-0.5$ in {opposition[1]['n_opposite_below_minus_0p5']}/"
        f"{opposition[1]['n_patients']} patients",
        transform=d_axes[1].transAxes, ha="center", va="top", fontsize=5.5,
        color="#B2182B",
    )

    c_ax = figure.add_axes([0.105, 0.040, 0.345, 0.088])
    panel_c(c_ax, patient_interictal, interictal_stats)
    e_ax = figure.add_axes([0.615, 0.040, 0.280, 0.088])
    panel_e(e_ax, patient_ictal, ictal_stats, str(readout["supportive_subject"]))

    for label, axis, offset in (
        ("a", a_ax, (-0.045, 1.00)), ("b", b_axes[0][0], (-0.235, 1.30)),
        ("c", c_ax, (-0.185, 1.34)), ("d", d_axes[0], (-0.290, 1.13)),
        ("e", e_ax, (-0.230, 1.34)),
    ):
        axis.annotate(label, xy=offset, xycoords="axes fraction", fontsize=9,
                      fontweight="bold", annotation_clip=False)

    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / "topic5_figure6_source_conditioned_rnn"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    patient_interictal.to_csv(figures_dir / "figure6_panelC_source_data.csv", index=False)
    patient_ictal.to_csv(figures_dir / "figure6_panelE_source_data.csv", index=False)
    pd.DataFrame(
        {
            "contact": names,
            "learned_axis_coordinate": coordinate,
            "model_field_start1": field_minus,
            "model_field_start2": field_plus,
            "early_ictal_median": ictal_field,
        }
    ).to_csv(figures_dir / "figure6_panelD_source_data.csv", index=False)
    pd.DataFrame(
        [
            {"mode": MODE_LABEL[side], "source": source,
             "n_events": payload[side]["n_events"],
             **{f"mean_rank__{name}": value
                for name, value in zip(payload[side]["names"], payload[side]["profile"])}}
            for side in ("minus", "plus")
            for source, payload in (("observed", observed), ("model_rollout", rollout))
        ]
    ).to_csv(figures_dir / "figure6_panelB_source_data.csv", index=False)

    statistics = {
        "representative_subject": REPRESENTATIVE,
        "representative_fixed_before_target_unseal": True,
        "panel_b_event_counts": {
            MODE_LABEL[side]: {"observed": observed[side]["n_events"],
                               "model_rollout": rollout[side]["n_events"]}
            for side in ("minus", "plus")
        },
        "panel_c": interictal_stats,
        "panel_d_directional_field_opposition": {
            "summary": opposition[1], "per_patient": opposition[0],
            "note": "measured on the emitted fields, not assumed from the direction state",
        },
        "panel_e": ictal_stats,
        "not_implemented": [
            "rollout-vs-test20 participation / pairwise precedence / expected-rank "
            "distance consistency statistics are not computed by any current script"
        ],
    }
    (figures_dir / "figure6_statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False, default=float) + "\n"
    )

    counts = statistics["panel_b_event_counts"]
    opposition_summary = opposition[1]
    (figures_dir / "README.md").write_text(
        "# Figure 6 图说明\n\n"
        "### topic5_figure6_source_conditioned_rnn.png / .pdf / .svg\n\n"
        "a 是模型结构：左边是这一步观察到哪些触点在放电，中间是触点之间的循环网络"
        "（灰线是所有触点共用的对称连接，红线是唯一那条有方向的连接，它的正负号由每场事件"
        "最先放电的那几个触点决定），右边是模型每一步要输出的三件事。\n\n"
        f"b 固定用 {REPRESENTATIVE.replace('epilepsiae_', 'E')}，上下两行是模型自己认定的"
        "两个起点端。每一列是一场事件，颜色是该触点在这场事件里第几个放电（深紫最早、黄最晚），"
        "灰色表示这场事件里它没参与。左列是留出集里真实观察到的事件，右列是冻结模型自主推演的事件，"
        "最右是两者的平均先后次序曲线（实线观察、虚线模型）。\n\n"
        "d 把两张冻结的模型场和同一患者发作早期的宽带能量场画在**同一套真实电极几何**上"
        "（与既有的间期-发作共享场图同一个平面、同一套插值），圆点是真实触点位置。\n\n"
        "c 与 e 是队列统计，只放数据不放解释文字。\n\n"
        "**读图时必须知道的三件事**：\n\n"
        f"1. b 右列模型事件顶部/底部那条纯深色带是**被我们强制指定的起点**"
        f"（每次推演都从同一组触点出发），不是模型自己学出来的；观察事件没有这个约束。"
        f"所以两列不能按逐格对应去读，只能比中间部分的走向。\n"
        f"2. 模型确实产出了两套不同的推演（上下两行明显不同），但那**主要是因为我们给了不同起点**，"
        f"不等于模型自发学到了两种模式。\n"
        f"3. d 里两张模型场实测**高度相似而非相反**"
        f"（全队列秩相关中位 {opposition_summary['median_rho']:+.2f}，"
        f"真正相反的只有 {opposition_summary['n_opposite_below_minus_0p5']}/"
        f"{opposition_summary['n_patients']} 人），不要按「两个相反方向场」去读。\n\n"
        "**关注点**：b 中间部分观察与模型的走向是否一致（实线与虚线是否贴合）；"
        "c 中结构化模型与打乱顺序对照是否分开；"
        "e 中三个模型谁高于各自的随机基线——实测最简单的静态基线最高，"
        "结构化模型没有把方向信息转化成跨状态优势。\n\n"
        f"事件数：模式 1 观察 {counts['Mode 1']['observed']:,} / 模型 "
        f"{counts['Mode 1']['model_rollout']:,}；模式 2 观察 "
        f"{counts['Mode 2']['observed']:,} / 模型 {counts['Mode 2']['model_rollout']:,}。\n"
    )
    print(json.dumps({"status": "COMPLETE", "figure": str(stem.with_suffix(".png"))}))


if __name__ == "__main__":
    main()

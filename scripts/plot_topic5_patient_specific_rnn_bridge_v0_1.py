#!/usr/bin/env python3
"""Paper-ready six-panel figure for the patient-specific RNN bridge."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLORS = {
    "full_history_gru": "#2F6B9A",
    "linear_state": "#5AAE61",
    "rank_shuffle_gru": "#BDBDBD",
    "static_fit60": "#E69F00",
    "empirical_test20": "#222222",
}
LABELS = {
    "full_history_gru": "Patient-only GRU",
    "linear_state": "Linear state",
    "rank_shuffle_gru": "Rank-shuffle GRU",
    "static_fit60": "Static fit60",
    "empirical_test20": "Empirical test20",
}


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def normalized_rank(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    return (ranked - 1) / max(len(ranked) - 1, 1)


def plot_design(ax) -> None:
    ax.axis("off")
    boxes = [
        (0.02, 0.56, 0.27, 0.28, "Patient's interictal\ncontact-rank events", "#EAF2F8"),
        (0.37, 0.56, 0.25, 0.28, "Self-supervised\npatient-only RNN", "#DDEEDB"),
        (0.70, 0.56, 0.27, 0.28, "RNN-derived\ncontact fields", "#FCE8C3"),
        (0.37, 0.08, 0.25, 0.25, "Same-patient early-ictal\n1–150 Hz field", "#F4D6D6"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#555555", linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    for start, end in [((0.29, 0.70), (0.37, 0.70)), ((0.62, 0.70), (0.70, 0.70)), ((0.835, 0.56), (0.62, 0.30))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    ax.text(0.02, 0.39, "Training: fit60 → validation20 → untouched test20", fontsize=8, color="#333333")
    ax.text(0.02, 0.27, "No other patients, A/B labels, SOZ, or ictal target", fontsize=8, color="#8B1A1A")
    ax.text(0.02, 0.15, "Ictal data enter only after checkpoint + rollout freeze", fontsize=8, color="#333333")


def paired_model_plot(ax, frame, metric, models, ylabel) -> None:
    wide = frame.pivot(index="subject", columns="model", values=metric).dropna(subset=models)
    x = np.arange(len(models))
    for _, row in wide.iterrows():
        ax.plot(x, row[models], color="#BBBBBB", alpha=0.55, linewidth=0.7, zorder=1)
        ax.scatter(x, row[models], c=[COLORS[m] for m in models], s=16, zorder=2, edgecolor="white", linewidth=0.3)
    medians = [wide[m].median() for m in models]
    ax.plot(x, medians, color="#111111", linewidth=2.0, marker="o", markersize=5, zorder=3)
    ax.set_xticks(x, [LABELS[m].replace(" ", "\n", 1) for m in models], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def representative_rows(output: Path, cache_root: Path, subject: str, seeds: list[int]):
    model_fields = {}
    names = None
    for model in ("full_history_gru", "rank_shuffle_gru"):
        rows = []
        for seed in seeds:
            with np.load(output / "units" / subject / model / f"seed_{seed}/free_rollouts.npz", allow_pickle=False) as data:
                current = np.asarray(data["contact_names"]).astype(str)
                if names is None:
                    names = current
                elif not np.array_equal(names, current):
                    raise RuntimeError("representative contact order drift")
                rows.append(np.asarray(data["field__weighted_earliness"], float))
        model_fields[model] = np.median(np.stack(rows), axis=0)
    with np.load(output / "units" / subject / "full_history_gru" / f"seed_{seeds[0]}/empirical_references.npz", allow_pickle=False) as data:
        participation = np.asarray(data["test_participation"], float)
        mean_rank = np.asarray(data["test_mean_rank"], float)
        empirical = participation * (1.0 - np.where(np.isfinite(mean_rank), mean_rank, 0.5))
    target_rows = []
    target_names = None
    for path in sorted((cache_root / f"outer_{subject}").glob(f"{subject}__*.npz")):
        with np.load(path, allow_pickle=False) as data:
            current = np.asarray(data["contact_names"]).astype(str)
            values = np.asarray(data["target_1_150"], float)
        if target_names is None:
            target_names = current
        if np.array_equal(current, target_names):
            target_rows.append(values)
    target = np.median(np.stack(target_rows), axis=0)
    lookup = {name: index for index, name in enumerate(names)}
    keep = np.asarray([lookup[name] for name in target_names], int)
    rows = np.row_stack([
        normalized_rank(empirical[keep]),
        normalized_rank(model_fields["full_history_gru"][keep]),
        normalized_rank(model_fields["rank_shuffle_gru"][keep]),
        normalized_rank(target),
    ])
    order = np.argsort(rows[0])
    return target_names[order], rows[:, order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_root"]
    interictal = pd.read_csv(output / "interictal_patient_metrics.csv")
    ictal = pd.read_csv(output / "early_ictal_patient_metrics.csv")
    primary = ictal.loc[(ictal.band == "1_150") & ~ictal.development_supportive.astype(bool)].copy()

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(12.0, 7.1), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.08, 1.0, 1.0])
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]

    plot_design(axes[0]); panel_label(axes[0], "A"); axes[0].set_title("Patient-specific target-free bridge")
    paired_model_plot(
        axes[1], interictal, "test_nll",
        ["rank_shuffle_gru", "linear_state", "full_history_gru"],
        "Held-out event NLL (lower is better)",
    )
    panel_label(axes[1], "B"); axes[1].set_title("True within-event order improves prediction")
    paired_model_plot(
        axes[2], interictal, "precedence_correlation",
        ["rank_shuffle_gru", "linear_state", "full_history_gru"],
        "Generated vs observed precedence, Spearman ρ",
    )
    axes[2].axhline(0, color="#777777", lw=0.8); panel_label(axes[2], "C")
    axes[2].set_title("RNN rollouts recover contact ordering")

    names, rows = representative_rows(
        output, ROOT / config["target_cache_root"], "epilepsiae_1084",
        list(map(int, config["training"]["seeds"])),
    )
    im = axes[3].imshow(rows, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[3].set_yticks(np.arange(4), ["Observed test20", "Patient-only GRU", "Rank-shuffle GRU", "Early-ictal 1–150 Hz"], fontsize=7)
    axes[3].set_xticks(np.arange(len(names)), names, rotation=70, fontsize=6)
    axes[3].set_xlabel("Contacts ordered by observed interictal earliness")
    axes[3].set_title("Representative same-patient field (E1084)")
    panel_label(axes[3], "D"); figure.colorbar(im, ax=axes[3], fraction=0.035, pad=0.02, label="Within-row contact rank")

    paired_model_plot(
        axes[4], primary, "observed_max_abs_rho",
        ["static_fit60", "rank_shuffle_gru", "linear_state", "full_history_gru"],
        "Early-ictal max |Spearman ρ|",
    )
    panel_label(axes[4], "E"); axes[4].set_title("Same-patient early-ictal correspondence")

    wide = primary.pivot(index="subject", columns="model", values="all_contact_margin").dropna()
    axes[5].axhline(0, color="#777777", lw=0.8)
    axes[5].axvline(0, color="#777777", lw=0.8)
    limit = max(0.15, float(np.nanmax(np.abs(wide[["static_fit60", "full_history_gru"]].to_numpy()))) * 1.1)
    axes[5].plot([-limit, limit], [-limit, limit], ls="--", color="#999999", lw=0.8)
    axes[5].scatter(wide.static_fit60, wide.full_history_gru, s=28, color=COLORS["full_history_gru"], edgecolor="white", linewidth=0.4)
    axes[5].set_xlim(-limit, limit); axes[5].set_ylim(-limit, limit)
    axes[5].set_xlabel("Static fit60 margin vs all-contact null")
    axes[5].set_ylabel("Patient-only GRU margin vs all-contact null")
    axes[5].grid(alpha=0.2); axes[5].spines[["top", "right"]].set_visible(False)
    axes[5].set_title("Patient heterogeneity and GRU increment")
    panel_label(axes[5], "F")

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "patient_specific_target_free_rnn_bridge_six_panel.png"
    pdf = figure_dir / "patient_specific_target_free_rnn_bridge_six_panel.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    (figure_dir / "README.md").write_text(
        """### patient_specific_target_free_rnn_bridge_six_panel.png

六联图按科学问题排列：A 明确患者内、自监督且 target-free 的训练合同；B 检验真实事件内顺序是否改善 held-out 预测；C 检验模型自由生成是否保留真实 contact 先后关系；D 展示 E1084 的真实间期、GRU、rank-shuffle 与发作早期场；E 展示 15 名 primary 患者的跨状态绝对对应；F 展示 GRU 与静态 fit60 相对 all-contact null 的患者异质性。

**关注点**：B/C 回答 RNN 是否学到间期传播结构，E/F 回答这一模型结构是否与同一患者发作早期场联系；两者必须分开解读。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

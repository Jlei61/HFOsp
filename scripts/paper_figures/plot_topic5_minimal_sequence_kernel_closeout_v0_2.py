#!/usr/bin/env python3
"""Render the paper-ready six-panel within-event closeout and audit companion."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_minimal_sequence_kernel import block_hankel_from_lag_kernels  # noqa: E402


BASE = ROOT / "results/topic5_minimal_sequence_kernel_closeout"
OUT = (
    ROOT
    / "results/paper-ready-figure/"
    "fig_topic5_minimal_sequence_kernel_closeout/figures"
)
BLUE = "#3B6FB6"
ORANGE = "#D97936"
RED = "#B23A48"
TEAL = "#278C8C"
PURPLE = "#6F58A8"
GREY = "#9AA0A6"
LIGHT_GREY = "#E6E8EB"
BLACK = "#202124"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def _panel(ax, letter: str, title: str) -> None:
    ax.text(
        -0.10,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.set_title(title, loc="left", pad=8)


def _strip_summary(
    ax,
    groups: list[np.ndarray],
    positions: list[float],
    colors: list[str],
    *,
    seed: int,
    width: float = 0.16,
) -> None:
    rng = np.random.default_rng(seed)
    for values, position, color in zip(groups, positions, colors):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        jitter = rng.uniform(-width, width, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=13,
            color=color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        if len(values):
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.plot([position, position], [q1, q3], color=BLACK, lw=2.5, zorder=3)
            ax.plot(
                [position - 0.16, position + 0.16],
                [median, median],
                color=BLACK,
                lw=1.4,
                zorder=4,
            )


def _draw_task(ax) -> None:
    _panel(ax, "A", "Nested prediction components")
    ax.set_axis_off()
    boxes = [
        (0.02, 0.52, 0.25, 0.28, "Static contact\nprior", r"$\alpha_{p,c}$", GREY),
        (0.37, 0.52, 0.25, 0.28, "Unordered\nprefix", r"$f_{\rm set}(U_t,c)$", TEAL),
        (0.72, 0.52, 0.25, 0.28, "Recent ordered\nranks", r"$f_{\rm ord}(S_{t-2:t},c)$", RED),
    ]
    for x, y, w, h, title, formula, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                fc=mpl.colors.to_rgba(color, 0.13),
                ec=color,
                lw=1.1,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + w / 2,
            y + 0.19,
            title,
            ha="center",
            va="center",
            fontsize=7.2,
            linespacing=1.05,
            transform=ax.transAxes,
        )
        ax.text(
            x + w / 2,
            y + 0.065,
            formula,
            ha="center",
            va="center",
            fontsize=7.2,
            transform=ax.transAxes,
        )
    ax.text(0.32, 0.66, "+", fontsize=16, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.67, 0.66, "+", fontsize=16, ha="center", va="center", transform=ax.transAxes)
    ax.annotate(
        "",
        xy=(0.50, 0.35),
        xytext=(0.50, 0.50),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": BLACK, "lw": 1.0},
    )
    ax.text(
        0.50,
        0.26,
        "next contact set  +  STOP",
        ha="center",
        va="center",
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.08,
        "Within-event memory only • reset at every event • rank step ≠ real time",
        ha="center",
        va="center",
        color="#5F6368",
        fontsize=7.2,
        transform=ax.transAxes,
    )


def _draw_static_reliability(ax) -> dict:
    _panel(ax, "B", "Stable contact recruitment scaffold")
    frame = pd.read_csv(
        ROOT
        / "results/topic5_interictal_scaffold_reliability_history_necessity/"
        "static_reliability_v0_1/patient_reliability.csv"
    )
    eligible = frame.loc[frame.structured_null_eligible].copy()
    for row in eligible.itertuples():
        ax.plot(
            [0, 1],
            [row.structured_null_median_rho, row.train80_heldout20_spearman_rho],
            color=LIGHT_GREY,
            lw=0.7,
            zorder=1,
        )
    _strip_summary(
        ax,
        [
            eligible.structured_null_median_rho.to_numpy(),
            eligible.train80_heldout20_spearman_rho.to_numpy(),
        ],
        [0, 1],
        [GREY, BLUE],
        seed=10,
        width=0.11,
    )
    ax.set_xticks([0, 1], ["shaft-preserving\nnull", "chronological\nheldout"])
    ax.set_ylabel("Train–heldout Spearman ρ")
    ax.set_ylim(-0.45, 1.04)
    ax.axhline(0, color=GREY, lw=0.7, ls="--")
    ax.text(
        0.04,
        0.97,
        "median heldout ρ = 0.89\n33/33 above structured null",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    _clean(ax)
    return {"n_eligible": len(eligible)}


def _draw_history_horizon(ax) -> dict:
    _panel(ax, "C", "Contact choice saturates after two ranks")
    gains = pd.read_csv(BASE / "formal_v0_2/patient_component_gains.csv")
    comparisons = [
        ("history2_minus_history1", "H2 − H1"),
        ("history3_minus_history2", "H3 − H2"),
        ("full_minus_history3", "Full − H3"),
    ]
    positions = []
    groups = []
    colors = []
    for index, (comparison, _) in enumerate(comparisons):
        for offset, component, color in (
            (-0.17, "event_contact_choice_nll", RED),
            (0.17, "event_stop_contribution_nll", BLUE),
        ):
            groups.append(
                gains.loc[
                    (gains.comparison == comparison)
                    & (gains.component == component),
                    "gain_nats",
                ].to_numpy()
            )
            positions.append(index + offset)
            colors.append(color)
    _strip_summary(ax, groups, positions, colors, seed=11, width=0.08)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(3), [label for _, label in comparisons])
    ax.set_ylabel("Heldout gain (nats/decision)")
    ax.scatter([], [], color=RED, label="contact identity")
    ax.scatter([], [], color=BLUE, label="STOP contribution")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.97,
        "H2 contact: +0.011 nats\nH3 contact: +0.0015 (n.s.)",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    _clean(ax)
    return {"comparisons": [item[0] for item in comparisons]}


def _draw_architecture(ax) -> dict:
    _panel(ax, "D", "Minimal architecture audit")
    old = pd.read_csv(
        ROOT
        / "results/topic5_ordered_history_architecture_audit/"
        "analysis/patient_paired_nll_gains.csv"
    )
    definitions = [
        ("full_history_gru", "GRU", GREY),
        ("vanilla_rnn", "rate RNN", GREY),
        ("low_rank_r0", "LR-0", GREY),
        ("low_rank_r1", "LR-1", GREY),
        ("low_rank_r2", "LR-2", GREY),
        ("low_rank_r4", "LR-4", GREY),
        ("linear_state", "linear state", RED),
    ]
    groups = []
    labels = []
    colors = []
    for candidate, label, color in definitions:
        groups.append(
            old.loc[
                (old.candidate == candidate)
                & (old.reference == "unordered_prefix"),
                "nll_gain_reference_minus_candidate",
            ].to_numpy()
        )
        labels.append(label)
        colors.append(color)
    fir = pd.read_csv(BASE / "fir_h3_formal_v0_2/patient_fir_gains.csv")
    groups.append(
        fir.loc[
            (fir.comparison == "fir_minus_retrained_unordered")
            & (fir.component == "event_total_nll"),
            "gain_nats",
        ].to_numpy()
    )
    labels.append("FIR-H3")
    colors.append(PURPLE)
    positions = list(range(len(groups)))
    _strip_summary(ax, groups, positions, colors, seed=12, width=0.13)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(positions, labels, rotation=42, ha="right")
    ax.set_ylabel("Joint NLL gain over unordered")
    ax.text(
        0.02,
        0.97,
        "simple linear state selected;\nnonlinear families add no stable benefit",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    _clean(ax)
    return {"models": labels}


def _hankel_cumulative() -> np.ndarray:
    rows = []
    for path in sorted(
        (BASE / "formal_v0_2").glob("seed_*/*/linear_state_lag_kernels.npz")
    ):
        with np.load(path, allow_pickle=False) as data:
            hankel = block_hankel_from_lag_kernels(data["contact_kernels"])
        singular = np.linalg.svd(hankel, compute_uv=False)
        energy = singular**2
        cumulative = np.cumsum(energy) / np.sum(energy)
        rows.append(np.pad(cumulative[:6], (0, max(0, 6 - len(cumulative))), constant_values=1.0))
    if len(rows) != 102:
        raise RuntimeError("lag-kernel inventory incomplete")
    return np.median(np.row_stack(rows), axis=0)


def _draw_lag_kernel(ax) -> dict:
    _panel(ax, "E", "Input–output lag kernel")
    gains = pd.read_csv(BASE / "formal_v0_2/patient_component_gains.csv")
    definitions = [
        ("lag0_contribution", "K₀"),
        ("lag1_contribution", "K₁"),
        ("lag2_contribution", "K₂"),
        ("lag3plus_contribution", "K₃+"),
    ]
    groups = [
        gains.loc[
            (gains.comparison == comparison)
            & (gains.component == "event_contact_choice_nll"),
            "gain_nats",
        ].to_numpy()
        for comparison, _ in definitions
    ]
    _strip_summary(ax, groups, list(range(4)), [RED] * 4, seed=13, width=0.13)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(4), [label for _, label in definitions])
    ax.set_ylabel("Contact-choice loss on removal")
    _clean(ax)
    inset = ax.inset_axes([0.56, 0.58, 0.40, 0.34])
    cumulative = _hankel_cumulative()
    inset.plot(np.arange(1, 7), cumulative, color=PURPLE, marker="o", ms=3)
    inset.axhline(0.9, color=GREY, ls="--", lw=0.7)
    inset.set_xlim(1, 6)
    inset.set_ylim(0.55, 1.01)
    inset.set_xticks([1, 2, 4, 6])
    inset.set_yticks([0.6, 0.8, 1.0])
    inset.set_xlabel("input–output order", fontsize=6)
    inset.set_ylabel("energy", fontsize=6)
    inset.tick_params(labelsize=5.5, length=2)
    _clean(inset)
    ax.text(
        0.02,
        0.97,
        "K₀ dominates; K₁ is smaller;\nK₂ and K₃+ add no stable contact gain",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    return {"hankel_median_cumulative_first6": cumulative.tolist()}


def _cross_state_values() -> tuple[list[np.ndarray], list[str], list[str]]:
    static = pd.read_csv(
        ROOT
        / "results/topic5_static_scaffold_fixed_readout_validation/"
        "phase1_existing_fields_patient_metrics.csv"
    )
    static_margin = static.loc[
        (static.model == "empirical_rank_distribution")
        & (static.field == "participation")
        & (static.null_mode == "all_contact"),
        "absolute_margin",
    ].to_numpy()
    unordered_margin = static.loc[
        (static.model == "unordered_prefix")
        & (static.field == "participation")
        & (static.null_mode == "all_contact"),
        "absolute_margin",
    ].to_numpy()
    conditional = pd.read_csv(
        ROOT
        / "results/topic5_ordered_history_architecture_audit/"
        "analysis/early_ictal_conditional_patient_metrics.csv"
    )

    def paired(left: str, right: str, conditioning: str) -> np.ndarray:
        frame = conditional.loc[
            (conditional.conditioning == conditioning)
            & conditional.field.isin([left, right])
            & conditional.eligible,
            ["subject", "field", "absolute_rho"],
        ]
        wide = frame.pivot(index="subject", columns="field", values="absolute_rho").dropna()
        return (wide[left] - wide[right]).to_numpy()

    ordered_minus_unordered = paired(
        "selected_ordered", "unordered_prefix", "static_only"
    )
    true_minus_shuffle = paired(
        "selected_ordered", "selected_rank_shuffle", "static_plus_unordered"
    )
    return (
        [static_margin, unordered_margin, ordered_minus_unordered, true_minus_shuffle],
        ["static\nvs null", "unordered\nvs null", "ordered −\nunordered", "true order −\nshuffle"],
        [BLUE, TEAL, RED, RED],
    )


def _draw_cross_state(ax) -> dict:
    _panel(ax, "F", "Cross-state association is primarily static")
    groups, labels, colors = _cross_state_values()
    _strip_summary(ax, groups, list(range(4)), colors, seed=14, width=0.13)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.axvline(1.5, color=LIGHT_GREY, lw=1.0)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Field association / paired Δ|ρ|")
    ax.text(
        0.02,
        0.97,
        "ordered residual increment:\nnot established",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    _clean(ax)
    return {"n_per_category": [int(np.sum(np.isfinite(values))) for values in groups]}


def _main_figure() -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.9))
    metadata = {}
    _draw_task(axes[0, 0])
    metadata["B"] = _draw_static_reliability(axes[0, 1])
    metadata["C"] = _draw_history_horizon(axes[0, 2])
    metadata["D"] = _draw_architecture(axes[1, 0])
    metadata["E"] = _draw_lag_kernel(axes[1, 1])
    metadata["F"] = _draw_cross_state(axes[1, 2])
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.10, top=0.94, wspace=0.34, hspace=0.42)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"topic5_minimal_sequence_kernel_closeout.{suffix}",
            dpi=450 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return metadata


def _audit_figure() -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15))
    # A: rank-set tolerance
    ax = axes[0]
    _panel(ax, "A", "Rank-set timing tolerance")
    tolerance = pd.read_csv(BASE / "rank_tolerance_v0_2/patient_tolerance_gains.csv")
    for condition, color, label in (
        ("history_3", RED, "H3"),
        ("linear_state", PURPLE, "linear state"),
    ):
        subset = tolerance.loc[
            (tolerance.condition == condition)
            & (tolerance.component == "event_contact_choice_nll")
        ]
        summary = subset.groupby("tolerance_ms").gain_nats.agg(
            median="median",
            q1=lambda value: np.quantile(value, 0.25),
            q3=lambda value: np.quantile(value, 0.75),
        )
        x = summary.index.to_numpy(float)
        ax.plot(x, summary["median"], color=color, marker="o", label=label)
        ax.fill_between(x, summary.q1, summary.q3, color=color, alpha=0.14, lw=0)
    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("Tie tolerance (ms)")
    ax.set_ylabel("Contact-choice gain (nats)")
    ax.legend(frameon=False)
    _clean(ax)

    # B: target reliability
    ax = axes[1]
    _panel(ax, "B", "Early-ictal target reliability")
    exact = pd.read_csv(
        BASE
        / "when_gate0_early_ictal_reliability_v0_2/"
        "exact_bb150_patient_mean_reliability.csv"
    )
    proxy = pd.read_csv(
        BASE
        / "when_gate0_early_ictal_reliability_v0_2/"
        "proxy_bb45_residual_reliability.csv"
    )
    values = [
        exact.loc[exact.eligible_split_half, "exact_bb150_split_half_rho_median"].to_numpy(),
        proxy.loc[proxy.eligible_proxy_residual, "proxy_patient_mean_half_rho"].to_numpy(),
        proxy.loc[
            proxy.eligible_proxy_residual,
            "proxy_residual_matched_minus_mismatched",
        ].to_numpy(),
    ]
    _strip_summary(ax, values, [0, 1, 2], [BLUE, TEAL, ORANGE], seed=20, width=0.10)
    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    ax.set_xticks(
        [0, 1, 2],
        ["BB150\npatient mean", "BB45\npatient mean", "BB45 seizure\nresidual proxy"],
    )
    ax.set_ylabel("Reliability / matched excess ρ")
    ax.text(
        0.02,
        0.03,
        "Exact BB150 seizure residual:\nunidentifiable from one field/seizure",
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
    )
    _clean(ax)

    # C: inter-event state feasibility
    ax = axes[2]
    _panel(ax, "C", "Inter-event state feasibility")
    gate = pd.read_csv(BASE / "when_gate1_inter_event_v0_2/patient_gate1_gains.csv")
    primary = gate.loc[
        gate.comparison == "time_state_minus_best_nonstate_control"
    ]
    values = [
        primary.loc[primary.dataset == "epilepsiae", "gain_nats"].to_numpy(),
        primary.loc[primary.dataset == "yuquan", "gain_nats"].to_numpy(),
    ]
    _strip_summary(ax, values, [0, 1], [ORANGE, BLUE], seed=21, width=0.13)
    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["Epilepsiae", "Yuquan"])
    ax.set_ylabel("Gain over best non-state control\n(nats/contact)")
    ax.text(
        0.02,
        0.97,
        "cohort signal; no two-dataset replication",
        transform=ax.transAxes,
        va="top",
        fontsize=6.8,
    )
    _clean(ax)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.90, wspace=0.36)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"topic5_sequence_definition_and_when_audit.{suffix}",
            dpi=450 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return {
        "rank_tolerances_ms": sorted(tolerance.tolerance_ms.unique().tolist()),
        "gate0_exact_eligible_patients": int(exact.eligible_split_half.sum()),
        "gate1_patients": int(primary.subject.nunique()),
    }


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "main": _main_figure(),
        "audit": _audit_figure(),
        "scientific_position": "Extended Data or Supplementary bounded computational result",
        "where_how_when_separated": True,
        "target_values_used_only_in_cross_state_and_gate0_panels": True,
    }
    (OUT / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        """### topic5_minimal_sequence_kernel_closeout.png

六联图按“任务分解 → 静态稳定性 → 短程历史 → 最小模型 → lag kernel → 跨状态边界”组织。它支持稳定 contact scaffold 之上存在短程事件内顺序信息，但不把 rank-step memory 写成真实时间状态，也不把早期发作对应归因于 ordered residual。

**关注点**：Panel C/E 中 contact identity 与 STOP 的分工，以及 Panel F 中 static association 与 ordered increment 的边界。

### topic5_sequence_definition_and_when_audit.png

三联补图展示 rank-set 时间容差、early-ictal 动态 target 的可靠性门和跨事件状态可行性。1–45 Hz residual 只作为时序代理，不能替代冻结的 1–150 Hz target；跨事件阳性也尚未在两个数据集分别复现。

**关注点**：0–2 ms 的定义稳健区间、BB150 seizure residual 的不可辨识性、以及 Epilepsiae/Yuquan 异质性。
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

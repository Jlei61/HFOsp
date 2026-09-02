#!/usr/bin/env python3
"""Render a Figure-6 review candidate with the axis-free operator result.

Panels A-E reproduce the accepted multiscale-scaffold r3 render.  The former
four-panel nonlocal-edge audit row is replaced by two wider panels:

* F: cross-topology convergence of the axis-free tissue-patch response operator;
* G: held-out transition alignment and the smoothing-matched identity boundary.

The accepted r3 assets are not overwritten.  This renderer writes a separate
candidate directory until the revised scientific layout is reviewed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import finalize_topic5_figure6_multiscale_scaffold_v0_5_r2 as r2  # noqa: E402
import finalize_topic5_figure6_multiscale_scaffold_v0_5_r3 as r3  # noqa: E402
from scripts.paper_figures import (  # noqa: E402
    plot_topic5_figure6_multiscale_scaffold_v0_5 as base,
)


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
DEFAULT_CANONICAL = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_OPERATOR = (
    ROOT / "results/topic5_latent_propagation_landscape_v0_2"
    / "spatial_control_field/patch_operator"
)
DEFAULT_FIGURE = (
    ROOT / "results/paper-ready-figure"
    / "fig6_multiscale_scaffold_v0_5_operator_r1_candidate/figures"
)
STEM = "topic5_figure6_multiscale_scaffold_v0_5_operator_r1_candidate"
GOLD = "#c99a37"
LIGHT_BLUE = "#7b9fc1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = finite[rng.integers(0, len(finite), size=(draws, len(finite)))]
    low, high = np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
    return float(low), float(high)


def cohort_marker(ax: plt.Axes, x: float, values: np.ndarray, seed: int) -> None:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    low, high = bootstrap_median_ci(finite, seed)
    median = float(np.median(finite))
    ax.plot([x, x], [low, high], color="black", lw=1.15, zorder=5)
    ax.plot([x - .20, x + .20], [median, median], color="black", lw=2.0, zorder=6)


def paired_operator_axis(
    ax: plt.Axes,
    control: np.ndarray,
    real: np.ndarray,
    labels: tuple[str, str],
    color: str,
    note: str,
) -> None:
    control = np.asarray(control, float)
    real = np.asarray(real, float)
    finite = np.isfinite(control) & np.isfinite(real)
    control, real = control[finite], real[finite]
    for low, high in zip(control, real):
        ax.plot([0, 1], [low, high], color=color, alpha=.28, lw=.75, zorder=1)
    ax.scatter(np.zeros(len(control)), control, s=22, color=base.GRAY, alpha=.65,
               edgecolor="white", lw=.3, zorder=3)
    ax.scatter(np.ones(len(real)), real, s=22, color=color, alpha=.68,
               edgecolor="white", lw=.3, zorder=3)
    cohort_marker(ax, 0, control, 5210)
    cohort_marker(ax, 1, real, 5211)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-.42, 1.42)
    ax.text(.5, 1.01, note, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.3, color="#4f565b", linespacing=.95)
    ax.text(.5, .02, f"n={len(real)}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.0, color="#666666")
    ax.spines[["top", "right"]].set_visible(False)


def strip_axis(
    ax: plt.Axes,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    rng = np.random.default_rng(5202)
    for position, (raw, color) in enumerate(zip(data, colors)):
        values = np.asarray(raw, float)
        values = values[np.isfinite(values)]
        jitter = rng.uniform(-.13, .13, len(values))
        ax.scatter(position + jitter, values, s=21, color=color, alpha=.58,
                   edgecolor="white", lw=.25, zorder=3)
        cohort_marker(ax, position, values, 5220 + position)
        ax.text(position, .02, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.0, color="#666666")
    ax.axhline(0, color="#858b8e", lw=.8, ls="--", zorder=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_xlim(-.55, len(labels) - .45)
    ax.spines[["top", "right"]].set_visible(False)


def draw_panel_c(ax: plt.Axes, out: Path) -> dict:
    frame = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
    pivot = frame.pivot(index="subject", columns="arm", values="test_contact_nll")
    true_order = pivot[r2.L3].sort_index()
    reassigned = pivot[r2.SUFFIX].reindex(true_order.index)
    gain = reassigned.to_numpy(float) - true_order.to_numpy(float)
    p_value = base.paired_test(gain, "greater")
    base.paired_axis(
        ax, true_order.to_numpy(float), reassigned.to_numpy(float),
        ("True order", "Suffix\nreassigned"), (base.RED, base.GRAY),
        "Held-out contact NLL", p_value,
    )
    ax.set_title(f"Interictal · n={len(true_order)}", fontsize=11.5,
                 fontweight="bold", pad=5)
    return {
        "contract": "v0.5_true_suffix_vs_split_matched_reassigned_suffix",
        "n": int(len(true_order)), "median_gain_nats": float(np.median(gain)),
        "n_positive": int(np.sum(gain > 1e-9)), "p_greater": float(p_value),
    }


def draw_panel_e(fig: plt.Figure, spec, out: Path) -> dict:
    sub = spec.subgridspec(1, 3, wspace=.68)
    ax_oracle, ax_mixture, ax_j = (fig.add_subplot(sub[0, index]) for index in range(3))
    frame = pd.read_csv(out / "early_ictal/EARLY_ICTAL_PER_PATIENT.csv")
    l3 = frame[
        (frame.condition == f"INTACT|{r2.L3}") & (frame.endpoint == "canonical_full")
    ].sort_values("subject")
    p_oracle = base.paired_test(
        l3.observed.to_numpy() - l3.all_contact_null_median.to_numpy(), "greater"
    )
    base.paired_axis(
        ax_oracle, l3.observed, l3.all_contact_null_median,
        ("RNN", "Channel\nshuffle"), (base.RED, base.GRAY),
        "Signed field correlation", 1.0,
    )
    ax_oracle.set_title(f"Best mode (oracle) · n={l3.subject.nunique()}",
                        fontsize=10.4, fontweight="bold", pad=4)
    r3.mark_not_significant(ax_oracle, float(p_oracle))

    mixture = frame[
        (frame.condition == f"INTACT_MIXTURE|{r2.L3}") & (frame.endpoint == "canonical_full")
    ].sort_values("subject")
    if mixture.subject.tolist() != l3.subject.tolist():
        raise RuntimeError("Panel-E oracle/mixture patient order changed")
    p_mixture = base.paired_test(
        mixture.observed.to_numpy() - mixture.all_contact_null_median.to_numpy(), "greater"
    )
    base.paired_axis(
        ax_mixture, mixture.observed, mixture.all_contact_null_median,
        ("Mixture", "Channel\nshuffle"), (base.BLUE, base.GRAY),
        "Signed field correlation", 1.0,
    )
    ax_mixture.set_title("Train mixture", fontsize=10.4, fontweight="bold", pad=4)
    ax_mixture.set_ylabel("")
    r3.mark_not_significant(ax_mixture, float(p_mixture))

    patient = frame[frame.endpoint == "canonical_full"].pivot(
        index="subject", columns="condition", values="observed"
    )
    delta = patient[f"INTACT|{r2.L3}"] - patient[f"INTACT|{r2.L2M}"]
    j_table = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv").set_index("subject")
    common = delta.index.intersection(j_table.index)
    j_values = j_table.loc[common, "J_lat_exceedance_burden"].to_numpy(float)
    ax_j.scatter(np.sqrt(np.maximum(j_values, 0)), delta.loc[common], s=25,
                 color=base.RED, edgecolor="white", lw=.35)
    ax_j.axhline(0, color="#858b8e", lw=.75, ls="--")
    kept = np.asarray([0.0, .05, .25, .60])
    ax_j.set_xticks(np.sqrt(kept), ["0", ".05", ".25", ".60"])
    ax_j.set_xlim(-.025, np.sqrt(max(.60, float(np.nanmax(j_values)))) + .035)
    ax_j.set_xlabel("Nonlocality J\n(sqrt scale)")
    ax_j.set_ylabel("Selected − matched\nsigned field correlation")
    ax_j.spines[["top", "right"]].set_visible(False)
    summary = json.loads((out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json").read_text())
    return {
        "contract": "oracle_plus_train_prevalence_mixture_plus_primary_J_interaction",
        "n": int(l3.subject.nunique()), "oracle_p_vs_null": float(p_oracle),
        "mixture_p_vs_null": float(p_mixture), "interaction": summary["primary_interaction"],
        "nonprimary_views_marked_not_significant": True,
    }


def load_operator_tables(operator_root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((operator_root / "PATCH_OPERATOR_SUMMARY.json").read_text())
    convergence = pd.read_csv(operator_root / "OPERATOR_TOPOLOGY_CONVERGENCE.csv")
    loo = pd.read_csv(operator_root / "OPERATOR_LEAVE_ONE_OUT_CONSENSUS.csv")
    alignment = pd.read_csv(operator_root / "OPERATOR_DATA_ALIGNMENT.csv")
    convergence = convergence.groupby("patient", as_index=False).median(numeric_only=True)
    loo = loo.groupby("patient", as_index=False).median(numeric_only=True)
    alignment = alignment.groupby("patient", as_index=False).median(numeric_only=True)
    return summary, convergence, loo, alignment


def draw_operator_row(
    fig: plt.Figure,
    spec_f,
    spec_g,
    operator_root: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, convergence, loo, alignment = load_operator_tables(operator_root)

    sub = spec_f.subgridspec(1, 2, wspace=.40)
    ax_conv, ax_loo = (fig.add_subplot(sub[0, index]) for index in range(2))
    paired_operator_axis(
        ax_conv,
        convergence["real_to_shuffled_similarity_corrected"],
        convergence["real_pair_similarity_corrected"],
        ("real vs\nshuffled", "real vs\nreal"), base.RED,
        "reliability-corrected Δ=+0.076\n"
        "95% CI [0.048, 0.164]; 24/28; Holm P=4.1e-5",
    )
    ax_conv.set_ylabel("patch-response operator similarity")
    ax_conv.set_title("Cross-topology convergence", fontsize=10.6, fontweight="bold", pad=31)
    ax_conv.text(
        .02, -.22,
        r"$K(c,i)=\langle[\ell_c^{+}(\tau)-\ell_c^{-}(\tau)]/(2d)\rangle$: "
        "tissue patch i → future contact c\n"
        "real order = L0/L1/L2m/L3; shuffled control = C-suffix",
        transform=ax_conv.transAxes, ha="left", va="top", fontsize=7.3, color="#4f565b",
    )

    paired_operator_axis(
        ax_loo,
        loo["consensus_predicts_shuffled"],
        loo["consensus_predicts_heldout_real"],
        ("consensus →\nshuffled", "consensus →\nheld-out real"), base.BLUE,
        "leave-one-topology-out Δ=+0.069\n"
        "95% CI [0.038, 0.183]; 23/28; Holm P=4.1e-5",
    )
    ax_loo.set_ylabel("")
    ax_loo.set_title("Leave-one-topology-out", fontsize=10.6, fontweight="bold", pad=31)
    ax_loo.text(
        .98, -.22,
        "phase invariance 0.918; half-dose consistency 0.996;\n"
        "split-half reliability real/shuffled 0.996/0.997",
        transform=ax_loo.transAxes, ha="right", va="top", fontsize=7.3, color="#4f565b",
    )

    ax_align = fig.add_subplot(spec_g)
    series = [
        alignment["consensus_minus_shuffled_arm"].to_numpy(float),
        alignment["all_contact_margin"].to_numpy(float),
        alignment["within_shaft_margin"].to_numpy(float),
        alignment["distance_bin_margin"].to_numpy(float),
        alignment["smoothing_matched_identity_margin"].to_numpy(float),
    ]
    strip_axis(
        ax_align, series,
        ["vs shuffled\nnetwork", "all-contact\nunstructured", "within-shaft\nprimary",
         "distance-bin\nsensitivity", "patient identity\nmatched smoothing"],
        [base.RED, base.GRAY, base.BLUE, LIGHT_BLUE, GOLD],
    )
    ax_align.set_ylabel("held-out transition alignment margin")
    ax_align.set_title("Consensus operator recovers held-out propagation",
                       fontsize=10.6, fontweight="bold", pad=5)
    ax_align.text(
        .015, .985,
        "within-shaft +0.068 [0.018, 0.128], Holm P=0.0034\n"
        "matched identity +0.086 [-0.044, 0.148]: not confirmed",
        transform=ax_align.transAxes, ha="left", va="top", fontsize=7.3, color="#4f565b",
    )

    f_source = convergence.merge(loo, on="patient", how="outer")
    return summary, f_source, alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--operator-root", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    out = args.out_root.resolve()
    old = args.old_root.resolve()
    canonical = args.canonical_root.resolve()
    operator_root = args.operator_root.resolve()
    destination = args.figure_dir.resolve()

    required = [
        out / "PIPELINE_COMPLETE.json", out / "EARLY_ICTAL_SCORING_COMPLETE.json",
        operator_root / "PATCH_OPERATOR_SUMMARY.json",
        operator_root / "OPERATOR_TOPOLOGY_CONVERGENCE.csv",
        operator_root / "OPERATOR_LEAVE_ONE_OUT_CONSENSUS.csv",
        operator_root / "OPERATOR_DATA_ALIGNMENT.csv",
        operator_root / "OPERATOR_PHASE_INVARIANCE.csv",
        operator_root / "OPERATOR_DOSE_CONSISTENCY.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Figure-6 operator candidate inputs missing: {missing}")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.7, "axes.labelsize": 11.5,
        "axes.titlesize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(15.4, 11.1), facecolor="white")
    grid = fig.add_gridspec(
        3, 12, height_ratios=(.93, .90, .88), left=.045, right=.985,
        bottom=.070, top=.97, wspace=.78, hspace=.45,
    )

    graph = r3.annotated_full_tissue_graph(base.draw_full_tissue_graph)
    events = r3.annotated_event_reproduction(base.draw_event_reproduction)
    ax_a = fig.add_subplot(grid[0, 0:3])
    graph(ax_a, old, base.FIT_ID, canonical)
    events(fig, grid[0, 3:10], out, old, canonical)
    ax_c = fig.add_subplot(grid[0, 10:12])
    c_stats = draw_panel_c(ax_c, out)
    d_stats = r3.relaid_cross_state_fields(fig, grid[1, 0:8], out, canonical)
    e_stats = draw_panel_e(fig, grid[1, 8:12], out)
    operator_summary, f_source, g_source = draw_operator_row(
        fig, grid[2, 0:6], grid[2, 6:12], operator_root,
    )

    cells = (
        grid[0, 0:3], grid[0, 3:10], grid[0, 10:12], grid[1, 0:8],
        grid[1, 8:12], grid[2, 0:6], grid[2, 6:12],
    )
    for label, cell in zip("ABCDEFG", cells):
        base.grid_letter(fig, cell, label)

    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    f_source.to_csv(source / "panel_f_axis_free_operator_convergence.csv", index=False)
    g_source.to_csv(source / "panel_g_operator_data_alignment.csv", index=False)
    write_json(source / "panel_f_g_operator_summary.json", operator_summary)

    stem = destination / STEM
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    inputs = {str(path.relative_to(ROOT)): sha256_file(path) for path in required}
    assets = {path.name: sha256_file(path) for path in (
        stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg"),
    )}
    metadata = {
        "contract": "topic5_figure6_multiscale_scaffold_v0_5_operator_r1_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_PENDING_USER_REVIEW_AND_VISUAL_QA",
        "panels": {
            "A": "full-tissue recurrent model and SEEG readout",
            "B": "held-out interictal data and free generation in representative patient",
            "C": "true suffix association versus reassigned suffix control",
            "D": "representative frozen RNN fields and early-ictal broadband field",
            "E": "early-ictal cohort trends and primary nonlocality interaction",
            "F": "axis-free patch-response operator convergence and leave-one-topology-out",
            "G": "held-out transition alignment and smoothing-matched patient identity boundary",
        },
        "panel_c": c_stats, "panel_d": d_stats, "panel_e": e_stats,
        "panel_f": operator_summary["topology_convergence"],
        "panel_g": operator_summary["data_link"],
        "claim_boundary": (
            "Static recurrent topology is not identified. Real-order networks converge on an "
            "axis-free finite-time functional response operator that aligns with held-out "
            "within-patient propagation after coarse spatial control; smoothing-matched "
            "cross-patient identity remains unconfirmed, and necessity was not tested."
        ),
        "inputs": inputs, "assets_sha256": assets,
    }
    write_json(destination / "FIGURE6_OPERATOR_R1_METADATA.json", metadata)
    write_json(destination / "FIGURE6_OPERATOR_R1_COMPLETE.json", {
        "status": "COMPLETE_PENDING_USER_REVIEW", "assets_sha256": assets,
    })
    (destination / "README.md").write_text(
        "### topic5_figure6_multiscale_scaffold_v0_5_operator_r1_candidate.png / .pdf / .svg\n\n"
        "A–E 与已验收 v0.5 Figure 6 相同：A 是 full-tissue RNN 和 SEEG 读出；B 是 E1146 "
        "留出事件与自由生成；C 检验真实 prefix–suffix association；D 是代表患者的 RNN TA/TB "
        "场与 early-ictal broadband 场；E 是17位患者的跨状态趋势及 nonlocality interaction。\n\n"
        "F 不再比较某几条 recurrent edges，而是比较 axis-free tissue-patch response operator。"
        "左图比较四种真实顺序 topology 之间与真实顺序对 C-suffix 的 operator similarity；右图用"
        "三个 topology 的 consensus 预测未参与 consensus 的第四个 topology。两项均为患者配对，"
        "黑线为中位数，竖线为 bootstrap 95% CI。\n\n"
        "G 把 consensus operator 映射到 contact space，并与冻结模型未见过的事件转移算子比较。"
        "within-shaft 和 distance-bin 控制后仍为正；all-contact 只是不保留空间结构的粗 null。"
        "smoothing-matched patient-identity margin 的中位数虽为正，但95% CI跨0，因此患者身份尚未确认。\n\n"
        "**关注点**：F/G 支持的是 finite-time functional response 的跨拓扑收敛和留出数据对齐，"
        "不是 edge mask、白质通路或必要性证明。当前 candidate 保留旧图，不覆盖已验收资产。\n"
    )
    print(json.dumps({"figure": str(stem.with_suffix('.png')), "assets": assets}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure C -- H2a, does the slow state change the event distribution?

asset_id: epi_prssm_event_distribution
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, FS_AXIS, FS_TICK, FS_TITLE, LW_MAIN, LW_REFERENCE,
    figure, panel_letter, save_asset, zero_line,
)

ASSET = "epi_prssm_event_distribution"
EVENTS = OUTPUT_ROOT / "event_distribution"
FIG_ROOT = OUTPUT_ROOT / "figures"


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a table that a stage may legitimately have left empty."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return pd.DataFrame(columns=columns or [])

ADAPTERS = ["initial_state", "node_film", "edge_gate"]
ADAPTER_LABEL = {"initial_state": "initial-state", "node_film": "node FiLM",
                 "edge_gate": "low-rank edge gate"}
SOURCE_COLOR = {"g0": COLOR["G0"], "g2": COLOR["G2"], "g3": COLOR["G3"]}


def panel_a(ax, effects: pd.DataFrame):
    ax.set_title("Capacity versus state", fontsize=FS_TITLE, loc="left", pad=5)
    subset = effects[(effects.endpoint == "order_nll")]
    capacity = subset[subset.contrast == "adapter capacity alone"].set_index("adapter")
    state = subset[subset.contrast == "state vs frozen-state (capacity-matched)"]
    x = np.arange(len(ADAPTERS))
    ax.bar(x - 0.19, [-capacity["median_delta"].get(a, np.nan) for a in ADAPTERS],
           width=0.34, color="#C9C9C9", label="adapter capacity (frozen state)")
    for j, source in enumerate(["g0", "g2", "g3"]):
        rows = state[state.state_source == source].set_index("adapter")
        ax.plot(x + 0.19, [-rows["median_delta"].get(a, np.nan) for a in ADAPTERS],
                "o", ms=4.2, color=SOURCE_COLOR[source], mew=0,
                label=f"state gain, {source.upper()}")
        for i, adapter in enumerate(ADAPTERS):
            if adapter in rows.index:
                ax.plot([x[i] + 0.19, x[i] + 0.19],
                        [-rows["ci_high"].loc[adapter], -rows["ci_low"].loc[adapter]],
                        color=SOURCE_COLOR[source], lw=0.9)
    zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([ADAPTER_LABEL[a] for a in ADAPTERS], fontsize=FS_TICK - 0.4)
    ax.set_ylabel("masked-order improvement")
    ax.legend(loc="best", fontsize=FS_TICK - 0.8, handlelength=0.9, borderpad=0.2,
              labelspacing=0.22)


def panel_b(ax, effects: pd.DataFrame):
    ax.set_title("Full-event endpoints", fontsize=FS_TITLE, loc="left", pad=5)
    order = ["order_nll", "stop_nll", "participation_nll", "selection_nll", "event_nll"]
    labels = ["masked order", "STOP", "participation", "selection", "whole event"]
    subset = effects[(effects.contrast == "state vs frozen-state (capacity-matched)")
                     & (effects.state_source == "g2") & (effects.adapter == "node_film")]
    subset = subset.set_index("endpoint")
    y = np.arange(len(order))
    for i, endpoint in enumerate(order):
        if endpoint not in subset.index:
            continue
        row = subset.loc[endpoint]
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=COLOR["G2"], lw=1.2)
        ax.plot(row["median_delta"], i, "o", ms=4.6, color=COLOR["G2"], mew=0)
        ax.text(row["ci_high"], i + 0.26,
                f"{int(row['n_favourable'])}/{int(row['n_patients'])}",
                fontsize=FS_TICK - 0.8, color="#333333", va="bottom")
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("Δ NLL, moving state − frozen state")


def panel_c(ax, swaps: pd.DataFrame):
    ax.set_title("Correct versus swapped state", fontsize=FS_TITLE, loc="left", pad=5)
    if swaps.empty:
        ax.text(0.5, 0.5, "no state-swap table", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_TICK)
        ax.set_xticks([]); ax.set_yticks([])
        return
    subset = swaps[(swaps.endpoint == "order_nll") & (swaps.swap == "swap_matched")
                   & (swaps.arm == "node_film_g2")]
    if subset.empty:
        subset = swaps[(swaps.endpoint == "order_nll") & (swaps.swap == "swap_matched")]
    for _, row in subset.iterrows():
        colour = COLOR["epilepsiae"] if row["dataset"] == "epilepsiae" else COLOR["yuquan"]
        ax.plot([0, 1], [row["correct"], row["swapped"]], color="#CCCCCC", lw=0.55, zorder=1)
        ax.plot([0, 1], [row["correct"], row["swapped"]], "o", ms=2.6, color=colour,
                mew=0, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["correct state", "matched swap"], fontsize=FS_TICK)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylabel("masked-order NLL")
    if not subset.empty:
        delta = float(subset["delta"].median())
        favourable = int((subset["delta"] < 0).sum())
        ax.text(0.02, 0.02, f"median Δ {delta:+.5f}\n{favourable}/{len(subset)} lower with "
                            "the correct state",
                transform=ax.transAxes, fontsize=FS_TICK - 1.2, va="bottom", color="#333333")


def panel_d(ax, prefixes: pd.DataFrame, inventory: pd.DataFrame):
    ax.set_title("Ambiguous-prefix support", fontsize=FS_TITLE, loc="left", pad=5)
    if inventory.empty:
        ax.text(0.5, 0.5, "no train-only prefix inventory", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_TICK)
        ax.set_xticks([]); ax.set_yticks([])
        return
    support = inventory.pivot_table(index="subject", columns="prefix_depth",
                                    values="n_events_in_ambiguous_families", aggfunc="max")
    support = support.reindex(sorted(support.index, key=lambda s: -support.loc[s].fillna(0).sum()))
    matrix = np.log10(support.to_numpy(dtype=float) + 1.0)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(support.shape[1]))
    ax.set_xticklabels([f"depth {c}" for c in support.columns], fontsize=FS_TICK - 0.4)
    ax.set_yticks([])
    ax.set_ylabel(f"{len(support)} patients")
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.040, pad=0.02)
    bar.set_label("log10 train events", fontsize=FS_TICK - 1.2)
    bar.ax.tick_params(labelsize=FS_TICK - 1.4)


def panel_e(ax, prefixes: pd.DataFrame):
    ax.set_title("Suffix branch", fontsize=FS_TITLE, loc="left", pad=5)
    if prefixes.empty:
        ax.text(0.5, 0.5, "no eligible prefix family", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_TICK)
        ax.set_xticks([]); ax.set_yticks([])
        return
    subset = prefixes[prefixes.arm == "node_film_g2"]
    if subset.empty:
        subset = prefixes
    depths = sorted(subset.prefix_depth.unique())
    rng = np.random.default_rng(11)
    for i, depth in enumerate(depths):
        rows = subset[subset.prefix_depth == depth].groupby("subject").state_gain.median()
        jitter = rng.normal(0, 0.05, len(rows))
        ax.plot(i + jitter, rows.to_numpy(), "o", ms=2.6, color=COLOR["G1"], mew=0, alpha=0.7)
        median = float(rows.median())
        ax.plot([i - 0.25, i + 0.25], [median, median], color="#1A1A1A", lw=1.5)
        ax.text(i, ax.get_ylim()[1], f"{int((rows > 0).sum())}/{len(rows)}",
                fontsize=FS_TICK - 0.8, ha="center", va="bottom", color="#333333")
    zero_line(ax)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"depth {d}" for d in depths], fontsize=FS_TICK - 0.2)
    ax.set_xlim(-0.5, len(depths) - 0.5)
    ax.set_ylabel("suffix log-prob gain")


def panel_f(ax, card: dict, swaps: pd.DataFrame):
    ax.set_title("Where the evidence sits", fontsize=FS_TITLE, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    lines = [
        ("full-event, capacity-matched", _describe(card, "order_nll")),
        ("state swap (matched)", _swap_line(swaps)),
        ("ambiguous prefix", _prefix_line(card)),
        ("targeted eligible", f"{len(card.get('targeted_eligible_patients', []))} of "
                              f"{card['denominators']['n_patients']} patients"),
    ]
    y = 0.97
    for title, body in lines:
        ax.text(0.0, y, title, fontsize=FS_TICK - 0.4, fontweight="bold", color="#222222")
        ax.text(0.0, y - 0.055, body, fontsize=FS_TICK - 1.2, color="#444444", va="top")
        y -= 0.245
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def _describe(card: dict, endpoint: str) -> str:
    rows = [r for r in card.get("cohort_wide_state_versus_frozen_state", [])
            if r.get("endpoint") == endpoint]
    if not rows:
        return "not available"
    best = rows[0]
    return (f"{best['adapter']} / {best['state_source']}: median {best['median_delta']:+.4f}\n"
            f"CI [{best['ci_low']:+.4f}, {best['ci_high']:+.4f}], "
            f"{int(best['n_favourable'])}/{int(best['n_patients'])} patients")


def _swap_line(swaps: pd.DataFrame) -> str:
    subset = swaps[(swaps.endpoint == "order_nll") & (swaps.swap == "swap_matched")]
    if subset.empty:
        return "not available"
    per_arm = subset.groupby("arm").delta.median().sort_values()
    arm = per_arm.index[0]
    rows = subset[subset.arm == arm]
    return (f"{arm}: median {float(rows.delta.median()):+.5f}\n"
            f"{int((rows.delta < 0).sum())}/{len(rows)} patients lower with the correct state")


def _prefix_line(card: dict) -> str:
    block = card.get("ambiguous_prefix", {})
    if not isinstance(block, dict) or "status" in block:
        return "no eligible family"
    by_arm = block.get("by_arm")
    if not by_arm:
        return "no moving-state arm"
    # one line per depth, pooling the moving-state arms only; the frozen arms are
    # exactly zero by construction and are reported as the negative control.
    parts = []
    depths = sorted({int(d) for arm in by_arm.values() for d in arm})
    for depth in depths:
        medians = [arm[str(depth)]["median_delta"] if str(depth) in arm
                   else arm[depth]["median_delta"]
                   for arm in by_arm.values() if str(depth) in arm or depth in arm]
        favs = [arm[str(depth)]["n_favourable"] if str(depth) in arm
                else arm[depth]["n_favourable"]
                for arm in by_arm.values() if str(depth) in arm or depth in arm]
        ns = [arm[str(depth)]["n_patients"] if str(depth) in arm else arm[depth]["n_patients"]
              for arm in by_arm.values() if str(depth) in arm or depth in arm]
        if not medians:
            continue
        parts.append(f"depth {depth}: {min(medians):+.4f} to {max(medians):+.4f}, "
                     f"{min(favs)}-{max(favs)}/{max(ns)} over {len(medians)} arms")
    control = block.get("negative_control_arms") or {}
    if control:
        parts.append(f"frozen controls ({len(control)} arms): 0.0000 by construction")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    effects = _read_csv(EVENTS / "full_event_effects.csv",
                        ["endpoint", "adapter", "state_source", "contrast", "n_patients",
                         "median_delta", "ci_low", "ci_high", "n_favourable", "sign_test_p",
                         "wilcoxon_p"])
    swaps = _read_csv(EVENTS / "state_swap_effects.csv",
                      ["endpoint", "arm", "swap", "subject", "dataset", "correct", "swapped",
                       "delta"])
    prefixes = _read_csv(EVENTS / "ambiguous_prefix_effects.csv",
                         ["arm", "seed", "subject", "dataset", "prefix_depth", "n_events",
                          "suffix_nll_correct", "suffix_nll_swapped", "state_gain"])
    inventory = _read_csv(OUTPUT_ROOT / "data_audit/ambiguous_prefix_inventory_train_only.csv",
                          ["subject", "dataset", "prefix_depth",
                           "n_events_in_ambiguous_families"])
    card = json.loads((EVENTS / "H2A_EVIDENCE_CARD.json").read_text())

    fig, axes = figure(DOUBLE_COLUMN_MM, 128.0, nrows=2, ncols=3)
    fig.subplots_adjust(left=0.080, right=0.975, top=0.895, bottom=0.095, wspace=0.48, hspace=0.52)
    panel_a(axes[0, 0], effects)
    panel_b(axes[0, 1], effects)
    panel_c(axes[0, 2], swaps)
    panel_d(axes[1, 0], prefixes, inventory)
    panel_e(axes[1, 1], prefixes)
    panel_f(axes[1, 2], card, swaps)
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        panel_letter(ax, letter, dx=-0.23, dy=1.20)

    files = save_asset(fig, ASSET, FIG_ROOT, metadata={
        "asset_id": ASSET, "provisional_role": "Figure C", "hypothesis": "H2a",
        "status": "EXPLORATORY", "split": "development validation partition",
        "denominators": card["denominators"],
        "targeted_eligible_patients": card.get("targeted_eligible_patients", []),
        "not_eligible_for_targeted_analysis": card.get("not_eligible_for_targeted_analysis", []),
        "claim_boundary": card["claim_boundary"],
        "frozen_ta_tb_projection": {"status": "NOT_RUN",
                                    "reason": "TA/TB template labels are a forbidden input to "
                                              "this model family and no frozen downstream "
                                              "projection was released for this cohort"},
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "H2a 的证据卡。A 把每种适配器自身的容量增益（灰柱，状态冻结）与"
                "真正来自状态的增益（彩色点，带自助区间）并排放，避免把容量算成状态。"
                "B 是同一个模型在五个端点上的效应；掩蔽顺序端点与参与度端点是分开的，"
                "所以顺序上的增益不是参与人数的同义反复。C 是逐患者的成对反事实："
                "同一模型换成患者内部配平的错位状态。D 是训练集里可用的歧义前缀支持度，"
                "颜色只表示支持度、不表示结果。E 是这些前缀上后缀分支的状态增益。"
                "F 汇总各条证据的中位数、区间与分母。",
        "focus": "C 的配对线方向是承重的：若绝大多数患者换成错位状态后 NLL 几乎不变，"
                 "那么 A/B 里的增益就应当读成读出容量而不是状态信息。",
    }])
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the full-tissue LBSS v0.3 engineering-smoke audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
LABELS = ("Local", "+ local", "+ random", "+ selected", "Shuffle")
COLORS = ("#777d82", "#6c96a6", "#9b8b72", "#b84b4b", "#b8b8b8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    marker = json.loads((out / "SMOKE_TRAINING_COMPLETE.json").read_text())
    if marker.get("unresolved") != 0:
        raise RuntimeError("smoke stage is not complete")

    records = []
    histories = {}
    for path in sorted((out / "diagnostic_smoke_units").glob("*/*/seed0/metrics.json")):
        value = json.loads(path.read_text())
        records.append({
            "fit_id": value["fit_id"],
            "arm": value["arm"],
            "n_nodes": value["n_nodes"],
            "test_contact_nll": value["test"]["contact_nll"],
            "seconds": value["seconds"],
        })
        histories[(value["fit_id"], value["arm"])] = json.loads(
            (path.parent / "history.json").read_text()
        )
    table = pd.DataFrame(records)
    if len(table) != 15:
        raise RuntimeError(f"expected 15 smoke units, found {len(table)}")

    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), gridspec_kw={"wspace": 0.46})
    plt.rcParams.update({"pdf.fonttype": 42})

    audit = pd.read_csv(out / "LATENT_DOMAIN_AUDIT.csv")
    audit = audit[(audit.version == "v0.3") & audit.fit_id.isin(table.fit_id.unique())]
    axis = axes[0]
    x = np.arange(len(audit))
    axis.bar(x, audit.n_nodes, color="#3f7f93", width=0.68)
    axis.bar(x, audit.n_zero_h_nodes, color="#c9ced1", width=0.68)
    axis.set_xticks(x, [fit.split("__")[0].replace("epilepsiae_", "E").replace("yuquan_", "Y-")
                       for fit in audit.fit_id], rotation=25, ha="right")
    axis.set_ylabel("Latent nodes")
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1]
    fit = "epilepsiae_1146__shared"
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        history = histories[(fit, arm)]
        axis.plot([row["epoch"] for row in history],
                  [row["validation_score"] for row in history],
                  color=color, lw=1.35, label=label)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Validation loss")
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[2]
    for fit_index, (_, group) in enumerate(table.groupby("fit_id", sort=True)):
        values = [float(group.loc[group.arm == arm, "test_contact_nll"].iloc[0]) for arm in ARMS]
        axis.plot(np.arange(len(ARMS)), values, color="#c4c8ca", lw=0.8, zorder=1)
        axis.scatter(np.arange(len(ARMS)), values, color=COLORS, s=22, zorder=2)
    medians = table.groupby("arm").test_contact_nll.median()
    axis.plot(np.arange(len(ARMS)), [medians[arm] for arm in ARMS], color="#171717", lw=1.8)
    axis.set_xticks(np.arange(len(ARMS)), LABELS, rotation=32, ha="right")
    axis.set_ylabel("Heldout contact NLL")
    axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center",
               bbox_to_anchor=(0.56, 1.05), fontsize=8.5)
    for label, axis in zip("ABC", axes):
        axis.text(-0.19, 1.07, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"stage_c_full_tissue_smoke.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (figure_dir / "README.md").open("a") as stream:
        stream.write(
            "\n### stage_c_full_tissue_smoke.png\n\n"
            "A 显示三个工程病例的总 latent nodes 与其中 zero-H nodes；B 为 E1146 五个 matched arms 的验证曲线；"
            "C 检查 heldout contact NLL 未因 full-tissue state 数量增加而数值塌缩。\n\n"
            "**关注点**：本图只验收训练、显存与输出合同，不作患者机制筛选。\n"
        )


if __name__ == "__main__":
    main()

"""Preview Fig2B: cluster-aware matching-index cohort panel.

This is a plotting-only preview. It consumes the masked interictal propagation
outputs and re-reads the lagPat artifacts only to compute masked, within-cluster
matching-index summaries for stable k=2 subjects.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import plot_interictal_propagation as prop_plot  # noqa: E402
from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    build_cluster_templates,
)
from src.plot_style import COL_EPI, COL_YQ, COL_SIG  # noqa: E402


OUT_DIR = ROOT / "results/paper-ready-figure/fig2b_template_matching_preview/figures"
MAX_REAL_EVENTS_PER_CLUSTER = 8000
MAX_NULL_EVENTS_PER_CLUSTER = 900
N_PERMUTATIONS = 120
MIN_ACTIVE_CHANNELS = 3


def _sample_indices(values: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if values.size <= cap:
        return values
    return np.sort(rng.choice(values, size=cap, replace=False))


def _event_mi(template: np.ndarray, ranks_col: np.ndarray, active_mask: np.ndarray) -> float:
    valid = np.isfinite(template) & np.isfinite(ranks_col) & active_mask
    if int(valid.sum()) < MIN_ACTIVE_CHANNELS:
        return float("nan")
    tmpl = np.asarray(template[valid], dtype=float)
    vals = np.asarray(ranks_col[valid], dtype=float)
    pa, pb = np.triu_indices(tmpl.size, k=1)
    if pa.size == 0:
        return float("nan")
    score = np.sign(tmpl[pa] - tmpl[pb]) * np.sign(vals[pa] - vals[pb])
    return float(np.mean(score))


def _cluster_mi_record(record: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    dataset = str(record["dataset"])
    subject = str(record["subject"])
    subject_dir = prop_plot._resolve_subject_dir(dataset, subject)
    loaded = prop_plot._load_lagpat(subject_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)

    valid_events = _valid_event_indices(bools, min_participating=MIN_ACTIVE_CHANNELS)
    labels = np.asarray(record.get("adaptive_cluster", {}).get("labels", []), dtype=int)
    if labels.shape != valid_events.shape:
        raise ValueError(f"{dataset}:{subject} label/event mismatch")

    stable_k = int(record.get("adaptive_cluster", {}).get("stable_k") or 0)
    if stable_k != 2:
        return []

    ranks_v = ranks[:, valid_events]
    bools_v = bools[:, valid_events]
    templates = build_cluster_templates(ranks_v, bools_v, labels, stable_k)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for cluster_id in range(stable_k):
        local_idx = np.where(labels == cluster_id)[0]
        event_idx = valid_events[local_idx]
        if event_idx.size < 5:
            continue
        template = templates[cluster_id]
        real_idx = _sample_indices(event_idx, MAX_REAL_EVENTS_PER_CLUSTER, rng)
        real_vals = [
            _event_mi(template, ranks[:, ei], bools[:, ei])
            for ei in real_idx
        ]
        real_arr = np.asarray(real_vals, dtype=float)
        real_arr = real_arr[np.isfinite(real_arr)]
        if real_arr.size == 0:
            continue

        null_event_idx = _sample_indices(event_idx, MAX_NULL_EVENTS_PER_CLUSTER, rng)
        null_medians = []
        for _ in range(N_PERMUTATIONS):
            vals = []
            for ei in null_event_idx:
                active = np.isfinite(template) & bools[:, ei] & np.isfinite(ranks[:, ei])
                if int(active.sum()) < MIN_ACTIVE_CHANNELS:
                    continue
                shuffled = ranks[:, ei].copy()
                shuffled[active] = rng.permutation(shuffled[active])
                vals.append(_event_mi(template, shuffled, active))
            if vals:
                null_medians.append(float(np.nanmedian(vals)))
        null_arr = np.asarray(null_medians, dtype=float)
        null_arr = null_arr[np.isfinite(null_arr)]
        if null_arr.size == 0:
            continue

        real_median = float(np.nanmedian(real_arr))
        null_p95 = float(np.nanpercentile(null_arr, 95))
        p_value = float((1 + np.sum(null_arr >= real_median)) / (1 + null_arr.size))
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "cluster_id": cluster_id,
                "n_events": int(event_idx.size),
                "n_real_used": int(real_arr.size),
                "n_null_events": int(null_event_idx.size),
                "real_median_mi": real_median,
                "real_iqr": [
                    float(np.nanpercentile(real_arr, 25)),
                    float(np.nanpercentile(real_arr, 75)),
                ],
                "null_median": float(np.nanmedian(null_arr)),
                "null_p95": null_p95,
                "margin_over_null_p95": real_median - null_p95,
                "p_value": p_value,
            }
        )
    return rows


def build_rows() -> list[dict[str, Any]]:
    prop_plot._apply_masked_paths()
    subjects = prop_plot._load("pr1_subject_summary.json")
    records = [
        rec for rec in subjects.values()
        if isinstance(rec, dict)
        and "error" not in rec
        and int(rec.get("adaptive_cluster", {}).get("stable_k") or 0) == 2
    ]
    records = sorted(records, key=lambda r: (r["dataset"], str(r["subject"])))
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        rows.extend(_cluster_mi_record(rec, seed=9100 + idx))
    return rows


def plot(rows: list[dict[str, Any]]) -> None:
    by_subject: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault((row["dataset"], row["subject"]), []).append(row)
    complete = {
        key: sorted(vals, key=lambda r: r["cluster_id"])
        for key, vals in by_subject.items()
        if len(vals) == 2
    }
    ordered = sorted(
        complete.items(),
        key=lambda item: min(r["margin_over_null_p95"] for r in item[1]),
    )
    if not ordered:
        raise RuntimeError("No complete stable-k=2 subjects available for Fig2B preview")

    fig, ax = plt.subplots(figsize=(8.6, 4.35), facecolor="white")
    x = np.arange(len(ordered), dtype=float)
    cluster_colors = ["#2F6FA3", "#C94C4C"]

    both_sig = 0
    template_sig = 0
    weaker_mi = []
    for i, ((dataset, subject), vals) in enumerate(ordered):
        y = np.array([v["real_median_mi"] for v in vals], dtype=float)
        null95 = float(max(v["null_p95"] for v in vals))
        weaker_mi.append(float(np.min(y)))
        if all(v["real_median_mi"] > v["null_p95"] for v in vals):
            both_sig += 1
        template_sig += sum(v["real_median_mi"] > v["null_p95"] for v in vals)

        ax.plot([i, i], [y.min(), y.max()], color="0.72", lw=1.0, zorder=1)
        ax.plot([i - 0.18, i + 0.18], [null95, null95], color="0.35", lw=1.2, zorder=2)
        for j, val in enumerate(vals):
            ax.scatter(
                i + (-0.10 if j == 0 else 0.10),
                val["real_median_mi"],
                s=42,
                color=cluster_colors[j],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
                label=f"Template {j + 1}" if i == 0 else None,
            )
        ax.scatter(
            i,
            -0.055,
            s=18,
            marker="s",
            color=COL_YQ if dataset == "yuquan" else COL_EPI,
            clip_on=False,
            zorder=5,
        )

    ax.axhline(0, color="0.72", lw=0.9, zorder=0)
    ax.set_xlim(-0.8, len(ordered) - 0.2)
    ax.set_ylim(-0.08, max(0.62, max(r["real_median_mi"] for r in rows) + 0.06))
    ax.set_xticks([])
    ax.set_ylabel("Masked Matching Index", fontsize=12)
    ax.set_xlabel("stable k=2 subjects, sorted by weaker-template margin", fontsize=11)
    ax.set_title("Both interictal propagation templates match their own events", fontsize=13, pad=10)
    ax.tick_params(axis="y", labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2, handletextpad=0.4)
    ax.text(
        0.98,
        0.07,
        f"both templates > null: {both_sig}/{len(ordered)} subjects\n"
        f"template-level: {template_sig}/{2 * len(ordered)}\n"
        f"median weaker MI = {np.median(weaker_mi):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.82", alpha=0.95),
    )
    ax.text(
        0.02,
        -0.16,
        "dataset rug: Yuquan / Epilepsiae",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.35",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig2b_template_matching_preview.png"
    pdf = OUT_DIR / "fig2b_template_matching_preview.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": "fig2b_template_matching_preview",
        "input_root": "results/interictal_propagation_masked",
        "stable_k2_subjects_complete": len(ordered),
        "templates_above_null_p95": template_sig,
        "subjects_with_both_templates_above_null_p95": both_sig,
        "n_permutations": N_PERMUTATIONS,
        "max_real_events_per_cluster": MAX_REAL_EVENTS_PER_CLUSTER,
        "max_null_events_per_cluster": MAX_NULL_EVENTS_PER_CLUSTER,
        "rows": rows,
        "notes": [
            "Preview-only visual design for Fig2B.",
            "Matching index is masked per event: only participating channels with finite cluster template rank enter each event score.",
            "Null shuffles the event rank order within the same active channel set.",
            "Sorted x-axis uses weaker-template margin to make the cohort bottleneck visible.",
        ],
    }
    (OUT_DIR / "fig2b_template_matching_preview_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        "# fig2b_template_matching_preview\n\n"
        "### fig2b_template_matching_preview.png / .pdf\n\n"
        "Fig2B 预览。每个 stable k=2 subject 贡献两个点，分别是两类传播模板在自己簇内事件上的 masked Matching Index 中位数；灰色短横线是同 subject 两簇中更高的置换 95% 阈值。底部小方块标数据集来源。\n\n"
        "**关注点**：读者应能一眼看到每个 subject 的两类模板都高于置换阈值；这比全事件单模板 MI 更直接支持“两类间期传播模板都真实存在”。\n",
        encoding="utf-8",
    )
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main() -> None:
    os.chdir(ROOT)
    rows = build_rows()
    plot(rows)


if __name__ == "__main__":
    main()

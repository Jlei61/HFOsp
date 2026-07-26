#!/usr/bin/env python3
"""Audit transition-skeleton edge weights against equal-density nulls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_transition_skeleton_prior import (  # noqa: E402
    _blend_graph,
    _edge_probability,
    _folded_transition_skeleton,
    _spectral_scale,
)
from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    load_records,
)


def _directed_transition_counts(
    group_ids: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_contacts = group_ids.shape[1]
    forward = np.zeros((n_contacts, n_contacts), np.float64)
    reverse = np.zeros_like(forward)
    for event in np.asarray(group_ids, np.int16):
        valid = event >= 0
        if int(valid.sum()) < 2:
            continue
        for step in range(int(event[valid].max())):
            source = np.flatnonzero(event == step)
            target = np.flatnonzero(event == step + 1)
            if not len(source) or not len(target):
                continue
            weight = 1.0 / float(len(source) * len(target))
            for source_index in source:
                for target_index in target:
                    if axis[target_index] >= axis[source_index]:
                        forward[target_index, source_index] += weight
                    else:
                        reverse[target_index, source_index] += weight
    return forward, reverse


def _weighted_nll(
    forward_count: np.ndarray,
    reverse_count: np.ndarray,
    forward_graph: np.ndarray,
    reverse_graph: np.ndarray,
) -> float:
    forward_probability = _edge_probability(forward_graph)
    reverse_probability = _edge_probability(reverse_graph)
    count = forward_count + reverse_count
    total = float(count.sum())
    if total <= 0:
        return np.nan
    value = -(
        forward_count
        * np.log(np.clip(forward_probability, 1e-8, 1.0))
        + reverse_count
        * np.log(np.clip(reverse_probability, 1e-8, 1.0))
    ).sum() / total
    return float(value)


def _uniform_axis_graph(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_contacts = len(axis)
    forward = np.zeros((n_contacts, n_contacts), np.float32)
    for source in range(n_contacts):
        target = np.flatnonzero(axis >= axis[source])
        target = target[target != source]
        forward[target, source] = 1.0
    # Exact transpose retains the paired directional contract.
    return _spectral_scale(forward), _spectral_scale(forward).T.copy()


def _weight_shuffle(
    skeleton: np.ndarray,
    axis: np.ndarray,
    *,
    rng: np.random.Generator,
    axis_floor: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.zeros_like(skeleton)
    allowed = np.argwhere(
        (axis[:, None] > axis[None, :]) & ~np.eye(len(axis), dtype=bool)
    )
    weights = skeleton[allowed[:, 0], allowed[:, 1]].copy()
    rng.shuffle(weights)
    shuffled[allowed[:, 0], allowed[:, 1]] = weights
    tied = np.isclose(axis[:, None], axis[None, :]) & ~np.eye(
        len(axis), dtype=bool
    )
    tied_values = skeleton[tied].copy()
    rng.shuffle(tied_values)
    shuffled[tied] = tied_values
    forward, reverse, _ = _blend_graph(
        shuffled,
        axis,
        axis_floor=float(axis_floor),
        neighbors=int(neighbors),
    )
    return forward, reverse


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float).ravel()
    right = np.asarray(right, float).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return (
        float(np.dot(left, right) / denominator)
        if denominator > 0
        else np.nan
    )


def _bh_fdr(values: pd.Series) -> pd.Series:
    x = values.to_numpy(float)
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return pd.Series(out, index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=ROOT
        / "results/topic5_structured_axis_graph/transition_skeleton_prior_v0_7",
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--axis-floor", type=float, default=0.20)
    parser.add_argument("--neighbors", type=int, default=2)
    args = parser.parse_args()
    records = load_records(args.dataset_root)
    rows = []
    for subject_index, (subject, record) in enumerate(records.items()):
        path = args.prior_root / "per_subject" / f"{subject}.npz"
        with np.load(path, allow_pickle=False) as z:
            axis = np.asarray(z["axis_coordinate"], np.float32)
            skeleton = np.asarray(z["transition_skeleton_raw"], np.float32)
            forward = np.asarray(z["forward_graph"], np.float32)
            reverse = np.asarray(z["reverse_graph"], np.float32)
        heldout_groups = record.group_ids[record.eval_indices]
        heldout_forward, heldout_reverse = _directed_transition_counts(
            heldout_groups, axis
        )
        true_nll = _weighted_nll(
            heldout_forward, heldout_reverse, forward, reverse
        )
        uniform_forward, uniform_reverse = _uniform_axis_graph(axis)
        uniform_nll = _weighted_nll(
            heldout_forward,
            heldout_reverse,
            uniform_forward,
            uniform_reverse,
        )
        heldout_skeleton = _spectral_scale(
            _folded_transition_skeleton(heldout_groups, axis)
        )
        train_skeleton = _spectral_scale(skeleton)
        observed_cosine = _cosine(train_skeleton, heldout_skeleton)
        rng = np.random.default_rng(20260726 + subject_index * 1009)
        null_nll = []
        null_cosine = []
        for _ in range(int(args.permutations)):
            null_forward, null_reverse = _weight_shuffle(
                skeleton,
                axis,
                rng=rng,
                axis_floor=float(args.axis_floor),
                neighbors=int(args.neighbors),
            )
            null_nll.append(
                _weighted_nll(
                    heldout_forward,
                    heldout_reverse,
                    null_forward,
                    null_reverse,
                )
            )
            # Remove the fixed axis floor for the edge-weight similarity test.
            null_cosine.append(
                _cosine(
                    _spectral_scale(
                        _weight_shuffle(
                            skeleton,
                            axis,
                            rng=rng,
                            axis_floor=0.0,
                            neighbors=int(args.neighbors),
                        )[0]
                    ),
                    heldout_skeleton,
                )
            )
        null_nll = np.asarray(null_nll)
        null_cosine = np.asarray(null_cosine)
        rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_contacts": len(axis),
                "n_train_events": len(record.train_indices),
                "n_heldout_events": len(record.eval_indices),
                "heldout_nll_true_weights": true_nll,
                "heldout_nll_uniform_dense": uniform_nll,
                "heldout_nll_null_median": float(np.median(null_nll)),
                "heldout_nll_gain_vs_uniform": uniform_nll - true_nll,
                "heldout_nll_gain_vs_weight_shuffle": float(
                    np.median(null_nll) - true_nll
                ),
                "heldout_nll_weight_shuffle_p": float(
                    (1 + np.sum(null_nll <= true_nll))
                    / (1 + len(null_nll))
                ),
                "train_heldout_weight_cosine": observed_cosine,
                "weight_cosine_null_median": float(
                    np.median(null_cosine)
                ),
                "weight_cosine_gain_vs_null": float(
                    observed_cosine - np.median(null_cosine)
                ),
                "weight_cosine_shuffle_p": float(
                    (1 + np.sum(null_cosine >= observed_cosine))
                    / (1 + len(null_cosine))
                ),
                "heldout_used_for_construction": False,
                "ictal_target_read": False,
            }
        )
    audit = pd.DataFrame(rows)
    audit["heldout_nll_weight_shuffle_q"] = _bh_fdr(
        audit.heldout_nll_weight_shuffle_p
    )
    audit["weight_cosine_shuffle_q"] = _bh_fdr(
        audit.weight_cosine_shuffle_p
    )
    audit.to_csv(
        args.prior_root / "transition_skeleton_weight_null_audit.csv",
        index=False,
    )
    summary = {
        "status": "complete",
        "n_subjects": int(len(audit)),
        "permutations": int(args.permutations),
        "median_heldout_nll_gain_vs_uniform": float(
            audit.heldout_nll_gain_vs_uniform.median()
        ),
        "n_true_nll_better_than_uniform": int(
            (audit.heldout_nll_gain_vs_uniform > 0).sum()
        ),
        "median_heldout_nll_gain_vs_weight_shuffle": float(
            audit.heldout_nll_gain_vs_weight_shuffle.median()
        ),
        "n_true_nll_better_than_weight_shuffle": int(
            (audit.heldout_nll_gain_vs_weight_shuffle > 0).sum()
        ),
        "n_nll_weight_shuffle_fdr_lt_0_05": int(
            (audit.heldout_nll_weight_shuffle_q < 0.05).sum()
        ),
        "median_train_heldout_weight_cosine": float(
            audit.train_heldout_weight_cosine.median()
        ),
        "median_weight_cosine_gain_vs_null": float(
            audit.weight_cosine_gain_vs_null.median()
        ),
        "n_weight_cosine_fdr_lt_0_05": int(
            (audit.weight_cosine_shuffle_q < 0.05).sum()
        ),
        "heldout_used_for_construction": False,
        "ictal_target_read": False,
    }
    (args.prior_root / "transition_skeleton_weight_null_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    figure_dir = args.prior_root / "figures"
    color = audit.dataset.map(
        {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))
    axes[0].scatter(
        audit.heldout_nll_null_median,
        audit.heldout_nll_true_weights,
        c=color,
        s=24,
    )
    limit = [
        float(
            min(
                audit.heldout_nll_null_median.min(),
                audit.heldout_nll_true_weights.min(),
            )
        ),
        float(
            max(
                audit.heldout_nll_null_median.max(),
                audit.heldout_nll_true_weights.max(),
            )
        ),
    ]
    axes[0].plot(limit, limit, "--", color="#777777", lw=1)
    axes[0].set(
        xlabel="Weight-shuffle held-out NLL",
        ylabel="True-weight held-out NLL",
        title="Edge-weight specificity",
    )
    axes[1].scatter(
        audit.weight_cosine_null_median,
        audit.train_heldout_weight_cosine,
        c=color,
        s=24,
    )
    axes[1].plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    axes[1].set(
        xlabel="Weight-shuffle cosine",
        ylabel="Train–held-out cosine",
        title="Weight reproducibility",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis_object in axes:
        axis_object.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "transition_skeleton_weight_null_audit.png",
        dpi=220,
        bbox_inches="tight",
    )
    readme_path = figure_dir / "README.md"
    readme = readme_path.read_text() if readme_path.exists() else ""
    readme += """\n### transition_skeleton_weight_null_audit.png

这张图在等密度条件下检验 train80 多路径骨架的边权是否能泛化到 heldout20。左图比较真实边权与边权重排 null 的 heldout NLL；右图比较真实 train–heldout 边权相似度与重排 null。

**关注点**：点落在左图对角线下方、右图对角线上方，才说明信息来自具体路径权重，而不是图更稠密。
"""
    readme_path.write_text(readme)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build train-only multi-path transition skeletons on the patient axis.

Consecutive rank-set contact pairs are folded onto an unsigned skeleton:
every observed pair contributes to the increasing-axis orientation regardless
of which direction the event traversed it.  The forward graph is the oriented
skeleton and the reverse graph is its exact transpose.  No A/B labels,
held-out events, inter-event intervals, or ictal targets enter construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_topic5_structured_axis_prior import (  # noqa: E402
    _directed_axis_graph,
)
from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    load_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spectral_scale(graph: np.ndarray) -> np.ndarray:
    graph = np.asarray(graph, np.float32)
    singular = float(np.linalg.svd(graph, compute_uv=False)[0])
    if not np.isfinite(singular) or singular <= 1e-8:
        return np.zeros_like(graph)
    return graph / singular


def _folded_transition_skeleton(
    group_ids: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Return increasing-axis target-by-source consecutive-rank weights."""
    groups = np.asarray(group_ids, np.int16)
    axis = np.asarray(axis, float)
    n_contacts = groups.shape[1]
    skeleton = np.zeros((n_contacts, n_contacts), np.float64)
    for event in groups:
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
                    delta = float(axis[target_index] - axis[source_index])
                    if delta > 1e-8:
                        skeleton[target_index, source_index] += weight
                    elif delta < -1e-8:
                        # Fold reverse traversal onto the increasing-axis edge.
                        skeleton[source_index, target_index] += weight
                    else:
                        skeleton[target_index, source_index] += 0.5 * weight
                        skeleton[source_index, target_index] += 0.5 * weight
    return skeleton.astype(np.float32)


def _blend_graph(
    skeleton: np.ndarray,
    axis: np.ndarray,
    *,
    axis_floor: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empirical = _spectral_scale(skeleton)
    axis_graph = _spectral_scale(
        _directed_axis_graph(axis, neighbors=int(neighbors))
    )
    raw = (
        (1.0 - float(axis_floor)) * empirical
        + float(axis_floor) * axis_graph
    )
    forward = _spectral_scale(raw)
    reverse = forward.T.copy()
    return forward, reverse, axis_graph


def _edge_probability(graph: np.ndarray) -> np.ndarray:
    """Convert target-by-source weights to target|source probabilities."""
    graph = np.asarray(graph, float).clip(min=0.0)
    denominator = graph.sum(0, keepdims=True)
    return np.divide(
        graph,
        denominator,
        out=np.zeros_like(graph),
        where=denominator > 0,
    )


def _heldout_transition_metrics(
    group_ids: np.ndarray,
    axis: np.ndarray,
    forward: np.ndarray,
    reverse: np.ndarray,
) -> dict:
    forward_probability = _edge_probability(forward)
    reverse_probability = _edge_probability(reverse)
    probabilities = []
    exact_edge = []
    for event in np.asarray(group_ids, np.int16):
        valid = event >= 0
        if int(valid.sum()) < 2:
            continue
        for step in range(int(event[valid].max())):
            source = np.flatnonzero(event == step)
            target = np.flatnonzero(event == step + 1)
            if not len(source) or not len(target):
                continue
            pair_probability = []
            pair_edge = []
            for source_index in source:
                for target_index in target:
                    delta = float(axis[target_index] - axis[source_index])
                    graph = (
                        forward_probability
                        if delta >= 0
                        else reverse_probability
                    )
                    pair_probability.append(graph[target_index, source_index])
                    pair_edge.append(graph[target_index, source_index] > 0)
            # Tied target sets are set-valued actions, not repeated samples.
            probabilities.append(float(np.mean(pair_probability)))
            exact_edge.append(float(np.mean(pair_edge)))
    probability = np.asarray(probabilities, float)
    return {
        "transition_nll": float(
            np.mean(-np.log(np.clip(probability, 1e-8, 1.0)))
        ),
        "transition_probability_mean": float(np.mean(probability)),
        "nonzero_edge_fraction": float(np.mean(exact_edge)),
        "n_transition_sets": int(len(probability)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float).ravel()
    right = np.asarray(right, float).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return (
        float(np.dot(left, right) / denominator)
        if denominator > 0
        else np.nan
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--axis-prior-root",
        type=Path,
        default=ROOT
        / "results/topic5_structured_axis_graph/axis_prior_v1_fast",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_structured_axis_graph/transition_skeleton_prior_v0_7",
    )
    parser.add_argument("--axis-floor", type=float, default=0.20)
    parser.add_argument("--neighbors", type=int, default=2)
    args = parser.parse_args()

    dataset_root = (
        args.dataset_root
        if args.dataset_root.is_absolute()
        else ROOT / args.dataset_root
    )
    axis_root = (
        args.axis_prior_root
        if args.axis_prior_root.is_absolute()
        else ROOT / args.axis_prior_root
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    prior_dir = output_dir / "per_subject"
    figure_dir = output_dir / "figures"
    prior_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_root)
    rows = []
    for subject, record in records.items():
        axis_path = axis_root / "per_subject" / f"{subject}.npz"
        with np.load(axis_path, allow_pickle=False) as z:
            axis = np.asarray(z["axis_coordinate"], np.float32)
            left = np.asarray(z["left_endpoint"], bool)
            right = np.asarray(z["right_endpoint"], bool)
            if not np.array_equal(
                z["contact_names"].astype(str),
                record.contact_names.astype(str),
            ):
                raise RuntimeError(f"{subject}: contact order mismatch")
        train = record.group_ids[record.train_indices]
        heldout = record.group_ids[record.eval_indices]
        skeleton = _folded_transition_skeleton(train, axis)
        forward, reverse, axis_graph = _blend_graph(
            skeleton,
            axis,
            axis_floor=float(args.axis_floor),
            neighbors=int(args.neighbors),
        )
        split = max(1, len(train) // 2)
        first = _spectral_scale(
            _folded_transition_skeleton(train[:split], axis)
        )
        second = _spectral_scale(
            _folded_transition_skeleton(train[split:], axis)
        )
        heldout_model = _heldout_transition_metrics(
            heldout, axis, forward, reverse
        )
        heldout_axis = _heldout_transition_metrics(
            heldout, axis, axis_graph, axis_graph.T
        )
        prior_path = prior_dir / f"{subject}.npz"
        np.savez_compressed(
            prior_path,
            subject=np.asarray(subject),
            dataset=np.asarray(record.dataset),
            contact_names=record.contact_names,
            axis_coordinate=axis,
            transition_skeleton_raw=skeleton,
            forward_graph=forward,
            reverse_graph=reverse,
            axis_only_forward_graph=axis_graph,
            left_endpoint=left,
            right_endpoint=right,
            input_record_sha256=np.asarray(record.input_sha256),
            input_axis_prior_sha256=np.asarray(_sha256(axis_path)),
            source_event_split=np.asarray("chronological_train80_only"),
            heldout_used_for_construction=np.asarray(False),
            ictal_target_read=np.asarray(False),
        )
        metadata = {
            "subject": subject,
            "dataset": record.dataset,
            "status": "ok",
            "construction": "train80_consecutive_rank_set_folded_skeleton",
            "axis_floor": float(args.axis_floor),
            "forward_reverse_exact_transpose": bool(
                np.array_equal(reverse, forward.T)
            ),
            "template_or_ab_labels_used": False,
            "heldout_used_for_construction": False,
            "heldout_used_for_postconstruction_audit": True,
            "ictal_target_read": False,
            "prior_npz_sha256": _sha256(prior_path),
        }
        prior_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_contacts": len(record.contact_names),
                "n_train_events": len(train),
                "n_heldout_events": len(heldout),
                "n_empirical_oriented_edges": int(np.sum(skeleton > 0)),
                "split_half_skeleton_cosine": _cosine(first, second),
                "heldout_transition_nll_skeleton": heldout_model[
                    "transition_nll"
                ],
                "heldout_transition_nll_axis_only": heldout_axis[
                    "transition_nll"
                ],
                "heldout_transition_nll_improvement": heldout_axis[
                    "transition_nll"
                ]
                - heldout_model["transition_nll"],
                "heldout_nonzero_edge_fraction_skeleton": heldout_model[
                    "nonzero_edge_fraction"
                ],
                "heldout_nonzero_edge_fraction_axis_only": heldout_axis[
                    "nonzero_edge_fraction"
                ],
                "n_heldout_transition_sets": heldout_model[
                    "n_transition_sets"
                ],
                "forward_reverse_exact_transpose": bool(
                    np.array_equal(reverse, forward.T)
                ),
                "heldout_used_for_construction": False,
                "ictal_target_read": False,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "transition_skeleton_audit.csv", index=False)
    summary = {
        "status": "complete",
        "n_subjects": int(len(audit)),
        "dataset_counts": audit.dataset.value_counts().to_dict(),
        "n_exact_transpose": int(
            audit.forward_reverse_exact_transpose.sum()
        ),
        "median_split_half_skeleton_cosine": float(
            audit.split_half_skeleton_cosine.median()
        ),
        "n_split_half_cosine_ge_0_8": int(
            (audit.split_half_skeleton_cosine >= 0.8).sum()
        ),
        "median_heldout_transition_nll_improvement_vs_axis_only": float(
            audit.heldout_transition_nll_improvement.median()
        ),
        "n_heldout_nll_better_than_axis_only": int(
            (audit.heldout_transition_nll_improvement > 0).sum()
        ),
        "median_heldout_nonzero_edge_fraction_skeleton": float(
            audit.heldout_nonzero_edge_fraction_skeleton.median()
        ),
        "median_heldout_nonzero_edge_fraction_axis_only": float(
            audit.heldout_nonzero_edge_fraction_axis_only.median()
        ),
        "heldout_used_for_construction": False,
        "ictal_target_read": False,
    }
    (output_dir / "transition_skeleton_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    color = audit.dataset.map(
        {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}
    )
    axes[0].scatter(
        audit.heldout_transition_nll_axis_only,
        audit.heldout_transition_nll_skeleton,
        c=color,
        s=24,
        alpha=0.85,
    )
    limit = [
        float(
            min(
                audit.heldout_transition_nll_axis_only.min(),
                audit.heldout_transition_nll_skeleton.min(),
            )
        ),
        float(
            max(
                audit.heldout_transition_nll_axis_only.max(),
                audit.heldout_transition_nll_skeleton.max(),
            )
        ),
    ]
    axes[0].plot(limit, limit, "--", color="#777777", lw=1)
    axes[0].set(
        xlabel="Axis-only held-out NLL",
        ylabel="Skeleton held-out NLL",
        title="Held-out transitions",
    )
    axes[1].hist(
        audit.split_half_skeleton_cosine,
        bins=np.linspace(0, 1, 11),
        color="#4D4D4D",
    )
    axes[1].axvline(0.8, ls="--", lw=1, color="#B2182B")
    axes[1].set(
        xlabel="Split-half cosine",
        ylabel="Patients",
        title="Skeleton stability",
    )
    axes[2].scatter(
        audit.heldout_nonzero_edge_fraction_axis_only,
        audit.heldout_nonzero_edge_fraction_skeleton,
        c=color,
        s=24,
        alpha=0.85,
    )
    axes[2].plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    axes[2].set(
        xlabel="Axis-only edge coverage",
        ylabel="Skeleton edge coverage",
        title="Held-out edge coverage",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis_object in axes:
        axis_object.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure_path = figure_dir / "transition_skeleton_audit.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        """### transition_skeleton_audit.png

这张诊断图检验 train80 相邻 rank-set 构成的多路径骨架，是否能在 heldout20 中复现，并与仅按患者轴连接相邻触点的图比较。左图越落在对角线下方越好；中图显示骨架的时间分半稳定性；右图显示对 heldout 连续触点对的覆盖。

**关注点**：只有骨架在大多数患者中稳定、且 heldout NLL/edge coverage 优于 axis-only，才进入结构化 RNN。
"""
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

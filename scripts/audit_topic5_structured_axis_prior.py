#!/usr/bin/env python3
"""Build and audit train-only bidirectional propagation-axis priors.

The paired template labels are arbitrary.  Only the unsigned patient axis,
its two endpoint cores, and the forward/reverse graph pair are consumed by the
structured model.  Ictal targets and chronological held-out events are sealed.
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
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.lagpat_rank_audit import (  # noqa: E402
    build_masked_kmeans_features,
    mask_phantom_ranks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_abs_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 4:
        return np.nan
    return float(abs(spearmanr(left[valid], right[valid]).statistic))


def _derive_axis(
    group_ids: np.ndarray,
    *,
    seed: int,
    max_fit_events: int,
) -> dict[str, np.ndarray | float | int]:
    groups = np.asarray(group_ids, np.int16)
    participating = groups >= 0
    ranks = groups.astype(float)
    features = build_masked_kmeans_features(
        ranks.T, participating.T, impute="event_median"
    )
    rng = np.random.default_rng(seed)
    if len(features) > max_fit_events:
        fit_indices = rng.choice(len(features), max_fit_events, replace=False)
        fit_features = features[fit_indices]
    else:
        fit_features = features
    model = MiniBatchKMeans(
        n_clusters=2,
        n_init=10,
        batch_size=min(2048, len(fit_features)),
        random_state=seed,
        reassignment_ratio=0.01,
    )
    model.fit(fit_features)
    labels = model.predict(features)
    masked = mask_phantom_ranks(
        ranks.T, participating.T, normalize=True
    )

    templates = []
    support = []
    for cluster in [0, 1]:
        selected = labels == cluster
        with np.errstate(all="ignore"):
            templates.append(np.nanmedian(masked[:, selected], axis=1))
        support.append(np.mean(participating[selected], axis=0))
    template_a, template_b = templates
    support_a, support_b = support
    joint_support = np.minimum(support_a, support_b)
    joint_valid = (
        np.isfinite(template_a)
        & np.isfinite(template_b)
        & (joint_support >= 0.02)
    )
    opposition = (
        -float(spearmanr(template_a[joint_valid], template_b[joint_valid]).statistic)
        if int(joint_valid.sum()) >= 4
        else np.nan
    )

    filled_a = np.where(np.isfinite(template_a), template_a, 0.5)
    filled_b = np.where(np.isfinite(template_b), template_b, 0.5)
    raw_axis = filled_a - filled_b
    reliability_weight = np.maximum(joint_support, 1e-3)
    raw_axis -= np.average(raw_axis, weights=reliability_weight)
    scale = float(np.max(np.abs(raw_axis)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise RuntimeError("train-only paired templates did not define an axis")
    axis = raw_axis / scale

    event_polarity = (features - 0.5) @ axis
    nonzero = event_polarity[np.abs(event_polarity) > 1e-12]
    polarity_balance = (
        float(min(np.mean(nonzero > 0), np.mean(nonzero < 0)))
        if len(nonzero)
        else 0.0
    )
    cluster_balance = float(min(np.mean(labels == 0), np.mean(labels == 1)))
    return {
        "axis": axis.astype(np.float32),
        "template_a": filled_a.astype(np.float32),
        "template_b": filled_b.astype(np.float32),
        "support_a": support_a.astype(np.float32),
        "support_b": support_b.astype(np.float32),
        "joint_valid": joint_valid.astype(bool),
        "cluster_labels": labels.astype(np.int8),
        "cluster_balance": cluster_balance,
        "polarity_balance": polarity_balance,
        "template_opposition": opposition,
        "n_joint_valid_contacts": int(joint_valid.sum()),
        "n_fit_events": int(len(fit_features)),
    }


def _directed_axis_graph(axis: np.ndarray, neighbors: int = 2) -> np.ndarray:
    """Return target-by-source graph for increasing axis propagation."""
    axis = np.asarray(axis, float)
    n_contacts = len(axis)
    order = np.argsort(axis, kind="stable")
    graph = np.zeros((n_contacts, n_contacts), np.float32)
    positive_steps = np.diff(np.sort(np.unique(axis)))
    positive_steps = positive_steps[positive_steps > 1e-6]
    scale = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    for source_position, source in enumerate(order):
        for offset in range(1, int(neighbors) + 1):
            target_position = source_position + offset
            if target_position >= n_contacts:
                break
            target = int(order[target_position])
            distance = max(float(axis[target] - axis[source]), 0.0)
            graph[target, source] = np.exp(-distance / max(scale, 1e-3))
    row_sum = graph.sum(axis=1, keepdims=True)
    graph = np.divide(graph, row_sum, out=np.zeros_like(graph), where=row_sum > 0)
    return graph


def _endpoint_masks(
    axis: np.ndarray, joint_support: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    n_contacts = len(axis)
    endpoint_k = max(1, min(3, n_contacts // 4))
    eligible = np.flatnonzero(joint_support >= 0.02)
    if len(eligible) < 2 * endpoint_k:
        eligible = np.arange(n_contacts)
    ordered = eligible[np.argsort(axis[eligible], kind="stable")]
    left = np.zeros(n_contacts, bool)
    right = np.zeros(n_contacts, bool)
    left[ordered[:endpoint_k]] = True
    right[ordered[-endpoint_k:]] = True
    return left, right, endpoint_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-events", type=int, default=20_000)
    parser.add_argument("--neighbors", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_dir = args.output_dir / "per_subject"
    figure_dir = args.output_dir / "figures"
    prior_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.dataset_root)
    rows = []

    for subject, record in records.items():
        train_groups = record.group_ids[record.train_indices]
        full = _derive_axis(
            train_groups, seed=20260726, max_fit_events=args.max_fit_events
        )
        split_at = max(2, len(train_groups) // 2)
        first = _derive_axis(
            train_groups[:split_at],
            seed=20260727,
            max_fit_events=args.max_fit_events,
        )
        second = _derive_axis(
            train_groups[split_at:],
            seed=20260728,
            max_fit_events=args.max_fit_events,
        )
        seed_axes = [
            _derive_axis(
                train_groups,
                seed=seed,
                max_fit_events=args.max_fit_events,
            )["axis"]
            for seed in [20260729, 20260730, 20260731]
        ]
        pairwise_seed = [
            _safe_abs_spearman(seed_axes[left], seed_axes[right])
            for left in range(len(seed_axes))
            for right in range(left + 1, len(seed_axes))
        ]
        forward = _directed_axis_graph(
            full["axis"], neighbors=int(args.neighbors)
        )
        reverse = forward.T.copy()
        reverse_row_sum = reverse.sum(axis=1, keepdims=True)
        reverse = np.divide(
            reverse,
            reverse_row_sum,
            out=np.zeros_like(reverse),
            where=reverse_row_sum > 0,
        )
        joint_support = np.minimum(full["support_a"], full["support_b"])
        left, right, endpoint_k = _endpoint_masks(
            full["axis"], joint_support
        )
        prior_path = prior_dir / f"{subject}.npz"
        np.savez_compressed(
            prior_path,
            subject=np.asarray(subject),
            dataset=np.asarray(record.dataset),
            contact_names=record.contact_names,
            axis_coordinate=full["axis"],
            forward_graph=forward,
            reverse_graph=reverse,
            left_endpoint=left,
            right_endpoint=right,
            template_a=full["template_a"],
            template_b=full["template_b"],
            support_a=full["support_a"],
            support_b=full["support_b"],
            joint_valid=full["joint_valid"],
            input_record_sha256=np.asarray(record.input_sha256),
            source_event_split=np.asarray("chronological_train80_only"),
            ictal_target_read=np.asarray(False),
        )
        metadata = {
            "subject": subject,
            "dataset": record.dataset,
            "status": "ok",
            "source": "chronological_train80_masked_rank_events",
            "heldout_events_read": False,
            "ictal_target_read": False,
            "template_labels_are_arbitrary": True,
            "graph_pair_is_label_swap_invariant": True,
            "endpoint_k": endpoint_k,
            "prior_npz_sha256": _sha256(prior_path),
            "input_record_sha256": record.input_sha256,
        }
        prior_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "status": "ok",
                "n_contacts": len(record.contact_names),
                "n_train_events": len(train_groups),
                "cluster_balance": full["cluster_balance"],
                "event_polarity_balance": full["polarity_balance"],
                "template_opposition": full["template_opposition"],
                "n_joint_valid_contacts": full["n_joint_valid_contacts"],
                "split_half_axis_abs_spearman": _safe_abs_spearman(
                    first["axis"], second["axis"]
                ),
                "kmeans_seed_axis_abs_spearman_median": float(
                    np.nanmedian(pairwise_seed)
                ),
                "endpoint_k": endpoint_k,
                "heldout_events_read": False,
                "ictal_target_read": False,
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(args.output_dir / "axis_prior_audit.csv", index=False)
    summary = {
        "status": "complete",
        "n_subjects": int(len(audit)),
        "dataset_counts": audit["dataset"].value_counts().to_dict(),
        "median_cluster_balance": float(audit["cluster_balance"].median()),
        "median_event_polarity_balance": float(
            audit["event_polarity_balance"].median()
        ),
        "median_template_opposition": float(
            audit["template_opposition"].median()
        ),
        "median_split_half_axis_abs_spearman": float(
            audit["split_half_axis_abs_spearman"].median()
        ),
        "median_kmeans_seed_axis_abs_spearman": float(
            audit["kmeans_seed_axis_abs_spearman_median"].median()
        ),
        "n_split_half_axis_rho_ge_0_5": int(
            (audit["split_half_axis_abs_spearman"] >= 0.5).sum()
        ),
        "n_seed_axis_rho_ge_0_5": int(
            (audit["kmeans_seed_axis_abs_spearman_median"] >= 0.5).sum()
        ),
        "heldout_events_read": False,
        "ictal_target_read": False,
    }
    (args.output_dir / "axis_prior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.45))
    metrics = [
        ("template_opposition", "Template opposition", "A"),
        ("split_half_axis_abs_spearman", "Split-half axis |ρ|", "B"),
        (
            "kmeans_seed_axis_abs_spearman_median",
            "KMeans-seed axis |ρ|",
            "C",
        ),
    ]
    colors = {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}
    for ax, (column, label, letter) in zip(axes, metrics):
        for dataset, frame in audit.groupby("dataset"):
            ax.scatter(
                np.repeat(0 if dataset == "epilepsiae" else 1, len(frame))
                + np.linspace(-0.12, 0.12, len(frame)),
                frame[column],
                s=20,
                alpha=0.65,
                color=colors[dataset],
                label=dataset.capitalize(),
            )
        ax.axhline(0.5, color="#555555", ls="--", lw=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Epilepsiae", "Yuquan"])
        ax.set_ylabel(label)
        ax.set_title(f"{letter}  {label}", loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.3)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"structured_axis_prior_audit.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        """### structured_axis_prior_audit.png

这张图只使用每名患者按时间划分的前80%间期事件，检查双向模板是否形成相反的触点轴、该轴能否跨时间复现，以及KMeans初始化是否改变轴。模板标签本身任意，正式模型只读取无符号轴、两端source core和互为转置的正反向图。

**关注点**：held-out事件和发作期数据均未读取；低稳定病例不删除，而是在全队列和高稳定性敏感性层分别报告。
"""
    )
    (args.output_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_subjects": int(len(audit)),
                "heldout_events_read": False,
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

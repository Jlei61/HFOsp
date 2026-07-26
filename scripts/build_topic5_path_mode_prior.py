#!/usr/bin/env python3
"""Build train80-only nonnegative patient path-mode graph priors."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_transition_skeleton_prior import (
    _blend_graph,
    _folded_transition_skeleton,
    _sha256,
)
from scripts.train_topic5_interictal_rank_distribution import load_records


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(
        right, axis=1
    )[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def event_transition_vectors(
    group_ids: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-event-weight canonical transition vectors.

    Each row is one event with at least one consecutive-rank transition.
    Edges are target-by-source and folded to increasing axis direction.
    """
    groups = np.asarray(group_ids, np.int16)
    axis = np.asarray(axis, float)
    n_contacts = groups.shape[1]
    vectors = []
    source_rows = []
    for event_index, event in enumerate(groups):
        valid = event >= 0
        if int(valid.sum()) < 2:
            continue
        vector = np.zeros((n_contacts, n_contacts), np.float32)
        for step in range(int(event[valid].max())):
            source = np.flatnonzero(event == step)
            target = np.flatnonzero(event == step + 1)
            if not len(source) or not len(target):
                continue
            pair_weight = 1.0 / float(len(source) * len(target))
            for source_index in source:
                for target_index in target:
                    delta = float(axis[target_index] - axis[source_index])
                    if delta > 1e-8:
                        vector[target_index, source_index] += pair_weight
                    elif delta < -1e-8:
                        vector[source_index, target_index] += pair_weight
                    else:
                        vector[target_index, source_index] += 0.5 * pair_weight
                        vector[source_index, target_index] += 0.5 * pair_weight
        total = float(vector.sum())
        if total > 0:
            vectors.append((vector / total).reshape(-1))
            source_rows.append(event_index)
    if not vectors:
        return (
            np.zeros((0, n_contacts * n_contacts), np.float32),
            np.zeros(0, np.int64),
        )
    return np.row_stack(vectors), np.asarray(source_rows, np.int64)


def _normalize_bases(bases: np.ndarray) -> np.ndarray:
    bases = np.asarray(bases, np.float64).clip(min=0.0)
    denominator = bases.sum(1, keepdims=True)
    if np.any(denominator <= 1e-12):
        raise RuntimeError("path-mode factorization produced an empty mode")
    return (bases / denominator).astype(np.float32)


def factor_path_modes(
    vectors: np.ndarray,
    *,
    mode_count: int,
    aggregate_skeleton: np.ndarray,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Factor edge co-occurrence into nonnegative event-level path bases."""
    vectors = np.asarray(vectors, np.float32)
    mode_count = int(mode_count)
    if mode_count < 1:
        raise ValueError("mode_count must be positive")
    n_edges = aggregate_skeleton.size
    if vectors.ndim != 2 or vectors.shape[1] != n_edges:
        raise ValueError("transition vector shape mismatch")
    if mode_count == 1:
        bases = _normalize_bases(
            np.asarray(aggregate_skeleton, np.float32).reshape(1, -1)
        )
        prior = np.ones(1, np.float32)
        return bases, prior, {
            "nmf_iterations": 0,
            "nmf_reconstruction_error": 0.0,
        }
    if len(vectors) < mode_count:
        raise RuntimeError("fewer transition events than requested modes")
    cooccurrence = (vectors.T @ vectors) / float(len(vectors))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        factorizer = NMF(
            n_components=mode_count,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            tol=1e-5,
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        factorizer.fit_transform(cooccurrence)
    bases = _normalize_bases(factorizer.components_)
    similarity = _cosine_rows(vectors, bases)
    # Soft event assignment is used only to define the train80 mode prior.
    # The RNN never consumes these event-wise scores or hard labels.
    score_sum = similarity.sum(0)
    if float(score_sum.sum()) <= 1e-12:
        prior = np.full(mode_count, 1.0 / mode_count, np.float32)
    else:
        prior = (score_sum / score_sum.sum()).astype(np.float32)
    order = np.lexsort(
        (
            np.argmax(bases, axis=1),
            -prior,
        )
    )
    bases = bases[order]
    prior = prior[order]
    return bases, prior, {
        "nmf_iterations": int(factorizer.n_iter_),
        "nmf_reconstruction_error": float(
            factorizer.reconstruction_err_
        ),
    }


def _mode_fit_metrics(vectors: np.ndarray, bases: np.ndarray) -> dict:
    if not len(vectors):
        return {
            "max_mode_cosine_median": np.nan,
            "soft_reconstruction_cosine_median": np.nan,
        }
    similarity = _cosine_rows(vectors, bases)
    weights = np.clip(similarity, 0.0, None)
    denominator = weights.sum(1, keepdims=True)
    weights = np.divide(
        weights,
        denominator,
        out=np.full_like(weights, 1.0 / bases.shape[0]),
        where=denominator > 1e-12,
    )
    reconstructed = weights @ bases
    numerator = np.sum(vectors * reconstructed, axis=1)
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(
        reconstructed, axis=1
    )
    reconstruction_cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    reconstruction_cosine = np.clip(reconstruction_cosine, -1.0, 1.0)
    return {
        "max_mode_cosine_median": float(
            np.median(np.max(similarity, axis=1))
        ),
        "soft_reconstruction_cosine_median": float(
            np.median(reconstruction_cosine)
        ),
    }


def _aligned_mode_cosine(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    similarity = _cosine_rows(first, second)
    left, right = linear_sum_assignment(-similarity)
    aligned = similarity[left, right]
    return float(np.median(aligned)), float(np.min(aligned))


def _pairwise_mode_cosine(bases: np.ndarray) -> float:
    if len(bases) == 1:
        return 0.0
    similarity = _cosine_rows(bases, bases)
    upper = similarity[np.triu_indices(len(bases), k=1)]
    return float(np.median(upper))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_persistent_path_mode_rnn_v0_9.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    output_dir = _resolve(
        args.output_dir
        if args.output_dir is not None
        else Path(cfg["outputs"]["prior"])
    )
    dataset_root = _resolve(Path(cfg["inputs"]["dataset"]))
    axis_root = _resolve(Path(cfg["inputs"]["axis_prior"]))
    aggregate_root = _resolve(Path(cfg["inputs"]["aggregate_prior"]))
    prior_cfg = cfg["prior"]
    mode_counts = [int(value) for value in prior_cfg["mode_counts"]]
    factorization_seed = int(prior_cfg["factorization_seed"])
    max_iter = int(prior_cfg["nmf_max_iter"])
    axis_floor = float(prior_cfg["axis_floor"])
    neighbors = int(prior_cfg["neighbors"])
    records = load_records(dataset_root)
    rows = []
    for subject, record in records.items():
        axis_path = axis_root / "per_subject" / f"{subject}.npz"
        aggregate_path = aggregate_root / "per_subject" / f"{subject}.npz"
        with np.load(axis_path, allow_pickle=False) as z:
            axis = np.asarray(z["axis_coordinate"], np.float32)
            left = np.asarray(z["left_endpoint"], bool)
            right = np.asarray(z["right_endpoint"], bool)
            axis_names = np.asarray(z["contact_names"]).astype(str)
        with np.load(aggregate_path, allow_pickle=False) as z:
            aggregate_skeleton = np.asarray(
                z["transition_skeleton_raw"], np.float32
            )
            aggregate_forward = np.asarray(z["forward_graph"], np.float32)
            aggregate_reverse = np.asarray(z["reverse_graph"], np.float32)
            aggregate_names = np.asarray(z["contact_names"]).astype(str)
        names = record.contact_names.astype(str)
        if not np.array_equal(axis_names, names) or not np.array_equal(
            aggregate_names, names
        ):
            raise RuntimeError(f"{subject}: contact order mismatch")
        train_groups = record.group_ids[record.train_indices]
        heldout_groups = record.group_ids[record.eval_indices]
        train_vectors, _ = event_transition_vectors(train_groups, axis)
        heldout_vectors, _ = event_transition_vectors(heldout_groups, axis)
        split = max(1, len(train_groups) // 2)
        first_vectors, _ = event_transition_vectors(
            train_groups[:split], axis
        )
        second_vectors, _ = event_transition_vectors(
            train_groups[split:], axis
        )
        for mode_count in mode_counts:
            subject_seed = int(
                hashlib.sha256(
                    f"{subject}:{mode_count}:{factorization_seed}".encode()
                ).hexdigest()[:8],
                16,
            )
            bases, mode_prior, fit = factor_path_modes(
                train_vectors,
                mode_count=mode_count,
                aggregate_skeleton=aggregate_skeleton,
                seed=subject_seed,
                max_iter=max_iter,
            )
            first_bases, _, _ = factor_path_modes(
                first_vectors,
                mode_count=mode_count,
                aggregate_skeleton=_folded_transition_skeleton(
                    train_groups[:split], axis
                ),
                seed=subject_seed,
                max_iter=max_iter,
            )
            second_bases, _, _ = factor_path_modes(
                second_vectors,
                mode_count=mode_count,
                aggregate_skeleton=_folded_transition_skeleton(
                    train_groups[split:], axis
                ),
                seed=subject_seed,
                max_iter=max_iter,
            )
            aligned_median, aligned_min = _aligned_mode_cosine(
                first_bases, second_bases
            )
            n_contacts = len(names)
            raw_modes = bases.reshape(mode_count, n_contacts, n_contacts)
            forward_graphs = []
            reverse_graphs = []
            for raw_mode in raw_modes:
                forward, reverse, _ = _blend_graph(
                    raw_mode,
                    axis,
                    axis_floor=axis_floor,
                    neighbors=neighbors,
                )
                forward_graphs.append(forward)
                reverse_graphs.append(reverse)
            forward_graphs = np.stack(forward_graphs).astype(np.float32)
            reverse_graphs = np.stack(reverse_graphs).astype(np.float32)
            if not np.array_equal(
                reverse_graphs, forward_graphs.transpose(0, 2, 1)
            ):
                raise RuntimeError(f"{subject} K={mode_count}: transpose fail")
            k_dir = output_dir / f"k_{mode_count}" / "per_subject"
            k_dir.mkdir(parents=True, exist_ok=True)
            prior_path = k_dir / f"{subject}.npz"
            np.savez_compressed(
                prior_path,
                subject=np.asarray(subject),
                dataset=np.asarray(record.dataset),
                contact_names=record.contact_names,
                axis_coordinate=axis,
                mode_skeleton_raw=raw_modes,
                mode_forward_graphs=forward_graphs,
                mode_reverse_graphs=reverse_graphs,
                mode_prior=mode_prior,
                aggregate_forward_graph=aggregate_forward,
                aggregate_reverse_graph=aggregate_reverse,
                left_endpoint=left,
                right_endpoint=right,
                input_record_sha256=np.asarray(record.input_sha256),
                input_axis_prior_sha256=np.asarray(_sha256(axis_path)),
                input_aggregate_prior_sha256=np.asarray(
                    _sha256(aggregate_path)
                ),
                source_event_split=np.asarray(
                    "chronological_train80_only"
                ),
                heldout_used_for_construction=np.asarray(False),
                ab_labels_used=np.asarray(False),
                iei_used=np.asarray(False),
                ictal_target_read=np.asarray(False),
                factorization_seed=np.asarray(factorization_seed),
            )
            train_fit = _mode_fit_metrics(train_vectors, bases)
            heldout_fit = _mode_fit_metrics(heldout_vectors, bases)
            weighted_basis = np.sum(
                mode_prior[:, None] * bases, axis=0, keepdims=True
            )
            aggregate_basis = _normalize_bases(
                aggregate_skeleton.reshape(1, -1)
            )
            rows.append(
                {
                    "subject": subject,
                    "dataset": record.dataset,
                    "mode_count": mode_count,
                    "n_contacts": n_contacts,
                    "n_train_events": len(train_groups),
                    "n_train_transition_events": len(train_vectors),
                    "n_heldout_events": len(heldout_groups),
                    "n_heldout_transition_events": len(heldout_vectors),
                    "mode_prior_min": float(np.min(mode_prior)),
                    "mode_prior_entropy": float(
                        -np.sum(
                            mode_prior
                            * np.log(np.clip(mode_prior, 1e-12, 1.0))
                        )
                    ),
                    "pairwise_mode_cosine_median": _pairwise_mode_cosine(
                        bases
                    ),
                    "split_half_aligned_mode_cosine_median": aligned_median,
                    "split_half_aligned_mode_cosine_min": aligned_min,
                    "aggregate_reconstruction_cosine": float(
                        _cosine_rows(weighted_basis, aggregate_basis)[0, 0]
                    ),
                    "train_max_mode_cosine_median": train_fit[
                        "max_mode_cosine_median"
                    ],
                    "train_soft_reconstruction_cosine_median": train_fit[
                        "soft_reconstruction_cosine_median"
                    ],
                    "heldout_max_mode_cosine_median": heldout_fit[
                        "max_mode_cosine_median"
                    ],
                    "heldout_soft_reconstruction_cosine_median": heldout_fit[
                        "soft_reconstruction_cosine_median"
                    ],
                    "nmf_iterations": fit["nmf_iterations"],
                    "nmf_reconstruction_error": fit[
                        "nmf_reconstruction_error"
                    ],
                    "forward_reverse_exact_transpose": True,
                    "heldout_used_for_construction": False,
                    "ab_labels_used": False,
                    "iei_used": False,
                    "ictal_target_read": False,
                    "prior_npz_sha256": _sha256(prior_path),
                }
            )
            prior_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "subject": subject,
                        "dataset": record.dataset,
                        "mode_count": mode_count,
                        "construction": (
                            "train80_edge_cooccurrence_nonnegative_modes"
                        ),
                        "component_semantics": (
                            "event_persistent_path_mode_x_direction"
                        ),
                        "hard_event_labels_written": False,
                        "heldout_used_for_construction": False,
                        "ab_labels_used": False,
                        "iei_used": False,
                        "ictal_target_read": False,
                        "prior_npz_sha256": _sha256(prior_path),
                    },
                    indent=2,
                )
            )
    audit = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "path_mode_prior_audit.csv", index=False)
    summary = {
        "status": "complete",
        "n_subjects": int(audit.subject.nunique()),
        "mode_counts": mode_counts,
        "n_rows": int(len(audit)),
        "dataset_counts": (
            audit.drop_duplicates("subject").dataset.value_counts().to_dict()
        ),
        "heldout_used_for_construction": False,
        "ab_labels_used": False,
        "iei_used": False,
        "ictal_target_read": False,
        "by_mode_count": {
            str(mode_count): {
                "median_split_half_aligned_cosine": float(
                    group.split_half_aligned_mode_cosine_median.median()
                ),
                "median_heldout_reconstruction_cosine": float(
                    group.heldout_soft_reconstruction_cosine_median.median()
                ),
                "median_pairwise_mode_cosine": float(
                    group.pairwise_mode_cosine_median.median()
                ),
                "minimum_mode_prior": float(group.mode_prior_min.min()),
            }
            for mode_count, group in audit.groupby("mode_count")
        },
    }
    (output_dir / "path_mode_prior_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    colors = audit.dataset.map(
        {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}
    )
    jitter_rng = np.random.default_rng(20260726)
    for panel, column, ylabel in [
        (
            axes[0],
            "split_half_aligned_mode_cosine_median",
            "Split-half aligned cosine",
        ),
        (
            axes[1],
            "heldout_soft_reconstruction_cosine_median",
            "Held-out reconstruction cosine",
        ),
        (
            axes[2],
            "pairwise_mode_cosine_median",
            "Within-patient mode cosine",
        ),
    ]:
        x = audit.mode_count.to_numpy(float) + jitter_rng.uniform(
            -0.08, 0.08, size=len(audit)
        )
        panel.scatter(x, audit[column], c=colors, s=12, alpha=0.58)
        medians = audit.groupby("mode_count")[column].median()
        panel.plot(
            medians.index,
            medians.values,
            color="#202020",
            marker="o",
            lw=1.4,
        )
        panel.set_xticks(mode_counts)
        panel.set_xlabel("Path modes K")
        panel.set_ylabel(ylabel)
        panel.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "path_mode_prior_audit.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        """### path_mode_prior_audit.png

三联图检查 train80-only 非负路径模式是否可解释为患者内持续路径，而不是任意矩阵分解。左图是 train80 时间分半后经最优匹配的 mode cosine；中图是冻结 modes 对 heldout20 事件路径的重建；右图是同一患者不同 modes 的相似度，过高表示模式塌缩。

**关注点**：K 增加时必须保持 split-half/heldout 稳定，同时不同 modes 不能全部退化为同一张 aggregate graph；本图只审计先验，不读取发作数据。
"""
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

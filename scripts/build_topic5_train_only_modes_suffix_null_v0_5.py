#!/usr/bin/env python3
"""Freeze train-only modes, prefix templates, and suffix-pairing nulls.

The v0.3 cache used full-record adaptive-cluster labels to decide which events
entered ``own_a`` and ``own_b`` fits.  That makes those split-plane fits
ineligible for strict held-out inference.  This builder replaces that join:

* K=2 is fitted only on development-training events;
* every fit trains/evaluates all eligible events, so a mode classifier cannot
  create empty held-out denominators;
* first-three-rank posteriors are frozen for template/flow stratification only;
* three deterministic cross-event suffix reassignments preserve the prefix and
  never cross train/validation/test boundaries.

No early-ictal artifact is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import build_cluster_templates  # noqa: E402
from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
SEEDS = (2026081301, 2026081302, 2026081303)
PREFIX_RANKS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def features(ranks: np.ndarray, prefix_ranks: int | None = None) -> np.ndarray:
    values = np.asarray(ranks, dtype=np.int16)
    present = values >= 0
    if prefix_ranks is not None:
        present &= values < int(prefix_ranks)
    return build_masked_kmeans_features(
        values.T.astype(float), present.T, impute="event_median"
    )


def train_only_modes(ranks: np.ndarray, base_split: np.ndarray) -> dict:
    train_index = np.flatnonzero(np.asarray(base_split) == 0)
    if len(train_index) < 20:
        raise RuntimeError("fewer than 20 train events cannot define K=2 modes")
    train_features = features(ranks[train_index])
    model = KMeans(n_clusters=2, n_init=10, random_state=0)
    train_labels = model.fit_predict(train_features)
    if np.min(np.bincount(train_labels, minlength=2)) < 5:
        raise RuntimeError("train-only K=2 produced a cluster with fewer than 5 events")

    bools = ranks[train_index].T >= 0
    templates = build_cluster_templates(
        ranks[train_index].T.astype(float), bools, train_labels, 2,
    )
    prefix_features = features(ranks, prefix_ranks=PREFIX_RANKS)
    squared = np.mean(
        (prefix_features[:, None, :] - model.cluster_centers_[None, :, :]) ** 2,
        axis=2,
    )
    train_margin = np.abs(squared[train_index, 0] - squared[train_index, 1])
    temperature = max(float(np.median(train_margin)), 1e-6)
    posterior = softmax(-squared / temperature, axis=1)
    prefix_mode = np.argmax(posterior, axis=1).astype(np.int8)
    full_train_mode = np.full(len(ranks), -1, dtype=np.int8)
    full_train_mode[train_index] = train_labels.astype(np.int8)
    entropy = -np.sum(posterior * np.log(np.maximum(posterior, 1e-12)), axis=1)
    return {
        "train_index": train_index,
        "full_train_mode": full_train_mode,
        "prefix_mode": prefix_mode,
        "prefix_posterior": posterior.astype(np.float32),
        "prefix_entropy": entropy.astype(np.float32),
        "templates": templates.astype(np.float32),
        "centers": model.cluster_centers_.astype(np.float32),
        "temperature": temperature,
        "train_counts": np.bincount(train_labels, minlength=2).astype(int),
    }


def scope_cluster(templates: np.ndarray, contacts_xy: np.ndarray) -> tuple[int, list[float]]:
    along = np.asarray(contacts_xy, float)[:, 0]
    scores = []
    for template in np.asarray(templates, float):
        finite = np.isfinite(template) & np.isfinite(along)
        if int(finite.sum()) < 3:
            scores.append(float("nan"))
            continue
        value = spearmanr(template[finite], along[finite]).statistic
        scores.append(abs(float(value)) if np.isfinite(value) else float("nan"))
    if not np.isfinite(scores).any():
        raise RuntimeError("neither train-only mode aligns with the frozen fit plane")
    return int(np.nanargmax(scores)), scores


def scope_split(
    base_split: np.ndarray,
    scope: str,
    modes: dict,
    own_cluster: int | None,
) -> np.ndarray:
    """Keep the prediction task identical across planes.

    ``own_a``/``own_b`` identify two retrospective geometry views, not two
    labels that may filter the held-out task.  Mode labels are used only by
    downstream stratified analyses.
    """
    base = np.asarray(base_split, dtype=np.int8)
    if scope != "shared" and own_cluster is None:
        raise ValueError("split-plane fit needs a descriptive own_cluster")
    return base.copy()


def _roll_derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(indices, dtype=int).copy()
    if len(values) < 2:
        return values
    for _ in range(100):
        candidate = rng.permutation(values)
        if np.all(candidate != values):
            return candidate
    return np.roll(values, 1)


def suffix_mapping(
    ranks: np.ndarray,
    split: np.ndarray,
    mode: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Map recipients to compatible same-split/mode suffix donors.

    Donors have the same number of suffix rank sets and their suffix contacts
    cannot overlap the recipient prefix.  Donor reuse is allowed because an
    exact one-to-one derangement is often impossible in a small stratum; self
    mapping is never allowed.
    """
    rng = np.random.default_rng(int(seed))
    values = np.asarray(ranks)
    n_groups = np.asarray([
        len(np.unique(row[row >= 0])) for row in values
    ], dtype=int)
    n_contacts = np.sum(values >= 0, axis=1)
    n_suffix_groups = np.maximum(n_groups - PREFIX_RANKS, 0)
    n_suffix_contacts = np.sum(values >= PREFIX_RANKS, axis=1)
    prefix_masks = (values >= 0) & (values < PREFIX_RANKS)
    suffix_masks = values >= PREFIX_RANKS
    mapping = np.arange(len(values), dtype=np.int64)
    eligible = np.flatnonzero((split >= 0) & (n_groups > PREFIX_RANKS))
    exact_pools: dict[tuple, np.ndarray] = {}
    relaxed_pools: dict[tuple, np.ndarray] = {}
    for index in eligible:
        exact_key = (
            int(split[index]), int(mode[index]), int(n_suffix_groups[index]),
            int(n_suffix_contacts[index]),
        )
        relaxed_key = exact_key[:3]
        exact_pools.setdefault(exact_key, []).append(int(index))
        relaxed_pools.setdefault(relaxed_key, []).append(int(index))
    exact_pools = {key: np.asarray(value, dtype=int) for key, value in exact_pools.items()}
    relaxed_pools = {key: np.asarray(value, dtype=int) for key, value in relaxed_pools.items()}
    unresolved = set()
    for recipient in eligible:
        exact_key = (
            int(split[recipient]), int(mode[recipient]), int(n_suffix_groups[recipient]),
            int(n_suffix_contacts[recipient]),
        )
        pools = (exact_pools[exact_key], relaxed_pools[exact_key[:3]])
        chosen = None
        for pool in pools:
            if len(pool) < 2:
                continue
            attempts = pool[rng.integers(0, len(pool), size=min(256, max(16, len(pool))))]
            for donor in np.concatenate([attempts, pool[:1024]]):
                donor = int(donor)
                if donor == int(recipient):
                    continue
                if np.any(prefix_masks[recipient] & suffix_masks[donor]):
                    continue
                chosen = donor
                break
            if chosen is not None:
                break
        if chosen is None:
            unresolved.add(int(recipient))
        else:
            mapping[recipient] = chosen
    fallback = 0
    still_unresolved = set()
    for index in unresolved:
        if n_suffix_groups[index] >= 2:
            mapping[index] = -int(index) - 1
            fallback += 1
        else:
            still_unresolved.add(int(index))
    changed = mapping != np.arange(len(mapping))
    donor_mask = changed & (mapping >= 0)
    donor = mapping[donor_mask]
    donor_reuse = 1.0 - len(np.unique(donor)) / max(len(donor), 1)
    return mapping, {
        "eligible_events": int(len(eligible)),
        "effectively_reassigned": int(changed.sum()),
        "effectively_reassigned_fraction": float(changed.sum() / max(len(eligible), 1)),
        "within_event_fallback": int(fallback),
        "unchanged_no_matched_donor": int(len(still_unresolved)),
        "unchanged_short_event": int(np.sum((split >= 0) & (n_groups <= PREFIX_RANKS))),
        "donor_reuse_fraction": float(donor_reuse),
        "same_suffix_rank_count_for_all_changed": bool(
            np.all(n_suffix_groups[donor_mask] == n_suffix_groups[mapping[donor_mask]])
        ),
        "prefix_suffix_overlap_for_any_changed": bool(
            any(np.any(prefix_masks[index] & suffix_masks[mapping[index]])
                for index in np.flatnonzero(donor_mask))
        ),
    }


def apply_suffix_mapping(
    ranks: np.ndarray, mapping: np.ndarray, *, seed: int = 0,
) -> np.ndarray:
    values = np.asarray(ranks, dtype=np.int16)
    output = values.copy()
    for recipient, donor in enumerate(np.asarray(mapping, dtype=int)):
        if recipient == donor:
            continue
        if donor < 0:
            row = values[recipient].copy()
            suffix_ranks = np.unique(row[row >= PREFIX_RANKS])
            rng = np.random.default_rng(int(seed) + 104729 * int(recipient + 1))
            permuted = _roll_derangement(suffix_ranks, rng)
            original = row.copy()
            for old_rank, new_rank in zip(suffix_ranks, permuted):
                row[original == old_rank] = new_rank
            output[recipient] = row
            continue
        prefix = (values[recipient] >= 0) & (values[recipient] < PREFIX_RANKS)
        donor_suffix = values[donor] >= PREFIX_RANKS
        donor_suffix &= ~prefix
        row = np.full(values.shape[1], -1, dtype=np.int16)
        row[prefix] = values[recipient, prefix]
        donor_ranks = np.unique(values[donor, donor_suffix])
        for offset, old_rank in enumerate(donor_ranks):
            row[donor_suffix & (values[donor] == old_rank)] = PREFIX_RANKS + offset
        output[recipient] = row
    return output


def suffix_position_frequency(ranks: np.ndarray, split: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks)
    select = np.asarray(split) >= 0
    out = np.zeros((values.shape[1], 4), dtype=float)
    for contact in range(values.shape[1]):
        row = values[select, contact]
        out[contact, 0] = np.mean(row == 0)
        out[contact, 1] = np.mean((row >= 1) & (row < PREFIX_RANKS))
        out[contact, 2] = np.mean(row >= PREFIX_RANKS)
        out[contact, 3] = np.mean(row < 0)
    return out


def suffix_distribution_audit(
    true_ranks: np.ndarray,
    null_ranks: np.ndarray,
    split: np.ndarray,
    mapping: np.ndarray,
    *,
    max_events: int = 1000,
) -> dict:
    eligible = np.flatnonzero((np.asarray(split) >= 0) & (mapping != np.arange(len(mapping))))
    sample = eligible[: int(max_events)]
    kendall_distance = []
    true_blocks = []
    null_blocks = []
    n_contacts = true_ranks.shape[1]
    true_count = np.zeros((n_contacts, n_contacts), dtype=float)
    true_wins = np.zeros_like(true_count)
    null_count = np.zeros_like(true_count)
    null_wins = np.zeros_like(true_count)
    for event in sample:
        true = np.asarray(true_ranks[event])
        null = np.asarray(null_ranks[event])
        true_suffix = true >= PREFIX_RANKS
        null_suffix = null >= PREFIX_RANKS
        true_blocks.append(len(np.unique(true[true_suffix])))
        null_blocks.append(len(np.unique(null[null_suffix])))
        shared = np.flatnonzero(true_suffix & null_suffix)
        if len(shared) >= 2:
            tau = kendalltau(true[shared], null[shared], variant="b").statistic
            if np.isfinite(tau):
                kendall_distance.append(0.5 * (1.0 - float(tau)))
        for row, count, wins in (
            (true, true_count, true_wins), (null, null_count, null_wins)
        ):
            present = np.flatnonzero(row >= PREFIX_RANKS)
            if len(present) < 2:
                continue
            rr = row[present]
            block = np.ix_(present, present)
            count[block] += 1.0
            wins[block] += (rr[:, None] < rr[None, :]).astype(float)
    true_precedence = np.divide(
        true_wins - true_wins.T, true_count,
        out=np.full_like(true_count, np.nan), where=true_count > 0,
    )
    null_precedence = np.divide(
        null_wins - null_wins.T, null_count,
        out=np.full_like(null_count, np.nan), where=null_count > 0,
    )
    common = np.isfinite(true_precedence) & np.isfinite(null_precedence)
    return {
        "distribution_audit_events": int(len(sample)),
        "mean_suffix_kendall_distance": float(np.mean(kendall_distance)) if kendall_distance else float("nan"),
        "mean_suffix_tie_block_shift": float(np.mean(null_blocks) - np.mean(true_blocks)) if true_blocks else float("nan"),
        "pairwise_precedence_l1": float(
            np.mean(np.abs(true_precedence[common] - null_precedence[common]))
        ) if np.any(common) else float("nan"),
    }


def compatibility(old_root: Path, fit_id: str, plane: dict, ranks: np.ndarray, split: np.ndarray) -> dict:
    old_fit = old_root / "cache" / fit_id
    if not old_fit.exists():
        return {
            "fit_id": fit_id, "old_fit": False, "checkpoint_reuse_eligible": False,
            "reason": "NEW_FIT",
        }
    old_plane = np.load(old_fit / "plane.npz", allow_pickle=False)
    old_events = np.load(old_fit / "events.npz", allow_pickle=False)
    geometry = all(np.array_equal(plane[key], old_plane[key]) for key in (
        "contacts_xy_mm", "nodes_xy_mm", "H", "D_mm",
    ))
    rank_equal = np.array_equal(ranks, old_events["ranks"])
    split_equal = np.array_equal(split, old_events["split"])
    shared = fit_id.endswith("__shared")
    eligible = bool(shared and geometry and rank_equal and split_equal)
    reason = "EXACT_SHARED_REUSE" if eligible else (
        "FULL_DATA_MODE_FILTER_REQUIRES_RETRAIN" if not shared else "CACHE_MISMATCH"
    )
    return {
        "fit_id": fit_id, "old_fit": True, "shared": shared,
        "geometry_exact": geometry, "ranks_exact": rank_equal,
        "split_exact": split_equal, "checkpoint_reuse_eligible": eligible,
        "reason": reason,
    }


def render_stage_b(fits: pd.DataFrame, audits: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    representative = "epilepsiae_1146__shared"
    small = fits.sort_values(["n_joint_contacts", "fit_id"]).iloc[0].fit_id
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 12,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 4, figsize=(12.8, 3.0), gridspec_kw={"wspace": 0.68})
    blue, red, grey = "#3274a1", "#b23b45", "#c5c9cc"
    for axis, fit_id in zip(axes[:2], (representative, small)):
        plane = np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False)
        observed = plane["H"].sum(axis=0) > 1e-12
        axis.scatter(plane["nodes_xy_mm"][:, 0], plane["nodes_xy_mm"][:, 1],
                     s=9, c=np.where(observed, blue, grey), edgecolors="none")
        axis.scatter(plane["contacts_xy_mm"][:, 0], plane["contacts_xy_mm"][:, 1],
                     s=28, facecolors="white", edgecolors="#171717", linewidths=0.9)
        axis.set_aspect("equal")
        axis.set_xlabel("Propagation axis (mm)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Transverse axis (mm)")

    templates = np.load(out / "cache" / representative / "train_only_modes.npz")
    image = axes[2].imshow(templates["templates"], aspect="auto", cmap="coolwarm")
    axes[2].set_xlabel("Contacts")
    axes[2].set_ylabel("Train-only mode")
    axes[2].set_yticks([0, 1])
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.025)
    colorbar.ax.set_title("Rank", fontsize=9.5, pad=4)

    pivot = audits.groupby("fit_id").suffix_position_l1.max()
    axes[3].scatter(np.arange(len(pivot)), pivot, color=red, s=20)
    axes[3].set_xlabel("Spatial fits")
    axes[3].set_ylabel("Suffix marginal shift")
    axes[3].set_xticks([])
    axes[3].spines[["top", "right"]].set_visible(False)
    for label, axis in zip("ABCD", axes):
        axis.text(-0.12, 1.10, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    stem = figures / "stage_b_v0_5_train_only_modes_and_suffix_null"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    readme = figures / "README.md"
    section = (
        "### stage_b_v0_5_train_only_modes_and_suffix_null.png\n\n"
        "A–B 展示 E1146 与最小 montage 的 full-tissue latent layout；蓝色节点可被 contact 直接读取，"
        "灰色节点只能通过 recurrence 参与。C 展示只用训练事件得到的两个 mode templates。"
        "D 显示三份 suffix-reassignment null 相对真实序列的 contact-position marginal 偏移。\n\n"
        "**关注点**：所有 planes 使用同一完整预测任务；train-only mode 只作分层，旧 split-fit checkpoint 不再复用。\n"
    )
    existing = readme.read_text() if readme.exists() else ""
    marker = "### stage_b_v0_5_train_only_modes_and_suffix_null.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    readme.write_text((existing.rstrip() + "\n\n" + section).lstrip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    old = args.old_root.resolve()
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    rows = []
    compatibility_rows = []
    audit_rows = []
    for item in census.itertuples():
        fit_id = str(item.fit_id)
        cache = out / "cache" / fit_id
        raw = np.load(cache / "events_raw.npz", allow_pickle=False)
        plane = np.load(cache / "plane.npz", allow_pickle=False)
        ranks = np.asarray(raw["ranks"], dtype=np.int16)
        base_split = np.asarray(raw["base_split"], dtype=np.int8)
        modes = train_only_modes(ranks, base_split)
        own_cluster = None
        alignment_scores: list[float] = []
        if str(item.scope) != "shared":
            own_cluster, alignment_scores = scope_cluster(
                modes["templates"], plane["contacts_xy_mm"]
            )
        split = scope_split(base_split, str(item.scope), modes, own_cluster)
        np.savez_compressed(
            cache / "events.npz",
            ranks=ranks, split=split,
            mode=modes["prefix_mode"],
            full_train_mode=modes["full_train_mode"],
            prefix_posterior=modes["prefix_posterior"],
            prefix_entropy=modes["prefix_entropy"],
            event_abs_time=raw["event_abs_time"],
            event_source_index=raw["event_source_index"],
            event_dataset_index=raw["event_dataset_index"],
        )
        np.savez_compressed(
            cache / "train_only_modes.npz",
            templates=modes["templates"], centers=modes["centers"],
            train_counts=modes["train_counts"],
            temperature=np.asarray([modes["temperature"]], np.float32),
            own_cluster=np.asarray([-1 if own_cluster is None else own_cluster], np.int8),
        )
        true_frequency = suffix_position_frequency(ranks, split)
        for null_index, seed in enumerate(SEEDS):
            routing_mode = np.where(
                base_split == 0, modes["full_train_mode"], modes["prefix_mode"]
            ).astype(np.int8)
            mapping, audit = suffix_mapping(ranks, split, routing_mode, seed)
            null_ranks = apply_suffix_mapping(ranks, mapping, seed=seed)
            null_frequency = suffix_position_frequency(null_ranks, split)
            l1 = float(np.mean(np.abs(true_frequency - null_frequency)))
            distribution = suffix_distribution_audit(
                ranks, null_ranks, split, mapping,
            )
            np.savez_compressed(
                cache / f"events_suffix_null_seed{null_index}.npz",
                ranks=null_ranks, split=split, mode=modes["prefix_mode"],
                suffix_donor_index=mapping,
                event_abs_time=raw["event_abs_time"],
                event_source_index=raw["event_source_index"],
            )
            audit_rows.append({
                "fit_id": fit_id, "null_seed_index": null_index,
                "suffix_null_seed": seed, **audit,
                "suffix_position_l1": l1, **distribution,
            })
        compatible = compatibility(old, fit_id, plane, ranks, split)
        compatibility_rows.append(compatible)
        rows.append({
            "fit_id": fit_id, "subject": item.subject, "scope": item.scope,
            "n_events_train": int(np.sum(split == 0)),
            "n_events_validation": int(np.sum(split == 1)),
            "n_events_test": int(np.sum(split == 2)),
            "train_mode0": int(modes["train_counts"][0]),
            "train_mode1": int(modes["train_counts"][1]),
            "own_cluster": -1 if own_cluster is None else own_cluster,
            "plane_alignment_scores": alignment_scores,
            "events_sha256": sha256(cache / "events.npz"),
            "templates_sha256": sha256(cache / "train_only_modes.npz"),
            "checkpoint_reuse_eligible": compatible["checkpoint_reuse_eligible"],
            "target_values_read": False,
        })
    table = pd.DataFrame(rows)
    compat = pd.DataFrame(compatibility_rows)
    audits = pd.DataFrame(audit_rows)
    table.to_csv(out / "TRAIN_ONLY_MODE_FIT_CENSUS.csv", index=False)
    compat.to_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv", index=False)
    audits.to_csv(out / "SUFFIX_NULL_DESTRUCTION_AUDIT.csv", index=False)
    shared_reuse = int(compat.checkpoint_reuse_eligible.sum())
    new_or_retrain = int(len(compat) - shared_reuse)
    units = shared_reuse * 2 * 3 + new_or_retrain * 5 * 3
    write_json(out / "STAGE_B_COMPLETE.json", {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "fits": int(len(table)), "patients": int(table.subject.nunique()),
        "shared_checkpoint_reuse_fits": shared_reuse,
        "mandatory_full_retrain_fits": new_or_retrain,
        "formal_training_units": units,
        "suffix_null_seeds": list(SEEDS),
        "minimum_effectively_reassigned_fraction": float(
            audits.effectively_reassigned_fraction.min()
        ),
        "maximum_suffix_position_l1": float(audits.suffix_position_l1.max()),
        "train_only_mode_manifest_sha256": sha256(out / "TRAIN_ONLY_MODE_FIT_CENSUS.csv"),
        "reuse_audit_sha256": sha256(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv"),
        "suffix_audit_sha256": sha256(out / "SUFFIX_NULL_DESTRUCTION_AUDIT.csv"),
    })
    render_stage_b(census, audits, out)
    print(
        f"Stage B PASS: {len(table)} fits; {shared_reuse} exact shared reuses; "
        f"{new_or_retrain} full retrains; formal units={units}"
    )


if __name__ == "__main__":
    main()

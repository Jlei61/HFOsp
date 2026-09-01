"""Explore direction-versus-extent factorization after the SA K=2 AMI stop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score

sys.path.insert(0, os.getcwd())
from scripts.build_topic4_rev10_sa_shaft_aware_target import (  # noqa: E402
    _atomic_json,
)
from src.topic4_shaft_aware import (  # noqa: E402
    PAIR_CLASS_ORDER,
    SHAFT_ORDER,
    contract_groups,
    contract_pairs,
    describe_events,
    descriptor_distances,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_shaft_aware.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_provenance(config_path):
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/topic4_shaft_aware.py",
        config_path.resolve(),
    ]
    relative = [str(path.relative_to(ROOT)) for path in paths]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runtime_files_dirty": bool(dirty),
        "runtime_file_sha256": {
            name: _sha256(ROOT / name) for name in relative
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def build_strata(blocks, onsets, labels, groups):
    """Index events by block and exact ICL/SCL recruitment counts."""
    mask = np.isfinite(onsets)
    n_icl = mask[:, np.asarray(groups["ICL"], dtype=int)].sum(axis=1)
    n_scl = mask[:, np.asarray(groups["SCL"], dtype=int)].sum(axis=1)
    strata = defaultdict(lambda: {0: [], 1: []})
    for index, (block, left, right, label) in enumerate(
        zip(blocks, n_icl, n_scl, labels)
    ):
        strata[(int(block), int(left), int(right))][int(label)].append(index)
    return {
        key: {label: np.asarray(index, dtype=int) for label, index in value.items()}
        for key, value in strata.items()
        if len(value[0]) and len(value[1])
    }


def matched_label_draw(strata, rng, max_events_per_label_per_stratum):
    """Return observed and within-stratum permuted equal-count A/B indices."""
    observed = {0: [], 1: []}
    null = {0: [], 1: []}
    used = []
    for key in sorted(strata):
        left, right = strata[key][0], strata[key][1]
        take = min(len(left), len(right), int(max_events_per_label_per_stratum))
        if take < 1:
            continue
        selected_left = rng.choice(left, size=take, replace=False)
        selected_right = rng.choice(right, size=take, replace=False)
        combined = np.r_[selected_left, selected_right]
        shuffled = rng.permutation(combined)
        observed[0].append(selected_left)
        observed[1].append(selected_right)
        null[0].append(shuffled[:take])
        null[1].append(shuffled[take:])
        used.append(key)
    if not used:
        raise RuntimeError("no block-by-recruitment stratum contains both old modes")
    return (
        {label: np.concatenate(parts) for label, parts in observed.items()},
        {label: np.concatenate(parts) for label, parts in null.items()},
        used,
    )


def _distance(onsets, indices, groups, pairs):
    left = describe_events(onsets[indices[0]], groups, pairs)
    right = describe_events(onsets[indices[1]], groups, pairs)
    return descriptor_distances(left, right)


def _flatten(row):
    return {
        f"{family}.{key}": float(value)
        for family, values in row.items()
        for key, value in values.items()
    }


def _label_extent_audit(old, new, extent):
    contingency = np.asarray([
        [np.sum((old == left) & (new == right)) for right in (0, 1)]
        for left in (0, 1)
    ], dtype=int)
    quartiles = np.quantile(extent, np.linspace(0.0, 1.0, 5))
    rows = []
    for index in range(4):
        selected = (
            (extent >= quartiles[index])
            & (extent <= quartiles[index + 1] if index == 3 else extent < quartiles[index + 1])
        )
        rows.append({
            "quartile": index + 1,
            "lower": float(quartiles[index]),
            "upper": float(quartiles[index + 1]),
            "n_events": int(selected.sum()),
            "old_mode_b_fraction": float(np.mean(old[selected] == 1)),
            "shaft_aware_cluster_b_fraction": float(np.mean(new[selected] == 1)),
        })
    return {
        "ami_old_vs_shaft_aware": float(adjusted_mutual_info_score(old, new)),
        "contingency_old_by_shaft_aware": contingency,
        "extent_auc_for_old_mode_b": float(roc_auc_score(old, extent)),
        "extent_auc_for_shaft_aware_cluster_b": float(roc_auc_score(new, extent)),
        "extent_quartiles": rows,
        "all_four_crossed_cells_supported": bool(np.all(contingency >= 100)),
    }


def _plot(result, arrays, contract, output_root):
    old, new, extent = arrays["old"], arrays["new"], arrays["extent"]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    colors = ("#E76F51", "#277DA1")
    bins = np.linspace(float(extent.min()), float(extent.max()), 25)
    for label, name in ((old, "old"), (new, "SA")):
        axis = axes[0] if name == "old" else axes[1]
        for mode in (0, 1):
            axis.hist(extent[label == mode], bins=bins, density=True,
                      histtype="step", linewidth=2, color=colors[mode],
                      label=f"{name} {'AB'[mode]}")
        axis.set_xlabel("mean shaft recruitment fraction")
        axis.set_ylabel("density")
        axis.set_title(
            ("A  Old direction labels" if name == "old" else "B  SA KMeans labels"),
            loc="left", weight="bold",
        )
        axis.legend(frameon=False)

    keys = list(PAIR_CLASS_ORDER)
    x = np.arange(len(keys))
    observed = [result["matched_direction_contrast"][f"precedence.{key}"]["observed"]["median"] for key in keys]
    null = [result["matched_direction_contrast"][f"precedence.{key}"]["null"]["median"] for key in keys]
    null_low = [result["matched_direction_contrast"][f"precedence.{key}"]["null"]["q05"] for key in keys]
    null_high = [result["matched_direction_contrast"][f"precedence.{key}"]["null"]["q95"] for key in keys]
    axes[2].bar(x - 0.18, observed, width=0.36, color="#4C78A8", label="old A vs B")
    axes[2].bar(x + 0.18, null, width=0.36, color="#B8B8B8", label="stratified null")
    axes[2].errorbar(x + 0.18, null, yerr=[np.asarray(null) - null_low, np.asarray(null_high) - null],
                     fmt="none", ecolor="black", capsize=3, linewidth=1)
    axes[2].set_xticks(x, keys)
    axes[2].set_ylabel("pair-state JS divergence")
    axes[2].set_title("C  Extent-matched direction", loc="left", weight="bold")
    axes[2].legend(frameon=False)

    contingency = np.asarray(result["extent_factor"]["contingency_old_by_shaft_aware"])
    fraction = contingency / contingency.sum(axis=1, keepdims=True)
    image = axes[3].imshow(fraction, vmin=0, vmax=1, cmap="Blues")
    for left in (0, 1):
        for right in (0, 1):
            axes[3].text(right, left, f"{contingency[left, right]}\n{fraction[left, right]:.1%}",
                         ha="center", va="center",
                         color="white" if fraction[left, right] > 0.5 else "black")
    axes[3].set_xticks([0, 1], ["extent A", "extent B"])
    axes[3].set_yticks([0, 1], ["direction A", "direction B"])
    axes[3].set_title("D  Crossed factors", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[3], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    figure_dir = output_root / "figures"
    stem = figure_dir / "rev10_sa_direction_extent_factorization"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    with (figure_dir / "README.md").open("a") as stream:
        stream.write("""

### rev10_sa_direction_extent_factorization

这张图只使用患者训练 blocks。A/B 比较旧传播方向标签与 shaft-aware KMeans 对事件招募范围的分割；C 在 `recording block × ICL 招募数 × SCL 招募数` 内严格配平 A/B 后，比较三类 precedence 与同层置换 null；D 显示方向与范围的四个交叉组合都有充足事件。

**关注点**：若 KMeans 几乎完全由事件范围预测，而范围配平后旧 A/B 的 ICL-ICL 和 ICL-SCL precedence 仍高于 null，则两者应作为不同因子进入 target，不能强迫一个平面 K=2 同时承担两种结构。
""")
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    summary_path = output_root / "shaft_aware_target_summary.json"
    target_path = output_root / "shaft_aware_patient_training_target.npz"
    if not summary_path.exists() or not target_path.exists():
        raise FileNotFoundError("run build_topic4_rev10_sa_shaft_aware_target.py first")
    target_summary = json.loads(summary_path.read_text())
    if target_summary["status"] != "PATIENT_MODE_DEFINITION_UNRESOLVED":
        raise RuntimeError("factorization audit is only licensed by the AMI stop")
    contract = target_summary["contact_contract"]
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    with np.load(target_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["patient_train_onsets"], dtype=float)
        blocks = np.asarray(loaded["patient_train_block_ids"])
        old = np.asarray(loaded["patient_train_old_labels"], dtype=int)
        new = np.asarray(loaded["patient_train_shaft_aware_k2_labels"], dtype=int)
    mask = np.isfinite(onsets)
    f_icl = mask[:, groups["ICL"]].mean(axis=1)
    f_scl = mask[:, groups["SCL"]].mean(axis=1)
    extent = 0.5 * (f_icl + f_scl)
    extent_audit = _label_extent_audit(old, new, extent)

    audit = config["mode_factorization_audit"]
    rng = np.random.default_rng(int(audit["seed"]))
    strata = build_strata(blocks, onsets, old, groups)
    observed_rows, null_rows, counts = [], [], []
    used_strata = set()
    representative = None
    for _ in range(int(audit["repeats"])):
        observed, null, used = matched_label_draw(
            strata, rng, audit["max_events_per_label_per_stratum"],
        )
        representative = observed
        observed_rows.append(_flatten(_distance(onsets, observed, groups, pairs)))
        null_rows.append(_flatten(_distance(onsets, null, groups, pairs)))
        counts.append(len(observed[0]))
        used_strata.update(used)
    keys = sorted(observed_rows[0])
    contrasts = {}
    for key in keys:
        observed_values = [row[key] for row in observed_rows]
        null_values = [row[key] for row in null_rows]
        observed_summary, null_summary = _summary(observed_values), _summary(null_values)
        contrasts[key] = {
            "observed": observed_summary,
            "null": null_summary,
            "observed_median_above_null_q95": (
                observed_summary["median"] > null_summary["q95"]
            ),
        }

    precedence_survives = {
        key: contrasts[f"precedence.{key}"]["observed_median_above_null_q95"]
        for key in PAIR_CLASS_ORDER
    }
    status = (
        "DIRECTION_AND_EXTENT_FACTORS_BOTH_SUPPORTED_EXPLORATORY"
        if (precedence_survives["ICL-ICL"] and precedence_survives["ICL-SCL"]
            and extent_audit["extent_auc_for_shaft_aware_cluster_b"] >= 0.9
            and extent_audit["all_four_crossed_cells_supported"])
        else "DIRECTION_EXTENT_FACTORIZATION_UNRESOLVED"
    )
    result = {
        "status": status,
        "scientific_role": (
            "patient-training-only exploratory resolution of the pre-registered "
            "PATIENT_MODE_DEFINITION_UNRESOLVED stop; no model artifact scored"
        ),
        "extent_factor": extent_audit,
        "matched_direction_contrast": contrasts,
        "matched_design": {
            "strata": "recording_block x recruited_ICL_count x recruited_SCL_count",
            "n_eligible_strata": len(strata),
            "n_strata_used": len(used_strata),
            "events_per_old_mode": _summary(counts),
            "max_events_per_label_per_stratum": audit["max_events_per_label_per_stratum"],
            "repeats": audit["repeats"],
            "seed": audit["seed"],
            "null": "permute old A/B within each exactly matched stratum",
        },
        "precedence_direction_survives_extent_matching": precedence_survives,
        "recommended_target_factorization": {
            "primary_mode_identity": "frozen old direction A/B labels",
            "within_mode_distribution": (
                "shaft-balanced recruitment extent and fixed-contact event cloud; "
                "do not replace A/B with the extent KMeans labels"
            ),
            "flat_shaft_aware_k2_as_patient_mode": "REJECT",
            "model_optimization_resume": "requires explicit spec amendment after review",
        },
        "inputs": {
            "target_npz": str(target_path.relative_to(ROOT)),
            "target_sha256": _sha256(target_path),
            "target_summary": str(summary_path.relative_to(ROOT)),
            "target_summary_sha256": _sha256(summary_path),
            "heldout_scores_read": False,
        },
        "provenance": _runtime_provenance(config_path),
    }
    output_path = output_root / "direction_extent_factorization_audit.json"
    _atomic_json(output_path, result)
    stem = _plot(result, {"old": old, "new": new, "extent": extent},
                 contract, output_root)
    print(status)
    print("extent AUC old/new:",
          extent_audit["extent_auc_for_old_mode_b"],
          extent_audit["extent_auc_for_shaft_aware_cluster_b"])
    print("precedence survives:", precedence_survives)
    print("matched events per mode:", _summary(counts))
    print(f"wrote {output_path} and {stem}.png/.pdf")


if __name__ == "__main__":
    main()

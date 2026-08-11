"""Zero-simulation V3 audit with the corrected factorized event objective."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev10_sa_spline_field_search import (  # noqa: E402
    _classifier_from_manifest,
    _objective,
)
from scripts.freeze_topic4_rev10_sa_spline_field_v4_candidates import (  # noqa: E402
    _patient_classifier,
)
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
    score_mode_conditioned_events,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_shaft_aware import contract_groups, contract_pairs  # noqa: E402
from src.topic4_shaft_aware_direction import (  # noqa: E402
    all_event_shaft_participation,
    assign_direction_modes,
)


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v3.json"
V4_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v4.json"


def _json_classifier(classifier):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in classifier.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-config", default=str(V3_CONFIG))
    parser.add_argument("--objective-config", default=str(V4_CONFIG))
    args = parser.parse_args()
    v3 = json.loads(Path(args.v3_config).read_text())
    objective_config = json.loads(Path(args.objective_config).read_text())
    output_root = ROOT / v3["output_root"]
    audit_root = output_root / "factorized_rescore"
    figure_dir = audit_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output_root / "candidate_manifest.json").read_text())
    contract = _load_json_input(v3["inputs"]["contact_contract"])
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    classifier = _classifier_from_manifest({
        "direction_classifier": _json_classifier(
            _patient_classifier(objective_config, contract)
        )
    })
    target_path = v3["inputs"]["shaft_aware_target_npz"]["path"]
    floor_path = v3["inputs"]["shaft_aware_floors"]["path"]
    names, embedding, targets, floors6 = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=6,
    )
    _, _, _, floors3 = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=3,
    )
    scoring_config = _load_json_input(
        objective_config["inputs"]["shaft_aware_scoring_config"]
    )
    rows = []
    for candidate in manifest["candidate_set"]["candidates"]:
        blocks = []
        for seed in v3["search"]["network_seeds"]:
            path = output_root / "workers" / f"{candidate['candidate_id']}_seed_{seed}.npz"
            with np.load(path, allow_pickle=False) as loaded:
                worker_names = np.asarray(loaded["contact_names"]).astype(str)
                if not np.array_equal(worker_names, names):
                    raise RuntimeError(f"contact order changed: {path}")
                blocks.append(np.asarray(loaded["onsets"], float))
        onsets = np.concatenate(blocks, axis=0)
        if len(onsets):
            assigned = assign_direction_modes(
                onsets, groups=groups, embedding=embedding, classifier=classifier,
            )
            labels, ood = assigned["labels"], assigned["ood"]
        else:
            labels, ood = np.empty(0, int), np.empty(0, bool)
        counts = np.bincount(labels, minlength=2)
        participation = all_event_shaft_participation(onsets, groups)
        score6 = score_mode_conditioned_events(
            onsets, labels, groups=groups, pairs=pairs, embedding=embedding,
            targets=targets, floors=floors6, config=scoring_config,
            fixed_events_per_mode=6,
        )
        score3 = score_mode_conditioned_events(
            onsets, labels, groups=groups, pairs=pairs, embedding=embedding,
            targets=targets, floors=floors3, config=scoring_config,
            fixed_events_per_mode=3,
        )
        objective = _objective(
            score6, score3, counts, participation,
            float(np.mean(ood)) if len(ood) else 1.0, objective_config,
        )
        rows.append({
            "candidate_id": candidate["candidate_id"], "role": candidate["role"],
            "center_xy_mm": candidate.get("uniform_center_xy_mm"),
            **participation, **objective,
            "mode_A_count": int(counts[0]), "mode_B_count": int(counts[1]),
            "ood_fraction": float(np.mean(ood)) if len(ood) else 1.0,
        })
    rows.sort(key=lambda row: (row["selection_score"], row["candidate_id"]))
    payload = {
        "status": "REV10SA_V3_FACTORIZED_ZERO_SIM_RESCORE_COMPLETE",
        "safe_claim": (
            "V3 location probes can switch event shaft or create sparse joint events, "
            "but were previously unscorable because the old single-axis entry filter "
            "discarded them before the shaft-aware objective"
        ),
        "rows": rows,
        "direction_classifier": {
            "heldout_balanced_accuracy": classifier["heldout_balanced_accuracy"],
            "heldout_roc_auc": classifier["heldout_roc_auc"],
        },
        "inputs": {
            "v3_manifest_sha256": _sha256(output_root / "candidate_manifest.json"),
            "v3_summary_sha256": _sha256(output_root / "spectral_field_search_summary.json"),
            "objective_config_sha256": _sha256(args.objective_config),
        },
        "provenance": _runtime_provenance(),
    }
    (audit_root / "v3_factorized_rescore.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    with open(audit_root / "v3_factorized_rescore.csv", "w", newline="") as stream:
        keys = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grid_rows = [row for row in rows if row["candidate_id"].startswith("v3_uniform_")]
    by_index = {int(row["candidate_id"].split("_")[-1]): row for row in grid_rows}
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), constrained_layout=True)
    fields = [
        ("joint_fraction", "joint-shaft fraction", "viridis", 0.0, 1.0),
        ("n_events", "detected events", "magma", None, None),
        ("mode_A_count", "direction A count", "Blues", None, None),
        ("mode_B_count", "direction B count", "Greens", None, None),
    ]
    for ax, (key, title, cmap, lower, upper) in zip(axes, fields):
        values = np.asarray([[by_index[4 * y + x][key] for x in range(4)] for y in range(4)])
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=lower, vmax=upper,
                          extent=(0, 20, 0, 20))
        for y in range(4):
            for x in range(4):
                text = f"{values[y, x]:.2f}" if key == "joint_fraction" else str(int(values[y, x]))
                ax.text(2.5 + 5 * x, 2.5 + 5 * y, text, ha="center", va="center",
                        color="white" if values[y, x] > np.nanmedian(values) else "black",
                        fontsize=8)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("sheet x (mm)")
        if ax is axes[0]:
            ax.set_ylabel("sheet y (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("V3 zero-simulation factorized audit | all events retained", weight="bold")
    stem = figure_dir / "rev10_sa_v3_factorized_rescore"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        """### rev10_sa_v3_factorized_rescore

这张图不重跑 SNN，而是对 V3 的全部已检测事件重新计算同一事件双杆参与和监督式 A/B 方向标签。四张热图分别显示 joint-shaft fraction、总事件数和两个方向的支持度。

**关注点**：SCL-only 增加不是患者多杆恢复；只有 joint fraction 与 A/B 支持同时存在才进入下一轮。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], "selected": rows[0]["candidate_id"],
        "selected_score": rows[0]["selection_score"], "figure": str(stem),
    }, indent=2))


if __name__ == "__main__":
    main()

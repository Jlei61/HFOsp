#!/usr/bin/env python3
"""Freeze target-free L2m graph controls and candidate-capacity audits."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_lbss_rnn_v0_2 import build_pool_contract  # noqa: E402
from src.topic5_multiscale_scaffold_v0_5 import (  # noqa: E402
    construct_macro_matched_nonlocal,
    exact_macro_match_audit,
    stable_seed,
)


OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3/per_fit"
ARM_L1 = "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"
ARM_L3 = "L3_LOCAL_PLUS_LEARNED_LR"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def candidate_capacity_rows(census: pd.DataFrame, reuse: pd.DataFrame) -> list[dict]:
    reuse_lookup = reuse.set_index("fit_id")["checkpoint_reuse_eligible"].to_dict()
    rows: list[dict] = []
    for fit_id in census.fit_id:
        plane = np.load(OUT_ROOT / "cache" / fit_id / "plane.npz", allow_pickle=False)
        pools = build_pool_contract(plane["D_mm"])
        supported = np.abs(plane["H"]).sum(axis=0) > 0
        extra_source = pools.extra_local_pool[:, supported].sum(axis=0)
        lr_source = pools.nonlocal_pool[:, supported].sum(axis=0)
        extra_target = pools.extra_local_pool[supported, :].sum(axis=1)
        lr_target = pools.nonlocal_pool[supported, :].sum(axis=1)
        only_one = ((extra_source > 0) ^ (lr_source > 0)) | ((extra_target > 0) ^ (lr_target > 0))
        row = {
            "fit_id": fit_id,
            "extra_local_pool_size": int(pools.extra_local_pool.sum()),
            "nonlocal_pool_size": int(pools.nonlocal_pool.sum()),
            "pool_size_ratio_L3_over_L1": float(
                pools.nonlocal_pool.sum() / max(1, int(pools.extra_local_pool.sum()))
            ),
            "h_supported_nodes": int(supported.sum()),
            "h_supported_nodes_only_one_arm_opportunity": int(only_one.sum()),
            "candidate_opportunity_severe": bool(only_one.any()),
            "l1_source_candidates_min": int(extra_source.min()),
            "l1_source_candidates_median": float(np.median(extra_source)),
            "l3_source_candidates_min": int(lr_source.min()),
            "l3_source_candidates_median": float(np.median(lr_source)),
            "l1_target_candidates_min": int(extra_target.min()),
            "l3_target_candidates_min": int(lr_target.min()),
            "old_shared_checkpoint_reuse": bool(reuse_lookup.get(fit_id, False)),
            "exposure_status": "PENDING_FORMAL_TRAIN" if not reuse_lookup.get(fit_id, False) else "AUDITED_OLD_SHARED",
            "l1_unique_exposure_fraction": float("nan"),
            "l3_unique_exposure_fraction": float("nan"),
            "exposure_fraction_ratio_L3_over_L1": float("nan"),
            "exposure_severe": False,
        }
        if row["old_shared_checkpoint_reuse"]:
            fractions = {}
            for arm, label in ((ARM_L1, "l1"), (ARM_L3, "l3")):
                graph = np.load(OLD_ROOT / fit_id / arm / "seed0" / "graph.npz", allow_pickle=False)
                pool = graph["candidate_pool"].astype(bool)
                seen = (
                    graph["initial_added_mask"].astype(bool)
                    | graph["added_mask"].astype(bool)
                    | (graph["exposure_count"] > 0)
                    | (graph["proposal_count"] > 0)
                )
                fractions[label] = float(np.count_nonzero(seen & pool) / max(1, int(pool.sum())))
            ratio = fractions["l3"] / max(1e-12, fractions["l1"])
            row.update({
                "l1_unique_exposure_fraction": fractions["l1"],
                "l3_unique_exposure_fraction": fractions["l3"],
                "exposure_fraction_ratio_L3_over_L1": ratio,
                "exposure_severe": bool(ratio < 0.5 or ratio > 2.0),
            })
        rows.append(row)
    return rows


def build_available_l2m(census: pd.DataFrame, reuse: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    reuse_lookup = reuse.set_index("fit_id")["checkpoint_reuse_eligible"].to_dict()
    for fit_id in census.fit_id:
        plane = np.load(OUT_ROOT / "cache" / fit_id / "plane.npz", allow_pickle=False)
        pools = build_pool_contract(plane["D_mm"])
        for model_seed in range(3):
            formal_graph = (
                OUT_ROOT / "formal_units" / fit_id / ARM_L3 / f"seed{model_seed}" / "graph.npz"
            )
            old_graph = OLD_ROOT / fit_id / ARM_L3 / f"seed{model_seed}" / "graph.npz"
            if formal_graph.exists():
                graph_path = formal_graph
                l3_source = "FORMAL_V0_5_L3"
            elif bool(reuse_lookup.get(fit_id, False)) and old_graph.exists():
                graph_path = old_graph
                l3_source = "EXACT_REUSED_V0_3_SHARED_L3"
            else:
                continue
            graph = np.load(graph_path, allow_pickle=False)
            if not np.array_equal(graph["candidate_pool"], pools.nonlocal_pool):
                raise RuntimeError(f"{fit_id} seed{model_seed}: frozen L3 pool differs from v0.5")
            if int(graph["added_mask"].sum()) != int(pools.k_added):
                raise RuntimeError(f"{fit_id} seed{model_seed}: k_added mismatch")
            graph_null_seed = stable_seed(fit_id, 2026081300 + model_seed)
            result = construct_macro_matched_nonlocal(
                graph["added_mask"], pools.nonlocal_pool, plane["D_mm"], graph_null_seed,
                max_restarts=100, attempts_per_restart=10_000,
                # Exact matching is the inferential contract.  A 25% pairing
                # disruption floor prevents a merely cosmetic two-edge swap
                # without making small, highly constrained graphs ineligible.
                minimum_disruption_fraction=0.25,
            )
            audit = exact_macro_match_audit(
                graph["added_mask"], result.mask, pools.nonlocal_pool, result.bin_labels
            )
            audit.update(result.audit)
            if not audit["all_exact"] or not audit.get("disruption_target_met", False):
                raise RuntimeError(f"{fit_id} seed{model_seed}: L2m exact matching failed")
            unit = OUT_ROOT / "graph_controls" / fit_id / f"seed{model_seed}"
            unit.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                unit / "L2M_GRAPH_CONTROL.npz",
                added_mask=result.mask,
                reference_l3_final_added_mask=graph["added_mask"].astype(np.uint8),
                reference_l3_initial_added_mask=graph["initial_added_mask"].astype(np.uint8),
                nonlocal_pool=pools.nonlocal_pool.astype(np.uint8),
                distance_bin_labels=result.bin_labels,
                distance_cutpoints_mm=result.cutpoints_mm,
                graph_null_seed=np.asarray(graph_null_seed, dtype=np.uint32),
                model_seed=np.asarray(model_seed, dtype=np.int16),
            )
            payload = {
                "contract": "topic5_l2m_graph_control_v0_5",
                "fit_id": fit_id,
                "model_seed": model_seed,
                "graph_null_seed": graph_null_seed,
                "reference_l3_graph_sha256": sha256_file(graph_path),
                "reference_l3_source": l3_source,
                "plane_sha256": sha256_file(OUT_ROOT / "cache" / fit_id / "plane.npz"),
                "audit": audit,
                "target_values_read": False,
                "status": "EXACT_MATCH_PASS",
            }
            atomic_json(unit / "L2M_GRAPH_CONTROL.json", payload)
            rows.append({
                "fit_id": fit_id,
                "model_seed": model_seed,
                "status": payload["status"],
                "graph_null_seed": graph_null_seed,
                "reference_l3_source": l3_source,
                "l3_edges": audit["reference_edge_count"],
                "l2m_edges": audit["candidate_edge_count"],
                "pairing_disruption_fraction": audit["pairing_disruption_fraction"],
                "reciprocity_count": audit["candidate_reciprocity_count"],
                "accepted_swaps": audit["accepted_swaps_this_restart"],
                "all_exact": audit["all_exact"],
                "target_values_read": False,
                "graph_control_path": str(unit / "L2M_GRAPH_CONTROL.npz"),
                "graph_control_sha256": sha256_file(unit / "L2M_GRAPH_CONTROL.npz"),
            })
    return rows


def plot_stage_c(candidate: pd.DataFrame, matched: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), constrained_layout=True)
    axis = axes[0]
    axis.scatter(candidate.extra_local_pool_size, candidate.nonlocal_pool_size, s=22, color="#315b8a")
    low = min(candidate.extra_local_pool_size.min(), candidate.nonlocal_pool_size.min())
    high = max(candidate.extra_local_pool_size.max(), candidate.nonlocal_pool_size.max())
    axis.plot([low, high], [low, high], color="0.65", lw=1, ls="--")
    axis.set(xscale="log", yscale="log", xlabel="Extra-local candidates", ylabel="Nonlocal candidates")
    axis.text(-0.20, 1.06, "A", transform=axis.transAxes, fontsize=15, fontweight="bold")

    axis = axes[1]
    values = candidate.h_supported_nodes_only_one_arm_opportunity.to_numpy()
    axis.hist(values, bins=np.arange(values.max() + 2) - 0.5, color="#668fb5", edgecolor="white")
    axis.set(xlabel="H-supported nodes\nwith one-arm opportunity", ylabel="Fits")
    axis.text(-0.20, 1.06, "B", transform=axis.transAxes, fontsize=15, fontweight="bold")

    axis = axes[2]
    axis.scatter(np.arange(len(matched)), matched.pairing_disruption_fraction, s=24, color="#b74248")
    axis.axhline(0.25, color="0.35", ls="--", lw=1)
    axis.set(xlabel="Frozen L2m controls", ylabel="Pairing disruption", ylim=(0, 1.02), xticks=[])
    axis.text(-0.20, 1.06, "C", transform=axis.transAxes, fontsize=15, fontweight="bold")
    output = OUT_ROOT / "figures/stage_c_v0_5_graph_control_audit.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    if not (OUT_ROOT / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json").exists():
        raise RuntimeError("target physical embargo is not active")
    census = pd.read_csv(OUT_ROOT / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(OUT_ROOT / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    candidate = pd.DataFrame(candidate_capacity_rows(census, reuse))
    candidate.to_csv(OUT_ROOT / "CANDIDATE_CAPACITY_AUDIT.csv", index=False)
    matched = pd.DataFrame(build_available_l2m(census, reuse))
    matched.to_csv(OUT_ROOT / "L2M_GRAPH_CONTROL_MANIFEST.csv", index=False)
    figure = plot_stage_c(candidate, matched)
    payload = {
        "status": "PASS_AVAILABLE_SHARED_CONTROLS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "fits_audited": int(candidate.fit_id.nunique()),
        "patients_audited": int(census.subject.nunique()),
        "candidate_opportunity_severe_fits": int(candidate.candidate_opportunity_severe.sum()),
        "available_l2m_controls": int(len(matched)),
        "available_l2m_fits": int(matched.fit_id.nunique()),
        "old_shared_l2m_controls": int((matched.reference_l3_source == "EXACT_REUSED_V0_3_SHARED_L3").sum()),
        "formal_v0_5_l2m_controls": int((matched.reference_l3_source == "FORMAL_V0_5_L3").sum()),
        "all_available_controls_exact": bool(matched.all_exact.all()),
        "minimum_pairing_disruption_fraction": float(matched.pairing_disruption_fraction.min()),
        "future_l2m_dependency": "GENERATE_AFTER_EACH_FORMAL_L3_FINAL_MASK_BEFORE_L2M_REFIT",
        "matching_algorithm": {
            "max_restarts": 100,
            "attempts_per_restart": 10000,
            "distance_bins": "nonlocal_pool_deciles",
            "minimum_pairing_disruption_fraction": 0.25,
        },
        "figure": str(figure),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(OUT_ROOT / "STAGE_C_GRAPH_CONTROL_COMPLETE.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

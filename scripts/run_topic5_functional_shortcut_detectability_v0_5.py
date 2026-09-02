#!/usr/bin/env python3
"""Low-cost functional-class positive control on three real geometries."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_topic5_lbss_unit_v0_2 import (  # noqa: E402
    DEFAULTS, decision_rows, train_unit,
)
from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DETECT_ROOT = OUT_ROOT / "functional_shortcut_detectability"
GEOMETRIES = (
    "epilepsiae_1077__own_a",
    "epilepsiae_384__shared",
    "epilepsiae_958__shared",
)
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
)


def synthesize_events(contact_xy: np.ndarray, seed: int, n_events: int = 900) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    xy = np.asarray(contact_xy, dtype=float)
    n = len(xy)
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    off = ~np.eye(n, dtype=bool)
    far = np.argwhere(off & (distance >= np.quantile(distance[off], 0.80)))
    # Candidate tuple is [target, source] to match recurrent matrix semantics.
    order = np.argsort(distance[far[:, 0], far[:, 1]])[::-1]
    bridge_pairs = []
    used_sources = set()
    for index in order:
        target, source = map(int, far[index])
        if source in used_sources:
            continue
        bridge_pairs.append((target, source))
        used_sources.add(source)
        if len(bridge_pairs) >= max(2, min(4, n // 3)):
            break
    local_scale = max(float(np.median(distance[off])), 1e-3)
    ranks = np.full((n_events, n), -1, dtype=np.int16)
    length = min(n, 6)
    bridge_sources = np.asarray([source for _, source in bridge_pairs], dtype=int)
    for event in range(n_events):
        current = int(rng.choice(bridge_sources)) if event % 2 == 0 else int(rng.integers(n))
        recruited = {current}
        ranks[event, current] = 0
        for rank in range(1, length):
            available = np.asarray([contact for contact in range(n) if contact not in recruited], dtype=int)
            score = np.exp(-distance[available, current] / local_scale)
            for target, source in bridge_pairs:
                if source == current and target in available:
                    score[available == target] *= 25.0
            score = score / score.sum()
            current = int(rng.choice(available, p=score))
            recruited.add(current)
            ranks[event, current] = rank
    return ranks, bridge_pairs


def prepare_geometry(fit_id: str) -> dict:
    source = OUT_ROOT / "cache" / fit_id
    destination = DETECT_ROOT / "cache" / fit_id
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "plane.npz", destination / "plane.npz")
    plane = np.load(source / "plane.npz", allow_pickle=False)
    ranks, bridges = synthesize_events(plane["contacts_xy_mm"], seed=1701 + len(fit_id))
    split = np.full(len(ranks), 0, dtype=np.int8)
    split[int(0.70 * len(ranks)):int(0.85 * len(ranks))] = 1
    split[int(0.85 * len(ranks)):] = 2
    np.savez_compressed(
        destination / "events.npz", ranks=ranks, split=split,
        mode=np.zeros(len(ranks), dtype=np.int8),
        event_source_index=np.arange(len(ranks), dtype=np.int64),
        event_abs_time=np.arange(len(ranks), dtype=float),
    )
    provenance = {
        "subject": f"synthetic_{fit_id}", "scope": "functional_shortcut_detectability",
        "n_contacts": int(ranks.shape[1]), "n_nodes": int(plane["D_mm"].shape[0]),
        "known_bridge_pairs_target_source": bridges, "target_values_read": False,
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def attenuated_distal_nll(fit_id: str, seed: int, metrics: dict, device: torch.device) -> float:
    cache = DETECT_ROOT / "cache" / fit_id
    plane = np.load(cache / "plane.npz", allow_pickle=False)
    events = np.load(cache / "events.npz", allow_pickle=False)
    pools = build_pool_contract(plane["D_mm"])
    config = LBSSConfig(
        arm="L3_LOCAL_PLUS_LEARNED_LR", n_contacts=events["ranks"].shape[1],
        n_nodes=plane["D_mm"].shape[0], observation_operator=plane["H"],
        node_distance_mm=plane["D_mm"], local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool, nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added, seed=seed, state_dim=1,
    )
    model = LBSSModel(config).to(device)
    unit = DETECT_ROOT / "units" / fit_id / "L3_LOCAL_PLUS_LEARNED_LR" / f"seed{seed}"
    model.load_state_dict(torch.load(unit / "weights.pt", map_location=device, weights_only=True))
    with torch.no_grad():
        added = model.added_mask.bool()
        model.recurrent[0][added] = 0.0
    tensors = build_event_tensors(events["ranks"])
    test_idx = np.flatnonzero(events["split"] == 2)
    rows = decision_rows(
        model, tensors, events["ranks"], test_idx, plane["contacts_xy_mm"], device
    )
    threshold = float(metrics["distance_thresholds_mm"]["q80"])
    values = [row["contact_nll"] for row in rows if row["frontier_distance_mm"] > threshold]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    DETECT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("INPUT_CACHE_MANIFEST.json", "RUN_CONTRACT.json"):
        (DETECT_ROOT / name).write_text(json.dumps({
            "contract": "topic5_functional_shortcut_detectability_v0_5",
            "target_values_read": False,
        }, indent=2))
    for fit_id in GEOMETRIES:
        prepare_geometry(fit_id)
    config = dict(DEFAULTS)
    config.update({
        "epochs_warmup": 5, "epochs_rewire": 20, "epochs_freeze": 40,
        "patience": 8, "max_batches_per_epoch": 60, "resume_every_epochs": 5,
    })
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for fit_id in GEOMETRIES:
        for arm in ARMS:
            metrics = train_unit(
                fit_id, arm, 0, DETECT_ROOT, device, config,
                resume=True, unit_root_name="units",
                contract_label="topic5_functional_shortcut_detectability_v0_5",
            )
            row = {
                "fit_id": fit_id, "arm": arm,
                "all_contact_nll": metrics["test"]["contact_nll"],
                "distal_contact_nll": metrics["distance_bins"]["distal"]["contact_nll"],
                "rollout_spearman": metrics["rollout"]["seed_removed_spearman_median"],
                "target_values_read": False,
            }
            if arm == "L3_LOCAL_PLUS_LEARNED_LR":
                row["attenuated_distal_contact_nll"] = attenuated_distal_nll(
                    fit_id, 0, metrics, device
                )
                row["attenuation_delta_distal_nll"] = (
                    row["attenuated_distal_contact_nll"] - row["distal_contact_nll"]
                )
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DETECT_ROOT / "FUNCTIONAL_DETECTABILITY.csv", index=False)
    pivot = table.pivot(index="fit_id", columns="arm", values="distal_contact_nll")
    l3 = pivot["L3_LOCAL_PLUS_LEARNED_LR"]
    l0 = pivot["L0_LOCAL_ONLY"]
    attenuation = table.loc[
        table.arm == "L3_LOCAL_PLUS_LEARNED_LR", "attenuation_delta_distal_nll"
    ].to_numpy()
    payload = {
        "status": "PASS" if bool(np.all(l3 < l0) and np.all(attenuation > 0)) else "LIMITED_DETECTABILITY",
        "geometries": list(GEOMETRIES), "n_units": int(len(table)),
        "l3_better_than_l0_distal_count": int(np.sum(l3 < l0)),
        "l3_attenuation_harms_distal_count": int(np.sum(attenuation > 0)),
        "exact_edge_recovery_required": False, "target_values_read": False,
    }
    (DETECT_ROOT / "FUNCTIONAL_DETECTABILITY_SUMMARY.json").write_text(json.dumps(payload, indent=2))
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for index, fit_id in enumerate(GEOMETRIES):
        values = table[table.fit_id == fit_id]
        axes[0].plot(np.arange(len(ARMS)), [
            float(values.loc[values.arm == arm, "distal_contact_nll"].iloc[0]) for arm in ARMS
        ], marker="o", lw=1, alpha=0.8)
    axes[0].set_xticks(np.arange(len(ARMS)), ["L0", "L1", "L2", "L3"])
    axes[0].set(ylabel="Distal contact NLL", xlabel="Functional-class model")
    axes[0].text(-0.20, 1.05, "A", transform=axes[0].transAxes, fontsize=15, fontweight="bold")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].scatter(np.arange(len(attenuation)), attenuation, color="#b74349", s=35)
    axes[1].set(xticks=[], xlabel="Real geometries", ylabel="L3 attenuation cost")
    axes[1].text(-0.20, 1.05, "B", transform=axes[1].transAxes, fontsize=15, fontweight="bold")
    figure_root = DETECT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_root / "functional_shortcut_detectability.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    (figure_root / "README.md").write_text(
        "### functional_shortcut_detectability.png\n\n"
        "A 在三套真实患者几何上比较已知 nonlocal shortcut 生成数据的 distal NLL；"
        "B 显示冻结 L3 added edges 后，完全 attenuation 对 distal NLL 的影响。\n\n"
        "**关注点**：这是 functional-class detectability，不要求恢复精确 edge；本轮仅有限通过。\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

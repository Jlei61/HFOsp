#!/usr/bin/env python3
"""Functional-class detectability control on real patient geometries."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_topic5_lbss_unit_v0_2 import decision_rows  # noqa: E402
from src.topic5_lbss_rnn_v0_2 import (  # noqa: E402
    LBSSConfig,
    LBSSModel,
    build_pool_contract,
    source_balanced_sample,
)
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
GEOMETRIES = (
    "epilepsiae_1084__shared",
    "epilepsiae_1146__shared",
    "yuquan_chengshuai__shared",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def build_ground_truth_model(plane: dict[str, np.ndarray], seed: int) -> tuple[LBSSModel, np.ndarray]:
    distance = plane["D_mm"]
    pools = build_pool_contract(distance)
    planted = source_balanced_sample(pools.nonlocal_pool, pools.k_added, seed=seed + 991)
    config = LBSSConfig(
        arm="L3_LOCAL_PLUS_LEARNED_LR",
        n_contacts=plane["H"].shape[0],
        n_nodes=plane["H"].shape[1],
        observation_operator=plane["H"],
        node_distance_mm=distance,
        local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added,
        seed=seed,
    )
    model = LBSSModel(config)
    model.added_mask.copy_(torch.as_tensor(planted, dtype=torch.float32))
    model._refresh_node_mask()
    with torch.no_grad():
        model.recurrent.zero_()
        local = model.local_mask.bool()
        added = model.added_mask.bool()
        local_scale = torch.exp(-model.D_mm / max(1e-3, pools.r_local_mm / 2.0))
        model.recurrent[0, local] = 0.42 * local_scale[local]
        model.recurrent[0, added] = 3.0
        model.input_gain.fill_(1.0)
        model.bias.zero_()
        model.contact_bias.zero_()
        model.kappa_logit.fill_(1.5)
        model.readout_gain.fill_(5.0)
    model.freeze_mask()
    return model, planted


@torch.no_grad()
def simulate_events(
    model: LBSSModel,
    n_events: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_contacts = model.n_contacts
    ranks = np.full((n_events, n_contacts), -1, dtype=np.int16)
    anchors = np.argmax(model.H.detach().cpu().numpy(), axis=1)
    active_sources = np.flatnonzero(model.added_mask.detach().cpu().numpy()[:, anchors].sum(axis=0) > 0)
    if active_sources.size == 0:
        active_sources = np.arange(n_contacts)
    for event in range(n_events):
        start = int(rng.choice(active_sources if rng.random() < 0.8 else np.arange(n_contacts)))
        length = int(rng.integers(5, min(9, n_contacts + 1)))
        recruited = np.zeros(n_contacts, dtype=bool)
        recruited[start] = True
        ranks[event, start] = 0
        h = torch.zeros(1, model.n_nodes * model.state_dim)
        current = start
        for rank_index in range(1, length):
            x = torch.zeros(1, n_contacts)
            x[0, current] = 1.0
            h = model._step(h, x)
            logits = model._readout(h)[0].numpy()
            eligible = np.flatnonzero(~recruited)
            scaled = logits[eligible] / 0.35
            probability = np.exp(scaled - scaled.max())
            probability /= probability.sum()
            current = int(rng.choice(eligible, p=probability))
            recruited[current] = True
            ranks[event, current] = rank_index
    return ranks


def make_synthetic_cache(real_root: Path, synthetic_root: Path) -> list[str]:
    fit_ids = []
    input_files = []
    for index, geometry_fit in enumerate(GEOMETRIES):
        source = real_root / "cache" / geometry_fit
        plane_npz = np.load(source / "plane.npz", allow_pickle=False)
        plane = {name: plane_npz[name] for name in plane_npz.files}
        provenance = json.loads((source / "provenance.json").read_text())
        model, planted = build_ground_truth_model(plane, seed=1200 + index)
        ranks = simulate_events(model, n_events=1200, seed=2200 + index)
        split = np.zeros(1200, dtype=np.int8)
        split[840:1020] = 1
        split[1020:] = 2
        fit_id = f"synthetic_{geometry_fit}"
        fit_ids.append(fit_id)
        destination = synthetic_root / "cache" / fit_id
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "plane.npz", **plane)
        np.savez_compressed(
            destination / "events.npz",
            ranks=ranks,
            split=split,
            mode=np.zeros(1200, dtype=np.int8),
            event_abs_time=np.arange(1200, dtype=float),
            event_source_index=np.arange(1200, dtype=np.int64),
        )
        synthetic_provenance = dict(provenance)
        synthetic_provenance.update({
            "fit_id": fit_id,
            "subject": fit_id,
            "scope": "synthetic_detectability",
            "n_events_kept": 1200,
            "n_train": 840,
            "n_validation": 180,
            "n_test": 180,
        })
        write_json(destination / "provenance.json", synthetic_provenance)
        np.savez_compressed(destination / "planted_shortcuts.npz", planted_mask=planted)
        for name in ("plane.npz", "events.npz", "provenance.json"):
            input_files.append({"fit_id": fit_id, "file": name})
    write_json(synthetic_root / "INPUT_CACHE_MANIFEST.json", {
        "contract": "topic5_lbss_functional_detectability_inputs_v0_2",
        "target_values_read": False,
        "files": input_files,
    })
    write_json(synthetic_root / "RUN_CONTRACT.json", {
        "contract": "topic5_lbss_functional_detectability_v0_2",
        "purpose": "functional-class detectability, never exact edge recovery",
        "geometries": list(GEOMETRIES),
        "n_events_per_geometry": 1200,
        "target_values_read": False,
    })
    return fit_ids


def run_job(trainer: Path, synthetic_root: Path, fit_id: str, arm: str, device: str) -> dict:
    log = synthetic_root / "run_logs" / fit_id / f"{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = synthetic_root / "per_fit" / fit_id / arm / "seed0" / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if existing.get("converged") and existing.get("target_values_read") is False:
            return {"fit_id": fit_id, "arm": arm, "returncode": 0, "log": str(log), "reused": True}
    command = [
        sys.executable, str(trainer), "--fit-id", fit_id, "--arm", arm, "--seed", "0",
        "--out-root", str(synthetic_root), "--device", device, "--epochs-freeze", "300",
    ]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return {"fit_id": fit_id, "arm": arm, "returncode": result.returncode, "log": str(log)}


def attenuation_delta(synthetic_root: Path, fit_id: str, device: torch.device) -> float:
    cache = synthetic_root / "cache" / fit_id
    plane_npz = np.load(cache / "plane.npz", allow_pickle=False)
    plane = {name: plane_npz[name] for name in plane_npz.files}
    provenance = json.loads((cache / "provenance.json").read_text())
    events = np.load(cache / "events.npz", allow_pickle=False)
    pools = build_pool_contract(plane["D_mm"])
    model = LBSSModel(LBSSConfig(
        arm="L3_LOCAL_PLUS_LEARNED_LR",
        n_contacts=provenance["n_contacts"],
        n_nodes=provenance["n_nodes"],
        observation_operator=plane["H"],
        node_distance_mm=plane["D_mm"],
        local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added,
        seed=0,
    )).to(device)
    unit = synthetic_root / "per_fit" / fit_id / "L3_LOCAL_PLUS_LEARNED_LR" / "seed0"
    state = torch.load(unit / "weights.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    ranks = events["ranks"]
    tensors = build_event_tensors(ranks)
    test_idx = np.flatnonzero(events["split"] == 2)
    intact = decision_rows(model, tensors, ranks, test_idx, plane["contacts_xy_mm"], device)
    metrics = json.loads((unit / "metrics.json").read_text())
    q80 = metrics["distance_thresholds_mm"]["q80"]
    intact_distal = np.mean([row["contact_nll"] for row in intact if row["frontier_distance_mm"] > q80])
    with torch.no_grad():
        model.recurrent[:, model.added_mask.bool()] = 0.0
    attenuated = decision_rows(model, tensors, ranks, test_idx, plane["contacts_xy_mm"], device)
    attenuated_distal = np.mean([
        row["contact_nll"] for row in attenuated if row["frontier_distance_mm"] > q80
    ])
    return float(attenuated_distal - intact_distal)


def analyse(synthetic_root: Path, fit_ids: list[str], device: torch.device) -> dict:
    rows = []
    all_converged = True
    for fit_id in fit_ids:
        by_arm = {}
        for arm in ARMS:
            path = synthetic_root / "per_fit" / fit_id / arm / "seed0" / "metrics.json"
            if not path.exists():
                raise RuntimeError(f"missing detectability result: {path}")
            by_arm[arm] = json.loads(path.read_text())
            all_converged &= bool(by_arm[arm]["converged"])
        l3 = by_arm["L3_LOCAL_PLUS_LEARNED_LR"]["distance_bins"]["distal"]["contact_nll"]
        row = {
            "fit_id": fit_id,
            "l3_minus_l0_distal_gain": by_arm["L0_LOCAL_ONLY"]["distance_bins"]["distal"]["contact_nll"] - l3,
            "l3_minus_l1_distal_gain": by_arm["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"]["distance_bins"]["distal"]["contact_nll"] - l3,
            "l3_minus_l2_distal_gain": by_arm["L2_LOCAL_PLUS_RANDOM_LR"]["distance_bins"]["distal"]["contact_nll"] - l3,
            "true_minus_shuffle_distal_gain": by_arm["C_L3_ORDER_SHUFFLED"]["distance_bins"]["distal"]["contact_nll"] - l3,
            "l3_attenuation_distal_nll_increase": attenuation_delta(synthetic_root, fit_id, device),
        }
        rows.append(row)
    keys = [key for key in rows[0] if key != "fit_id"]
    summary = {
        "contract": "topic5_lbss_functional_detectability_summary_v0_2",
        "n_geometries": len(rows),
        "rows": rows,
        "median": {key: float(np.median([row[key] for row in rows])) for key in keys},
        "all_units_converged": bool(all_converged),
        "functional_class_detected": bool(
            all_converged
            and
            np.median([row["l3_minus_l0_distal_gain"] for row in rows]) > 0
            and np.median([row["l3_minus_l1_distal_gain"] for row in rows]) > 0
            and np.median([row["l3_minus_l2_distal_gain"] for row in rows]) > 0
            and np.median([row["true_minus_shuffle_distal_gain"] for row in rows]) > 0
            and np.median([row["l3_attenuation_distal_nll_increase"] for row in rows]) > 0
        ),
        "exact_edge_recovery_claimed": False,
        "target_values_read": False,
    }
    write_json(synthetic_root / "FUNCTIONAL_DETECTABILITY_SUMMARY.json", summary)

    figure_dir = synthetic_root / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.35))
    labels = ["vs local", "vs extra local", "vs random LR", "vs shuffle", "LR off"]
    values = np.asarray([[row[key] for key in keys] for row in rows])
    for patient_values in values:
        ax.plot(np.arange(5), patient_values, color="#b6bdc5", lw=0.8, zorder=1)
        ax.scatter(np.arange(5), patient_values, color="#68737d", s=20, zorder=2)
    ax.scatter(np.arange(5), np.median(values, axis=0), color="#c73d32", s=42, zorder=3)
    ax.axhline(0, color="#1e1e1e", lw=0.8)
    ax.set_xticks(np.arange(5), labels, rotation=25, ha="right")
    ax.set_ylabel("Distal NLL benefit")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"stage_c_functional_detectability.{suffix}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        "### stage_c_functional_detectability.png\n\n"
        "在三套真实患者几何中植入功能性非局部 shortcut 后，比较 L3 相对局部、额外局部、随机非局部和顺序打乱的远端 NLL 增益。"
        "最后一列表示关闭 L3 自身新增边后远端 NLL 的上升量；灰线是几何，红点是中位数。\n\n"
        "**关注点**：本正对照只验证流水线能辨识需要非局部通信的功能类别，不主张恢复精确边。\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2/synthetic_detectability"))
    parser.add_argument("--trainer", type=Path, default=Path("scripts/train_topic5_lbss_unit_v0_2.py"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    real_root, synthetic_root = args.real_root.resolve(), args.out_root.resolve()
    fit_ids = make_synthetic_cache(real_root, synthetic_root)
    if args.prepare_only:
        return
    jobs = [(fit_id, arm) for fit_id in fit_ids for arm in ARMS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        results = list(executor.map(
            lambda job: run_job(args.trainer.resolve(), synthetic_root, job[0], job[1], args.device),
            jobs,
        ))
    write_json(synthetic_root / "TRAINING_STATUS.json", {"jobs": results})
    failed = [row for row in results if row["returncode"] != 0]
    if failed:
        raise RuntimeError(f"detectability training failed: {failed}")
    summary = analyse(synthetic_root, fit_ids, torch.device(args.device))
    write_json(synthetic_root / "PIPELINE_COMPLETE.json", {
        "status": "PASS",
        "functional_class_detected": summary["functional_class_detected"],
        "target_values_read": False,
    })


if __name__ == "__main__":
    main()

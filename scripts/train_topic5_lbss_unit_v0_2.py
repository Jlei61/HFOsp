#!/usr/bin/env python3
"""Train one target-free LBSS-RNN v0.2 fit/arm/seed unit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_lbss_rnn_v0_2 import (  # noqa: E402
    LBSSConfig,
    LBSSModel,
    build_pool_contract,
    checkpoint_is_eligible,
    clear_recurrent_optimizer_state,
    derange_training_validation_only,
    semantic_snapshot_epochs,
    transition_frontier_distance,
)
from src.topic5_rnn_motif_v0_4 import (  # noqa: E402
    fit_rollout_size_head,
    rollout_with_size_head,
)
from src.topic5_wiring_economy_rnn import (  # noqa: E402
    build_event_tensors,
    cardinality_conditioned_nll,
    next_rank_stop_loss,
    zeta_schedule,
)


DEFAULTS: dict[str, Any] = {
    "lr": 6e-3,
    "density": 0.10,
    "added_fraction": 0.10,
    "r_local_multiplier": 2.0,
    "state_dim": 1,
    "stop_weight": 1.0,
    "epochs_warmup": 10,
    "epochs_rewire": 40,
    "epochs_freeze": 3000,
    "zeta0": 0.20,
    "patience": 12,
    "min_relative_improvement": 1e-4,
    "max_batches_per_epoch": 120,
    "max_batch": 1024,
    "min_updates_per_epoch": 8,
    "resume_every_epochs": 10,
    "gradient_clip": 5.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str, salt: int) -> int:
    value = hashlib.sha256(f"{label}|{salt}".encode()).digest()
    return int.from_bytes(value[:4], "little")


def resolve_batch(n_train: int, cfg: dict[str, Any]) -> int:
    return int(min(cfg["max_batch"], max(1, int(np.ceil(n_train / cfg["min_updates_per_epoch"])))))


def evaluate(
    model: LBSSModel,
    tensors: dict[str, torch.Tensor],
    event_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    model.eval()
    indices = np.asarray(event_indices, dtype=int)
    if indices.size == 0:
        return {key: float("nan") for key in ("next_bce", "stop_bce", "contact_nll", "top1")}
    totals = {"next_bce": 0.0, "stop_bce": 0.0, "contact_nll": 0.0, "top1": 0.0}
    decisions = 0.0
    with torch.no_grad():
        for begin in range(0, indices.size, int(batch_size)):
            chosen = torch.as_tensor(indices[begin:begin + int(batch_size)])
            batch = {key: value[chosen].to(device) for key, value in tensors.items()}
            logits, stop = model(batch["x"], batch["recruited"], batch["valid"])
            _, next_bce, stop_bce = next_rank_stop_loss(
                logits, stop, batch["target"], batch["available"], batch["valid"], batch["is_last"]
            )
            predict = batch["valid"] & ~batch["is_last"]
            nll = cardinality_conditioned_nll(logits, batch["target"], batch["available"], predict)
            masked = logits.masked_fill(~batch["available"], -1e9)
            top1 = ((masked.argmax(-1) == batch["target"].argmax(-1)) & predict).float().sum()
            weight = float(predict.float().sum())
            totals["next_bce"] += float(next_bce) * weight
            totals["stop_bce"] += float(stop_bce) * weight
            totals["contact_nll"] += float(nll) * weight
            totals["top1"] += float(top1)
            decisions += weight
    result = {key: value / max(1.0, decisions) for key, value in totals.items()}
    result["n_continue_decisions"] = int(decisions)
    return result


@torch.no_grad()
def decision_rows(
    model: LBSSModel,
    tensors: dict[str, torch.Tensor],
    ranks: np.ndarray,
    event_indices: np.ndarray,
    contact_xy_mm: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    indices = np.asarray(event_indices, dtype=int)
    for begin in range(0, len(indices), int(batch_size)):
        chosen_np = indices[begin:begin + int(batch_size)]
        chosen = torch.as_tensor(chosen_np)
        batch = {key: value[chosen].to(device) for key, value in tensors.items()}
        logits, _ = model(batch["x"], batch["recruited"], batch["valid"])
        masked = logits.masked_fill(~batch["available"], -1e9)
        log_prob = torch.log_softmax(masked, -1).detach().cpu().numpy()
        prediction = masked.argmax(-1).detach().cpu().numpy()
        target = batch["target"].detach().cpu().numpy()
        for local_index, event_index in enumerate(chosen_np):
            row = ranks[event_index]
            max_rank = int(row[row >= 0].max()) if np.any(row >= 0) else -1
            recruited: set[int] = set()
            for rank_index in range(max_rank):
                current = np.flatnonzero(row == rank_index)
                recruited.update(current.tolist())
                nxt = np.flatnonzero(row == rank_index + 1)
                distance = transition_frontier_distance(current, recruited, nxt, contact_xy_mm)
                next_target = target[local_index, rank_index]
                denominator = max(1.0, float(next_target.sum()))
                nll = float(-(log_prob[local_index, rank_index] * next_target).sum() / denominator)
                rows.append({
                    "event_index": int(event_index),
                    "rank_index": int(rank_index),
                    "frontier_distance_mm": distance,
                    "contact_nll": nll,
                    "top1": int(int(prediction[local_index, rank_index]) in set(nxt.tolist())),
                    "n_current": int(current.size),
                    "n_next": int(nxt.size),
                })
    return rows


def sequence_agreement(observed: np.ndarray, generated: list[list[int]]) -> float:
    from scipy.stats import spearmanr
    generated_rank = {contact: rank for rank, rank_set in enumerate(generated) for contact in rank_set}
    shared = [contact for contact in np.flatnonzero(observed >= 0) if int(contact) in generated_rank]
    if len(shared) < 3:
        return float("nan")
    value = spearmanr([observed[c] for c in shared], [generated_rank[int(c)] for c in shared]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def save_resume(
    path: Path,
    model: LBSSModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_rng: np.random.Generator,
    rewire_rng: np.random.Generator,
    best: float,
    best_epoch: int,
    best_runtime: dict | None,
    stale: int,
    history: list[dict],
) -> None:
    payload = {
        "runtime": model.runtime_state(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "train_rng_state": train_rng.bit_generator.state,
        "rewire_rng_state": rewire_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "best": float(best),
        "best_epoch": int(best_epoch),
        "best_runtime": best_runtime,
        "stale": int(stale),
        "history": history,
    }
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(
    path: Path,
    model: LBSSModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, np.random.Generator, np.random.Generator, float, int, dict | None, int, list[dict]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.restore_runtime_state(payload["runtime"])
    optimizer.load_state_dict(payload["optimizer"])
    train_rng = np.random.default_rng()
    train_rng.bit_generator.state = payload["train_rng_state"]
    rewire_rng = np.random.default_rng()
    rewire_rng.bit_generator.state = payload["rewire_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return (
        int(payload["epoch"]) + 1,
        train_rng,
        rewire_rng,
        float(payload["best"]),
        int(payload["best_epoch"]),
        payload["best_runtime"],
        int(payload["stale"]),
        list(payload["history"]),
    )


def train_unit(
    fit_id: str,
    arm: str,
    seed: int,
    out_root: Path,
    device: torch.device,
    cfg: dict[str, Any],
    resume: bool = True,
    unit_root_name: str = "per_fit",
    events_file_name: str | None = None,
    fixed_added_mask_path: Path | None = None,
    contract_label: str = "topic5_lbss_unit_v0_2",
) -> dict[str, Any]:
    started = time.time()
    cache = out_root / "cache" / fit_id
    plane = np.load(cache / "plane.npz", allow_pickle=False)
    events_path = cache / "events.npz"
    events = np.load(events_path, allow_pickle=False)
    provenance = json.loads((cache / "provenance.json").read_text())
    observed_ranks = events["ranks"][events["split"] >= 0].copy()
    ranks = observed_ranks.copy()
    split = events["split"][events["split"] >= 0]
    mode = events["mode"][events["split"] >= 0]
    shuffle_audit = None
    null_events_sha256 = None
    if events_file_name is not None:
        null_path = cache / events_file_name
        null_events = np.load(null_path, allow_pickle=False)
        if not np.array_equal(null_events["split"], events["split"]):
            raise RuntimeError("suffix-null split does not match the frozen real split")
        null_ranks = null_events["ranks"][events["split"] >= 0]
        # The null changes train/validation information only. Every arm is
        # evaluated against the exact same real heldout decisions.
        train_validation = (split == 0) | (split == 1)
        ranks[train_validation] = null_ranks[train_validation]
        if not np.array_equal(ranks[split == 2], observed_ranks[split == 2]):
            raise RuntimeError("suffix-null changed the heldout evaluation ranks")
        null_events_sha256 = sha256_file(null_path)
        shuffle_audit = {
            "scope": "precomputed_suffix_pairing_train_and_validation_only",
            "events_file_name": events_file_name,
            "heldout_test_unchanged": True,
            "null_events_sha256": null_events_sha256,
        }
    elif arm == "C_L3_ORDER_SHUFFLED":
        ranks, shuffle_audit = derange_training_validation_only(
            ranks, split, stable_seed(fit_id, 7717)
        )
    keep = events["split"] >= 0
    tensors = build_event_tensors(ranks)
    train_idx = np.flatnonzero(split == 0)
    val_idx = np.flatnonzero(split == 1)
    test_idx = np.flatnonzero(split == 2)
    batch_size = resolve_batch(len(train_idx), cfg)

    pools = build_pool_contract(
        plane["D_mm"], cfg["density"], cfg["added_fraction"],
        cfg["r_local_multiplier"],
    )
    fixed_added_mask = None
    graph_control = None
    if fixed_added_mask_path is not None:
        graph_control = np.load(fixed_added_mask_path, allow_pickle=False)
        fixed_added_mask = graph_control["added_mask"].astype(np.uint8)
    # v0.3 provenance used ``n_contacts``; the automated v0.5 census names
    # the same exact denominator ``n_joint_contacts``.  Accept both schemas
    # without changing the model or cohort definition.
    n_contacts = int(
        provenance["n_contacts"]
        if "n_contacts" in provenance else provenance["n_joint_contacts"]
    )
    model = LBSSModel(LBSSConfig(
        arm=arm,
        n_contacts=n_contacts,
        n_nodes=int(provenance["n_nodes"]),
        observation_operator=plane["H"],
        node_distance_mm=plane["D_mm"],
        local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added,
        seed=int(seed),
        state_dim=int(cfg["state_dim"]),
        fixed_added_mask=fixed_added_mask,
    )).to(device)
    initial_weight_match = None
    if arm == "L2M_MACRO_MATCHED_RANDOM_LR":
        if graph_control is None:
            raise RuntimeError("L2M requires --fixed-added-mask-path")
        if int(cfg["state_dim"]) != 1 or model.recurrent.shape[0] != 1:
            raise RuntimeError("v0.5 exact initial-weight matching is frozen for state_dim=1 leaky RNN")
        reference_initial = graph_control["reference_l3_initial_added_mask"].astype(bool)
        target_added = model.added_mask.detach().cpu().numpy().astype(bool)
        if int(reference_initial.sum()) != int(target_added.sum()):
            raise RuntimeError("L2M and reference L3 initial added-edge counts differ")
        with torch.no_grad():
            source_values = model.recurrent[0][torch.as_tensor(reference_initial, device=device)].clone()
            target_index = torch.as_tensor(target_added, device=device)
            model.recurrent[0][target_index] = source_values
            assigned = model.recurrent[0][target_index].detach().cpu().numpy()
        reference_values = source_values.detach().cpu().numpy()
        initial_weight_match = {
            "exact_multiset": bool(np.array_equal(np.sort(reference_values), np.sort(assigned))),
            "n_values": int(reference_values.size),
            "reference_mean": float(reference_values.mean()),
            "target_mean": float(assigned.mean()),
            "reference_std": float(reference_values.std()),
            "target_std": float(assigned.std()),
            "graph_control_sha256": sha256_file(fixed_added_mask_path),
        }
        if not initial_weight_match["exact_multiset"]:
            raise RuntimeError("L2M failed exact initial added-weight multiset matching")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    unit_dir = out_root / unit_root_name / fit_id / arm / f"seed{seed}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    resume_path = unit_dir / "resume.pt"
    snapshots = semantic_snapshot_epochs(cfg["epochs_warmup"], cfg["epochs_rewire"])
    snapshot_dir = unit_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    np.savez_compressed(snapshot_dir / "SNAPSHOT_INIT.npz", **model.graph_snapshot())

    train_rng = np.random.default_rng(seed)
    rewire_rng = np.random.default_rng(stable_seed(fit_id, seed + 29031))
    begin_epoch, best, best_epoch, best_runtime, stale, history = 0, float("inf"), -1, None, 0, []
    if resume and resume_path.exists() and not (unit_dir / "DONE.json").exists():
        begin_epoch, train_rng, rewire_rng, best, best_epoch, best_runtime, stale, history = load_resume(
            resume_path, model, optimizer
        )

    warmup = int(cfg["epochs_warmup"])
    rewiring = int(cfg["epochs_rewire"])
    freeze_epoch = snapshots["SNAPSHOT_MASK_FREEZE"]
    ceiling = warmup + rewiring + int(cfg["epochs_freeze"])
    hit_ceiling = True
    for epoch in range(begin_epoch, ceiling):
        model.train()
        order = train_rng.permutation(train_idx)
        n_batches = min(int(cfg["max_batches_per_epoch"]), int(np.ceil(len(order) / batch_size)))
        for batch_index in range(n_batches):
            chosen = torch.as_tensor(order[batch_index * batch_size:(batch_index + 1) * batch_size])
            if chosen.numel() == 0:
                continue
            batch = {key: value[chosen].to(device) for key, value in tensors.items()}
            logits, stop = model(batch["x"], batch["recruited"], batch["valid"])
            loss, _, _ = next_rank_stop_loss(
                logits, stop, batch["target"], batch["available"], batch["valid"], batch["is_last"],
                stop_weight=float(cfg["stop_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            optimizer.step()

        zeta = zeta_schedule(epoch, warmup, rewiring, float(cfg["zeta0"]))
        rewire = model.rewire_added(zeta, rewire_rng)
        clear_recurrent_optimizer_state(model, optimizer, rewire["touched"])
        if epoch == freeze_epoch:
            model.freeze_mask()
        for label, snapshot_epoch in snapshots.items():
            if label != "SNAPSHOT_INIT" and epoch == snapshot_epoch:
                np.savez_compressed(snapshot_dir / f"{label}.npz", **model.graph_snapshot())

        validation = evaluate(model, tensors, val_idx, device)
        score = validation["next_bce"] + float(cfg["stop_weight"]) * validation["stop_bce"]
        history.append({
            "epoch": epoch,
            "validation_score": score,
            "validation_contact_nll": validation["contact_nll"],
            "zeta": zeta,
            "n_rewired": int(rewire["n_drop"]),
        })
        if checkpoint_is_eligible(epoch, freeze_epoch):
            if score < best * (1.0 - float(cfg["min_relative_improvement"])):
                best = score
                best_epoch = epoch
                best_runtime = {
                    "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "mask_frozen": bool(model.mask_frozen),
                }
                stale = 0
            else:
                stale += 1
                if stale >= int(cfg["patience"]):
                    hit_ceiling = False
                    save_resume(resume_path, model, optimizer, epoch, train_rng, rewire_rng,
                                best, best_epoch, best_runtime, stale, history)
                    break
        if epoch % int(cfg["resume_every_epochs"]) == 0 or epoch == freeze_epoch:
            save_resume(resume_path, model, optimizer, epoch, train_rng, rewire_rng,
                        best, best_epoch, best_runtime, stale, history)

    if best_runtime is None or not checkpoint_is_eligible(best_epoch, freeze_epoch):
        raise RuntimeError("no checkpoint eligible after structural mask freeze")
    model.restore_runtime_state(best_runtime)
    np.savez_compressed(snapshot_dir / "SNAPSHOT_FINAL.npz", **model.graph_snapshot())

    decoder, decoder_metrics = fit_rollout_size_head(
        model, tensors, train_idx, val_idx, device, seed=seed
    )
    validation = evaluate(model, tensors, val_idx, device)
    test = evaluate(model, tensors, test_idx, device)
    distance_rows = decision_rows(model, tensors, ranks, test_idx, plane["contacts_xy_mm"], device)
    finite_train_distances = []
    for event_index in train_idx:
        row = observed_ranks[event_index]
        max_rank = int(row[row >= 0].max()) if np.any(row >= 0) else -1
        recruited: set[int] = set()
        for rank_index in range(max_rank):
            current = np.flatnonzero(row == rank_index)
            recruited.update(current.tolist())
            value = transition_frontier_distance(
                current, recruited, np.flatnonzero(row == rank_index + 1), plane["contacts_xy_mm"]
            )
            if np.isfinite(value):
                finite_train_distances.append(value)
    q50, q80 = np.quantile(finite_train_distances, [0.50, 0.80])
    for row in distance_rows:
        value = row["frontier_distance_mm"]
        row["distance_bin"] = (
            "invalid" if not np.isfinite(value) else
            "local" if value <= q50 else
            "intermediate" if value <= q80 else "distal"
        )
    bin_metrics = {}
    for label in ("local", "intermediate", "distal"):
        selected = [row for row in distance_rows if row["distance_bin"] == label]
        bin_metrics[label] = {
            "n": len(selected),
            "contact_nll": float(np.mean([row["contact_nll"] for row in selected])) if selected else float("nan"),
            "top1": float(np.mean([row["top1"] for row in selected])) if selected else float("nan"),
            "distance_median_mm": float(np.median([row["frontier_distance_mm"] for row in selected])) if selected else float("nan"),
            "inferential_eligible": len(selected) >= 20,
        }

    starts = [np.flatnonzero(ranks[index] == 0) for index in test_idx]
    generated = rollout_with_size_head(model, decoder, starts, device)
    agreements = [sequence_agreement(ranks[index], sequence) for index, sequence in zip(test_idx, generated)]
    length_ratio = [
        sum(map(len, sequence)) / max(1, int((ranks[index] >= 0).sum()))
        for index, sequence in zip(test_idx, generated)
    ]
    source_index = events["event_source_index"][keep]
    event_time = events["event_abs_time"][keep]
    rollout_records = [
        {
            "kept_event_index": int(index),
            "event_source_index": int(source_index[index]),
            "event_abs_time": float(event_time[index]),
            "mode": int(mode[index]),
            "seed_contacts": np.flatnonzero(ranks[index] == 0).astype(int).tolist(),
            "generated_rank_sets": sequence,
        }
        for index, sequence in zip(test_idx, generated)
    ]

    graph = model.graph_snapshot()
    metrics = {
        "contract": contract_label,
        "fit_id": fit_id,
        "subject": provenance["subject"],
        "scope": provenance["scope"],
        "arm": arm,
        "seed": int(seed),
        "n_contacts": n_contacts,
        "n_nodes": int(provenance["n_nodes"]),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "batch_size": batch_size,
        "converged": bool(not hit_ceiling),
        "hit_ceiling": bool(hit_ceiling),
        "n_epochs": len(history),
        "best_epoch": int(best_epoch),
        "mask_freeze_epoch": int(freeze_epoch),
        "best_checkpoint_eligible": bool(checkpoint_is_eligible(best_epoch, freeze_epoch)),
        "validation": validation,
        "test": test,
        "distance_thresholds_mm": {"q50": float(q50), "q80": float(q80)},
        "distance_bin_reference": "observed_true_order_train_events",
        "distance_bin_reference_sha256": hashlib.sha256(
            np.ascontiguousarray(observed_ranks[train_idx]).view(np.uint8)
        ).hexdigest(),
        "distance_bins": bin_metrics,
        "rollout": {
            "n": len(generated),
            "seed_removed_spearman_median": float(np.nanmedian(agreements)),
            "length_ratio_median": float(np.median(length_ratio)),
        },
        "shuffle_audit": shuffle_audit,
        "initial_weight_match": initial_weight_match,
        "graph": {
            "local_edges": int(graph["local_mask"].sum()),
            "added_edges": int(graph["added_mask"].sum()),
            "rewire_counter": int(graph["rewire_counter"]),
            "candidate_exposure_total": int(graph["exposure_count"].sum()),
            "candidate_proposal_total": int(graph["proposal_count"].sum()),
        },
        "rollout_decoder": {key: value for key, value in decoder_metrics.items() if key != "curve"},
        "config": cfg,
        "device": str(device),
        "seconds": round(time.time() - started, 2),
        "target_values_read": False,
        "producer_hashes": {
            "trainer": sha256_file(Path(__file__).resolve()),
            "model": sha256_file(ROOT / "src/topic5_lbss_rnn_v0_2.py"),
            "input_manifest": sha256_file(
                out_root / ("INPUT_CACHE_MANIFEST.json" if (out_root / "INPUT_CACHE_MANIFEST.json").exists()
                            else "FULL_PARENT_CACHE_MANIFEST.json")
            ),
            "run_contract": sha256_file(out_root / "RUN_CONTRACT.json"),
        },
    }
    temporary = unit_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, allow_nan=True))
    temporary.replace(unit_dir / "metrics.json")
    np.savez_compressed(unit_dir / "graph.npz", **graph)
    torch.save(model.state_dict(), unit_dir / "weights.pt")
    torch.save(decoder.state_dict(), unit_dir / "rollout_size_head.pt")
    (unit_dir / "history.json").write_text(json.dumps(history))
    (unit_dir / "distance_decisions.json").write_text(json.dumps(distance_rows, allow_nan=True))
    with gzip.open(unit_dir / "heldout_rollouts.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(rollout_records, stream, separators=(",", ":"))
    write_done = unit_dir / "DONE.json.tmp"
    write_done.write_text(json.dumps({
        "ok": True,
        "converged": bool(not hit_ceiling),
        "target_values_read": False,
    }))
    write_done.replace(unit_dir / "DONE.json")
    resume_path.unlink(missing_ok=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs-freeze", type=int)
    parser.add_argument(
        "--config-json", type=Path,
        help=("Optional target-free development override. Only spatial-search keys "
              "lr/density/added_fraction/r_local_multiplier/zeta0/state_dim are accepted."),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--unit-root-name", default="per_fit")
    parser.add_argument("--events-file-name")
    parser.add_argument("--fixed-added-mask-path", type=Path)
    parser.add_argument("--contract-label", default="topic5_lbss_unit_v0_2")
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    if args.config_json is not None:
        overrides = json.loads(args.config_json.read_text())
        if not isinstance(overrides, dict):
            raise ValueError("config-json must contain a JSON object")
        allowed = {"lr", "density", "added_fraction", "r_local_multiplier", "zeta0", "state_dim"}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"unsupported target-free search keys: {unknown}")
        cfg.update(overrides)
        if int(cfg["state_dim"]) not in (1, 2, 4):
            raise ValueError("target-free state_dim must be one of 1, 2, or 4")
        cfg["state_dim"] = int(cfg["state_dim"])
    if args.epochs_freeze is not None:
        cfg["epochs_freeze"] = int(args.epochs_freeze)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    metrics = train_unit(
        args.fit_id, args.arm, args.seed, args.out_root.resolve(), torch.device(args.device), cfg,
        resume=not args.no_resume, unit_root_name=args.unit_root_name,
        events_file_name=args.events_file_name,
        fixed_added_mask_path=(args.fixed_added_mask_path.resolve()
                               if args.fixed_added_mask_path is not None else None),
        contract_label=args.contract_label,
    )
    print(json.dumps({
        "fit_id": args.fit_id,
        "arm": args.arm,
        "seed": args.seed,
        "n_epochs": metrics["n_epochs"],
        "converged": metrics["converged"],
        "test_contact_nll": metrics["test"]["contact_nll"],
        "rollout_spearman": metrics["rollout"]["seed_removed_spearman_median"],
        "seconds": metrics["seconds"],
    }))


if __name__ == "__main__":
    main()

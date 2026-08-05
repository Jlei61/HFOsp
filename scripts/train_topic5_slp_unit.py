"""Train one (patient, arm, seed) unit of the SLP-RNN.

Three phases, per spec §6: a functional warm-up with open gates, a structure
phase where the wiring penalty ramps in and the Concrete temperature anneals,
and a freeze phase where the topology is fixed and only retained weights move.

Selection reads the validation partition only.  The test partition is touched
once, after the best checkpoint is chosen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_latent_rnn import (
    LATENT_ARMS,
    ModelConfig,
    SLPModel,
    build_event_tensors,
    cardinality_conditioned_nll,
    next_set_stop_loss,
)
from src.topic5_virtual_seeg_operator import (
    hop_reachability,
    knn_edge_mask,
    normalised_distance,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

DEFAULTS: Dict[str, Any] = {
    "hidden": 4,
    "microsteps": 3,
    "ordinary_hidden": 64,
    "edge_budget": 6.0,
    "knn_k": 6,
    "wiring_strength": 0.1,
    "node_density": "spec",
    "batch_size": 256,
    "lr": 3e-3,
    # Budget is deliberately generous and identical across arms.  An earlier
    # Topic 5 comparison was distorted by runs that stopped at the budget
    # ceiling, which is conservative for a positive and anti-conservative for a
    # negative; every run here records whether it converged or hit the ceiling.
    "epochs_warmup": 10,
    "epochs_structure": 25,
    "epochs_freeze": 30,
    "temperature_start": 1.0,
    "temperature_end": 0.3,
    "stop_weight": 1.0,
    "patience": 6,
    "min_relative_improvement": 1e-4,
    "max_batches_per_epoch": 120,
    "use_contact_bias": True,
    "gate_init_log_alpha": 2.0,
}


def load_patient(subject: str, cache_root: Path):
    d = cache_root / subject
    plane = np.load(d / "plane_coordinates.npz", allow_pickle=True)
    nodes = np.load(d / "latent_nodes.npz")
    operator = np.load(d / "seeg_operator.npz")
    events = np.load(d / "events.npz")
    provenance = json.loads((d / "provenance.json").read_text())
    return plane, nodes, operator, events, provenance


def make_model(arm: str, cfg: Dict[str, Any], plane, nodes, operator, seed: int) -> SLPModel:
    xy = plane["xy_mm"]
    nodes_xy = nodes["nodes_xy"]
    H = operator["H"]
    geometry = xy if arm == "CONTACT_GRAPH_RNN" else nodes_xy
    config = ModelConfig(
        arm=arm,
        n_contacts=len(xy),
        n_nodes=len(nodes_xy),
        hidden=int(cfg["hidden"]),
        microsteps=int(cfg["microsteps"]),
        ordinary_hidden=int(cfg["ordinary_hidden"]),
        edge_budget=float(cfg["edge_budget"]),
        knn_k=int(cfg["knn_k"]),
        use_contact_bias=bool(cfg["use_contact_bias"]),
        gate_init_log_alpha=float(cfg["gate_init_log_alpha"]),
        seed=seed,
        normalised_distance=normalised_distance(geometry),
        fixed_edge_mask=knn_edge_mask(nodes_xy, int(cfg["knn_k"])),
        observation_operator=H if arm in LATENT_ARMS else None,
    )
    return SLPModel(config)


def evaluate(model, tensors, temperature, device, batch_size=512) -> Dict[str, float]:
    model.eval()
    n = tensors.x.shape[0]
    totals = {"next_bce": 0.0, "stop_bce": 0.0, "contact_nll": 0.0, "top1": 0.0}
    weight = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = slice(start, min(start + batch_size, n))
            batch = _slice(tensors, chunk).to(device)
            logits, stop = model(batch.x, batch.recruited, batch.valid, temperature)
            _, next_bce, stop_bce = next_set_stop_loss(
                logits, stop, batch.target, batch.available, batch.valid, batch.is_last
            )
            predict = batch.valid & ~batch.is_last
            nll = cardinality_conditioned_nll(logits, batch.target, batch.available, predict)
            masked = logits.masked_fill(~batch.available, -1e9)
            top1 = ((masked.argmax(-1)[..., None] ==
                     batch.target.argmax(-1)[..., None]) & predict[..., None]).float().sum()
            m = float(predict.float().sum())
            totals["next_bce"] += float(next_bce) * m
            totals["stop_bce"] += float(stop_bce) * m
            totals["contact_nll"] += float(nll) * m
            totals["top1"] += float(top1)
            weight += m
    return {k: v / max(weight, 1.0) for k, v in totals.items()}


def _slice(tensors, chunk):
    from src.topic5_spatial_latent_rnn import EventTensors
    return EventTensors(
        x=tensors.x[chunk], recruited=tensors.recruited[chunk],
        available=tensors.available[chunk], target=tensors.target[chunk],
        valid=tensors.valid[chunk], is_last=tensors.is_last[chunk],
    )


def train_unit(subject: str, arm: str, seed: int, cfg: Dict[str, Any],
               out_dir: Path, device: torch.device,
               cache_root: Path | None = None) -> Dict[str, Any]:
    cache_root = cache_root or (OUT_ROOT / "cache")
    plane, nodes, operator, events, provenance = load_patient(subject, cache_root)
    group_ids = events["group_ids"]
    split = events["split"]

    partitions = {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        idx = np.flatnonzero(split == code)
        partitions[name] = build_event_tensors(group_ids[idx])

    model = make_model(arm, cfg, plane, nodes, operator, seed).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    rng = np.random.default_rng(seed)

    train = partitions["train"]
    n_train = train.x.shape[0]
    batch_size = int(cfg["batch_size"])
    learnable = arm in ("CONTACT_GRAPH_RNN", "LATENT_LEARNED_SPATIAL_RNN")

    warmup, structure, freeze = (
        int(cfg["epochs_warmup"]), int(cfg["epochs_structure"]), int(cfg["epochs_freeze"])
    )
    total_epochs = warmup + structure + freeze
    t_start, t_end = float(cfg["temperature_start"]), float(cfg["temperature_end"])

    lambda_wire = 0.0
    log_rows = []
    best = {"validation_next_bce": float("inf"), "epoch": -1}
    best_state = None
    stale = 0
    frozen_edges = None

    for epoch in range(total_epochs):
        if epoch < warmup:
            phase, temperature, ramp = "warmup", t_start, 0.0
        elif epoch < warmup + structure:
            phase = "structure"
            progress = (epoch - warmup) / max(structure - 1, 1)
            temperature = t_start + (t_end - t_start) * progress
            ramp = progress
        else:
            phase, temperature, ramp = "freeze", t_end, 1.0
            if learnable and frozen_edges is None:
                frozen_edges = model.graph.freeze_topology(temperature)
                model.graph.gate.log_alpha.requires_grad_(False)

        model.train()
        order = rng.permutation(n_train)
        n_batches = min(int(cfg["max_batches_per_epoch"]),
                        max(1, int(np.ceil(n_train / batch_size))))
        epoch_loss = 0.0
        for b in range(n_batches):
            picks = order[b * batch_size:(b + 1) * batch_size]
            if not len(picks):
                picks = order[:batch_size]
            batch = _slice(train, picks).to(device)
            logits, stop = model(batch.x, batch.recruited, batch.valid, temperature)
            task, next_bce, _ = next_set_stop_loss(
                logits, stop, batch.target, batch.available, batch.valid,
                batch.is_last, stop_weight=float(cfg["stop_weight"])
            )
            wire = model.wiring_loss(temperature)
            budget = model.edge_budget_loss(temperature)
            if learnable and phase == "structure" and lambda_wire == 0.0 and float(wire) > 0:
                # calibrate once: the requested fraction of the task loss
                lambda_wire = float(cfg["wiring_strength"]) * float(task) / float(wire)
            loss = task + ramp * lambda_wire * wire + ramp * 0.01 * budget
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            epoch_loss += float(task)

        validation = evaluate(model, partitions["validation"], temperature, device)
        row = {
            "epoch": epoch, "phase": phase, "temperature": temperature,
            "lambda_wire": lambda_wire, "train_task": epoch_loss / max(n_batches, 1),
            **{f"validation_{k}": v for k, v in validation.items()},
        }
        if learnable:
            row.update(model.graph.edge_statistics(temperature))
        log_rows.append(row)

        threshold = best["validation_next_bce"] * (1.0 - float(cfg["min_relative_improvement"]))
        if validation["next_bce"] < threshold:
            best = {"validation_next_bce": validation["next_bce"], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        # early stopping only inside the last phase, so structure formation always runs
        if phase == "freeze" and stale >= int(cfg["patience"]):
            break

    # A run that was still improving when the budget ran out cannot carry a
    # negative verdict: the number would be an artefact of the epoch ceiling.
    last_epochs = [r["validation_next_bce"] for r in log_rows[-3:]]
    still_improving = len(last_epochs) == 3 and last_epochs[-1] < last_epochs[0] * (
        1.0 - float(cfg["min_relative_improvement"])
    )
    hit_ceiling = (epoch == total_epochs - 1) and still_improving

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, partitions["test"], t_end, device)

    metrics: Dict[str, Any] = {
        "subject": subject, "arm": arm, "seed": seed,
        "n_contacts": int(provenance["n_contacts"]),
        "n_latent_nodes": int(provenance["n_latent_nodes"]),
        "n_train_events": int(partitions["train"].x.shape[0]),
        "n_validation_events": int(partitions["validation"].x.shape[0]),
        "n_test_events": int(partitions["test"].x.shape[0]),
        "best_epoch": best["epoch"],
        "epochs_run": epoch + 1,
        "epoch_budget": total_epochs,
        "hit_epoch_ceiling_while_improving": bool(hit_ceiling),
        "converged": bool(not hit_ceiling),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        **{f"test_{k}": v for k, v in test.items()},
        "validation_next_bce": best["validation_next_bce"],
        "lambda_wire": lambda_wire,
    }

    if arm in ("CONTACT_GRAPH_RNN", "LATENT_LEARNED_SPATIAL_RNN", "LATENT_FIXED_LOCAL_RNN",
               "LATENT_DENSE_RNN"):
        with torch.no_grad():
            adjacency = model.graph.adjacency(t_end).cpu().numpy()
        live = np.abs(adjacency) > 0
        distance = model.edge_distance.cpu().numpy()
        metrics.update({
            "n_edges": int(live.sum()),
            "mean_degree": float(live.sum() / adjacency.shape[0]),
            "wiring_cost": float((np.abs(adjacency) * distance).sum()),
            "mean_edge_length": float(distance[live].mean()) if live.any() else float("nan"),
        })
        anchors = np.load(cache_root / subject / "latent_nodes.npz")["contact_anchor_node"]
        if arm == "CONTACT_GRAPH_RNN":
            anchors = np.arange(provenance["n_contacts"])
        transitions = observed_transitions(group_ids[split == 2])
        metrics["hop_reachability"] = hop_reachability(
            live, anchors, transitions, int(cfg["microsteps"])
        )
        np.savez_compressed(out_dir / "graph.npz", adjacency=adjacency, distance=distance)
        with (out_dir / "graph_edges.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "target", "weight", "normalised_distance"])
            for i, j in zip(*np.nonzero(live)):
                writer.writerow([int(i), int(j), float(adjacency[i, j]), float(distance[i, j])])

    with (out_dir / "training_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in log_rows for k in r}))
        writer.writeheader()
        writer.writerows(log_rows)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out_dir / "checkpoint.pt")
    return metrics


def observed_transitions(group_ids: np.ndarray, limit: int = 2000):
    """Pairs (a, b) of contacts on consecutive ranks, sampled for the diagnostic."""
    pairs = []
    for row in group_ids[:limit]:
        order = {}
        for c, r in enumerate(row):
            if r >= 0:
                order.setdefault(int(r), []).append(c)
        for r in sorted(order)[:-1]:
            for a in order[r]:
                for b in order.get(r + 1, []):
                    pairs.append((a, b))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE.json").exists():
        print(f"skip {out_dir} (already done)")
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (out_dir / "config.json").write_text(json.dumps(
        {"subject": args.subject, "arm": args.arm, "seed": args.seed, **cfg}, indent=1
    ))
    try:
        metrics = train_unit(args.subject, args.arm, args.seed, cfg, out_dir,
                             torch.device(args.device), args.cache_root)
    except Exception as exc:  # noqa: BLE001 - a failed unit must not stop the cohort
        (out_dir / "FAILED.json").write_text(json.dumps(
            {"subject": args.subject, "arm": args.arm, "seed": args.seed,
             "error": f"{type(exc).__name__}: {exc}"}, indent=1
        ))
        print(f"FAILED {args.subject} {args.arm} seed{args.seed}: {exc}")
        return 1

    code_hash = hashlib.sha256(
        (ROOT / "src/topic5_spatial_latent_rnn.py").read_bytes()
    ).hexdigest()
    metrics["code_sha256"] = code_hash
    (out_dir / "DONE.json").write_text(json.dumps(metrics, indent=1))
    print(
        f"{args.subject:24s} {args.arm:28s} seed{args.seed} "
        f"test_bce={metrics['test_next_bce']:.4f} nll={metrics['test_contact_nll']:.4f} "
        f"top1={metrics['test_top1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

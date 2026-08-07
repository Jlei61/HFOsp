"""Train one WE-SLP-RNN v0.3 unit: one fit, one arm, one seed.

Convergence is a precondition for entering the analysis, not a footnote.  In
v0.1 every arm was given the same epoch budget and the arm that carried the
negative conclusion was the only one still improving when the budget ran out;
the conclusion had to be withdrawn.  Here the mask anneals to frozen first, the
early-stopping clock only starts after that, and a unit that reaches the ceiling
is written out with ``converged=false`` and is excluded downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_wiring_economy_rnn import (  # noqa: E402
    WEConfig,
    arm_uses_wiring_cost,
    WEModel,
    build_event_tensors,
    cardinality_conditioned_nll,
    next_rank_stop_loss,
    rollout,
    zeta_schedule,
)

OUT_ROOT = ROOT / "results/topic5_wiring_economy_slp_rnn_v0_3"

DEFAULTS: Dict[str, Any] = {
    "density": 0.10,
    "eta": 0.03,
    "d0_mm": 10.0,
    "state_dim": 1,
    "stop_weight": 1.0,
    "lr": 6e-3,
    "epochs_warmup": 10,
    "epochs_rewire": 40,
    "epochs_freeze": 350,
    "zeta0": 0.20,
    "patience": 12,
    "min_relative_improvement": 1e-4,
    "max_batches_per_epoch": 120,
    "max_batch": 1024,
    "min_updates_per_epoch": 8,
    "rollout_events": 256,
    "generator_guard_fraction": 0.15,
}


def resolve_batch(n_train: int) -> int:
    """Batch is a property of the patient, never a memory knob.

    A 249-event patient at batch 1024 gets one gradient step per epoch and its
    static baseline lands 0.3 nats from the analytic optimum.  Concurrency is
    the only legitimate way to trade memory here; changing the batch would make
    two arms of the same patient incomparable.
    """
    return int(min(DEFAULTS["max_batch"], max(1, int(np.ceil(n_train / DEFAULTS["min_updates_per_epoch"])))))


def shuffle_targets(ranks: np.ndarray, seed: int) -> np.ndarray:
    """Permute which contact holds which rank, per event.

    Keeps the participating set and the event length; destroys the order.  This
    is the control that asks how much topology the prune/regrow dynamics build
    on their own, with the task signal removed but everything else identical.
    """
    rng = np.random.default_rng(seed)
    out = ranks.copy()
    for e, row in enumerate(out):
        present = np.flatnonzero(row >= 0)
        if present.size > 1:
            out[e, present] = row[rng.permutation(present)]
    return out


def evaluate(model: WEModel, tensors: Dict[str, torch.Tensor], device, batch_size=512,
             event_mask: np.ndarray | None = None) -> Dict[str, float]:
    model.eval()
    n = tensors["x"].shape[0]
    index = np.arange(n) if event_mask is None else np.flatnonzero(event_mask)
    if index.size == 0:
        return {k: float("nan") for k in ("next_bce", "stop_bce", "contact_nll", "top1")}
    totals = {"next_bce": 0.0, "stop_bce": 0.0, "contact_nll": 0.0, "top1": 0.0}
    weight = 0.0
    with torch.no_grad():
        for start in range(0, index.size, batch_size):
            chunk = torch.as_tensor(index[start:start + batch_size])
            batch = {k: v[chunk].to(device) for k, v in tensors.items()}
            logits, stop = model(batch["x"], batch["recruited"], batch["valid"])
            _, next_bce, stop_bce = next_rank_stop_loss(
                logits, stop, batch["target"], batch["available"], batch["valid"],
                batch["is_last"])
            predict = batch["valid"] & ~batch["is_last"]
            nll = cardinality_conditioned_nll(logits, batch["target"], batch["available"], predict)
            masked = logits.masked_fill(~batch["available"], -1e9)
            top1 = ((masked.argmax(-1) == batch["target"].argmax(-1)) & predict).float().sum()
            m = float(predict.float().sum())
            totals["next_bce"] += float(next_bce) * m
            totals["stop_bce"] += float(stop_bce) * m
            totals["contact_nll"] += float(nll) * m
            totals["top1"] += float(top1)
            weight += m
    return {k: v / max(weight, 1.0) for k, v in totals.items()}


def sequence_agreement(observed: np.ndarray, generated: list[list[int]]) -> float:
    """Spearman between the observed contact order and the generated one."""
    order = {c: i for i, seq in enumerate(generated) for c in seq}
    shared = [c for c in np.flatnonzero(observed >= 0) if c in order]
    if len(shared) < 3:
        return float("nan")
    r = spearmanr([observed[c] for c in shared], [order[c] for c in shared]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def rank_disagreement(a: list[list[list[int]]], b: list[list[list[int]]]) -> float:
    """Fraction of positions where two generated repertoires differ.

    A generator with no information produces the same event whatever you condition
    it on; a claim of "the two modes are not separable" read off such a generator
    describes the sampler, not the model.
    """
    pairs = 0
    differ = 0
    for seq_a, seq_b in zip(a, b):
        flat_a = [c for s in seq_a for c in s]
        flat_b = [c for s in seq_b for c in s]
        for i in range(max(len(flat_a), len(flat_b))):
            pairs += 1
            if i >= len(flat_a) or i >= len(flat_b) or flat_a[i] != flat_b[i]:
                differ += 1
    return differ / max(1, pairs)


def train_unit(fit_id: str, arm: str, seed: int, cfg: Dict[str, Any], out_root: Path,
               device: torch.device, shuffled: bool = False, out_tag: str = "") -> Dict[str, Any]:
    cache = out_root / "cache" / fit_id
    plane = np.load(cache / "plane.npz")
    events = np.load(cache / "events.npz")
    provenance = json.loads((cache / "provenance.json").read_text())

    ranks = events["ranks"]
    split = events["split"]
    mode = events["mode"]
    if shuffled:
        ranks = shuffle_targets(ranks, seed=seed + 7717)

    keep = split >= 0
    tensors = build_event_tensors(ranks[keep])
    part = split[keep]
    mode_kept = mode[keep]
    train_idx = np.flatnonzero(part == 0)
    n_train = int(train_idx.size)
    batch_size = resolve_batch(n_train)

    config = WEConfig(
        arm=arm, cell=cfg["cell"], n_contacts=int(provenance["n_contacts"]),
        n_nodes=int(provenance["n_nodes"]), state_dim=int(cfg["state_dim"]),
        density=float(cfg["density"]), eta=float(cfg["eta"]), d0_mm=float(cfg["d0_mm"]),
        seed=seed,
        observation_operator=None if arm == "STATIC_CONTACT" else plane["H"],
        node_distance_mm=None if arm == "STATIC_CONTACT" else plane["D_mm"],
    )
    model = WEModel(config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    warmup, rewire_epochs = int(cfg["epochs_warmup"]), int(cfg["epochs_rewire"])
    ceiling = warmup + rewire_epochs + int(cfg["epochs_freeze"])
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    rng = np.random.default_rng(seed)
    started = time.time()

    for epoch in range(ceiling):
        model.train()
        order = rng.permutation(train_idx)
        n_batches = min(int(cfg["max_batches_per_epoch"]),
                        int(np.ceil(order.size / batch_size)))
        for b in range(n_batches):
            chunk = torch.as_tensor(order[b * batch_size:(b + 1) * batch_size])
            if chunk.numel() == 0:
                break
            batch = {k: v[chunk].to(device) for k, v in tensors.items()}
            logits, stop = model(batch["x"], batch["recruited"], batch["valid"])
            loss, _, _ = next_rank_stop_loss(
                logits, stop, batch["target"], batch["available"], batch["valid"],
                batch["is_last"], stop_weight=float(cfg["stop_weight"]))
            if arm_uses_wiring_cost(arm):
                loss = loss + float(cfg["eta"]) * model.wiring_cost()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

        zeta = zeta_schedule(epoch, warmup, rewire_epochs, float(cfg["zeta0"]))
        if zeta > 0.0:
            model.rewire(zeta)
        if epoch == warmup + rewire_epochs - 1:
            model.freeze_mask()

        val = evaluate(model, tensors, device, event_mask=(part == 1))
        score = val["next_bce"] + float(cfg["stop_weight"]) * val["stop_bce"]
        history.append({"epoch": epoch, "val": score, "zeta": zeta})

        # The early-stopping clock only runs once the graph has stopped moving.
        if epoch < warmup + rewire_epochs:
            continue
        if score < best * (1.0 - float(cfg["min_relative_improvement"])):
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break

    n_epochs = len(history)
    hit_ceiling = n_epochs >= ceiling
    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, tensors, device, event_mask=(part == 2))
    by_mode = {
        str(m): evaluate(model, tensors, device, event_mask=(part == 2) & (mode_kept == m))
        for m in (0, 1) if int(((part == 2) & (mode_kept == m)).sum()) > 0
    }

    # Same-start free generation on held-out events, per mode where both exist.
    test_idx = np.flatnonzero(part == 2)
    kept_ranks = ranks[keep]
    max_steps = int(tensors["valid"].shape[1])
    roll: Dict[str, Any] = {}
    generated_by_mode: Dict[str, list] = {}
    for label, sel in [("all", test_idx)] + [
            (str(m), test_idx[mode_kept[test_idx] == m]) for m in (0, 1)]:
        if sel.size == 0:
            continue
        pick = sel[:int(cfg["rollout_events"])]
        starts = [np.flatnonzero(kept_ranks[i] == 0) for i in pick]
        generated = rollout(model, starts, config.n_contacts, max_steps, device)
        agreement = [sequence_agreement(kept_ranks[i], g) for i, g in zip(pick, generated)]
        lengths = [len([c for s in g for c in s]) for g in generated]
        observed_lengths = [int((kept_ranks[i] >= 0).sum()) for i in pick]
        roll[label] = {
            "n": int(pick.size),
            "spearman_median": float(np.nanmedian(agreement)),
            "length_ratio_median": float(np.median(np.asarray(lengths, float)
                                                   / np.maximum(1, observed_lengths))),
        }
        generated_by_mode[label] = generated

    degenerate = None
    if "0" in generated_by_mode and "1" in generated_by_mode:
        n = min(len(generated_by_mode["0"]), len(generated_by_mode["1"]))
        disagreement = rank_disagreement(generated_by_mode["0"][:n], generated_by_mode["1"][:n])
        roll["mode_disagreement"] = disagreement
        degenerate = bool(disagreement < float(cfg["generator_guard_fraction"]))

    snapshot = model.graph_snapshot()
    metrics: Dict[str, Any] = {
        "fit_id": fit_id, "arm": arm, "cell": cfg["cell"], "seed": seed,
        "shuffled_targets": bool(shuffled),
        "fit_scope": provenance["scope"], "subject": provenance["subject"],
        "n_contacts": provenance["n_contacts"], "n_nodes": provenance["n_nodes"],
        "n_train": n_train, "n_validation": int((part == 1).sum()),
        "n_test": int((part == 2).sum()), "batch_size": batch_size,
        "thin": bool(n_train < 500),
        "converged": bool(not hit_ceiling), "hit_ceiling": bool(hit_ceiling),
        "n_epochs": n_epochs, "val_score": float(best),
        "uses_wiring_cost": bool(arm_uses_wiring_cost(arm)),
        "test": test, "test_by_mode": by_mode, "rollout": roll,
        "generator_degenerate": degenerate,
        "label_coverage": provenance["label_coverage"],
        "device": str(device), "seconds": round(time.time() - started, 1),
        "config": {k: cfg[k] for k in sorted(cfg)},
    }
    if snapshot:
        mask = snapshot["mask"].astype(bool)
        d = snapshot["D_mm"]
        metrics["edge_count"] = int(mask.sum())
        metrics["mean_edge_len_mm"] = float(d[mask].mean()) if mask.any() else float("nan")
        metrics["long_edge_fraction"] = float((d[mask] > float(cfg["d0_mm"])).mean()) if mask.any() else float("nan")
        metrics["c_wiring"] = float(model.wiring_cost().detach())
    else:
        metrics["edge_count"] = 0
        metrics["c_wiring"] = 0.0

    suffix = "_shuffled" if shuffled else ""
    out_dir = (out_root / "per_subject" / fit_id
               / f"{arm}{suffix}_{cfg['cell']}{out_tag}" / f"seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, indent=2)
    metrics["config_sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    (out_dir / "metrics.json.tmp").write_text(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json.tmp").rename(out_dir / "metrics.json")
    if snapshot:
        np.savez_compressed(out_dir / "graph.npz.tmp.npz", **snapshot)
        (out_dir / "graph.npz.tmp.npz").rename(out_dir / "graph.npz")
    (out_dir / "history.json").write_text(json.dumps(history))
    (out_dir / "DONE.json").write_text(json.dumps({"ok": True, "converged": metrics["converged"]}))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cell", default="rnn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--epochs-freeze", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["cell"] = args.cell
    for key, value in (("eta", args.eta), ("density", args.density),
                       ("state_dim", args.state_dim), ("epochs_freeze", args.epochs_freeze)):
        if value is not None:
            cfg[key] = value

    torch.manual_seed(args.seed)
    metrics = train_unit(args.fit_id, args.arm, args.seed, cfg, args.out_root,
                         torch.device(args.device), shuffled=args.shuffled,
                         out_tag=args.out_tag)
    print(f"{args.fit_id} {args.arm}{'_shuffled' if args.shuffled else ''} {args.cell} "
          f"seed{args.seed} epochs={metrics['n_epochs']} converged={metrics['converged']} "
          f"test_bce={metrics['test']['next_bce']:.4f} c_wiring={metrics['c_wiring']:.4f} "
          f"{metrics['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

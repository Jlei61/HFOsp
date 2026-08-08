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
import gzip
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
    zeta_schedule,
)
from src.topic5_rnn_motif_v0_4 import (  # noqa: E402
    fit_rollout_size_head,
    rollout_with_size_head,
    shuffle_rank_sets,
)
from src.topic5_we_readouts import module_lesion, unit_tuning  # noqa: E402

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
    "epochs_freeze": 3000,
    "zeta0": 0.20,
    "patience": 12,
    "min_relative_improvement": 1e-4,
    "max_batches_per_epoch": 120,
    "max_batch": 1024,
    "min_updates_per_epoch": 8,
    "rollout_events": None,
    "generator_guard_fraction": 0.15,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_if_present(path: Path) -> str:
    return sha256_file(path) if path.exists() else "NOT_PRESENT_TEST_CACHE"


def resolve_batch(n_train: int) -> int:
    """Batch is a property of the patient, never a memory knob.

    A 249-event patient at batch 1024 gets one gradient step per epoch and its
    static baseline lands 0.3 nats from the analytic optimum.  Concurrency is
    the only legitimate way to trade memory here; changing the batch would make
    two arms of the same patient incomparable.
    """
    return int(min(DEFAULTS["max_batch"], max(1, int(np.ceil(n_train / DEFAULTS["min_updates_per_epoch"])))))


def shuffle_targets(ranks: np.ndarray, seed: int) -> np.ndarray:
    """Backward-compatible alias for the v0.4 first-rank-preserving control."""
    return shuffle_rank_sets(ranks, seed=seed, keep_first=True)


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
               device: torch.device, shuffled: bool = False, out_tag: str = "",
               model_id: str | None = None, shuffle_mode: str = "none") -> Dict[str, Any]:
    cache = out_root / "cache" / fit_id
    plane = np.load(cache / "plane.npz")
    events = np.load(cache / "events.npz")
    provenance = json.loads((cache / "provenance.json").read_text())

    ranks = events["ranks"]
    split = events["split"]
    mode = events["mode"]
    if shuffled and shuffle_mode == "none":
        shuffle_mode = "keep_first"
    if shuffle_mode == "keep_first":
        ranks = shuffle_rank_sets(ranks, seed=seed + 7717, keep_first=True)
    elif shuffle_mode == "full":
        ranks = shuffle_rank_sets(ranks, seed=seed + 7717, keep_first=False)
    elif shuffle_mode != "none":
        raise ValueError(f"unknown shuffle_mode {shuffle_mode!r}")

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

    snapshots: Dict[str, Dict[str, np.ndarray]] = {
        "INIT": model.graph_snapshot(),
    }
    cost_trajectory: Dict[str, Dict[str, float]] = {}

    def record_cost(label: str, task_value: float) -> None:
        cost = (float(model.wiring_cost().detach())
                if arm != "STATIC_CONTACT" else 0.0)
        cost_trajectory[label] = {
            "task_loss": float(task_value),
            "c_wiring": cost,
            "eta_c_wiring_over_task": (
                float(cfg["eta"]) * cost / max(abs(float(task_value)), 1e-12)
            ),
        }

    initial_val = evaluate(model, tensors, device, event_mask=(part == 1))
    record_cost("INIT", initial_val["next_bce"] + float(cfg["stop_weight"]) * initial_val["stop_bce"])

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
        if epoch == warmup + max(0, rewire_epochs // 2) - 1:
            snapshots["REWIRE_MID"] = model.graph_snapshot()
        if epoch == warmup + rewire_epochs - 1:
            model.freeze_mask()
            snapshots["MASK_FREEZE"] = model.graph_snapshot()

        val = evaluate(model, tensors, device, event_mask=(part == 1))
        score = val["next_bce"] + float(cfg["stop_weight"]) * val["stop_bce"]
        history.append({"epoch": epoch, "val": score, "zeta": zeta})
        if epoch == warmup + max(0, rewire_epochs // 2) - 1:
            record_cost("REWIRE_MID", score)
        if epoch == warmup + rewire_epochs - 1:
            record_cost("MASK_FREEZE", score)

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

    snapshots["FINAL"] = model.graph_snapshot()
    final_val = evaluate(model, tensors, device, event_mask=(part == 1))
    record_cost("FINAL", final_val["next_bce"] + float(cfg["stop_weight"]) * final_val["stop_bce"])

    # The recurrent model is frozen before the common cardinality decoder sees
    # any state.  Test events are never used for this calibration.
    decoder, decoder_metrics = fit_rollout_size_head(
        model, tensors, np.flatnonzero(part == 0), np.flatnonzero(part == 1),
        device, seed=seed,
    )

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
    rollout_records: list[dict[str, Any]] = []
    for label, sel in [("all", test_idx)] + [
            (str(m), test_idx[mode_kept[test_idx] == m]) for m in (0, 1)]:
        if sel.size == 0:
            continue
        limit = cfg.get("rollout_events")
        pick = sel if limit is None else sel[:int(limit)]
        starts = [np.flatnonzero(kept_ranks[i] == 0) for i in pick]
        generated = rollout_with_size_head(model, decoder, starts, device)
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
        if label == "all":
            source_index = (events["event_source_index"][keep] if "event_source_index" in events
                            else np.arange(len(kept_ranks), dtype=np.int64))
            event_time = (events["event_abs_time"][keep] if "event_abs_time" in events
                          else np.arange(len(kept_ranks), dtype=float))
            rollout_records = [
                {
                    "kept_event_index": int(i),
                    "event_source_index": int(source_index[i]),
                    "event_abs_time": float(event_time[i]),
                    "mode": int(mode_kept[i]),
                    "seed_contacts": np.flatnonzero(kept_ranks[i] == 0).astype(int).tolist(),
                    "generated_rank_sets": sequence,
                }
                for i, sequence in zip(pick, generated)
            ]

    degenerate = None
    if "0" in generated_by_mode and "1" in generated_by_mode:
        n = min(len(generated_by_mode["0"]), len(generated_by_mode["1"]))
        disagreement = rank_disagreement(generated_by_mode["0"][:n], generated_by_mode["1"][:n])
        roll["mode_disagreement"] = disagreement
        degenerate = bool(disagreement < float(cfg["generator_guard_fraction"]))

    # Functional portrait and module lesion run here rather than in a later pass:
    # both need the trained weights, and re-instantiating them from disk is one
    # more place for a silent mismatch between what was scored and what was saved.
    tuning = unit_tuning(model, tensors, test_idx[:2048], mode_kept, device)
    lesion: Dict[str, Any] = {}
    if arm != "STATIC_CONTACT":
        nodes_xy = plane["nodes_xy_mm"]
        lesion = module_lesion(
            model, nodes_xy,
            evaluate=lambda: evaluate(model, tensors, device, event_mask=(part == 2)),
            evaluate_mode=lambda m: evaluate(model, tensors, device,
                                             event_mask=(part == 2) & (mode_kept == m)),
            seed=seed)

    snapshot = model.graph_snapshot()
    metrics: Dict[str, Any] = {
        "fit_id": fit_id, "arm": arm, "cell": cfg["cell"], "seed": seed,
        "model_id": model_id or arm,
        "shuffled_targets": bool(shuffle_mode != "none"),
        "shuffle_mode": shuffle_mode,
        "fit_scope": provenance["scope"], "subject": provenance["subject"],
        "n_contacts": provenance["n_contacts"], "n_nodes": provenance["n_nodes"],
        "n_train": n_train, "n_validation": int((part == 1).sum()),
        "n_test": int((part == 2).sum()), "batch_size": batch_size,
        "thin": bool(n_train < 500),
        "converged": bool(not hit_ceiling), "hit_ceiling": bool(hit_ceiling),
        "n_epochs": n_epochs, "val_score": float(best),
        "uses_wiring_cost": bool(arm_uses_wiring_cost(arm)),
        "validation": final_val, "test": test, "test_by_mode": by_mode, "rollout": roll,
        "generator_degenerate": degenerate,
        "label_coverage": provenance["label_coverage"],
        "device": str(device), "seconds": round(time.time() - started, 1),
        "config": {k: cfg[k] for k in sorted(cfg)},
        "rollout_decoder": {k: v for k, v in decoder_metrics.items() if k != "curve"},
        "cost_trajectory": cost_trajectory,
        "producer_hashes": {
            "trainer": sha256_file(Path(__file__).resolve()),
            "model": sha256_file(ROOT / "src/topic5_wiring_economy_rnn.py"),
            "v0_4_contract": sha256_file(ROOT / "src/topic5_rnn_motif_v0_4.py"),
            "input_manifest": sha256_if_present(out_root / "INPUT_MANIFEST.json"),
        },
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

    suffix = "_shuffled" if shuffle_mode != "none" else ""
    directory_name = model_id or f"{arm}{suffix}_{cfg['cell']}{out_tag}"
    out_dir = (out_root / "per_subject" / fit_id
               / directory_name / f"seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, indent=2)
    metrics["config_sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    (out_dir / "metrics.json.tmp").write_text(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json.tmp").rename(out_dir / "metrics.json")
    if snapshot:
        np.savez_compressed(out_dir / "graph.npz.tmp.npz", **snapshot)
        (out_dir / "graph.npz.tmp.npz").rename(out_dir / "graph.npz")
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    for label, values in snapshots.items():
        if values:
            np.savez_compressed(snapshot_dir / f"{label}.npz", **values)
    if tuning.size:
        np.savez_compressed(out_dir / "unit_tuning.npz.tmp.npz", tuning=tuning)
        (out_dir / "unit_tuning.npz.tmp.npz").rename(out_dir / "unit_tuning.npz")
    if lesion:
        (out_dir / "lesion.json").write_text(json.dumps(lesion, indent=2))
    torch.save(model.state_dict(), out_dir / "weights.pt")
    torch.save(decoder.state_dict(), out_dir / "rollout_size_head.pt")
    (out_dir / "rollout_decoder_history.json").write_text(
        json.dumps(decoder_metrics["curve"])
    )
    with gzip.open(out_dir / "heldout_rollouts.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(rollout_records, handle, separators=(",", ":"))
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
    parser.add_argument("--shuffle-mode", choices=("none", "keep_first", "full"), default="none")
    parser.add_argument("--model-id", default=None)
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
                         out_tag=args.out_tag, model_id=args.model_id,
                         shuffle_mode=args.shuffle_mode)
    print(f"{args.fit_id} {args.arm}{'_shuffled' if args.shuffled else ''} {args.cell} "
          f"seed{args.seed} epochs={metrics['n_epochs']} converged={metrics['converged']} "
          f"test_bce={metrics['test']['next_bce']:.4f} c_wiring={metrics['c_wiring']:.4f} "
          f"{metrics['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

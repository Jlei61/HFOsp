#!/usr/bin/env python3
"""Phase B3: train, audit and freeze the two unordered baselines per patient.

The baselines are the bypass the ordered models must beat.  They are frozen
before any ordered model exists, they are identical for every structure, seed
and use-phase operation, and their permutation-invariance is audited (with a
deliberate bug injection proving the audit can fail).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_data_v0_2 import (  # noqa: E402
    PRIMARY_PREFIX_LEN,
    SENSITIVITY_PREFIX_LEN,
    load_sample_set,
)
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    BASELINE_LEVELS,
    TrainConfig,
    UnorderedBaseline,
    checkpoint_objective,
    combine_logits,
    evaluate,
    fit,
    tensors_from_samples,
    training_loss,
    unordered_features,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
BASELINE_ROOT = RESULT_ROOT / "baseline"
BASELINE_RANK_GRID = (1, 2, 4, 8, 16)
BASELINE_SEED = 20260817


def contact_xy_for(patient: str) -> np.ndarray:
    frame = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2" / patient
    if frame.exists():
        return np.asarray(np.load(frame / "plane.npz")["contacts_xy_mm"], dtype=np.float32)
    from src.topic5_strict_history_data_v0_2 import load_ecog_patient
    subject = patient[1:]
    root = ROOT / "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
    return np.asarray(load_ecog_patient(root, subject).contacts_xy_mm, dtype=np.float32)


def leaky_features(batch, level: str) -> torch.Tensor:
    """Deliberately broken variant that exposes the last ordered rank set."""
    honest = unordered_features(batch, level)
    return torch.cat([honest, batch.prefix_sets[:, -1]], dim=1)


def middle_order_permutations(prefix_len: int, n_draws: int) -> list[torch.Tensor]:
    """Orderings of the prefix that keep the start set, cumulative set, prefix
    length and recruited fraction fixed.

    With a three-step prefix the group has only two elements, so the audit
    enumerates it exhaustively instead of pretending 1,000 random draws are
    distinct; longer prefixes fall back to random draws.
    """
    middle = list(range(1, prefix_len))
    if math.factorial(max(len(middle), 1)) <= 24:
        orders = [torch.as_tensor([0, *permutation]) for permutation in itertools.permutations(middle)]
    else:
        generator = torch.Generator().manual_seed(4242)
        orders = [
            torch.cat([torch.zeros(1, dtype=torch.long),
                       torch.randperm(prefix_len - 1, generator=generator) + 1])
            for _ in range(n_draws)
        ]
    return orders


def order_permutation_audit(batch, module, level: str, n_draws: int, feature_fn) -> dict:
    with torch.no_grad():
        reference = module(feature_fn(batch, level))
    orders = middle_order_permutations(batch.prefix_len, n_draws)
    max_deviation = 0.0
    invariants_held = True
    for order in orders:
        permuted = batch.index(torch.arange(batch.n_samples))
        permuted.prefix_sets = batch.prefix_sets[:, order]
        invariants_held &= bool(
            torch.equal(permuted.prefix_sets.sum(dim=1).clamp(max=1),
                        batch.cumulative_set.clamp(max=1))
        )
        with torch.no_grad():
            candidate = module(feature_fn(permuted, level))
        for key in reference:
            max_deviation = max(max_deviation, float((candidate[key] - reference[key]).abs().max()))
    group_size = math.factorial(max(batch.prefix_len - 1, 1))
    return {
        "n_orders_tested": len(orders),
        "exhaustive": group_size <= 24,
        "middle_order_group_size": group_size,
        # With a two-step prefix there is a start set and one further step, so the
        # only order-preserving permutation is the identity: the audit is vacuous
        # by construction there and cannot be made to fail.  Order-blindness at
        # prefix 2 rests on the shared feature builder, which *is* tested at
        # prefix 3.
        "applicable": group_size > 1,
        "start_and_cumulative_set_preserved": bool(invariants_held),
        "max_abs_deviation": max_deviation,
        "bitwise_invariant": max_deviation == 0.0,
    }


def train_one(job: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    device = job.get("device", "cpu")
    patient, prefix_len, level = job["patient"], job["prefix_len"], job["level"]
    samples = load_sample_set(RESULT_ROOT / "sample_cache" / f"prefix{prefix_len}" / f"{patient}.npz")
    all_rows = np.arange(samples.n_samples)
    batch = tensors_from_samples(samples, all_rows, device=device)
    contact_xy = torch.as_tensor(contact_xy_for(patient)).to(device)
    train_rows = torch.as_tensor(np.flatnonzero(samples.split == 0)).to(device)
    valid_rows = torch.as_tensor(np.flatnonzero(samples.split == 1)).to(device)
    train_batch = batch.index(train_rows)
    valid_batch = batch.index(valid_rows)
    n_features = unordered_features(train_batch, level).shape[1]
    max_rank = max(1, min(samples.n_contacts, n_features))

    selection = []
    best = None
    for rank in BASELINE_RANK_GRID:
        if rank > max_rank:
            continue
        module = UnorderedBaseline(
            level=level, n_contacts=samples.n_contacts, n_features=n_features,
            n_horizons=batch.n_horizons, max_cardinality=samples.max_cardinality, rank=rank,
        ).to(device)
        features_train = unordered_features(train_batch, level)
        features_valid = unordered_features(valid_batch, level)

        def forward(piece, rows, _module=module, _features=features_train):
            merged = combine_logits({**_module(_features[rows]), }, None)
            return training_loss(merged, piece, "full_suffix")

        def objective(_module, _features=features_valid, _batch=valid_batch):
            with torch.no_grad():
                outputs = _module(_features)
            logits = {key: value for key, value in outputs.items()}
            result = evaluate(None, logits, _batch, contact_xy)
            return checkpoint_objective(result, None)

        history = fit(
            module, forward, train_batch, valid_batch, objective,
            TrainConfig(seed=BASELINE_SEED + rank, **job.get("train", {})),
        )
        selection.append({"rank": rank, **{k: v for k, v in history.items() if k != "history"}})
        if best is None or history["best_valid_objective"] < best["history"]["best_valid_objective"]:
            best = {"rank": rank, "module": module, "history": history, "n_features": n_features}

    module = best["module"]
    with torch.no_grad():
        logits = module(unordered_features(batch, level))
    honest_audit = order_permutation_audit(batch, module, level, job["n_permutations"],
                                           unordered_features)
    leaky_module = UnorderedBaseline(
        level=level, n_contacts=samples.n_contacts, n_features=n_features + samples.n_contacts,
        n_horizons=batch.n_horizons, max_cardinality=samples.max_cardinality, rank=best["rank"],
    ).to(device)
    torch.manual_seed(7)
    with torch.no_grad():
        for parameter in leaky_module.parameters():
            parameter.normal_(0.0, 0.5)
    leaky_audit = order_permutation_audit(batch, leaky_module, level, 8, leaky_features)
    leaky_audit["audit_applicable"] = leaky_audit["applicable"]

    directory = BASELINE_ROOT / level / f"prefix{prefix_len}" / patient
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in module.state_dict().items()}, "rank": best["rank"],
                "level": level, "n_features": n_features}, directory / "checkpoint.pt")
    payload = {key: value.cpu().numpy().astype(np.float32) for key, value in logits.items()}
    temporary = directory / "logits.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, split=samples.split, **payload)
    temporary.replace(directory / "logits.npz")

    digest = hashlib.sha256()
    with (directory / "logits.npz").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    scores = {}
    for split_value, split_name in ((1, "calibration"), (2, "development_test")):
        rows = torch.as_tensor(np.flatnonzero(samples.split == split_value)).to(device)
        if rows.numel() == 0:
            continue
        piece = batch.index(rows)
        base = {key: value[rows] for key, value in logits.items()}
        result = evaluate(None, base, piece, contact_xy)
        scores[split_name] = {"per_horizon": result.per_horizon, "scalars": result.scalars}

    summary = {
        "patient": patient, "prefix_len": prefix_len, "level": level,
        "selected_rank": best["rank"], "n_features": n_features,
        "max_cardinality": samples.max_cardinality, "n_contacts": samples.n_contacts,
        "n_train": int(train_rows.numel()), "n_valid": int(valid_rows.numel()),
        "rank_selection": selection,
        "training": {k: v for k, v in best["history"].items() if k != "history"},
        "order_invariance_audit": honest_audit,
        "bug_injection_audit": {
            **leaky_audit,
            "audit_correctly_fails": (not leaky_audit["bitwise_invariant"])
            if leaky_audit["applicable"] else None,
        },
        "logits_sha256": digest.hexdigest(),
        "held_out_scores": scores,
        "trainable_parameters": int(sum(p.numel() for p in module.parameters())),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def safe_train(job: dict) -> dict:
    try:
        return train_one(job)
    except Exception:  # keep one failed unit from killing the pool
        return {"patient": job["patient"], "prefix_len": job["prefix_len"], "level": job["level"],
                "error": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--patients", default="")
    parser.add_argument("--prefix-lengths", default=f"{PRIMARY_PREFIX_LEN},{SENSITIVITY_PREFIX_LEN}")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-updates", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-only", action="store_true",
                        help="re-aggregate the on-disk per-unit summaries without training")
    arguments = parser.parse_args()

    available = sorted(
        path.stem for path in (RESULT_ROOT / "sample_cache" / f"prefix{PRIMARY_PREFIX_LEN}").glob("*.npz")
    )
    patients = arguments.patients.split(",") if arguments.patients else available
    jobs = [
        {"patient": patient, "prefix_len": int(prefix), "level": level,
         "n_permutations": arguments.permutations, "device": arguments.device,
         "train": {"max_epochs": arguments.max_epochs, "max_seconds": arguments.max_seconds,
                   "batch_size": arguments.batch_size,
                   "min_updates_per_epoch": arguments.min_updates}}
        for patient in patients
        for prefix in arguments.prefix_lengths.split(",")
        for level in BASELINE_LEVELS
    ]
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.audit_only:
        results = []
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            results = list(pool.map(safe_train, jobs))

    failures = [row for row in results if "error" in row]
    on_disk = [json.loads(path.read_text())
               for path in sorted(BASELINE_ROOT.rglob("summary.json"))]
    audit = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_unordered_invariance",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "n_units_this_invocation": len(results),
        "n_units_on_disk": len(on_disk),
        "n_failed": len(failures),
        "all_bitwise_invariant": all(
            row["order_invariance_audit"]["bitwise_invariant"] for row in on_disk),
        "n_units_with_a_nontrivial_order_group": sum(
            1 for row in on_disk if row["order_invariance_audit"]["middle_order_group_size"] > 1),
        "n_units_with_a_vacuous_order_group": sum(
            1 for row in on_disk if row["order_invariance_audit"]["middle_order_group_size"] <= 1),
        "vacuous_order_group_reason": "prefix length 2 leaves no middle rank set to permute; "
                                      "order-blindness there rests on the shared feature builder, "
                                      "which is tested at prefix length 3",
        "bug_injection_correctly_fails": all(
            row["bug_injection_audit"]["audit_correctly_fails"] for row in on_disk
            if row["bug_injection_audit"]["middle_order_group_size"] > 1),
        "min_gradient_updates": min(
            (row["training"]["gradient_updates"] for row in on_disk), default=0),
        "units": on_disk,
    }
    (BASELINE_ROOT / "UNORDERED_INVARIANCE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"baseline units this run: {len(results)}  on disk: {len(on_disk)}  failed: {len(failures)}")
    print(f"  min gradient updates       : {audit['min_gradient_updates']}")
    print(f"  units with testable order group: {audit['n_units_with_a_nontrivial_order_group']} "
          f"(vacuous at prefix 2: {audit['n_units_with_a_vacuous_order_group']})")
    print(f"  all bitwise order-invariant : {audit['all_bitwise_invariant']}")
    print(f"  bug injection detected      : {audit['bug_injection_correctly_fails']}")
    for row in failures:
        print(f"  FAILED {row['patient']} prefix{row['prefix_len']} {row['level']}")
        print(row["error"].splitlines()[-1])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

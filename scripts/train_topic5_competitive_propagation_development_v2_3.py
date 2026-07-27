#!/usr/bin/env python3
"""Train one resumable v2.3 development run on sealed interictal inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_competitive_propagation_v2_3 import (  # noqa: E402
    CompetitivePropagationRNN,
    has_non_source_tie,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_node_hazard,
    logit,
)


DATASET = (
    ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
)
OUT = (
    ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
)
AUDIT = OUT / "input_audit"
V22_TARGET = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2"
    / "target_audit/TARGET_METADATA_GATE.json"
)
DEVELOPMENT = (
    "epilepsiae_1077",
    "epilepsiae_1146",
    "yuquan_chengshuai",
)
PERSISTENCE = {
    "p025_c050": (0.25, 0.50),
    "p050_c075": (0.50, 0.75),
    "p050_c090": (0.50, 0.90),
}
LEARNING_RATES = (0.003, 0.01)
SEEDS = (17, 29)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_subject(subject: str) -> dict[str, Any]:
    path = DATASET / "per_subject" / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        counts = np.asarray(data["event_group_count"], dtype=np.int64)
        split = np.asarray(data["event_split"], dtype=np.uint8)
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
        times = np.asarray(data["event_abs_time"], dtype=np.float64)
    if (
        groups.ndim != 2
        or counts.shape != (len(groups),)
        or split.shape != (len(groups),)
        or coords.shape != (groups.shape[1], 3)
        or not np.all(np.isfinite(coords))
    ):
        raise ValueError(f"{subject}: invalid dataset schema")
    if not np.all(np.diff(times) >= 0):
        raise ValueError(f"{subject}: events are not chronological")
    if np.any(counts < 2):
        raise ValueError(f"{subject}: event without next-contact decision")
    keep = np.asarray(
        [not has_non_source_tie(event) for event in groups], dtype=bool
    )
    train80 = np.flatnonzero((split == 0) & keep)
    heldout20 = np.flatnonzero((split == 1) & keep)
    fit_count = int(np.floor(0.75 * len(train80)))
    if fit_count <= 0 or fit_count >= len(train80) or len(heldout20) == 0:
        raise ValueError(f"{subject}: empty development partition")
    return {
        "path": path,
        "groups": groups,
        "counts": counts,
        "coords": coords,
        "partitions": {
            "fit60": train80[:fit_count],
            "validation20": train80[fit_count:],
            "heldout20_sealed": heldout20,
        },
        "train80": train80,
        "n_tied_excluded": int((~keep).sum()),
    }


def axis_for_subject(subject: str) -> np.ndarray:
    table = pd.read_csv(AUDIT / "development_axis_inventory.csv")
    row = table.loc[table.subject.astype(str) == subject]
    if len(row) != 1:
        raise ValueError(f"{subject}: development axis missing or duplicated")
    return row[["axis_x", "axis_y", "axis_z"]].to_numpy(float)[0]


def node_only_event_losses(
    groups: torch.Tensor,
    counts: torch.Tensor,
    node_logit: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(groups)
    dtype = node_logit.dtype
    loss_sum = torch.zeros(batch_size, dtype=dtype, device=groups.device)
    decision_count = torch.zeros_like(loss_sum)
    for step in range(int(counts.max().item()) - 1):
        active = counts > (step + 1)
        seen = (groups >= 0) & (groups <= step)
        eligible = ~seen
        score = node_logit[None, :].expand_as(groups)
        masked = score.masked_fill(~eligible, -torch.inf)
        target = groups == (step + 1)
        target_score = torch.where(
            target, score, torch.zeros_like(score)
        ).sum(dim=1)
        loss = torch.logsumexp(masked, dim=1) - target_score
        loss_sum += torch.where(active, loss, torch.zeros_like(loss))
        decision_count += active.to(dtype)
    return loss_sum / decision_count.clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: CompetitivePropagationRNN,
    groups: torch.Tensor,
    counts: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> dict[str, float | int | bool]:
    model.eval()
    full_values: list[np.ndarray] = []
    node_values: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch = torch.as_tensor(
            indices[start : start + batch_size],
            dtype=torch.long,
            device=groups.device,
        )
        current_groups = groups[batch]
        current_counts = counts[batch]
        full_values.append(
            model.forward_batch(
                current_groups, current_counts
            ).event_losses.detach().cpu().numpy()
        )
        node_values.append(
            node_only_event_losses(
                current_groups, current_counts, model.node_logit
            ).detach().cpu().numpy()
        )
    full = np.concatenate(full_values)
    node = np.concatenate(node_values)
    return {
        "n_events": int(len(full)),
        "full_categorical_nll": float(full.mean()),
        "node_categorical_nll": float(node.mean()),
        "full_over_node_benefit": float(node.mean() - full.mean()),
        "finite": bool(np.all(np.isfinite(full)) and np.all(np.isfinite(node))),
    }


def truncate_epoch_log(path: Path, start_epoch: int) -> None:
    if not path.exists():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["epoch"]) < start_epoch:
            rows.append(json.dumps(row, ensure_ascii=False))
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.subject not in DEVELOPMENT:
        raise SystemExit("subject is outside the frozen development cohort")
    if args.persistence not in PERSISTENCE:
        raise SystemExit("persistence pair is outside the frozen grid")
    if args.learning_rate not in LEARNING_RATES:
        raise SystemExit("learning rate is outside the frozen grid")
    if args.seed not in SEEDS:
        raise SystemExit("seed is outside the frozen development set")
    if args.batch_size not in (512, 1024, 2048):
        raise SystemExit("batch size is outside the frozen OOM fallback set")
    status = json.loads(
        (AUDIT / "INPUT_AUDIT_STATUS.json").read_text(encoding="utf-8")
    )
    target = json.loads(V22_TARGET.read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("target_values_read"):
        raise SystemExit("v2.3 input audit is not sealed")
    if any(
        bool(target.get(key, False))
        for key in (
            "energy_values_read",
            "recruitment_values_read",
            "target_values_read",
        )
    ):
        raise SystemExit("early-ictal target seal is not intact")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    set_determinism(args.seed)
    record = load_subject(args.subject)
    partitions = record["partitions"]
    if args.smoke:
        partitions = {
            name: values[: min(args.smoke_events, len(values))]
            for name, values in partitions.items()
        }
    node_hazard = estimate_node_hazard(
        record["groups"], record["train80"]
    )
    node_logit = logit(node_hazard)
    rho_p, rho_c = PERSISTENCE[args.persistence]
    model = CompetitivePropagationRNN(
        coords=record["coords"],
        axis=axis_for_subject(args.subject),
        node_logit=node_logit,
        rho_propagation=rho_p,
        rho_competition=rho_c,
    ).to(device)
    groups = torch.as_tensor(
        record["groups"], dtype=torch.long, device=device
    )
    counts = torch.as_tensor(
        record["counts"], dtype=torch.long, device=device
    )

    learning_label = f"lr_{args.learning_rate:g}".replace(".", "p")
    mode = "smoke" if args.smoke else "grid"
    run_root = (
        OUT
        / "development"
        / mode
        / args.subject
        / args.persistence
        / learning_label
        / f"seed_{args.seed}"
    )
    complete = run_root / "COMPLETE"
    if complete.exists() and not args.force:
        print(f"already complete: {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "contract": "topic5_symmetric_axis_competitive_propagation_v2_3",
        "subject": args.subject,
        "persistence_label": args.persistence,
        "rho_propagation": rho_p,
        "rho_competition": rho_c,
        "learning_rate": args.learning_rate,
        "weight_decay": 1.0e-4,
        "gradient_clip": 5.0,
        "max_epochs": 3 if args.smoke else 200,
        "patience": 2 if args.smoke else 20,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "smoke": args.smoke,
        "partitions": {key: int(len(value)) for key, value in partitions.items()},
        "node_bias_estimation": "chronological_train80_Beta_1_1",
        "axis_estimation": "chronological_train80_32_fixed_directions",
        "input_npz": str(record["path"].relative_to(ROOT)),
        "input_sha256": sha256(record["path"]),
        "input_audit_sha256": sha256(AUDIT / "INPUT_AUDIT_STATUS.json"),
        "core_sha256": sha256(
            ROOT / "src/topic5_competitive_propagation_v2_3.py"
        ),
        "trainer_sha256": sha256(Path(__file__)),
        "git_commit": git_commit(),
        "n_tied_events_excluded": record["n_tied_excluded"],
        "heldout_used_for_optimizer_selection": False,
        "target_values_read": False,
    }
    atomic_json(run_root / "resolved_config.json", resolved)
    atomic_json(
        run_root / "run_state.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_unix": time.time(),
            "resuming": (run_root / "last.pt").exists(),
            "target_values_read": False,
        },
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    last_path = run_root / "last.pt"
    best_path = run_root / "best.pt"
    epoch_log = run_root / "epochs.jsonl"
    start_epoch = 0
    best_validation = float("inf")
    best_epoch = -1
    stale_epochs = 0
    if last_path.exists() and not args.force:
        checkpoint = torch.load(
            last_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
    elif args.force:
        for path in (last_path, best_path, complete):
            path.unlink(missing_ok=True)
        epoch_log.write_text("", encoding="utf-8")
    truncate_epoch_log(epoch_log, start_epoch)

    max_epochs = 3 if args.smoke else 200
    patience = 2 if args.smoke else 20
    started = time.time()
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, max_epochs):
            last_epoch = epoch
            model.train()
            order = np.random.default_rng(args.seed + epoch).permutation(
                partitions["fit60"]
            )
            training_values: list[float] = []
            last_gradient = float("nan")
            for start in range(0, len(order), args.batch_size):
                batch = torch.as_tensor(
                    order[start : start + args.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = model.forward_batch(
                    groups[batch], counts[batch]
                ).event_losses.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training objective")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )
                optimizer.step()
                training_values.append(float(loss.detach().cpu()))
                last_gradient = float(gradient.detach().cpu())

            validation = evaluate(
                model,
                groups,
                counts,
                partitions["validation20"],
                max(args.batch_size, 2048),
            )
            score = float(validation["full_categorical_nll"])
            improved = score < best_validation - 1.0e-7
            if improved:
                best_validation = score
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "validation_nll": score,
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation": best_validation,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                },
                last_path,
            )
            epoch_record = {
                "epoch": epoch,
                "training_categorical_nll": float(np.mean(training_values)),
                "validation_categorical_nll": score,
                "validation_node_nll": validation["node_categorical_nll"],
                "validation_full_over_node_benefit": validation[
                    "full_over_node_benefit"
                ],
                "gradient_norm_last_batch": last_gradient,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "elapsed_seconds": time.time() - started,
                "parameters": model.parameter_summary(),
                "target_values_read": False,
            }
            with epoch_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record) + "\n")
            print(json.dumps(epoch_record), flush=True)
            if stale_epochs >= patience:
                break

        if not best_path.exists():
            raise RuntimeError("training ended without a best checkpoint")
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        metrics = {
            name: evaluate(
                model,
                groups,
                counts,
                indices,
                max(args.batch_size, 2048),
            )
            for name, indices in partitions.items()
            if name != "heldout20_sealed"
        }
        result = {
            "status": "COMPLETE",
            "subject": args.subject,
            "persistence_label": args.persistence,
            "rho_propagation": rho_p,
            "rho_competition": rho_c,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "smoke": args.smoke,
            "best_epoch": int(best["epoch"]),
            "epochs_completed": int(last_epoch + 1),
            "early_stopped": bool(last_epoch + 1 < max_epochs),
            "metrics": metrics,
            "parameters": model.parameter_summary(),
            "resource": {
                "runtime_seconds": time.time() - started,
                "peak_rss_gb": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / (1024.0**2)
                ),
                "peak_cuda_allocated_gb": (
                    torch.cuda.max_memory_allocated(device) / (1024.0**3)
                    if device.type == "cuda"
                    else 0.0
                ),
                "peak_cuda_reserved_gb": (
                    torch.cuda.max_memory_reserved(device) / (1024.0**3)
                    if device.type == "cuda"
                    else 0.0
                ),
            },
            "heldout20_status": "sealed_not_evaluated_in_development_grid",
            "target_values_read": False,
        }
        atomic_json(run_root / "metrics.json", result)
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "COMPLETE",
                "pid": os.getpid(),
                "finished_unix": time.time(),
                "target_values_read": False,
            },
        )
        complete.write_text("COMPLETE\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "FAILED",
                "pid": os.getpid(),
                "finished_unix": time.time(),
                "error": repr(error),
                "target_values_read": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--persistence", required=True, choices=PERSISTENCE)
    parser.add_argument(
        "--learning-rate", type=float, required=True, choices=LEARNING_RATES
    )
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-events", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

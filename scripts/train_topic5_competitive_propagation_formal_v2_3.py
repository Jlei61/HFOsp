#!/usr/bin/env python3
"""Train one patient/seed across the five frozen v2.3 formal variants."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
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

from scripts.train_topic5_competitive_propagation_development_v2_3 import (  # noqa: E402
    atomic_json,
    evaluate,
    load_subject,
    set_determinism,
    sha256,
)
from src.topic5_competitive_propagation_v2_3 import (  # noqa: E402
    CompetitivePropagationRNN,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_node_hazard,
    logit,
)


BASE = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
AUDIT = BASE / "input_audit"
FREEZE = BASE / "development/DEVELOPMENT_FREEZE.json"
TARGET = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2"
    / "target_audit/TARGET_METADATA_GATE.json"
)
SEEDS = (17, 29, 43)
VARIANTS: dict[str, dict[str, bool]] = {
    "local_isotropic_two_state": {
        "local_only": True,
        "no_competition": False,
        "no_source": True,
        "no_history": False,
    },
    "axis_one_state_no_competition": {
        "local_only": False,
        "no_competition": True,
        "no_source": True,
        "no_history": False,
    },
    "axis_two_state_no_source": {
        "local_only": False,
        "no_competition": False,
        "no_source": True,
        "no_history": False,
    },
    "axis_instantaneous_no_history": {
        "local_only": False,
        "no_competition": False,
        "no_source": False,
        "no_history": True,
    },
    "axis_two_state_source_full": {
        "local_only": False,
        "no_competition": False,
        "no_source": False,
        "no_history": False,
    },
}


def axis_for_subject(subject: str) -> np.ndarray:
    table = pd.read_csv(AUDIT / "formal_axis_inventory.csv")
    row = table.loc[table.subject.astype(str) == subject]
    if len(row) != 1:
        raise ValueError(f"{subject}: frozen formal axis missing or duplicated")
    return row[["axis_x", "axis_y", "axis_z"]].to_numpy(float)[0]


def trim_epoch_log(path: Path, start_epoch: int) -> None:
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


def build_model(
    *,
    variant: str,
    coords: np.ndarray,
    axis: np.ndarray,
    node_logit: np.ndarray,
    rho_propagation: float,
    rho_competition: float,
    device: torch.device,
) -> CompetitivePropagationRNN:
    return CompetitivePropagationRNN(
        coords=coords,
        axis=axis,
        node_logit=node_logit,
        rho_propagation=rho_propagation,
        rho_competition=rho_competition,
        **VARIANTS[variant],
    ).to(device)


def fit_variant(
    *,
    variant: str,
    model: CompetitivePropagationRNN,
    groups: torch.Tensor,
    counts: torch.Tensor,
    partitions: dict[str, np.ndarray],
    run_root: Path,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    variant_root = run_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    complete = variant_root / "COMPLETE"
    metrics_path = variant_root / "metrics.json"
    if complete.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
    last_path = variant_root / "last.pt"
    best_path = variant_root / "best.pt"
    epoch_log = variant_root / "epochs.jsonl"
    start_epoch = 0
    best_validation = float("inf")
    best_epoch = -1
    stale_epochs = 0
    if last_path.exists():
        checkpoint = torch.load(
            last_path, map_location=groups.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
    trim_epoch_log(epoch_log, start_epoch)
    atomic_json(
        variant_root / "run_state.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "variant": variant,
            "start_epoch": start_epoch,
            "target_values_read": False,
        },
    )

    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        last_epoch = epoch
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(
            partitions["fit60"]
        )
        training_losses: list[float] = []
        last_gradient = float("nan")
        for start in range(0, len(order), batch_size):
            batch = torch.as_tensor(
                order[start : start + batch_size],
                dtype=torch.long,
                device=groups.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.forward_batch(
                groups[batch], counts[batch]
            ).event_losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant}: non-finite training NLL")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
            last_gradient = float(gradient.detach().cpu())

        validation = evaluate(
            model,
            groups,
            counts,
            partitions["validation20"],
            batch_size,
        )
        score = float(validation["full_categorical_nll"])
        if score < best_validation - 1.0e-7:
            best_validation = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_nll": score,
                    "variant": variant,
                    "seed": seed,
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
                "variant": variant,
                "seed": seed,
            },
            last_path,
        )
        with epoch_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "training_categorical_nll": float(
                            np.mean(training_losses)
                        ),
                        "validation_categorical_nll": score,
                        "gradient_norm_last_batch": last_gradient,
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                        "parameters": model.parameter_summary(),
                        "elapsed_seconds": time.time() - started,
                        "target_values_read": False,
                    }
                )
                + "\n"
            )
        if stale_epochs >= patience:
            break

    if not best_path.exists():
        raise RuntimeError(f"{variant}: no best checkpoint")
    best = torch.load(best_path, map_location=groups.device, weights_only=False)
    model.load_state_dict(best["model_state"])
    partition_metrics = {
        name: evaluate(model, groups, counts, indices, batch_size)
        for name, indices in partitions.items()
    }
    result = {
        "status": "COMPLETE",
        "variant": variant,
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "epochs_completed": int(last_epoch + 1),
        "early_stopped": bool(last_epoch + 1 < max_epochs),
        "metrics": partition_metrics,
        "parameters": model.parameter_summary(),
        "runtime_seconds": time.time() - started,
        "target_values_read": False,
    }
    atomic_json(metrics_path, result)
    atomic_json(
        variant_root / "run_state.json",
        {
            "status": "COMPLETE",
            "pid": os.getpid(),
            "variant": variant,
            "finished_unix": time.time(),
            "target_values_read": False,
        },
    )
    complete.write_text("COMPLETE\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-events", type=int, default=1024)
    args = parser.parse_args()

    audit = json.loads(
        (AUDIT / "INPUT_AUDIT_STATUS.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    target = json.loads(TARGET.read_text(encoding="utf-8"))
    formal_subjects = list(map(str, audit["physical_axis_formal_patients"]))
    if args.subject not in formal_subjects:
        raise SystemExit("subject is outside the frozen n=22 physical cohort")
    if audit.get("target_values_read") or freeze.get(
        "early_ictal_target_values_read"
    ):
        raise SystemExit("v2.3 target seal failed")
    if any(
        bool(target.get(key, False))
        for key in (
            "energy_values_read",
            "recruitment_values_read",
            "target_values_read",
        )
    ):
        raise SystemExit("v2.2 target seal failed")
    if freeze.get("status") != "FROZEN":
        raise SystemExit("development hyperparameters are not frozen")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    device = torch.device(args.device)
    set_determinism(args.seed)
    record = load_subject(args.subject)
    if args.smoke:
        record["partitions"] = {
            name: indices[: min(args.smoke_events, len(indices))]
            for name, indices in record["partitions"].items()
        }
    node_logit = logit(
        estimate_node_hazard(record["groups"], record["train80"])
    )
    axis = axis_for_subject(args.subject)
    groups = torch.as_tensor(
        record["groups"], dtype=torch.long, device=device
    )
    counts = torch.as_tensor(
        record["counts"], dtype=torch.long, device=device
    )
    run_kind = "smoke" if args.smoke else "runs"
    run_root = BASE / "formal" / run_kind / args.subject / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "contract": "topic5_symmetric_axis_competitive_propagation_v2_3",
        "subject": args.subject,
        "seed": args.seed,
        "device": str(device),
        "smoke": args.smoke,
        "variants": list(VARIANTS),
        "rho_propagation": freeze["rho_propagation"],
        "rho_competition": freeze["rho_competition"],
        "learning_rate": freeze["learning_rate"],
        "batch_size": freeze["batch_size"],
        "maximum_epochs": (
            3 if args.smoke else freeze["maximum_epochs"]
        ),
        "patience": 2 if args.smoke else freeze["patience"],
        "partitions": {
            name: int(len(indices))
            for name, indices in record["partitions"].items()
        },
        "node_bias_estimation": "chronological_train80_Beta_1_1",
        "axis_estimation": "frozen_transition_decomposition_train80",
        "input_npz": str(record["path"].relative_to(ROOT)),
        "input_sha256": sha256(record["path"]),
        "input_audit_sha256": sha256(AUDIT / "INPUT_AUDIT_STATUS.json"),
        "development_freeze_sha256": sha256(FREEZE),
        "core_sha256": sha256(
            ROOT / "src/topic5_competitive_propagation_v2_3.py"
        ),
        "trainer_sha256": sha256(Path(__file__)),
        "heldout_used_for_training_or_epoch_selection": False,
        "target_values_read": False,
    }
    atomic_json(run_root / "resolved_config.json", resolved)
    atomic_json(
        run_root / "run_state.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "target_values_read": False,
        },
    )
    summaries: dict[str, Any] = {}
    started = time.time()
    try:
        for variant in VARIANTS:
            set_determinism(args.seed)
            model = build_model(
                variant=variant,
                coords=record["coords"],
                axis=axis,
                node_logit=node_logit,
                rho_propagation=float(freeze["rho_propagation"]),
                rho_competition=float(freeze["rho_competition"]),
                device=device,
            )
            summaries[variant] = fit_variant(
                variant=variant,
                model=model,
                groups=groups,
                counts=counts,
                partitions=record["partitions"],
                run_root=run_root,
                seed=args.seed,
                learning_rate=float(freeze["learning_rate"]),
                batch_size=int(freeze["batch_size"]),
                max_epochs=(3 if args.smoke else int(freeze["maximum_epochs"])),
                patience=(2 if args.smoke else int(freeze["patience"])),
            )
            print(
                f"{args.subject} seed={args.seed} {variant} complete",
                flush=True,
            )
        result = {
            "status": "COMPLETE",
            "subject": args.subject,
            "seed": args.seed,
            "smoke": args.smoke,
            "variants": summaries,
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
        (run_root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
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


if __name__ == "__main__":
    main()

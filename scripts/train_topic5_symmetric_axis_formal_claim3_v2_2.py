#!/usr/bin/env python3
"""Fit one vectorized fixed-random-axis chunk for formal Claim 3.

Shared dynamics come from the corresponding completed Claim-2 LOSO checkpoint
and remain frozen.  The physical axis is fixed to a pre-generated null vector;
only patient-specific gamma and gain are re-estimated from train80.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import Tensor, nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_symmetric_axis_formal_claim2_v2_2 import (  # noqa: E402
    SharedDynamics,
    load_subject,
    make_model,
)
from src.topic5_symmetric_axis_random_controls_v2_2 import (  # noqa: E402
    fixed_axis_event_losses_batch,
    fixed_axis_operator_batch,
)


BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _require_claim3_unlock() -> tuple[dict[str, Any], Path]:
    status_path = BASE / "formal/analysis/CLAIM2_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "complete"
        or status.get("claim2_next") != "PASS"
        or status.get("claim2_future") != "PASS"
        or status.get("next_stage_allowed") is not True
    ):
        raise SystemExit("Claim 3 locked: both Claim-2 endpoints must PASS")
    null_lock = BASE / "formal/claim3_random_axis_nulls/RANDOM_AXIS_NULL_LOCK.json"
    if not null_lock.is_file():
        raise SystemExit("random-axis null lock is absent")
    return status, null_lock


def _per_direction_clip(
    gamma_raw: nn.Parameter, gain_raw: nn.Parameter, maximum: float
) -> float:
    if gamma_raw.grad is None or gain_raw.grad is None:
        raise RuntimeError("missing fixed-axis gradient")
    norm = torch.sqrt(gamma_raw.grad.square() + gain_raw.grad.square())
    scale = torch.clamp(maximum / torch.clamp(norm, min=1.0e-12), max=1.0)
    gamma_raw.grad.mul_(scale)
    gain_raw.grad.mul_(scale)
    return float(torch.max(norm).detach().cpu())


def _operator(
    *,
    scalar_model: nn.Module,
    axes: Tensor,
    gamma_raw: Tensor,
    gain_raw: Tensor,
) -> Tensor:
    return fixed_axis_operator_batch(
        coords=scalar_model.coords,
        axes=axes,
        anisotropy_ratio=scalar_model.anisotropy_ratio,
        gamma_raw=gamma_raw,
        gain_raw=gain_raw,
        local_scale=scalar_model.local_scale,
        eps=scalar_model.eps,
    )


def _fit_chunk(
    *,
    scalar_model: nn.Module,
    axes: Tensor,
    groups: Tensor,
    counts: Tensor,
    train_indices: np.ndarray,
    seed: int,
    h_train: int,
    optimizer_cfg: dict[str, Any],
    epochs: int,
    batch_size: int,
    log_path: Path,
) -> tuple[nn.Parameter, nn.Parameter, dict[str, Any]]:
    set_determinism(seed + 2_000_003)
    gamma_raw = nn.Parameter(torch.zeros(len(axes), device=axes.device))
    gain_raw = nn.Parameter(torch.zeros(len(axes), device=axes.device))
    optimizer = torch.optim.AdamW(
        [gamma_raw, gain_raw],
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    clip = float(optimizer_cfg["gradient_clip"])
    generator = np.random.default_rng(seed + 2_000_003)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    started = time.time()
    last_gradient_max = float("nan")

    for epoch in range(epochs):
        order = generator.permutation(train_indices)
        epoch_direction_loss_sum = torch.zeros(
            len(axes), device=axes.device, dtype=axes.dtype
        )
        n_events = 0
        for start in range(0, len(order), batch_size):
            batch = torch.as_tensor(
                order[start : start + batch_size],
                device=groups.device,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            operator = _operator(
                scalar_model=scalar_model,
                axes=axes,
                gamma_raw=gamma_raw,
                gain_raw=gain_raw,
            )
            losses = fixed_axis_event_losses_batch(
                operator=operator,
                groups=groups[batch],
                counts=counts[batch],
                node_bias=scalar_model.node_bias,
                rho_p=scalar_model.rho_p,
                c0=scalar_model.c0,
                c_p=scalar_model.c_p,
                c_n=scalar_model.c_n,
                training_horizon=h_train,
                eps=scalar_model.eps,
            )["event_objective"]
            direction_loss = losses.mean(dim=1)
            if not torch.isfinite(direction_loss).all():
                raise FloatingPointError("non-finite random-axis training loss")
            # Sum across independent directions preserves each direction's
            # scalar gradient.  Per-direction clipping below prevents coupling.
            direction_loss.sum().backward()
            last_gradient_max = _per_direction_clip(
                gamma_raw, gain_raw, clip
            )
            optimizer.step()
            epoch_direction_loss_sum += (
                losses.detach().sum(dim=1)
            )
            n_events += len(batch)
        if epoch % 10 == 0 or epoch + 1 == epochs:
            direction_mean = epoch_direction_loss_sum / max(n_events, 1)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "direction_loss_median": float(
                                torch.median(direction_mean).cpu()
                            ),
                            "direction_loss_min": float(
                                torch.min(direction_mean).cpu()
                            ),
                            "direction_loss_max": float(
                                torch.max(direction_mean).cpu()
                            ),
                            "gradient_norm_max": last_gradient_max,
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                    + "\n"
                )
    return gamma_raw, gain_raw, {
        "epochs": epochs,
        "runtime_seconds": time.time() - started,
        "last_gradient_norm_max": last_gradient_max,
    }


@torch.no_grad()
def _evaluate_next_nll(
    *,
    scalar_model: nn.Module,
    axes: Tensor,
    gamma_raw: Tensor,
    gain_raw: Tensor,
    groups: Tensor,
    counts: Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    operator = _operator(
        scalar_model=scalar_model,
        axes=axes,
        gamma_raw=gamma_raw,
        gain_raw=gain_raw,
    )
    total = torch.zeros(len(axes), device=axes.device, dtype=axes.dtype)
    n_events = 0
    for start in range(0, len(indices), batch_size):
        batch = torch.as_tensor(
            indices[start : start + batch_size],
            device=groups.device,
            dtype=torch.long,
        )
        values = fixed_axis_event_losses_batch(
            operator=operator,
            groups=groups[batch],
            counts=counts[batch],
            node_bias=scalar_model.node_bias,
            rho_p=scalar_model.rho_p,
            c0=scalar_model.c0,
            c_p=scalar_model.c_p,
            c_n=scalar_model.c_n,
            training_horizon=0,
            eps=scalar_model.eps,
        )["event_next_nll"]
        total += values.sum(dim=1)
        n_events += len(batch)
    return (total / n_events).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--direction-start", type=int, required=True)
    parser.add_argument("--direction-stop", type=int, required=True)
    # Match the frozen Claim-2 heldout-patient optimizer step schedule.
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claim2_status, null_lock_path = _require_claim3_unlock()
    physical_lock_path = BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json"
    physical_lock = json.loads(physical_lock_path.read_text(encoding="utf-8"))
    subjects = list(map(str, physical_lock["subjects"]))
    seeds = list(map(int, physical_lock["seeds"]))
    n_directions = int(cfg["statistics"]["random_directions"])
    if args.subject not in subjects or args.seed not in seeds:
        raise SystemExit("subject/seed outside the physical-axis formal lock")
    if not (0 <= args.direction_start < args.direction_stop <= n_directions):
        raise SystemExit("invalid direction half-open interval")
    if physical_lock["H_train"] != 3:
        raise SystemExit("frozen H3 contract drifted")

    device_name = args.device or cfg["resources"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)
    set_determinism(args.seed)

    dataset = ROOT / cfg["inputs"]["rank_dataset"] / "per_subject"
    subject = load_subject(dataset / f"{args.subject}.npz")
    directions_path = (
        BASE / "formal/claim3_random_axis_nulls" / f"{args.subject}.npy"
    )
    all_axes = np.load(directions_path, allow_pickle=False)
    if all_axes.shape != (n_directions, 3):
        raise SystemExit("random-axis file shape drifted")
    axes = torch.as_tensor(
        all_axes[args.direction_start : args.direction_stop],
        dtype=torch.float32,
        device=device,
    )

    claim2_run = (
        BASE
        / "formal/claim2_runs"
        / args.subject
        / f"seed_{args.seed}"
    )
    checkpoint_path = claim2_run / "full_heldout_model.pt"
    metrics_path = claim2_run / "metrics.json"
    if not checkpoint_path.is_file() or not (claim2_run / "COMPLETE").is_file():
        raise SystemExit("corresponding Claim-2 run is incomplete")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    shared = SharedDynamics().to(device)
    scalar_model = make_model(
        subject=subject,
        shared=shared,
        isotropic=False,
        device=device,
    )
    scalar_model.load_state_dict(checkpoint["model_state"])
    scalar_model.eval()
    for parameter in scalar_model.parameters():
        parameter.requires_grad_(False)

    groups = torch.as_tensor(
        subject["groups"], dtype=torch.long, device=device
    )
    counts = torch.as_tensor(
        subject["counts"], dtype=torch.long, device=device
    )
    chunk = f"{args.direction_start:03d}_{args.direction_stop - 1:03d}"
    run_root = (
        BASE
        / "formal/claim3_random_axis_runs"
        / args.subject
        / f"seed_{args.seed}"
        / f"chunk_{chunk}"
    )
    if (run_root / "COMPLETE").is_file() and not args.force:
        print(f"already complete: {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_root / "run_state.json",
        {
            "status": "RUNNING",
            "subject": args.subject,
            "seed": args.seed,
            "direction_start": args.direction_start,
            "direction_stop": args.direction_stop,
            "target_values_read": False,
            "started_unix": time.time(),
        },
    )
    resolved = {
        "contract": cfg["contract"]["name"],
        "version": cfg["contract"]["version"],
        "subject": args.subject,
        "seed": args.seed,
        "direction_start": args.direction_start,
        "direction_stop": args.direction_stop,
        "n_directions": args.direction_stop - args.direction_start,
        "H_train": 3,
        "fit_partition": "train80",
        "evaluation_partition": "heldout20",
        "optimized_parameters": ["gamma_raw", "gain_raw"],
        "fixed_parameters": [
            "axis",
            "shared_anisotropy_ratio",
            "shared_rho_p",
            "shared_stop",
            "node_bias",
        ],
        "epochs": int(cfg["optimizer"]["max_epochs"]),
        "batch_size": args.batch_size,
        "device": str(device),
        "input_sha256": subject["input_sha256"],
        "node_bias_sha256": subject["bias_sha256"],
        "directions_sha256": sha256(directions_path),
        "claim2_checkpoint_sha256": sha256(checkpoint_path),
        "claim2_metrics_sha256": sha256(metrics_path),
        "claim2_status_sha256": sha256(
            BASE / "formal/analysis/CLAIM2_STATUS.json"
        ),
        "null_lock_sha256": sha256(null_lock_path),
        "physical_lock_sha256": sha256(physical_lock_path),
        "core_sha256": sha256(
            ROOT / "src/topic5_symmetric_axis_propagation_state_v2_2.py"
        ),
        "vectorized_control_sha256": sha256(
            ROOT / "src/topic5_symmetric_axis_random_controls_v2_2.py"
        ),
        "trainer_sha256": sha256(Path(__file__)),
        "target_values_read": False,
    }
    atomic_json(run_root / "resolved_config.json", resolved)

    try:
        gamma_raw, gain_raw, fit_summary = _fit_chunk(
            scalar_model=scalar_model,
            axes=axes,
            groups=groups,
            counts=counts,
            train_indices=subject["train"],
            seed=args.seed,
            h_train=3,
            optimizer_cfg=cfg["optimizer"],
            epochs=int(cfg["optimizer"]["max_epochs"]),
            batch_size=args.batch_size,
            log_path=run_root / "epochs.jsonl",
        )
        random_nll = _evaluate_next_nll(
            scalar_model=scalar_model,
            axes=axes,
            gamma_raw=gamma_raw,
            gain_raw=gain_raw,
            groups=groups,
            counts=counts,
            indices=subject["heldout"],
            batch_size=max(args.batch_size, 1024),
        )
        learned = json.loads(metrics_path.read_text(encoding="utf-8"))
        learned_nll = float(
            learned["models"]["full"]["heldout_fit"]["metrics"]["heldout20"][
                "next_nll"
            ]
        )
        result = {
            "status": "complete",
            "subject": args.subject,
            "seed": args.seed,
            "direction_indices": list(
                range(args.direction_start, args.direction_stop)
            ),
            "heldout20_random_next_nll": random_nll.tolist(),
            "heldout20_learned_next_nll": learned_nll,
            "random_minus_learned": (random_nll - learned_nll).tolist(),
            "gamma": torch.sigmoid(gamma_raw).detach().cpu().tolist(),
            "gain": (
                torch.nn.functional.softplus(gain_raw) + scalar_model.eps
            ).detach().cpu().tolist(),
            "fit": fit_summary,
            "node_bias_sha256": subject["bias_sha256"],
            "target_values_read": False,
            "resource": {
                "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.0**2),
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
        }
        atomic_json(run_root / "metrics.json", result)
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "COMPLETE",
                "subject": args.subject,
                "seed": args.seed,
                "direction_start": args.direction_start,
                "direction_stop": args.direction_stop,
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        (run_root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "median_random_minus_learned": float(
                        np.median(random_nll - learned_nll)
                    ),
                    "resource": result["resource"],
                },
                indent=2,
            )
        )
    except Exception as exc:
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "FAILED",
                "subject": args.subject,
                "seed": args.seed,
                "direction_start": args.direction_start,
                "direction_stop": args.direction_stop,
                "error": repr(exc),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()

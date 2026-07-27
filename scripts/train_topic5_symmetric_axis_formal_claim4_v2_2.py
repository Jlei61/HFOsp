#!/usr/bin/env python3
"""Fit one patient-seed formal Claim-4 shared-scaffold comparison."""
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
from scripts.train_topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    batch_event_losses,
)
from src.topic5_symmetric_axis_claim4_v2_2 import (  # noqa: E402
    SOURCE_LEFT,
    SOURCE_RIGHT,
    meets_claim4_event_thresholds,
    partition_source_sides,
    side_event_indices,
)


BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
SHARED_KEYS = ("raw_anisotropy", "raw_rho", "c0", "raw_c_p", "raw_c_n")


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


def _require_claim4_unlock() -> Path:
    path = BASE / "formal/analysis/CLAIM3_STATUS.json"
    if not path.is_file():
        raise SystemExit("Claim 4 locked: Claim-3 status is absent")
    status = json.loads(path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "complete"
        or status.get("claim3_random_axis") != "PASS"
        or status.get("next_stage_allowed") is not True
    ):
        raise SystemExit("Claim 4 locked: random-axis specificity did not PASS")
    return path


def _load_checkpoint_model(
    *,
    subject: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
    isotropic: bool,
) -> nn.Module:
    shared = SharedDynamics().to(device)
    model = make_model(
        subject=subject,
        shared=shared,
        isotropic=isotropic,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _new_cross_side_model(
    *,
    subject: dict[str, Any],
    full_checkpoint: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """New patient scaffold with LOSO shared values but no full-train axis."""
    shared = SharedDynamics().to(device)
    model = make_model(
        subject=subject,
        shared=shared,
        isotropic=False,
        device=device,
    )
    state = full_checkpoint["model_state"]
    with torch.no_grad():
        for key in SHARED_KEYS:
            getattr(model, key).copy_(state[key])
        model.gamma_raw.zero_()
        model.gain_raw.zero_()
        # axis_raw deliberately remains the geometry-only PCA1 initialization.
    return model


def _fit_subset(
    *,
    model: nn.Module,
    indices: np.ndarray,
    groups: Tensor,
    counts: Tensor,
    train_axis: bool,
    seed: int,
    optimizer_cfg: dict[str, Any],
    epochs: int,
    batch_size: int,
    log_path: Path,
) -> dict[str, Any]:
    if len(indices) == 0:
        raise ValueError("cannot fit an empty source-side partition")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gamma_raw.requires_grad_(True)
    model.gain_raw.requires_grad_(True)
    if train_axis:
        model.axis_raw.requires_grad_(True)
    parameters = [model.gamma_raw, model.gain_raw]
    if train_axis:
        parameters.insert(0, model.axis_raw)
    set_determinism(seed)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    clip = float(optimizer_cfg["gradient_clip"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    started = time.time()
    last_gradient = float("nan")
    for epoch in range(epochs):
        order = generator.permutation(indices)
        losses_epoch = []
        for start in range(0, len(order), batch_size):
            batch = torch.as_tensor(
                order[start : start + batch_size],
                device=groups.device,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = batch_event_losses(
                model=model,
                groups=groups[batch],
                counts=counts[batch],
                training_horizon=3,
                evaluate_full_future=False,
            )["event_objective"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Claim-4 subset loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, clip)
            last_gradient = float(gradient.detach().cpu())
            optimizer.step()
            losses_epoch.append(float(loss.detach().cpu()))
        if epoch % 10 == 0 or epoch + 1 == epochs:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "objective": float(np.mean(losses_epoch)),
                            "gradient_norm_last_batch": last_gradient,
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                    + "\n"
                )
    model.eval()
    return {
        "n_train_events": int(len(indices)),
        "epochs": epochs,
        "runtime_seconds": time.time() - started,
        "last_gradient_norm": last_gradient,
        "axis": (
            model.axis.detach().cpu().numpy().astype(float).tolist()
        ),
        "gamma": float(model.gamma.detach().cpu()),
        "gain": float(model.gain.detach().cpu()),
    }


@torch.no_grad()
def _event_next_nll(
    *,
    model: nn.Module,
    groups: Tensor,
    counts: Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    values = []
    for start in range(0, len(indices), batch_size):
        batch = torch.as_tensor(
            indices[start : start + batch_size],
            device=groups.device,
            dtype=torch.long,
        )
        result = batch_event_losses(
            model=model,
            groups=groups[batch],
            counts=counts[batch],
            training_horizon=0,
            evaluate_full_future=False,
        )["event_next_nll"]
        values.append(result.cpu().numpy())
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(values).astype(np.float64, copy=False)


def _metric(values: np.ndarray) -> float:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("empty or non-finite heldout event metric")
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claim3_status_path = _require_claim4_unlock()
    lock_path = BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        args.subject not in lock["subjects"]
        or args.seed not in list(map(int, lock["seeds"]))
        or int(lock["H_train"]) != 3
    ):
        raise SystemExit("subject/seed/H3 outside the physical-axis formal lock")
    device_name = args.device or cfg["resources"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)
    set_determinism(args.seed)

    dataset = ROOT / cfg["inputs"]["rank_dataset"] / "per_subject"
    subject = load_subject(dataset / f"{args.subject}.npz")
    claim2_run = (
        BASE / "formal/claim2_runs" / args.subject / f"seed_{args.seed}"
    )
    full_checkpoint_path = claim2_run / "full_heldout_model.pt"
    iso_checkpoint_path = claim2_run / "local_isotropic_heldout_model.pt"
    if not (claim2_run / "COMPLETE").is_file():
        raise SystemExit("corresponding Claim-2 fold is incomplete")
    full_checkpoint = torch.load(
        full_checkpoint_path, map_location=device, weights_only=False
    )
    iso_checkpoint = torch.load(
        iso_checkpoint_path, map_location=device, weights_only=False
    )
    full_model = _load_checkpoint_model(
        subject=subject,
        checkpoint=full_checkpoint,
        device=device,
        isotropic=False,
    )
    iso_model = _load_checkpoint_model(
        subject=subject,
        checkpoint=iso_checkpoint,
        device=device,
        isotropic=True,
    )
    axis = full_model.axis.detach().cpu().numpy()
    partition = partition_source_sides(
        groups=subject["groups"],
        coords=subject["coords"],
        axis=axis,
        train_indices=subject["train"],
        heldout_indices=subject["heldout"],
    )
    counts_by_side = partition.counts()
    eligible = meets_claim4_event_thresholds(
        partition,
        min_train_per_side=int(
            cfg["statistics"]["min_train_events_per_side"]
        ),
        min_heldout_per_side=int(
            cfg["statistics"]["min_heldout_events_per_side"]
        ),
    )
    run_root = (
        BASE
        / "formal/claim4_shared_scaffold_runs"
        / args.subject
        / f"seed_{args.seed}"
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
            "target_values_read": False,
            "started_unix": time.time(),
        },
    )
    resolved = {
        "contract": cfg["contract"]["name"],
        "version": cfg["contract"]["version"],
        "subject": args.subject,
        "seed": args.seed,
        "H_train": 3,
        "source_side_axis": axis.astype(float).tolist(),
        "source_thresholds_fit_partition": "train80_only",
        "q25": partition.q25,
        "q75": partition.q75,
        "source_side_counts": counts_by_side,
        "min_train_events_per_side": int(
            cfg["statistics"]["min_train_events_per_side"]
        ),
        "min_heldout_events_per_side": int(
            cfg["statistics"]["min_heldout_events_per_side"]
        ),
        "eligible": eligible,
        "node_bias_sha256": subject["bias_sha256"],
        "input_sha256": subject["input_sha256"],
        "full_checkpoint_sha256": sha256(full_checkpoint_path),
        "isotropic_checkpoint_sha256": sha256(iso_checkpoint_path),
        "claim3_status_sha256": sha256(claim3_status_path),
        "physical_lock_sha256": sha256(lock_path),
        "core_sha256": sha256(
            ROOT / "src/topic5_symmetric_axis_propagation_state_v2_2.py"
        ),
        "source_side_code_sha256": sha256(
            ROOT / "src/topic5_symmetric_axis_claim4_v2_2.py"
        ),
        "trainer_sha256": sha256(Path(__file__)),
        "target_values_read": False,
    }
    atomic_json(run_root / "resolved_config.json", resolved)

    try:
        if not eligible:
            result = {
                "status": "not_estimable",
                "reason": (
                    partition.status
                    if partition.status != "ok"
                    else "source_side_event_count_threshold"
                ),
                "subject": args.subject,
                "seed": args.seed,
                "source_side_counts": counts_by_side,
                "q25": partition.q25,
                "q75": partition.q75,
                "node_bias_sha256": subject["bias_sha256"],
                "target_values_read": False,
            }
        else:
            train_left = side_event_indices(
                partition_indices=subject["train"],
                side=partition.train_side,
                wanted=SOURCE_LEFT,
            )
            train_right = side_event_indices(
                partition_indices=subject["train"],
                side=partition.train_side,
                wanted=SOURCE_RIGHT,
            )
            heldout_left = side_event_indices(
                partition_indices=subject["heldout"],
                side=partition.heldout_side,
                wanted=SOURCE_LEFT,
            )
            heldout_right = side_event_indices(
                partition_indices=subject["heldout"],
                side=partition.heldout_side,
                wanted=SOURCE_RIGHT,
            )
            groups = torch.as_tensor(
                subject["groups"], dtype=torch.long, device=device
            )
            event_counts = torch.as_tensor(
                subject["counts"], dtype=torch.long, device=device
            )
            full_left = _event_next_nll(
                model=full_model,
                groups=groups,
                counts=event_counts,
                indices=heldout_left,
                batch_size=args.batch_size,
            )
            full_right = _event_next_nll(
                model=full_model,
                groups=groups,
                counts=event_counts,
                indices=heldout_right,
                batch_size=args.batch_size,
            )
            iso_left = _event_next_nll(
                model=iso_model,
                groups=groups,
                counts=event_counts,
                indices=heldout_left,
                batch_size=args.batch_size,
            )
            iso_right = _event_next_nll(
                model=iso_model,
                groups=groups,
                counts=event_counts,
                indices=heldout_right,
                batch_size=args.batch_size,
            )

            two_left = _load_checkpoint_model(
                subject=subject,
                checkpoint=full_checkpoint,
                device=device,
                isotropic=False,
            )
            two_right = _load_checkpoint_model(
                subject=subject,
                checkpoint=full_checkpoint,
                device=device,
                isotropic=False,
            )
            two_left_fit = _fit_subset(
                model=two_left,
                indices=train_left,
                groups=groups,
                counts=event_counts,
                train_axis=False,
                seed=args.seed + 4_000_001,
                optimizer_cfg=cfg["optimizer"],
                epochs=int(cfg["optimizer"]["max_epochs"]),
                batch_size=args.batch_size,
                log_path=run_root / "twoW_left_epochs.jsonl",
            )
            two_right_fit = _fit_subset(
                model=two_right,
                indices=train_right,
                groups=groups,
                counts=event_counts,
                train_axis=False,
                seed=args.seed + 4_000_003,
                optimizer_cfg=cfg["optimizer"],
                epochs=int(cfg["optimizer"]["max_epochs"]),
                batch_size=args.batch_size,
                log_path=run_root / "twoW_right_epochs.jsonl",
            )
            two_left_nll = _event_next_nll(
                model=two_left,
                groups=groups,
                counts=event_counts,
                indices=heldout_left,
                batch_size=args.batch_size,
            )
            two_right_nll = _event_next_nll(
                model=two_right,
                groups=groups,
                counts=event_counts,
                indices=heldout_right,
                batch_size=args.batch_size,
            )

            cross_left = _new_cross_side_model(
                subject=subject,
                full_checkpoint=full_checkpoint,
                device=device,
            )
            cross_right = _new_cross_side_model(
                subject=subject,
                full_checkpoint=full_checkpoint,
                device=device,
            )
            cross_left_fit = _fit_subset(
                model=cross_left,
                indices=train_left,
                groups=groups,
                counts=event_counts,
                train_axis=True,
                seed=args.seed + 5_000_001,
                optimizer_cfg=cfg["optimizer"],
                epochs=int(cfg["optimizer"]["max_epochs"]),
                batch_size=args.batch_size,
                log_path=run_root / "cross_train_left_epochs.jsonl",
            )
            cross_right_fit = _fit_subset(
                model=cross_right,
                indices=train_right,
                groups=groups,
                counts=event_counts,
                train_axis=True,
                seed=args.seed + 5_000_003,
                optimizer_cfg=cfg["optimizer"],
                epochs=int(cfg["optimizer"]["max_epochs"]),
                batch_size=args.batch_size,
                log_path=run_root / "cross_train_right_epochs.jsonl",
            )
            cross_left_to_right = _event_next_nll(
                model=cross_left,
                groups=groups,
                counts=event_counts,
                indices=heldout_right,
                batch_size=args.batch_size,
            )
            cross_right_to_left = _event_next_nll(
                model=cross_right,
                groups=groups,
                counts=event_counts,
                indices=heldout_left,
                batch_size=args.batch_size,
            )
            shared_pooled = np.concatenate([full_left, full_right])
            iso_pooled = np.concatenate([iso_left, iso_right])
            two_pooled = np.concatenate([two_left_nll, two_right_nll])
            shared_nll = _metric(shared_pooled)
            iso_nll = _metric(iso_pooled)
            two_nll = _metric(two_pooled)
            delta_two = shared_nll - two_nll
            delta_axis = iso_nll - shared_nll
            margin = float(cfg["statistics"]["two_w_margin_fraction"])
            result = {
                "status": "complete",
                "subject": args.subject,
                "seed": args.seed,
                "source_side_counts": counts_by_side,
                "q25": partition.q25,
                "q75": partition.q75,
                "shared_left_next_nll": _metric(full_left),
                "isotropic_left_next_nll": _metric(iso_left),
                "left_axis_benefit": _metric(iso_left) - _metric(full_left),
                "shared_right_next_nll": _metric(full_right),
                "isotropic_right_next_nll": _metric(iso_right),
                "right_axis_benefit": _metric(iso_right) - _metric(full_right),
                "shared_pooled_next_nll": shared_nll,
                "isotropic_pooled_next_nll": iso_nll,
                "twoW_pooled_next_nll": two_nll,
                "delta_two": delta_two,
                "delta_axis": delta_axis,
                "twoW_margin_fraction": margin,
                "M": delta_two - margin * delta_axis,
                "cross_train_left_test_right_next_nll": _metric(
                    cross_left_to_right
                ),
                "cross_left_to_right_isotropic_benefit": (
                    _metric(iso_right) - _metric(cross_left_to_right)
                ),
                "cross_train_right_test_left_next_nll": _metric(
                    cross_right_to_left
                ),
                "cross_right_to_left_isotropic_benefit": (
                    _metric(iso_left) - _metric(cross_right_to_left)
                ),
                "fits": {
                    "twoW_left": two_left_fit,
                    "twoW_right": two_right_fit,
                    "cross_train_left": cross_left_fit,
                    "cross_train_right": cross_right_fit,
                },
                "node_bias_sha256": subject["bias_sha256"],
                "target_values_read": False,
                "resource": {
                    "peak_rss_gb": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / (1024.0**2)
                    ),
                    "peak_cuda_reserved_gb": (
                        torch.cuda.max_memory_reserved(device) / (1024.0**3)
                        if device.type == "cuda"
                        else 0.0
                    ),
                },
            }
            for name, model in (
                ("twoW_left", two_left),
                ("twoW_right", two_right),
                ("cross_train_left", cross_left),
                ("cross_train_right", cross_right),
            ):
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "subject": args.subject,
                        "seed": args.seed,
                        "variant": name,
                    },
                    run_root / f"{name}_model.pt",
                )
        atomic_json(run_root / "metrics.json", result)
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "COMPLETE",
                "analysis_status": result["status"],
                "subject": args.subject,
                "seed": args.seed,
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        (run_root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "counts": counts_by_side}, indent=2))
    except Exception as exc:
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "FAILED",
                "subject": args.subject,
                "seed": args.seed,
                "error": repr(exc),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()

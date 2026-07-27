#!/usr/bin/env python3
"""Formal LOSO Claim-2 fit: symmetric-axis full versus local-isotropic.

For each fold, shared dynamics are fitted only on the other 21 formal
patients' train80 events.  Shared parameters are then frozen, and only the
heldout patient's allowed scaffold parameters are fitted on its train80.
Heldout20 is evaluated once and never enters optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
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

from scripts.train_topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    batch_event_losses,
    evaluate_partition,
)
from src.topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    SymmetricAxisPropagationStateRNN,
    canonicalize_axis,
    estimate_node_hazard_bias,
    node_bias_fingerprint,
)


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
    torch.use_deterministic_algorithms(True, warn_only=False)


def load_subject(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        counts = np.asarray(data["event_group_count"], dtype=np.int64)
        split = np.asarray(data["event_split"], dtype=np.uint8)
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
        names = [str(value) for value in data["contact_names"]]
        times = np.asarray(data["event_abs_time"], dtype=np.float64)
    if groups.ndim != 2 or counts.shape != (groups.shape[0],):
        raise ValueError(f"{npz_path}: invalid event schema")
    if split.shape != (groups.shape[0],) or set(np.unique(split)) != {0, 1}:
        raise ValueError(f"{npz_path}: formal split must contain train80 and heldout20")
    if coords.shape != (groups.shape[1], 3) or not np.all(np.isfinite(coords)):
        raise ValueError(f"{npz_path}: physical-axis fold requires complete geometry")
    if not np.all(np.diff(times) >= 0):
        raise ValueError(f"{npz_path}: events are not chronological")
    train = np.flatnonzero(split == 0)
    heldout = np.flatnonzero(split == 1)
    bias = estimate_node_hazard_bias(groups[train])["bias"]
    return {
        "groups": groups,
        "counts": counts,
        "coords": coords,
        "names": names,
        "train": train,
        "heldout": heldout,
        "bias": bias,
        "bias_sha256": node_bias_fingerprint(bias),
        "input_sha256": sha256(npz_path),
    }


class SharedDynamics(nn.Module):
    """Exactly the five shared raw parameters allowed by v2.2."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_anisotropy = nn.Parameter(torch.tensor(0.0))
        self.raw_rho = nn.Parameter(torch.tensor(0.0))
        self.c0 = nn.Parameter(torch.tensor(-1.0))
        self.raw_c_p = nn.Parameter(torch.tensor(0.0))
        self.raw_c_n = nn.Parameter(torch.tensor(0.0))


def make_model(
    *,
    subject: dict[str, Any],
    shared: SharedDynamics,
    isotropic: bool,
    device: torch.device,
) -> SymmetricAxisPropagationStateRNN:
    model = SymmetricAxisPropagationStateRNN(
        coords=subject["coords"],
        node_bias=subject["bias"],
        shared_raw_anisotropy=shared.raw_anisotropy,
        shared_raw_rho=shared.raw_rho,
        shared_c0=shared.c0,
        shared_raw_c_p=shared.raw_c_p,
        shared_raw_c_n=shared.raw_c_n,
        isotropic=isotropic,
    )
    return model.to(device)


def _patient_parameters(
    model: SymmetricAxisPropagationStateRNN,
) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name in {"axis_raw", "gamma_raw", "gain_raw"} and parameter.requires_grad
    ]


def fit_shared(
    *,
    subjects: dict[str, dict[str, Any]],
    training_subjects: list[str],
    isotropic: bool,
    seed: int,
    device: torch.device,
    optimizer_cfg: dict[str, Any],
    h_train: int,
    epochs: int,
    events_per_patient: int,
    log_path: Path,
) -> tuple[SharedDynamics, dict[str, SymmetricAxisPropagationStateRNN], dict[str, Any]]:
    set_determinism(seed)
    shared = SharedDynamics().to(device)
    models = {
        subject: make_model(
            subject=subjects[subject],
            shared=shared,
            isotropic=isotropic,
            device=device,
        )
        for subject in training_subjects
    }
    parameters: list[nn.Parameter] = list(shared.parameters())
    for model in models.values():
        parameters.extend(_patient_parameters(model))
    # Remove the unused anisotropy parameter from the isotropic optimizer.
    parameters = [
        parameter
        for parameter in parameters
        if parameter.requires_grad
        and not (isotropic and parameter is shared.raw_anisotropy)
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    clip = float(optimizer_cfg["gradient_clip"])
    generators = {
        subject: np.random.default_rng(seed + 1009 * index)
        for index, subject in enumerate(training_subjects)
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    started = time.time()
    last_gradient_norm = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        patient_losses = []
        for subject in training_subjects:
            record = subjects[subject]
            train = record["train"]
            size = min(events_per_patient, len(train))
            sample = generators[subject].choice(train, size=size, replace=False)
            groups = torch.as_tensor(
                record["groups"][sample], dtype=torch.long, device=device
            )
            counts = torch.as_tensor(
                record["counts"][sample], dtype=torch.long, device=device
            )
            loss = batch_event_losses(
                model=models[subject],
                groups=groups,
                counts=counts,
                training_horizon=h_train,
                evaluate_full_future=False,
            )["event_objective"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{subject}: non-finite shared loss")
            (loss / len(training_subjects)).backward()
            patient_losses.append(float(loss.detach().cpu()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
        last_gradient_norm = float(gradient_norm.detach().cpu())
        optimizer.step()
        if epoch % 10 == 0 or epoch + 1 == epochs:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "variant": "local_isotropic" if isotropic else "full",
                            "patient_mean_training_objective": float(
                                np.mean(patient_losses)
                            ),
                            "patient_min_training_objective": float(
                                np.min(patient_losses)
                            ),
                            "patient_max_training_objective": float(
                                np.max(patient_losses)
                            ),
                            "gradient_norm": last_gradient_norm,
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                    + "\n"
                )
    summary = {
        "training_subjects": training_subjects,
        "n_training_subjects": len(training_subjects),
        "epochs": epochs,
        "events_per_patient_per_epoch": events_per_patient,
        "last_gradient_norm": last_gradient_norm,
        "runtime_seconds": time.time() - started,
    }
    return shared, models, summary


def _freeze_shared(shared: SharedDynamics) -> None:
    for parameter in shared.parameters():
        parameter.requires_grad_(False)


def fit_heldout_patient(
    *,
    subject: dict[str, Any],
    shared: SharedDynamics,
    isotropic: bool,
    seed: int,
    device: torch.device,
    optimizer_cfg: dict[str, Any],
    h_train: int,
    epochs: int,
    batch_size: int,
    log_path: Path,
) -> tuple[SymmetricAxisPropagationStateRNN, dict[str, Any]]:
    # Shared values have already been fitted without the heldout patient.
    _freeze_shared(shared)
    set_determinism(seed + 1_000_003)
    model = make_model(
        subject=subject, shared=shared, isotropic=isotropic, device=device
    )
    parameters = _patient_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    clip = float(optimizer_cfg["gradient_clip"])
    generator = np.random.default_rng(seed + 2_000_003)
    train = subject["train"]
    log_path.write_text("", encoding="utf-8")
    started = time.time()
    for epoch in range(epochs):
        order = generator.permutation(train)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            groups = torch.as_tensor(
                subject["groups"][indices], dtype=torch.long, device=device
            )
            counts = torch.as_tensor(
                subject["counts"][indices], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = batch_event_losses(
                model=model,
                groups=groups,
                counts=counts,
                training_horizon=h_train,
                evaluate_full_future=False,
            )["event_objective"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite heldout-train loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch % 10 == 0 or epoch + 1 == epochs:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "variant": "local_isotropic" if isotropic else "full",
                            "training_objective": float(np.mean(epoch_losses)),
                            "gradient_norm_last_batch": float(
                                gradient_norm.detach().cpu()
                            ),
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                    + "\n"
                )
    all_groups = torch.as_tensor(
        subject["groups"], dtype=torch.long, device=device
    )
    all_counts = torch.as_tensor(
        subject["counts"], dtype=torch.long, device=device
    )
    metrics = {
        "train80": evaluate_partition(
            model=model,
            groups=all_groups,
            counts=all_counts,
            indices=subject["train"],
            batch_size=max(batch_size, 4096),
        ),
        "heldout20": evaluate_partition(
            model=model,
            groups=all_groups,
            counts=all_counts,
            indices=subject["heldout"],
            batch_size=max(batch_size, 4096),
        ),
    }
    parameters_out = {
        "axis": canonicalize_axis(model.axis.detach().cpu().numpy()).tolist(),
        "gamma": float(model.gamma.detach().cpu()),
        "gain": float(model.gain.detach().cpu()),
        "anisotropy_ratio": float(model.anisotropy_ratio.detach().cpu()),
        "rho_p": float(model.rho_p.detach().cpu()),
        "c0": float(model.c0.detach().cpu()),
        "c_p": float(model.c_p.detach().cpu()),
        "c_n": float(model.c_n.detach().cpu()),
    }
    return model, {
        "epochs": epochs,
        "runtime_seconds": time.time() - started,
        "metrics": metrics,
        "parameters": parameters_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--events-per-patient", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    development_lock = (
        ROOT / cfg["outputs"]["root"] / "development/DEVELOPMENT_LOCK.json"
    )
    lock = json.loads(development_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "pass":
        raise SystemExit("development is not locked PASS")
    h_train = int(lock["H_train"])
    if h_train != 3 or lock["selected_objective"] != "next_plus_rollout_h3":
        raise SystemExit("development objective does not match frozen H3 contract")
    cohort_path = (
        ROOT
        / cfg["outputs"]["root"]
        / "input_audit/physical_axis_formal_cohort.json"
    )
    formal_subjects = list(
        map(
            str,
            json.loads(cohort_path.read_text(encoding="utf-8"))["subjects"],
        )
    )
    if len(formal_subjects) != 22 or args.heldout_subject not in formal_subjects:
        raise SystemExit("formal physical-axis cohort/fold mismatch")
    if args.seed not in list(map(int, cfg["optimizer"]["seeds"])):
        raise SystemExit("seed outside frozen set")
    training_subjects = [
        subject for subject in formal_subjects if subject != args.heldout_subject
    ]
    if len(training_subjects) != 21:
        raise AssertionError("LOSO training set must contain 21 patients")

    device_name = args.device or cfg["resources"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)
    set_determinism(args.seed)
    dataset = ROOT / cfg["inputs"]["rank_dataset"] / "per_subject"
    subjects = {
        subject: load_subject(dataset / f"{subject}.npz")
        for subject in formal_subjects
    }
    mode = "claim2_smoke" if args.smoke else "claim2_runs"
    run_root = (
        ROOT
        / cfg["outputs"]["root"]
        / "formal"
        / mode
        / args.heldout_subject
        / f"seed_{args.seed}"
    )
    if (run_root / "COMPLETE").exists() and not args.force:
        print(f"already complete: {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    epochs = 2 if args.smoke else int(cfg["optimizer"]["max_epochs"])
    events_per_patient = min(args.events_per_patient, 64) if args.smoke else args.events_per_patient
    atomic_json(
        run_root / "run_state.json",
        {
            "status": "RUNNING",
            "heldout_subject": args.heldout_subject,
            "seed": args.seed,
            "shared_training_subjects": training_subjects,
            "target_values_read": False,
            "started_unix": time.time(),
        },
    )
    resolved = {
        "heldout_subject": args.heldout_subject,
        "shared_training_subjects": training_subjects,
        "n_shared_training_subjects": len(training_subjects),
        "seed": args.seed,
        "selected_objective": lock["selected_objective"],
        "H_train": h_train,
        "epochs": epochs,
        "events_per_patient_per_epoch": events_per_patient,
        "batch_size": args.batch_size,
        "device": str(device),
        "node_bias_sha256": {
            subject: subjects[subject]["bias_sha256"]
            for subject in formal_subjects
        },
        "input_sha256": {
            subject: subjects[subject]["input_sha256"]
            for subject in formal_subjects
        },
        "config_sha256": sha256(config_path),
        "development_lock_sha256": sha256(development_lock),
        "core_sha256": sha256(
            ROOT / "src/topic5_symmetric_axis_propagation_state_v2_2.py"
        ),
        "development_trainer_sha256": sha256(
            ROOT / "scripts/train_topic5_symmetric_axis_propagation_state_v2_2.py"
        ),
        "formal_trainer_sha256": sha256(Path(__file__)),
        "git_commit": git_commit(),
        "target_values_read": False,
    }
    atomic_json(run_root / "resolved_config.json", resolved)

    summaries: dict[str, Any] = {}
    try:
        for variant, isotropic in (("full", False), ("local_isotropic", True)):
            shared, _, shared_summary = fit_shared(
                subjects=subjects,
                training_subjects=training_subjects,
                isotropic=isotropic,
                seed=args.seed,
                device=device,
                optimizer_cfg=cfg["optimizer"],
                h_train=h_train,
                epochs=epochs,
                events_per_patient=events_per_patient,
                log_path=run_root / f"{variant}_shared_epochs.jsonl",
            )
            model, heldout_summary = fit_heldout_patient(
                subject=subjects[args.heldout_subject],
                shared=shared,
                isotropic=isotropic,
                seed=args.seed,
                device=device,
                optimizer_cfg=cfg["optimizer"],
                h_train=h_train,
                epochs=epochs,
                batch_size=args.batch_size,
                log_path=run_root / f"{variant}_heldout_fit_epochs.jsonl",
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "heldout_subject": args.heldout_subject,
                    "seed": args.seed,
                    "shared_training_subjects": training_subjects,
                },
                run_root / f"{variant}_heldout_model.pt",
            )
            summaries[variant] = {
                "shared_fit": shared_summary,
                "heldout_fit": heldout_summary,
            }
        full = summaries["full"]["heldout_fit"]["metrics"]["heldout20"]
        isotropic = summaries["local_isotropic"]["heldout_fit"]["metrics"]["heldout20"]
        result = {
            "contract": cfg["contract"]["name"],
            "version": cfg["contract"]["version"],
            "status": "complete",
            "smoke": args.smoke,
            "heldout_subject": args.heldout_subject,
            "seed": args.seed,
            "shared_training_subjects": training_subjects,
            "models": summaries,
            "heldout20_comparison": {
                "next_benefit": isotropic["next_nll"] - full["next_nll"],
                "future_benefit": isotropic["future_nll"] - full["future_nll"],
            },
            "node_bias_sha256": subjects[args.heldout_subject]["bias_sha256"],
            "full_control_bias_identical": True,
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
            "target_values_read": False,
        }
        atomic_json(run_root / "metrics.json", result)
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "COMPLETE",
                "heldout_subject": args.heldout_subject,
                "seed": args.seed,
                "finished_unix": time.time(),
                "target_values_read": False,
            },
        )
        (run_root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        print(json.dumps(result["heldout20_comparison"], indent=2))
        print(json.dumps(result["resource"], indent=2))
    except Exception as exc:
        atomic_json(
            run_root / "run_state.json",
            {
                "status": "FAILED",
                "heldout_subject": args.heldout_subject,
                "seed": args.seed,
                "error": repr(exc),
                "finished_unix": time.time(),
                "target_values_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()

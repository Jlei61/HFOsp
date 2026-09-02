#!/usr/bin/env python3
"""Run, evaluate, assay and export the v0.3.2 residual-state model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.v032_model.config import load_config  # noqa: E402
from src.topic5_group_event_state.v032_model.data import load_subject_bundle  # noqa: E402
from src.topic5_group_event_state.v032_model.evaluate import evaluate_arms  # noqa: E402
from src.topic5_group_event_state.v032_model.paths import (  # noqa: E402
    FROZEN_SUBJECTS, MODEL_ROOT, SEEDS, atomic_write_json,
)
from src.topic5_group_event_state.v032_model.registry import (  # noqa: E402
    export_checkpoint_trajectory, write_frozen_registry,
)
from src.topic5_group_event_state.v032_model.synthetic import (  # noqa: E402
    judge_synthetic, run_synthetic_assay,
)
from src.topic5_group_event_state.v032_model.trainer import (  # noqa: E402
    bundle_tensors, load_checkpoint_model, train_residual_model,
)


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _run_dir(architecture: str, subject: str, seed: int, arm: str) -> Path:
    suffix = "" if arm == "learned" else f"_{arm}"
    return MODEL_ROOT / "runs" / architecture / subject / f"seed_{seed}{suffix}"


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, architecture=args.architecture)
    bundle = load_subject_bundle(args.subject, allow_provisional_h=False)
    report = train_residual_model(
        bundle, cfg, args.seed, device=_device(args.device),
        out_dir=_run_dir(args.architecture, args.subject, args.seed, args.arm),
        arm=args.arm, overwrite=args.force,
    )
    print(json.dumps({k: report.get(k) for k in ("subject", "seed", "architecture", "arm", "status",
                                                  "selected_step", "n_steps_run", "elapsed_seconds",
                                                  "best_validation")}, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, architecture=args.architecture)
    bundle = load_subject_bundle(args.subject, allow_provisional_h=False)
    device = _device(args.device)
    learned_dir = _run_dir(args.architecture, args.subject, args.seed, "learned")
    result = json.loads((learned_dir / "result.json").read_text())
    model = load_checkpoint_model(learned_dir / "checkpoint.pt", in_dim=bundle.x_std.shape[1], device=device)
    random_model = None
    random_dir = _run_dir(args.architecture, args.subject, args.seed, "random_reservoir")
    if (random_dir / "checkpoint.pt").exists():
        random_model = load_checkpoint_model(random_dir / "checkpoint.pt", in_dim=bundle.x_std.shape[1], device=device)
    tensors = bundle_tensors(bundle, device)
    payload = {
        "format": "group_event_state_v0_3_2_model_evaluation",
        "subject": args.subject, "seed": args.seed, "architecture": args.architecture,
        "checkpoint": str(learned_dir / "checkpoint.pt"),
        "h_source": bundle.history.source,
        "selection_phase": "dev_val", "dev_test_used_for_selection": False,
        "phases": {
            phase: evaluate_arms(
                model, bundle, cfg, device=device, phase=phase,
                horizon=float(cfg.horizon_seconds), log_r_h=float(result["log_r_h"]),
                random_model=random_model, tensors=tensors,
            ) for phase in ("dev_val", "dev_test")
        },
    }
    atomic_write_json(learned_dir / "evaluation.json", payload)
    print(json.dumps({p: v["contrasts"] for p, v in payload["phases"].items()}, ensure_ascii=False, indent=2))


def synthetic(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, architecture=args.architecture)
    bundle = load_subject_bundle(args.subject, allow_provisional_h=False)
    out = MODEL_ROOT / "synthetic" / args.architecture / args.subject / args.kind / f"replicate_{args.replicate}"
    report = run_synthetic_assay(
        bundle, cfg, kind=args.kind, replicate=args.replicate, seed=args.seed,
        device=_device(args.device), out_dir=out, overwrite=args.force,
    )
    print(json.dumps({"status": report.get("status"), "kind": args.kind, "replicate": args.replicate,
                      "dev_test": report.get("dev_test", {}).get("contrasts")}, ensure_ascii=False, indent=2))


def judge(args: argparse.Namespace) -> None:
    root = MODEL_ROOT / "synthetic" / args.architecture / args.subject / args.kind
    assays = [json.loads(p.read_text()) for p in sorted(root.glob("replicate_*/assay.json"))]
    report = judge_synthetic(assays, args.kind)
    atomic_write_json(root / "judgement.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def export(args: argparse.Namespace) -> None:
    device = _device(args.device)
    entries = []
    for subject in args.subjects:
        for seed in args.seeds:
            checkpoint = _run_dir(args.architecture, subject, seed, "learned") / "checkpoint.pt"
            entry = export_checkpoint_trajectory(
                subject=subject, seed=seed, architecture=args.architecture,
                checkpoint=checkpoint, device=device,
            )
            entries.append((subject, seed, entry))
    registry = write_frozen_registry(entries)
    print(json.dumps({"status": registry["status"], "n_complete_entries": registry["n_complete_entries"],
                      "patients": sorted(registry["patients"])}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    sub = p.add_subparsers(dest="command", required=True)
    for name, func in (("train", train), ("evaluate", evaluate)):
        q = sub.add_parser(name)
        q.add_argument("--subject", required=True)
        q.add_argument("--seed", type=int, required=True)
        q.add_argument("--architecture", choices=("leaky_bank", "repaired_rnn"), default="leaky_bank")
        if name == "train":
            q.add_argument("--arm", choices=("learned", "random_reservoir"), default="learned")
        q.add_argument("--force", action="store_true")
        q.set_defaults(func=func)
    q = sub.add_parser("synthetic")
    q.add_argument("--subject", required=True)
    q.add_argument("--seed", type=int, required=True)
    q.add_argument("--replicate", type=int, required=True)
    q.add_argument("--kind", choices=("positive", "null"), required=True)
    q.add_argument("--architecture", choices=("leaky_bank", "repaired_rnn"), default="leaky_bank")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=synthetic)
    q = sub.add_parser("judge")
    q.add_argument("--subject", required=True)
    q.add_argument("--kind", choices=("positive", "null"), required=True)
    q.add_argument("--architecture", choices=("leaky_bank", "repaired_rnn"), default="leaky_bank")
    q.set_defaults(func=judge)
    q = sub.add_parser("export")
    q.add_argument("--subjects", nargs="+", default=list(FROZEN_SUBJECTS))
    q.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    q.add_argument("--architecture", choices=("leaky_bank", "repaired_rnn"), default="leaky_bank")
    q.set_defaults(func=export)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


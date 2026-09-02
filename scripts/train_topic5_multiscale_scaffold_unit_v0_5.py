#!/usr/bin/env python3
"""Train one frozen Topic 5.1 v0.5 target-free unit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_topic5_lbss_unit_v0_2 import DEFAULTS, train_unit  # noqa: E402


OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
ARM_MAP = {
    "L0": "L0_LOCAL_ONLY",
    "L1": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2m": "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3": "L3_LOCAL_PLUS_LEARNED_LR",
    "C-suffix": "C_L3_ORDER_SHUFFLED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_MAP), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs-freeze", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if not (OUT_ROOT / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json").exists():
        raise RuntimeError("target physical embargo is not active")
    cfg = dict(DEFAULTS)
    if args.epochs_freeze is not None:
        cfg["epochs_freeze"] = int(args.epochs_freeze)
    events_file_name = None
    fixed_added_mask_path = None
    if args.arm == "C-suffix":
        events_file_name = f"events_suffix_null_seed{args.seed}.npz"
    elif args.arm == "L2m":
        fixed_added_mask_path = (
            OUT_ROOT / "graph_controls" / args.fit_id / f"seed{args.seed}" /
            "L2M_GRAPH_CONTROL.npz"
        )
        if not fixed_added_mask_path.exists():
            raise FileNotFoundError(f"missing frozen L2m graph control: {fixed_added_mask_path}")
    internal_arm = ARM_MAP[args.arm]
    metrics = train_unit(
        args.fit_id, internal_arm, args.seed, OUT_ROOT, torch.device(args.device), cfg,
        resume=not args.no_resume, unit_root_name="formal_units",
        events_file_name=events_file_name,
        fixed_added_mask_path=fixed_added_mask_path,
        contract_label="topic5_multiscale_scaffold_unit_v0_5",
    )
    unit = OUT_ROOT / "formal_units" / args.fit_id / internal_arm / f"seed{args.seed}"
    metrics_path = unit / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    payload["v0_5_public_arm"] = args.arm
    payload["producer_hashes"]["v0_5_wrapper"] = sha256_file(Path(__file__).resolve())
    payload["target_values_read"] = False
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True))
    temporary.replace(metrics_path)
    print(json.dumps({
        "fit_id": args.fit_id, "arm": args.arm, "seed": args.seed,
        "converged": metrics["converged"], "best_epoch": metrics["best_epoch"],
        "test_contact_nll": metrics["test"]["contact_nll"],
        "target_values_read": False,
    }))


if __name__ == "__main__":
    main()

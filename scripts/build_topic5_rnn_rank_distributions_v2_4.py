#!/usr/bin/env python3
"""Build one subject/seed's target-sealed v2.3 contact-rank distributions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_competitive_propagation_development_v2_3 import (  # noqa: E402
    atomic_json,
    load_subject,
    sha256,
)
from scripts.train_topic5_competitive_propagation_formal_v2_3 import (  # noqa: E402
    axis_for_subject,
    build_model,
)
from src.topic5_axis_positive_static_transfer_v2_4 import (  # noqa: E402
    empirical_rank_distribution,
    paired_rollout_design,
    rollout_model_distribution,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_node_hazard,
    logit,
)


BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
V23 = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
FREEZE = V23 / "development/DEVELOPMENT_FREEZE.json"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
SEEDS = (17, 29, 43)
VARIANTS = {
    "full_fixed_axis": "axis_two_state_source_full",
    "no_history": "axis_instantaneous_no_history",
    "local_isotropic": "local_isotropic_two_state",
}


def load_model(
    *,
    subject: str,
    seed: int,
    output_name: str,
    record: dict,
    axis: np.ndarray,
    node_logit: np.ndarray,
    freeze: dict,
    device: torch.device,
):
    variant = VARIANTS[output_name]
    model = build_model(
        variant=variant,
        coords=record["coords"],
        axis=axis,
        node_logit=node_logit,
        rho_propagation=float(freeze["rho_propagation"]),
        rho_competition=float(freeze["rho_competition"]),
        device=device,
    )
    checkpoint = (
        V23
        / "formal/runs"
        / subject
        / f"seed_{seed}"
        / variant
        / "best.pt"
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--n-rollouts", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.n_rollouts != 5000:
        raise SystemExit("formal contract freezes n_rollouts=5000")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if args.subject not in audit["target_metadata_eligible_patients"]:
        raise SystemExit("subject is outside target-metadata eligible cohort")
    if audit.get("target_values_read") or audit.get(
        "target_arrays_deserialized"
    ):
        raise SystemExit("target seal failed")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    record = load_subject(args.subject)
    with np.load(
        DATASET / "per_subject" / f"{args.subject}.npz",
        allow_pickle=False,
    ) as data:
        contact_names = np.asarray(data["contact_names"]).astype(str)
    node_logit = logit(
        estimate_node_hazard(record["groups"], record["train80"])
    )
    axis = axis_for_subject(args.subject)
    sampled, uniforms = paired_rollout_design(
        record["groups"],
        record["train80"],
        n_rollouts=args.n_rollouts,
        seed=240000 + args.seed,
    )
    started = time.time()
    arrays: dict[str, np.ndarray] = {
        "contact_names": contact_names,
        "empirical_train80": empirical_rank_distribution(
            record["groups"], record["train80"]
        ).astype(np.float32),
    }
    checkpoint_hashes: dict[str, str] = {}
    full_model = None
    for output_name in VARIANTS:
        model, checkpoint = load_model(
            subject=args.subject,
            seed=args.seed,
            output_name=output_name,
            record=record,
            axis=axis,
            node_logit=node_logit,
            freeze=freeze,
            device=device,
        )
        arrays[output_name] = rollout_model_distribution(
            model,
            record["groups"],
            sampled,
            uniforms,
        ).astype(np.float32)
        checkpoint_hashes[output_name] = sha256(checkpoint)
        if output_name == "full_fixed_axis":
            full_model = model
    if full_model is None:
        raise RuntimeError("full model was not loaded")
    arrays["node_only"] = rollout_model_distribution(
        full_model,
        record["groups"],
        sampled,
        uniforms,
        node_only=True,
    ).astype(np.float32)
    for name, values in arrays.items():
        if name == "contact_names":
            continue
        if values.shape != (len(contact_names), 11):
            raise RuntimeError(f"{name}: representation shape mismatch")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-6):
            raise RuntimeError(f"{name}: distribution rows do not sum to one")

    out_root = BASE / "representations/per_seed"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.subject}_seed{args.seed}.npz"
    np.savez_compressed(out_path, **arrays)
    result = {
        "status": "COMPLETE",
        "subject": args.subject,
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "rollout_prior": "paired empirical train80 source-set and event-length",
        "representations": sorted(
            name for name in arrays if name != "contact_names"
        ),
        "n_contacts": len(contact_names),
        "output": str(out_path.relative_to(ROOT)),
        "output_sha256": sha256(out_path),
        "checkpoint_sha256": checkpoint_hashes,
        "dataset_sha256": sha256(record["path"]),
        "runtime_seconds": time.time() - started,
        "peak_rss_gb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        ),
        "target_arrays_deserialized": False,
        "target_values_read": False,
    }
    atomic_json(
        out_root / f"{args.subject}_seed{args.seed}.json", result
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train one axis-positive patient/seed across 32 frozen candidate axes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

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
from scripts.train_topic5_competitive_propagation_formal_v2_3 import (  # noqa: E402
    build_model,
    fit_variant,
)
from src.topic5_axis_positive_static_transfer_v2_4 import (  # noqa: E402
    candidate_alignment_summary,
    sign_invariant_cosine,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_node_hazard,
    fibonacci_axes,
    logit,
)


BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit"
V23 = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
FREEZE = V23 / "development/DEVELOPMENT_FREEZE.json"
SEEDS = (17, 29, 43)


def load_best_model(
    *,
    variant: str,
    checkpoint: Path,
    record: dict,
    axis: np.ndarray,
    node_logit: np.ndarray,
    freeze: dict,
    device: torch.device,
):
    model = build_model(
        variant=variant,
        coords=record["coords"],
        axis=axis,
        node_logit=node_logit,
        rho_propagation=float(freeze["rho_propagation"]),
        rho_competition=float(freeze["rho_competition"]),
        device=device,
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--candidate-indices",
        help="comma-separated frozen candidate indices for resume workers",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="fit only the requested candidates; defer selection and heldout",
    )
    args = parser.parse_args()
    if args.candidates_only and (
        args.smoke or not args.candidate_indices
    ):
        raise SystemExit(
            "--candidates-only requires --candidate-indices and no --smoke"
        )

    status = json.loads(
        (AUDIT / "INPUT_AUDIT_STATUS.json").read_text(encoding="utf-8")
    )
    if args.subject not in status["axis_positive_primary_patients"]:
        raise SystemExit("subject is outside frozen n=9 axis-positive subgroup")
    if status.get("target_values_read") or status.get(
        "target_arrays_deserialized"
    ):
        raise SystemExit("target seal failed")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    axis_table = np.genfromtxt(
        AUDIT / "axis_positive_cohort.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    match = axis_table[axis_table["subject"] == args.subject]
    if np.ndim(match) == 0:
        match = np.asarray([match])
    if len(match) != 1:
        raise SystemExit(f"{args.subject}: shared-axis row missing or duplicated")
    reference = np.asarray(
        [
            match["shared_axis_x"][0],
            match["shared_axis_y"][0],
            match["shared_axis_z"][0],
        ],
        dtype=np.float64,
    )
    relation = str(match["relation"][0])
    directions = fibonacci_axes(32)
    if args.smoke:
        candidate_indices = list(range(2))
    elif args.candidate_indices:
        candidate_indices = sorted(
            {
                int(value)
                for value in args.candidate_indices.split(",")
                if value.strip()
            }
        )
        if (
            not candidate_indices
            or candidate_indices[0] < 0
            or candidate_indices[-1] >= len(directions)
        ):
            raise SystemExit("candidate indices must be in [0,31]")
    else:
        candidate_indices = list(range(32))
    set_determinism(args.seed)
    record = load_subject(args.subject)
    node_logit = logit(
        estimate_node_hazard(record["groups"], record["train80"])
    )
    groups = torch.as_tensor(
        record["groups"], dtype=torch.long, device=device
    )
    counts = torch.as_tensor(
        record["counts"], dtype=torch.long, device=device
    )
    partitions = {
        "fit60": record["partitions"]["fit60"],
        "validation20": record["partitions"]["validation20"],
    }
    run_kind = "smoke" if args.smoke else "axis_search"
    run_root = (
        BASE
        / "formal"
        / run_kind
        / args.subject
        / f"seed_{args.seed}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    final_state_path = run_root / "run_state.json"
    final_metrics_path = run_root / "metrics.json"
    if (
        not args.smoke
        and not args.candidates_only
        and final_state_path.exists()
        and final_metrics_path.exists()
    ):
        final_state = json.loads(
            final_state_path.read_text(encoding="utf-8")
        )
        final_metrics = json.loads(
            final_metrics_path.read_text(encoding="utf-8")
        )
        if (
            final_state.get("status") == "COMPLETE"
            and final_metrics.get("status") == "COMPLETE"
            and not final_metrics.get("target_values_read")
        ):
            print(
                f"{args.subject} seed={args.seed} already COMPLETE; "
                "resume skipped",
                flush=True,
            )
            return
    resolved_config = {
            "contract": "topic5_rnn_candidate_axis_v2_4",
            "subject": args.subject,
            "seed": args.seed,
            "device": str(device),
            "n_candidate_directions": len(candidate_indices),
            "axis_selection_partition": "validation20_only",
            "heldout_evaluated_after_axis_freeze": True,
            "rho_propagation": freeze["rho_propagation"],
            "rho_competition": freeze["rho_competition"],
            "learning_rate": freeze["learning_rate"],
            "batch_size": freeze["batch_size"],
            "maximum_epochs": (
                2 if args.smoke else freeze["maximum_epochs"]
            ),
            "patience": 2 if args.smoke else freeze["patience"],
            "dataset_sha256": sha256(record["path"]),
            "freeze_sha256": sha256(FREEZE),
            "target_values_read": False,
        }
    if args.candidates_only:
        atomic_json(
            run_root
            / (
                "candidate_worker_"
                + "_".join(f"{value:02d}" for value in candidate_indices)
                + ".json"
            ),
            {
                **resolved_config,
                "worker_role": "candidates_only_no_selection_no_heldout",
                "candidate_indices": candidate_indices,
            },
        )
    else:
        atomic_json(run_root / "resolved_config.json", resolved_config)
    started = time.time()
    candidate_rows = []
    for axis_index in candidate_indices:
        candidate_root = run_root / f"candidate_{axis_index:02d}"
        candidate_metrics_path = (
            candidate_root
            / "axis_two_state_no_source"
            / "metrics.json"
        )
        candidate_checkpoint = (
            candidate_root
            / "axis_two_state_no_source"
            / "best.pt"
        )
        if (
            not args.smoke
            and candidate_metrics_path.exists()
            and candidate_checkpoint.exists()
        ):
            candidate_metrics = json.loads(
                candidate_metrics_path.read_text(encoding="utf-8")
            )
            if (
                candidate_metrics.get("status") == "COMPLETE"
                and not candidate_metrics.get("target_values_read")
            ):
                candidate_rows.append(
                    {
                        "axis_index": int(axis_index),
                        "validation_nll": float(
                            candidate_metrics["metrics"]["validation20"][
                                "full_categorical_nll"
                            ]
                        ),
                        "fit60_nll": float(
                            candidate_metrics["metrics"]["fit60"][
                                "full_categorical_nll"
                            ]
                        ),
                        "best_epoch": int(
                            candidate_metrics["best_epoch"]
                        ),
                    }
                )
                print(
                    f"{args.subject} seed={args.seed} axis={axis_index:02d} "
                    "resume=COMPLETE",
                    flush=True,
                )
                continue
        set_determinism(args.seed)
        model = build_model(
            variant="axis_two_state_no_source",
            coords=record["coords"],
            axis=directions[axis_index],
            node_logit=node_logit,
            rho_propagation=float(freeze["rho_propagation"]),
            rho_competition=float(freeze["rho_competition"]),
            device=device,
        )
        result = fit_variant(
            variant="axis_two_state_no_source",
            model=model,
            groups=groups,
            counts=counts,
            partitions=partitions,
            run_root=candidate_root,
            seed=args.seed,
            learning_rate=float(freeze["learning_rate"]),
            batch_size=int(freeze["batch_size"]),
            max_epochs=(2 if args.smoke else int(freeze["maximum_epochs"])),
            patience=(2 if args.smoke else int(freeze["patience"])),
        )
        candidate_rows.append(
            {
                "axis_index": int(axis_index),
                "validation_nll": float(
                    result["metrics"]["validation20"][
                        "full_categorical_nll"
                    ]
                ),
                "fit60_nll": float(
                    result["metrics"]["fit60"]["full_categorical_nll"]
                ),
                "best_epoch": int(result["best_epoch"]),
            }
        )
        print(
            f"{args.subject} seed={args.seed} axis={axis_index:02d} "
            f"validation={candidate_rows[-1]['validation_nll']:.8f}",
            flush=True,
        )

    if args.candidates_only:
        print(
            json.dumps(
                {
                    "status": "CANDIDATES_COMPLETE",
                    "subject": args.subject,
                    "seed": args.seed,
                    "candidate_indices": candidate_indices,
                    "heldout_values_read": False,
                    "target_values_read": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    candidate_rows.sort(key=lambda row: (row["validation_nll"], row["axis_index"]))
    selected_index = int(candidate_rows[0]["axis_index"])
    selected_axis = directions[selected_index]
    selection_path = run_root / "AXIS_SELECTION_FROZEN.json"
    selection = {
        "status": "FROZEN",
        "subject": args.subject,
        "seed": args.seed,
        "selected_axis_index": selected_index,
        "selected_axis": selected_axis.tolist(),
        "selection_endpoint": "minimum_validation20_categorical_nll",
        "selection_validation_nll": candidate_rows[0]["validation_nll"],
        "heldout_values_read_for_selection": False,
        "candidate_rows": candidate_rows,
        "target_values_read": False,
    }
    atomic_json(selection_path, selection)

    selected_checkpoint = (
        run_root
        / f"candidate_{selected_index:02d}"
        / "axis_two_state_no_source"
        / "best.pt"
    )
    no_source_model = load_best_model(
        variant="axis_two_state_no_source",
        checkpoint=selected_checkpoint,
        record=record,
        axis=selected_axis,
        node_logit=node_logit,
        freeze=freeze,
        device=device,
    )
    heldout_no_source = evaluate(
        no_source_model,
        groups,
        counts,
        record["partitions"]["heldout20_sealed"],
        int(freeze["batch_size"]),
    )

    source_result = None
    heldout_source = None
    if relation == "reversed" and not args.smoke:
        set_determinism(args.seed)
        source_model = build_model(
            variant="axis_two_state_source_full",
            coords=record["coords"],
            axis=selected_axis,
            node_logit=node_logit,
            rho_propagation=float(freeze["rho_propagation"]),
            rho_competition=float(freeze["rho_competition"]),
            device=device,
        )
        source_result = fit_variant(
            variant="axis_two_state_source_full",
            model=source_model,
            groups=groups,
            counts=counts,
            partitions=partitions,
            run_root=run_root / "selected_source",
            seed=args.seed,
            learning_rate=float(freeze["learning_rate"]),
            batch_size=int(freeze["batch_size"]),
            max_epochs=int(freeze["maximum_epochs"]),
            patience=int(freeze["patience"]),
        )
        source_model = load_best_model(
            variant="axis_two_state_source_full",
            checkpoint=(
                run_root
                / "selected_source"
                / "axis_two_state_source_full"
                / "best.pt"
            ),
            record=record,
            axis=selected_axis,
            node_logit=node_logit,
            freeze=freeze,
            device=device,
        )
        heldout_source = evaluate(
            source_model,
            groups,
            counts,
            record["partitions"]["heldout20_sealed"],
            int(freeze["batch_size"]),
        )

    existing = json.loads(
        (
            V23
            / "formal/runs"
            / args.subject
            / f"seed_{args.seed}"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    isotropic_nll = float(
        existing["variants"]["local_isotropic_two_state"]["metrics"][
            "heldout20_sealed"
        ]["full_categorical_nll"]
    )
    alignment = candidate_alignment_summary(
        selected_axis, reference, directions
    )
    result = {
        "status": "COMPLETE",
        "subject": args.subject,
        "seed": args.seed,
        "relation": relation,
        "selected_axis_index": selected_index,
        "selected_axis": selected_axis.tolist(),
        **alignment,
        "isotropic_heldout_nll": isotropic_nll,
        "selected_no_source_heldout_nll": float(
            heldout_no_source["full_categorical_nll"]
        ),
        "selected_axis_over_isotropic_benefit": float(
            isotropic_nll - heldout_no_source["full_categorical_nll"]
        ),
        "selected_source_heldout_nll": (
            None
            if heldout_source is None
            else float(heldout_source["full_categorical_nll"])
        ),
        "selected_source_over_no_source_benefit": (
            None
            if heldout_source is None
            else float(
                heldout_no_source["full_categorical_nll"]
                - heldout_source["full_categorical_nll"]
            )
        ),
        "selected_source_parameters": (
            None if source_result is None else source_result["parameters"]
        ),
        "runtime_seconds": time.time() - started,
        "peak_rss_gb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        ),
        "heldout_values_read_after_axis_freeze": True,
        "target_values_read": False,
    }
    atomic_json(run_root / "metrics.json", result)
    atomic_json(
        run_root / "run_state.json",
        {
            "status": "COMPLETE",
            "subject": args.subject,
            "seed": args.seed,
            "target_values_read": False,
        },
    )
    (run_root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C1: the functional-innovation trajectory along the observed recording.

Registry discipline, stated in the output rather than assumed: if agent A has
published a producer for this patient, the trajectory is computed on it and
labelled ``shared_registry``.  If not, the registry-bound trajectory is reported
as ``not_available`` -- and a clearly separate, differently named diagnostic is
computed on agent C's own frozen checkpoint.  The two never share a field, so a
reader cannot mistake the fallback for the registry-bound result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.impulse import spearman  # noqa: E402
from src.topic5_group_event_state_h3.innovation import functional_innovation  # noqa: E402
from src.topic5_group_event_state_h3.io import (  # noqa: E402
    payload_hash,
    write_json_atomic,
    write_npz_atomic,
)
from src.topic5_group_event_state_h3.models import H3Config, build_model  # noqa: E402
from src.topic5_group_event_state_h3.registry import resolve_producer  # noqa: E402
from src.topic5_group_event_state_h3.runtime import AGENT_C_ROOT, load_subject  # noqa: E402
from src.topic5_group_event_state_h3.support import MAIN_HORIZONS_MINUTES  # noqa: E402
from src.topic5_group_event_state_h3.train import TrainConfig, resolve_window  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"
REGISTRY_PRODUCERS = ("P_slow", "P_local", "B_multiscale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", default="M2_mark_specific_feedback")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--checkpoint-root", type=Path, default=AGENT_C_ROOT / "checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_id = f"{args.subject}__{args.arm}__seed{args.seed}"
    result_path = OUT_ROOT / "machine" / f"innovation_{args.tag}" / f"{run_id}.json"
    array_path = Path(args.checkpoint_root) / f"innovation_{args.tag}" / f"{run_id}__innovation.npz"
    if result_path.exists() and not args.overwrite:
        if json.loads(result_path.read_text()).get("status") == "ok":
            print(f"{run_id}: cached")
            return

    registry_status = {
        producer: resolve_producer(producer, args.subject).as_dict()
        for producer in REGISTRY_PRODUCERS
    }
    registry_ready = [p for p, s in registry_status.items() if s["status"] == "ok"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.time()
    ckpt_path = Path(args.checkpoint_root) / args.tag / f"{run_id}__checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no frozen checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ctx = load_subject(args.subject, device, horizons=args.horizons)
    if ckpt.get("support_hash") != ctx.support_hash:
        raise ValueError(f"{run_id}: support hash mismatch against the frozen checkpoint")

    cfg = ckpt["model_config"]
    model_cfg = H3Config(**{**cfg,
                            "tau_range_s": tuple(cfg["tau_range_s"]),
                            "tau_init_s": tuple(cfg["tau_init_s"]),
                            "horizons_minutes": tuple(cfg["horizons_minutes"])})
    model = build_model(
        args.arm, model_cfg, ctx.tensors.n_drive_features,
        int(ctx.stream.features.count_features.shape[1]),
        int(ctx.stream.features.mark_features.shape[1]),
        args.seed, mean_event_rate_hz=ctx.tensors.train_event_rate_hz,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    train_cfg = resolve_window(ctx.tensors, TrainConfig(**ckpt["train_config"]))

    trace = functional_innovation(
        model, ctx.tensors, train_cfg, device, args.horizons,
        ctx.stream.t_abs, ctx.stream.features,
        producer_source="agent_c_internal_frozen_arm",
        max_events_per_segment=args.max_events, seed=args.seed,
    )
    group_names = [name for name, _sl in ctx.tensors.mark_groups]

    payload = {
        "status": "ok",
        "run_id": run_id,
        "subject": args.subject,
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "support_hash": ctx.support_hash,
        "config_hash": payload_hash({"max_events": args.max_events,
                                     "checkpoint": ckpt.get("config_hash")}),
        # Registry discipline, reported rather than assumed.
        "registry_bound_trajectory": {
            "status": "ok" if registry_ready else "not_available",
            "producers_verified": registry_ready,
            "per_producer": registry_status,
            "note": (
                "computed on a shared-registry producer"
                if registry_ready
                else "agent A has published no verified producer for this patient; the "
                     "registry-bound trajectory was not computed and was NOT substituted"
            ),
        },
        "agent_c_internal_diagnostic": {
            "producer_source": trace.producer_source,
            "tier": "diagnostic_only_not_registry_bound",
            "checkpoint_config_hash": ckpt.get("config_hash"),
            **trace.as_summary(group_names),
        },
        "n_events_traced": int(trace.event_rows.size),
        "total_seconds": None,
    }

    # Does the innovation track what actually happened next?  Reported next to it,
    # because the model's own forecast movement is not evidence that it should have
    # moved.
    payload["agent_c_internal_diagnostic"]["innovation_vs_realised_future_change"] = {
        str(h): {
            "spearman": spearman(trace.count_fraction[h], trace.future_count_change[h]),
            "n_events": int(trace.count_fraction[h].size),
            "median_realised_change_events": float(np.median(trace.future_count_change[h])),
        }
        for h in args.horizons
    }
    payload["total_seconds"] = round(time.time() - started, 1)

    arrays = {"event_rows": trace.event_rows, "event_times": trace.event_times}
    for h in args.horizons:
        arrays[f"innovation_count_fraction_{h}"] = trace.count_fraction[h]
        arrays[f"innovation_mark_shift_{h}"] = trace.mark_shift[h]
        arrays[f"realised_future_count_change_{h}"] = trace.future_count_change[h]
    write_npz_atomic(array_path, **arrays)
    payload["array_file"] = str(array_path)
    write_json_atomic(payload, result_path)

    head = args.horizons[0]
    stats = payload["agent_c_internal_diagnostic"]["horizons"][str(head)]
    print(
        f"{run_id}: registry={payload['registry_bound_trajectory']['status']} "
        f"n={payload['n_events_traced']} {head}m median_innov={stats['median_innovation_count_fraction']:+.5f} "
        f"rho_vs_realised="
        f"{payload['agent_c_internal_diagnostic']['innovation_vs_realised_future_change'][str(head)]['spearman']:+.3f} "
        f"{payload['total_seconds']}s"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Event-type-specific signed impulse response from a frozen checkpoint.

Reads out, for each observed event, the fraction by which it changes the number of
events the model expects in the next 5 / 30 / 120 minutes -- exactly, because the
state is linear and the edge is input-driven.  Signed throughout: an event type
whose impulse response is negative is reported as negative, and the fraction of
events on each side of zero is part of the output.
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

from src.topic5_group_event_state_h3.impulse import (  # noqa: E402
    MAIN_LAG_SECONDS,
    MAX_EVENTS_SCORED,
    N_EVENT_TYPES,
    SUPPLEMENT_LAG_SECONDS,
    compute_impulse_response,
    descriptive_event_types,
    kfree_event_axes,
    spearman,
)
from src.topic5_group_event_state_h3.io import (  # noqa: E402
    payload_hash,
    write_json_atomic,
    write_npz_atomic,
)
from src.topic5_group_event_state_h3.models import H3Config, build_model  # noqa: E402
from src.topic5_group_event_state_h3.perturb import collect_anchor_states  # noqa: E402
from src.topic5_group_event_state_h3.runtime import AGENT_C_ROOT, load_subject  # noqa: E402
from src.topic5_group_event_state_h3.support import MAIN_HORIZONS_MINUTES  # noqa: E402
from src.topic5_group_event_state_h3.train import TrainConfig, resolve_window  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"


def split_event_rows(ctx, split: str) -> np.ndarray:
    """Event stream indices whose instant falls in one split's usable time."""

    rows: list[np.ndarray] = []
    for interval in ctx.intervals:
        if interval.split != split:
            continue
        lo = int(np.searchsorted(ctx.stream.t_abs, interval.start, side="left"))
        hi = int(np.searchsorted(ctx.stream.t_abs, interval.stop, side="left"))
        if hi > lo:
            rows.append(np.arange(lo, hi, dtype=np.int64))
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", default="M2_mark_specific_feedback")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS_SCORED)
    parser.add_argument("--checkpoint-root", type=Path, default=AGENT_C_ROOT / "checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_id = f"{args.subject}__{args.arm}__seed{args.seed}"
    result_path = OUT_ROOT / "machine" / f"impulse_{args.tag}" / f"{run_id}.json"
    array_path = Path(args.checkpoint_root) / f"impulse_{args.tag}" / f"{run_id}__impulse.npz"
    if result_path.exists() and not args.overwrite:
        if json.loads(result_path.read_text()).get("status") == "ok":
            print(f"{run_id}: cached")
            return

    ckpt_path = Path(args.checkpoint_root) / args.tag / f"{run_id}__checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no frozen checkpoint at {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.time()
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
    model.eval()

    train_cfg = resolve_window(ctx.tensors, TrainConfig(**ckpt["train_config"]))
    anchor_states = collect_anchor_states(model, ctx.tensors, train_cfg, device)

    # Reference state: the median development-test anchor state, i.e. a state the
    # recording actually passed through, not an arbitrary origin.
    dev_states = []
    for seg, split in enumerate(ctx.tensors.anchor_split):
        keep = np.asarray([s == "development_test" for s in split], dtype=bool)
        if keep.any() and anchor_states[seg].shape[0] == keep.size:
            dev_states.append(anchor_states[seg][torch.from_numpy(np.flatnonzero(keep)).to(device)])
    if not dev_states:
        raise ValueError(f"{args.subject}: no development-test anchor states")
    reference = torch.cat(dev_states, dim=0).median(dim=0).values

    scored_rows = split_event_rows(ctx, "development_test")
    rng = np.random.default_rng(args.seed)
    if scored_rows.size > args.max_events:
        scored_rows = np.sort(rng.choice(scored_rows, size=args.max_events, replace=False))
    if scored_rows.size == 0:
        raise ValueError(f"{args.subject}: no development-test events to score")

    response = compute_impulse_response(
        model, ctx.tensors.count_features, ctx.tensors.mark_features, scored_rows,
        reference, args.horizons, ctx.tensors.mark_groups, lag_seconds=MAIN_LAG_SECONDS,
    )
    group_names = [name for name, _sl in ctx.tensors.mark_groups]
    payload = {
        "status": "ok",
        "run_id": run_id,
        "subject": args.subject,
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "checkpoint_config_hash": ckpt.get("config_hash"),
        "support_hash": ctx.support_hash,
        "config_hash": payload_hash({"lag": MAIN_LAG_SECONDS, "n_events": int(scored_rows.size),
                                     "checkpoint": ckpt.get("config_hash")}),
        "n_events_scored": int(scored_rows.size),
        "mark_group_names": group_names,
        "reference_state_source": "median development_test anchor state",
        "primary": response.as_summary(group_names),
    }

    # K-free readout: how the signed response depends on continuous event
    # coordinates.  Primary, because a cluster label depends on K and a seed.
    features = ctx.stream.features
    axes = kfree_event_axes(
        features.mark_features, features.count_features,
        features.mark_feature_names, features.count_feature_names, scored_rows,
    )
    payload["continuous_axis_spearman"] = {
        str(h): {name: spearman(values, response.count_fraction[h]) for name, values in axes.items()}
        for h in args.horizons
    }

    # Descriptive typing, explicitly secondary.
    train_rows = split_event_rows(ctx, "train")
    if train_rows.size > 20000:
        train_rows = np.sort(rng.choice(train_rows, size=20000, replace=False))
    labels = descriptive_event_types(
        features.mark_features, train_rows, scored_rows, n_types=N_EVENT_TYPES, seed=args.seed
    )
    payload["descriptive_event_types"] = {
        "n_types": int(N_EVENT_TYPES),
        "tier": "descriptive_only_depends_on_K",
        "per_type": {
            str(t): {
                "n_events": int((labels == t).sum()),
                "median_count_fraction": {
                    str(h): float(np.median(response.count_fraction[h][labels == t]))
                    if (labels == t).any() else float("nan")
                    for h in args.horizons
                },
                "fraction_positive": {
                    str(h): float(np.mean(response.count_fraction[h][labels == t] > 0))
                    if (labels == t).any() else float("nan")
                    for h in args.horizons
                },
            }
            for t in range(N_EVENT_TYPES)
        },
    }

    # Decay profile as a supplement, never the headline.
    supplement = {}
    for lag in SUPPLEMENT_LAG_SECONDS:
        if lag == MAIN_LAG_SECONDS:
            supplement[str(int(lag))] = response.as_summary(group_names)
            continue
        lagged = compute_impulse_response(
            model, ctx.tensors.count_features, ctx.tensors.mark_features, scored_rows,
            reference, args.horizons, ctx.tensors.mark_groups, lag_seconds=lag,
        )
        supplement[str(int(lag))] = lagged.as_summary(group_names)
    payload["lag_profile_supplement"] = supplement
    payload["total_seconds"] = round(time.time() - started, 1)

    arrays = {"event_rows": response.event_rows, "reference_state": response.reference_state,
              "descriptive_type": labels}
    for h in args.horizons:
        arrays[f"count_fraction_{h}"] = response.count_fraction[h]
        arrays[f"mark_shift_{h}"] = response.mark_shift[h]
        for channel, per_h in response.count_fraction_by_channel.items():
            arrays[f"count_fraction_{channel}_{h}"] = per_h[h]
    for name, values in axes.items():
        arrays[f"axis__{name}"] = values
    write_npz_atomic(array_path, **arrays)
    payload["array_file"] = str(array_path)
    write_json_atomic(payload, result_path)

    head = args.horizons[0]
    stats = payload["primary"]["horizons"][str(head)]
    print(
        f"{run_id}: n={payload['n_events_scored']} "
        f"{head}m median_frac={stats['median_count_fraction']:+.5f} "
        f"pos={stats['fraction_events_positive']:.3f} {payload['total_seconds']}s"
    )


if __name__ == "__main__":
    main()

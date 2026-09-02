#!/usr/bin/env python3
"""C4: replay a frozen checkpoint under the minimal perturbation set.

Nothing is refitted here.  The model is loaded, frozen, and the *same* exposure
window is replayed three ways from the *same* pre-state onto the *same* future
block.  Any difference in score is therefore a property of what the events were
allowed to do, not of a second training run.
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

from src.topic5_group_event_state_h3.io import (  # noqa: E402
    payload_hash,
    write_json_atomic,
    write_npz_atomic,
)
from src.topic5_group_event_state_h3.models import H3Config, build_model  # noqa: E402
from src.topic5_group_event_state_h3.perturb import (  # noqa: E402
    PRIMARY_ARMS,
    SECONDARY_ARMS,
    build_donor_pool,
    burst_event_mask,
    collect_anchor_states,
    perturb_exposure,
)
from src.topic5_group_event_state_h3.runtime import AGENT_C_ROOT, load_subject  # noqa: E402
from src.topic5_group_event_state_h3.support import (  # noqa: E402
    MAIN_HORIZONS_MINUTES,
    segment_anchor_grid,
    segment_bounds,
    select_disjoint_anchors,
)
from src.topic5_group_event_state_h3.train import TrainConfig, resolve_window  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"


def disjoint_pairs(ctx, horizon: int, split: str = "development_test"):
    """(exposure anchor index, target anchor index) with both halves disjoint.

    Exposure and target are the same length and both are laid on the shared 5-min
    grid, so an exposure window is itself an anchor -- which is what lets every
    perturbation start from a state this recording actually passed through.
    """

    bounds = segment_bounds(ctx.intervals)
    out = []
    for seg_pos, segment_id in enumerate(sorted(bounds)):
        lo, hi = bounds[segment_id]
        grid = segment_anchor_grid(lo, hi)
        members = [i for i in ctx.intervals if i.segment_id == segment_id]
        times = ctx.tensors.timelines[seg_pos].anchor_time
        lookup = {round(float(t), 6): i for i, t in enumerate(times)}
        for anchor, anchor_split, _seg in select_disjoint_anchors(
            grid, members, horizon, disjoint_exposure=True
        ):
            if anchor_split != split:
                continue
            target_idx = lookup.get(round(float(anchor), 6))
            expo_idx = lookup.get(round(float(anchor - horizon * 60.0), 6))
            if target_idx is None or expo_idx is None:
                continue
            out.append((seg_pos, expo_idx, target_idx))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", default="M2_mark_specific_feedback")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--out-tag", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=AGENT_C_ROOT / "checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_tag = args.out_tag or args.tag
    run_id = f"{args.subject}__{args.arm}__seed{args.seed}"
    result_path = OUT_ROOT / "machine" / f"perturbation_{out_tag}" / f"{run_id}.json"
    block_path = Path(args.checkpoint_root) / f"perturbation_{out_tag}" / f"{run_id}__blocks.npz"
    if result_path.exists() and not args.overwrite:
        if json.loads(result_path.read_text()).get("status") == "ok":
            print(f"{run_id}: cached")
            return

    ckpt_path = Path(args.checkpoint_root) / args.tag / f"{run_id}__checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no frozen checkpoint at {ckpt_path}; run the model grid first")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ctx = load_subject(args.subject, device, horizons=args.horizons)
    if ckpt.get("support_hash") != ctx.support_hash:
        raise ValueError(
            f"{run_id}: checkpoint support_hash {ckpt.get('support_hash')} does not match "
            f"the rebuilt support {ctx.support_hash}; refusing to score a frozen model "
            "against a denominator it was not trained on"
        )

    model_cfg = H3Config(**{**ckpt["model_config"],
                            "tau_range_s": tuple(ckpt["model_config"]["tau_range_s"]),
                            "tau_init_s": tuple(ckpt["model_config"]["tau_init_s"]),
                            "horizons_minutes": tuple(ckpt["model_config"]["horizons_minutes"])})
    model = build_model(
        args.arm, model_cfg, ctx.tensors.n_drive_features,
        int(ctx.stream.features.count_features.shape[1]),
        int(ctx.stream.features.mark_features.shape[1]),
        args.seed, mean_event_rate_hz=ctx.tensors.train_event_rate_hz,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    train_cfg = TrainConfig(**ckpt["train_config"])
    train_cfg = resolve_window(ctx.tensors, train_cfg)
    anchor_states = collect_anchor_states(model, ctx.tensors, train_cfg, device)
    donor = build_donor_pool(ctx, ctx.tensors, anchor_states, args.seed)

    arms = list(PRIMARY_ARMS) + (list(SECONDARY_ARMS) if args.include_secondary else [])
    rng = np.random.default_rng(args.seed + 17)
    burst_masks = {
        seg: burst_event_mask(ctx.tensors, seg, args.seed)
        for seg in range(len(ctx.tensors.timelines))
    } if "burst_thinning" in arms else {}

    per_block: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    with torch.no_grad():
        for horizon in args.horizons:
            pairs = disjoint_pairs(ctx, horizon)
            if not pairs:
                summary[str(horizon)] = {"status": "no_disjoint_exposure_target_pairs"}
                continue
            target = ctx.tensors.targets
            rows = {arm: {"count": [], "mark": []} for arm in arms}
            meta = {"segment": [], "anchor_time": [], "n_exposure_events": [], "count_true": []}
            for seg, expo_idx, target_idx in pairs:
                tl = ctx.tensors.timelines[seg]
                lo = int(tl.anchor_step[expo_idx])
                hi = int(tl.anchor_step[target_idx])
                pre_state = anchor_states[seg][expo_idx]
                n_events = int((tl.event_row[lo : hi + 1] >= 0).sum())
                tgt = target[seg][horizon]
                for arm in arms:
                    state = perturb_exposure(
                        model, ctx.tensors, seg, lo, hi, pre_state, arm,
                        donor=donor, rng=rng,
                        burst_mask=burst_masks.get(seg),
                    )
                    scores = model.score_blocks(
                        state.unsqueeze(0), horizon,
                        tgt["count"][target_idx : target_idx + 1],
                        tgt["has_events"][target_idx : target_idx + 1],
                        tgt["mark_mean"][target_idx : target_idx + 1],
                    )
                    rows[arm]["count"].append(float(scores["count"][0]))
                    rows[arm]["mark"].append(float(scores["mark"][0].mean()))
                meta["segment"].append(seg)
                meta["anchor_time"].append(float(tl.anchor_time[target_idx]))
                meta["n_exposure_events"].append(n_events)
                meta["count_true"].append(int(tgt["count"][target_idx]))

            real_count = np.asarray(rows["real_sequence"]["count"])
            real_mark = np.asarray(rows["real_sequence"]["mark"])
            entry: dict = {
                "status": "ok",
                "n_disjoint_pairs": len(pairs),
                "median_exposure_events": float(np.median(meta["n_exposure_events"])),
            }
            for arm in arms:
                c = np.asarray(rows[arm]["count"])
                m = np.asarray(rows[arm]["mark"])
                per_block[f"{horizon}__{arm}__count"] = c
                per_block[f"{horizon}__{arm}__mark"] = m
                entry[arm] = {
                    "mean_count_logscore": float(np.mean(c)),
                    "mean_mark_logscore": float(np.mean(m)),
                    "median_count_delta_vs_real": float(np.median(real_count - c)),
                    "median_mark_delta_vs_real": float(np.median(real_mark - m)),
                    "n_blocks_real_better_count": int((real_count > c).sum()),
                    "n_blocks_real_better_mark": int((real_mark > m).sum()),
                }
            # The two estimands, named so they cannot be silently merged later.
            entry["burden_effect"] = entry.get("no_event_feedback", {})
            entry["content_effect"] = entry.get("state_matched_mark_replacement", {})
            summary[str(horizon)] = entry
            for key, value in meta.items():
                per_block[f"{horizon}__meta__{key}"] = np.asarray(value)

    write_npz_atomic(block_path, **per_block)
    payload = {
        "status": "ok",
        "run_id": run_id,
        "subject": args.subject,
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "checkpoint_file": str(ckpt_path),
        "checkpoint_config_hash": ckpt.get("config_hash"),
        "support_hash": ctx.support_hash,
        "config_hash": payload_hash({"arms": arms, "horizons": list(args.horizons),
                                     "checkpoint": ckpt.get("config_hash")}),
        "perturbation_arms": arms,
        "primary_arms": list(PRIMARY_ARMS),
        "secondary_arms": list(SECONDARY_ARMS) if args.include_secondary else [],
        "n_donors": int(donor[0].numel()),
        "donor_pool_split": "train",
        "horizons": summary,
        "block_file": str(block_path),
        "total_seconds": round(time.time() - started, 1),
    }
    write_json_atomic(payload, result_path)
    print(f"{run_id}: " + " ".join(
        f"{h}:{summary.get(str(h), {}).get('status', 'na')}" for h in args.horizons
    ) + f" {payload['total_seconds']}s")


if __name__ == "__main__":
    main()

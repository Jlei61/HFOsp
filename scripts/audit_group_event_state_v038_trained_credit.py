#!/usr/bin/env python3
"""Measure how far trained H1 losses send credit into real event history.

This is deliberately different from the v0.3.7 architecture sanity check.
It loads the selected human checkpoint and its fitted endpoint readout, then
back-propagates held-out endpoint losses to the actual historical burden and
grammar marks.  A fixed tau bank can carry a gradient by construction; this
audit asks whether the *trained selected readout* actually uses that route.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import (
    DualStreamEventCTSSM,
    dual_stream_features_at_queries,
)
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_train import (
    H1TrainConfig,
    NestedH1Readout,
    _endpoint_losses,
    _target_bundle,
)


BINS_HOURS = (0.0, 0.5, 2.0, 6.0, 8.0, 16.0, 32.0, float("inf"))


def _bin_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}h" if np.isfinite(hi) else f">={lo:g}h"


def _one(subject: str, seed: int, h1_root: Path, device: torch.device,
         maximum_anchors: int) -> dict:
    directory = h1_root / "event" / subject / f"seed{seed}"
    card = json.loads((directory / "card.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    config_names = {field.name for field in fields(H1TrainConfig)}
    config = H1TrainConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in config_names
    })
    horizons = tuple(float(value) for value in card["horizons_seconds"])
    data = build_h1_subject_data(subject, seed=seed, horizons_seconds=horizons)
    observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1],
        taus_seconds=config.taus_seconds,
        burden_channels_per_tau=config.burden_channels_per_tau,
        grammar_channels_per_tau=config.grammar_channels_per_tau,
    ).to(device)
    observer.load_state_dict(checkpoint["observer"])
    widths = {key: int(value) for key, value in checkpoint["widths"].items()}
    with np.load(directory / "trajectory_and_targets.npz", allow_pickle=False) as stored:
        bmark_np = np.asarray(stored["fixed_mark_state"], dtype=np.float32)
        learned_np = np.asarray(stored["learned_state"], dtype=np.float32)
    q_np = np.clip(
        (data.rate.q_raw - np.asarray(checkpoint["q_centre"])) / np.asarray(checkpoint["q_scale"]),
        -12.0, 12.0,
    ).astype(np.float32)
    bmark_burden_dim = len(config.taus_seconds) * data.burden_mark.shape[1]
    state_burden_dim = len(config.taus_seconds) * config.burden_channels_per_tau
    readout = NestedH1Readout(
        q_np.shape[1], bmark_burden_dim, bmark_np.shape[1] - bmark_burden_dim,
        state_burden_dim, observer.state_dim - state_burden_dim,
        widths, len(horizons),
        state_readout_init_std=config.state_readout_init_std,
        bmark_readout_init_std=config.bmark_readout_init_std,
    ).to(device)
    readout.load_state_dict(checkpoint["readout"])
    observer.eval(); readout.eval()
    target, valid, _ = _target_bundle(data, device)
    exposure = torch.as_tensor(data.rate.target_exposure_seconds, dtype=torch.float32, device=device)
    q = torch.as_tensor(q_np, device=device)
    bmark = torch.as_tensor(bmark_np, device=device)
    horizon_index = len(horizons) - 1
    selection = np.flatnonzero(
        (data.rate.phase == "SELECTION") & data.rate.target_valid[:, horizon_index]
    )
    eligible = []
    for anchor in selection:
        event_rows = np.flatnonzero(
            (data.event_segment == data.rate.segment[anchor])
            & (data.event_time < data.rate.anchor_time[anchor])
        )
        if event_rows.size and data.rate.anchor_time[anchor] - data.event_time[event_rows[0]] >= 8.0 * 3600.0:
            eligible.append(int(anchor))
    if eligible:
        positions = np.linspace(0, len(eligible) - 1, min(maximum_anchors, len(eligible)), dtype=int)
        anchors = [eligible[index] for index in np.unique(positions)]
    else:
        anchors = []
    endpoint_bin_total = {
        endpoint: {_bin_name(BINS_HOURS[i], BINS_HOURS[i + 1]): 0.0
                   for i in range(len(BINS_HOURS) - 1)}
        for endpoint in NestedH1Readout.ENDPOINTS
    }
    endpoint_events = {endpoint: {name: 0 for name in endpoint_bin_total[endpoint]}
                       for endpoint in NestedH1Readout.ENDPOINTS}
    used_anchors = 0
    state_parity_max_abs = 0.0
    maximum_history_age_hours = 0.0
    for anchor in anchors:
        query_time = float(data.rate.anchor_time[anchor])
        # Use the complete preceding carry segment.  Truncating at 32 h is not
        # equivalent to the trained observer: a 32 h channel retains exp(-1)
        # of a 32 h-old event and still retains older events.  The final bin
        # therefore genuinely measures >=32 h history rather than being an
        # impossible structural zero.
        event_rows = np.flatnonzero(
            (data.event_segment == data.rate.segment[anchor])
            & (data.event_time < query_time)
        )
        if event_rows.size == 0:
            continue
        times = torch.as_tensor(data.event_time[event_rows], dtype=torch.float64, device=device)
        burden = torch.as_tensor(data.burden_mark[event_rows], dtype=torch.float32, device=device).requires_grad_(True)
        grammar = torch.as_tensor(data.grammar_mark[event_rows], dtype=torch.float32, device=device).requires_grad_(True)
        age_seconds = query_time - data.event_time[event_rows]
        maximum_history_age_hours = max(
            maximum_history_age_hours, float(np.max(age_seconds) / 3600.0)
        )
        observer_output = observer(times, burden, grammar, scan="associative")
        state = dual_stream_features_at_queries(
            observer_output,
            times,
            torch.as_tensor([query_time], dtype=torch.float64, device=device),
        )[0]
        parity = float(np.max(np.abs(state.detach().cpu().numpy() - learned_np[anchor])))
        state_parity_max_abs = max(state_parity_max_abs, parity)
        if parity > 1e-4:
            raise ValueError(
                f"trained state reconstruction does not match frozen trajectory: {parity}"
            )
        prediction = readout.predict(q[anchor:anchor + 1], bmark=bmark[anchor:anchor + 1], state=state[None])
        local_target = {name: value[anchor:anchor + 1] for name, value in target.items()}
        local_valid = {}
        for name, value in valid.items():
            mask = torch.zeros_like(value[anchor:anchor + 1])
            mask[:, horizon_index] = value[anchor:anchor + 1, horizon_index]
            local_valid[name] = mask
        losses = _endpoint_losses(
            prediction, local_target, local_valid, exposure[anchor:anchor + 1],
            readout.log_dispersion, torch.zeros(1, dtype=torch.long, device=device),
        )
        for endpoint, loss in losses.items():
            grad_b, grad_g = torch.autograd.grad(
                loss, (burden, grammar), retain_graph=True, allow_unused=True,
            )
            norm = torch.zeros(event_rows.size, device=device)
            if grad_b is not None: norm = norm + grad_b.square().sum(1)
            if grad_g is not None: norm = norm + grad_g.square().sum(1)
            norm = norm.sqrt().detach().cpu().numpy()
            for index in range(len(BINS_HOURS) - 1):
                lo, hi = BINS_HOURS[index], BINS_HOURS[index + 1]
                name = _bin_name(lo, hi)
                use = (age_seconds >= lo * 3600.0) & (age_seconds < hi * 3600.0)
                endpoint_bin_total[endpoint][name] += float(norm[use].sum())
                endpoint_events[endpoint][name] += int(use.sum())
        used_anchors += 1
    endpoint_summary = {}
    for endpoint, totals in endpoint_bin_total.items():
        total = sum(totals.values())
        endpoint_summary[endpoint] = {
            "gradient_mass": totals,
            "mean_gradient_per_event": {
                name: (value / endpoint_events[endpoint][name]
                       if endpoint_events[endpoint][name] > 0 else None)
                for name, value in totals.items()
            },
            "gradient_fraction": {name: (value / total if total > 0 else 0.0)
                                  for name, value in totals.items()},
            "events": endpoint_events[endpoint],
            "fraction_beyond_2h": (
                sum(value for name, value in totals.items()
                    if name in {"2-6h", "6-8h", "8-16h", "16-32h", ">=32h"}) / total
                if total > 0 else 0.0
            ),
            "fraction_beyond_6h": (
                sum(value for name, value in totals.items()
                    if name in {"6-8h", "8-16h", "16-32h", ">=32h"}) / total
                if total > 0 else 0.0
            ),
        }
    state_stage = card["stages"]["state"]
    return {
        "format": "group_event_state_v0_3_8_trained_credit_audit_v1",
        "subject": subject, "seed": seed, "source_card": str(directory / "card.json"),
        "horizon_hours": horizons[horizon_index] / 3600.0,
        "eligible_anchors": len(eligible), "audited_anchors": used_anchors,
        "maximum_audited_history_age_hours": maximum_history_age_hours,
        "endpoint_credit": endpoint_summary,
        "history_contract": "complete preceding carry segment; bins extend through >=32h",
        "frozen_trajectory_state_parity_max_abs": state_parity_max_abs,
        "selected_state_training": {
            key: state_stage.get(key) for key in (
                "selected_step", "selected_at_init", "first_step_gradient_norm",
                "peak_parameter_delta_from_stage_start", "gain_over_parent",
                "training_budget_exhausted",
            )
        },
        "interpretation_contract": (
            "measurable old-event gradient is trained observer credit, not evidence that an IED "
            "physically changed the brain; H3 remains a separate generative comparison"
        ),
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-anchors", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    atomic_json(args.out_root / "queue_status.json", {
        "status": "RUNNING", "failures": [],
        "subjects": args.subjects, "seeds": args.seeds,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    failures = []
    for subject in args.subjects:
        for seed in args.seeds:
            output = args.out_root / subject / f"seed{seed}.json"
            if output.exists():
                continue
            try:
                payload = _one(subject, seed, args.h1_root, device, args.maximum_anchors)
                atomic_json(output, payload)
            except Exception as error:
                failures.append({"subject": subject, "seed": seed, "error": repr(error)})
    atomic_json(args.out_root / "queue_status.json", {
        "status": "FAILED" if failures else "COMPLETE", "failures": failures,
        "subjects": args.subjects, "seeds": args.seeds,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Random-background capacity control for the trained dual-stream H1 model.

The primary dual card already compares the learned event observer against a
frozen random event observer.  It did not contain the symmetric control for
the persistent background observer.  This audit keeps the fitted B_rate,
B_mark and current-background predictor fixed, replaces only the persistent
background encoder by a frozen random encoder with the same state width and
physical-time bank, and fits a residual readout on FIT/INNER only.

The resulting contrast is a capacity control, not another state producer and
never participates in checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import GridBackgroundCTSSM
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_dual_train import (
    BackgroundStateComputer,
    H1DualTrainConfig,
    NestedDualReadout,
    _background_features,
)
from src.topic5_group_event_state.v037.h1_train import (
    _endpoint_losses,
    _selection_score,
    _target_bundle,
    _weighted_total,
)


def _configuration(raw: dict[str, Any]) -> H1DualTrainConfig:
    names = {field.name for field in fields(H1DualTrainConfig)}
    return H1DualTrainConfig(**{key: value for key, value in raw.items() if key in names})


def _score_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    values = [abs(float(left["total"]) - float(right["total"]))]
    for endpoint in NestedDualReadout.ENDPOINTS:
        values.append(abs(float(left["endpoints"][endpoint]) - float(right["endpoints"][endpoint])))
    return max(values)


def _parent_parity_audit(
    rebuilt: dict[str, Any],
    frozen: dict[str, Any],
    horizons: tuple[float, ...],
    *,
    ordinary_absolute_tolerance: float = 5e-4,
    ordinary_relative_tolerance: float = 2e-4,
    reconstructed_mixture_tolerance: float = 2e-3,
    reconstructed_mark_tolerance: float = 3e-3,
) -> dict[str, Any]:
    """Audit each endpoint instead of hiding one reconstructed target in a total.

    The source run did not persist either its K-means repertoire targets or its
    randomized-PCA compressed grammar targets.  A FIT-only label alignment
    recovers the repertoire convention, but very small reconstruction drift can
    remain in ``mixture`` or ``mark``.  Such drift must never silently validate
    a random-background contrast: the affected endpoint is reported unavailable
    whenever ordinary exact-parent parity is not met.  Other endpoints remain
    usable because their residual heads are disjoint.
    """

    endpoint_max_abs: dict[str, float] = {}
    endpoint_max_tolerance_ratio: dict[str, float] = {}
    for endpoint in NestedDualReadout.ENDPOINTS:
        pairs = [(float(rebuilt["endpoints"][endpoint]), float(frozen["endpoints"][endpoint]))]
        for horizon in horizons:
            key = str(int(horizon))
            pairs.append((
                float(rebuilt["by_horizon"][key]["endpoints"][endpoint]),
                float(frozen["by_horizon"][key]["endpoints"][endpoint]),
            ))
        differences = [abs(left - right) for left, right in pairs]
        tolerances = [
            float(ordinary_absolute_tolerance)
            + float(ordinary_relative_tolerance) * abs(right)
            for _left, right in pairs
        ]
        endpoint_max_abs[endpoint] = max(differences)
        endpoint_max_tolerance_ratio[endpoint] = max(
            difference / max(tolerance, 1e-12)
            for difference, tolerance in zip(differences, tolerances)
        )
    auditable = [
        endpoint for endpoint, value in endpoint_max_tolerance_ratio.items()
        if value <= 1.0
    ]
    excluded = [endpoint for endpoint in NestedDualReadout.ENDPOINTS if endpoint not in auditable]
    reconstructed_tolerances = {
        "mixture": float(reconstructed_mixture_tolerance),
        "mark": float(reconstructed_mark_tolerance),
    }
    illegal = {
        endpoint: endpoint_max_abs[endpoint]
        for endpoint in excluded
        if endpoint not in reconstructed_tolerances
        or endpoint_max_abs[endpoint] > reconstructed_tolerances[endpoint]
    }
    if illegal:
        raise ValueError(f"frozen parent endpoint parity failed: {illegal}")
    return {
        "overall_max_abs": _score_delta(rebuilt, frozen),
        "endpoint_max_abs_including_horizons": endpoint_max_abs,
        "endpoint_max_tolerance_ratio": endpoint_max_tolerance_ratio,
        "ordinary_absolute_tolerance": float(ordinary_absolute_tolerance),
        "ordinary_relative_tolerance": float(ordinary_relative_tolerance),
        "reconstructed_mixture_hard_tolerance": float(reconstructed_mixture_tolerance),
        "reconstructed_mark_hard_tolerance": float(reconstructed_mark_tolerance),
        "auditable_endpoints": auditable,
        "excluded_endpoints": excluded,
        "all_endpoint_total_auditable": not excluded,
    }


def _fit_only_repertoire_alignment(
    target: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    parent_prediction: dict[str, torch.Tensor],
    fit: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], tuple[int, ...], float]:
    """Resolve arbitrary repertoire-label permutations using FIT rows only.

    The original H1 run did not persist its fitted dictionary.  Rebuilding a
    K-means dictionary can therefore reproduce the same partition with a
    different label order.  This is a nuisance symmetry, not target drift.
    Aligning on FIT avoids looking at INNER/SELECTION and restores the source
    checkpoint's output convention before any capacity-control fitting.
    """

    width = int(target["mixture"].shape[-1])
    logp = torch.log_softmax(parent_prediction["mixture"][fit], dim=-1)
    mask = valid["mixture"][fit]
    best_loss = float("inf")
    best = tuple(range(width))
    for permutation in itertools.permutations(range(width)):
        value = target["mixture"][fit][..., list(permutation)]
        loss = -(value * logp).sum(dim=-1)
        if not bool(mask.any()):
            score = 0.0
        else:
            # Equal horizon weighting matches the H1 training contract.
            terms = [loss[:, h][mask[:, h]].mean() for h in range(mask.shape[1])
                     if bool(mask[:, h].any())]
            score = float(torch.stack(terms).mean()) if terms else 0.0
        if score < best_loss:
            best_loss = score
            best = tuple(int(value) for value in permutation)
    aligned = dict(target)
    aligned["mixture"] = target["mixture"][..., list(best)]
    return aligned, best, best_loss


def _one(subject: str, seed: int, h1_root: Path, out_root: Path,
         device: torch.device) -> dict[str, Any]:
    source = h1_root / "dual" / subject / f"seed{seed}"
    card = json.loads((source / "card.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    with np.load(source / "trajectory_and_targets.npz", allow_pickle=False) as stored:
        trajectory = {name: np.asarray(stored[name]) for name in stored.files}
    horizons = tuple(float(value) for value in card["horizons_seconds"])
    data = build_h1_subject_data(subject, seed=seed, horizons_seconds=horizons)
    if not np.array_equal(trajectory["anchor_time"], data.rate.anchor_time):
        raise ValueError("random-background audit anchor drift")
    config = _configuration(checkpoint["config"])
    fit_np, inner_np, selection_np = (
        np.flatnonzero(data.rate.phase == split) for split in ("FIT", "INNER", "SELECTION")
    )
    fit, inner, selection = (
        torch.as_tensor(rows, dtype=torch.long, device=device)
        for rows in (fit_np, inner_np, selection_np)
    )
    q_raw = np.clip(
        (data.rate.q_raw - np.asarray(checkpoint["q_centre"]))
        / np.asarray(checkpoint["q_scale"]),
        -12.0, 12.0,
    ).astype(np.float32)
    q = torch.as_tensor(q_raw, device=device)
    # Parent parity must use the exact frozen tensors saved by the source run.
    # Recomputing the fixed EWMA changes float rounding by ~1e-3, which can be
    # amplified by a large fitted readout and falsely fail (or pass) parity.
    bmark = torch.as_tensor(
        trajectory["fixed_mark_state"], dtype=torch.float32, device=device,
    )
    bg_current = torch.as_tensor(
        trajectory["background_current"], dtype=torch.float32, device=device,
    )
    bg_available = torch.as_tensor(
        trajectory["background_available"], dtype=torch.float32, device=device,
    )
    rebuilt_current, rebuilt_available, _audit, _names, centre, scale = _background_features(
        data, device
    )
    if not np.allclose(centre, np.asarray(checkpoint["background_centre"]), rtol=0.0, atol=1e-6):
        raise ValueError("background centre does not reproduce the frozen H1 checkpoint")
    if not np.allclose(scale, np.asarray(checkpoint["background_scale"]), rtol=0.0, atol=1e-6):
        raise ValueError("background scale does not reproduce the frozen H1 checkpoint")
    if not torch.allclose(rebuilt_current, bg_current, rtol=0.0, atol=1e-6):
        raise ValueError("background current differs from frozen H1 trajectory")
    if not torch.equal(rebuilt_available, bg_available):
        raise ValueError("background availability differs from frozen H1 trajectory")

    # Use a seed stream disjoint from the trained model initialization.  The
    # encoder stays frozen; only its capacity-matched residual readout learns.
    torch.manual_seed(int(seed) + 1_000_003)
    random_background = GridBackgroundCTSSM(
        bg_current.shape[1], taus_seconds=config.taus_seconds,
        channels_per_tau=config.background_channels_per_tau,
    ).to(device)
    for parameter in random_background.parameters():
        parameter.requires_grad_(False)
    computer = BackgroundStateComputer(
        data, random_background, bg_current, bg_available, device,
    )
    with torch.no_grad():
        random_state = computer()

    widths = {key: int(value) for key, value in checkpoint["widths"].items()}
    bmark_burden_dim = len(config.taus_seconds) * data.burden_mark.shape[1]
    event_burden_dim = len(config.taus_seconds) * config.burden_channels_per_tau
    readout = NestedDualReadout(
        q.shape[1], bmark_burden_dim, bmark.shape[1] - bmark_burden_dim,
        bg_current.shape[1], random_background.state_dim,
        event_burden_dim,
        len(config.taus_seconds) * (config.grammar_channels_per_tau + 1),
        widths, len(horizons),
        state_readout_init_std=config.state_readout_init_std,
        bmark_readout_init_std=config.bmark_readout_init_std,
    ).to(device)
    readout.load_state_dict(checkpoint["readout"])
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    for layer in readout.background_state.values():
        layer.weight.data.zero_()
        layer.weight.requires_grad_(True)
    parameters = list(readout.background_state.parameters())
    target, valid, _scales = _target_bundle(data, device)
    exposure = torch.as_tensor(
        data.rate.target_exposure_seconds, dtype=torch.float32, device=device,
    )

    def predict(use_random: bool) -> dict[str, torch.Tensor]:
        return readout.predict(
            q, bmark=bmark, background_current=bg_current,
            background_state=random_state if use_random else None,
        )

    with torch.no_grad():
        target, repertoire_permutation, repertoire_fit_loss = _fit_only_repertoire_alignment(
            target, valid, predict(False), fit,
        )

    with torch.no_grad():
        parent_inner = float(_weighted_total(_endpoint_losses(
            predict(False), target, valid, exposure, readout.log_dispersion, inner,
        )))
    best = parent_inner
    best_step = 0
    best_state = copy.deepcopy(readout.background_state.state_dict())
    first_gradient = None
    peak_delta = 0.0
    start = [parameter.detach().cpu().clone() for parameter in parameters]
    stale = 0
    history = [{"step": 0, "inner_loss": parent_inner}]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config.lr_background_head),
        weight_decay=float(config.weight_decay),
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
    )
    for step in range(1, int(config.max_steps_background_state) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_total(_endpoint_losses(
            predict(True), target, valid, exposure, readout.log_dispersion, fit,
        ))
        if not torch.isfinite(loss):
            raise FloatingPointError("random-background residual produced non-finite loss")
        loss.backward()
        if step == 1:
            first_gradient = float(torch.sqrt(sum(
                parameter.grad.detach().float().square().sum()
                for parameter in parameters if parameter.grad is not None
            )))
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        if step % int(config.validate_every) != 0 and step != int(config.max_steps_background_state):
            continue
        peak_delta = max(peak_delta, max(
            float((parameter.detach().cpu() - initial).abs().max())
            for parameter, initial in zip(parameters, start)
        ))
        with torch.no_grad():
            value = float(_weighted_total(_endpoint_losses(
                predict(True), target, valid, exposure, readout.log_dispersion, inner,
            )))
        history.append({"step": step, "fit_loss": float(loss.detach()), "inner_loss": value})
        if np.isfinite(value) and value < best - 1e-5:
            best = value
            best_step = step
            best_state = copy.deepcopy(readout.background_state.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience_checks):
            break
    readout.background_state.load_state_dict(best_state)
    with torch.no_grad():
        parent = _selection_score(
            predict(False), target, valid, exposure, readout.log_dispersion,
            selection, selection_np, horizons,
        )
        random_score = _selection_score(
            predict(True), target, valid, exposure, readout.log_dispersion,
            selection, selection_np, horizons,
        )
    frozen_parent = card["selection_scores"]["B_mark_current_background"]
    parity = _parent_parity_audit(parent, frozen_parent, horizons)
    auditable_endpoints = set(parity["auditable_endpoints"])
    total_auditable = bool(parity["all_endpoint_total_auditable"])
    learned = card["selection_scores"]["B_background_persistent"]
    by_horizon = {}
    for horizon in horizons:
        key = str(int(horizon))
        by_horizon[key] = {
            "random_gain_over_current_background": (
                float(parent["by_horizon"][key]["total"])
                - float(random_score["by_horizon"][key]["total"])
            ),
            "learned_gain_over_random_background": (
                float(random_score["by_horizon"][key]["total"])
                - float(learned["by_horizon"][key]["total"])
            ) if total_auditable else None,
            "endpoint_learned_gain_over_random": {
                endpoint: ((
                    float(random_score["by_horizon"][key]["endpoints"][endpoint])
                    - float(learned["by_horizon"][key]["endpoints"][endpoint])
                ) if endpoint in auditable_endpoints else None)
                for endpoint in NestedDualReadout.ENDPOINTS
            },
        }
    output = {
        "format": "group_event_state_v0_3_8_dual_random_background_control_v2",
        "subject": subject, "seed": int(seed), "source_card": str(source / "card.json"),
        "horizons_seconds": list(horizons),
        "parent_reconstruction_max_abs": parity["overall_max_abs"],
        "parent_reconstruction_audit": parity,
        "fit_only_repertoire_label_alignment": {
            "permutation": list(repertoire_permutation),
            "fit_cross_entropy": repertoire_fit_loss,
            "inner_or_selection_used": False,
        },
        "training": {
            "selected_step": best_step, "selected_at_init": best_step == 0,
            "first_step_gradient_norm": first_gradient,
            "peak_parameter_delta_from_stage_start": peak_delta,
            "steps_run": history[-1]["step"],
            "training_budget_exhausted": bool(
                history[-1]["step"] == int(config.max_steps_background_state)
                and stale < int(config.patience_checks)
            ),
            "history": history,
        },
        "scores": {
            "current_background": parent,
            "random_persistent_background": random_score,
            "learned_persistent_background": learned,
        },
        "primary_contrasts": {
            "learned_background_gain_over_random": (
                float(random_score["total"] - learned["total"])
                if total_auditable else None
            ),
            "random_background_gain_over_current": float(parent["total"] - random_score["total"]),
        },
        "by_horizon": by_horizon,
        "interpretation_contract": (
            "this is a frozen random-background capacity control; it does not select or modify "
            "the trained dual observer"
        ),
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    destination = out_root / subject / f"seed{seed}.json"
    atomic_json(destination, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    atomic_json(args.out_root / "queue_status.json", {
        "status": "RUNNING", "failures": [],
        "subjects": list(args.subjects), "seeds": list(args.seeds),
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    failures = []
    for subject in args.subjects:
        for seed in args.seeds:
            destination = args.out_root / subject / f"seed{seed}.json"
            if destination.exists():
                continue
            try:
                _one(subject, seed, args.h1_root, args.out_root, torch.device(args.device))
            except Exception as error:
                failures.append({"subject": subject, "seed": seed, "error": repr(error)})
    atomic_json(args.out_root / "queue_status.json", {
        "status": "FAILED" if failures else "COMPLETE", "failures": failures,
        "subjects": list(args.subjects), "seeds": list(args.seeds),
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

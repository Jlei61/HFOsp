"""Fixed-time multi-event open-loop evaluation for the v0.3 pilot."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.v02.subject import load_subject_timeline

from .pilot import (
    DATASET_ROOT,
    PilotConfig,
    SOURCE_COMMIT,
    _atomic_json,
    _group_count,
    _to_device,
    build_model,
)


def _poisson_nll(count: np.ndarray, mean: np.ndarray) -> np.ndarray:
    y = np.asarray(count, dtype=np.float64)
    mu = np.clip(np.asarray(mean, dtype=np.float64), 1e-8, None)
    return mu - y * np.log(mu) + np.vectorize(math.lgamma)(y + 1.0)


def _eligible_baseline_columns(names: tuple[str, ...]) -> np.ndarray:
    """H1 baseline may use past seizures, never a future seizure time."""

    # Remove the complete seizure-bookkeeping family in the pilot so its H1
    # comparison is visibly event/clock/coverage-only.  H2b will add past-seizure
    # covariates under its own contract.
    return np.array(["seizure" not in name for name in names], dtype=bool)


def _fit_count_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z_train = (x_train - mean) / scale
    z_val = (x_val - mean) / scale
    z_test = (x_test - mean) / scale
    target = np.log1p(y_train.astype(np.float64))
    target_mean = float(target.mean())
    centered = target - target_mean
    grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
    best = None
    path = []
    for ridge in grid:
        gram = z_train.T @ z_train + float(ridge) * np.eye(z_train.shape[1])
        weight = np.linalg.solve(gram, z_train.T @ centered)
        val_mu = np.expm1(np.clip(target_mean + z_val @ weight, 0.0, 20.0))
        score = float(_poisson_nll(y_val, val_mu).mean())
        path.append({"ridge": ridge, "validation_poisson_nll": score})
        if best is None or score < best[0]:
            best = (score, ridge, weight)
    assert best is not None
    pred = np.expm1(np.clip(target_mean + z_test @ best[2], 0.0, 20.0))
    return pred, {
        "selected_ridge": float(best[1]),
        "ridge_at_edge": bool(best[1] in (grid[0], grid[-1])),
        "validation_poisson_nll": float(best[0]),
        "path": path,
        "n_features": int(x_train.shape[1]),
    }


@torch.no_grad()
def collect_event_post_states(
    model,
    seq: SubjectSequence,
    timeline,
    *,
    device: torch.device,
    batch_size: int = 128,
    amp: bool = True,
) -> np.ndarray:
    """Causal post-event state for every retained interictal event."""

    model.eval()
    out = np.zeros((timeline.event_times.size, model.state.cfg.state_dim), dtype=np.float32)
    for segment_index, segment in enumerate(timeline.segments):
        positions = np.flatnonzero(timeline.event_segment == segment_index)
        if not positions.size:
            continue
        state = model.state.initial(1, device)
        previous_time = float(segment.start_epoch)
        for lo in range(0, positions.size, batch_size):
            pos = positions[lo : lo + batch_size]
            raw = seq.gather_positions(timeline.stream_positions[pos])
            batch = _to_device(raw, device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
            ):
                event_embedding, _ = model.event_encoder(batch)
            event_embedding = event_embedding.float()
            for j, event_pos in enumerate(pos):
                event_time = float(timeline.event_times[event_pos])
                state_pre = model.state.evolve(
                    state, torch.tensor([event_time - previous_time], device=device)
                )
                state = model.state.update(state_pre, event_embedding[j : j + 1])
                out[event_pos] = state.squeeze(0).cpu().numpy()
                previous_time = event_time
    return out


@torch.no_grad()
def anchor_states(model, timeline, post_states: np.ndarray, device: torch.device) -> np.ndarray:
    out = np.zeros((timeline.grid.n_anchors, model.state.cfg.state_dim), dtype=np.float32)
    for segment_index, segment in enumerate(timeline.segments):
        anchor_pos = np.flatnonzero(timeline.grid.segment_index == segment_index)
        if not anchor_pos.size:
            continue
        for a in anchor_pos:
            last = int(timeline.grid.last_event_pos[a])
            if last >= 0 and int(timeline.event_segment[last]) == segment_index:
                state = torch.from_numpy(post_states[last : last + 1]).to(device)
                base_time = float(timeline.event_times[last])
            else:
                state = model.state.initial(1, device)
                base_time = float(segment.start_epoch)
            flowed = model.state.evolve(
                state,
                torch.tensor([float(timeline.grid.t_anchor[a]) - base_time], device=device),
            )
            out[a] = flowed.squeeze(0).cpu().numpy()
    return out


def _block_shift_donor(timeline, anchors: np.ndarray, horizon: float) -> np.ndarray:
    """Half-segment circular shift, retaining only physically distant donors."""

    donor = np.full(anchors.size, -1, dtype=np.int64)
    for segment_index in np.unique(timeline.grid.segment_index[anchors]):
        local = np.flatnonzero(timeline.grid.segment_index[anchors] == segment_index)
        if local.size < 3:
            continue
        shift = max(local.size // 2, 1)
        candidate = np.roll(local, shift)
        dt = np.abs(
            timeline.grid.t_anchor[anchors[local]]
            - timeline.grid.t_anchor[anchors[candidate]]
        )
        ok = dt > float(horizon)
        donor[local[ok]] = candidate[ok]
    return donor


@torch.no_grad()
def _expected_count(model, states: np.ndarray, horizon: float, device: torch.device) -> np.ndarray:
    if not states.size:
        return np.zeros(0, dtype=np.float64)
    state = torch.from_numpy(states).to(device).float()
    grid = torch.linspace(0.0, float(horizon), 33, device=device)
    b, d = state.shape
    repeated = state[:, None, :].expand(b, grid.numel(), d).reshape(-1, d)
    dt = grid[None, :].expand(b, -1).reshape(-1)
    flowed = model.state.evolve(repeated, dt)
    intensity = model.state.intensity(flowed).reshape(b, -1)
    expected = torch.trapezoid(intensity, grid, dim=1)
    return expected.cpu().numpy().astype(np.float64)


@torch.no_grad()
def _score_future_marks(
    model,
    timeline,
    anchor_indices: np.ndarray,
    states: np.ndarray,
    horizon_index: int,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Autonomous state flow; actual future event marks are only scoring targets."""

    pair_anchor: list[int] = []
    pair_event: list[int] = []
    for local, anchor in enumerate(anchor_indices):
        lo = int(timeline.grid.window_lo[anchor, horizon_index])
        hi = int(timeline.grid.window_hi[anchor, horizon_index])
        pair_anchor.extend([local] * (hi - lo))
        pair_event.extend(range(lo, hi))
    n_anchor = anchor_indices.size
    size_sum = np.zeros(n_anchor, dtype=np.float64)
    subset_sum = np.zeros(n_anchor, dtype=np.float64)
    active_sum = np.zeros(n_anchor, dtype=np.float64)
    select_sum = np.zeros(n_anchor, dtype=np.float64)
    if not pair_event:
        return {
            "size_nll_per_step": np.full(n_anchor, np.nan),
            "subset_nll_per_group": np.full(n_anchor, np.nan),
            "n_event_pairs": 0,
        }
    pair_anchor_a = np.asarray(pair_anchor, dtype=np.int64)
    pair_event_a = np.asarray(pair_event, dtype=np.int64)
    group_ids_all = np.asarray(timeline.marks.participation, dtype=bool)  # shape check only
    del group_ids_all
    root = DATASET_ROOT / timeline.subject
    seq = SubjectSequence(root)
    raw_ids = np.asarray(
        seq.gather_positions(timeline.stream_positions[pair_event_a])["tied_group_id"],
        dtype=np.int64,
    )
    counts = _group_count(raw_ids)
    for lo in range(0, pair_event_a.size, batch_size):
        sl = slice(lo, lo + batch_size)
        a_local = pair_anchor_a[sl]
        event_pos = pair_event_a[sl]
        base = torch.from_numpy(states[a_local]).to(device).float()
        dt = torch.from_numpy(
            timeline.event_times[event_pos]
            - timeline.grid.t_anchor[anchor_indices[a_local]]
        ).to(device).float()
        flowed = model.state.evolve(base, dt)
        ids = torch.from_numpy(raw_ids[sl]).to(device).long()
        count = torch.from_numpy(counts[sl]).to(device).long()
        terms, _ = model.grammar(ids, count, flowed)
        np.add.at(size_sum, a_local, -terms.group_size_step_log_prob.sum(-1).cpu().numpy())
        np.add.at(subset_sum, a_local, -terms.subset_step_log_prob.sum(-1).cpu().numpy())
        np.add.at(active_sum, a_local, terms.active_step.sum(-1).cpu().numpy())
        np.add.at(select_sum, a_local, terms.select_step.sum(-1).cpu().numpy())
    return {
        "size_nll_per_step": np.divide(
            size_sum, active_sum, out=np.full_like(size_sum, np.nan), where=active_sum > 0
        ),
        "subset_nll_per_group": np.divide(
            subset_sum, select_sum, out=np.full_like(subset_sum, np.nan), where=select_sum > 0
        ),
        "n_event_pairs": int(pair_event_a.size),
    }


def evaluate_open_loop(
    subject: str,
    seed: int,
    *,
    checkpoint: Path,
    grammar_checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    cfg: PilotConfig = PilotConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    report_path = out_dir / "open_loop.json"
    array_path = out_dir / "open_loop_arrays.npz"
    if report_path.exists() and array_path.exists() and not overwrite:
        return json.loads(report_path.read_text())
    seq = SubjectSequence(DATASET_ROOT / subject)
    timeline = load_subject_timeline(subject)
    model = build_model(subject, seq, grammar_checkpoint, seed=seed, device=device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    post = collect_event_post_states(model, seq, timeline, device=device, amp=cfg.amp)
    anchor = anchor_states(model, timeline, post, device)
    arrays: dict[str, np.ndarray] = {
        "anchor_time": timeline.grid.t_anchor,
        "anchor_state": anchor,
        "event_post_state": post,
    }
    report: dict[str, Any] = {
        "format": "group_event_state_v0_3_open_loop",
        "subject": subject,
        "seed": seed,
        "status": "complete",
        "open_loop_definition": (
            "state at each fixed test anchor is evolved in real time without "
            "reading future events; actual future counts and marks are targets only"
        ),
        "horizons": {},
        "sealed_partition_opened": False,
        "source_commit": SOURCE_COMMIT,
    }
    baseline_keep = _eligible_baseline_columns(timeline.baseline.names)
    for h_i, horizon in enumerate(timeline.config.horizons_seconds):
        masks = {
            name: timeline.anchor_mask(name, h_i)
            for name in ("train", "val", "test")
        }
        indices = {name: np.flatnonzero(mask) for name, mask in masks.items()}
        count = {
            name: (
                timeline.grid.window_hi[idx, h_i] - timeline.grid.window_lo[idx, h_i]
            ).astype(np.int64)
            for name, idx in indices.items()
        }
        test_idx = indices["test"]
        state_test = anchor[test_idx]
        expected = _expected_count(model, state_test, horizon, device)
        equilibrium_state = np.broadcast_to(
            model.state.mean.detach().cpu().numpy(), state_test.shape
        ).copy()
        zero_state = np.zeros_like(state_test)  # neutral input of calibrated grammar
        expected_zero = _expected_count(model, equilibrium_state, horizon, device)
        baseline_pred, baseline_fit = _fit_count_ridge(
            timeline.baseline.x[indices["train"]][:, baseline_keep], count["train"],
            timeline.baseline.x[indices["val"]][:, baseline_keep], count["val"],
            timeline.baseline.x[test_idx][:, baseline_keep],
        )
        donor = _block_shift_donor(timeline, test_idx, horizon)
        donor_ok = donor >= 0
        shifted_states = np.zeros_like(state_test)
        shifted_states[donor_ok] = state_test[donor[donor_ok]]
        expected_shift = np.full(test_idx.size, np.nan)
        expected_shift[donor_ok] = _expected_count(
            model, shifted_states[donor_ok], horizon, device
        )
        correct_mark = _score_future_marks(
            model, timeline, test_idx, state_test, h_i, device=device
        )
        zero_mark = _score_future_marks(
            model, timeline, test_idx, zero_state, h_i, device=device
        )
        shift_mark = _score_future_marks(
            model, timeline, test_idx[donor_ok], shifted_states[donor_ok], h_i,
            device=device,
        )
        key = f"{int(horizon)}s"
        actual = count["test"]
        entry = {
            "horizon_seconds": float(horizon),
            "n_test_anchors": int(test_idx.size),
            "n_shift_matched_anchors": int(donor_ok.sum()),
            "count_poisson_nll": {
                "correct_state": float(_poisson_nll(actual, expected).mean()),
                "state_free": float(_poisson_nll(actual, expected_zero).mean()),
                "multiscale_history": float(_poisson_nll(actual, baseline_pred).mean()),
                "block_shifted_state": float(
                    _poisson_nll(actual[donor_ok], expected_shift[donor_ok]).mean()
                ) if donor_ok.any() else None,
            },
            "mark_nll": {
                "correct_state": {
                    "size": float(np.nanmean(correct_mark["size_nll_per_step"])),
                    "subset": float(np.nanmean(correct_mark["subset_nll_per_group"])),
                },
                "state_free": {
                    "size": float(np.nanmean(zero_mark["size_nll_per_step"])),
                    "subset": float(np.nanmean(zero_mark["subset_nll_per_group"])),
                },
                "block_shifted_state": {
                    "size": float(np.nanmean(shift_mark["size_nll_per_step"])),
                    "subset": float(np.nanmean(shift_mark["subset_nll_per_group"])),
                } if donor_ok.any() else None,
            },
            "multiscale_count_fit": baseline_fit,
            "state_free_definition": {
                "count": "fixed dynamical equilibrium with TRAIN marginal intensity",
                "mark": "zero state adapter input to the calibrated grammar",
            },
            "n_future_event_pairs": int(correct_mark["n_event_pairs"]),
        }
        report["horizons"][key] = entry
        prefix = f"h{int(horizon)}"
        arrays.update({
            f"{prefix}_test_anchor_index": test_idx,
            f"{prefix}_count_true": actual,
            f"{prefix}_count_expected_state": expected,
            f"{prefix}_count_expected_state_free": expected_zero,
            f"{prefix}_count_expected_multiscale": baseline_pred,
            f"{prefix}_shift_donor_local": donor,
            f"{prefix}_correct_size_nll": correct_mark["size_nll_per_step"],
            f"{prefix}_correct_subset_nll": correct_mark["subset_nll_per_group"],
            f"{prefix}_state_free_size_nll": zero_mark["size_nll_per_step"],
            f"{prefix}_state_free_subset_nll": zero_mark["subset_nll_per_group"],
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = array_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, array_path)
    report["arrays"] = str(array_path)
    report["config"] = asdict(cfg)
    _atomic_json(report_path, report)
    return report

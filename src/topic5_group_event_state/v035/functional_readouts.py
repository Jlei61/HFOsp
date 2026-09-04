"""Frozen functional readouts for v0.3.5 H1/H2a.

The recurrent state is learned only from interictal group events.  This module
asks what that state predicts beyond the causal multiscale baseline ``q(t)``.
It deliberately scores both event-indexed futures (1/5/20 events) and fixed
physical futures (5/30/120 min), and separates participation, continuous lag,
multiband expression, cross-band timing and waveform morphology.

Readouts are low-capacity masked ridge models.  Their regularisation is chosen
on INNER and their final numbers are reported once on SELECTION.  No seizure or
development target is read here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
    FrozenDecoderBundle, decoder_tensors,
)

from .contracts import FORMAT_PREFIX, atomic_json
from .full_mark_state import FullMarkData, FullMarkStateModel, FullMarkTrainConfig


RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
PHYSICAL_HORIZONS = (300.0, 1800.0, 7200.0)
EVENT_OFFSETS = (1, 5, 20)


@dataclass(frozen=True)
class Endpoint:
    name: str
    values: np.ndarray
    valid: np.ndarray
    binary: bool = False


def _waveform_signature(waveform: np.ndarray, rows: np.ndarray, participation: np.ndarray,
                        contact_ok: np.ndarray, has_waveform: np.ndarray,
                        batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Five deterministic summaries of every full stored contact/view trace.

    This is intentionally an assay endpoint, not another learned representation:
    mean, RMS, peak absolute amplitude, peak time and line length are computed
    from all waveform samples in each bipolar/CAR/detector view.
    """

    n = rows.size
    sample = np.asarray(waveform[rows[:1]], dtype=np.float32)
    c, v = sample.shape[1:3]
    out = np.zeros((n, c * v * 5), dtype=np.float32)
    valid = np.zeros_like(out, dtype=bool)
    for lo in range(0, n, batch_size):
        rr = rows[lo : lo + batch_size]
        x = np.nan_to_num(np.asarray(waveform[rr], dtype=np.float32))
        mean = x.mean(-1)
        rms = np.sqrt(np.mean(np.square(x), axis=-1))
        peak = np.max(np.abs(x), axis=-1)
        peak_t = np.argmax(np.abs(x), axis=-1).astype(np.float32) / max(x.shape[-1] - 1, 1)
        line = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
        feat = np.stack((mean, rms, peak, peak_t, line), axis=-1).reshape(rr.size, -1)
        ok_contact = participation[lo : lo + rr.size] & contact_ok[lo : lo + rr.size]
        ok = np.repeat(ok_contact[:, :, None, None], v, axis=2)
        ok = np.repeat(ok, 5, axis=3).reshape(rr.size, -1)
        ok &= has_waveform[lo : lo + rr.size, None]
        out[lo : lo + rr.size] = feat
        valid[lo : lo + rr.size] = ok & np.isfinite(feat)
    return out, valid


def build_endpoints(data: FullMarkData) -> dict[str, Endpoint]:
    seq = data.seq
    raw_rows = seq.order[data.source_position]
    part = np.asarray(seq.arrays["participation"][raw_rows], dtype=bool)
    contact_ok = np.asarray(seq.arrays["contact_ok"][raw_rows], dtype=bool)
    delay = np.asarray(seq.arrays["relative_delay"][raw_rows], dtype=np.float32)
    band = np.asarray(seq.arrays["band_features"][raw_rows], dtype=np.float32)
    cross = np.asarray(seq.arrays["cross_band_lag"][raw_rows], dtype=np.float32)
    source_scalar = seq.scalars
    has = np.asarray(source_scalar["has_waveform"][raw_rows], dtype=bool)
    wave, wave_ok = _waveform_signature(
        seq.arrays["waveform"], raw_rows, part, contact_ok, has,
    )
    band_available = np.asarray(seq.index["band_available"], dtype=bool)
    band_ok = part[:, :, None] & contact_ok[:, :, None] & band_available[None, None, :]
    band_ok = np.broadcast_to(band_ok, band[..., 0].shape)
    cross_ok = np.broadcast_to((part & contact_ok)[:, :, None], cross.shape)
    out = {
        "participation_field": Endpoint(
            "participation_field", part.astype(np.float32),
            np.broadcast_to(seq.contact_valid[None], part.shape).copy(), binary=True,
        ),
        "extent_fraction": Endpoint(
            "extent_fraction", (part.sum(1, keepdims=True) / max(part.shape[1], 1)).astype(np.float32),
            np.ones((part.shape[0], 1), dtype=bool),
        ),
        "continuous_lag_field": Endpoint(
            "continuous_lag_field", delay, part & contact_ok & np.isfinite(delay),
        ),
        "multiband_log_energy_field": Endpoint(
            "multiband_log_energy_field", band[..., 2].reshape(part.shape[0], -1),
            (band_ok & np.isfinite(band[..., 2])).reshape(part.shape[0], -1),
        ),
        "multiband_peak_time_field": Endpoint(
            "multiband_peak_time_field", band[..., 0].reshape(part.shape[0], -1),
            (band_ok & np.isfinite(band[..., 0])).reshape(part.shape[0], -1),
        ),
        "cross_band_lag_field": Endpoint(
            "cross_band_lag_field", cross.reshape(part.shape[0], -1),
            (cross_ok & np.isfinite(cross)).reshape(part.shape[0], -1),
        ),
        "waveform_morphology": Endpoint("waveform_morphology", wave, wave_ok),
    }
    return out


def _design(q: np.ndarray, state: np.ndarray | None) -> np.ndarray:
    pieces = [np.ones((q.shape[0], 1), dtype=np.float64), np.asarray(q, dtype=np.float64)]
    if state is not None:
        pieces.append(np.asarray(state, dtype=np.float64))
    return np.concatenate(pieces, axis=1)


def _fit_scaler(values: np.ndarray, valid: np.ndarray, rows: np.ndarray, binary: bool) -> tuple[np.ndarray, np.ndarray]:
    if binary:
        return np.zeros(values.shape[1]), np.ones(values.shape[1])
    # Some contact-band coordinates can be absent throughout FIT.  Compute
    # robust statistics only on genuinely observed values so this expected
    # assay limitation does not emit an all-NaN warning or hide a real one.
    centre = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        use = valid[rows, column] & np.isfinite(values[rows, column])
        observed = values[rows[use], column]
        if observed.size == 0:
            continue
        centre[column] = float(np.median(observed))
        mad = float(np.median(np.abs(observed - centre[column])))
        if np.isfinite(mad) and mad > 1e-6:
            scale[column] = 1.4826 * mad
    return centre, scale


def _fit_masked_ridge(x: np.ndarray, y: np.ndarray, valid: np.ndarray, rows: np.ndarray,
                      alpha: float, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    ys = (y - centre[None]) / scale[None]
    coef = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    for j in range(y.shape[1]):
        rr = rows[valid[rows, j] & np.isfinite(ys[rows, j])]
        if rr.size < max(8, x.shape[1] // 2):
            coef[0, j] = np.nanmean(ys[rr, j]) if rr.size else 0.0
            continue
        gram = x[rr].T @ x[rr] + penalty
        coef[:, j] = np.linalg.solve(gram, x[rr].T @ ys[rr, j])
    return coef


def _score(pred_scaled: np.ndarray, y: np.ndarray, valid: np.ndarray, rows: np.ndarray,
           centre: np.ndarray, scale: np.ndarray, binary: bool) -> tuple[float | None, int]:
    pred = pred_scaled * scale[None] + centre[None]
    mask = valid[rows] & np.isfinite(y[rows]) & np.isfinite(pred[rows])
    if not mask.any():
        return None, 0
    if binary:
        loss = (np.clip(pred[rows], 0.0, 1.0) - y[rows]) ** 2
    else:
        loss = ((pred[rows] - y[rows]) / scale[None]) ** 2
    return float(np.mean(loss[mask])), int(mask.sum())


def _select_and_score(endpoint: Endpoint, q: np.ndarray, state: np.ndarray,
                      fit: np.ndarray, inner: np.ndarray, selection: np.ndarray,
                      shifted_state: np.ndarray, shift_valid: np.ndarray) -> dict[str, Any]:
    centre, scale = _fit_scaler(endpoint.values, endpoint.valid, fit, endpoint.binary)
    arms: dict[str, Any] = {}
    designs = (
        ("static_only", np.zeros_like(q), None),
        ("q_only", q, None),
        ("state_only", np.zeros_like(q), state),
        ("q_plus_state", q, state),
    )
    for name, q_used, state_used in designs:
        x = _design(q_used, state_used)
        best = None
        for alpha in RIDGES:
            coef = _fit_masked_ridge(x, endpoint.values, endpoint.valid, fit, alpha, centre, scale)
            value, n = _score(x @ coef, endpoint.values, endpoint.valid, inner, centre, scale, endpoint.binary)
            if value is not None and (best is None or value < best[0]):
                best = (value, alpha, coef, n)
        if best is None:
            arms[name] = {"status": "NOT_ESTIMABLE"}
            continue
        value, n = _score(x @ best[2], endpoint.values, endpoint.valid, selection, centre, scale, endpoint.binary)
        arms[name] = {"selection_loss": value, "n_values": n, "alpha": best[1],
                      "inner_loss": best[0]}
        if name == "q_plus_state":
            x_shift = _design(q, shifted_state)
            shifted_rows = selection[shift_valid[selection]]
            sv, sn = _score(x_shift @ best[2], endpoint.values, endpoint.valid, shifted_rows,
                            centre, scale, endpoint.binary)
            arms["block_shift_state"] = {"selection_loss": sv, "n_values": sn,
                                         "alpha": best[1]}
            # The block-shift null is only defined on anchors with a distant
            # donor.  Score the correct-time state on exactly that support so
            # the timing contrast never mixes two anchor populations
            # (review 2026-09-04).
            cv_support, cn_support = _score(x @ best[2], endpoint.values, endpoint.valid,
                                            shifted_rows, centre, scale, endpoint.binary)
            arms["correct_state_on_shift_support"] = {"selection_loss": cv_support,
                                                      "n_values": cn_support,
                                                      "alpha": best[1]}
            mean_state = np.nanmean(state[fit], axis=0, keepdims=True)
            x_const = _design(q, np.broadcast_to(mean_state, state.shape))
            cv, cn = _score(x_const @ best[2], endpoint.values, endpoint.valid, selection,
                            centre, scale, endpoint.binary)
            arms["fit_period_mean_state"] = {"selection_loss": cv, "n_values": cn,
                                               "alpha": best[1]}
    # A null arm that cannot be evaluated is still a registered scientific
    # outcome.  Keep it explicit instead of omitting the key: omission is
    # indistinguishable from an interrupted pipeline, whereas
    # ``NOT_ESTIMABLE`` preserves the assay limitation (most often too little
    # 120-min support) without manufacturing a zero contrast.
    arms.setdefault("block_shift_state", {
        "status": "NOT_ESTIMABLE",
        "reason": "q_plus_state readout or valid distant block-shift support unavailable",
    })
    arms.setdefault("fit_period_mean_state", {
        "status": "NOT_ESTIMABLE",
        "reason": "q_plus_state readout unavailable on FIT/INNER support",
    })
    arms.setdefault("correct_state_on_shift_support", {
        "status": "NOT_ESTIMABLE",
        "reason": "q_plus_state readout or valid distant block-shift support unavailable",
    })
    qloss = arms.get("q_only", {}).get("selection_loss")
    sloss = arms.get("q_plus_state", {}).get("selection_loss")
    staticloss = arms.get("static_only", {}).get("selection_loss")
    stateonlyloss = arms.get("state_only", {}).get("selection_loss")
    shiftloss = arms.get("block_shift_state", {}).get("selection_loss")
    supportloss = arms.get("correct_state_on_shift_support", {}).get("selection_loss")
    constantloss = arms.get("fit_period_mean_state", {}).get("selection_loss")
    arms["contrasts"] = {
        "rate_gain_over_static": (
            None if staticloss is None or qloss is None else staticloss - qloss
        ),
        "state_only_gain_over_static": (
            None if staticloss is None or stateonlyloss is None else staticloss - stateonlyloss
        ),
        "state_gain_over_q": None if qloss is None or sloss is None else qloss - sloss,
        "correct_time_gain_over_shift": (
            None if shiftloss is None or supportloss is None else shiftloss - supportloss
        ),
        "correct_time_support": "block_shift_state versus correct_state_on_shift_support on identical anchors",
        "dynamic_gain_over_fit_period_mean": (
            None if constantloss is None or sloss is None else constantloss - sloss
        ),
        "loss_kind": "Brier" if endpoint.binary else "TRAIN-robust-standardized MSE",
    }
    return arms


def _block_shift(state: np.ndarray, time_: np.ndarray, segment: np.ndarray,
                 rows: np.ndarray, min_shift: float) -> tuple[np.ndarray, np.ndarray]:
    out = state.copy()
    valid = np.zeros(time_.size, dtype=bool)
    for seg in np.unique(segment[rows]):
        rr = rows[segment[rows] == seg]
        if rr.size < 4:
            continue
        # Search a half-block rotation first, then accept only genuinely distant donors.
        donor = np.roll(rr, rr.size // 2)
        ok = np.abs(time_[donor] - time_[rr]) >= float(min_shift)
        out[rr[ok]] = state[donor[ok]]
        valid[rr[ok]] = True
    return out, valid


def _event_offset_endpoint(endpoint: Endpoint, next_index: np.ndarray, offset_j: int) -> Endpoint:
    target = next_index[:, offset_j]
    good = target >= 0
    values = np.zeros((target.size, endpoint.values.shape[1]), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    values[good] = endpoint.values[target[good]]
    valid[good] = endpoint.valid[target[good]]
    return Endpoint(endpoint.name, values, valid, endpoint.binary)


def _states_at_grid(trajectory: dict[str, np.ndarray], anchor_time: np.ndarray,
                    anchor_segment: np.ndarray) -> np.ndarray:
    et = trajectory["event_time"]
    es = trajectory["event_segment"]
    post = trajectory["state_post"]
    mean = trajectory["state_mean"]
    taus = trajectory["fixed_taus_seconds"]
    out = np.broadcast_to(mean, (anchor_time.size, mean.size)).copy().astype(np.float32)
    for seg in np.unique(anchor_segment):
        ar = np.flatnonzero(anchor_segment == seg)
        er = np.flatnonzero(es == seg)
        if er.size == 0:
            continue
        pos = np.searchsorted(et[er], anchor_time[ar], side="left") - 1
        ok = pos >= 0
        donor = er[np.maximum(pos, 0)]
        dt = anchor_time[ar] - et[donor]
        decayed = mean[None] + (post[donor] - mean[None]) * np.exp(-dt[:, None] / taus[None])
        out[ar[ok]] = decayed[ok]
    return out


def _physical_endpoint(endpoint: Endpoint, event_time: np.ndarray, event_segment: np.ndarray,
                       anchor_time: np.ndarray, anchor_segment: np.ndarray,
                       target_valid: np.ndarray, horizon_j: int, horizon: float) -> Endpoint:
    values = np.zeros((anchor_time.size, endpoint.values.shape[1]), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    for i, (t, seg) in enumerate(zip(anchor_time, anchor_segment)):
        if not target_valid[i, horizon_j]:
            continue
        rows = np.flatnonzero((event_segment == seg) & (event_time >= t) & (event_time < t + horizon))
        if rows.size == 0:
            continue
        mask = endpoint.valid[rows]
        count = mask.sum(0)
        values[i] = np.where(count > 0, np.where(mask, endpoint.values[rows], 0.0).sum(0) / np.maximum(count, 1), 0.0)
        valid[i] = count > 0
    return Endpoint(endpoint.name, values, valid, endpoint.binary)


def run_functional_readouts(
    data: FullMarkData,
    trajectory_path: Path,
    rate_trajectory_path: Path,
    *,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    with np.load(trajectory_path, allow_pickle=False) as z:
        trajectory = {k: np.asarray(z[k]) for k in z.files}
    endpoints = build_endpoints(data)
    fit = np.flatnonzero(data.phase == "FIT")
    inner = np.flatnonzero(data.phase == "INNER")
    selection = np.flatnonzero(data.phase == "SELECTION")
    state = trajectory["state_post"]
    shifted, shift_valid = _block_shift(state, data.event_time, data.event_segment, selection, 1800.0)
    event_results: dict[str, Any] = {}
    for j, offset in enumerate(data.event_offsets):
        event_results[f"next_{offset}_events"] = {
            name: _select_and_score(
                _event_offset_endpoint(endpoint, data.next_index, j), data.q_context, state,
                fit, inner, selection, shifted, shift_valid,
            )
            for name, endpoint in endpoints.items()
        }

    with np.load(rate_trajectory_path, allow_pickle=False) as z:
        at = np.asarray(z["anchor_time"], dtype=np.float64)
        aseg = np.asarray(z["segment"], dtype=np.int64)
        phase = np.asarray(z["phase"]).astype(str)
        q = np.asarray(z["q_standardized"], dtype=np.float32)
        target_valid = np.asarray(z["target_valid"], dtype=bool)
    grid_state = _states_at_grid(trajectory, at, aseg)
    pfit, pinner, pselection = (np.flatnonzero(phase == p) for p in ("FIT", "INNER", "SELECTION"))
    physical_results: dict[str, Any] = {}
    physical_horizons = tuple(data.physical_horizons_seconds)
    if len(physical_horizons) != target_valid.shape[1]:
        raise ValueError("physical horizon metadata and target mask disagree")
    for j, horizon in enumerate(physical_horizons):
        shifted_grid, shifted_valid = _block_shift(
            grid_state, at, aseg, pselection, max(1800.0, horizon),
        )
        physical_results[f"future_{int(horizon // 60)}min"] = {
            name: _select_and_score(
                _physical_endpoint(endpoint, data.event_time, data.event_segment, at, aseg,
                                   target_valid, j, horizon),
                q, grid_state, pfit, pinner, pselection, shifted_grid, shifted_valid,
            )
            for name, endpoint in endpoints.items()
        }
    card = {
        "format": f"{FORMAT_PREFIX}_functional_readouts_v1",
        "subject": data.subject,
        "trajectory": str(trajectory_path),
        "rate_trajectory": str(rate_trajectory_path),
        "event_horizons": event_results,
        "physical_horizons": physical_results,
        "endpoint_semantics": {
            "participation_field": "per-contact participation; Brier score",
            "continuous_lag_field": "per-contact continuous lag, participating contacts only",
            "multiband_log_energy_field": "per-contact per-band log integrated energy",
            "multiband_peak_time_field": "per-contact per-band peak time",
            "cross_band_lag_field": "per-contact cross-band timing differences",
            "waveform_morphology": "full-trace mean/RMS/peak/peak-time/line-length per contact and reference view",
        },
        "split_rule": "FIT readout fitting; INNER ridge selection; SELECTION one-time report",
        "state_nulls": ["fit-period mean state", "within-segment block-circular shift"],
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_json(card_path, card)
    return card

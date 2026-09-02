"""Residual-positive synthetic targets planted on a DataView (design A6).

Until Agent A's D0-D4 generators arrive, "synthetic recovery" uses the v0.3.2
construction: a hidden marked leaky component (tau = 30 min) of a fixed random
non-linear projection of the *real* event tokens, standardised on TRAIN, added
to ``log mu_H`` of every bin with strength ``beta``; counts drawn NB(mu, r).
The hidden component is information ``H`` does not carry (R^2 audit).  Source
label: ``v032_residual_positive_proxy``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from src.topic5_group_event_state.v03.evaluate import _eligible_baseline_columns
from src.topic5_group_event_state.v032_model.readout import fit_nb_log_dispersion
from src.topic5_group_event_state.v032_model.state import anchor_states, leaky_bank_trajectory

from .data import DataView
from .paths import payload_hash

SOURCE = "v032_residual_positive_proxy"
HIDDEN_TAU_SECONDS = 1800.0
HIDDEN_WIDTH = 4


def hidden_component(view: DataView, *, generator_seed: int, hidden_tau: float = HIDDEN_TAU_SECONDS) -> np.ndarray:
    """TRAIN-standardised hidden leaky component ``z`` (A,) of the real tokens."""

    rng = np.random.default_rng(int(generator_seed))
    d = view.n_features
    weight = rng.normal(size=(d, HIDDEN_WIDTH)) / np.sqrt(d)
    bias = rng.normal(size=HIDDEN_WIDTH) * 0.5
    g = np.tanh(view.x_scaled.astype(np.float64) @ weight + bias)
    times = torch.from_numpy(view.event_times)
    seg = torch.from_numpy(view.event_segment)
    _pre, post = leaky_bank_trajectory(torch.from_numpy(g).float(), times, seg, torch.tensor([float(hidden_tau)]),
                                       chunk_seconds=3600.0)
    s = anchor_states(post, times, torch.from_numpy(view.t_anchor), torch.from_numpy(view.last_event_pos),
                      torch.full((HIDDEN_WIDTH,), float(hidden_tau))).numpy().astype(np.float64)
    raw = s @ rng.normal(size=HIDDEN_WIDTH)
    train = view.phase_index["train"]
    return (raw - raw[train].mean()) / max(float(raw[train].std()), 1e-9)


def hidden_r2_against_baseline(view: DataView, z: np.ndarray, *, phase: str = "train", ridge: float = 1e-2) -> float:
    """In-sample linear R^2 of ``z`` from the explicit-history features (seizure columns removed)."""

    idx = view.phase_index[phase]
    bundle = view.bundle
    if bundle is not None and getattr(bundle, "baseline_x", None) is not None:
        keep = _eligible_baseline_columns(tuple(bundle.baseline_names))
        x = np.asarray(bundle.baseline_x, dtype=np.float64)[idx][:, keep]
    else:
        x = np.nan_to_num(view.log_mu_h[idx], nan=0.0)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    xs = np.column_stack([np.ones(idx.size), (x - mean) / scale])
    beta = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ z[idx])
    resid = z[idx] - xs @ beta
    total = float(((z[idx] - z[idx].mean()) ** 2).sum())
    return float(1.0 - (resid ** 2).sum() / max(total, 1e-12))


def plant_residual_signal(
    view: DataView,
    *,
    beta: float,
    dispersion_r: float,
    generator_seed: int,
    noise_seed: int,
    hidden_tau: float = HIDDEN_TAU_SECONDS,
) -> tuple[DataView, dict[str, Any]]:
    """Copy of ``view`` whose exposed counts follow ``NB(exp(log mu_H + beta z), r)`` on every bin."""

    z = hidden_component(view, generator_seed=generator_seed, hidden_tau=hidden_tau)
    exposed = np.concatenate([view.phase_index["train"], view.phase_index["inner_val"]])
    log_mu_true = view.log_mu_h + float(beta) * z[:, None]
    if not np.isfinite(log_mu_true[exposed]).all():
        raise ValueError("non-finite log mu_H on an exposed anchor; cannot synthesise counts")
    noise = np.random.default_rng(int(noise_seed))
    r = float(dispersion_r)
    mu = np.exp(np.where(np.isfinite(log_mu_true), log_mu_true, 0.0))
    counts = np.full(view.counts.shape, -1, dtype=np.int64)
    counts[exposed] = noise.negative_binomial(r, r / (r + mu[exposed])).astype(np.int64)
    train = view.phase_index["train"]
    log_r_h = np.array([fit_nb_log_dispersion(counts[train, b], np.exp(view.log_mu_h[train, b]))
                        for b in range(view.n_bins)], dtype=np.float64)
    spec = {"source": SOURCE, "beta": float(beta), "dispersion_r": r, "generator_seed": int(generator_seed),
            "noise_seed": int(noise_seed), "hidden_tau_seconds": float(hidden_tau), "hidden_width": HIDDEN_WIDTH}
    fingerprint = dict(view.fingerprint)
    fingerprint["synthetic"] = spec
    planted = replace(view, counts=counts, log_r_h=log_r_h, fingerprint=fingerprint,
                      input_hash=payload_hash({"base": view.input_hash, "synthetic": spec}))
    info = {**spec, "z": z, "log_mu_true": log_mu_true,
            "r2_hidden_vs_baseline_train": hidden_r2_against_baseline(view, z)}
    return planted, info

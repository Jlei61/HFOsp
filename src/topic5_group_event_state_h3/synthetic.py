"""Synthetic patients with a known feedback edge, for identifiability only.

This calibrates the *instrument*: under a generator that really has a count edge,
does M1 beat M0?  Under one that really has a mark edge at fixed count and time,
does M2 beat M1?  Under one with no edge at all, do they tie?

A pass here says the comparison can see an edge of this size at this support.  It
says nothing whatsoever about H3 in patients, and it is never a gate on whether
the human analysis proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .features import EventFeatures

TRUTHS = ("zero_feedback", "count_feedback", "mark_feedback", "intercept_only", "linear_drift")


@dataclass
class SyntheticPatient:
    features: EventFeatures
    background_time: np.ndarray
    background_features: np.ndarray
    block_ranges: list[tuple[float, float]]
    truth: str
    latent: np.ndarray
    latent_time: np.ndarray
    feedback: np.ndarray
    params: dict[str, Any]


def generate(
    truth: Literal["zero_feedback", "count_feedback", "mark_feedback", "intercept_only", "linear_drift"],
    *,
    hours: float = 120.0,
    base_rate_hz: float = 0.08,
    n_contacts: int = 8,
    tau_latent_s: float = 3600.0,
    rate_gain: float = 0.8,
    feedback_strength: float = 0.6,
    seed: int = 0,
    grid_seconds: float = 30.0,
) -> SyntheticPatient:
    """One synthetic recording whose feedback edge is known by construction.

    The latent process is simulated on a fine grid and events are drawn by
    thinning, so the event times are a genuine inhomogeneous point process rather
    than a rate curve sampled at fixed spacing.

    ``count_feedback`` kicks the latent by how far the realised count fell from
    what the current rate implied -- a *raw* count kick on a log-rate is positive
    feedback and diverges, which would test the estimator against a regime no
    recording is in.  ``mark_feedback`` kicks it by the sum of the events' mark
    signs, whose mean over marks is zero, so it is invisible to anything that only
    counts events: exactly the separation the M2-vs-M1 contrast has to make.
    """

    if truth not in TRUTHS:
        raise ValueError(f"unknown truth {truth!r}; expected one of {TRUTHS}")
    rng = np.random.default_rng(seed)
    t0 = 1_200_000_000.0
    n_grid = int(hours * 3600.0 / grid_seconds)
    grid = t0 + grid_seconds * np.arange(n_grid, dtype=np.float64)

    # Slow background process, common to every truth.
    innovation = rng.normal(0.0, 1.0, n_grid)
    decay = float(np.exp(-grid_seconds / tau_latent_s))
    background = np.zeros(n_grid)
    for i in range(1, n_grid):
        background[i] = decay * background[i - 1] + np.sqrt(1 - decay**2) * innovation[i]

    latent = np.zeros(n_grid)
    feedback_series = np.zeros(n_grid)
    times: list[float] = []
    mark_sign: list[float] = []
    feedback = 0.0
    drift = np.linspace(0.0, 1.0, n_grid)

    for i in range(n_grid):
        if truth == "intercept_only":
            latent[i] = 1.0
        elif truth == "linear_drift":
            latent[i] = drift[i]
        else:
            latent[i] = float(np.clip(background[i] + feedback, -4.0, 4.0))
        feedback_series[i] = feedback
        rate = base_rate_hz * float(np.exp(rate_gain * latent[i]))
        expected = rate * grid_seconds
        n_here = rng.poisson(expected)
        if n_here:
            offsets = np.sort(rng.uniform(0.0, grid_seconds, n_here))
            times.extend((grid[i] + offsets).tolist())
            signs = rng.choice([-1.0, 1.0], size=n_here)
            mark_sign.extend(signs.tolist())
            if truth == "count_feedback":
                # Driven by the *excess* over what the current rate already
                # implies, so the edge is mean-zero under its own dynamics.  A
                # raw count kick on a log-rate is positive feedback and runs away
                # -- an unstable generator would test the estimator against a
                # regime no recording is in.
                feedback += (
                    feedback_strength * (n_here - expected) / max(expected, 1.0) * 0.05
                )
            elif truth == "mark_feedback":
                feedback += feedback_strength * float(signs.sum()) / max(expected, 1.0) * 0.05
        feedback = float(np.clip(feedback * decay, -3.0, 3.0))

    t_abs = np.asarray(times, dtype=np.float64)
    if t_abs.size < 100:
        raise ValueError("synthetic generator produced too few events; raise base_rate_hz")
    signs = np.asarray(mark_sign, dtype=np.float32)

    # Marks: a sign channel that carries the mark-feedback truth, plus filler
    # dimensions with no relationship to anything, so M2's extra capacity has
    # somewhere to overfit if it is going to.
    n_mark = n_contacts + 4
    marks = rng.normal(0.0, 1.0, size=(t_abs.size, n_mark)).astype(np.float32)
    marks[:, 0] = signs
    participation = rng.random((t_abs.size, n_contacts)) < 0.5
    marks[:, 1 : 1 + n_contacts] = participation.astype(np.float32)

    dt_prev = np.full(t_abs.size, np.nan)
    dt_prev[1:] = np.diff(t_abs)
    size = participation.sum(1).astype(np.float32)
    count_features = np.stack(
        [
            np.ones(t_abs.size, dtype=np.float32),
            np.log1p(size),
            size / n_contacts,
            np.log1p(np.nan_to_num(dt_prev, nan=0.0)).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    features = EventFeatures(
        t_abs=t_abs,
        count_features=count_features,
        mark_features=marks,
        mark_group_slices={
            "participation": (1, 1 + n_contacts),
            "extent": (0, 1),
            "multiband": (1 + n_contacts, 2 + n_contacts),
            "waveform_crossband": (2 + n_contacts, n_mark),
        },
        count_feature_names=["occurrence", "log1p_size", "size_fraction", "log1p_dt_prev"],
        mark_feature_names=[f"m{i}" for i in range(n_mark)],
        participation=participation,
        size=size,
        band_available=np.ones(5, dtype=bool),
    )

    # Background observations the common-drive arm is allowed to see: a noisy view
    # of the same latent process, on its own fixed grid.
    anchor_time = grid.copy()
    anchor_features = np.stack(
        [background + rng.normal(0.0, 0.5, n_grid), np.roll(background, 1)], axis=1
    ).astype(np.float32)

    return SyntheticPatient(
        features=features,
        background_time=anchor_time,
        background_features=anchor_features,
        block_ranges=[(float(t0), float(t0 + hours * 3600.0))],
        truth=truth,
        latent=latent,
        latent_time=grid,
        feedback=feedback_series,
        params={
            "hours": hours,
            "base_rate_hz": base_rate_hz,
            "tau_latent_s": tau_latent_s,
            "rate_gain": rate_gain,
            "feedback_strength": feedback_strength,
            "seed": seed,
            "n_events": int(t_abs.size),
            "grid_seconds": grid_seconds,
            # The effect size in the only units that travel: how much of the
            # latent's own variance the feedback edge accounts for.  "Did the
            # instrument see it" is meaningless without this number.
            "feedback_variance_fraction": float(
                np.var(feedback_series) / max(np.var(latent), 1e-12)
            ),
            "feedback_std": float(np.std(feedback_series)),
            "latent_std": float(np.std(latent)),
        },
    )

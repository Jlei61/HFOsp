"""Train-only fixed repertoire ``mu_p`` and the static / dynamic variance split.

``logit p(E_{p,e}) = mu_p + D_psi(z_{p,e}^-)``.  ``mu_p`` is estimated from this
patient's train events alone and frozen; the state only ever explains motion
around it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .event_marks import PatientEvents


@dataclass(frozen=True)
class PatientBaseline:
    subject: str
    participation_logit: np.ndarray   # (N,) train-only logit of marginal participation
    order_score: np.ndarray           # (N,) train-only Plackett-Luce score
    stop_logit: float                 # train-only global continuation bias
    mean_load: float
    n_train_events: int

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "participation_logit": self.participation_logit.tolist(),
            "order_score": self.order_score.tolist(),
            "stop_logit": float(self.stop_logit),
            "mean_load": float(self.mean_load),
            "n_train_events": int(self.n_train_events),
        }


def estimate_baseline(events: PatientEvents, *, split: str = "train",
                      pseudo_count: float = 1.0, n_iter: int = 60) -> PatientBaseline:
    mask = events.split_mask(split)
    part = events.participation[mask]
    gid = events.group_ids[mask]
    n_events, n_contacts = part.shape
    if n_events == 0:
        raise ValueError(f"{events.subject}: empty {split} partition")

    rate = (part.sum(axis=0) + pseudo_count) / (n_events + 2.0 * pseudo_count)
    participation_logit = np.log(rate) - np.log1p(-rate)

    order_score = _plackett_luce_scores(part, gid, n_iter=n_iter)

    n_part = part.sum(axis=1)
    continue_steps = float(np.sum(np.maximum(n_part - 1, 0)))
    stop_steps = float(n_events)
    stop_logit = float(np.log((stop_steps + 1.0) / (continue_steps + 1.0)))

    return PatientBaseline(
        subject=events.subject,
        participation_logit=participation_logit.astype(np.float32),
        order_score=order_score.astype(np.float32),
        stop_logit=stop_logit,
        mean_load=float(events.load[mask].mean()),
        n_train_events=int(n_events),
    )


def _plackett_luce_scores(part: np.ndarray, gid: np.ndarray, *, n_iter: int) -> np.ndarray:
    """Minorisation-maximisation fit of a static tie-aware Plackett-Luce model.

    ``w_i`` is updated as ``wins_i / sum_over_steps(indicator i was a candidate) /
    normaliser``; the classical Hunter MM update, run on the recruitment steps of
    every train event with ties treated as exchangeable multi-selections.
    """
    n_events, n_contacts = part.shape
    wins = np.zeros(n_contacts)
    # candidate exposure has to be accumulated per step, so precompute the
    # step-wise candidate sets as an (events x contacts) rank-of-group array
    order = np.where(part, gid.astype(np.int32), np.iinfo(np.int32).max)
    max_group = int(np.max(np.where(part, gid, 0))) + 1
    for e in range(n_events):
        idx = np.flatnonzero(part[e])
        wins[idx] += 1.0
    weights = np.ones(n_contacts)
    for _ in range(n_iter):
        denom = np.zeros(n_contacts)
        # accumulate 1/normaliser for every step where a contact was a candidate
        for k in range(max_group):
            candidate = order >= k           # (events, contacts) still available
            active = candidate & part
            has_step = (np.where(part, gid, -1) == k).any(axis=1)
            if not has_step.any():
                continue
            rows = np.flatnonzero(has_step)
            block = active[rows]
            normaliser = block @ weights
            normaliser = np.where(normaliser > 0, normaliser, 1.0)
            multiplicity = ((np.where(part, gid, -1) == k)[rows]).sum(axis=1)
            denom += block.T @ (multiplicity / normaliser)
        weights = np.where(denom > 0, wins / np.maximum(denom, 1e-12), weights)
        weights = weights / max(weights.mean(), 1e-12)
    return np.log(np.maximum(weights, 1e-8))


def variance_decomposition(events: PatientEvents, baseline: PatientBaseline,
                           *, split: str = "train", block: int = 200) -> dict:
    """How much of a patient's repertoire is the fixed ``mu_p`` and how much moves.

    The dynamic part is the variance of block-mean participation around the
    patient's own train mean, corrected for the binomial sampling variance a
    fixed-repertoire patient would show at the same block size.
    """
    mask = events.split_mask(split)
    part = events.participation[mask].astype(np.float64)
    n_events, n_contacts = part.shape
    if n_events < block * 2:
        block = max(20, n_events // 4)
    n_blocks = n_events // block
    if n_blocks < 2:
        return {"subject": events.subject, "status": "insufficient_blocks",
                "n_events": n_events, "block": block, "n_blocks": n_blocks}
    trimmed = part[: n_blocks * block].reshape(n_blocks, block, n_contacts)
    block_mean = trimmed.mean(axis=1)                      # (n_blocks, N)
    grand = part.mean(axis=0)                              # (N,)
    between = block_mean.var(axis=0, ddof=1)               # (N,)
    expected_sampling = grand * (1.0 - grand) / block      # binomial null
    dynamic = np.maximum(between - expected_sampling, 0.0)
    static = grand * (1.0 - grand)
    total = static + dynamic
    share = np.divide(dynamic, total, out=np.zeros_like(dynamic), where=total > 0)
    return {
        "subject": events.subject,
        "dataset": events.dataset,
        "status": "ok",
        "n_events": int(n_events),
        "n_contacts": int(n_contacts),
        "block_events": int(block),
        "n_blocks": int(n_blocks),
        "static_variance_mean": float(static.mean()),
        "dynamic_variance_mean": float(dynamic.mean()),
        "dynamic_share_mean": float(share.mean()),
        "dynamic_share_max": float(share.max()),
        "between_block_variance_mean": float(between.mean()),
        "expected_sampling_variance_mean": float(expected_sampling.mean()),
    }

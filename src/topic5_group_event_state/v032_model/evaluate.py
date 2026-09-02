"""Five evaluation arms on one shared anchor set (design §5).

    H                 log mu_H, TRAIN-fitted (or registry) NB dispersion
    H + S_correct     trained residual model
    H + S_shifted     same model, anchor state replaced by a same-segment donor
                      at least one horizon away (block-circular shift)
    H + mean(S_train) same model, anchor state replaced by the TRAIN-mean state
                      (a pure constant offset = intercept recalibration of H)
    H + S_random      random-reservoir model (frozen random encoder, trained adapter)

Contrasts are per-anchor paired NB-NLL differences in nats/anchor, positive when
the correct dynamic state is better, with a within-segment moving-block
bootstrap for the uncertainty.  ``effective_independent_windows`` is reported
next to every anchor count because 5-minute anchors overlap heavily at 30 min.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .data import SubjectBundle
from .model import ResidualStateModel
from .shift import block_circular_donor
from .trainer import anchor_terms, bundle_tensors, h_only_nll


def block_bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    block_len: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Moving-block bootstrap of a mean: blocks never cross a group (segment)."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    n = int(values.size)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0, "n_blocks": 0, "block_len": int(block_len),
                "n_boot": int(n_boot), "fraction_bootstrap_positive": float("nan")}
    block_sum: list[float] = []
    block_size: list[int] = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for start in range(0, idx.size, int(block_len)):
            member = idx[start : start + int(block_len)]
            block_sum.append(float(values[member].sum()))
            block_size.append(int(member.size))
    sums = np.asarray(block_sum)
    sizes = np.asarray(block_size, dtype=np.float64)
    n_blocks = sums.size
    rng = np.random.default_rng(int(seed))
    pick = rng.integers(0, n_blocks, size=(int(n_boot), n_blocks))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "se": float(means.std(ddof=1)) if n_boot > 1 else float("nan"),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": n,
        "n_blocks": int(n_blocks),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "fraction_bootstrap_positive": float(np.mean(means > 0)),
    }


def _segment_favourability(diff: np.ndarray, segments: np.ndarray) -> dict[str, int]:
    finite = np.isfinite(diff)
    positive = 0
    total = 0
    for g in np.unique(segments[finite]):
        member = finite & (segments == g)
        total += 1
        positive += int(diff[member].mean() > 0)
    return {"n_segments_favourable": positive, "n_segments": total}


def evaluate_arms(
    model: ResidualStateModel,
    bundle: SubjectBundle,
    cfg: ModelConfig,
    *,
    device: torch.device,
    phase: str,
    horizon: float,
    log_r_h: float,
    random_model: ResidualStateModel | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    t = tensors or bundle_tensors(bundle, device)
    model.eval()
    with torch.no_grad():
        correct = anchor_terms(model, bundle, phase=phase, horizon=horizon, device=device, tensors=t)
        idx = correct["idx"]
        n = int(idx.size)
        state = correct["state"]
        nll_correct = correct["nll"].cpu().numpy().astype(np.float64)
        nll_h = h_only_nll(bundle, phase=phase, horizon=horizon, log_r_h=log_r_h)
        mean_state = model.train_mean_state.to(state.dtype).unsqueeze(0).expand_as(state)
        mean_terms = anchor_terms(model, bundle, phase=phase, horizon=horizon, device=device,
                                  tensors=t, state_override=mean_state)
        nll_mean = mean_terms["nll"].cpu().numpy().astype(np.float64)
        shifted: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for fraction in cfg.shift_fractions:
            donor = block_circular_donor(bundle.t_anchor, bundle.anchor_segment, idx,
                                         horizon=horizon, fraction=float(fraction))
            ok = donor >= 0
            shifted_state = state.clone()
            if ok.any():
                src = torch.from_numpy(donor[ok]).to(device)
                dst = torch.from_numpy(np.flatnonzero(ok)).to(device)
                shifted_state[dst] = state[src]
            terms = anchor_terms(model, bundle, phase=phase, horizon=horizon, device=device,
                                 tensors=t, state_override=shifted_state)
            nll = terms["nll"].cpu().numpy().astype(np.float64)
            nll[~ok] = np.nan
            shifted[float(fraction)] = (nll, donor)
        nll_random = None
        if random_model is not None:
            random_model.eval()
            rnd = anchor_terms(random_model, bundle, phase=phase, horizon=horizon, device=device, tensors=t)
            nll_random = rnd["nll"].cpu().numpy().astype(np.float64)
    segments = bundle.anchor_segment[idx]
    primary_fraction = float(cfg.shift_fractions[0])
    nll_shifted, donor = shifted[primary_fraction]
    boot = dict(block_len=cfg.bootstrap_block_anchors, n_boot=cfg.bootstrap_resamples, seed=0)

    def contrast(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        diff = a - b
        out = block_bootstrap_mean_ci(diff, segments, **boot)
        out.update(_segment_favourability(diff, segments))
        return out

    modulation = correct["modulation"].cpu().numpy().astype(np.float64)
    contrasts = {
        "h_minus_correct": contrast(nll_h, nll_correct),
        "shifted_minus_correct": contrast(nll_shifted, nll_correct),
        "mean_minus_correct": contrast(nll_mean, nll_correct),
        "h_minus_mean": contrast(nll_h, nll_mean),
        "random_minus_correct": None if nll_random is None else contrast(nll_random, nll_correct),
        "h_minus_random": None if nll_random is None else contrast(nll_h, nll_random),
    }
    alternatives = {}
    for fraction, (nll_alt, donor_alt) in shifted.items():
        if fraction == primary_fraction:
            continue
        alternatives[f"{fraction:g}"] = {**contrast(nll_alt, nll_correct),
                                         "n_valid_donors": int((donor_alt >= 0).sum())}
    ok = donor >= 0
    return {
        "phase": phase,
        "horizon_seconds": float(horizon),
        "n_anchors": n,
        "effective_independent_windows": bundle.effective_independent_windows(phase, horizon),
        "arms": {
            "h": {"nll_mean": float(nll_h.mean()), "log_r": float(log_r_h), "n": n},
            "h_plus_s_correct": {
                "nll_mean": float(nll_correct.mean()),
                "modulation_rms": float(np.sqrt(np.mean(modulation ** 2))),
                "modulation_std": float(modulation.std()),
                "n": n,
            },
            "h_plus_s_shifted": {
                "fraction": primary_fraction,
                "nll_mean": float(np.nanmean(nll_shifted)) if ok.any() else float("nan"),
                "n_valid_donors": int(ok.sum()),
                "correct_nll_mean_on_same_anchors": float(nll_correct[ok].mean()) if ok.any() else float("nan"),
            },
            "h_plus_mean_s": {
                "nll_mean": float(nll_mean.mean()),
                "modulation_std": float(mean_terms["modulation"].cpu().numpy().std()),
                "modulation_value": float(mean_terms["modulation"].cpu().numpy().mean()),
                "n": n,
            },
            "h_plus_s_random": None if nll_random is None else {"nll_mean": float(nll_random.mean()), "n": n},
        },
        "contrasts": contrasts,
        "shift_alternatives": alternatives,
        "per_anchor": {
            "idx": idx.tolist(),
            "segment": segments.tolist(),
            "nll_h": nll_h.tolist(),
            "nll_correct": nll_correct.tolist(),
            "nll_shifted": [None if not np.isfinite(v) else float(v) for v in nll_shifted],
            "nll_mean": nll_mean.tolist(),
            "nll_random": None if nll_random is None else nll_random.tolist(),
            "donor": donor.tolist(),
            "modulation_correct": modulation.tolist(),
        },
        "units": "nats per anchor; positive contrast favours the correct dynamic state",
    }

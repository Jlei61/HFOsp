"""The five R0.1 core analyses (scientific spec section 8.2) -- and nothing else.

1. :func:`horizon_curve`        model vs each baseline at 1 / 5 / 10 / 100 min, with skill scores.
2. :func:`open_loop_trajectory` observed vs open-loop decoded field for a representative patient.
3. :func:`matched_state_swap`   substitute the state of a field-matched distant minute.
4. :func:`state_consistency`    E_cons per window and its per-patient quantiles.
5. :func:`mode_readout`         each 2-D mode's time constant, period and contact/frequency loading.

Spec section 8.3 forbids anything else this round. Patients are the unit of the
cohort statistics: minute windows inside one patient are not independent
biological samples, so per-window sign tests are reported as *within-patient
descriptives* and the cohort test runs over per-patient medians.

Reporting discipline (spec section 2.4): the forecast layer and the consistency
layer are separate statements. A positive horizon curve with a failing E_cons
supports "forecastable latent code", not "a single evolvable state".
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import contract
from .train import (apply_dynamics, call_filtered, encoder_inputs, model_forward,
                    move_batch)

#: The state-swap partner must be more than two hours away (spec section 8.2).
SWAP_MIN_SEPARATION_MINUTES = 120


# ---------------------------------------------------------------------------
# 1. Horizon curve
# ---------------------------------------------------------------------------

#: The five arms of spec section 8.1, in reporting order.
ARM_ORDER: Tuple[str, ...] = (
    "patient_mean", "persistence", "feature_ar", "identity_dynamics", "model",
)
BASELINE_ARMS: Tuple[str, ...] = ("patient_mean", "persistence", "feature_ar",
                                  "identity_dynamics")


def model_device(model) -> "torch.device":
    """Where the model's parameters live.

    Every post-hoc analysis reads a host-side latent cache and then pushes it
    back through the model's decoder and dynamics. Assuming CPU makes the whole
    analysis block work after a CPU smoke run and fail after a real GPU run,
    which is the worst possible place to discover it.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model with no parameters
        return torch.device("cpu")


def skill_score(mse_model: float, mse_baseline: float) -> float:
    """1 - MSE_model / MSE_baseline. Positive means the model is better."""
    if not (np.isfinite(mse_model) and np.isfinite(mse_baseline)) or mse_baseline <= 0:
        return float("nan")
    return 1.0 - float(mse_model) / float(mse_baseline)


def horizon_curve(per_subject: Dict[str, Dict[int, Dict[str, float]]]):
    """Long-format table: one row per (subject, horizon, arm) plus skill scores.

    ``per_subject[subject][horizon][arm] = mse``. Extra keys ``n_windows`` /
    ``n_elements`` inside the horizon dict are carried through so every row
    reports its own denominator (execution plan section 9: never collapse
    per-patient denominators into a single cohort n).
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for subject in sorted(per_subject):
        for horizon in sorted(per_subject[subject]):
            entry = per_subject[subject][horizon]
            model_mse = float(entry.get("model", float("nan")))
            for arm in ARM_ORDER:
                if arm not in entry:
                    continue
                rows.append({
                    "subject": subject,
                    "horizon_min": int(horizon),
                    "arm": arm,
                    "mse": float(entry[arm]),
                    "skill_vs_arm": (float("nan") if arm == "model"
                                     else skill_score(model_mse, float(entry[arm]))),
                    "n_windows": int(entry.get("n_windows", 0)),
                    "n_elements": int(entry.get("n_elements", 0)),
                })
    return pd.DataFrame(rows, columns=["subject", "horizon_min", "arm", "mse",
                                       "skill_vs_arm", "n_windows", "n_elements"])


# ---------------------------------------------------------------------------
# 2. Open-loop trajectory
# ---------------------------------------------------------------------------


def slice_observed_future(
    arrays: Dict[str, Any], t_index: int, horizons_out: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed normalised field at t+h for each requested h.

    Returns ``(h_grid, observed (H,C,F), observed_mask (H,C,F))``; horizons whose
    minute is absent from the cache are dropped from ``h_grid`` rather than
    filled, because an unrecorded minute is not a stable background (spec 4.3).
    """
    pos = {int(m): i for i, m in enumerate(arrays["minute_index"])}
    keep = [int(h) for h in horizons_out if int(t_index) + int(h) in pos]
    if not keep:
        shape = (0,) + tuple(arrays["field"].shape[1:])
        return np.zeros(0, dtype=int), np.zeros(shape), np.zeros(shape, dtype=bool)
    obs = np.stack([arrays["field"][pos[int(t_index) + h]] for h in keep])
    valid = np.stack([arrays["valid"][pos[int(t_index) + h]] for h in keep])
    mask = np.repeat(valid[:, :, None], obs.shape[2], axis=2)
    return np.asarray(keep, dtype=int), obs, mask


@torch.no_grad()
def open_loop_trajectory(
    model,
    window_batch: Dict[str, Any],
    observed: np.ndarray,
    observed_mask: np.ndarray,
    h_grid: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    autocast_ctx: Optional[Callable[[], Any]] = None,
) -> Dict[str, np.ndarray]:
    """Decode z_t forward over ``h_grid`` with no further input, vs the truth.

    ``window_batch`` is a single-window collated batch at time t. The encoder is
    run once; every horizon is then ``decode(B(h) z_t)``, i.e. the input really
    is cut off at t. Returns arrays only -- plotting lives in ``make_figures.py``.
    """
    device = torch.device(device)
    autocast_ctx = autocast_ctx or (lambda: contextlib.nullcontext())
    model.eval()
    batch = move_batch(window_batch, device)
    n_contacts, n_freq = int(observed.shape[1]), int(observed.shape[2])
    with autocast_ctx():
        z = call_filtered(model.encode, **encoder_inputs(batch))
        preds = [decode_field(model, apply_dynamics(model.dynamics, z, float(h)),
                              n_contacts, n_freq)[0] for h in h_grid]
    predicted = np.stack(preds) if preds else np.zeros_like(observed)
    persistence = np.asarray(batch["persistence"].float().detach().cpu().numpy()[0])
    m = np.asarray(observed_mask, dtype=bool)
    err_model, err_pers = [], []
    for i in range(predicted.shape[0]):
        sel = m[i]
        err_model.append(float(((predicted[i] - observed[i]) ** 2)[sel].mean()) if sel.any() else np.nan)
        err_pers.append(float(((persistence - observed[i]) ** 2)[sel].mean()) if sel.any() else np.nan)
    return {
        "h_grid": np.asarray(h_grid, dtype=int),
        "observed": np.asarray(observed),
        "observed_mask": m,
        "predicted": predicted,
        "persistence": persistence,
        "mse_open_loop": np.asarray(err_model, dtype=float),
        "mse_persistence": np.asarray(err_pers, dtype=float),
        "t_index": int(np.asarray(window_batch["t_index"]).reshape(-1)[0]),
        "latent": z.float().detach().cpu().numpy()[0],
    }


# ---------------------------------------------------------------------------
# Latent cache -- encode once, then every analysis is post-hoc
# ---------------------------------------------------------------------------


def decode_field(model, z: torch.Tensor, n_contacts: int, n_freq: int) -> np.ndarray:
    """``model.decode`` may return (B, C*F) or (B, C, F); normalise to (B, C, F)."""
    out = model.decode(z)
    arr = out.float().detach().cpu().numpy()
    return arr.reshape(arr.shape[0], n_contacts, n_freq)


@torch.no_grad()
def build_latent_cache(
    model, loader: Iterable[Dict[str, Any]], horizons: Sequence[int], *,
    device: torch.device | str = "cpu", subject: Optional[str] = None,
    split_name: str = "validation",
    autocast_ctx: Optional[Callable[[], Any]] = None,
    cfg=None,
) -> Dict[str, Any]:
    """Encode every scored validation minute exactly once.

    This single pass is what ``latent_trajectory.zarr`` stores, and every later
    analysis -- horizon curve, persistence, patient mean, consistency, matched
    state swap -- is decoded from it. Nothing downstream runs the encoder again,
    which is why the identical-window property is structural rather than a
    convention that has to be re-checked at each call site.
    """
    device = torch.device(device)
    autocast_ctx = autocast_ctx or (lambda: contextlib.nullcontext())
    model.eval()
    hs = [int(h) for h in horizons]
    z_all, f_all, v_all, t_all, e_all = [], [], [], [], []
    tgt = {h: [] for h in hs}
    msk = {h: [] for h in hs}
    for batch in loader:
        if cfg is not None:
            # the same eval-side arm transform the trainer applies, so the
            # latent cache is encoded from exactly the inputs the arm defines
            from .train import apply_arm_transform
            batch = apply_arm_transform(batch, cfg, training=False)
        batch = move_batch(batch, device)
        if subject is not None and "t_epoch" in batch:
            contract.assert_not_sealed(subject, batch["t_epoch"].detach().cpu().numpy())
        with autocast_ctx():
            z = call_filtered(model.encode, **encoder_inputs(batch))
        z_all.append(z.float().detach().cpu().numpy())
        f_all.append(batch["persistence"].float().detach().cpu().numpy())
        v_all.append(batch["minute_valid"][:, :, -1].bool().detach().cpu().numpy())
        t_all.append(batch["t_index"].detach().cpu().numpy().astype(int))
        e_all.append(batch["t_epoch"].detach().cpu().numpy().astype(float)
                     if "t_epoch" in batch else np.full(z.shape[0], np.nan))
        for h in hs:
            tgt[h].append((batch["target"][h] if h in batch["target"]
                           else batch["target"][int(h)]).float().detach().cpu().numpy())
            msk[h].append((batch["target_mask"][h] if h in batch["target_mask"]
                           else batch["target_mask"][int(h)]).bool().detach().cpu().numpy())
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
    field = cat(f_all, (0, 0, 0))
    return {
        "subject": subject, "split": split_name, "horizons": hs,
        "z": cat(z_all, (0, 0)),
        "field_now": field,
        "valid_now": cat(v_all, (0, 0)).astype(bool),
        "minute_index": cat(t_all, (0,)).astype(int),
        "t_epoch": cat(e_all, (0,)),
        "target": {h: cat(tgt[h], (0, 0, 0)) for h in hs},
        "target_mask": {h: cat(msk[h], (0, 0, 0)).astype(bool) for h in hs},
        "n_contacts": int(field.shape[1]) if field.ndim == 3 else 0,
        "n_freq": int(field.shape[2]) if field.ndim == 3 else 0,
    }


def save_latent_cache(cache: Dict[str, Any], path: Path) -> Path:
    """Write the cache as ``latent_trajectory.zarr``."""
    import zarr

    path = Path(path)
    store = zarr.open(str(path), mode="w")
    for key, dtype in (("z", "float32"), ("minute_index", "int64"),
                       ("t_epoch", "float64")):
        arr = np.asarray(cache[key])
        store.create_array(key, shape=arr.shape, dtype=dtype, overwrite=True)[:] = arr
    store.attrs["subject"] = cache.get("subject")
    store.attrs["split"] = cache.get("split")
    store.attrs["horizons"] = [int(h) for h in cache.get("horizons", [])]
    return path


def next_minute_pairs(minute_index: np.ndarray, t_epoch: np.ndarray,
                      tolerance_sec: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rows (i, j) where j is the very next minute of i, same session and split.

    Presence in the cache already implies the minute is usable and inside the
    same partition; the wall-clock check rejects a pair whose two minutes are
    not actually 60 s apart, which is what a session gap would look like.
    """
    minute_index = np.asarray(minute_index, dtype=int)
    t_epoch = np.asarray(t_epoch, dtype=float)
    pos = {int(m): i for i, m in enumerate(minute_index)}
    i_idx, j_idx = [], []
    for i, m in enumerate(minute_index):
        j = pos.get(int(m) + 1)
        if j is None:
            continue
        dt = t_epoch[j] - t_epoch[i]
        if np.isfinite(dt) and abs(dt - 60.0) > tolerance_sec:
            continue
        i_idx.append(i)
        j_idx.append(j)
    return np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int)


def consistency_from_cache(model, cache: Dict[str, Any], loss_bundle) -> Dict[str, Any]:
    """E_cons for every cached minute whose successor is also cached."""
    from .train import latent_diagnostics, ratio_parts, summarise_consistency

    i_idx, j_idx = next_minute_pairs(cache["minute_index"], cache["t_epoch"])
    if i_idx.size == 0:
        return {"pairs": (i_idx, j_idx),
                "summary": summarise_consistency([], [], [], cache["z"], [])}
    z = torch.as_tensor(np.asarray(cache["z"]), dtype=torch.float32,
                        device=model_device(model))
    z_now, z_next = z[i_idx], z[j_idx]
    z_pred = apply_dynamics(model.dynamics, z_now, 1.0)
    ratio, num, den = ratio_parts(loss_bundle, z_next, z_pred, z_now)
    diag = latent_diagnostics(loss_bundle, z_now, z_next)
    summary = summarise_consistency(np.atleast_1d(ratio).tolist(),
                                    np.atleast_1d(num).tolist(),
                                    np.atleast_1d(den).tolist(),
                                    cache["z"], [diag])
    return {"pairs": (i_idx, j_idx), "e_cons": np.atleast_1d(ratio),
            "residual_norm": np.atleast_1d(num), "step_norm": np.atleast_1d(den),
            "summary": summary}


@torch.no_grad()
def _score_arms(pred: np.ndarray, persistence: np.ndarray, target: np.ndarray,
                mask: np.ndarray, ids: np.ndarray) -> Dict[str, Any]:
    """Model / persistence / patient-mean MSE under one mask, on one window set."""
    n = int(mask.sum())
    keep = mask.reshape(mask.shape[0], -1).any(axis=1)
    window_ids = sorted(int(x) for x in ids[keep])
    # Per-band disaggregation. Not a new experiment -- the same masked MSE, split
    # along the frequency axis -- but it is needed to read the pooled number
    # honestly: the train stats show the per-contact standard deviation of log
    # power running ~0.45 in the lowest bands and ~0.07-0.13 in the highest, so
    # after unit normalisation the bands carry very different amounts of real
    # minute-to-minute movement. A pooled win could be a low-frequency win with
    # the high bands riding along on their own stability. Descriptive only; no
    # statistic is attached to it.
    def _per_band(p_):
        out = []
        for f in range(target.shape[-1]):
            mf = mask[..., f]
            nf = int(mf.sum())
            out.append(float((((p_[..., f] - target[..., f]) ** 2)[mf]).sum() / nf)
                       if nf else float("nan"))
        return out

    return {
        "model_mse": float((((pred - target) ** 2)[mask]).sum() / n) if n else float("nan"),
        "persistence_mse": float((((persistence - target) ** 2)[mask]).sum() / n) if n else float("nan"),
        "patient_mean_mse": float(((target ** 2)[mask]).sum() / n) if n else float("nan"),
        "model_mse_per_band": _per_band(pred) if n else [],
        "persistence_mse_per_band": _per_band(persistence) if n else [],
        "n_elements_per_band": [int(mask[..., f].sum()) for f in range(target.shape[-1])],
        "n_elements": n,
        "n_windows": len(window_ids),
        "model_window_ids": window_ids,
        "persistence_window_ids": window_ids,
        "patient_mean_window_ids": window_ids,
    }


def common_window_selector(cache: Dict[str, Any], horizons: Sequence[int]) -> np.ndarray:
    """Boolean over cached windows: scoreable at EVERY requested horizon.

    This is ``contract.EVAL_SET_PRIMARY``. Without it the horizon curve compares
    different window sets at different horizons, so a rise in error could mean
    "the far horizon is harder" or "the far horizon kept only the awkward
    windows", and the figure cannot tell you which.
    """
    sel = None
    for h in (int(x) for x in horizons):
        m = np.asarray(cache["target_mask"][h], dtype=bool)
        any_h = m.reshape(m.shape[0], -1).any(axis=1)
        sel = any_h if sel is None else (sel & any_h)
    return sel if sel is not None else np.zeros(0, dtype=bool)


def evaluate_from_cache(model, cache: Dict[str, Any], *, horizons: Sequence[int],
                        loss_bundle=None) -> Dict[str, Any]:
    """Model, persistence and patient-mean error from one encoder pass.

    Returns BOTH evaluation window sets (contract section 3):

    * ``contract.EVAL_SET_PRIMARY`` -- only windows scoreable at all four
      horizons. Every arm is scored on this identical index set, so the horizon
      curve measures horizon difficulty and nothing else. This is what Figure R2
      plots.
    * ``contract.EVAL_SET_SECONDARY`` -- each horizon on its own windows, with
      its own denominator. Four subjects (gaolan, litengsheng, songzishuo,
      sunyuanxin) have zero validation windows at h=100 because their validation
      span is under 110 minutes; without this set they would vanish from the
      h=1/5/10 results as well.

    ``per_horizon`` at the top level stays the secondary set for backward
    compatibility with the training-loop curve.

    All arms read the same cached targets under the same mask, so their window
    sets are identical by construction; ``baselines.align_windows`` still checks
    it, because the feature-AR baseline is produced by a separate script.
    """
    hs = [int(h) for h in horizons]
    n_contacts, n_freq = int(cache["n_contacts"]), int(cache["n_freq"])
    # The latent cache is a numpy array on the host, but the model (and its
    # dynamics parameters) stay on whatever device training left them on. Put z
    # on the model's device rather than assuming CPU: an all-post-hoc analysis
    # that silently only works after a CPU run is worse than one that fails.
    z = torch.as_tensor(np.asarray(cache["z"]), dtype=torch.float32,
                        device=model_device(model))
    ids = np.asarray(cache["minute_index"], dtype=int)
    common = common_window_selector(cache, hs)

    secondary: Dict[int, Dict[str, Any]] = {}
    primary: Dict[int, Dict[str, Any]] = {}
    for h in hs:
        target = np.asarray(cache["target"][h], dtype=np.float64)
        mask = np.asarray(cache["target_mask"][h], dtype=bool)
        pred = decode_field(model, apply_dynamics(model.dynamics, z, float(h)),
                            n_contacts, n_freq).astype(np.float64)
        persistence = np.asarray(cache["field_now"], dtype=np.float64)
        secondary[h] = _score_arms(pred, persistence, target, mask, ids)
        primary[h] = _score_arms(pred, persistence, target,
                                 mask & common[:, None, None], ids)

    def _mean_finite(d):
        vals = [v["model_mse"] for v in d.values() if math.isfinite(v["model_mse"])]
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    sec_loss, sec_n = _mean_finite(secondary)
    pri_loss, pri_n = _mean_finite(primary)
    out = {
        "per_horizon": secondary,
        "forecast_loss": sec_loss,
        "n_horizons_scored": sec_n,
        "n_windows_encoded": int(z.shape[0]),
        "n_windows_common": int(common.sum()),
        "eval_sets": {
            contract.EVAL_SET_PRIMARY: {
                "per_horizon": primary, "forecast_loss": pri_loss,
                "n_horizons_scored": pri_n, "n_windows": int(common.sum()),
                "common_window_ids": sorted(int(x) for x in ids[common]),
            },
            contract.EVAL_SET_SECONDARY: {
                "per_horizon": secondary, "forecast_loss": sec_loss,
                "n_horizons_scored": sec_n,
            },
        },
        "primary_set_empty": bool(common.sum() == 0),
    }
    if loss_bundle is not None:
        out["e_cons"] = consistency_from_cache(model, cache, loss_bundle)["summary"]
    return out


# ---------------------------------------------------------------------------
# 3. Matched state swap
# ---------------------------------------------------------------------------


def find_matched_partners(
    fields: np.ndarray,
    valid: np.ndarray,
    t_index: np.ndarray,
    split: Sequence[str],
    *,
    min_separation_minutes: int = SWAP_MIN_SEPARATION_MINUTES,
) -> Dict[str, np.ndarray]:
    """Nearest neighbour in normalised field space, >2 h away, same split.

    ``fields`` (N,C,F) is the observed normalised field at the last context
    minute; ``valid`` (N,C) is contact validity there. Distance is the RMS
    difference over contacts valid in *both* windows, so windows with different
    artifact patterns are still comparable.

    Returns ``partner`` (-1 where no eligible partner exists), the matched
    distance, its percentile rank among that window's eligible candidates, and
    the ratio to the median eligible distance -- the last two are what tells a
    reader whether "matched" really matched.
    """
    fields = np.asarray(fields, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    t_index = np.asarray(t_index, dtype=int)
    split = np.asarray(split)
    n = fields.shape[0]
    partner = np.full(n, -1, dtype=int)
    distance = np.full(n, np.nan)
    pct = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    n_cand = np.zeros(n, dtype=int)
    if n == 0:
        return {"partner": partner, "distance": distance, "percentile": pct,
                "ratio_to_median": ratio, "n_candidates": n_cand}
    flat = fields.reshape(n, fields.shape[1], -1)
    for i in range(n):
        eligible = (np.abs(t_index - t_index[i]) > int(min_separation_minutes)) & (split == split[i])
        eligible[i] = False
        idx = np.flatnonzero(eligible)
        n_cand[i] = idx.size
        if idx.size == 0:
            continue
        both = valid[i][None, :] & valid[idx]                       # (K, C)
        diff = (flat[idx] - flat[i][None]) ** 2                     # (K, C, F)
        num = (diff * both[:, :, None]).sum(axis=(1, 2))
        den = both.sum(axis=1) * flat.shape[2]
        d = np.divide(num, den, out=np.full(num.shape, np.inf), where=den > 0)
        d = np.sqrt(d)
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            continue
        partner[i] = int(idx[j])
        distance[i] = float(d[j])
        finite = d[np.isfinite(d)]
        pct[i] = float((finite <= d[j]).mean() * 100.0)
        med = float(np.median(finite))
        ratio[i] = float(d[j] / med) if med > 0 else np.nan
    return {"partner": partner, "distance": distance, "percentile": pct,
            "ratio_to_median": ratio, "n_candidates": n_cand}


@torch.no_grad()
def matched_state_swap(
    model, cache: Dict[str, Any], *, horizons: Sequence[int],
    min_separation_minutes: int = SWAP_MIN_SEPARATION_MINUTES,
) -> Dict[str, Any]:
    """Swap in the state of a field-matched, >2 h distant minute and re-decode.

    Runs entirely off the latent cache: no second encoder pass, so the swapped
    and the true prediction differ in nothing but which row of z was fed to the
    same dynamics and the same decoder.

    Positive extra error means the future got worse when the state came from a
    minute whose *current* field looks the same -- i.e. the state carries
    information beyond the current snapshot. The per-window sign test is a
    within-patient descriptive; the cohort test is over per-patient medians
    (see :func:`cohort_summary_from_rows`).
    """
    from scipy import stats as sps

    hs = [int(h) for h in horizons]
    subject = cache.get("subject")
    split_name = cache.get("split", "validation")
    t_index = np.asarray(cache["minute_index"], dtype=int)
    n = int(t_index.size)
    n_contacts, n_freq = int(cache["n_contacts"]), int(cache["n_freq"])
    split = np.array([split_name] * n)
    match = find_matched_partners(cache["field_now"], cache["valid_now"], t_index,
                                  split, min_separation_minutes=min_separation_minutes)

    z = torch.as_tensor(np.asarray(cache["z"]), dtype=torch.float32,
                        device=model_device(model))
    rows: List[Dict[str, Any]] = []
    per_horizon: Dict[int, Dict[str, Any]] = {}
    ok = np.flatnonzero(match["partner"] >= 0)
    for h in hs:
        target = np.asarray(cache["target"][h], dtype=np.float64)
        mask = np.asarray(cache["target_mask"][h], dtype=bool)
        pred_true = decode_field(model, apply_dynamics(model.dynamics, z, float(h)),
                                 n_contacts, n_freq).astype(np.float64)
        pred_swap = pred_true[match["partner"]]
        d_list, mse_true_list, mse_swap_list = [], [], []
        for i in ok:
            sel = mask[i]
            if not sel.any():
                continue
            mse_t = float(((pred_true[i] - target[i]) ** 2)[sel].mean())
            mse_s = float(((pred_swap[i] - target[i]) ** 2)[sel].mean())
            d_list.append(mse_s - mse_t)
            mse_true_list.append(mse_t)
            mse_swap_list.append(mse_s)
            rows.append({
                "subject": subject, "split": split_name, "horizon_min": int(h),
                "t_index": int(t_index[i]),
                "partner_t_index": int(t_index[match["partner"][i]]),
                "separation_minutes": int(abs(t_index[i] - t_index[match["partner"][i]])),
                "match_distance": float(match["distance"][i]),
                "match_distance_percentile": float(match["percentile"][i]),
                "match_distance_ratio_to_median": float(match["ratio_to_median"][i]),
                "n_candidates": int(match["n_candidates"][i]),
                "mse_true_state": mse_t, "mse_swapped_state": mse_s,
                "dmse": mse_s - mse_t,
            })
        d = np.asarray(d_list, dtype=float)
        n_pos = int((d > 0).sum())
        n_eff = int((d != 0).sum())
        p = float(sps.binomtest(n_pos, n_eff, 0.5).pvalue) if n_eff else float("nan")
        per_horizon[int(h)] = {
            "n_windows": int(d.size),
            "median_dmse": float(np.median(d)) if d.size else float("nan"),
            "mean_dmse": float(np.mean(d)) if d.size else float("nan"),
            "frac_positive": float(n_pos / n_eff) if n_eff else float("nan"),
            "sign_test_p_windows": p,
            "median_mse_true_state": float(np.median(mse_true_list)) if mse_true_list else float("nan"),
            "median_mse_swapped_state": float(np.median(mse_swap_list)) if mse_swap_list else float("nan"),
        }
    return {
        "subject": subject,
        "split": split_name,
        "min_separation_minutes": int(min_separation_minutes),
        "n_windows_total": n,
        "n_windows_with_partner": int(ok.size),
        "match_quality": {
            "median_distance": float(np.nanmedian(match["distance"])) if n else float("nan"),
            "median_distance_ratio_to_median": float(np.nanmedian(match["ratio_to_median"])) if n else float("nan"),
            "median_distance_percentile": float(np.nanmedian(match["percentile"])) if n else float("nan"),
            "median_separation_minutes": float(np.median(
                np.abs(t_index[ok] - t_index[match["partner"][ok]]))) if ok.size else float("nan"),
        },
        "per_horizon": per_horizon,
        "rows": rows,
        "note": ("per-window sign test is a within-patient descriptive; minute "
                 "windows are not independent biological samples"),
    }


# ---------------------------------------------------------------------------
# 4. State consistency
# ---------------------------------------------------------------------------


def state_consistency(
    model, cache: Dict[str, Any], *, loss_bundle,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Per-window E_cons plus the per-patient quantiles required by the spec.

    Reported alongside E_cons, never on its own: the residual norm, the step
    norm, the spread of the latent cloud and the number of active latent
    dimensions. A collapsed state makes E_cons small for an uninteresting
    reason, and the output has to make that visible rather than leave it to be
    inferred.
    """
    import pandas as pd

    subject = cache.get("subject")
    split_name = cache.get("split", "validation")
    result = consistency_from_cache(model, cache, loss_bundle)
    i_idx, j_idx = result["pairs"]
    t_index = np.asarray(cache["minute_index"], dtype=int)
    t_epoch = np.asarray(cache["t_epoch"], dtype=float)
    frame = pd.DataFrame({
        "subject": subject, "split": split_name,
        "t_index": t_index[i_idx] if i_idx.size else np.zeros(0, dtype=int),
        "t_epoch": t_epoch[i_idx] if i_idx.size else np.zeros(0),
        "next_t_index": t_index[j_idx] if j_idx.size else np.zeros(0, dtype=int),
        "e_cons": result.get("e_cons", np.zeros(0)),
        "residual_norm": result.get("residual_norm", np.zeros(0)),
        "step_norm": result.get("step_norm", np.zeros(0)),
    }, columns=["subject", "split", "t_index", "t_epoch", "next_t_index",
                "e_cons", "residual_norm", "step_norm"])
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(out_path)
    summary = dict(result["summary"])
    summary.update({"subject": subject, "split": split_name,
                    "n_windows": int(i_idx.size),
                    "n_windows_encoded": int(t_index.size),
                    "path": str(out_path) if out_path is not None else None})
    return {"frame": frame, "summary": summary}


# ---------------------------------------------------------------------------
# 5. Latent mode readout
# ---------------------------------------------------------------------------


def _normalise_mode_description(described: Any) -> List[Dict[str, float]]:
    if isinstance(described, dict):
        taus = np.atleast_1d(np.asarray(
            described.get("tau_minutes", described.get("tau", []))), )
        omegas = np.atleast_1d(np.asarray(
            described.get("omega_rad_per_min", described.get("omega", []))))
        return [{"tau_minutes": float(t), "omega_rad_per_min": float(w)}
                for t, w in zip(taus, omegas)]
    out = []
    for entry in described:
        out.append({
            "tau_minutes": float(entry.get("tau_minutes", entry.get("tau"))),
            "omega_rad_per_min": float(entry.get("omega_rad_per_min", entry.get("omega"))),
        })
    return out


def mode_readout(
    dynamics: Any, decoder_weight: np.ndarray, n_contacts: int,
    n_freq: int = contract.N_FREQ_BINS,
) -> Dict[str, Any]:
    """Per 2-D mode: time constant, rotation period, and its contact/frequency map.

    ``decoder_weight`` is the (C*F, latent) linear decoder matrix. Mode ``j``
    occupies latent dims ``2j`` and ``2j+1``; its loading is the Euclidean norm
    of those two decoder columns, reshaped to (C, F). The loading says where in
    the contact-by-frequency field the mode writes -- it does NOT say the mode is
    excitatory, inhibitory, or epileptogenic (spec section 2.2).
    """
    modes = _normalise_mode_description(dynamics.describe_modes())
    w = np.asarray(decoder_weight, dtype=np.float64)
    if w.shape[0] != n_contacts * n_freq:
        raise ValueError(
            f"decoder weight has {w.shape[0]} output rows, expected "
            f"{n_contacts}*{n_freq}={n_contacts * n_freq}"
        )
    n_modes = len(modes)
    if w.shape[1] < 2 * n_modes:
        raise ValueError(
            f"decoder weight has {w.shape[1]} latent columns, need {2 * n_modes}"
        )
    loading = np.zeros((n_modes, n_contacts, n_freq), dtype=np.float64)
    records: List[Dict[str, Any]] = []
    for j, mode in enumerate(modes):
        cols = w[:, 2 * j:2 * j + 2]
        norm = np.sqrt((cols ** 2).sum(axis=1)).reshape(n_contacts, n_freq)
        loading[j] = norm
        omega = float(mode["omega_rad_per_min"])
        tau = float(mode["tau_minutes"])
        records.append({
            "mode": j,
            "latent_dims": [2 * j, 2 * j + 1],
            "tau_minutes": tau,
            "half_life_minutes": tau * math.log(2.0),
            "omega_rad_per_min": omega,
            "period_minutes": (2.0 * math.pi / abs(omega)) if abs(omega) > 1e-12 else float("inf"),
            "decay_at_1min": math.exp(-1.0 / tau) if tau > 0 else float("nan"),
            "decay_at_100min": math.exp(-100.0 / tau) if tau > 0 else float("nan"),
            "loading_total": float(norm.sum()),
            "loading_peak_contact": int(np.argmax(norm.sum(axis=1))),
            "loading_peak_freq_bin": int(np.argmax(norm.sum(axis=0))),
        })
    return {"modes": records, "loading": loading,
            "n_modes": n_modes, "n_contacts": int(n_contacts), "n_freq": int(n_freq),
            "freq_edges_hz": contract.FREQ_EDGES.tolist()}


# ---------------------------------------------------------------------------
# Cohort statistics -- patients are the unit
# ---------------------------------------------------------------------------


def cohort_statistic(values: Sequence[float], null_value: float = 0.0) -> Dict[str, Any]:
    """Median / IQR / exact sign test over the supplied per-patient values.

    Each element must already be one patient's summary. Nothing in this function
    knows how many windows a patient contributed, which is exactly the point:
    a patient with ten times more minutes must not get ten times the weight.
    """
    from scipy import stats as sps

    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n_subjects": 0, "median": float("nan"), "q25": float("nan"),
                "q75": float("nan"), "n_above_null": 0, "n_nonzero": 0,
                "sign_test_p": float("nan")}
    diff = arr - float(null_value)
    n_pos = int((diff > 0).sum())
    n_eff = int((diff != 0).sum())
    p = float(sps.binomtest(n_pos, n_eff, 0.5).pvalue) if n_eff else float("nan")
    return {
        "n_subjects": n,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "n_above_null": n_pos,
        "n_nonzero": n_eff,
        "sign_test_p": p,
    }


def cohort_summary_from_rows(
    rows: Sequence[Dict[str, Any]], value_key: str,
    group_keys: Sequence[str] = ("horizon_min",), null_value: float = 0.0,
) -> Dict[str, Any]:
    """Group per-patient rows and summarise each group with patients as the unit.

    ``rows`` must contain at most one row per (subject, group). A duplicate
    subject inside a group is an error, not something to average away: it would
    silently reweight the cohort statistic.
    """
    grouped: Dict[Tuple, Dict[str, float]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        subject = row["subject"]
        bucket = grouped.setdefault(key, {})
        if subject in bucket:
            raise ValueError(
                f"subject {subject!r} appears twice for group {key}; the cohort "
                "statistic weights patients equally and cannot absorb duplicates"
            )
        bucket[subject] = float(row[value_key])
    out: Dict[str, Any] = {}
    for key in sorted(grouped, key=lambda k: tuple(str(x) for x in k)):
        bucket = grouped[key]
        stat = cohort_statistic([bucket[s] for s in sorted(bucket)], null_value=null_value)
        stat["subjects"] = sorted(bucket)
        out["|".join(str(x) for x in key)] = stat
    return out

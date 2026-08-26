"""The five R0.1 baselines (scientific spec section 8.1) -- and nothing else.

1. ``patient_mean_baseline``      predict the train mean, i.e. 0 in normalised space.
2. ``persistence_baseline``       predict the observed field at the last context minute.
3. ``spectral_feature_ar_baseline`` low-capacity ridge on the 10-minute context spectra.
4. ``identity_dynamics_arm_config`` config factory only -- the arm is the full model
   trained with ``dynamics.identity_mode=True`` (same encoder capacity, only B(h) swapped).
5. the full model, trained by :mod:`train`.

Spec section 8.3 forbids adding anything else this round (no time shuffles, no
patient swaps, no coordinate nulls, no nuisance removal, no sleep/day-night
stratification, no capacity sweeps). Do not extend this module.

Every baseline is scored on **exactly** the validation windows the model is
scored on. ``align_windows`` is the enforcement point: it raises rather than
silently intersecting, because a silent intersection would change the
denominator of the headline comparison without anyone noticing.
"""

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import contract
from .train import TrainConfig, move_batch, resolve_arm, sequential_loader

#: Fixed ridge grid, spec section 8.1 baseline 3.
RIDGE_ALPHA_GRID: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RIDGE_N_FOLDS = 5


# ---------------------------------------------------------------------------
# Window alignment
# ---------------------------------------------------------------------------


def align_windows(sources: Dict[str, Dict[int, Sequence[int]]]) -> Dict[int, List[int]]:
    """Assert that every source scored the same windows; return that window set.

    ``sources`` maps a label ("model", "persistence", "feature_ar", ...) to a
    per-horizon list of window ids. Raises on any disagreement -- a mismatched
    denominator is invalidation condition 7 of the spec, not a rounding detail.
    """
    if not sources:
        raise ValueError("align_windows needs at least one source")
    labels = sorted(sources)
    horizons = sorted(sources[labels[0]])
    for label in labels[1:]:
        if sorted(sources[label]) != horizons:
            raise ValueError(
                f"horizon sets differ: {labels[0]}={horizons} vs {label}={sorted(sources[label])}"
            )
    out: Dict[int, List[int]] = {}
    for h in horizons:
        ref = sorted(int(x) for x in sources[labels[0]][h])
        for label in labels[1:]:
            other = sorted(int(x) for x in sources[label][h])
            if other != ref:
                only_ref = sorted(set(ref) - set(other))[:8]
                only_other = sorted(set(other) - set(ref))[:8]
                raise ValueError(
                    f"window sets differ at horizon {h}: {labels[0]} has {len(ref)}, "
                    f"{label} has {len(other)}; only in {labels[0]}: {only_ref}; "
                    f"only in {label}: {only_other}"
                )
        out[h] = ref
    return out


# ---------------------------------------------------------------------------
# Baseline 1 and 2 -- trivially cheap, computed straight off the eval loader
# ---------------------------------------------------------------------------


def _accumulate(loader: Iterable[Dict[str, Any]], horizons: Sequence[int],
                predict, restrict_ids: Optional[Sequence[int]] = None
                ) -> Dict[int, Dict[str, Any]]:
    """``restrict_ids`` limits scoring to those minute indices.

    Pass ``contract.EVAL_SET_PRIMARY``'s common window ids to score a baseline on
    exactly the windows the model was scored on. Restriction is applied to the
    MASK, not to the batch, so a partially-in-set batch contributes only its
    in-set rows and the element counts stay exact.
    """
    hs = [int(h) for h in horizons]
    keep_ids = None if restrict_ids is None else set(int(x) for x in restrict_ids)
    acc = {h: {"sse": 0.0, "n_elements": 0, "window_ids": []} for h in hs}
    for batch in loader:
        t_index = np.asarray(batch["t_index"]).astype(int)
        in_set = (None if keep_ids is None
                  else torch.as_tensor(np.array([int(i) in keep_ids for i in t_index])))
        for h in hs:
            tgt = torch.as_tensor(batch["target"][h]).double()
            msk = torch.as_tensor(batch["target_mask"][h]).bool()
            if in_set is not None:
                msk = msk & in_set.reshape((-1,) + (1,) * (msk.dim() - 1))
            if not bool(msk.any()):
                continue
            m = msk.double()
            pred = predict(batch, h).double()
            acc[h]["sse"] += float((((pred - tgt) ** 2) * m).sum())
            acc[h]["n_elements"] += int(m.sum())
            keep = msk.reshape(msk.shape[0], -1).any(dim=1).numpy()
            acc[h]["window_ids"].extend(t_index[keep].tolist())
    return {
        h: {"mse": (a["sse"] / a["n_elements"]) if a["n_elements"] else float("nan"),
            "n_elements": int(a["n_elements"]),
            "n_windows": int(len(a["window_ids"])),
            "window_ids": sorted(a["window_ids"])}
        for h, a in acc.items()
    }


def patient_mean_baseline(loader: Iterable[Dict[str, Any]],
                          horizons: Sequence[int],
                          restrict_ids: Optional[Sequence[int]] = None
                          ) -> Dict[int, Dict[str, Any]]:
    """Predict the train mean. In normalised space that is exactly zero.

    On the train split the normalised MSE is 1.0 by construction; the value
    reported on validation is therefore directly readable as "fraction of the
    patient's own train variance still unexplained".
    """
    return _accumulate(loader, horizons,
                       lambda batch, h: torch.zeros_like(torch.as_tensor(batch["target"][h])),
                       restrict_ids=restrict_ids)


def persistence_baseline(loader: Iterable[Dict[str, Any]],
                         horizons: Sequence[int],
                         restrict_ids: Optional[Sequence[int]] = None
                         ) -> Dict[int, Dict[str, Any]]:
    """Predict the observed normalised field at the last context minute."""
    return _accumulate(loader, horizons,
                       lambda batch, h: torch.as_tensor(batch["persistence"]),
                       restrict_ids=restrict_ids)


def assert_patient_mean_is_unit_on_train(train_metrics: Dict[int, Dict[str, Any]],
                                         tol: float = 0.35) -> None:
    """Sanity gate: the targets are standardised, so the train MSE must be ~1.

    The tolerance is deliberately loose. ``target_mean`` / ``target_std`` are
    estimated over the artifact-clean TRAIN contact-minutes, so the unit-variance
    property is exact only on that population. This baseline is scored on a
    different one -- the target minutes of eligible training WINDOWS, which is a
    subset shifted by the context and horizon requirements -- and lands near 1.1
    rather than exactly 1.0. That gap is real and expected.

    What the gate is for is the failure mode that actually happened: before the
    artifact-aware second standardisation pass, the denominator was inflated by
    the artifact tail (1.35 % of contact-minutes carrying 87 % of the variance
    on epilepsiae_620) and this number came out at **0.13**, a factor of eight.
    A 35 % band catches that and every failure of its size, without failing on
    the legitimate population difference.
    """
    bad = {h: v["mse"] for h, v in train_metrics.items()
           if np.isfinite(v["mse"]) and abs(v["mse"] - 1.0) > tol}
    if bad:
        raise AssertionError(
            "patient-mean baseline is far from 1.0 on train -- targets are not "
            f"standardised on the train-only clean population: {bad}"
        )


# ---------------------------------------------------------------------------
# Baseline 3 -- low-capacity spectral feature AR
# ---------------------------------------------------------------------------


@dataclass
class FeatureArDesign:
    """Design matrix for one (horizon, frequency-bin) ridge problem."""

    x: np.ndarray            # (n_rows, 20) own 10 context minutes + across-contact mean
    y: np.ndarray            # (n_rows,)
    window_pos: np.ndarray   # (n_rows,) index into the window axis, for grouped CV
    window_id: np.ndarray    # (n_rows,) minute index of the window
    contact: np.ndarray      # (n_rows,)
    imputed: np.ndarray      # (n_rows,) bool -- own context had to be filled with 0


def build_feature_ar_design(
    context: np.ndarray,
    context_valid: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    window_id: np.ndarray,
    freq_bin: int,
    *,
    drop_incomplete_context: bool,
) -> FeatureArDesign:
    """Assemble the 20 features + label rows for one frequency bin.

    ``context``       (N, C, 10, F) normalised log power of the 10 context minutes
    ``context_valid`` (N, C, 10)    per contact-minute artifact mask
    ``target``        (N, C, F)     normalised log power at t+h
    ``target_mask``   (N, C, F)     which target entries are scoreable

    Features per row = the contact's own 10 context values at this frequency
    plus the across-contact mean at the same frequency over the same 10 minutes.
    The intercept is fitted by the estimator, so it is not a column here.
    ``drop_incomplete_context=True`` (fitting) drops rows whose own 10-minute
    context is not fully valid; ``False`` (prediction) keeps them and fills the
    missing minutes with 0, which is the train mean, and flags them as imputed.
    """
    n_win, n_ch, n_ctx = context_valid.shape
    own = context[:, :, :, freq_bin]                            # (N, C, 10)
    valid = context_valid.astype(bool)
    w = valid.astype(np.float64)
    denom = w.sum(axis=1)                                       # (N, 10)
    across = np.divide((own * w).sum(axis=1), denom,
                       out=np.zeros_like(denom), where=denom > 0)   # (N, 10)
    own_filled = np.where(valid, own, 0.0)
    full = valid.all(axis=2)                                    # (N, C)
    y = target[:, :, freq_bin]
    keep = target_mask[:, :, freq_bin].astype(bool) & np.isfinite(y)
    if drop_incomplete_context:
        keep &= full
    wi, ci = np.nonzero(keep)
    x = np.concatenate(
        [own_filled[wi, ci, :], across[wi, :]], axis=1
    ).astype(np.float64)
    return FeatureArDesign(
        x=x, y=y[wi, ci].astype(np.float64), window_pos=wi.astype(int),
        window_id=np.asarray(window_id)[wi].astype(int), contact=ci.astype(int),
        imputed=~full[wi, ci],
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """Ridge with unpenalised intercept, closed form, deterministic."""
    xm = x.mean(axis=0)
    ym = float(y.mean())
    xc = x - xm
    gram = xc.T @ xc + float(alpha) * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, xc.T @ (y - ym))
    return coef, ym - float(xm @ coef)


def _select_alpha(design: FeatureArDesign, alphas: Sequence[float],
                  n_folds: int) -> Tuple[float, Dict[float, float]]:
    """K-fold CV *inside train only*, folds cut on the window axis.

    Folds are contiguous blocks of windows (no shuffle): rows from one window
    share the across-contact mean feature, so splitting rows instead of windows
    would leak that feature across the fold boundary.
    """
    windows = np.unique(design.window_pos)
    n_folds = int(min(max(2, n_folds), len(windows))) if len(windows) >= 2 else 1
    scores: Dict[float, float] = {}
    if n_folds < 2:
        return float(alphas[0]), {float(a): float("nan") for a in alphas}
    bounds = np.array_split(windows, n_folds)
    for alpha in alphas:
        sse, n = 0.0, 0
        for held in bounds:
            te = np.isin(design.window_pos, held)
            tr = ~te
            if tr.sum() < design.x.shape[1] + 1 or te.sum() == 0:
                continue
            coef, b = _ridge_fit(design.x[tr], design.y[tr], alpha)
            resid = design.y[te] - (design.x[te] @ coef + b)
            sse += float((resid ** 2).sum())
            n += int(te.sum())
        scores[float(alpha)] = (sse / n) if n else float("inf")
    best = min(scores, key=lambda a: (scores[a], a))
    return float(best), scores


def spectral_feature_ar_baseline(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    horizons: Sequence[int],
    *,
    alphas: Sequence[float] = RIDGE_ALPHA_GRID,
    n_folds: int = RIDGE_N_FOLDS,
    restrict_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Per subject, per horizon, per frequency bin: one ridge shared over contacts.

    ``train`` / ``val`` are dicts with ``context (N,C,10,F)``,
    ``context_valid (N,C,10)``, ``window_id (N,)`` and per-horizon
    ``target[h] (N,C,F)`` / ``target_mask[h] (N,C,F)``.

    Fitting uses TRAIN rows only, and alpha is chosen by K-fold CV inside train
    only -- validation never touches model selection. Prediction covers every
    scoreable validation element (missing own-context minutes filled with the
    train mean 0) so the metric shares the model's denominator; the
    fully-valid-context subset is reported alongside as
    ``mse_full_context_only``.

    ``restrict_ids`` limits SCORING (never fitting) to those validation window
    ids, so the primary ``common_all_horizons`` set can be evaluated without
    refitting the ridge on a different training pool.
    """
    hs = [int(h) for h in horizons]
    keep_ids = None if restrict_ids is None else set(int(x) for x in restrict_ids)
    n_freq = int(train["context"].shape[3])
    out: Dict[str, Any] = {"per_horizon": {}, "alpha": {}, "cv_scores": {},
                           "coef": {}, "intercept": {}}
    for h in hs:
        alpha_by_bin: List[float] = []
        cv_by_bin: List[Dict[float, float]] = []
        coef_by_bin = np.zeros((n_freq, 20), dtype=np.float64)
        icpt_by_bin = np.zeros(n_freq, dtype=np.float64)
        sse = sse_full = 0.0
        n_elem = n_full = 0
        n_imputed = 0
        window_ids: List[int] = []
        for f in range(n_freq):
            tr = build_feature_ar_design(
                train["context"], train["context_valid"], train["target"][h],
                train["target_mask"][h], train["window_id"], f,
                drop_incomplete_context=True,
            )
            if tr.x.shape[0] <= tr.x.shape[1] + 1:
                alpha_by_bin.append(float("nan"))
                cv_by_bin.append({})
                continue
            alpha, scores = _select_alpha(tr, alphas, n_folds)
            coef, icpt = _ridge_fit(tr.x, tr.y, alpha)
            alpha_by_bin.append(alpha)
            cv_by_bin.append(scores)
            coef_by_bin[f] = coef
            icpt_by_bin[f] = icpt

            va = build_feature_ar_design(
                val["context"], val["context_valid"], val["target"][h],
                val["target_mask"][h], val["window_id"], f,
                drop_incomplete_context=False,
            )
            if va.x.shape[0] == 0:
                continue
            if keep_ids is not None:
                sel = np.array([int(w) in keep_ids for w in va.window_id], dtype=bool)
                if not sel.any():
                    continue
                va = dataclasses.replace(
                    va, x=va.x[sel], y=va.y[sel], window_pos=va.window_pos[sel],
                    window_id=va.window_id[sel], contact=va.contact[sel],
                    imputed=va.imputed[sel])
            resid = va.y - (va.x @ coef + icpt)
            sse += float((resid ** 2).sum())
            n_elem += int(resid.size)
            n_imputed += int(va.imputed.sum())
            full = ~va.imputed
            sse_full += float((resid[full] ** 2).sum())
            n_full += int(full.sum())
            window_ids.extend(np.unique(va.window_id).tolist())
        ids = sorted(set(int(x) for x in window_ids))
        out["per_horizon"][h] = {
            "mse": (sse / n_elem) if n_elem else float("nan"),
            "mse_full_context_only": (sse_full / n_full) if n_full else float("nan"),
            "n_elements": int(n_elem),
            "n_windows": len(ids),
            "window_ids": ids,
            "frac_context_imputed": (n_imputed / n_elem) if n_elem else float("nan"),
        }
        out["alpha"][h] = alpha_by_bin
        out["cv_scores"][h] = cv_by_bin
        out["coef"][h] = coef_by_bin
        out["intercept"][h] = icpt_by_bin
    return out


# ---------------------------------------------------------------------------
# Baseline 4 -- config factory only
# ---------------------------------------------------------------------------


def identity_dynamics_arm_config(base: TrainConfig) -> TrainConfig:
    """Baseline 4 is not a separate estimator: it is the full model with B(h)=I.

    Encoder capacity, horizons, loss weights and schedule are inherited from the
    full-model config verbatim, per spec section 10 ("只换 B(h)").
    """
    return replace(base, arm="identity", identity_dynamics=True)


# ---------------------------------------------------------------------------
# Adapter: load the normalised minute spectra the feature-AR baseline needs
# ---------------------------------------------------------------------------


def load_subject_spectral_arrays(
    subject: str,
    *,
    spectral_path: Optional[Path] = None,
    window_index_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read Worker B's minute spectra + window index and normalise with train stats.

    Returns ``{"field": (M,C,F) normalised, "valid": (M,C) bool,
    "minute_index": (M,), "split": (M,), "minute_epoch": (M,)}``. This is the
    single adapter that touches Worker B's on-disk layout; if that layout moves,
    only this function changes.
    """
    import pandas as pd
    import zarr

    spectral_path = Path(spectral_path or contract.spectral_target_path(subject))
    window_index_path = Path(window_index_path or (contract.DATA_DIR / "window_index.parquet"))
    stats_path = Path(stats_path or contract.subject_stats_path(subject))

    field = np.asarray(zarr.open(str(spectral_path), mode="r")[:], dtype=np.float64)
    idx = pd.read_parquet(window_index_path)
    idx = idx[idx["subject"] == subject].sort_values("minute_index").reset_index(drop=True)
    stats = json.loads(stats_path.read_text())
    mean, std = _extract_train_stats(stats, field.shape[1:])
    normalised = (field - mean[None]) / np.where(std[None] > 0, std[None], 1.0)

    minute_index = idx["minute_index"].to_numpy().astype(int)
    if field.shape[0] != minute_index.size:
        raise ValueError(
            f"{subject}: spectral cache has {field.shape[0]} minutes but the window "
            f"index has {minute_index.size} rows -- channel/minute alignment is "
            "invalidation condition 4"
        )
    contract.assert_not_sealed(subject, idx["minute_start_epoch"].to_numpy())
    valid = np.isfinite(normalised).all(axis=2)
    return {
        "field": normalised,
        "valid": valid,
        "minute_index": minute_index,
        "minute_epoch": idx["minute_start_epoch"].to_numpy(dtype=float),
        "split": idx["split"].to_numpy(),
        "minute_usable": idx["minute_usable"].to_numpy().astype(bool),
        "ctx_ok": idx["ctx_ok"].to_numpy().astype(bool),
        "horizon_ok": {int(h): idx[f"h{int(h)}_ok"].to_numpy().astype(bool)
                       for h in contract.HORIZONS_MIN if f"h{int(h)}_ok" in idx.columns},
    }


def _extract_train_stats(stats: Dict[str, Any], shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    for mk, sk in (("target_mean", "target_std"), ("mean", "std"),
                   ("spectral_mean", "spectral_std")):
        if mk in stats and sk in stats:
            return (np.asarray(stats[mk], dtype=np.float64).reshape(shape),
                    np.asarray(stats[sk], dtype=np.float64).reshape(shape))
    raise KeyError(
        f"train_stats.json has no recognised mean/std pair; keys = {sorted(stats)}"
    )


def assemble_feature_ar_inputs(
    arrays: Dict[str, Any], split: str, horizons: Sequence[int],
    *, context_minutes: int = contract.CONTEXT_MINUTES,
) -> Dict[str, Any]:
    """Turn the per-minute arrays into the (N,C,10,F) window tensors the ridge wants."""
    field = arrays["field"]
    valid = arrays["valid"]
    minute_index = arrays["minute_index"]
    pos = {int(m): i for i, m in enumerate(minute_index)}
    in_split = arrays["split"] == split
    hs = [int(h) for h in horizons]

    rows: List[int] = []
    for i, m in enumerate(minute_index):
        if not (in_split[i] and arrays["ctx_ok"][i]):
            continue
        if any(not arrays["horizon_ok"].get(h, np.zeros_like(in_split))[i] for h in hs):
            continue
        if any(int(m) + h not in pos or not in_split[pos[int(m) + h]] for h in hs):
            continue
        if int(m) - (context_minutes - 1) not in pos:
            continue
        rows.append(i)

    ctx = np.stack([
        np.stack([field[pos[int(minute_index[i]) - k]] for k in range(context_minutes - 1, -1, -1)],
                 axis=1)
        for i in rows
    ]) if rows else np.zeros((0, field.shape[1], context_minutes, field.shape[2]))
    ctx_valid = np.stack([
        np.stack([valid[pos[int(minute_index[i]) - k]] for k in range(context_minutes - 1, -1, -1)],
                 axis=1)
        for i in rows
    ]) if rows else np.zeros((0, field.shape[1], context_minutes), dtype=bool)

    target = {h: np.stack([field[pos[int(minute_index[i]) + h]] for i in rows])
              if rows else np.zeros((0, field.shape[1], field.shape[2])) for h in hs}
    target_mask = {h: np.stack([np.repeat(valid[pos[int(minute_index[i]) + h]][:, None],
                                          field.shape[2], axis=1) for i in rows])
                   if rows else np.zeros((0, field.shape[1], field.shape[2]), dtype=bool)
                   for h in hs}
    return {
        "context": ctx, "context_valid": ctx_valid,
        "target": target, "target_mask": target_mask,
        "window_id": np.asarray([int(minute_index[i]) for i in rows], dtype=int),
        "persistence": ctx[:, :, -1, :] if rows else np.zeros((0, field.shape[1], field.shape[2])),
    }

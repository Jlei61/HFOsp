"""Losses and the reported consistency diagnostic (spec sections 6 and 7).

    L = L_forecast + lambda_cons * L_cons

* ``L_forecast``: masked MSE on the (contact x log-frequency) field, computed
  per horizon and **equally weighted** across the four horizons.  Equal weight
  means the mean of the per-horizon means -- *not* a pooled mean over all valid
  cells, which would silently down-weight a horizon that happens to have fewer
  valid (contact, freq) entries in the batch.
* ``L_cons``: Huber between the state the encoder produces at ``t+1`` and the
  state the one-step map produces from ``t``.
* ``consistency_ratio`` is the spec section 7 quantity ``E_cons``.  It is
  **reported**, never optimised.
"""

from __future__ import annotations

from typing import Dict, Mapping, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from . import contract

__all__ = [
    "masked_forecast_loss",
    "consistency_loss",
    "total_loss",
    "consistency_ratio",
    "ConsistencyParts",
    "latent_diagnostics",
    "ACTIVE_DIM_STD_THRESHOLD",
]

#: A latent dimension counts as "in use" when its across-batch std exceeds this.
ACTIVE_DIM_STD_THRESHOLD = 1e-3


class ConsistencyParts(NamedTuple):
    """``E_cons`` together with the two norms it is built from.

    Reporting the ratio alone is ambiguous: ``E_cons = 0.02`` from a state that
    moves a lot and is predicted well, and ``E_cons = 0.02`` from a state that
    has collapsed to a constant (both norms ~ 0), are the same number and
    opposite findings.  ``denominator`` is exactly ``latent_diagnostics``'
    ``z_step_norm`` per sample, so a collapsed run is visible in the same row.
    """

    ratio: Tensor
    numerator: Tensor
    denominator: Tensor


def masked_forecast_loss(
    pred: Mapping[int, Tensor],
    target: Mapping[int, Tensor],
    mask: Mapping[int, Tensor],
) -> Tuple[Tensor, Dict[int, Tensor]]:
    """Equal-weight masked MSE across horizons.

    Parameters
    ----------
    pred, target:
        ``{horizon_minutes: (B, C, F)}``.  ``target`` may contain NaN in cells
        that ``mask`` marks invalid; those cells never touch the value or the
        gradient (the difference is taken through ``torch.where``, and the
        target is sanitised first, so ``0 * NaN`` can never appear).
    mask:
        ``{horizon_minutes: (B, C, F)}`` bool, True = usable cell.

    Returns
    -------
    total:
        ``sum_h mse_h / n_horizons_with_data``.  A horizon with **zero** valid
        cells in this batch contributes 0 to the numerator *and* is dropped
        from the denominator, so it neither rewards nor penalises the model.
    per_horizon:
        ``{horizon: scalar tensor}``.  A horizon with zero valid cells reports
        ``0.0`` (not NaN, per hard-invalid condition 6 of the spec).  A logged
        0.0 therefore means "nothing to score", not "perfect" -- callers that
        log per-horizon values should log ``mask[h].sum()`` beside them.
    """
    if set(pred) != set(target) or set(pred) != set(mask):
        raise ValueError(
            f"pred/target/mask horizons disagree: {sorted(pred)} / "
            f"{sorted(target)} / {sorted(mask)}"
        )
    per_horizon: Dict[int, Tensor] = {}
    total: Optional[Tensor] = None
    n_used = 0
    for h in sorted(pred):
        p, t, m = pred[h], target[h], mask[h]
        if p.shape != t.shape or p.shape != m.shape:
            raise ValueError(
                f"horizon {h}: shapes disagree pred={tuple(p.shape)} "
                f"target={tuple(t.shape)} mask={tuple(m.shape)}"
            )
        m = m.to(torch.bool)
        n_valid = int(m.sum())
        if n_valid == 0:
            # graph-connected exact zero: keeps .backward() valid on a batch
            # in which some horizon has no scorable cell at all.
            per_horizon[h] = (p * 0.0).sum()
            continue
        t_safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        diff = torch.where(m, p - t_safe, torch.zeros((), dtype=p.dtype, device=p.device))
        mse = (diff * diff).sum() / n_valid
        per_horizon[h] = mse
        total = mse if total is None else total + mse
        n_used += 1
    if total is None:
        total = sum(per_horizon.values())
        return total, per_horizon
    return total / n_used, per_horizon


def consistency_loss(z_next_encoded: Tensor, z_next_predicted: Tensor) -> Tensor:
    """Huber(z_enc(t+1), Phi_1(z_enc(t))), mean over batch and latent dims.

    ``delta = contract.CONSISTENCY_HUBER_DELTA``.  Neither argument is
    detached -- that is the literal spec section 6 form.  The gradient therefore
    also flows into the encoder, so the term can in principle be reduced by
    flattening the state; only the forecast term prevents that, which is why
    R0.1 compares ``lambda_cons = 0.1`` against ``lambda_cons = 0`` rather than
    tuning it.
    """
    if z_next_encoded.shape != z_next_predicted.shape:
        raise ValueError(
            f"shape mismatch {tuple(z_next_encoded.shape)} vs "
            f"{tuple(z_next_predicted.shape)}"
        )
    return F.huber_loss(
        z_next_predicted,
        z_next_encoded,
        reduction="mean",
        delta=float(contract.CONSISTENCY_HUBER_DELTA),
    )


def total_loss(
    pred: Mapping[int, Tensor],
    target: Mapping[int, Tensor],
    mask: Mapping[int, Tensor],
    z_next_encoded: Optional[Tensor] = None,
    z_next_predicted: Optional[Tensor] = None,
    lambda_cons: float = contract.LAMBDA_CONS_DEFAULT,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """``L_forecast + lambda_cons * L_cons`` plus the parts, for logging.

    The consistency term is skipped (and reported as exactly 0) when
    ``lambda_cons == 0`` or when no state pair is supplied; the two settings
    R0.1 compares are ``lambda_cons = LAMBDA_CONS_DEFAULT`` and ``0.0``.
    """
    forecast, per_horizon = masked_forecast_loss(pred, target, mask)
    parts: Dict[str, Tensor] = {"forecast": forecast}
    for h, v in per_horizon.items():
        parts[f"forecast_h{h}"] = v
    have_pair = z_next_encoded is not None and z_next_predicted is not None
    if have_pair and float(lambda_cons) != 0.0:
        cons = consistency_loss(z_next_encoded, z_next_predicted)
        total = forecast + float(lambda_cons) * cons
    else:
        cons = torch.zeros((), dtype=forecast.dtype, device=forecast.device)
        total = forecast
    parts["consistency"] = cons
    parts["total"] = total
    return total, parts


def consistency_ratio(
    z_enc_next: Tensor,
    z_pred_next: Tensor,
    z_enc_now: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Spec section 7 ``E_cons``, per sample.

    ``|| z_enc(t+1) - Phi_1(z_enc(t)) || / ( || z_enc(t+1) - z_enc(t) || + eps )``

    Returns a :class:`ConsistencyParts` named tuple ``(ratio, numerator,
    denominator)``, each of shape ``(B,)`` -- **not** a bare tensor: a small
    ratio has to be readable against the size of the step it normalises by.

    Reported (median + quartiles per patient on validation), never optimised.
    ``E_cons << 1`` means one step of the dynamics explains most of the change
    in the state; ``E_cons >~ 1`` means the encoder re-encodes each minute
    independently, which triggers the downgraded wording of spec section 2.4.
    """
    if z_enc_next.shape != z_pred_next.shape or z_enc_next.shape != z_enc_now.shape:
        raise ValueError(
            f"shape mismatch {tuple(z_enc_next.shape)} / {tuple(z_pred_next.shape)} "
            f"/ {tuple(z_enc_now.shape)}"
        )
    num = torch.linalg.vector_norm(z_enc_next - z_pred_next, dim=-1)
    den = torch.linalg.vector_norm(z_enc_next - z_enc_now, dim=-1)
    return ConsistencyParts(ratio=num / (den + eps), numerator=num, denominator=den)


def latent_diagnostics(z_now: Tensor, z_next: Tensor) -> Dict[str, float]:
    """Collapse observability for the state itself, logged every epoch.

    ``consistency_loss`` detaches neither side (the literal spec section 6
    form), so ``z = const`` is a trivial minimiser of the consistency term and
    only the forecast term argues against it.  That makes collapse a thing to
    *watch*, not to infer after the fact:

    * ``z_std_per_dim``  -- mean over latent dims of the across-batch std of
      ``z_now``.  Collapse drives this to 0.
    * ``z_step_norm``    -- mean ``||z(t+1) - z(t)||``, i.e. the denominator of
      ``consistency_ratio``.  A collapsed state has a tiny step norm, which is
      what makes a small ``E_cons`` meaningless.
    * ``n_active_dims``  -- latent dims whose across-batch std exceeds
      ``ACTIVE_DIM_STD_THRESHOLD``.  Partial collapse shows up here first.
    * ``n_samples``      -- batch size.  The std is the *population* std (so it
      is 0, never NaN, at B = 1); with B = 1 the two std-based numbers are
      structurally 0 and must not be read as collapse.
    """
    if z_now.shape != z_next.shape:
        raise ValueError(f"shape mismatch {tuple(z_now.shape)} vs {tuple(z_next.shape)}")
    if z_now.ndim != 2:
        raise ValueError(f"expected (B, latent); got {tuple(z_now.shape)}")
    z = z_now.detach().float()
    std = z.std(dim=0, unbiased=False)                     # population std: no NaN at B=1
    step = torch.linalg.vector_norm(z_next.detach().float() - z, dim=-1)
    return {
        "z_std_per_dim": float(std.mean()),
        "z_step_norm": float(step.mean()),
        "n_active_dims": int((std > ACTIVE_DIM_STD_THRESHOLD).sum()),
        "n_samples": int(z.shape[0]),
    }

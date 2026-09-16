"""Fixed multi-task weights and masked, per-endpoint loss normalisation.

Clause C9. Weights come from the FIT intercept model's per-endpoint mean NLL
and are therefore identical for F, L and N at the same subject/view. A target
window with zero events has no observed recruitment vector, so it is excluded
from the spatial term instead of being trained on a fabricated all-zero label.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..v039.transition import endpoint_loss


def intercept_reference(counts, recruitment, valid, spatial_valid, fit_rows):
    """Closed-form FIT intercept: pooled over the rotated leads, no fitting."""
    y = counts[fit_rows][valid[fit_rows]]
    if y.numel() < 8:
        raise ValueError('FIT count support too small for an intercept reference')
    mean = y.mean()
    variance = y.var(unbiased=False)
    r = (mean.square() / (variance - mean).clamp_min(.1)).clamp(.05, 100)
    log_dispersion = torch.log(torch.expm1(r.clamp_max(30)))
    p = recruitment[fit_rows][spatial_valid[fit_rows]]
    probability = p.mean(0).clamp(.01, .99) if p.numel() else None
    return dict(log_mean=mean.clamp_min(1e-3).log(), log_dispersion=log_dispersion, probability=probability,
                n_count_rows=int(y.numel()), n_spatial_rows=int(p.shape[0]) if p.numel() else 0)


def fit_intercept_weights(counts, recruitment, valid, spatial_valid, fit_rows):
    reference = intercept_reference(counts, recruitment, valid, spatial_valid, fit_rows)
    y = counts[fit_rows][valid[fit_rows]]
    mu = reference['log_mean'].expand(y.shape)
    logits = (torch.logit(reference['probability']) if reference['probability'] is not None
              else torch.zeros(recruitment.shape[-1], device=counts.device))
    spatial_target = recruitment[fit_rows][spatial_valid[fit_rows]]
    _, nb, _ = endpoint_loss(mu, torch.zeros(y.shape + logits.shape, device=y.device),
                             y, torch.zeros(y.shape + logits.shape, device=y.device),
                             reference['log_dispersion'], 'joint')
    count_nll = float(nb.mean())
    if spatial_target.numel():
        spatial_nll = float(nn.functional.binary_cross_entropy_with_logits(
            logits.expand(spatial_target.shape), spatial_target, reduction='none').mean())
    else:
        spatial_nll = None
    w_count = 1. / max(count_nll, .1)
    w_recruit = (1. / max(spatial_nll, .05)) if spatial_nll is not None else 0.
    scale = 2. / (w_count + w_recruit) if (w_count + w_recruit) > 0 else 0.
    return dict(intercept_count_nll=count_nll, intercept_recruitment_nll=spatial_nll,
                weight_count=w_count * scale, weight_recruitment=w_recruit * scale,
                rule='w=1/max(L,floor) with floors 0.1/0.05, renormalised to sum 2; shared across F/L/N',
                n_count_rows=reference['n_count_rows'], n_spatial_rows=reference['n_spatial_rows'])


def masked_objective(log_mean, logits, counts, recruitment, spatial_mask, log_dispersion,
                     weights, view, denominator_count, denominator_spatial):
    """Sum-form objective divided by WHOLE-batch denominators (clause C7).

    Dividing every micro-batch by the same global denominators makes gradient
    accumulation numerically identical to a single 128-sample step.
    """
    _, nb, spatial = endpoint_loss(log_mean, logits, counts, recruitment, log_dispersion, 'joint')
    mask = spatial_mask.to(spatial.dtype)
    if view == 'count':
        return nb.sum() / max(denominator_count, 1), nb.detach(), spatial.detach()
    if view == 'recruitment':
        return (spatial * mask).sum() / max(denominator_spatial, 1), nb.detach(), spatial.detach()
    value = weights['weight_count'] * nb.sum() / max(denominator_count, 1)
    if denominator_spatial:
        value = value + weights['weight_recruitment'] * (spatial * mask).sum() / denominator_spatial
    return value, nb.detach(), spatial.detach()


def aggregate_objective(nb, spatial, spatial_mask, weights, view):
    """Report form: weighted objective plus both raw per-endpoint NLLs."""
    nb = np.asarray(nb, float); spatial = np.asarray(spatial, float); mask = np.asarray(spatial_mask, bool)
    count_nll = float(nb.mean()) if nb.size else None
    spatial_nll = float(spatial[mask].mean()) if mask.any() else None
    if view == 'count':
        objective = count_nll
    elif view == 'recruitment':
        objective = spatial_nll
    else:
        objective = (weights['weight_count'] * count_nll if count_nll is not None else 0.)
        if spatial_nll is not None:
            objective = objective + weights['weight_recruitment'] * spatial_nll
    return dict(objective=objective, count_nll=count_nll, recruitment_nll=spatial_nll,
                n_count=int(nb.size), n_recruitment=int(mask.sum()))

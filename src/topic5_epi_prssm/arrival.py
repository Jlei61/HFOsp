"""Conditional intensity for when the next discharge arrives.

v0.1 optimised only ``p(marks | t_e, history)``: the timestamps were fed in but no
term in the loss ever asked the state to explain them, so the state was free to
track "which event number is this" and ignore the clock.  It did: shuffling the
intervals cost about 0.0012 nats and an event-index kernel scored slightly better
than a real-time one.

This module supplies the missing channel, as a modulated renewal process

    lambda(u) = exp(eta_p + beta . z_rate(t_{e-1}))  *  h(a),      a = u - t_{e-1}

with a learned baseline hazard ``h`` over elapsed time.  Splitting the intensity
into a slow multiplicative part and a renewal shape keeps the two questions apart:
``h`` carries refractoriness and the lognormal-ish interval shape that Topic 2
already established, while ``eta_p + beta . z_rate`` is the part a slow state can
move.  Only the multiplicative part is allowed to depend on the state, so "the
state modulates the rate" is a single interpretable coefficient rather than a
capacity story.

Two rules the survival integral has to honour:

  * an interval that spans a **metadata gap** was never recorded, so no evidence of
    absence exists across it and it must not enter the integral;
  * an interval that was recorded and simply had no discharge in it **is** data and
    must enter it -- that silence is most of what identifies the rate.
"""
from __future__ import annotations

import math

import torch
from torch import nn

#: elapsed-time grid for the baseline hazard, 100 ms to two days
LOG_A_MIN, LOG_A_MAX = math.log(0.1), math.log(172800.0)
N_QUADRATURE = 256


class RenewalIntensity(nn.Module):
    """Baseline hazard over elapsed time plus a state-modulated multiplier."""

    def __init__(self, n_patients: int, state_dim: int, *, n_basis: int = 10,
                 markov_renewal: bool = False):
        super().__init__()
        #: Depend on the *previous* interval as well as the elapsed one.  Without it
        #: the intensity is a modulated renewal process: the multiplier moves on the
        #: scale of minutes to days and the shape depends only on time since the last
        #: event, so neither term can produce short-range interval correlation.  The
        #: data has plenty -- lag-1 log-interval correlation is +0.300 in 33 of 34
        #: patients, matching what Topic 2 recorded independently -- and its absence
        #: is what pushed the rescaled residuals off unit variance.
        self.markov_renewal = bool(markov_renewal)
        self.previous_interval_weight = nn.Parameter(torch.zeros(1))
        centres = torch.linspace(LOG_A_MIN, LOG_A_MAX, n_basis)
        self.register_buffer("centres", centres)
        self.log_width = nn.Parameter(torch.tensor(
            float(math.log((LOG_A_MAX - LOG_A_MIN) / max(n_basis - 1, 1)))))
        self.basis_weight = nn.Parameter(torch.zeros(n_basis))
        self.baseline = nn.Parameter(torch.zeros(()))
        #: per-patient rate offset; the slow state must beat this, not merely beat zero
        self.patient_offset = nn.Parameter(torch.zeros(n_patients))
        #: the single coefficient that says whether the slow state moves the rate
        self.state_weight = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.state_weight.weight)
        nn.init.zeros_(self.state_weight.bias)

        grid = torch.linspace(LOG_A_MIN, LOG_A_MAX, N_QUADRATURE)
        self.register_buffer("log_grid", grid)
        self.register_buffer("grid", torch.exp(grid))

    @torch.no_grad()
    def initialise_from(self, elapsed: torch.Tensor, recorded: torch.Tensor) -> None:
        """Start at the empirical rate, so training does not begin thousands of nats out."""
        kept = elapsed[recorded.bool()]
        if kept.numel() == 0:
            return
        mean_interval = float(kept.clamp(min=1e-3).mean())
        self.baseline.fill_(math.log(1.0 / max(mean_interval, 1e-3)))

    def log_baseline_hazard(self, log_a: torch.Tensor) -> torch.Tensor:
        width = torch.exp(self.log_width).clamp(min=1e-3)
        z = (log_a.unsqueeze(-1) - self.centres) / width
        response = torch.exp(-0.5 * z ** 2) @ self.basis_weight
        return response + self.baseline

    def cumulative_baseline(self) -> torch.Tensor:
        """H(a) on the quadrature grid, integrating h in *linear* elapsed time."""
        hazard = torch.exp(self.log_baseline_hazard(self.log_grid))
        widths = torch.diff(self.grid, prepend=self.grid.new_zeros(1))
        return torch.cumsum(hazard * widths, dim=0)

    def _interpolate(self, cumulative: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        log_a = torch.log(a.clamp(min=math.exp(LOG_A_MIN))).clamp(max=LOG_A_MAX)
        index = torch.searchsorted(self.log_grid, log_a.reshape(-1).contiguous())
        index = index.clamp(1, len(self.log_grid) - 1)
        lo, hi = index - 1, index
        span = (self.log_grid[hi] - self.log_grid[lo]).clamp(min=1e-9)
        frac = ((log_a.reshape(-1) - self.log_grid[lo]) / span).clamp(0.0, 1.0)
        out = cumulative[lo] + frac * (cumulative[hi] - cumulative[lo])
        return out.reshape(a.shape)

    def forward(self, elapsed: torch.Tensor, state_summary: torch.Tensor,
                patient_index: torch.Tensor, recorded: torch.Tensor,
                previous_elapsed: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        """Negative log-likelihood of the arrival times.

        ``elapsed``        (E,) seconds since the previous event
        ``state_summary``  (E, D) the slow rate state as it stood *before* the event
        ``patient_index``  (E,) which patient each event belongs to
        ``recorded``       (E,) bool; False where the interval spans a metadata gap
        """
        a = elapsed.clamp(min=math.exp(LOG_A_MIN))
        log_multiplier = (self.patient_offset[patient_index]
                          + self.state_weight(state_summary).squeeze(-1))
        if self.markov_renewal:
            if previous_elapsed is None:
                raise ValueError("markov_renewal needs the previous interval; passing "
                                 "None would silently fall back to a plain renewal "
                                 "process and reintroduce the misfit it exists to fix")
            prior = torch.log(previous_elapsed.clamp(min=math.exp(LOG_A_MIN)))
            log_multiplier = log_multiplier + self.previous_interval_weight * prior
        log_lambda = log_multiplier + self.log_baseline_hazard(torch.log(a))
        compensator = torch.exp(log_multiplier) * self._interpolate(
            self.cumulative_baseline(), a)
        # an unrecorded interval contributes neither an arrival term nor a survival
        # term: nothing was observed across it, in either direction
        keep = recorded.to(log_lambda.dtype)
        return {"log_intensity": log_lambda * keep,
                "compensator": compensator * keep,
                "n_recorded": keep.sum(),
                "nll": -(log_lambda * keep).sum() + (compensator * keep).sum()}

    @torch.no_grad()
    def rescaled_times(self, elapsed: torch.Tensor, state_summary: torch.Tensor,
                       patient_index: torch.Tensor, recorded: torch.Tensor,
                       previous_elapsed: torch.Tensor | None = None
                       ) -> torch.Tensor:
        """Time-rescaling residuals; a correct intensity makes these unit-exponential.

        This is the goodness-of-fit the mark-only model could never be checked against.
        """
        out = self.forward(elapsed, state_summary, patient_index, recorded,
                           previous_elapsed)
        return out["compensator"][recorded.bool()]


@torch.no_grad()
def goodness_of_fit(rescaled: torch.Tensor) -> dict[str, float]:
    """Mean and sd are not enough: a wrong intensity can match both and still leave
    structure in the residuals, which is exactly what a missing lag-1 term does."""
    import numpy as np
    from scipy import stats

    values = rescaled.detach().cpu().numpy()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 50:
        return {"n": int(values.size), "status": "TOO_FEW"}
    ks = stats.kstest(values, "expon")
    centred = values - values.mean()
    denominator = float((centred ** 2).sum()) or 1.0
    acf = [float((centred[:-k] * centred[k:]).sum() / denominator) for k in (1, 2, 5)]
    quantiles = np.linspace(0.02, 0.98, 25)
    qq = float(np.max(np.abs(np.quantile(values, quantiles)
                             + np.log(1.0 - quantiles))))
    # A diagnostic that is computed and then ignored is decoration.  ``status`` used
    # to be "OK" for any sample of 50 or more, so a model with the right first two
    # moments could carry serial structure in its residuals and still be waved
    # through by a gate that only read the cohort median of mean and sd.
    checks = {
        "mean_within_20pct": bool(0.8 <= values.mean() <= 1.2),
        "sd_within_20pct": bool(0.8 <= values.std() <= 1.2),
        "no_residual_serial_structure": bool(max(abs(a) for a in acf) < 0.05),
        "qq_within_tolerance": bool(qq < 0.5),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {"n": int(values.size),
            "mean": float(values.mean()), "sd": float(values.std()),
            "ks_statistic": float(ks.statistic), "ks_p": float(ks.pvalue),
            "acf_lag1": acf[0], "acf_lag2": acf[1], "acf_lag5": acf[2],
            "qq_max_abs_deviation": qq,
            "checks": checks, "failed_checks": failed,
            "status": "OK" if not failed else "MISSPECIFIED"}

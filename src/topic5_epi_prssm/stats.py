"""Patient-first statistics.

The unit of cohort inference is the patient.  Seeds are aggregated inside a
patient before any cohort statistic is formed; events, windows and seizures are
repeated measures and never enter as independent samples.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np

from .contracts import FROZEN


@dataclass
class PairedEffect:
    """One paired patient-level contrast."""

    label: str
    n_patients: int
    median_delta: float
    mean_delta: float
    ci_low: float
    ci_high: float
    n_favourable: int
    sign_test_p: float
    wilcoxon_p: float
    per_patient: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def aggregate_seeds(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Median across seeds inside each patient -- the only legal first step."""
    subjects = sorted({s for v in values for s in v})
    out = {}
    for subject in subjects:
        got = [float(v[subject]) for v in values if subject in v and np.isfinite(v[subject])]
        if got:
            out[subject] = float(np.median(got))
    return out


def paired_effect(a: Mapping[str, float], b: Mapping[str, float], *, label: str,
                  lower_is_better: bool = True, seed: int | None = None) -> PairedEffect:
    """``a - b`` per patient, on the intersection of the two subject sets.

    Subject keys are matched explicitly; array order is never trusted.
    """
    subjects = sorted(set(a) & set(b))
    delta = {s: float(a[s]) - float(b[s]) for s in subjects
             if np.isfinite(a[s]) and np.isfinite(b[s])}
    subjects = sorted(delta)
    values = np.array([delta[s] for s in subjects], dtype=float)
    if len(values) == 0:
        return PairedEffect(label, 0, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, {})
    favourable = int((values < 0).sum() if lower_is_better else (values > 0).sum())
    rng = np.random.default_rng(FROZEN["bootstrap_seed"] if seed is None else seed)
    draws = rng.integers(0, len(values), size=(FROZEN["bootstrap_draws"], len(values)))
    boot = np.median(values[draws], axis=1)
    return PairedEffect(
        label=label, n_patients=len(values),
        median_delta=float(np.median(values)), mean_delta=float(values.mean()),
        ci_low=float(np.quantile(boot, 0.025)), ci_high=float(np.quantile(boot, 0.975)),
        n_favourable=favourable,
        sign_test_p=_sign_test(values, lower_is_better),
        wilcoxon_p=_wilcoxon(values),
        per_patient=delta,
    )


def _sign_test(values: np.ndarray, lower_is_better: bool) -> float:
    from scipy import stats
    non_zero = values[values != 0]
    if len(non_zero) == 0:
        return float("nan")
    favourable = int((non_zero < 0).sum() if lower_is_better else (non_zero > 0).sum())
    return float(stats.binomtest(favourable, len(non_zero), 0.5, alternative="two-sided").pvalue)


def _wilcoxon(values: np.ndarray) -> float:
    from scipy import stats
    non_zero = values[values != 0]
    if len(non_zero) < 3:
        return float("nan")
    try:
        return float(stats.wilcoxon(non_zero).pvalue)
    except ValueError:
        return float("nan")


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-corrected p-values inside one declared multiplicity family."""
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    n = len(items)
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        adjusted = min(1.0, (n - i) * p)
        running = max(running, adjusted)
        out[key] = running
    for key, value in pvalues.items():
        out.setdefault(key, float("nan"))
    return out


def stratify(effect: PairedEffect, strata: Mapping[str, str]) -> dict[str, dict]:
    """Split a paired effect by a per-patient label such as dataset."""
    out: dict[str, dict] = {}
    groups: dict[str, list[float]] = {}
    for subject, value in effect.per_patient.items():
        groups.setdefault(strata.get(subject, "unknown"), []).append(value)
    for name, values in groups.items():
        array = np.array(values, dtype=float)
        out[name] = {
            "n_patients": int(len(array)),
            "median_delta": float(np.median(array)),
            "n_favourable": int((array < 0).sum()),
        }
    return out

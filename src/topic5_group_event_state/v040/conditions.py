"""Common measurement conditions C, offered to every arm through one interface.

C9/C10: intraday clock, recording-segment age, time since the available clinical
reset, count/marks publication-support age, readable exposure and the matching
missing flags.  Continuous entries are log1p-compressed and standardised with
FIT-frozen medians and IQRs.  A field that does not exist is a declared missing
flag, never a guessed value.  The explicit recent history H stays a consumer
input and is deliberately absent here (C11).
"""
from __future__ import annotations
import numpy as np

from . import data as D

# name, is_continuous.  Flags stay raw 0/1 so a missing marker is never rescaled.
FIELDS = (('log_episode_age_hours', True),
          ('log_clinical_reset_age_hours', True),
          ('clinical_reset_known', False),
          ('log_count_support_age_minutes', True),
          ('count_support_missing', False),
          ('log_marks_support_age_minutes', True),
          ('marks_support_missing', False),
          ('short_window_published_coverage', True),
          ('log_published_exposure_hours', True),
          ('has_observation', False))
NAMES = tuple(n for n, _ in FIELDS)
CONTINUOUS = np.array([c for _, c in FIELDS], bool)
DIM = len(FIELDS)
CLOCK_DIM = 2
COND_DIM = CLOCK_DIM + DIM


def raw_conditions(payload, split, queries, short_minutes, role='descriptive', last_observed=None):
    """Deterministic timing/support covariates readable at each query time."""
    pk = payload['packets']
    if last_observed is None:
        last_observed = D.observed_support_end(payload)
    resets = np.asarray(split['excluded_intervals'], float).reshape(-1, 2)
    out = np.zeros((len(np.asarray(queries)), DIM), np.float64)
    for j, q in enumerate(np.asarray(queries, int)):
        t = float(pk['end'][q])
        start = int(split['episode_start'][q])
        ix = D.readable_prefix(payload, split, q, role)
        episode_age = max(0., t - float(pk['start'][start])) / 3600.
        ends = resets[:, 1][resets[:, 1] <= t + 1e-6]
        reset_known = bool(len(ends))
        reset_age = (t - float(ends.max())) / 3600. if reset_known else 0.
        support_end = float(np.nanmax(last_observed[ix])) if len(ix) and np.isfinite(last_observed[ix]).any() else None
        support_age = 0. if support_end is None else max(0., t - support_end) / 60.
        missing = support_end is None
        recent = ix[pk['start'][ix] >= t - short_minutes * 60. - 1e-6]
        coverage = float(pk['exposure'][recent].sum()) / (short_minutes * 60.)
        exposure_hours = float(pk['exposure'][ix].sum()) / 3600.
        out[j] = (np.log1p(episode_age), np.log1p(reset_age), float(reset_known),
                  np.log1p(support_age), float(missing),
                  np.log1p(support_age), float(missing),
                  coverage, np.log1p(exposure_hours), float(len(ix) > 0))
    return out


def fit_condition_scaling(payload, split, short_minutes, stride=30):
    """FIT-frozen centre/scale for the continuous entries only."""
    fit = np.flatnonzero(split['train_packet'])
    if not len(fit):
        raise ValueError('no FIT packets for condition scaling')
    queries = fit[::stride] if len(fit) > stride else fit
    raw = raw_conditions(payload, split, queries, short_minutes, 'fit')
    center = np.zeros(DIM)
    scale = np.ones(DIM)
    for k in range(DIM):
        if not CONTINUOUS[k]:
            continue
        c, s = D.v0312.legacy._robust(raw[:, k])
        center[k] = c
        scale[k] = s
    return dict(center=center, scale=scale, names=list(NAMES), continuous=CONTINUOUS.tolist(),
                short_minutes=int(short_minutes), fit_query_count=int(len(queries)),
                fit_query_digest=D.digest(np.asarray(queries, np.int64)),
                note='flags are never rescaled; a missing field is a declared flag, not an imputed value')


def standardize(raw, scaling):
    z = (np.asarray(raw, float) - scaling['center']) / scaling['scale']
    keep = np.asarray(scaling['continuous'], bool)
    return np.where(keep[None, :], np.clip(z, -8., 8.), np.asarray(raw, float)).astype(np.float32)


def query_conditions(payload, split, queries, scaling, role='descriptive', last_observed=None):
    return standardize(raw_conditions(payload, split, queries, scaling['short_minutes'], role, last_observed), scaling)


def availability_note():
    return dict(
        online_available=['log_episode_age_hours', 'log_count_support_age_minutes', 'count_support_missing',
                          'log_marks_support_age_minutes', 'marks_support_missing',
                          'short_window_published_coverage', 'log_published_exposure_hours', 'has_observation'],
        offline_only=['log_clinical_reset_age_hours', 'clinical_reset_known'],
        offline_reason=('seizure and post-ictal boundaries are clinically annotated after the fact; they already '
                        'define the registered conditional-interictal scope, and they are supplied identically to '
                        'every arm, so they support offline conditional analysis rather than prospective claims'),
        release_streams=('count and marks share one closed one-hour block release in the current packets; both '
                         'ages are exported separately so an asynchronous stream would be visible'),
        explicit_history_excluded='H (recent readable event content) is a consumer input and is not folded into C')

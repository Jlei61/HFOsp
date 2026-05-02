"""Data-driven ictal-onset SOZ audit — PR-T3-1 v1.1 (SUPERSEDED).

⚠️  **OBSOLETE 2026-05-03**: PR-T3-1 v1.1 is superseded by v2.1 pivot
(``docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md``).
The v1.1 M1 (HFO rate) + M2 (HFO band-power log-ratio) helpers in this
module are KEPT BUT MARKED OBSOLETE — they remain importable so the
24-subject v1.1 cohort run can be reproduced from
``per_subject_hfo_rate_obsolete_v1_1/`` if needed for audit.

The v2.1 pivot introduces a two-layer design:

- Layer A: ``src/ictal_er_rank.py`` — new ictal ER-rank producer (NOT
  in this module).
- Layer B: ``src/data_driven_soz_pivot.py`` — label builder + audit
  consumer (NOT in this module).

Helpers in this module split into two groups:

**KEPT and reused by v2.1** (still primary, no obsolete marker):

- ``annotate_clinical_soz`` — 3-state (SOZ/nonSOZ/unknown) channel
  matcher (used by both v1.1 audit and v2.1 Layer B).
- ``matched_clinical_contacts`` — unmatched-name reporter (used by
  audit.csv schema in both versions).
- ``check_channel_schema_consistency`` — cross-block channel order
  validator (still used by the ``--audit`` CLI mode).
- ``compute_overlap`` / ``random_expected_jaccard`` — overlap metrics
  (reused by Layer B).
- ``SOZ_LABEL`` / ``NON_SOZ_LABEL`` / ``UNKNOWN_LABEL`` constants.

**OBSOLETE (kept for reproducibility, not for new analysis)**:

- ``compute_hfo_onset_metrics`` (M1)
- ``rank_top_k_per_seizure``
- ``aggregate_consensus`` / ``aggregate_median_rank``
- ``_bandpass_power`` / ``compute_er_logratio`` (M2)
- ``estimate_per_channel_eps`` / ``select_m2_eligible_channels`` /
  ``PerChannelEps``
- ``prefilter_seizures_by_block_window``
- ``time_shifted_seizure_onsets``
- ``compute_per_subject_audit``

These are NOT to be invoked by new code. They are reachable by tests
and by the deprecated ``--per-subject`` / ``--cohort-overlap`` /
``--build-data-driven-soz-labels`` CLI modes (which now print a
deprecation warning).

See:

- ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``
  (v1.1 plan, superseded)
- ``docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md``
  (v2.1 pivot, plan-of-record)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from src.event_periodicity import _normalize_channel_name


SOZ_LABEL = "soz"
NON_SOZ_LABEL = "non_soz"
UNKNOWN_LABEL = "unknown"


def matched_clinical_contacts(
    analysis_channels: Iterable[str],
    clinical_soz: Iterable[str],
) -> Set[str]:
    """Set of normalized clinical SOZ contacts that touch ≥1 analysis channel.

    Reuses ``_normalize_channel_name`` so the unmatched complement is
    consistent with ``annotate_clinical_soz`` (plan §3.2). For bipolar
    channels ``X-Y``, both contacts are checked. For CAR / monopolar
    ``X``, the single contact is checked.

    Empty / whitespace-only contact parts (malformed names) are skipped
    rather than silently producing a spurious match.
    """
    norm_soz = {_normalize_channel_name(s) for s in clinical_soz if s}
    matched: Set[str] = set()
    for ch in analysis_channels:
        if not ch:
            continue
        normalized = _normalize_channel_name(ch)
        for raw_part in normalized.split("-"):
            # Re-normalize each side so a dual-prefix bipolar like
            # "EEG A1-EEG A2" resolves both contacts (the top-level
            # ``_normalize_channel_name`` only strips the leading prefix).
            p = _normalize_channel_name(raw_part)
            if not p:
                continue
            if p in norm_soz:
                matched.add(p)
    return matched


def check_channel_schema_consistency(
    channel_lists: Sequence[Sequence[str]],
) -> Dict[str, object]:
    """Verify ``chns_names`` schema is consistent across blocks.

    Plan §3.2 requires HFO npz channel order to be aligned across blocks
    of the same subject; the per-subject runner concatenates events
    across blocks and indexes by channel position, so an order mismatch
    in even one block silently corrupts every downstream metric.

    Returns
    -------
    dict
        ``{"n_blocks": int, "all_consistent": bool,
        "mismatched_block_indices": list[int], "n_channels_min": int,
        "n_channels_max": int}``.
    """
    n_blocks = len(channel_lists)
    if n_blocks == 0:
        return {
            "n_blocks": 0,
            "all_consistent": True,
            "mismatched_block_indices": [],
            "n_channels_min": 0,
            "n_channels_max": 0,
        }
    reference = list(channel_lists[0])
    mismatched: List[int] = []
    n_min = n_max = len(reference)
    for i, channels in enumerate(channel_lists):
        ch_list = list(channels)
        n_min = min(n_min, len(ch_list))
        n_max = max(n_max, len(ch_list))
        if i == 0:
            continue
        if ch_list != reference:
            mismatched.append(i)
    return {
        "n_blocks": n_blocks,
        "all_consistent": not mismatched,
        "mismatched_block_indices": mismatched,
        "n_channels_min": n_min,
        "n_channels_max": n_max,
    }


def annotate_clinical_soz(
    analysis_channels: Iterable[str],
    clinical_soz: Iterable[str],
) -> Dict[str, str]:
    """Annotate each analysis channel as SOZ / non_soz / unknown.

    Plan §3.2 contract:

    - Bipolar ``X-Y``: if X or Y is in ``clinical_soz`` → ``"soz"``;
      else → ``"non_soz"``.
    - If ``X`` or ``Y`` is empty / whitespace-only (malformed name)
      → ``"unknown"``.
    - CAR / monopolar ``X``: same logic with single contact.

    Per-endpoint normalization is applied to each side after splitting on
    ``-``. This matches ``matched_clinical_contacts`` so audit rows stay
    consistent with annotation: a dual-prefix bipolar like
    ``EEG A1-EEG A2`` resolves both contacts via
    ``_normalize_channel_name`` (the canonical helper only strips the
    *leading* prefix from the whole string, leaving the second side
    prefixed). Delegating to ``match_bipolar_soz`` would silently
    mis-label this case as ``non_soz`` even though the audit reports
    ``A2`` as matched — exactly the inconsistency the audit was added
    to prevent.

    Returns ``{channel_name: label}`` preserving the input ordering of
    ``analysis_channels``.
    """
    soz_set = {_normalize_channel_name(s) for s in clinical_soz if s}
    out: Dict[str, str] = {}
    for ch in analysis_channels:
        normalized = _normalize_channel_name(ch)
        raw_parts = [p.strip() for p in normalized.split("-")]
        if any(not p for p in raw_parts):
            out[ch] = UNKNOWN_LABEL
            continue
        parts = [_normalize_channel_name(p) for p in raw_parts]
        if any(p in soz_set for p in parts):
            out[ch] = SOZ_LABEL
        else:
            out[ch] = NON_SOZ_LABEL
    return out


# ---------------------------------------------------------------------------
# Step 1 — M1 (HFO-onset rate enrichment)
# ---------------------------------------------------------------------------


def compute_hfo_onset_metrics(
    hfo_event_times_per_channel: Mapping[str, np.ndarray],
    seizure_onset: float,
    w_pre: float = 30.0,
    w_post: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """Per-channel HFO-onset enrichment (PR-T3-1 v1.1 §3.3).

    ⚠️  **OBSOLETE 2026-05-03** — superseded by Layer A in v2.1 pivot
    (HFO-rate-based proxy is too tightly coupled with HFO event
    detection; downstream "stereotype-node ↔ SOZ" validation would be
    circular). Kept importable so the v1.1 cohort is reproducible from
    ``per_subject_hfo_rate_obsolete_v1_1/``. Do NOT call from new code.

    Parameters
    ----------
    hfo_event_times_per_channel
        ``{channel_name: ndarray of absolute event timestamps}``.
        Timestamps must be in the same time base as ``seizure_onset``.
    seizure_onset
        Seizure onset (absolute seconds).
    w_pre, w_post
        Pre / post window lengths in seconds (defaults 30 / 10 s).

    Returns
    -------
    dict
        ``{ch: {"n_pre": int, "n_post": int, "rate_pre": float,
        "rate_post": float, "M1_raw": float, "M1_log": float,
        "M1_pois": float}}``.

    Notes
    -----
    Per plan §3.3:

    - Pre window is half-open ``[t_s - w_pre, t_s)``.
    - Post window is half-open ``(t_s, t_s + w_post]``.
    - ``M1_raw   = rate_post - rate_pre``.
    - ``M1_log   = log(n_post + 1) - log(n_pre + 1) - log(W_post / W_pre)``.
    - ``M1_pois  = (n_post - μ_pre) / sqrt(μ_pre + 1)`` with
      ``μ_pre = rate_pre × w_post``.
    - When a channel has zero events in both windows, all three variants
      short-circuit to 0 (plan T4: "通道全无 events → 三 variant 全 0").
      Without the short-circuit, ``M1_log`` would equal
      ``-log(W_post / W_pre)`` and rank silent channels above zero-baseline
      channels with sparse post events, which is not what the plan wants.
    """
    out: Dict[str, Dict[str, float]] = {}
    if w_pre <= 0 or w_post <= 0:
        raise ValueError(f"w_pre ({w_pre}) and w_post ({w_post}) must be positive")
    log_window_correction = math.log(w_post / w_pre)
    for ch, times in hfo_event_times_per_channel.items():
        arr = np.asarray(times, dtype=float).reshape(-1)
        pre_mask = (arr >= seizure_onset - w_pre) & (arr < seizure_onset)
        post_mask = (arr > seizure_onset) & (arr <= seizure_onset + w_post)
        n_pre = int(pre_mask.sum())
        n_post = int(post_mask.sum())
        rate_pre = n_pre / w_pre
        rate_post = n_post / w_post
        if n_pre == 0 and n_post == 0:
            m1_raw = 0.0
            m1_log = 0.0
            m1_pois = 0.0
        else:
            m1_raw = rate_post - rate_pre
            m1_log = (
                math.log(n_post + 1.0)
                - math.log(n_pre + 1.0)
                - log_window_correction
            )
            mu_pre = rate_pre * w_post
            m1_pois = (n_post - mu_pre) / math.sqrt(mu_pre + 1.0)
        out[ch] = {
            "n_pre": n_pre,
            "n_post": n_post,
            "rate_pre": rate_pre,
            "rate_post": rate_post,
            "M1_raw": m1_raw,
            "M1_log": m1_log,
            "M1_pois": m1_pois,
        }
    return out


def rank_top_k_per_seizure(
    per_channel_score: Mapping[str, float],
    k: int,
    nan_handling: str = "rank_last",
) -> List[str]:
    """Top-k channels by ``per_channel_score`` (PR-T3-1 v1.1 §3.5).

    ⚠️  **OBSOLETE 2026-05-03** — Layer A (v2.1) uses
    ``rank_channels_by_n_d`` instead. Kept for v1.1 reproducibility.

    Finite scores rank first (descending score, ties broken by
    ascending channel name). NaN-score channels fill the tail in
    ascending channel-name order. The truncation to ``k`` happens after
    both groups are concatenated, so size-matched primary k =
    ``|clinical_matched|`` (plan §3.6) returns exactly ``k`` channels
    even when many channels have NaN scores (e.g. zero-baseline
    Poisson z that the implementation might emit). Only the
    ``"rank_last"`` policy is implemented.
    """
    if k <= 0:
        return []
    if nan_handling != "rank_last":
        raise ValueError(
            f"unsupported nan_handling={nan_handling!r}; only 'rank_last' is supported"
        )
    finite_items: List[tuple] = []
    nan_channels: List[str] = []
    for ch, score in per_channel_score.items():
        s = float(score)
        if math.isnan(s):
            nan_channels.append(ch)
        else:
            finite_items.append((ch, s))
    finite_items.sort(key=lambda x: (-x[1], x[0]))
    nan_channels.sort()
    ordered = [ch for ch, _ in finite_items] + nan_channels
    return ordered[:k]


def aggregate_consensus(
    per_seizure_topk: Sequence[Sequence[str]],
    min_seizure_fraction: float = 0.5,
) -> Set[str]:
    """Channels appearing in ≥ ``min_seizure_fraction`` of per-seizure top-k
    lists (PR-T3-1 v1.1 §3.5).

    ⚠️  **OBSOLETE 2026-05-03** — v2.1 Layer B uses median-rank only as
    primary aggregation; consensus is no longer reported. Kept for v1.1
    reproducibility.

    .. warning::

        **Output size is data-dependent and does NOT equal the per-seizure
        ``k``.** It can be smaller (no channel meets the threshold) or
        larger (many channels are stable enough to clear it). Step 3/4
        enrichment (Fisher / hypergeometric / random expected overlap)
        must compute basket size as ``len(B)``, not as the input ``k``.
        Computing expected overlap as ``k * |annotated_soz| / n_channels``
        is a contract violation; use ``len(B) * |annotated_soz| /
        n_channels``. If a fixed-size basket is required, post-process
        with a deterministic tie-break, do not assume the consensus
        rule already enforces it.
    """
    n_seizures = len(per_seizure_topk)
    if n_seizures == 0:
        return set()
    threshold_count = min_seizure_fraction * n_seizures
    counts: Dict[str, int] = {}
    for topk in per_seizure_topk:
        for ch in set(topk):  # guard against duplicate names within one list
            counts[ch] = counts.get(ch, 0) + 1
    return {ch for ch, c in counts.items() if c >= threshold_count}


def aggregate_median_rank(
    per_seizure_ranks: Sequence[Mapping[str, int]],
    k: int,
) -> Set[str]:
    """Top-k channels by median rank across seizures (PR-T3-1 v1.1 §3.5).

    ⚠️  **OBSOLETE 2026-05-03 for v1.1 M1 use case** — kept for v1.1
    reproducibility; v2.1 Layer A re-implements median-rank in
    ``compute_per_subject_r_sz`` against per-channel n_d ranks (NOT
    HFO-rate scores).

    Channels missing from a given seizure dict are assigned the worst
    available rank for that seizure (``n_channels``), so a channel that
    only ranks high in one seizure but is missing in others does not
    crowd out a channel that ranks moderately in every seizure. Ties on
    median are broken alphabetically.
    """
    if k <= 0 or not per_seizure_ranks:
        return set()
    all_channels: Set[str] = set()
    max_observed_rank = 0
    for ranks in per_seizure_ranks:
        all_channels.update(ranks.keys())
        if ranks:
            max_observed_rank = max(max_observed_rank, max(ranks.values()))
    worst_rank = max(max_observed_rank, len(all_channels))
    medians: Dict[str, float] = {}
    for ch in all_channels:
        ranks = [s.get(ch, worst_rank) for s in per_seizure_ranks]
        medians[ch] = float(np.median(ranks))
    sorted_channels = sorted(medians.items(), key=lambda x: (x[1], x[0]))
    return {ch for ch, _ in sorted_channels[:k]}


# ---------------------------------------------------------------------------
# Step 2 — M2 (ER log-ratio + Nyquist / filter padding guards)
# ---------------------------------------------------------------------------


class NyquistGuardError(ValueError):
    """Raised when ``sfreq / 2`` is below the M2 band's safety-margined
    upper edge (PR-T3-1 v1.1 §3.4: ``sfreq >= band[1] * 2.1``).

    ⚠️  **OBSOLETE 2026-05-03** — only raised by v1.1 M2 path. Layer A
    in v2.1 enforces Nyquist via PR-6A's already-validated bandpass
    contract upstream.
    """


class FilterPaddingError(ValueError):
    """Raised when the input signal is shorter than the ``filtfilt``
    padding required for the requested band's lowest cutoff (PR-T3-1
    v1.1 §3.4: ``padlen >= int(1.5 * sfreq / band[0])``).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 M2 path only.
    """


def _bandpass_power(
    signal_2d: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Bandpass-filter a 2D signal and return per-sample instantaneous power.

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 M2 helper. Layer A uses PR-6A
    Step 0-2's ``compute_er`` (spectrogram-based ER) directly, not
    Butter+filtfilt power.

    Parameters
    ----------
    signal_2d
        ``(n_samples, n_channels)`` array.
    sfreq
        Sampling rate (Hz).
    band
        ``(low_hz, high_hz)``.

    Returns
    -------
    np.ndarray
        ``(n_samples, n_channels)`` array of squared filtered samples
        (``filtered ** 2``). Callers compute window-mean to get power
        per pre/post window — keeping the per-sample resolution lets
        downstream callers slice arbitrary windows from a single filter
        pass.

    Raises
    ------
    NyquistGuardError
        If ``sfreq / 2 < band[1] * 1.05`` — the same plan-§3.4 5%
        safety margin enforced by ``compute_er_logratio``. The helper
        replicates it so Step 3 power-pre precomputation paths cannot
        silently bypass the contract.
    FilterPaddingError
        If ``n_samples <= padlen`` where
        ``padlen = max(filtfilt_default, int(1.5 * sfreq / band[0]))``.
    """
    from scipy.signal import butter, filtfilt

    if signal_2d.ndim != 2:
        raise ValueError(
            f"signal_2d must be 2D (n_samples, n_channels), got shape {signal_2d.shape}"
        )

    band_lo, band_hi = float(band[0]), float(band[1])
    sfreq = float(sfreq)
    nyq = sfreq / 2.0
    if nyq < band_hi * 1.05:
        raise NyquistGuardError(
            f"sfreq={sfreq} below 5% Nyquist safety margin for band {band}; "
            f"need sfreq >= {band_hi * 2.1}"
        )
    if band_lo <= 0 or band_hi <= band_lo:
        raise ValueError(f"invalid band {band}: must satisfy 0 < lo < hi < nyq")

    b, a = butter(4, [band_lo / nyq, band_hi / nyq], btype="bandpass")

    # filtfilt's default padlen is ``3 * max(len(a), len(b))``.
    # Plan §3.4: also require ``int(1.5 * sfreq / band[0])`` cycles of
    # the lowest-frequency edge so the IIR settles before the data
    # window starts.
    default_padlen = 3 * max(len(a), len(b))
    band_padlen = int(1.5 * sfreq / band_lo)
    required_padlen = max(default_padlen, band_padlen)
    n_samples = signal_2d.shape[0]
    if n_samples <= required_padlen:
        raise FilterPaddingError(
            f"signal too short for filter padding: n_samples={n_samples}, "
            f"required > {required_padlen} (band={band}, sfreq={sfreq})"
        )

    filtered = filtfilt(b, a, signal_2d, axis=0, padlen=required_padlen)
    return filtered ** 2


SignalLoader = Callable[[float, float, Sequence[str]], Tuple[np.ndarray, float]]


def compute_er_logratio(
    signal_loader: SignalLoader,
    channels: Sequence[str],
    seizure_onset: float,
    eps_per_channel: Mapping[str, float],
    w_pre: float = 30.0,
    w_post: float = 10.0,
    edge_buffer: float = 2.0,
    band: Tuple[float, float] = (80.0, 250.0),
) -> Dict[str, float]:
    """Per-channel ER log-ratio: ``log(P_post + ε) − log(P_pre + ε)`` (PR-T3-1 v1.1 §3.4).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 M2 helper. Definition has drifted
    from PR-6A's validated ER (gamma 60-100 / slow 4-20 with z-score +
    CUSUM). v2.1 Layer A re-uses PR-6A's ``compute_er`` directly. Do
    NOT call from new code.

    Parameters
    ----------
    signal_loader
        Callable ``(t_start, t_end, channels) -> (signal[T, C], sfreq)``.
        ``signal`` must be a 2D array with ``shape[1] == len(channels)``
        and channel order matching the requested ``channels`` list.
    channels
        Channel names to score.
    seizure_onset
        Absolute time of the seizure onset (same time base as the
        loader).
    eps_per_channel
        Per-channel noise floor (typically from
        ``estimate_per_channel_eps``). **Every queried channel must
        have an entry**; missing entries raise ``ValueError`` so a
        caller bug (forgetting to estimate eps for a freshly-added
        channel) cannot silently produce a plausible-looking
        log-ratio via the floor.
    w_pre, w_post
        Pre / post window length in seconds.
    edge_buffer
        Seconds to skip on each side of the onset to avoid
        bandpass-filter ringing leaking into either window.
    band
        ``(lo, hi)`` for the Butter passband; primary M2 is
        ``(80, 250)`` Hz.

    Returns
    -------
    dict[str, float]
        Per-channel log-ratio.

    Raises
    ------
    NyquistGuardError
        If ``sfreq / 2 < band[1] * 1.05``.
    FilterPaddingError
        If the loaded signal is too short for ``filtfilt``.

    Notes
    -----
    - ``compute_er_logratio`` is the user-facing entry point for the
      Nyquist guard, mirroring plan §3.4. The check happens *before*
      the bandpass call so a low-sfreq subject is rejected with the
      domain-specific exception rather than a numpy / scipy
      ``ValueError``.
    - Plan T15: ``power_pre = 0`` falls back to ``log(P_post + ε) -
      log(0 + ε)``, which is finite and large but never inf / NaN.
    """
    if w_pre <= 0 or w_post <= 0 or edge_buffer < 0:
        raise ValueError(
            f"w_pre/w_post must be positive and edge_buffer >= 0; "
            f"got w_pre={w_pre}, w_post={w_post}, edge_buffer={edge_buffer}"
        )
    if not channels:
        return {}

    band_lo, band_hi = float(band[0]), float(band[1])

    t_start = float(seizure_onset) - w_pre - edge_buffer
    t_end = float(seizure_onset) + w_post + edge_buffer

    signal, sfreq = signal_loader(t_start, t_end, list(channels))
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[1] != len(channels):
        raise ValueError(
            f"signal_loader returned shape {signal.shape}; "
            f"expected (T, {len(channels)})"
        )
    sfreq = float(sfreq)

    # User-facing Nyquist guard (plan §3.4 — band[1] * 1.05 safety margin).
    if sfreq / 2.0 < band_hi * 1.05:
        raise NyquistGuardError(
            f"sfreq={sfreq} too low for band {band}; need sfreq >= {band_hi * 2.1}"
        )

    inst_power = _bandpass_power(signal, sfreq, band)

    # Window indices relative to t_start (signal sample 0 corresponds to
    # ``seizure_onset - w_pre - edge_buffer``).
    pre_start = 0
    pre_end = int(round(w_pre * sfreq))
    post_start = int(round((w_pre + 2 * edge_buffer) * sfreq))
    post_end = int(round((w_pre + 2 * edge_buffer + w_post) * sfreq))

    if post_end > inst_power.shape[0]:
        raise ValueError(
            f"signal_loader returned {inst_power.shape[0]} samples; "
            f"need at least {post_end} for the post window to fit"
        )

    missing_eps = [ch for ch in channels if ch not in eps_per_channel]
    if missing_eps:
        raise ValueError(
            f"eps_per_channel missing entries for {missing_eps}; "
            f"estimate eps via estimate_per_channel_eps() and pass "
            f"every queried channel"
        )

    out: Dict[str, float] = {}
    for i, ch in enumerate(channels):
        eps = float(eps_per_channel[ch])
        if eps <= 0:
            raise ValueError(f"eps for {ch!r} must be positive, got {eps}")
        power_pre = float(np.mean(inst_power[pre_start:pre_end, i]))
        power_post = float(np.mean(inst_power[post_start:post_end, i]))
        out[ch] = math.log(power_post + eps) - math.log(power_pre + eps)
    return out


class PerChannelEps(NamedTuple):
    """Result of ``estimate_per_channel_eps`` (PR-T3-1 v1.1 §3.4).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 M2 helper.

    Attributes
    ----------
    eps
        Shape ``(n_channels,)``. ``max(raw_percentile, floor)`` per
        channel — pass directly to ``compute_er_logratio`` for
        eligible channels.
    m2_ineligible
        Shape ``(n_channels,)``, bool. ``True`` for channels whose
        pre-power is **strictly zero across every seizure** — these
        must be dropped per plan §3.4 ("如果某通道全 cohort 都 0，
        drop 该通道, 写入 ineligible_channels"). The Step 3 runner
        consumes this mask to populate ``m2_ineligible_channels``
        instead of letting the floor mechanism produce a spurious
        ``log(P_post / 1e-18) ≈ +40`` log-ratio that would dominate
        top-k ranking.
    raw_percentile
        Shape ``(n_channels,)``. The unclamped 1st percentile —
        useful for diagnostics and for the audit doc to report the
        actual noise-floor distribution before the floor is applied.
    """

    eps: np.ndarray
    m2_ineligible: np.ndarray
    raw_percentile: np.ndarray


def estimate_per_channel_eps(
    power_pre_matrix: np.ndarray,
    floor: float = 1e-18,
) -> PerChannelEps:
    """Per-channel noise floor estimate (PR-T3-1 v1.1 §3.4).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 M2 helper.

    Parameters
    ----------
    power_pre_matrix
        Shape ``(n_seizures, n_channels)``. Element ``[s, ch]`` is the
        pre-window band-power for seizure ``s``, channel ``ch`` (output
        of ``np.mean(_bandpass_power(...)[pre_window], axis=0)``).
    floor
        Lower bound for the returned eps (plan §3.4: ``1e-18``).

    Returns
    -------
    PerChannelEps
        See class docstring. Critically: ``m2_ineligible`` is the
        **strict drop signal** (every pre-power value == 0), not the
        weaker "floor activated" condition (raw 1st percentile <
        floor). The floor still kicks in for floor-active channels so
        ``eps`` stays positive for any caller that indexes without
        first filtering by ``m2_ineligible``.

    Raises
    ------
    ValueError
        If ``power_pre_matrix`` is not 2D, or if it has zero rows
        (``np.percentile`` on an empty axis would silently return NaN).
    """
    arr = np.asarray(power_pre_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"power_pre_matrix must be 2D (n_seizures, n_channels), got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(
            f"power_pre_matrix has 0 rows (no seizures); cannot estimate eps"
        )

    raw_percentile = np.percentile(arr, 1, axis=0)
    floor_val = float(floor)
    eps = np.maximum(raw_percentile, floor_val)
    m2_ineligible = np.all(arr == 0.0, axis=0)
    return PerChannelEps(
        eps=eps,
        m2_ineligible=m2_ineligible,
        raw_percentile=raw_percentile,
    )


def select_m2_eligible_channels(
    channels: Sequence[str],
    eps_result: PerChannelEps,
    channel_index: Mapping[str, int],
) -> Tuple[List[str], List[str]]:
    """Filter ``channels`` by ``eps_result.m2_ineligible`` (PR-T3-1 v1.1 §3.4).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 M2 helper.

    Step 3 must call this **before** passing channels into
    ``compute_er_logratio``. Without it, an ineligible channel (every
    pre-power == 0 across every seizure) gets the ``1e-18`` floor as
    its eps and ``log(P_post / 1e-18)`` ≈ +40, dominating the M2
    ranking with a noise-floor artefact. The dropped list goes into
    the per-subject JSON's ``m2_ineligible_channels`` field so the
    audit / cohort summary can report it.

    Parameters
    ----------
    channels
        Names to filter. Order is preserved in the eligible list.
    eps_result
        Output of ``estimate_per_channel_eps`` (cohort pre-power).
    channel_index
        Map from channel name to its row index in the cohort
        ``power_pre_matrix`` that produced ``eps_result``.

    Returns
    -------
    (eligible, dropped) : tuple[list[str], list[str]]
        Both lists preserve the input ``channels`` order. Channels
        with ``m2_ineligible[channel_index[ch]] == True`` go into
        ``dropped``; the rest into ``eligible``.

    Raises
    ------
    KeyError
        If a queried channel is missing from ``channel_index`` —
        this is a Step 3 caller bug (cohort mismatch); fail loudly
        rather than silently dropping.
    """
    eligible: List[str] = []
    dropped: List[str] = []
    mask = np.asarray(eps_result.m2_ineligible, dtype=bool)
    for ch in channels:
        if ch not in channel_index:
            raise KeyError(
                f"channel {ch!r} not in channel_index; "
                f"cohort eps was estimated on a different channel set"
            )
        if bool(mask[channel_index[ch]]):
            dropped.append(ch)
        else:
            eligible.append(ch)
    return eligible, dropped


# ---------------------------------------------------------------------------
# Step 3 — per-subject orchestrator + supporting helpers
# ---------------------------------------------------------------------------


def prefilter_seizures_by_block_window(
    seizure_onsets: Sequence[float],
    seizure_block_ids: Sequence[str],
    block_windows: Mapping[str, Tuple[float, float]],
    w_pre: float = 30.0,
    w_post: float = 10.0,
    edge_buffer: float = 2.0,
) -> Tuple[List[int], List[int], List[str]]:
    """Drop seizures whose M2 window does not fit in their containing block.

    ⚠️  **OBSOLETE 2026-05-03 for v1.1 M2 use case** — Layer A in v2.1
    re-implements the boundary check against PR-6A's baseline window
    contract instead. This helper is kept for v1.1 reproducibility.

    Plan §3.4 + §6 step-3 carry-over: the M2 window is
    ``[t_s − W_pre − edge_buffer, t_s + W_post + edge_buffer]``. Any
    seizure within ``W_pre + edge_buffer`` of its block start, or within
    ``W_post + edge_buffer`` of its block end, has no signal to load.
    The runner must drop these via this **forward inventory check** —
    NOT by catching ``FilterPaddingError`` inside ``compute_er_logratio``
    (which would conflate boundary failure with sfreq problems and waste
    a partial signal load).

    Parameters
    ----------
    seizure_onsets
        Absolute epoch seconds, parallel to ``seizure_block_ids``.
    seizure_block_ids
        Block stem (e.g. ``"107300102_0000"`` for Epilepsiae or the
        Yuquan record name) for each seizure.
    block_windows
        ``{block_id: (block_start_epoch, block_end_epoch)}`` from
        ``epilepsiae_block_inventory.csv`` or the Yuquan EDF
        ``start_time + duration``.
    w_pre, w_post, edge_buffer
        Same defaults as the M2 contract.

    Returns
    -------
    (kept_indices, dropped_indices, reasons) : tuple
        Indices into the input ``seizure_onsets`` list. ``reasons`` is
        parallel to the input length; entries for kept seizures are
        empty strings. Drop reasons are ``"missing_block"``,
        ``"boundary_pre"``, or ``"boundary_post"``.
    """
    n = len(seizure_onsets)
    if len(seizure_block_ids) != n:
        raise ValueError(
            f"length mismatch: seizure_onsets ({n}) vs "
            f"seizure_block_ids ({len(seizure_block_ids)})"
        )
    if w_pre <= 0 or w_post <= 0 or edge_buffer < 0:
        raise ValueError(
            f"w_pre/w_post must be positive and edge_buffer >= 0; "
            f"got w_pre={w_pre}, w_post={w_post}, edge_buffer={edge_buffer}"
        )

    pre_pad = float(w_pre) + float(edge_buffer)
    post_pad = float(w_post) + float(edge_buffer)
    kept: List[int] = []
    dropped: List[int] = []
    reasons: List[str] = [""] * n
    for i, (t_s, blk) in enumerate(zip(seizure_onsets, seizure_block_ids)):
        if blk not in block_windows:
            dropped.append(i)
            reasons[i] = "missing_block"
            continue
        b0, b1 = block_windows[blk]
        if t_s - pre_pad < b0:
            dropped.append(i)
            reasons[i] = (
                f"boundary_pre (onset={t_s} block_start={b0} need_pad={pre_pad})"
            )
            continue
        if t_s + post_pad > b1:
            dropped.append(i)
            reasons[i] = (
                f"boundary_post (onset={t_s} block_end={b1} need_pad={post_pad})"
            )
            continue
        kept.append(i)
    return kept, dropped, reasons


def random_expected_jaccard(a_size: int, b_size: int, n_total: int) -> float:
    """Expected Jaccard for two uniform random subsets of ``[n_total]``.

    Closed-form approximation:

    .. math::

        E[J] \\approx \\frac{|A||B|/N}{|A| + |B| - |A||B|/N}

    Plan §3.7: this is the random null for the Jaccard, not for the raw
    intersection — keep it separate from ``random_expected_intersection``
    (which the enrichment uses).
    """
    if a_size < 0 or b_size < 0 or n_total <= 0:
        raise ValueError(
            f"sizes must be >= 0 and n_total > 0; got "
            f"a={a_size}, b={b_size}, n_total={n_total}"
        )
    if a_size == 0 or b_size == 0:
        return 0.0
    e_inter = a_size * b_size / n_total
    e_union = a_size + b_size - e_inter
    if e_union <= 0:
        return 0.0
    return e_inter / e_union


def compute_overlap(
    a: Set[str],
    b: Set[str],
    n_total: int,
    enrichment_floor: float = 0.5,
) -> Dict[str, float]:
    """Compute Jaccard / precision / recall / F1 + enrichment for two sets.

    Plan §3.7 contract — ``|B|`` is the *actual* size of the second set,
    which for the consensus aggregation may not equal ``k``. The helper
    intentionally takes no ``k`` argument so callers cannot accidentally
    use it.

    Parameters
    ----------
    a, b
        Compared channel sets. ``a`` is typically ``clinical_matched``,
        ``b`` is the data-driven top-k (medianrank) or consensus.
    n_total
        ``|analysis_channel_set|``.
    enrichment_floor
        Plan §3.4: ``enrichment = observed / max(expected, floor)`` so
        a denominator near zero (e.g. tiny |A|) does not produce inf.

    Returns
    -------
    dict with keys: ``jaccard``, ``precision``, ``recall``, ``f1``,
    ``observed_intersection``, ``random_expected_intersection``,
    ``enrichment``, ``random_expected_jaccard``.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be > 0, got {n_total}")
    a_set = set(a)
    b_set = set(b)
    inter = a_set & b_set
    union = a_set | b_set
    n_inter = len(inter)
    n_union = len(union)
    a_size = len(a_set)
    b_size = len(b_set)
    if a_size == 0 or b_size == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = n_inter / b_size
        recall = n_inter / a_size
        if precision + recall <= 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    jaccard = n_inter / n_union if n_union else 0.0
    e_inter = a_size * b_size / n_total
    enrichment = n_inter / max(e_inter, float(enrichment_floor))
    return {
        "jaccard": float(jaccard),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "observed_intersection": int(n_inter),
        "random_expected_intersection": float(e_inter),
        "enrichment": float(enrichment),
        "random_expected_jaccard": random_expected_jaccard(
            a_size, b_size, n_total
        ),
    }


def _per_seizure_topk_for_method(
    per_seizure_scores: Sequence[Mapping[str, float]],
    k: int,
) -> List[List[str]]:
    """Build the per-seizure top-k list for a single method.

    Each seizure has ``{ch: score}`` (NaN scores routed to the bottom
    via ``rank_top_k_per_seizure``).
    """
    return [rank_top_k_per_seizure(scores, k) for scores in per_seizure_scores]


def _per_seizure_rank_for_method(
    per_seizure_scores: Sequence[Mapping[str, float]],
    n_channels: int,
) -> List[Dict[str, int]]:
    """Convert per-seizure scores to per-seizure ``{channel: rank}``
    dicts for ``aggregate_median_rank``. Channels with NaN scores get
    rank ``n_channels`` (worst).
    """
    out: List[Dict[str, int]] = []
    for scores in per_seizure_scores:
        ordered = rank_top_k_per_seizure(scores, n_channels)
        rank_map = {ch: i + 1 for i, ch in enumerate(ordered)}
        out.append(rank_map)
    return out


def _aggregate_topk(
    per_seizure_scores: Sequence[Mapping[str, float]],
    k: int,
    n_channels: int,
    aggregation: str,
    min_seizure_fraction: float = 0.5,
) -> Set[str]:
    """Cross-seizure aggregation. ``aggregation`` ∈ {``medianrank``, ``consensus``}.

    Note: consensus output size is data-dependent and may not equal ``k``.
    """
    if aggregation == "medianrank":
        ranks = _per_seizure_rank_for_method(per_seizure_scores, n_channels)
        return aggregate_median_rank(ranks, k)
    if aggregation == "consensus":
        topks = _per_seizure_topk_for_method(per_seizure_scores, k)
        return aggregate_consensus(topks, min_seizure_fraction=min_seizure_fraction)
    raise ValueError(
        f"unknown aggregation {aggregation!r}; expected 'medianrank' or 'consensus'"
    )


def _per_seizure_consistency(
    per_seizure_scores: Sequence[Mapping[str, float]],
    k: int,
) -> float:
    """Median pairwise Jaccard of per-seizure top-k lists (plan §3.8)."""
    topks = [
        set(rank_top_k_per_seizure(scores, k))
        for scores in per_seizure_scores
    ]
    if len(topks) < 2:
        return float("nan")
    pair_jaccards: List[float] = []
    for i in range(len(topks)):
        for j in range(i + 1, len(topks)):
            inter = topks[i] & topks[j]
            union = topks[i] | topks[j]
            pair_jaccards.append(len(inter) / len(union) if union else 0.0)
    return float(np.median(pair_jaccards)) if pair_jaccards else float("nan")


def time_shifted_seizure_onsets(
    seizure_onsets: Sequence[float],
    seizure_block_ids: Sequence[str],
    block_windows: Mapping[str, Tuple[float, float]],
    w_pre: float,
    w_post: float,
    edge_buffer: float,
    n_iter: int,
    rng_seed: int,
    exclusion_radius_sec: float = 300.0,
) -> np.ndarray:
    """Generate ``n_iter`` shifted onset matrices (PR-T3-1 v1.1 §5.1).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 helper. Layer A in v2.1
    re-implements time-shifted r_sz null inside Step A.2.

    For each seizure, draw ``n_iter`` random onsets uniformly inside the
    block's safe window ``[block_start + W_pre + edge,
    block_end - W_post - edge]`` while avoiding ± ``exclusion_radius_sec``
    of any true seizure in the same block.

    Returns
    -------
    np.ndarray
        Shape ``(n_iter, n_seizures)`` of shifted absolute-epoch onsets.
        If a block has no admissible window for a given seizure (very
        short block or every shifted draw rejected), that seizure column
        falls back to the original onset (so downstream M1/M2 still get
        a defined value but the surrogate is essentially identity for
        that seizure).
    """
    n_sz = len(seizure_onsets)
    if len(seizure_block_ids) != n_sz:
        raise ValueError("seizure_onsets and seizure_block_ids length mismatch")
    if n_iter <= 0:
        return np.empty((0, n_sz), dtype=float)
    rng = np.random.default_rng(int(rng_seed))
    pre_pad = float(w_pre) + float(edge_buffer)
    post_pad = float(w_post) + float(edge_buffer)
    excl = float(exclusion_radius_sec)
    out = np.empty((int(n_iter), n_sz), dtype=float)
    by_block: Dict[str, List[float]] = defaultdict(list)
    for t, blk in zip(seizure_onsets, seizure_block_ids):
        by_block[blk].append(float(t))
    for j, (t_s, blk) in enumerate(zip(seizure_onsets, seizure_block_ids)):
        if blk not in block_windows:
            out[:, j] = float(t_s)
            continue
        b0, b1 = block_windows[blk]
        lo = b0 + pre_pad
        hi = b1 - post_pad
        if hi <= lo:
            out[:, j] = float(t_s)
            continue
        block_seizures = by_block[blk]
        for it in range(int(n_iter)):
            for _ in range(50):
                cand = rng.uniform(lo, hi)
                if all(abs(cand - st) > excl for st in block_seizures):
                    out[it, j] = cand
                    break
            else:
                out[it, j] = float(t_s)
    return out


def _build_per_method_results(
    per_seizure_scores_by_method: Mapping[str, Sequence[Mapping[str, float]]],
    k_grid: Sequence[Tuple[str, int]],
    n_channels_in_ranking: int,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Compute all (method × aggregation × k) top-k sets.

    Returns
    -------
    dict
        ``results[method][aggregation][k_label] -> sorted list[str]``.
    """
    out: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for method, per_seizure_scores in per_seizure_scores_by_method.items():
        out[method] = {"medianrank": {}, "consensus": {}}
        for k_label, k in k_grid:
            for agg in ("medianrank", "consensus"):
                topk = _aggregate_topk(
                    per_seizure_scores, k, n_channels_in_ranking, agg
                )
                out[method][agg][k_label] = sorted(topk)
    return out


def _build_overlap_table(
    method_results: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    clinical_matched_set: Set[str],
    n_total: int,
) -> Dict[str, Dict[str, float]]:
    """Compute overlap metrics for every (method × aggregation × k).

    Key format: ``{method}_{aggregation}_{k_label}`` (plan §9 schema).
    """
    out: Dict[str, Dict[str, float]] = {}
    for method, by_agg in method_results.items():
        for agg, by_k in by_agg.items():
            for k_label, ranking in by_k.items():
                key = f"{method}_{agg}_{k_label}"
                out[key] = compute_overlap(
                    clinical_matched_set, set(ranking), n_total
                )
    return out


def _baseline_rate_per_channel(
    per_seizure_m1: Sequence[Mapping[str, Mapping[str, float]]],
    channels: Sequence[str],
) -> Dict[str, float]:
    """Mean ``rate_pre`` per channel across seizures (plan §3.3)."""
    out: Dict[str, float] = {}
    for ch in channels:
        rates = [m[ch]["rate_pre"] for m in per_seizure_m1 if ch in m]
        out[ch] = float(np.mean(rates)) if rates else 0.0
    return out


def compute_per_subject_audit(
    *,
    dataset: str,
    subject: str,
    seizure_onsets: Sequence[float],
    seizure_block_ids: Sequence[str],
    block_windows: Mapping[str, Tuple[float, float]],
    hfo_event_times_per_channel: Mapping[str, np.ndarray],
    signal_loader: Optional[SignalLoader],
    sfreq: float,
    clinical_soz: Sequence[str],
    analysis_channels: Sequence[str],
    m2_eligible: bool,
    block_sfreqs: Optional[Mapping[str, float]] = None,
    band: Tuple[float, float] = (80.0, 250.0),
    w_pre: float = 30.0,
    w_post: float = 10.0,
    edge_buffer: float = 2.0,
    null_n_iter: int = 200,
    null_rng_seed: int = 0,
    null_exclusion_radius_sec: float = 300.0,
    signal_loader_path: str = "unknown",
) -> Dict[str, object]:
    """Step 3 orchestrator (PR-T3-1 v1.1 §9 Step 3.2 / §3.3 schema).

    ⚠️  **OBSOLETE 2026-05-03** — v1.1 orchestrator. v2.1 splits
    Layer A (``src/ictal_er_rank.py`` — ER + CUSUM + r_sz) and Layer B
    (``src/data_driven_soz_pivot.py`` — label builder + audit). Kept
    importable so the v1.1 cohort can be reproduced from
    ``per_subject_hfo_rate_obsolete_v1_1/``. Do NOT call from new code.

    The runner:

    1. **Block-boundary prefilter** via ``prefilter_seizures_by_block_window``
       BEFORE any signal load (plan §3.4 + §6 carry-over).
    2. Annotates ``analysis_channels`` against ``clinical_soz`` 3-state.
    3. If ``m2_eligible``, pre-computes the cohort ``power_pre_matrix``
       across kept seizures, runs ``estimate_per_channel_eps`` and
       ``select_m2_eligible_channels`` to drop strict-zero channels.
    4. Per kept seizure: ``compute_hfo_onset_metrics`` (M1 three variants)
       and (if M2) ``compute_er_logratio`` over eligible channels only.
    5. Cross-seizure aggregation: medianrank (primary) + consensus
       (sensitivity) at multiple k values centered on
       ``k_primary = n_clinical_matched``.
    6. Overlap with clinical: jaccard / precision / recall / f1 +
       enrichment using ``len(B)``, NOT k.
    7. Per-seizure consistency (median pairwise Jaccard).
    8. Time-shifted null surrogate (n_iter draws per seizure).

    Returns the per-subject JSON dict (plan §9 Step 3.3 locked schema).
    NO verdict-style fields, NO replacement of ``soz_core_channels.json``.
    """
    if len(seizure_onsets) != len(seizure_block_ids):
        raise ValueError(
            f"seizure_onsets ({len(seizure_onsets)}) vs seizure_block_ids "
            f"({len(seizure_block_ids)}) length mismatch"
        )

    annotation = annotate_clinical_soz(analysis_channels, clinical_soz)
    matched_set = matched_clinical_contacts(analysis_channels, clinical_soz)
    norm_clinical = {_normalize_channel_name(s) for s in clinical_soz if s}
    unmatched = sorted(norm_clinical - matched_set)
    n_clinical_matched = sum(1 for v in annotation.values() if v == SOZ_LABEL)
    clinical_matched_channels = [
        ch for ch, lab in annotation.items() if lab == SOZ_LABEL
    ]

    kept_idx, dropped_idx, drop_reasons = prefilter_seizures_by_block_window(
        seizure_onsets,
        seizure_block_ids,
        block_windows,
        w_pre=w_pre,
        w_post=w_post,
        edge_buffer=edge_buffer,
    )
    kept_onsets = [seizure_onsets[i] for i in kept_idx]
    kept_block_ids = [seizure_block_ids[i] for i in kept_idx]

    n_channels_total = len(analysis_channels)

    # ----- M1 per seizure (three variants) -----
    per_seizure_m1: List[Dict[str, Dict[str, float]]] = []
    for t_s in kept_onsets:
        per_seizure_m1.append(
            compute_hfo_onset_metrics(
                hfo_event_times_per_channel,
                seizure_onset=float(t_s),
                w_pre=w_pre,
                w_post=w_post,
            )
        )

    def _scores_for(metric: str) -> List[Dict[str, float]]:
        return [
            {ch: vals[metric] for ch, vals in m.items()}
            for m in per_seizure_m1
        ]

    per_seizure_scores_by_method: Dict[str, List[Dict[str, float]]] = {
        "M1_raw": _scores_for("M1_raw"),
        "M1_log": _scores_for("M1_log"),
        "M1_pois": _scores_for("M1_pois"),
    }

    # ----- M2 per seizure (eligible-channel-filtered) -----
    m2_ineligible_channels: List[str] = []
    n_seizures_m2_dropped_low_sfreq = 0
    band_hi_with_safety = float(band[1]) * 2.1  # plan §3.4 5% Nyquist margin
    # Per-seizure M2 eligibility: a seizure landing inside a low-sfreq
    # block (e.g. epilepsiae 583 has both 1024 Hz and 256 Hz blocks)
    # would fail compute_er_logratio's NyquistGuardError. Drop those
    # seizures from M2 BEFORE any signal load — M1 still runs because
    # it doesn't need the signal. This forward gate is the M2-side
    # analog of prefilter_seizures_by_block_window.
    if block_sfreqs is not None:
        m2_kept_seizure_indices = [
            i for i, blk in enumerate(kept_block_ids)
            if float(block_sfreqs.get(blk, 0.0)) >= band_hi_with_safety
        ]
        n_seizures_m2_dropped_low_sfreq = len(kept_onsets) - len(m2_kept_seizure_indices)
    else:
        m2_kept_seizure_indices = list(range(len(kept_onsets)))
    m2_kept_onsets = [kept_onsets[i] for i in m2_kept_seizure_indices]

    # Track whether M2 actually ran. ``m2_ran=False`` triggers the NaN /
    # None propagation through every M2-derived field below — Step 4
    # cohort code must be able to distinguish "M2 not run" from
    # "M2 ran and found zero overlap" (P1.1 review fix).
    m2_ran = bool(m2_eligible and signal_loader is not None and m2_kept_onsets)

    if m2_ran:
        # Cohort power_pre matrix (n_m2_seizures, n_channels) over the full
        # analysis_channels set so eps_per_channel covers every channel.
        n_m2_kept = len(m2_kept_onsets)
        power_pre_matrix = np.zeros((n_m2_kept, n_channels_total), dtype=float)
        for s_idx, t_s in enumerate(m2_kept_onsets):
            t_start = float(t_s) - w_pre - edge_buffer
            t_end = float(t_s) + w_post + edge_buffer
            sig, sfreq_ret = signal_loader(t_start, t_end, list(analysis_channels))
            sig = np.asarray(sig, dtype=float)
            if sig.shape[1] != n_channels_total:
                raise ValueError(
                    f"signal_loader returned {sig.shape[1]} channels, "
                    f"expected {n_channels_total}"
                )
            inst_power = _bandpass_power(sig, float(sfreq_ret), band)
            pre_end = int(round(w_pre * float(sfreq_ret)))
            power_pre_matrix[s_idx, :] = np.mean(inst_power[:pre_end, :], axis=0)
        eps_result = estimate_per_channel_eps(power_pre_matrix)
        channel_index = {ch: i for i, ch in enumerate(analysis_channels)}
        eligible_channels, m2_ineligible_channels = select_m2_eligible_channels(
            list(analysis_channels), eps_result, channel_index
        )
        eps_map = {ch: float(eps_result.eps[channel_index[ch]]) for ch in eligible_channels}
        per_seizure_m2: List[Dict[str, float]] = []
        for t_s in m2_kept_onsets:
            scores = compute_er_logratio(
                signal_loader=signal_loader,
                channels=eligible_channels,
                seizure_onset=float(t_s),
                eps_per_channel=eps_map,
                w_pre=w_pre,
                w_post=w_post,
                edge_buffer=edge_buffer,
                band=band,
            )
            per_seizure_m2.append(scores)
        per_seizure_scores_by_method["M2_logratio"] = per_seizure_m2
    else:
        per_seizure_scores_by_method["M2_logratio"] = [
            {} for _ in m2_kept_onsets
        ]

    # ----- Aggregation grid + overlap -----
    k_primary = max(int(n_clinical_matched), 1)
    k_grid: List[Tuple[str, int]] = [
        ("k3", 3),
        ("k5", 5),
        ("k10", 10),
        ("k_primary", k_primary),
        ("k_primary_minus2", max(1, k_primary - 2)),
        ("k_primary_plus2", k_primary + 2),
    ]
    method_results = _build_per_method_results(
        per_seizure_scores_by_method, k_grid, n_channels_total
    )
    clinical_matched_set = set(clinical_matched_channels)
    overlap_table = _build_overlap_table(
        method_results, clinical_matched_set, n_channels_total
    )

    # P1.1: when M2 didn't actually run, replace every M2-derived ranking
    # / overlap entry with None so Step 4 cohort code can distinguish
    # "M2 not computed" from "M2 ran and overlapped zero channels".
    # Returning 0.0 for these subjects would silently drag the cohort
    # M2 median toward 0.
    if not m2_ran:
        for agg in method_results["M2_logratio"]:
            for k_label in method_results["M2_logratio"][agg]:
                method_results["M2_logratio"][agg][k_label] = None
        for key in list(overlap_table):
            if key.startswith("M2_logratio"):
                overlap_table[key] = None

    headline_primary = {
        "H_M1_pois_medianrank_size_matched": overlap_table.get(
            "M1_pois_medianrank_k_primary", {}
        ).get("enrichment", float("nan")),
    }
    if m2_ran:
        m2_overlap = overlap_table.get("M2_logratio_medianrank_k_primary")
        headline_primary["H_M2_logratio_medianrank_size_matched"] = (
            m2_overlap["enrichment"] if m2_overlap is not None else None
        )
        m1_topk = set(method_results["M1_pois"]["medianrank"]["k_primary"])
        m2_topk = set(method_results["M2_logratio"]["medianrank"]["k_primary"])
        concord = m1_topk & m2_topk
        concord_overlap = compute_overlap(
            clinical_matched_set, concord, n_channels_total
        )
        headline_primary["H_concord_M1_M2_size_matched"] = concord_overlap[
            "enrichment"
        ]
    else:
        headline_primary["H_M2_logratio_medianrank_size_matched"] = None
        headline_primary["H_concord_M1_M2_size_matched"] = None

    # ----- Per-seizure consistency -----
    # P1.1: M2 consistency must be None (not NaN/0) when M2 didn't run.
    per_seizure_consistency = {
        "M1_pois_kPrimary": _per_seizure_consistency(
            per_seizure_scores_by_method["M1_pois"], k_primary
        ),
        "M2_logratio_kPrimary": (
            _per_seizure_consistency(
                per_seizure_scores_by_method["M2_logratio"], k_primary
            )
            if m2_ran else None
        ),
    }

    # ----- Time-shifted null -----
    if null_n_iter > 0 and kept_onsets:
        shifted = time_shifted_seizure_onsets(
            kept_onsets,
            kept_block_ids,
            block_windows,
            w_pre=w_pre,
            w_post=w_post,
            edge_buffer=edge_buffer,
            n_iter=null_n_iter,
            rng_seed=null_rng_seed,
            exclusion_radius_sec=null_exclusion_radius_sec,
        )
        shift_enrich_m1: List[float] = []
        shift_enrich_m2: List[float] = []
        for it in range(null_n_iter):
            shifted_onsets = shifted[it].tolist()
            null_m1 = [
                compute_hfo_onset_metrics(
                    hfo_event_times_per_channel, seizure_onset=float(t),
                    w_pre=w_pre, w_post=w_post,
                )
                for t in shifted_onsets
            ]
            null_scores_pois = [
                {ch: vals["M1_pois"] for ch, vals in m.items()}
                for m in null_m1
            ]
            null_topk_m1 = _aggregate_topk(
                null_scores_pois, k_primary, n_channels_total, "medianrank"
            )
            ov = compute_overlap(
                clinical_matched_set, null_topk_m1, n_channels_total
            )
            shift_enrich_m1.append(ov["enrichment"])
        H_M1_obs = headline_primary["H_M1_pois_medianrank_size_matched"]
        H_M1_shift_med = float(np.median(shift_enrich_m1)) if shift_enrich_m1 else float("nan")
        true_over_shift_m1 = (
            H_M1_obs / H_M1_shift_med
            if (H_M1_shift_med and not math.isnan(H_M1_shift_med))
            else float("nan")
        )
        # P1.2: M2 surrogate is intentionally skipped at the per-subject
        # level — each shifted draw would re-bandpass the full block,
        # multiplying the per-seizure load count by ``n_iter`` (= 200
        # by default). The user-flagged review accepted this as a Step 3
        # engineering trade-off but requires a structured boolean flag
        # so Step 4 cohort code can deterministically gate the M2
        # true-vs-shifted reporting (the previous free-text "note_M2"
        # required string parsing).
        time_shifted_null_dict: Dict[str, object] = {
            "n_iter": null_n_iter,
            "rng_seed": null_rng_seed,
            "exclusion_radius_sec": null_exclusion_radius_sec,
            "H_M1_pois_shifted_median": H_M1_shift_med,
            "enrichment_true_over_shift_M1_pois": true_over_shift_m1,
            "m2_surrogate_skipped": True,
            "m2_skip_reason": (
                "per-subject M2 surrogate not computed: each iteration "
                "would re-bandpass the full window for every seizure "
                "(n_iter * n_seizures additional loads); deferred to "
                "Step 4 cohort-level computation if needed"
            ),
            "enrichment_true_over_shift_M2_logratio": None,
        }
    else:
        time_shifted_null_dict = {
            "n_iter": int(null_n_iter),
            "rng_seed": int(null_rng_seed),
            "skipped": "null_n_iter == 0 or no kept seizures",
            "m2_surrogate_skipped": True,
            "m2_skip_reason": "M1 surrogate also skipped (no kept seizures or null_n_iter == 0)",
            "enrichment_true_over_shift_M1_pois": None,
            "enrichment_true_over_shift_M2_logratio": None,
        }

    # P2.1: record the preprocessing path the runner actually used so
    # Step 4 sensitivity comparisons (partial-loader / no-notch vs
    # legacy / full-block + notch) can split deterministically without
    # relying on directory names.
    preprocessing_meta = {
        "signal_loader_path": str(signal_loader_path),
        "band": [float(band[0]), float(band[1])],
        "w_pre": float(w_pre),
        "w_post": float(w_post),
        "edge_buffer": float(edge_buffer),
    }

    return {
        "dataset": dataset,
        "subject": subject,
        "n_seizures_used": len(kept_onsets),
        "n_seizures_dropped": len(dropped_idx),
        "dropped_seizure_reasons": [drop_reasons[i] for i in dropped_idx],
        "n_seizures_m2_dropped_low_sfreq": int(n_seizures_m2_dropped_low_sfreq),
        "preprocessing": preprocessing_meta,
        "n_channels_total": n_channels_total,
        "sfreq": float(sfreq),
        "m2_eligible": bool(m2_eligible),
        "channel_matching": {
            "n_clinical_total": len(list(clinical_soz)),
            "n_clinical_matched": int(n_clinical_matched),
            "n_clinical_unmatched": len(unmatched),
            "unmatched_clinical_names": unmatched,
        },
        "baseline_rate_per_channel": _baseline_rate_per_channel(
            per_seizure_m1, analysis_channels
        ),
        "k_primary_size_matched": int(k_primary),
        "results": method_results,
        "overlap_with_clinical": overlap_table,
        "headline_primary": headline_primary,
        "per_seizure_consistency": per_seizure_consistency,
        "time_shifted_null": time_shifted_null_dict,
        "m2_ineligible_channels": m2_ineligible_channels,
    }


__all__ = [
    "annotate_clinical_soz",
    "matched_clinical_contacts",
    "check_channel_schema_consistency",
    "SOZ_LABEL",
    "NON_SOZ_LABEL",
    "UNKNOWN_LABEL",
    "compute_hfo_onset_metrics",
    "rank_top_k_per_seizure",
    "aggregate_consensus",
    "aggregate_median_rank",
    "NyquistGuardError",
    "FilterPaddingError",
    "PerChannelEps",
    "_bandpass_power",
    "compute_er_logratio",
    "estimate_per_channel_eps",
    "select_m2_eligible_channels",
    "prefilter_seizures_by_block_window",
    "random_expected_jaccard",
    "compute_overlap",
    "time_shifted_seizure_onsets",
    "compute_per_subject_audit",
]

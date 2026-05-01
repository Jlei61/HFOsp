"""Data-driven ictal-onset SOZ audit (PR-T3-1).

Step 0 helpers:

- ``annotate_clinical_soz``: 3-state (SOZ/nonSOZ/unknown) annotation of an
  analysis channel set against a clinical SOZ list, using the canonical
  bipolar-to-any matcher from ``src.event_periodicity``.
- ``matched_clinical_contacts`` / ``check_channel_schema_consistency``:
  unmatched-name normalization and cross-block schema validation
  helpers used by the audit script.

Step 1 helpers (M1 — HFO-onset rate enrichment):

- ``compute_hfo_onset_metrics``: per-seizure / per-channel three M1
  variants (``M1_raw``, ``M1_log``, ``M1_pois``).
- ``rank_top_k_per_seizure``: deterministic top-k by score with NaN
  routed to the bottom and alphabetical tie-break.
- ``aggregate_consensus``: channel must appear in ≥ ``min_fraction`` of
  the per-seizure top-k lists.
- ``aggregate_median_rank``: median rank across seizures (channels
  missing in a seizure receive the worst rank), top-k smallest medians.

Step 2 helpers (M2 — band-power log-ratio enrichment):

- ``_bandpass_power``: Butter order 4 + ``filtfilt`` zero-phase →
  instantaneous (per-sample) power. Raises
  ``NyquistGuardError`` / ``FilterPaddingError`` when the signal can't
  support the requested band.
- ``compute_er_logratio``: per-channel ER log-ratio (post / pre band
  power) with edge-buffer windows, per-channel ε floor, and the
  Nyquist guard at the contract boundary.
- ``estimate_per_channel_eps``: 1st-percentile-across-seizures noise
  floor per channel, lower-bounded by ``floor`` (default ``1e-18``).

See ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Set, Tuple

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
    """Per-channel HFO-onset enrichment (plan §3.3).

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
    """Top-k channels by ``per_channel_score`` (plan §3.5).

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
    lists (plan §3.5).

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
    """Top-k channels by median rank across seizures (plan §3.5).

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
    upper edge (plan §3.4: ``sfreq >= band[1] * 2.1``).
    """


class FilterPaddingError(ValueError):
    """Raised when the input signal is shorter than the ``filtfilt``
    padding required for the requested band's lowest cutoff (plan §3.4:
    ``padlen >= int(1.5 * sfreq / band[0])``).
    """


def _bandpass_power(
    signal_2d: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Bandpass-filter a 2D signal and return per-sample instantaneous power.

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
    """Per-channel ER log-ratio: ``log(P_post + ε) − log(P_pre + ε)`` (plan §3.4).

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
    """Result of ``estimate_per_channel_eps`` (plan §3.4).

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
    """Per-channel noise floor estimate (plan §3.4).

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
    """Filter ``channels`` by ``eps_result.m2_ineligible`` (plan §3.4).

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
]

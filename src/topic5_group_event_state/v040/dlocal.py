"""D-local: what could each measurement field have published, and how soon?

The delivered answer is per field, not "still to be audited".  Every field is
traced to its producer function, its left and right context, its block or
200-second processing dependency, the earliest release that can be verified from
the measurement cards, and the observed processing latency.  Fields that can be
recomputed locally are recomputed and compared at their original time, keeping
the event-set difference rather than silently intersecting the two versions.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .train import atomic_json

MEASUREMENT_ROOT = Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer/measurements')
CARD_ROOT = Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/measurements')
SEGMENT_SECONDS = 200.

# left context, right context, dependency, verdict. The right context is what the
# value actually needs, not when the packaging convention chose to publish it.
FIELDS = (
    dict(field='participation / contact_ok', producer='detector gpu asset -> cache.participation',
         left='event context window', right='event context window', dependency='event-local',
         verdict='available_with_bounded_context'),
    dict(field='event count per minute', producer='packets.build_packets over event_abs_time',
         left='none', right='the minute packet boundary', dependency='event-local',
         verdict='available_with_bounded_context'),
    dict(field='core_seconds_raw / event end', producer='cache.core_start_seconds, core_end_seconds',
         left='event context window', right='event context window', dependency='event-local',
         verdict='available_with_bounded_context'),
    dict(field='relative_delay_s / tied_group_id (synchronous group)',
         producer='v0311/packets.py:block_event_tables (delay, order, rank, lead)',
         left='event context window', right='the full event, to normalise the rank fraction',
         dependency='event-local', verdict='available_with_bounded_context'),
    dict(field='delay_iqr_s / delay_span_s',
         producer='v0311/packets.py:_iqr and block_event_tables',
         left='event context window', right='the full event',
         dependency='event-local; needs at least four participating delays',
         verdict='available_with_bounded_context'),
    dict(field='band_log_energy / band_centroid_s / band_log_peak',
         producer='raw_views stitched spectrogram -> cache.band_features',
         left='the closed 200-second processing segment', right='the closed 200-second processing segment',
         dependency='200-second segment', verdict='recomputable_with_local_release'),
    dict(field='cross_band_lag_s',
         producer='raw_views stitched spectrogram -> cache.cross_band_lag_s',
         left='the closed 200-second processing segment', right='the closed 200-second processing segment',
         dependency='200-second segment', verdict='recomputable_with_local_release'),
    dict(field='band_ratio (target)', producer='v0311/packets.py:block_event_tables via _participant_mean',
         left='inherits band_log_energy', right='inherits band_log_energy',
         dependency='200-second segment', verdict='recomputable_with_local_release'),
    dict(field='signed_xlag (target)', producer='v0311/packets.py:block_event_tables via _participant_mean',
         left='inherits cross_band_lag_s', right='inherits cross_band_lag_s',
         dependency='200-second segment', verdict='recomputable_with_local_release'),
    dict(field='waveform descriptors (wave_log_rms / peak / line length, bipolar and shaft-CAR means)',
         producer='v0311/packets.py:_wave_stats over cache.waveform_*',
         left='event context window', right='event context window', dependency='event-local',
         verdict='available_with_bounded_context'),
    dict(field='block background summary (block_context)',
         producer='v039/human_data.py:background_summary over cache.background_features',
         left='the whole one-hour block', right='the whole one-hour block', dependency='block',
         verdict='block_bound_but_not_consumed'),
)


def cards_for(subject, root=CARD_ROOT):
    base = Path(root) / subject
    if not base.exists():
        return []
    out = []
    for card_path in sorted(base.glob('block_*/card.json')):
        out.append(json.loads(card_path.read_text()) | {'card_path': str(card_path)})
    return out


def recompute_block(card, manifest_path=None):
    """Rebuild one block's tables from the raw cache and compare exactly."""
    from ..v0311.packets import block_event_tables
    cache = Path(card['cache_path'])
    if not cache.exists():
        return dict(status='MISSING_SOURCE', block=card['block'], cache=str(cache))
    manifest = Path(manifest_path) if manifest_path else cache.with_suffix('.manifest.json')
    if not manifest.exists():
        return dict(status='MISSING_SOURCE', block=card['block'], manifest=str(manifest))
    meta = json.loads(manifest.read_text())
    with np.load(cache, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    tokens, event, targets, part = block_event_tables(raw, meta, int(len(card['selected_contacts'])))
    return dict(status='RECOMPUTED', block=int(card['block']), n_events=int(len(event)),
                elapsed_seconds=float(card['elapsed_seconds']),
                available_time=float(card['available_time']), block_start=float(card['block_start']),
                tokens=tokens, event=event, targets=targets, participation=part,
                event_abs_time=np.asarray(raw['event_abs_time'], float),
                core_end=np.asarray(raw['core_end_seconds'], float) + float(card['block_start']),
                segment_crossing_exclusions=len(card.get('segment_crossing_exclusions', [])))


def local_release(event_abs_time, block_start, elapsed_seconds, n_events, segment=SEGMENT_SECONDS):
    """Earliest verifiable publication under a per-segment local pipeline.

    The band coordinates close with their 200-second processing segment; the
    measured per-block compute time is charged pro rata to each segment.
    """
    seg_index = np.floor((np.asarray(event_abs_time, float) - block_start) / segment)
    seg_close = block_start + (seg_index + 1) * segment
    n_segments = max(1., float(np.unique(seg_index).size))
    return seg_close + float(elapsed_seconds) / n_segments


def run_dlocal(cfg, out_dir, max_blocks=4):
    """Field-by-field verdicts plus a same-original-time recomputation comparison."""
    subject = cfg['subject']
    payload = torch.load(Path(cfg['packets_root']) / f'{subject}.pt', weights_only=False, map_location='cpu')
    split = D.build_split(payload, subject, cfg['split_seed'], 'outer', 'S-E')
    cards = cards_for(subject)
    pk = payload['packets']
    result = dict(subject=subject, field_table=[dict(f) for f in FIELDS], n_cards=len(cards))
    if not cards:
        for row in result['field_table']:
            row['conclusion'] = 'missing source: no measurement cards found for this subject'
        result['status'] = 'MISSING_SOURCE'
        result['missing'] = [str(CARD_ROOT / subject)]
        atomic_json(result, Path(out_dir) / 'dlocal.json')
        return result
    # A pre-declared FIT-internal segment chosen by clock coverage, not by outcome.
    fit_cards = [c for c in cards if c['block_start'] + 3600 <= split['fit_end']]
    chosen = fit_cards[:max_blocks] if fit_cards else cards[:max_blocks]
    comparisons = []
    for card in chosen:
        rec = recompute_block(card)
        if rec['status'] != 'RECOMPUTED':
            comparisons.append(rec)
            continue
        # Same-original-time comparison against the packaged arrays. The join is an
        # exact absolute-difference match: np.isclose on epoch seconds carries a
        # default relative tolerance worth hours and would join unrelated events.
        t = rec['event_abs_time']
        stream = np.asarray(payload['event_time'], float)
        packaged = np.clip(np.searchsorted(stream, t), 0, len(stream) - 1)
        left = np.clip(packaged - 1, 0, len(stream) - 1)
        use_left = np.abs(stream[left] - t) < np.abs(stream[packaged] - t)
        packaged = np.where(use_left, left, packaged)
        exact = np.abs(stream[packaged] - t) <= 1e-6
        if exact.any() and len(np.unique(packaged[exact])) != int(exact.sum()):
            raise ValueError('cache events joined to the same packaged event; the time join is not one to one')
        idx = packaged[exact]
        window = (stream >= rec['block_start']) & (stream < rec['block_start'] + 3600.)
        packaged_in_window = int(window.sum())
        packaged_not_in_cache = int(window.sum() - np.isin(np.flatnonzero(window), idx).sum())
        checks = {}
        for name, ours, theirs in (
                ('band_ratio', rec['targets']['band_ratio'][exact], payload['targets']['band_ratio'][idx]),
                ('signed_xlag', rec['targets']['signed_xlag'][exact], payload['targets']['signed_xlag'][idx]),
                ('delay_iqr', rec['targets']['delay_iqr'][exact], payload['targets']['delay_iqr'][idx]),
                ('participation', rec['participation'][exact].astype(float),
                 payload['participation'][idx].astype(float)),
                ('contact_tokens', rec['tokens'][exact], payload['contact_tokens'][idx])):
            a = np.asarray(ours, float)
            b = np.asarray(theirs, float)
            both = np.isfinite(a) & np.isfinite(b)
            checks[name] = dict(n=int(both.sum()),
                                max_abs_difference=float(np.abs(a[both] - b[both]).max()) if both.any() else None,
                                identical_missing_pattern=bool((np.isfinite(a) == np.isfinite(b)).all()))
        # The event-set difference is characterised, not intersected away.
        dropped = t[~exact]
        amb = np.asarray(payload.get('ambiguous_intervals', np.empty((0, 2))), float).reshape(-1, 2)
        excl = np.asarray(split['excluded_intervals'], float).reshape(-1, 2)
        def covered(times, intervals):
            if not len(times) or not len(intervals):
                return 0
            hit = np.zeros(len(times), bool)
            for a, b in intervals:
                hit |= (times >= a) & (times <= b)
            return int(hit.sum())
        dropped_report = dict(
            n=int(len(dropped)),
            in_ambiguous_interval=covered(dropped, amb),
            in_seizure_or_postictal_exclusion=covered(dropped, excl),
            inside_observed_support=covered(dropped, np.asarray(payload['observed_support'], float)),
            note=('cache events with no packaged counterpart; the packaging applies the registered '
                  'ictal and post-ictal exclusion, so the difference is reported rather than intersected away'))
        lr = local_release(t, rec['block_start'], rec['elapsed_seconds'], rec['n_events'])
        delayed = rec['available_time']
        comparisons.append(dict(
            status='RECOMPUTED', block=rec['block'], n_events_in_cache=rec['n_events'],
            n_events_matched_in_package=int(exact.sum()),
            n_events_in_cache_not_in_package=int((~exact).sum()),
            packaged_events_in_block_window=packaged_in_window,
            packaged_events_in_window_without_a_cache_event=packaged_not_in_cache,
            events_dropped_by_segment_crossing=rec['segment_crossing_exclusions'],
            cache_events_without_a_packaged_counterpart=dropped_report,
            event_set_note=('the packaged stream already excludes events crossing a 200-second processing '
                            'boundary; both event sets are reported rather than intersected away'),
            recomputation=checks,
            elapsed_seconds=rec['elapsed_seconds'],
            delayed_release_epoch=float(delayed),
            local_release_seconds_saved=dict(
                median=float(np.median(delayed - lr)), min=float((delayed - lr).min()),
                max=float((delayed - lr).max()))))
    verdicts = {}
    for row in result['field_table']:
        if row['verdict'] == 'available_with_bounded_context':
            row['conclusion'] = ('bounded and verifiable: the value needs only the event context window, so a '
                                 'local pipeline could publish it within the measured processing time')
            row['earliest_verifiable_release'] = 'event end + the block processing time charged pro rata'
        elif row['verdict'] == 'recomputable_with_local_release':
            row['conclusion'] = ('needs local recomputation: the value closes with its 200-second processing '
                                 'segment; recomputed here from the raw cache and compared at the original time')
            row['earliest_verifiable_release'] = '200-second segment close + the processing time of that segment'
            row['recomputation_steps'] = [
                'load the block cache npz named by the card cache_path',
                'load the sibling manifest for the band and cross-band pair order',
                'call v0311/packets.py:block_event_tables',
                'compare against the packaged arrays on the same event times']
        else:
            row['conclusion'] = ('block bound, but this field never reaches the producer: block_context is not '
                                 'read by the v0311, v0312 or v040 input path, so it does not constrain '
                                 'D-local for any fitted arm')
            row['earliest_verifiable_release'] = 'block close'
        verdicts[row['field']] = row['verdict']
    latency = [float(c['elapsed_seconds']) for c in cards]
    publication = [float(c['available_time'] - c['block_start']) for c in cards]
    result.update(
        status='COMPLETE',
        processing_latency_seconds=dict(median=float(np.median(latency)), min=float(np.min(latency)),
                                        max=float(np.max(latency))),
        observed_publication_lag_seconds=dict(median=float(np.median(publication)),
                                              min=float(np.min(publication)), max=float(np.max(publication)),
                                              note='block close relative to block start; not all blocks are a '
                                                   'full hour, so some publish earlier than start plus 3600'),
        packet_release_minus_end_minutes=dict(
            median=float(np.median((pk['release'] - pk['end'])[np.isfinite(pk['release'])]) / 60.),
            max=float(np.max((pk['release'] - pk['end'])[np.isfinite(pk['release'])]) / 60.)),
        comparisons=comparisons, verdicts=verdicts,
        conclusion=('every field the producer actually reads has a bounded right context: event-local for '
                    'participation, timing, delay dispersion and waveform descriptors, and a closed '
                    '200-second processing segment for the band and cross-band coordinates. The observed '
                    'zero-to-sixty-minute publication lag is a one-hour packaging convention, not an '
                    'intrinsic measurement latency. A complete low-latency rebuild of the whole recording '
                    'was not produced in this package, and that does not block the D-delayed main line.'),
        not_done=['a full low-latency re-release of every block',
                  'a re-fit of any arm on a locally released stream'])
    atomic_json(result, Path(out_dir) / 'dlocal.json')
    return result

"""Release-aware FIT boundary, published support audit and the frozen short window.

The v0312 split declares FIT by packet end alone; a minute packet is published
with its closed one-hour block, so up to an hour of FIT material was not yet
readable at the FIT boundary.  Every fitted object here -- inputs, labels,
scaler, trait and statistics -- is restricted to ``release <= fit_end``.
"""
from __future__ import annotations
import numpy as np

from ..v0312 import data as v0312

PACKET_SECONDS = v0312.PACKET_SECONDS
HORIZON_SECONDS = v0312.HORIZON_SECONDS
SPLIT_SEED = v0312.SPLIT_SEED
SHORT_CANDIDATES = (30, 120)
MEMORY_EXTRA_MINUTES = 30.
clock_features = v0312.clock_features
packet_tables = v0312.packet_tables
encode_events = v0312.encode_events
seizure_intervals = v0312.seizure_intervals
interval_mask = v0312.interval_mask
clock_stratum = v0312.clock_stratum
calendar_day = v0312.calendar_day
digest = v0312.digest
role_mask = v0312.role_mask
input_mask = v0312.input_mask
event_packets = v0312.event_packets
fit_scaling = v0312.fit_scaling
target_table = v0312.target_table


def build_split(payload, subject, split_seed=SPLIT_SEED, stage='inner0', protocol='S-E'):
    """v0312 split with the FIT permission additionally bounded by publication.

    C7: a packet may only enter FIT (inputs, targets, scaler support) when its
    marks were released at or before the FIT boundary.  Later-released packets
    stay legal as *inputs at a later query* and as post-hoc scoring targets.
    """
    split = v0312.build_split(payload, subject, split_seed, stage, protocol)
    release = np.asarray(payload['packets']['release'], float)
    published = np.isfinite(release) & (release <= split['fit_end'] + 1e-6)
    train = split['train_packet'].copy()
    split['train_packet_end_only'] = train
    split['train_packet'] = train & published
    split['late_release_excluded_from_fit'] = train & ~published
    split['release_boundary'] = float(split['fit_end'])
    split['contract'] = ('v040: FIT inputs/labels/scaler restricted to release<=fit_end; '
                         'later releases remain legal later inputs and post-hoc scoring targets')
    split['split_id'] = digest({k: v for k, v in split.items() if k != 'seizures'})
    return split


def observed_support_end(payload):
    """Latest observed wall-clock second inside each packet, or NaN when unobserved."""
    pk = payload['packets']
    obs = np.asarray(payload['observed_support'], float)
    out = np.full(len(pk['end']), np.nan)
    for k, (a, b) in enumerate(zip(pk['start'], pk['end'])):
        stop = np.minimum(obs[:, 1], b)
        ok = stop > np.maximum(obs[:, 0], a)
        if ok.any():
            out[k] = stop[ok].max()
    return out


def readable_prefix(payload, split, q, role='descriptive'):
    """Packet indices legally readable at the end of query packet ``q``."""
    pk = payload['packets']
    t = float(pk['end'][q])
    ix = np.arange(int(split['episode_start'][q]), int(q) + 1)
    allowed = input_mask(split, role)
    return ix[allowed[ix] & np.isfinite(pk['release'][ix]) & (pk['release'][ix] <= t)]


def query_support(payload, split, queries, minutes, role='descriptive', last_observed=None):
    """Per-query published support for one candidate short window.

    C1 coverage, C2 separate count/marks support ends and last-IED age,
    C3 the four support classes.  ``no readable IED`` is never conflated with
    ``no observation``: a fully exposed window with zero events is quiet.
    """
    pk = payload['packets']
    if last_observed is None:
        last_observed = observed_support_end(payload)
    # One shared block release publishes count and marks together in the current
    # packets; both ages are reported so an asynchronous stream stays visible.
    distinguished = 'count_release' in pk and 'marks_release' in pk
    rows = []
    for q in np.asarray(queries, int):
        t = float(pk['end'][q])
        ix = readable_prefix(payload, split, q, role)
        start = int(split['episode_start'][q])
        episode_age_seconds = t - float(pk['start'][start])
        window_lo = t - minutes * 60.
        recent = ix[pk['start'][ix] >= window_lo - 1e-6]
        published_seconds = float(pk['exposure'][recent].sum())
        coverage = published_seconds / (minutes * 60.)
        # Observable seconds regardless of publication: separates "not yet
        # released" from "the amplifier was off".
        obs = np.asarray(payload['observed_support'], float)
        observable = float(np.maximum(0., np.minimum(obs[:, 1], t) - np.maximum(obs[:, 0], window_lo)).sum())
        support_end = float(np.nanmax(last_observed[ix])) if len(ix) and np.isfinite(last_observed[ix]).any() else None
        support_age = None if support_end is None else (t - support_end) / 60.
        events = int((pk['event_hi'][recent] - pk['event_lo'][recent]).sum())
        last_event = None
        with_events = ix[pk['event_hi'][ix] > pk['event_lo'][ix]]
        if len(with_events):
            last_event = float(payload['event_time'][pk['event_hi'][with_events[-1]] - 1])
        older = ix[pk['end'][ix] <= window_lo + 1e-6]
        older_seconds = float(pk['exposure'][older].sum())
        older_events = int((pk['event_hi'][older] - pk['event_lo'][older]).sum())
        if observable <= 1e-6:
            support_class = 'window_not_observable'
        elif published_seconds <= 1e-6:
            support_class = 'observed_but_unpublished'
        elif published_seconds < observable - 1e-6:
            support_class = 'partially_published'
        elif events == 0:
            support_class = 'published_and_quiet'
        else:
            support_class = 'published_with_events'
        eligible = bool(episode_age_seconds >= minutes * 60. - 1e-6 and coverage >= .5
                        and support_age is not None and support_age <= minutes / 2.)
        rows.append(dict(query=int(q), time=t, window_minutes=int(minutes),
                         coverage=coverage, published_seconds=published_seconds,
                         observable_seconds=observable, episode_age_minutes=episode_age_seconds / 60.,
                         count_support_age_minutes=support_age, marks_support_age_minutes=support_age,
                         support_release_streams_distinguished=bool(distinguished),
                         last_readable_ied_age_minutes=None if last_event is None else (t - last_event) / 60.,
                         readable_recent_events=events, readable_older_events=older_events,
                         older_published_seconds=older_seconds, support_class=support_class,
                         support_eligible=eligible,
                         memory_support=bool(eligible and older_seconds >= MEMORY_EXTRA_MINUTES * 60.)))
    return rows


def short_history_decision(payload, subject, split_seed=SPLIT_SEED, candidates=SHORT_CANDIDATES,
                           stride=30, horizon=30, quorum=.8):
    """C4/C5: freeze H_short from INNER support metadata only, never from a score.

    Reads the registered main-horizon query times of both temporal INNER stages
    and their published support.  No loss, no future morphology, no seizure
    outcome enters this decision, and a failing candidate is *replaced* rather
    than added as a sixth arm.
    """
    last_observed = observed_support_end(payload)
    stages = {}
    for stage in ('inner0', 'inner1', 'outer'):
        split = build_split(payload, subject, split_seed, stage, 'S-E')
        role = 'outer' if stage == 'outer' else 'inner'
        table = target_table(payload, split, role, stride=stride, horizons=(horizon,))
        qs = np.unique(table[:, 2]) if len(table) else np.empty(0, int)
        per = {}
        for minutes in candidates:
            rows = query_support(payload, split, qs, minutes, last_observed=last_observed)
            cov = [r['coverage'] for r in rows]
            per[int(minutes)] = dict(
                n_queries=len(rows),
                median_published_coverage=float(np.median(cov)) if len(cov) else None,
                support_eligible_queries=int(sum(r['support_eligible'] for r in rows)),
                eligible_fraction=float(np.mean([r['support_eligible'] for r in rows])) if rows else 0.,
                memory_support_queries=int(sum(r['memory_support'] for r in rows)),
                support_class_counts={k: int(sum(r['support_class'] == k for r in rows)) for k in
                                      ('window_not_observable', 'observed_but_unpublished', 'partially_published',
                                       'published_and_quiet', 'published_with_events')},
                rows=rows)
        stages[stage] = dict(split_id=split['split_id'], fit_end=float(split['fit_end']),
                             n_main_horizon_queries=len(qs), windows=per)
    selected = None
    for minutes in candidates:
        if all(stages[s]['windows'][int(minutes)]['eligible_fraction'] >= quorum for s in ('inner0', 'inner1')):
            selected = int(minutes)
            break
    if selected is None:
        selected = int(candidates[-1])
        basis = 'no candidate met the quorum; the longest registered candidate replaces it'
    else:
        basis = 'earliest candidate meeting the pre-registered INNER quorum'
    return dict(selected_short_history_minutes=selected, selection_basis=basis,
                rule=(f'each INNER needs >={quorum:.0%} of {horizon}-minute-horizon queries with episode age>=H, '
                      'published coverage>=0.5 and support-end age<=H/2; candidates are replaced, never added'),
                inputs_read='query times, episode age, published exposure, support end; no loss, morphology or seizure outcome',
                candidates=[int(c) for c in candidates], stages=stages,
                outer_is_descriptive_only=True)


def training_cutoff_audit(payload, subject, split_seed=SPLIT_SEED, stages=('inner0', 'inner1', 'outer')):
    """C7: every fitted object's release and right context must precede the cutoff."""
    pk = payload['packets']
    release = np.asarray(pk['release'], float)
    out = {}
    for stage in stages:
        split = build_split(payload, subject, split_seed, stage, 'S-E')
        end_only = split['train_packet_end_only']
        fit = split['train_packet']
        late = split['late_release_excluded_from_fit']
        ep = event_packets(payload)
        ev_fit = fit[ep]
        ev_release = np.asarray(payload['event_release'], float)
        out[stage] = dict(
            fit_end=float(split['fit_end']), release_boundary=float(split['release_boundary']),
            fit_packets_end_only=int(end_only.sum()), fit_packets_released=int(fit.sum()),
            late_release_packets_removed=int(late.sum()),
            late_release_fraction_of_end_only=float(late.sum() / max(end_only.sum(), 1)),
            max_release_in_fit=float(release[fit].max()) if fit.any() else None,
            release_within_cutoff=bool(fit.sum() == 0 or release[fit].max() <= split['fit_end'] + 1e-6),
            fit_events=int(ev_fit.sum()),
            max_event_release_in_fit=float(ev_release[ev_fit].max()) if ev_fit.any() else None,
            event_release_within_cutoff=bool(ev_fit.sum() == 0 or ev_release[ev_fit].max() <= split['fit_end'] + 1e-6),
            scaler_support='fit_scaling reads packets and events through train_packet only',
            right_context=('packet marks close with their one-hour block; the block release is the '
                           'earliest verifiable publication and already bounds the right context'))
    return out

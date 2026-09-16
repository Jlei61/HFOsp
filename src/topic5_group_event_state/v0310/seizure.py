"""Rare-seizure query grid, clusters and matched pseudo-onsets (spec section 7).

Nothing here touches model state or outcome scores. Seizure labels are used
only for the existing ictal mask, already-past onsets, and post-freeze
evaluation. The 5-minute query grid is rebuilt from past coverage and release
delay alone -- the H1 anchors were filtered by FUTURE pure-interictal targets
and must never be reused as seizure queries.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..v039.human_data import exposure, merge_intervals, phase, subtract_intervals
from .history import past_coverage

POST_ICTAL_SECONDS = 1800.          # ictal onset .. offset + 0.5 h is excluded
CLUSTER_GAP_HOURS = 6.              # primary; 2 h and 12 h are registered sensitivities
PRE_ICTAL_WINDOW = (-7200., -1800.)  # primary -2 h .. -0.5 h
SECONDARY_WINDOWS = {'minus6h_to_minus2h': (-21600., -7200.), 'minus0p5h_to_onset': (-1800., 0.)}
TRAJECTORY = (-21600., 7200.)
MAX_CONTROLS = 10
CLOCK_TOLERANCE = 7200.
COVERAGE_TOLERANCE = 0.10


def load_subject(subject, bundle_root, dataset_root='/data/hfosp_group_event_state_v0_1/dataset'):
    import torch
    data = torch.load(Path(bundle_root) / f'{subject}.pt', map_location='cpu', weights_only=False)
    index = json.loads((Path(dataset_root) / subject / 'index.json').read_text())
    seizures = np.asarray([[s['onset_epoch'], s['offset_epoch']] for s in index['seizures']], float).reshape(-1, 2)
    order = np.argsort(seizures[:, 0])
    return data, seizures[order], [index['seizures'][i] for i in order]


def block_spans(data):
    """[block_start, available_time] for every measurement block."""
    spans = []
    for record in data['source_cards']:
        card = json.loads(Path(record['card_path']).read_text())
        spans.append([card['block_start'], card['available_time']])
    return merge_intervals(spans)


def seizure_excluded_support(data, seizures, spans=None):
    """Observed support minus every 1-hour block touching an ictal/post-ictal span.

    A block that intersects the exclusion interval is dropped whole; no
    sub-block clinical label is invented (spec section 7.2).
    """
    spans = block_spans(data) if spans is None else spans
    exclusion = merge_intervals([[lo, hi + POST_ICTAL_SECONDS] for lo, hi in seizures])
    touched = [list(span) for span in spans
               if np.any((exclusion[:, 0] < span[1]) & (exclusion[:, 1] > span[0]))] if len(exclusion) else []
    return subtract_intervals(data['observed_support'], merge_intervals(touched)), merge_intervals(touched), exclusion


def query_grid(data, seizures, history_hours, min_coverage=0.8):
    """Past-only 5-minute grid: coverage and release delay decide eligibility."""
    bounds = data['phase_boundaries']
    support, dropped, exclusion = seizure_excluded_support(data, seizures)
    grid = np.arange(np.ceil(bounds['20pct'] / 300) * 300, bounds['80pct'], 300.)
    inside = np.array([bool(np.any((support[:, 0] <= t) & (support[:, 1] > t))) for t in grid])
    coverage = np.array([past_coverage(support, t, history_hours) for t in grid])
    eligible = inside & (coverage >= min_coverage)
    previous = np.searchsorted(seizures[:, 0], grid, side='right') - 1
    since = np.where(previous >= 0, grid - seizures[np.maximum(previous, 0), 0], np.inf)
    return dict(times=grid, eligible=eligible, coverage=coverage, phase=phase(grid, bounds),
                inside_support=inside, seconds_since_previous_onset=since,
                excluded_blocks=dropped, exclusion_intervals=exclusion, support=support,
                history_hours=float(history_hours), min_coverage=float(min_coverage),
                rule='fresh 5-minute pre-80%% grid; eligibility uses only past measurement support and '
                     'block-level ictal/post-ictal exclusion; no future interictal target filtering')


def cluster_onsets(seizures, gap_hours=CLUSTER_GAP_HOURS):
    onsets = seizures[:, 0]
    if not len(onsets):
        return []
    breaks = np.r_[0, np.flatnonzero(np.diff(onsets) > gap_hours * 3600) + 1, len(onsets)]
    return [dict(members=list(range(int(a), int(b))), first_onset=float(onsets[a]),
                 last_onset=float(onsets[b - 1]), n_members=int(b - a))
            for a, b in zip(breaks[:-1], breaks[1:])]


def since_stratum(seconds):
    if not np.isfinite(seconds):
        return 'none'
    hours = seconds / 3600.
    if hours < 6:
        return '0p5_to_6h'
    if hours < 24:
        return '6_to_24h'
    return 'ge_24h'


def window_times(grid, centre, window):
    lo, hi = centre + window[0], centre + window[1]
    return (grid['times'] >= lo) & (grid['times'] < hi) & grid['eligible']


def window_profile(grid, seizures, centre, window=PRE_ICTAL_WINDOW):
    """Everything matching compares is measured on the EVALUATION WINDOW.

    The block containing a real onset is excluded whole, so onset-anchored
    coverage is structurally holed and is not comparable to a pseudo-onset's.
    The window the score is actually computed on is.
    """
    mask = window_times(grid, centre, window)
    query_start = centre + window[0]
    previous = np.searchsorted(seizures[:, 0], query_start, side='right') - 1
    since = query_start - seizures[previous, 0] if previous >= 0 else np.inf
    index = int(np.argmin(np.abs(grid['times'] - centre)))
    return dict(phase=str(grid['phase'][index]), clock=float(centre % 86400.),
                n_eligible=int(mask.sum()),
                window_support=float(np.mean(grid['coverage'][mask])) if mask.any() else 0.,
                since_stratum=since_stratum(since), seconds_since_previous_onset=float(since),
                inside_support=bool(grid['inside_support'][index]))


def matched_controls(grid, seizures, cluster_onset, taken, max_controls=MAX_CONTROLS,
                     extra_predicate=None):
    """Pseudo-onsets matched on phase, clock, evaluation-window coverage and
    time-since-previous-seizure stratum. Ordered by time distance, ties
    earliest first. Nothing about the model state enters the choice."""
    times = grid['times']
    target = window_profile(grid, seizures, cluster_onset)
    clock = np.abs(((times % 86400.) - target['clock'] + 43200.) % 86400. - 43200.)
    free = np.array([not np.any((seizures[:, 0] < t + 1800.) & (seizures[:, 1] > t + PRE_ICTAL_WINDOW[0]))
                     for t in times])
    ok = grid['inside_support'] & (grid['phase'] == target['phase']) & (clock <= CLOCK_TOLERANCE) & free
    ok &= np.abs(times - cluster_onset) > 3600.
    reserved = list(taken) + [(cluster_onset + PRE_ICTAL_WINDOW[0], cluster_onset + PRE_ICTAL_WINDOW[1])]
    picked, rejected = [], dict(no_eligible_query=0, coverage=0, stratum=0, overlap=0, extra=0)
    ordered = times[ok][np.lexsort((times[ok], np.abs(times[ok] - cluster_onset)))] if ok.any() else []
    for candidate in ordered:
        profile = window_profile(grid, seizures, float(candidate))
        if not profile['n_eligible']:
            rejected['no_eligible_query'] += 1; continue
        if abs(profile['window_support'] - target['window_support']) > COVERAGE_TOLERANCE:
            rejected['coverage'] += 1; continue
        if profile['since_stratum'] != target['since_stratum']:
            rejected['stratum'] += 1; continue
        if extra_predicate is not None and not extra_predicate(float(candidate)):
            rejected['extra'] += 1; continue
        lo, hi = candidate + PRE_ICTAL_WINDOW[0], candidate + PRE_ICTAL_WINDOW[1]
        if any(lo < b and hi > a for a, b in reserved):
            rejected['overlap'] += 1; continue
        picked.append(float(candidate)); reserved.append((lo, hi))
        if len(picked) >= max_controls:
            break
    return picked, reserved, dict(phase=target['phase'], clock_seconds=target['clock'],
                                  window_support=target['window_support'],
                                  n_eligible_in_window=target['n_eligible'],
                                  since_stratum=target['since_stratum'],
                                  n_candidates=int(ok.sum()), rejected=rejected,
                                  rule='same phase, clock within 2 h, evaluation-window past coverage within '
                                       '10 percentage points, same time-since-previous-seizure stratum, no '
                                       'real seizure in -2h..+0.5h, non-overlapping evaluation windows')


def build_ledger(subject, bundle_root, history_hours=2.0, min_coverage=0.8):
    data, seizures, records = load_subject(subject, bundle_root)
    bounds = data['phase_boundaries']
    grid = query_grid(data, seizures, history_hours, min_coverage)
    usable = seizures[:, 0] < bounds['80pct']
    design_seizures = seizures[usable]
    clusters = cluster_onsets(design_seizures)
    sensitivity = {f'{g}h': len(cluster_onsets(design_seizures, g)) for g in (2, 6, 12)}
    rows = []; taken = []
    for ci, cluster in enumerate(clusters):
        onset = cluster['first_onset']
        onset_phase = str(phase(np.array([onset]), bounds)[0])
        controls, taken, match = matched_controls(grid, seizures, onset, taken)
        pre = window_times(grid, onset, PRE_ICTAL_WINDOW)
        secondary = {name: int(window_times(grid, onset, w).sum()) for name, w in SECONDARY_WINDOWS.items()}
        trajectory = int(window_times(grid, onset, TRAJECTORY).sum())
        # Descriptive only. The block containing the onset is excluded whole, so
        # onset-anchored past coverage is structurally holed and can never be the
        # eligibility criterion; eligibility is per query inside the window.
        support = past_coverage(grid['support'], onset, history_hours)
        window_support = float(np.mean(grid['coverage'][pre])) if pre.any() else 0.
        for mi, member in enumerate(cluster['members']):
            rows.append(dict(
                subject=subject, seizure_id=records[member]['seizure_id'],
                classification=records[member]['classification'], pattern=records[member]['pattern'],
                onset_epoch=float(seizures[member, 0]),
                offset_epoch=float(seizures[member, 1]),
                cluster_id=ci, cluster_position=mi, cluster_first_onset=onset,
                cluster_n_members=cluster['n_members'], is_cluster_first=mi == 0,
                phase=onset_phase if mi == 0 else str(phase(np.array([seizures[member, 0]]), bounds)[0]),
                past_history_hours=history_hours,
                onset_anchored_support_fraction=round(support, 4),
                mean_query_past_support_in_primary_window=round(window_support, 4),
                query_support_ge_80pct=bool(window_support >= .8),
                query_support_ge_90pct=bool(window_support >= .9),
                n_eligible_preictal_queries=int(pre.sum()),
                n_eligible_minus6h_to_minus2h=secondary['minus6h_to_minus2h'],
                n_eligible_minus0p5h_to_onset=secondary['minus0p5h_to_onset'],
                n_eligible_trajectory_queries=trajectory,
                n_matched_controls=len(controls) if mi == 0 else 0,
                matching_status=('MATCHING_LIMITED' if len(controls) < 2 else 'MATCHED') if mi == 0 else 'CLUSTER_MEMBER',
                control_onsets=';'.join(f'{c:.0f}' for c in controls) if mi == 0 else '',
                match_phase=match['phase'], match_clock_seconds=round(match['clock_seconds']),
                match_window_support=round(match['window_support'], 4),
                match_since_stratum=match['since_stratum'],
                match_n_candidates=match['n_candidates'],
                match_rejected_no_query=match['rejected']['no_eligible_query'],
                match_rejected_coverage=match['rejected']['coverage'],
                match_rejected_stratum=match['rejected']['stratum'],
                match_rejected_overlap=match['rejected']['overlap'],
                s_a_eligible=bool(mi == 0 and pre.sum() > 0),
                trajectory_window='-6h..+2h'))
    first = [r for r in rows if r['is_cluster_first']]
    eligible_first = [r for r in first if r['s_a_eligible']]
    training_phase_clusters = [r for r in eligible_first if r['phase'] in ('FIT', 'INNER')]
    selection_clusters = [r for r in eligible_first if r['phase'] == 'SELECTION']
    summary = dict(
        subject=subject, n_raw_seizures=int(len(seizures)),
        n_design_seizures_before_80pct=int(usable.sum()),
        n_clusters=len(clusters), cluster_gap_sensitivity=sensitivity,
        n_eligible_first_onsets=len(eligible_first),
        eligible_by_phase={p: sum(r['phase'] == p for r in eligible_first)
                           for p in ('CALIBRATION', 'FIT', 'INNER', 'SELECTION', 'CLOSED')},
        clusters_outside_design_grid=[dict(cluster_id=r['cluster_id'], phase=r['phase'],
                                           n_members=r['cluster_n_members'])
                                      for r in first if not r['s_a_eligible']],
        eligibility_rule='a cluster enters S-A when at least one 5-minute query inside -2h..-0.5h is '
                         'eligible on past coverage; onset-anchored coverage is descriptive only because '
                         'the block containing the onset is excluded whole',
        n_grid_points=int(len(grid['times'])), n_eligible_queries=int(grid['eligible'].sum()),
        n_excluded_blocks=int(len(grid['excluded_blocks'])),
        excluded_hours=float((grid['excluded_blocks'][:, 1] - grid['excluded_blocks'][:, 0]).sum() / 3600)
        if len(grid['excluded_blocks']) else 0.,
        s_a_status='ESTIMABLE' if eligible_first else 'NOT_ESTIMABLE',
        s_b_status='EXPLORATORY_ESTIMABLE' if len(eligible_first) >= 2 else 'NOT_ESTIMABLE',
        s_b_reason=None if len(eligible_first) >= 2 else 'fewer than two eligible clusters',
        s_c_status='ESTIMABLE' if (training_phase_clusters and selection_clusters) else 'NOT_ESTIMABLE',
        s_c_reason=None if (training_phase_clusters and selection_clusters) else
                   'a truly prospective split needs an eligible cluster in FIT/INNER AND a later one in '
                   'SELECTION; upstream weights are fitted on data up to the 70%% boundary',
        matching_limited_clusters=[r['cluster_id'] for r in first if r['matching_status'] == 'MATCHING_LIMITED'],
        history_hours=history_hours, min_coverage=min_coverage,
        interpretation='counts of eligible clusters, not of independent replications; overlapping long '
                       'histories are retained and never claimed independent')
    return rows, summary, grid

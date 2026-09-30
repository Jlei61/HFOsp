#!/usr/bin/env python3
"""Section 5.2 readouts and finite-time state categories for native (and approximation) trajectories.

All rates use the figure's definition: all-E rate in 10-ms bins (Hz); quiet = <5 Hz for >=20 ms; a finite event is
an activity segment bounded by >=20-ms quiet on both sides, lasting >=20 ms with peak >=20 Hz.  Events are found on the
complete time series first; only events lying entirely inside a window are counted for that window.
"""
import numpy as np
from common import *  # noqa: F401,F403

QUIET_HZ = 5.
QUIET_MIN_MS = 20
EVENT_MIN_MS = 20
EVENT_PEAK_HZ = 20.
HIGH_HZ = 200.
HIGH_MS = 200
BIN_MS = 10
TAIL_S = 4
SPATIAL_HZ = (20., 50., 100.)
DUTY_LEVELS = (.8, .9)
MAIN_SPATIAL = (50., .8)
TRAILING_MAX_MS = 500


def load_chunks(folder, keys=('spikes_1ms', 'regions_1ms', 'field_1ms', 'core15_1ms', 'time_ms')):
    parts = {k: [] for k in keys}; start = None; end = None
    for path in sorted((Path(folder) / 'chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as a:
            s, e = int(a['start_step']), int(a['end_step'])
            if end is not None:
                assert s == end, ('chunk gap', path.name)
            start = s if start is None else start; end = e
            for k in keys:
                if k in a.files:
                    parts[k].append(a[k])
    out = {k: (np.concatenate(v) if v else None) for k, v in parts.items()}
    out['start_step'] = start; out['end_step'] = end
    return out


def rate_10ms(counts_1ms, n):
    """counts_1ms: (T,) spike counts per 1 ms -> per-neuron Hz in 10-ms bins."""
    c = np.asarray(counts_1ms, float); T = len(c) // BIN_MS * BIN_MS
    return c[:T].reshape(-1, BIN_MS).sum(1) / n / (BIN_MS / 1000.)


def runs_of(mask):
    """(start_bin, end_bin_exclusive) of True runs."""
    edges = np.diff(np.r_[0, np.asarray(mask, bool).astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def find_events(r10):
    """Return quiet separators (>=20 ms quiet runs) and complete finite events on the full series (bin indices)."""
    quiet = r10 < QUIET_HZ
    seps = [(a, b) for a, b in runs_of(quiet) if (b - a) * BIN_MS >= QUIET_MIN_MS]
    events = []
    for (a0, b0), (a1, b1) in zip(seps[:-1], seps[1:]):
        s, e = b0, a1                      # activity segment between two separators (may contain <20 ms dips)
        if e <= s:
            continue
        dur = (e - s) * BIN_MS; peak = float(r10[s:e].max())
        events.append(dict(start_bin=int(s), end_bin=int(e), duration_ms=int(dur), peak_hz=peak,
                           qualifies=bool(dur >= EVENT_MIN_MS and peak >= EVENT_PEAK_HZ)))
    return seps, events


def high_rate_entry(r10):
    hi = r10 >= HIGH_HZ; need = HIGH_MS // BIN_MS
    for a, b in runs_of(hi):
        if b - a >= need:
            return dict(onset_bin=int(a), confirmation_bin=int(a + need))
    return None


def window_stats(r10, seps, events, cell_rate10, weights, lo, hi, high):
    """Stats for bins [lo, hi)."""
    w = r10[lo:hi]; nb = hi - lo
    quiet = w < QUIET_HZ
    qruns = [(max(a, lo), min(b, hi)) for a, b in seps if b > lo and a < hi]
    qruns = [(a, b) for a, b in qruns if (b - a) * BIN_MS >= QUIET_MIN_MS]
    ev = [e for e in events if e['qualifies'] and e['start_bin'] >= lo and e['end_bin'] <= hi]
    onsets = [e['start_bin'] for e in ev]
    duty_cells = {}; persistent = {}
    for th in SPATIAL_HZ:
        duty = (cell_rate10[lo:hi] > th).mean(0)
        duty_cells[f'{th:g}Hz'] = duty
        for lvl in DUTY_LEVELS:
            persistent[f'{th:g}Hz_duty{int(lvl * 100)}'] = float(np.average(duty >= lvl, weights=weights))
    last_quiet_end = max([b for a, b in qruns], default=lo)
    trailing = (hi - last_quiet_end) * BIN_MS if qruns else nb * BIN_MS
    high_in = high is not None and lo <= high['confirmation_bin'] <= hi
    return dict(lo_bin=int(lo), hi_bin=int(hi), n_bins=int(nb), quiet_fraction=float(quiet.mean()),
                longest_quiet_ms=int(max([(b - a) for a, b in runs_of(quiet)], default=0) * BIN_MS),
                quiet_runs_ge20ms=int(len(qruns)), n_events=int(len(ev)),
                event_durations_ms=[e['duration_ms'] for e in ev], event_peaks_hz=[e['peak_hz'] for e in ev],
                inter_onset_ms=[int((b - a) * BIN_MS) for a, b in zip(onsets[:-1], onsets[1:])],
                activity_duty=float((w >= QUIET_HZ).mean()), mean_rate_hz=float(w.mean()), max_rate_hz=float(w.max()),
                trailing_activity_ms=int(trailing), persistent_fraction=persistent, high_confirmed_inside=bool(high_in),
                mean_instantaneous_fraction_50Hz=float(np.average(cell_rate10[lo:hi] > 50., axis=1, weights=weights).mean()))


def classify(tail, subs, high, r10_tail):
    """Finite-time category (design 5.2). Returns (category, reasons, sensitivity)."""
    key = f'{MAIN_SPATIAL[0]:g}Hz_duty{int(MAIN_SPATIAL[1] * 100)}'
    all_sub_quiet = all(s['quiet_runs_ge20ms'] >= 1 for s in subs)
    all_sub_persist = all(s['persistent_fraction'][key] > 0 for s in subs)
    reasons = {}
    if tail['n_events'] >= 2 and all_sub_quiet and tail['trailing_activity_ms'] <= TRAILING_MAX_MS:
        cat = 'SELF_LIMITED'
    elif tail['n_events'] == 0 and tail['quiet_fraction'] >= .8:
        cat = 'QUIESCENT'
    elif tail['quiet_runs_ge20ms'] == 0 and all_sub_persist:
        cat = 'PERSISTENT'
    else:
        cat = 'UNRESOLVED'
    reasons = dict(n_events_tail=tail['n_events'], every_subwindow_has_quiet=all_sub_quiet,
                   trailing_activity_ms=tail['trailing_activity_ms'], quiet_fraction_tail=tail['quiet_fraction'],
                   tail_has_no_quiet_run=tail['quiet_runs_ge20ms'] == 0, every_subwindow_persistent=all_sub_persist,
                   high_rate_endpoint_reached=high is not None)
    sens = {}
    for th in SPATIAL_HZ:
        for lvl in DUTY_LEVELS:
            k = f'{th:g}Hz_duty{int(lvl * 100)}'
            alt = all(s['persistent_fraction'][k] > 0 for s in subs)
            if cat == 'PERSISTENT':
                sens[k] = 'PERSISTENT' if alt else 'UNRESOLVED'
            elif cat == 'UNRESOLVED' and tail['quiet_runs_ge20ms'] == 0:
                sens[k] = 'PERSISTENT' if alt else 'UNRESOLVED'
            else:
                sens[k] = cat
    return cat, reasons, sens


def analyse_run(folder, start_step=None, ne=NE, ni=NI, region_counts=None, cell_counts=None, n_grid=20):
    """Complete readout for one native run folder (chunk format of the reference producer)."""
    d = load_chunks(folder)
    if start_step is not None:
        off = int(start_step) - int(d['start_step']); assert off >= 0 and off % 10 == 0
        for k in ('spikes_1ms', 'regions_1ms', 'field_1ms', 'core15_1ms', 'time_ms'):
            if d[k] is not None:
                d[k] = d[k][off // 10:]
        d['start_step'] = int(start_step)
    g = np.load(Path(folder).parent.parent / 'geometry.npz') if (Path(folder).parent.parent / 'geometry.npz').exists() else None
    if region_counts is None:
        region_counts = g['region_counts']
    if cell_counts is None:
        cell_counts = g['cell_e_counts']
    T = len(d['spikes_1ms']) // BIN_MS * BIN_MS
    r10 = rate_10ms(d['spikes_1ms'][:, 0], ne)
    ri10 = rate_10ms(d['spikes_1ms'][:, 1], ni)
    field = d['field_1ms'][:T].astype(float).reshape(-1, BIN_MS, n_grid * n_grid).sum(1)
    cell_rate10 = field / np.asarray(cell_counts, float)[None, :] / (BIN_MS / 1000.)
    reg = d['regions_1ms'][:T].astype(float).reshape(-1, BIN_MS, 6).sum(1) / np.asarray(region_counts, float)[None, :] / (BIN_MS / 1000.)
    core15 = None
    if d['core15_1ms'] is not None:
        c15 = d['core15_1ms'][:T].astype(float).reshape(-1, BIN_MS, 3).sum(1)
        core15 = c15
    seps, events = find_events(r10)
    high = high_rate_entry(r10)
    nb = len(r10); tail_lo = max(0, nb - TAIL_S * 1000 // BIN_MS)
    weights = np.asarray(cell_counts, float)
    tail = window_stats(r10, seps, events, cell_rate10, weights, tail_lo, nb, high)
    subs = [window_stats(r10, seps, events, cell_rate10, weights, tail_lo + k * 100, tail_lo + (k + 1) * 100, high) for k in range(TAIL_S)]
    cat, reasons, sens = classify(tail, subs, high, r10[tail_lo:])
    t0_s = d['start_step'] * DT_MS / 1000.

    def region_rates(lo, hi):
        out = dict(all_E_hz=float(r10[lo:hi].mean()), all_I_hz=float(ri10[lo:hi].mean()),
                   readout175_E_coreA_hz=float(reg[lo:hi, 0].mean()), readout175_E_coreB_hz=float(reg[lo:hi, 1].mean()),
                   readout175_E_surround_hz=float(reg[lo:hi, 2].mean()), readout175_I_coreA_hz=float(reg[lo:hi, 3].mean()),
                   readout175_I_coreB_hz=float(reg[lo:hi, 4].mean()), readout175_I_surround_hz=float(reg[lo:hi, 5].mean()))
        if core15 is not None:
            n15 = np.asarray(CORE15_COUNTS, float)
            out.update(core15_E_coreA_hz=float((core15[lo:hi, 0] / n15[0] / (BIN_MS / 1000.)).mean()),
                       core15_E_coreB_hz=float((core15[lo:hi, 1] / n15[1] / (BIN_MS / 1000.)).mean()),
                       core15_E_surround_hz=float((core15[lo:hi, 2] / n15[2] / (BIN_MS / 1000.)).mean()))
        return out
    tail['rates'] = region_rates(tail_lo, nb)
    for k, s in enumerate(subs):
        s['rates'] = region_rates(tail_lo + k * 100, tail_lo + (k + 1) * 100)
    ev_all = [dict(e, start_s=t0_s + e['start_bin'] * BIN_MS / 1000., end_s=t0_s + e['end_bin'] * BIN_MS / 1000.) for e in events if e['qualifies']]
    result = dict(folder=str(folder), start_s=t0_s, end_s=t0_s + nb * BIN_MS / 1000., n_bins_10ms=int(nb),
                  category=cat, category_reasons=reasons, category_sensitivity=sens,
                  high_rate=None if high is None else dict(onset_s=t0_s + high['onset_bin'] * BIN_MS / 1000.,
                                                           confirmation_s=t0_s + high['confirmation_bin'] * BIN_MS / 1000.),
                  high_rate_reached='REACHED' if high is not None else 'NOT_REACHED_IN_WINDOW',
                  tail_window_s=[t0_s + tail_lo * BIN_MS / 1000., t0_s + nb * BIN_MS / 1000.],
                  tail=tail, subwindows=subs, events_all=ev_all, n_events_all=len(ev_all),
                  definitions=dict(bin_ms=BIN_MS, quiet_hz=QUIET_HZ, quiet_min_ms=QUIET_MIN_MS, event_min_ms=EVENT_MIN_MS,
                                   event_peak_hz=EVENT_PEAK_HZ, high_hz=HIGH_HZ, high_ms=HIGH_MS, spatial_main=MAIN_SPATIAL,
                                   trailing_activity_max_ms_for_self_limited=TRAILING_MAX_MS,
                                   spatial_fraction='neuron-weighted fraction of 20x20 cells whose 10-ms E rate exceeds the threshold for >= duty of the window; grid-mean rate readout, not per-neuron participation'))
    arrays = dict(r10=r10, ri10=ri10, region10=reg, cell_rate10=cell_rate10, core15_10=core15, t0_s=t0_s)
    return result, arrays


CORE15_COUNTS = None


def set_core15_counts(counts):
    global CORE15_COUNTS
    CORE15_COUNTS = np.asarray(counts)


def summary_row(name, job, res):
    t = res['tail']; key = f'{MAIN_SPATIAL[0]:g}Hz_duty{int(MAIN_SPATIAL[1] * 100)}'
    row = dict(name=name, z_source_ms=job.get('z_source_ms'), history_ms=job.get('history_ms'), future=job.get('future'),
               kind=job.get('kind'), freeze_m=job.get('freeze_m'), m_source_ms=job.get('m_source_ms'),
               category=res['category'], high_rate=res['high_rate_reached'],
               high_onset_s=None if res['high_rate'] is None else res['high_rate']['onset_s'],
               high_confirmation_s=None if res['high_rate'] is None else res['high_rate']['confirmation_s'],
               tail_quiet_fraction=t['quiet_fraction'], tail_quiet_runs=t['quiet_runs_ge20ms'], tail_longest_quiet_ms=t['longest_quiet_ms'],
               tail_n_events=t['n_events'], tail_event_median_ms=(float(np.median(t['event_durations_ms'])) if t['event_durations_ms'] else None),
               tail_max_event_ms=(int(max(t['event_durations_ms'])) if t['event_durations_ms'] else None), tail_long_episodes_over_500ms=int(sum(1 for d in t['event_durations_ms'] if d > 500)),
               tail_activity_duty=t['activity_duty'], tail_persistent_50Hz_80=t['persistent_fraction'][key],
               tail_persistent_50Hz_90=t['persistent_fraction'][f'{MAIN_SPATIAL[0]:g}Hz_duty90'],
               tail_persistent_20Hz_80=t['persistent_fraction']['20Hz_duty80'], tail_persistent_100Hz_80=t['persistent_fraction']['100Hz_duty80'],
               tail_all_E_hz=t['rates']['all_E_hz'], tail_coreA_hz=t['rates']['readout175_E_coreA_hz'], tail_coreB_hz=t['rates']['readout175_E_coreB_hz'],
               tail_surround_hz=t['rates']['readout175_E_surround_hz'],
               tail_core15A_hz=t['rates'].get('core15_E_coreA_hz'), tail_core15B_hz=t['rates'].get('core15_E_coreB_hz'),
               sub_quiet_runs=[s['quiet_runs_ge20ms'] for s in res['subwindows']],
               sub_persistent=[s['persistent_fraction'][key] for s in res['subwindows']],
               sub_all_E_hz=[s['rates']['all_E_hz'] for s in res['subwindows']],
               category_sensitivity=res['category_sensitivity'], trailing_activity_ms=t['trailing_activity_ms'])
    return row

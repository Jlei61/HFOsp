#!/usr/bin/env python3
"""Compare both native event families with one independently timed high-state window."""
from pathlib import Path
import csv
import json
import warnings
import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT = BASE / 'early_spatial_v1'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ranked(values, valid):
    out = np.full(values.shape, np.nan)
    ids = np.flatnonzero(valid & np.isfinite(values))
    if len(ids) >= 2:
        out[ids] = (rankdata(values[ids], method='average') - 1) / (len(ids) - 1)
    return out


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return None
    value = spearmanr(x[valid], y[valid]).statistic
    return float(value) if np.isfinite(value) else None


def contact_order(lt, lfp, lo, hi, noise, valid_contacts, fraction=.1):
    ix = (lt >= lo - .020) & (lt < hi + .040)
    times, signal = lt[ix], lfp[ix]
    amplitude = np.ptp(signal, axis=0)
    valid = valid_contacts & (amplitude >= np.maximum(5 * noise, fraction * amplitude.max()))
    onsets = np.full(lfp.shape[1], np.nan)
    for c in np.flatnonzero(valid):
        peak = int(np.argmax(signal[:, c]))
        minimum = float(signal[:peak + 1, c].min())
        threshold = minimum + .2 * (signal[peak, c] - minimum)
        # Last upward crossing before the peak avoids counting an earlier falling tail.
        crossings = np.flatnonzero((signal[1:peak + 1, c] >= threshold) &
                                   (signal[:peak, c] < threshold)) + 1
        if len(crossings):
            onsets[c] = times[crossings[-1]]
    return onsets, ranked(onsets, valid)


def native_order(counts, ncell, lo, hi):
    segment = counts[int(round(lo * 1000)):int(round(hi * 1000))].astype(float)
    total = segment.sum(0)
    peak = gaussian_filter1d(segment / ncell * 1000, 2, axis=0).max(0)
    valid = (total >= 5) & (peak >= 20)
    onset = np.argmax(segment.cumsum(0) >= .1 * total, axis=0).astype(float)
    onset[~valid] = np.nan
    return ranked(onset, valid)


def main():
    a = np.load(BASE / 'runs/continuous_refill_release.npz')
    run = json.loads((BASE / 'runs/continuous_refill_release.json').read_text())
    substrate = json.loads((BASE.parent / 'historical_manual_hard_native_z_v1/substrate.json').read_text())
    core_radius = float(substrate['historical_core_radius_mm'])
    distance = np.linalg.norm(a['positions_e'][:, None] - a['centers_mm'][None], axis=2)
    core_n = (distance <= core_radius).sum(0)
    e = a['rate_e_hz'].reshape(-1, 50).mean(1)
    t = (np.arange(len(e)) + .5) * .005
    lt = a['lfp_time_ms'] / 1000
    lfp = a['lfp_effective']
    assert lfp.shape[1] == len(a['contact_names']) == len(a['contact_xy'])
    quiet = (lt >= .5) & (lt < 8) & (np.interp(lt, t, e) < 1)
    b = np.median(lfp[quiet], axis=0)
    noise = 1.4826 * np.median(abs(lfp[quiet] - b), axis=0)
    baseline = (lt >= .5) & (lt < 8)
    baseline_power = ((lfp[baseline] - b) ** 2).mean(0)

    # Recover the first actual 200-Hz run; detection occurs only after 200 ms.
    rate10 = a['rate_e_hz'].reshape(-1, 100).mean(1)
    changes = np.diff(np.r_[False, rate10 >= 200, False].astype(int))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    start = int(starts[np.flatnonzero(stops - starts >= 20)[0]])
    onset = start * .010
    assert np.isclose(onset + .200, run['first_trigger_ms'] / 1000)
    early = [onset, onset + .250]
    assert .5 < early[0] < early[1] < run['restore_start_ms'] / 1000

    contract = dict(
        source_run=str(BASE / 'runs/continuous_refill_release.npz'), simulation_rerun=False,
        question='Does a single model high-state entry share its early high-power contacts with A-leading or B-leading recurrent self-limited events?',
        statistical_units='Events nested in one noise realization and one fixed topology; one high-state entry. Parameter panel uses three paired noise realizations per cell.',
        event_detection='All global-E 5-ms intervals above 1 Hz (close one-bin gaps), peak >=20 Hz, wholly within 0.5-8 s. Intervals have quiet boundaries; no ranking by spatial agreement.',
        family_assignment='20%-of-local-peak regional E-rate onset (minimum 5 Hz), 1-ms rates smoothed sigma=2 ms. Both regional onsets required; separation >=5 ms => A-leading or B-leading; otherwise unclassified.',
        family_region_radius_mm=1.75,
        family_scope='Stored 1.75-mm core-neighborhood summaries, not the physical 1.5-mm threshold support, and not proof of causal ignition.',
        physical_core_radius_mm=core_radius, physical_core_n_E=core_n.tolist(),
        early_window_s=early, high_activity_start_s=onset, detection_time_s=run['first_trigger_ms'] / 1000,
        early_window_selection='First 250 ms of the first 200-Hz run lasting at least 200 ms; same full-field window for both event families, no spatial optimization and no patient-to-model time scaling.',
        power_definition='Delta P_c = mean_early[(L_c-b_c)^2] - mean_0.5-8s[(L_c-b_c)^2]; b_c is the 0.5-8s quiet-bin median. Native applied-current virtual SEEG proxy, not patient 1-150 Hz EEG band power.',
        power_units='(virtual SEEG proxy units)^2; mean-square power increment, not integrated energy or spike rate.',
        contact_participation='Event peak-to-trough >= max(5 quiet-bin MAD, 10% of event maximum channel amplitude); valid contact and upward crossing required.',
        contact_order='Last upward 20%-of-amplitude crossing before each event peak; masked within-event 0-1 ranks.',
        template='Equal-event median normalized rank; contact/cell must be observed in at least half the family events.',
        correspondence='Signed Spearman rho between earlier recruitment (-rank) and early power increment, scored at directly observed contacts without field interpolation or sign flipping.',
        native_check='Direct 20x20 spike grid: first 10% cumulative event spikes (>=5 spikes and peak >=20 Hz), versus early mean E-rate increment over 0.5-8s. No electrode readout, no spatial smoothing; this is a rate comparison, not a second energy measurement.',
        sensitivity='Early windows starting at t0-250 ms, t0, t0+250 ms, durations 125/250/500 ms if before refill; contact participation fractions 0.05/0.1/0.2.',
        interpretation='Compare both families honestly; a matching smoothed field cannot establish a shared core cause, independent patient validation, or two separate high-state realizations.'
    )
    write('analysis_contract.json', contract)

    def power_for(lo, hi):
        ix = (lt >= lo) & (lt < hi)
        return ((lfp[ix] - b) ** 2).mean(0) - baseline_power

    delta_power = power_for(*early)
    native_baseline = a['field_e_count_1ms'][500:8000].sum(0) / a['cell_e_counts'] / 7.5
    native_early = a['field_e_count_1ms'][round(early[0]*1000):round(early[1]*1000)].sum(0) / a['cell_e_counts'] / .25
    native_delta = native_early - native_baseline
    reg = gaussian_filter1d(a['region_spikes_1ms'][:, :2] / a['region_counts'][:2] * 1000, 2, axis=0)
    active = binary_closing(e > 1, structure=np.ones(2))
    changes = np.diff(np.r_[False, active, False].astype(int))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    rows, ranks, native_ranks, sensitivities = [], [], [], []
    for i, j in zip(starts, stops):
        lo, hi = i * .005, j * .005
        if lo < .5 or hi > 8 or e[i:j].max() < 20:
            continue
        local = reg[round(lo*1000):round(hi*1000)]
        onsets = []
        for c in range(2):
            ix = np.flatnonzero(local[:, c] >= max(5, .2 * local[:, c].max()))
            onsets.append(float(ix[0]) if len(ix) else None)
        lag = onsets[1] - onsets[0] if all(x is not None for x in onsets) else None
        group = 'A' if lag is not None and lag >= 5 else 'B' if lag is not None and lag <= -5 else 'unclassified'
        _, rank = contact_order(lt, lfp, lo, hi, noise, a['valid_contacts'])
        nrank = native_order(a['field_e_count_1ms'], a['cell_e_counts'], lo, hi)
        rows.append(dict(event=len(rows)+1, start_s=float(lo), end_s=float(hi),
            family=group, B_minus_A_local_onset_ms=lag, valid_contacts=int(np.isfinite(rank).sum()),
            direct_contact_rho=correlation(-rank, delta_power),
            native_cell_rho=correlation(-nrank, native_delta)))
        ranks.append(rank); native_ranks.append(nrank)
        sensitivities.append([contact_order(lt, lfp, lo, hi, noise, a['valid_contacts'], f)[1]
                              for f in [.05, .1, .2]])
    ranks = np.array(ranks); native_ranks = np.array(native_ranks)
    templates, ntemplates, summary, support = [], [], {}, []
    for family in ['A', 'B']:
        ix = np.array([row['family'] == family for row in rows])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            template = np.nanmedian(ranks[ix], axis=0)
            ntemplate = np.nanmedian(native_ranks[ix], axis=0)
        freq = np.mean(np.isfinite(ranks[ix]), axis=0)
        template[freq < .5] = np.nan
        ntemplate[np.mean(np.isfinite(native_ranks[ix]), axis=0) < .5] = np.nan
        templates.append(template); ntemplates.append(ntemplate); support.append(freq)
        summary[family] = dict(n_events=int(ix.sum()), template_valid_contacts=int(np.isfinite(template).sum()),
            template_contact_rho=correlation(-template, delta_power),
            template_native_rho=correlation(-ntemplate, native_delta))
        for key in ['direct_contact_rho', 'native_cell_rho']:
            vals = [row[key] for row in rows if row['family'] == family and row[key] is not None]
            summary[family][key+'_median'] = float(np.median(vals)) if vals else None
            summary[family][key+'_positive_events'] = int(np.sum(np.array(vals) > 0))
            summary[family][key+'_estimable_events'] = len(vals)
    sensitivity = []
    for offset in [-.25, 0, .25]:
        for length in [.125, .25, .5]:
            lo, hi = onset + offset, onset + offset + length
            if hi > run['restore_start_ms']/1000:
                continue
            dp = power_for(lo, hi)
            sensitivity.append(dict(window_start_s=lo, window_end_s=hi,
                A=correlation(-templates[0], dp), B=correlation(-templates[1], dp)))
    participation_checks = []
    for k, fraction in enumerate([.05, .1, .2]):
        for family in ['A', 'B']:
            rr = np.array([s[k] for s,row in zip(sensitivities, rows) if row['family'] == family])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(rr, axis=0)
            med[np.mean(np.isfinite(rr), axis=0) < .5] = np.nan
            participation_checks.append(dict(fraction=fraction, family=family,
                template_contact_rho=correlation(-med, delta_power), n_contacts=int(np.isfinite(med).sum())))

    # Reuse all 27 completed parameter results. No smoothing or filling a denser grid.
    tau_values = [2500., 5000., 10000.]
    thresholds = [75., 95.19851312666987, 120.]
    with (BASE / 'transition_times.csv').open() as f:
        parameter_rows = list(csv.DictReader(f))
    assert len(parameter_rows) == 27
    matrix, fractions, counts = np.empty((3,3)), np.empty((3,3)), np.empty((3,3), int)
    for y, threshold in enumerate(thresholds):
        for x, tau in enumerate(tau_values):
            selected = [r for r in parameter_rows if abs(float(r['I_th'])-threshold)<1e-6 and float(r['tau_z_ms']) == tau]
            assert len(selected) == 3 and len({r['seed'] for r in selected}) == 3
            matrix[y,x] = np.mean([float(r['time_without_transition_restricted_24s']) for r in selected])
            fractions[y,x] = np.mean([int(r['transition_by_24s']) for r in selected])
            counts[y,x] = len(selected)

    np.savez_compressed(OUT/'analysis_arrays.npz', contact_xy=a['contact_xy'],
        contact_names=a['contact_names'], centers_mm=a['centers_mm'], positions_e=a['positions_e'],
        template_rank=np.array(templates), template_support=np.array(support),
        native_template_rank=np.array(ntemplates), native_early_delta_hz=native_delta,
        early_delta_power=delta_power, baseline_power=baseline_power, quiet_baseline=b,
        event_contact_rank=ranks, event_native_rank=native_ranks,
        parameter_tau_s=np.array(tau_values)/1000, parameter_threshold=np.array(thresholds),
        restricted_mean_time_s=matrix, transition_fraction=fractions, parameter_n=counts)
    with (OUT/'events.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    direct_rows = [dict(contact=str(name), x_mm=float(xy[0]), y_mm=float(xy[1]),
        rank_A=None if not np.isfinite(templates[0][k]) else float(templates[0][k]),
        rank_B=None if not np.isfinite(templates[1][k]) else float(templates[1][k]),
        early_delta_power=float(delta_power[k])) for k,(name,xy) in enumerate(zip(a['contact_names'],a['contact_xy']))]
    with (OUT/'contact_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(direct_rows[0]));writer.writeheader();writer.writerows(direct_rows)
    write('summary.json', dict(groups=summary, n_events=len(rows),
        n_unclassified=sum(row['family']=='unclassified' for row in rows), early_window_s=early,
        early_window_sensitivity=sensitivity, participation_sensitivity=participation_checks,
        actual_core_radius_mm=core_radius, physical_core_n_E=core_n.tolist(),
        parameter_tau_s=(np.array(tau_values)/1000).tolist(), parameter_threshold=thresholds,
        parameter_restricted_mean_time_s=matrix.tolist(), parameter_transition_fraction=fractions.tolist(),
        statistical_scope='One detailed SNN realization; same early high-state map compared with both event families; three noise seeds per parameter cell on one topology.'))
    print(json.dumps(dict(groups=summary, n_events=len(rows), early_window_s=early,
        sensitivity=sensitivity, participation_sensitivity=participation_checks,
        core_radius_mm=core_radius, core_n=core_n.tolist())))


if __name__ == '__main__':
    main()

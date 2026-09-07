#!/usr/bin/env python3
"""Bound core activity from native cell totals; no causal or fit qualification."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_node_dualmode import sheet_bin_indices


def activity_bounds(counts, population, members):
    """Sharp per-cell bounds for distinct active members, with unknown identities."""
    arrays = [np.asarray(x) for x in (counts, population, members)]
    if any(not np.issubdtype(x.dtype, np.integer) for x in arrays):
        raise ValueError('counts and populations must be integers')
    a, n, g = [x.astype(np.int64, copy=False) for x in arrays]
    if n.shape != g.shape or a.shape[1:] != n.shape:
        raise ValueError('cell dimensions differ')
    if np.any(g < 0) or np.any(g > n) or np.any(a < 0) or np.any(a > n):
        raise ValueError('activity exceeds the available population')
    axes = tuple(range(1, a.ndim))
    return (np.maximum(0, a - (n - g)).sum(axis=axes),
            np.minimum(a, g).sum(axis=axes))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, default=6)
    args = ap.parse_args()
    root = ROOT / 'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v3'
    review_path = root / f'rounds/{args.round:03d}/raw_propagation_review.json'
    review = json.loads(review_path.read_text())
    hashes = {str(review_path): sha(review_path), str(Path(__file__)): sha(__file__)}
    rows = []
    for item in review['candidates']:
        cid = item['candidate_id']
        source = root / f'rounds/{args.round:03d}/combined_{cid}.json'
        candidate = json.loads(source.read_text())['candidates'][0]
        diagnostic = root / 'raw_propagation_audit' / cid / 'summary.json'
        expected = review['source_hashes'][str(diagnostic)]
        if sha(diagnostic) != expected:
            raise RuntimeError(f'diagnostic source changed: {diagnostic}')
        events = json.loads(diagnostic.read_text())['metrics']
        hashes.update({str(source): sha(source), str(diagnostic): expected})
        centers = np.asarray(candidate['candidate']['node_field']['centers_mm'])
        for unit in candidate['units']:
            wp = Path(unit['worker_path'])
            if sha(wp) != unit['worker_sha256']:
                raise RuntimeError(f'worker changed: {wp}')
            worker = json.loads(wp.read_text())
            npz = Path(worker['arrays']['path'])
            if sha(npz) != worker['arrays']['sha256']:
                raise RuntimeError(f'array source changed: {npz}')
            hashes.update({str(wp): unit['worker_sha256'], str(npz): worker['arrays']['sha256']})
            with np.load(npz) as z:
                counts = z['sheet_activity_counts']
                xy, h, dv = z['positions_E'], z['h'], z['delta_vtheta']
                frame_ms, bin_mm, sheet_mm = [float(z[k]) for k in
                    ('sheet_activity_frame_ms', 'source_bin_mm', 'source_sheet_mm')]
            if (frame_ms, bin_mm, sheet_mm) != (2., 1., 20.):
                raise RuntimeError('native movie contract changed')
            if not np.all(np.isin(h, [0., 1.])) or np.count_nonzero(h) != 1499:
                raise RuntimeError('expected the frozen binary VTH field')
            if not np.array_equal(h != 0, dv != 0):
                raise RuntimeError('core membership and actual VTH perturbation differ')
            bins, size = sheet_bin_indices(xy, bin_mm=bin_mm, sheet_mm=sheet_mm)
            population = np.bincount(bins, minlength=size * size).reshape(size, size)
            nearest = np.argmin(((xy[:, None, :] - centers[None, :, :]) ** 2).sum(2), axis=1)
            masks = {'target_all': h > 0, 'core_0': (h > 0) & (nearest == 0),
                     'core_1': (h > 0) & (nearest == 1),
                     'threshold_lowered': dv < 0, 'threshold_raised': dv > 0}
            times = (np.arange(len(counts)) + .5) * frame_ms
            windows = [('post_burnin', -1, 500., len(counts) * frame_ms)]
            windows += [('observed_event', e['event_index'], e['start_ms'], e['stop_ms'])
                        for e in events if e['seed'] == unit['seed']]
            total = counts.astype(np.int64).sum(axis=(1, 2))
            for group, mask in masks.items():
                members = np.bincount(bins[mask], minlength=size * size).reshape(size, size)
                lo, hi = activity_bounds(counts, population, members)
                ng, nb = int(mask.sum()), int((~mask).sum())
                if not ng or not nb:
                    raise RuntimeError('empty group or complement')
                for scope, event, start, stop in windows:
                    use = (times >= start) & (times < stop)
                    nf = int(use.sum())
                    if not nf:
                        raise RuntimeError('empty native observation window')
                    lower, upper, mass = int(lo[use].sum()), int(hi[use].sum()), int(total[use].sum())
                    def ratio(value):
                        return (value / ng) / ((mass - value) / nb) if mass > value else None
                    rows.append({'candidate_id': cid, 'seed': unit['seed'], 'scope': scope,
                        'event_index': event, 'start_ms': start, 'stop_ms': stop,
                        'group': group, 'n_group_neurons': ng, 'n_complement_neurons': nb,
                        'n_frames': nf, 'total_active_neuron_bins': mass,
                        'group_active_neuron_bins_lower': lower, 'group_active_neuron_bins_upper': upper,
                        'group_activity_probability_lower': lower / (ng * nf),
                        'group_activity_probability_upper': upper / (ng * nf),
                        'group_to_complement_per_neuron_ratio_lower': ratio(lower),
                        'group_to_complement_per_neuron_ratio_upper': ratio(upper)})
    out = root / f'rounds/{args.round:03d}/core_activity_bounds'
    out.mkdir(exist_ok=True)
    table = out / 'core_activity_bounds.csv'
    with table.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summaries = []
    for cid in [x['candidate_id'] for x in review['candidates']]:
        for group in masks:
            rr = [r for r in rows if r['candidate_id'] == cid and r['group'] == group and r['scope'] == 'post_burnin']
            lower = [r['group_to_complement_per_neuron_ratio_lower'] for r in rr]
            upper = [r['group_to_complement_per_neuron_ratio_upper'] for r in rr]
            summaries.append({'candidate_id': cid, 'group': group, 'n_networks': len(rr),
                'per_neuron_ratio_lower_min': min(lower), 'per_neuron_ratio_lower_max': max(lower),
                'per_neuron_ratio_upper_min': min(upper), 'per_neuron_ratio_upper_max': max(upper),
                'networks_with_lower_bound_above_one': sum(v > 1. for v in lower)})
    result = {'status': 'DESCRIPTIVE_CORE_ACTIVITY_BOUNDS_NOT_CAUSAL', 'round': args.round,
        'formula': 'Per cell: max(0, active - nonmembers) <= active members <= min(active, members). Sum over cells and time bins.',
        'quantity': 'Distinct active E-neuron time bins; not spikes, event counts, source probability, or causal credit.',
        'groups': 'Actual binary field; cores split by nearest center; threshold signs use saved delta_vtheta added to baseline VTH.',
        'event_windows': 'Same readout-defined physical windows; frame-center membership. No event redefinition.',
        'n_rows': len(rows), 'summaries': summaries, 'source_hashes': hashes,
        'output_csv': str(table), 'output_csv_sha256': sha(table), 'live_search_changed': False,
        'scientific_qualification': False, 'fig5_hold_released': False,
        'limitations': ['Bounds do not identify individual active neurons or propagation origin.',
                       'Core activity enrichment does not establish that cores are necessary or sufficient.',
                       'Shared seeds and adaptively selected candidates are not independent validation.',
                       'Event windows may overlap; do not sum event rows into whole-run activity.']}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'rows': len(rows), 'target_all':
        [s for s in summaries if s['group'] == 'target_all']}))


if __name__ == '__main__':
    main()

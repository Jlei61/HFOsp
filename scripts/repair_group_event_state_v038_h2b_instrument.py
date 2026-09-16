#!/usr/bin/env python3
"""Reassess original frozen H2b features with measured readout qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_group_event_state_v038_dual_credit import file_hash
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.h2b import run_h2b_outcomes, _phase_at, _phase_circular_shift
from src.topic5_group_event_state.v035.contracts import DATASET_ROOT
from src.topic5_group_event_state.v035.seizure_transfer import _hazard_rows_observed_support


def episode_support(index, frozen):
    bounds = json.loads(str(frozen['phase_boundaries_json'].item()))
    seizures = sorted((row for row in index['seizures'] if row['onset_epoch'] < bounds['80pct']), key=lambda r: r['onset_epoch'])
    onsets = np.unique([row['onset_epoch'] for row in seizures])
    phase = frozen['phase'].astype(str); times = frozen['anchor_time']
    phase_hi = {'CALIBRATION': bounds['20pct'], 'FIT': bounds['60pct'], 'INNER': bounds['70pct'], 'SELECTION': bounds['80pct']}
    ar, bins, y, _weights = _hazard_rows_observed_support(times, phase, frozen['observed_support_bounds'], phase_hi, onsets)
    selection = np.flatnonzero(phase == 'SELECTION')
    _shifted, shift_valid = _phase_circular_shift(frozen['S_event_N'], times, selection, 21600.0)
    supports = {}
    for label, use in [('full', phase[ar] == 'SELECTION'),
                       ('time_shift', (phase[ar] == 'SELECTION') & shift_valid[ar])]:
        positive = np.flatnonzero(use & (y > 0))
        next_index = np.searchsorted(onsets, times[ar[positive]], side='right')
        if np.any(next_index >= len(onsets)):
            raise ValueError('positive hazard row has no corresponding onset')
        matched = onsets[next_index]
        if np.any(matched > times[ar[positive]] + (bins[positive] + 1) * 300 + 1e-6):
            raise ValueError('positive hazard row maps to an incompatible onset')
        supports[label] = {'anchors': int(np.unique(ar[use]).size), 'person_period_rows': int(use.sum()),
                           'positive_rows': int(positive.size), 'distinct_onsets': np.unique(matched).tolist()}
    cluster_results = {}
    for hours in (1, 6, 24):
        clusters = []
        for row in seizures:
            onset, offset = float(row['onset_epoch']), float(row['offset_epoch'])
            if clusters and onset - clusters[-1]['end'] <= hours * 3600:
                clusters[-1]['end'] = max(clusters[-1]['end'], offset)
                clusters[-1]['onsets'].append(onset)
            else:
                clusters.append({'start': onset, 'end': offset, 'onsets': [onset]})
        counts = {}
        for label, support in supports.items():
            matching = set(support['distinct_onsets'])
            counted = []
            for i, cluster in enumerate(clusters):
                if matching.intersection(cluster['onsets']):
                    cluster_phase = _phase_at(np.asarray([cluster['start'], np.nextafter(cluster['end'], -np.inf)]), bounds)
                    counted.append({'cluster': i, 'onsets': cluster['onsets'], 'crosses_phase': len(set(cluster_phase)) > 1})
            counts[label] = {'clusters_with_scored_onsets': len(counted),
                             'phase_contained_clusters': sum(not row['crosses_phase'] for row in counted), 'clusters': counted}
        cluster_results[str(hours)] = counts
    return {'support': supports, 'cluster_gap_hours_sensitivity': cluster_results,
            'interpretation': 'operational temporal clusters; not proof of physiological independence'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-card', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ictal-cache-root', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.ictal_cache_root.is_dir():
        raise FileNotFoundError(args.ictal_cache_root)
    old = json.loads(args.source_card.read_text()); freeze_path = Path(old['freeze_card'])
    freeze = json.loads(freeze_path.read_text()); feature_path = Path(freeze['feature_path'])
    paths = [args.source_card, freeze_path, feature_path, DATASET_ROOT / old['subject'] / 'index.json']
    index = json.loads(paths[-1].read_text())
    subject_root = DATASET_ROOT / old['subject']
    paths.extend([subject_root / 'scalars.npz',
                  subject_root / index['arrays']['participation']['file'],
                  subject_root / index['arrays']['relative_delay']['file']])
    for prefix in ('source', 'event_source', 'grid_source'):
        checkpoint = freeze.get(prefix + '_checkpoint')
        if checkpoint:
            path = Path(checkpoint)
            if file_hash(path) != freeze[prefix + '_checkpoint_sha256']:
                raise ValueError(f'upstream checkpoint mismatch: {prefix}')
            paths.append(path)
    exact = args.ictal_cache_root / f"{old['subject']}.npz"
    if exact.exists():
        paths.append(exact)
    hashes = {str(p): file_hash(p) for p in paths}
    result = run_h2b_outcomes(old['subject'], old['seed'], freeze_dir=freeze_path.parent,
                              out_dir=args.output.parent, extend_budget=not args.preflight_only,
                              instrument_controls=True, ictal_cache_root=args.ictal_cache_root)
    with np.load(feature_path, allow_pickle=False) as archive:
        frozen = {key: archive[key] for key in archive.files}
    index = json.loads((DATASET_ROOT / old['subject'] / 'index.json').read_text())
    result['episode_support_audit'] = episode_support(index, frozen)
    result.update(status='COMPLETE', repair_source_card=str(args.source_card),
                  repair_source_sha256=hashes[str(args.source_card)], repair_input_hashes=hashes,
                  repair_upstream_weights_updated=False, repair_preflight_only=args.preflight_only,
                  repair_ictal_energy_cache_available=exact.exists(),
                  repair_ictal_energy_cache_path=str(exact), repair_elapsed_seconds=time.time() - started,
                  claim_tier='reassessment of previously examined selection; no independent confirmation')
    for path, digest in hashes.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f'input changed during H2b instrument repair: {path}')
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'subject', 'seed', 'repair_elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()

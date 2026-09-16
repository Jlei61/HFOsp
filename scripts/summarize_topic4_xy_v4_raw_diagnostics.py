#!/usr/bin/env python3
"""Numerical roll-up of the automatic raw-propagation diagnostics for one V4 round.

Verifies the watcher completion markers, summarises every observed event of every
expanded candidate (no event selection), runs the descriptive native boundary audit
and writes rounds/NNN/raw_propagation_numeric_audit.json. Visual QA is recorded
separately per candidate in figures/visual_qa.json; this script never qualifies.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4'
STATE = OUT / 'raw_propagation_audit'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quant(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {'n': int(len(v)), 'median': float(np.median(v)), 'q90': float(np.quantile(v, .9)), 'max': float(v.max()), 'min': float(v.min())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--round', type=int, required=True); args = ap.parse_args()
    stage = OUT / 'rounds' / f'{args.round:03d}'
    analysis = json.loads((stage / 'analysis.json').read_text())
    rows, hashes = [], {str(stage / 'analysis.json'): sha(stage / 'analysis.json')}
    for item in analysis['expanded']:
        cid = item['candidate_id']; folder = STATE / cid; marker = folder / 'automated_diagnostic_completion.json'
        done = json.loads(marker.read_text())
        if done['analysis_sha256'] != sha(stage / 'analysis.json') or done['round'] != args.round:
            raise RuntimeError(f'{cid}: completion marker does not match this round analysis')
        for p, h in done['outputs'].items():
            if sha(p) != h:
                raise RuntimeError(f'{cid}: locked diagnostic output changed: {p}')
        hashes.update(done['outputs'])
        summary = json.loads((folder / 'summary.json').read_text())
        if summary['n_events'] != item['n_events'] or summary['candidate_id'] != cid:
            raise RuntimeError(f'{cid}: diagnostic event denominator differs from analysis')
        with (folder / 'all_event_comparisons.csv').open() as f:
            recs = list(csv.DictReader(f))
        if len(recs) != item['n_events']:
            raise RuntimeError(f'{cid}: CSV rows differ from event count')
        flt = lambda k: [float(r[k]) if r[k] not in ('', 'None') else None for r in recs]
        boundary = folder / 'native_boundary_activity.json'
        if not boundary.exists():
            subprocess.run([sys.executable, str(ROOT / 'scripts/audit_topic4_xy_v4_native_boundary.py'), '--round', str(args.round), '--candidate', cid], cwd=ROOT, check=True)
        edge = json.loads(boundary.read_text())
        anim = summary.get('animation') or {}
        rows.append({'candidate_id': cid, 'n_events': item['n_events'], 'n_networks': summary['n_networks'],
                     'joint_distance': item['joint_distance'], 'assessment_pass': item['assessment']['pass'],
                     'events_per_seed': {str(s): int(sum(1 for r in recs if int(r['seed']) == s)) for s in sorted({int(r['seed']) for r in recs})},
                     'contacts_per_event': quant(flt('n_contacts')),
                     'native_pair_lag_mae_ms': quant(flt('native_pair_lag_mae_ms')),
                     'native_order_discordance': quant(flt('native_order_discordance')),
                     'native_participation_mismatch': quant(flt('native_participation_mismatch')),
                     'TA_order_discordance': quant(flt('TA_order_discordance')), 'TA_pair_lag_mae_ms': quant(flt('TA_pair_lag_mae_ms')),
                     'TA_participation_mismatch': quant(flt('TA_participation_mismatch')),
                     'TB_order_discordance': quant(flt('TB_order_discordance')), 'TB_pair_lag_mae_ms': quant(flt('TB_pair_lag_mae_ms')),
                     'TB_participation_mismatch': quant(flt('TB_participation_mismatch')),
                     'first_event_preview': {'seed': int(recs[0]['seed']) if recs else None, 'model_window_ms': anim.get('model_window_ms'),
                                             'n_frames': anim.get('n_frames'), 'gif_sha256': anim.get('sha256'), 'gif': anim.get('path'),
                                             'TA_order_discordance': float(recs[0]['TA_order_discordance']) if recs and recs[0]['TA_order_discordance'] not in ('', 'None') else None,
                                             'TA_pair_lag_mae_ms': float(recs[0]['TA_pair_lag_mae_ms']) if recs and recs[0]['TA_pair_lag_mae_ms'] not in ('', 'None') else None,
                                             'TA_participation_mismatch': int(float(recs[0]['TA_participation_mismatch'])) if recs else None,
                                             'n_contacts': int(float(recs[0]['n_contacts'])) if recs else None},
                     'edge_to_interior_activity_ratio_per_seed': [s['edge_to_interior_per_neuron_activity_ratio'] for s in edge['seeds']],
                     'visual_qa_recorded': (folder / 'figures/visual_qa.json').exists()})
    result = {'status': 'ROUND_RAW_DIAGNOSTICS_NUMERICALLY_AUDITED_MODEL_NOT_QUALIFIED', 'updated_unix': time.time(), 'round': args.round,
              'candidates': rows, 'event_records_across_candidates': int(sum(r['n_events'] for r in rows)),
              'all_observed_events_included': True, 'event_selection': 'none; every observed event of every network',
              'shared_network_seed_sets_not_independent_replications': True, 'patient_exemplars_unpaired_descriptive_only': True,
              'visual_review_scope': 'recorded per candidate in figures/visual_qa.json when performed; absence means not visually reviewed',
              'scientific_qualification': False, 'fig5_hold_released': False, 'author_acceptance': False,
              'source_hashes': {**hashes, str(Path(__file__)): sha(Path(__file__))}}
    (stage / 'raw_propagation_numeric_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    for r in rows:
        print(f"{r['candidate_id']:22s} N={r['n_events']:3d} nat_lag_med={r['native_pair_lag_mae_ms']['median']:.3f} max={r['native_pair_lag_mae_ms']['max']:.2f} "
              f"TA_disc_med={r['TA_order_discordance']['median']:.3f} TB_disc_med={r['TB_order_discordance']['median']:.3f} "
              f"TA_mismatch_med={r['TA_participation_mismatch']['median']:.1f} contacts_med={r['contacts_per_event']['median']:.1f} "
              f"edge={min(r['edge_to_interior_activity_ratio_per_seed']):.3f}-{max(r['edge_to_interior_activity_ratio_per_seed']):.3f}")


if __name__ == '__main__':
    main()

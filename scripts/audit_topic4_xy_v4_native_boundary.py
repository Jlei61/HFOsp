#!/usr/bin/env python3
"""Descriptive edge/interior activity check for V4 candidates; same rule as the V3 audit.

Identical computation to scripts/audit_topic4_xy_native_boundary.py with the search root
parameterised, so the historical V3 producer is left untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--search-root', default=str(ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4'))
    args = ap.parse_args()
    root = Path(args.search_root)
    source = root/f'rounds/{args.round:03d}/combined_{args.candidate}.json'
    row = json.loads(source.read_text())['candidates'][0]; rows = []; hashes = {}
    for u in row['units']:
        wp = Path(u['worker_path'])
        assert hashlib.sha256(wp.read_bytes()).hexdigest() == u['worker_sha256']
        m = json.loads(wp.read_text()); p = Path(m['arrays']['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == m['arrays']['sha256']
        hashes[str(p)] = m['arrays']['sha256']
        with np.load(p) as z:
            counts = z['sheet_activity_counts'].astype(float); pos = z['positions_E']
            dt = float(z['sheet_activity_frame_ms']); L = float(z['source_sheet_mm']); mm = float(z['source_bin_mm'])
            assert mm == 1 and dt == 2 and L == 20
            gx, gy = np.meshgrid((np.arange(counts.shape[2])+.5)*mm, (np.arange(counts.shape[1])+.5)*mm)
            edge = (gx < 2) | (gx >= L-2) | (gy < 2) | (gy >= L-2)
            en = (pos[:, 0] < 2) | (pos[:, 0] >= L-2) | (pos[:, 1] < 2) | (pos[:, 1] >= L-2)
            start = int(500/dt); e = counts[start:, edge].sum(); i = counts[start:, ~edge].sum()
            rows.append({'seed': u['seed'], 'edge_neuron_fraction': float(en.mean()),
                         'edge_active_neuron_time_fraction': float(e/(e+i)),
                         'edge_to_interior_per_neuron_activity_ratio': float((e/en.sum())/(i/(~en).sum()))})
    hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, Path(__file__))})
    result = {'status': 'DESCRIPTIVE_NATIVE_BOUNDARY_AUDIT', 'boundary_band_mm': 2, 'burnin_ms': 500, 'seeds': rows,
              'quantity': 'Active neuron-time bins; not spikes or number of independent events',
              'boundary_rule_fixed_before_reading_results': True, 'source_hashes': hashes,
              'claim': 'Boundary enrichment is descriptive. Does not establish independent source, artifact, or necessary mechanism.'}
    out = root/'raw_propagation_audit'/args.candidate; out.mkdir(parents=True, exist_ok=True)
    (out/'native_boundary_activity.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Edge/interior per-neuron activity ratio:', [r['edge_to_interior_per_neuron_activity_ratio'] for r in rows])


if __name__ == '__main__':
    main()

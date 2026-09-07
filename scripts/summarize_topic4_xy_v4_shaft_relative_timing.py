#!/usr/bin/env python3
"""Descriptive SCL-vs-ICL relative timing: frozen patient training table vs every raw-audited V4 candidate.

Uses all observed events of each candidate (readout centroids from the automatic raw diagnostics)
and the same frozen patient training table the kernel objective uses. Not a threshold, not an
acceptance metric, no event selection; written for the milestone review only.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_component_search as run


def stats(T, icl, scl):
    T = np.asarray(T, float); both = np.isfinite(T[:, icl]).any(1) & np.isfinite(T[:, scl]).any(1)
    d = np.nanmedian(T[both][:, scl], 1) - np.nanmedian(T[both][:, icl], 1)
    first = np.nanmin(T[both][:, scl], 1) - np.nanmin(T[both][:, icl], 1)
    q = lambda v: {'median': float(np.median(v)), 'q25': float(np.quantile(v, .25)), 'q75': float(np.quantile(v, .75)),
                   'iqr': float(np.quantile(v, .75) - np.quantile(v, .25))}
    return {'n_events': int(len(T)), 'events_with_both_groups': int(both.sum()), 'frac_both': float(both.mean()),
            'scl_minus_icl_group_median_ms': q(d), 'first_scl_minus_first_icl_ms': q(first),
            'mean_scl_participation': float(np.isfinite(T[:, scl]).mean()), 'mean_icl_participation': float(np.isfinite(T[:, icl]).mean())}


def main():
    out = run.OUT; obj = run.KernelObjective(run.v1, out, run.KERNEL)
    names = obj.training['contact_names']
    icl = [i for i, n in enumerate(names) if n.startswith('ICL')]; scl = [i for i, n in enumerate(names) if n.startswith('SCL')]
    result = {'patient_training': stats(obj.patient, icl, scl)}; hashes = {}
    for c in sorted((out / 'raw_propagation_audit').glob('*/all_event_contacts.csv')):
        rows = list(csv.DictReader(c.open())); ev = {}
        for r in rows:
            ev.setdefault((r['seed'], r['event_index']), {})[r['contact']] = float(r['readout_centroid_ms']) if r['participant'] == 'True' else np.nan
        result[c.parent.name] = stats([[e[n] for n in names] for e in ev.values()], icl, scl)
        hashes[str(c)] = hashlib.sha256(c.read_bytes()).hexdigest()
    result['_note'] = ('Descriptive only. Readout centroids in ms; SCL/ICL groups by contact name; patient = frozen training table '
                       '(masked lagPatRaw centroid). Group median difference and first-contact difference per event. Not an acceptance metric.')
    result['_source_hashes'] = {**hashes, str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['_updated_unix'] = time.time()
    (out / 'scl_icl_relative_timing_descriptive.json').write_text(json.dumps(result, indent=2) + '\n')
    for k, v in result.items():
        if k.startswith('_'):
            continue
        m = v['scl_minus_icl_group_median_ms']
        print(f"{k:24s} N={v['n_events']:5d} both={v['frac_both']:.2f} SCL-ICL median {m['median']:7.1f} IQR {m['iqr']:6.1f}  SCL part {v['mean_scl_participation']:.2f} ICL part {v['mean_icl_participation']:.2f}")


if __name__ == '__main__':
    main()

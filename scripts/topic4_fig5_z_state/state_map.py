#!/usr/bin/env python3
"""Aggregate native continuation readouts into native_state_map.csv/json and apply the section 5.3 selection rule."""
import csv
import numpy as np
from common import *  # noqa: F401,F403
import readouts as R
import native_continue as nc


def core15_counts():
    g = np.load(OUT / 'approx/coarse_20/geometry.npz') if (OUT / 'approx/coarse_20/geometry.npz').exists() else None
    if g is not None:
        return np.bincount(g['g15'], minlength=3)
    s = np.load(SUBSTRATE / 'substrate.npz'); centers = s['centers_mm']; pos = s['positions_e']
    d = np.linalg.norm(pos[:, None] - centers[None], axis=2)
    g15 = np.full(len(pos), 2); g15[d[:, 0] < 1.5] = 0; g15[(d[:, 1] < 1.5) & (d[:, 1] < d[:, 0])] = 1
    return np.bincount(g15, minlength=3)


def analyse_all(names=None, force=False):
    R.set_core15_counts(core15_counts())
    rows = []; results = {}
    jobs = sorted((nc.NATIVE / 'jobs').glob('*.json'))
    for jp in jobs:
        job = read(jp); name = job['name']
        if names is not None and name not in names:
            continue
        folder = nc.NATIVE / 'runs' / name
        if not (folder / 'result.json').exists():
            continue
        out = folder / 'readout.json'
        if out.exists() and not force:
            res = read(out)
        else:
            res, arrays = R.analyse_run(folder, start_step=job['start_step'] if job['kind'] == 'continuation' else None)
            res['job'] = job; write(out, res)
            np.savez_compressed(folder / 'readout_arrays.npz', **{k: v for k, v in arrays.items() if v is not None})
        results[name] = res
        rows.append(R.summary_row(name, job, res))
    return rows, results


def pair_selection(rows):
    """Section 5.3: earliest adjacent Z pair (low history, both futures SELF_LIMITED -> PERSISTENT); else earliest
    pair with any inconsistency or UNRESOLVED; else none."""
    low = HISTORY_MS[0]
    cat = {(r['z_source_ms'], r['future']): r['category'] for r in rows if r['kind'] == 'continuation' and r['history_ms'] == low and not r['freeze_m']}
    pairs = list(zip(Z_TIMES_MS[:-1], Z_TIMES_MS[1:]))
    consistent = None; inconsistent = None; detail = []
    for a, b in pairs:
        c = {w: (cat.get((a, w)), cat.get((b, w))) for w in ('W1', 'W2')}
        complete = all(x is not None and y is not None for x, y in c.values())
        both_sl_to_p = complete and all(c[w] == ('SELF_LIMITED', 'PERSISTENT') for w in c)
        any_unres = complete and any('UNRESOLVED' in c[w] for w in c)
        futures_disagree = complete and (c['W1'][0] != c['W2'][0] or c['W1'][1] != c['W2'][1])
        detail.append(dict(pair=[a, b], W1=list(c['W1']), W2=list(c['W2']), complete=complete,
                           both_self_limited_to_persistent=both_sl_to_p, any_unresolved=any_unres, futures_disagree=futures_disagree))
        if both_sl_to_p and consistent is None:
            consistent = (a, b)
        if (any_unres or futures_disagree) and inconsistent is None:
            inconsistent = (a, b)
    if consistent is not None:
        return dict(rule='consistent_self_limited_to_persistent', pair=list(consistent), detail=detail)
    if inconsistent is not None:
        return dict(rule='earliest_inconsistent_or_unresolved', pair=list(inconsistent), detail=detail)
    return dict(rule='no_pair', pair=None, detail=detail)


def write_map(rows, selection=None):
    rows = sorted(rows, key=lambda r: (r['kind'] != 'continuation', r['history_ms'] or 0, r['z_source_ms'] or 0, r['future'] or '', r['name']))
    write(OUT / 'native_state_map.json', dict(rows=rows, selection=selection, definitions=dict(
        statistical_unit='one complete continuation trajectory', tail='last 4 s of each trajectory, four 1-s sub-windows',
        categories='SELF_LIMITED / QUIESCENT / PERSISTENT / UNRESOLVED per design 5.2; finite-time descriptors only')))
    keys = [k for k in rows[0].keys() if not isinstance(rows[0][k], (list, dict))] if rows else []
    with (OUT / 'native_state_map.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); a = ap.parse_args()
    rows, _ = analyse_all(force=a.force)
    sel = pair_selection(rows)
    write_map(rows, sel)
    for r in rows:
        print(f"{r['name']:24s} {r['category']:13s} high={r['high_rate']:22s} quiet={r['tail_quiet_fraction']:.3f} runs={r['tail_quiet_runs']:2d} ev={r['tail_n_events']:2d} "
              f"persist50={r['tail_persistent_50Hz_80']:.3f} E={r['tail_all_E_hz']:6.1f} A={r['tail_coreA_hz']:6.1f} B={r['tail_coreB_hz']:6.1f}")
    print('selection', json.dumps(sel['rule']), sel['pair'])

#!/usr/bin/env python3
"""Pairwise native-vs-approximation scoring and the pre-registered correspondence gates (design 6.3).

Gate thresholds are operational standards of this milestone, not biological significance levels:
  1 state:      >=1 adjacent Z pair (low history, both futures) native SELF_LIMITED -> PERSISTENT with the
                approximation matching category point-by-point; high history and extensions must match too.
  2 magnitude:  on that pair's tail windows: |quiet fraction diff| <= 0.10, |persistent fraction diff| <= 0.10,
                rate error / max(native rate, 20 Hz) <= 25% (all-E, core A, core B), self-limited-side event
                duration median diff <= max(20 ms, 25% native median) (NOT_ESTIMABLE if < 3 complete events).
  3 tail/resolution: results survive the extension; 20x20 and 10x10 give the same state direction.
  4 trajectory: real-Z(t) path replays reproduce last quiet end / high onset / spatial expansion separately.
"""
import argparse
import numpy as np
from common import *  # noqa: F401,F403
from approx_system import APPROX
import native_continue as nc
import readouts as R

KEY = f'{R.MAIN_SPATIAL[0]:g}Hz_duty{int(R.MAIN_SPATIAL[1] * 100)}'
GATE = dict(quiet_abs=.10, persistent_abs=.10, rate_rel=.25, rate_floor_hz=20., duration_abs_ms=20., duration_rel=.25, min_events=3)


def native_readout(name):
    p = nc.NATIVE / 'runs' / name / 'readout.json'
    return read(p) if p.exists() else None


def approx_readout(version, name):
    p = APPROX / version / 'runs' / name / 'result.json'
    return read(p) if p.exists() else None


def pair_metrics(nat, app):
    tn, ta = nat['tail'], app['tail']
    out = dict(native_category=nat['category'], approx_category=app['category'], category_match=nat['category'] == app['category'],
               native_high=nat['high_rate_reached'], approx_high=app['high_rate_reached'],
               quiet_fraction=dict(native=tn['quiet_fraction'], approx=ta['quiet_fraction'], abs_diff=abs(tn['quiet_fraction'] - ta['quiet_fraction'])),
               persistent_fraction=dict(native=tn['persistent_fraction'][KEY], approx=ta['persistent_fraction'][KEY],
                                        abs_diff=abs(tn['persistent_fraction'][KEY] - ta['persistent_fraction'][KEY])),
               rates={})
    for key in ('all_E_hz', 'readout175_E_coreA_hz', 'readout175_E_coreB_hz'):
        n_, a_ = tn['rates'][key], ta['rates'][key]
        out['rates'][key] = dict(native=n_, approx=a_, rel_error=abs(a_ - n_) / max(n_, GATE['rate_floor_hz']))
    dn, da = tn['event_durations_ms'], ta['event_durations_ms']
    if len(dn) >= GATE['min_events'] and len(da) >= GATE['min_events']:
        mn, ma = float(np.median(dn)), float(np.median(da))
        out['event_duration'] = dict(native_median_ms=mn, approx_median_ms=ma, abs_diff=abs(ma - mn), tolerance=max(GATE['duration_abs_ms'], GATE['duration_rel'] * mn),
                                     status='PASS' if abs(ma - mn) <= max(GATE['duration_abs_ms'], GATE['duration_rel'] * mn) else 'FAIL')
    else:
        out['event_duration'] = dict(native_n=len(dn), approx_n=len(da), status='NOT_ESTIMABLE')
    out['magnitude_pass'] = bool(out['quiet_fraction']['abs_diff'] <= GATE['quiet_abs'] and out['persistent_fraction']['abs_diff'] <= GATE['persistent_abs']
                                 and all(v['rel_error'] <= GATE['rate_rel'] for v in out['rates'].values()))
    out['subwindow_categories'] = dict(native_quiet_runs=[s['quiet_runs_ge20ms'] for s in nat['subwindows']], approx_quiet_runs=[s['quiet_runs_ge20ms'] for s in app['subwindows']],
                                       native_persistent=[s['persistent_fraction'][KEY] for s in nat['subwindows']], approx_persistent=[s['persistent_fraction'][KEY] for s in app['subwindows']])
    return out


def score(version, futures=('W1',), names=None):
    rows = []
    for j in nc.main_jobs():
        if j['future'] not in futures or (names is not None and j['name'] not in names):
            continue
        nat, app = native_readout(j['name']), approx_readout(version, j['name'])
        if nat is None or app is None:
            rows.append(dict(name=j['name'], status='MISSING', native=nat is not None, approx=app is not None)); continue
        m = pair_metrics(nat, app); m.update(name=j['name'], z_source_ms=j['z_source_ms'], history_ms=j['history_ms'], future=j['future'], status='SCORED')
        rows.append(m)
    return rows


def state_gate(rows, selection):
    """Gate 1 on the native-selected pair (section 5.3) for the given futures."""
    if selection is None or selection.get('pair') is None or selection['rule'] != 'consistent_self_limited_to_persistent':
        return dict(status='NOT_ESTIMABLE', reason='native did not bracket a consistent SELF_LIMITED->PERSISTENT adjacent pair')
    a, b = selection['pair']; checks = []; ok = True
    for r in rows:
        if r.get('status') != 'SCORED' or r['z_source_ms'] not in (a, b):
            continue
        checks.append(dict(name=r['name'], native=r['native_category'], approx=r['approx_category'], match=r['category_match']))
        if not r['category_match']:
            ok = False
    return dict(status='PASS' if ok and checks else ('FAIL' if checks else 'NOT_ESTIMABLE'), pair=[a, b], checks=checks)


def magnitude_gate(rows, selection):
    if selection is None or selection.get('pair') is None:
        return dict(status='NOT_ESTIMABLE')
    a, b = selection['pair']; rs = [r for r in rows if r.get('status') == 'SCORED' and r['z_source_ms'] in (a, b)]
    if not rs:
        return dict(status='NOT_ESTIMABLE')
    per = []
    for r in rs:
        per.append(dict(name=r['name'], magnitude_pass=r['magnitude_pass'], event_duration=r['event_duration']['status'],
                        quiet_abs=r['quiet_fraction']['abs_diff'], persistent_abs=r['persistent_fraction']['abs_diff'],
                        rate_rel=max(v['rel_error'] for v in r['rates'].values())))
    ok = all(p['magnitude_pass'] and p['event_duration'] != 'FAIL' for p in per)
    return dict(status='PASS' if ok else 'FAIL', per_pair=per, thresholds=GATE)


def score_extensions(version):
    """Pairwise tail comparison of the extended continuations (last 4 s of the 20-s runs)."""
    rows = []
    for jp in sorted((nc.NATIVE / 'jobs').glob('*_ext.json')):
        name = read(jp)['name']; nat = native_readout(name); app = approx_readout(version, name)
        if nat is None or app is None:
            rows.append(dict(name=name, status='MISSING', native=nat is not None, approx=app is not None)); continue
        m = pair_metrics(nat, app); m.update(name=name, status='SCORED', parent=name[:-4]); rows.append(m)
    write(APPROX / version / 'validation_extension.json', dict(version=version, rows=rows))
    return rows


def write_csv(version):
    import csv
    rows = []
    for fut in (('W1',), ('W2',)):
        p = APPROX / version / f"validation_{'_'.join(fut)}.json"
        if not p.exists():
            continue
        for r in read(p)['rows']:
            if r.get('status') != 'SCORED':
                continue
            rows.append(dict(name=r['name'], future=r['future'], z_source_ms=r['z_source_ms'], history_ms=r['history_ms'],
                             native_category=r['native_category'], approx_category=r['approx_category'], category_match=r['category_match'],
                             native_high=r['native_high'], approx_high=r['approx_high'],
                             quiet_native=r['quiet_fraction']['native'], quiet_approx=r['quiet_fraction']['approx'], quiet_abs_diff=r['quiet_fraction']['abs_diff'],
                             persistent_native=r['persistent_fraction']['native'], persistent_approx=r['persistent_fraction']['approx'], persistent_abs_diff=r['persistent_fraction']['abs_diff'],
                             allE_native=r['rates']['all_E_hz']['native'], allE_approx=r['rates']['all_E_hz']['approx'], allE_rel_err=r['rates']['all_E_hz']['rel_error'],
                             coreA_native=r['rates']['readout175_E_coreA_hz']['native'], coreA_approx=r['rates']['readout175_E_coreA_hz']['approx'], coreA_rel_err=r['rates']['readout175_E_coreA_hz']['rel_error'],
                             coreB_native=r['rates']['readout175_E_coreB_hz']['native'], coreB_approx=r['rates']['readout175_E_coreB_hz']['approx'], coreB_rel_err=r['rates']['readout175_E_coreB_hz']['rel_error'],
                             event_duration_status=r['event_duration']['status'], magnitude_pass=r['magnitude_pass']))
    if rows:
        with (OUT / 'approximation_validation.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write(OUT / 'approximation_validation.json', dict(version=version, rows=rows, thresholds=GATE,
          gates={fut: {k: read(APPROX / version / f'validation_{fut}.json')[k] for k in ('state_gate', 'magnitude_gate')} for fut in ('W1', 'W2') if (APPROX / version / f'validation_{fut}.json').exists()}))
    return len(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--version', default='v1'); ap.add_argument('--futures', nargs='*', default=['W1']); a = ap.parse_args()
    rows = score(a.version, tuple(a.futures))
    sel = read(OUT / 'native_state_map.json').get('selection') if (OUT / 'native_state_map.json').exists() else None
    out = dict(version=a.version, futures=a.futures, rows=rows, state_gate=state_gate(rows, sel), magnitude_gate=magnitude_gate(rows, sel), selection=sel)
    write(APPROX / a.version / f"validation_{'_'.join(a.futures)}.json", out)
    for r in rows:
        if r.get('status') == 'SCORED':
            print(f"{r['name']:22s} native={r['native_category']:13s} approx={r['approx_category']:13s} quietΔ={r['quiet_fraction']['abs_diff']:.3f} persΔ={r['persistent_fraction']['abs_diff']:.3f} "
                  f"E {r['rates']['all_E_hz']['native']:.1f}/{r['rates']['all_E_hz']['approx']:.1f} A {r['rates']['readout175_E_coreA_hz']['native']:.1f}/{r['rates']['readout175_E_coreA_hz']['approx']:.1f} dur={r['event_duration']['status']}")
        else:
            print(r)
    print('state_gate', out['state_gate']['status'], 'magnitude_gate', out['magnitude_gate']['status'])
    ext = score_extensions(a.version); print('extension rows scored', sum(1 for r in ext if r.get('status') == 'SCORED'))
    print('csv rows', write_csv(a.version))

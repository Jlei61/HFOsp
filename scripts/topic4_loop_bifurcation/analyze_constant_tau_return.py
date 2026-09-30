#!/usr/bin/env python3
"""Full observed trajectory with an invariant native prefix and one continuation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
from continue_constant_tau_return import OUT, NAME, SOURCE
import run_topic4_loop_zk_conditional as native
import run_topic4_recovery_window as window
from analyze_native import activity_censor
from zoom_topic4_return_core_propagation import event_metrics, summarize

DEST = OUT/'return_analysis'
BASE = native.SOURCE/'runs'/native.NAME
END = 56.8


def part(folder, sub, clock, keys, lo, hi, right=False):
    values = {k: [] for k in [clock, *keys]}
    for path in sorted((folder/sub).glob('*.npz')):
        if '.tmp.' in path.name:continue
        a, b = map(int, path.stem.split('_'))
        if b <= round(lo*10000) or a >= round(hi*10000):continue
        with np.load(path) as z:
            t = z[clock]/1000
            m = ((t > lo+1e-9) & (t <= hi+1e-9) if right else
                 (t >= lo-1e-9) & (t < hi-1e-9))
            for key in values:values[key].append(z[key][m])
    return {key: np.concatenate(value) for key, value in values.items()}


def combined(sub, clock, keys, right=False):
    pieces = [part(folder, sub, clock, keys, lo, hi, right) for folder, lo, hi in
              [(BASE, 0, 16.8), (SOURCE, 16.8, 26.8), (OUT/'runs'/NAME, 26.8, END)]]
    return {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}


def inputs(folder, lo, hi):
    rows = []
    for path in sorted((folder/'chunks').glob('*.npz')):
        a, b = map(int, path.stem.split('_'))
        if b <= round(lo*10000) or a >= round(hi*10000):continue
        with np.load(path) as z:
            v = z['inputs'];t = v[:, 0]/1000
            rows.append(v[(t >= lo-1e-9) & (t < hi-1e-9)])
    return np.concatenate(rows)


def normalize(value):
    if isinstance(value, dict):return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, (list, tuple)):return [normalize(v) for v in value]
    if isinstance(value, np.generic):return value.item()
    return value


def main(wait):
    DEST.mkdir(exist_ok=True)
    while True:
        status = read(OUT/'supervisor.json')
        if status['status'] == 'COMPLETE_ANALYSIS_PENDING':break
        if status['status'] == 'FAILED':
            write(DEST/'progress.json', dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(DEST/'progress.json', dict(status='WAITING_ONE_CONSTANT_TAU_CONTINUATION', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'runs'/NAME/'result.json')['status'] == 'COMPLETE'
    prefix = read(ROOT/'natural_exit_mediator_probes/constant_tau_prefix_diagnostic.json')
    assert prefix['constant_tau_prefix_equivalence_supported']
    c = combined('chunks', 'time_ms', ['spikes_1ms', 'regions_1ms'])
    f = combined('chunks', 'field_time_ms', ['field_5ms'])
    z = combined('chunks', 'slow_time_ms', ['Z'])
    m = combined('mechanism_chunks', 'time_ms', ['global_E_rate_Hz', 'global_raw_conductance_ratio'])
    k = combined('intrinsic_adaptation_chunks', 'time_ms', ['sahp_mean_conductance_ratio'])
    b = combined('z_budget_chunks', 'time_ms', ['values'], right=True)
    assert len(c['time_ms']) == 56800 and len(z['slow_time_ms']) == 2840
    assert np.allclose(c['time_ms'], (np.arange(56800)+.5), rtol=0, atol=1e-9)
    assert np.allclose(m['time_ms'], np.arange(56800), rtol=0, atol=1e-9)
    assert np.array_equal(k['time_ms'], m['time_ms'])
    assert np.allclose(z['slow_time_ms'], np.arange(2840)*20, rtol=0, atol=1e-9)
    assert np.allclose(b['time_ms'], (np.arange(2840)+1)*20, rtol=0, atol=1e-9)
    assert np.max(abs(b['values'][:, :, 5])) < 1e-11
    assert np.max(abs(b['values'][1:, :, 0]-b['values'][:-1, :, 1])) < 1e-11
    assert np.max(abs(z['Z'][:, [0, 5, 6, 7]]-b['values'][:, :, 0])) < 1e-11
    geo = dict(np.load(OUT/'geometry.npz'));counts = np.r_[32000, geo['region_counts'][:3]]
    raw = np.c_[c['spikes_1ms'][:, 0], c['regions_1ms'][:, :3]]
    assert np.array_equal(raw[:, 0], raw[:, 1:].sum(1))
    assert np.array_equal(f['field_5ms'].sum(1), raw[:, 0].reshape(-1, 5).sum(1))
    rr = raw.reshape(-1, 10, 4).sum(1)/counts/.01
    r5 = raw.reshape(-1, 5, 4).sum(1)/counts/.005
    with np.load(native.SOURCE/'references/native_s9108405.npz') as reference:
        for key in ['spikes_1ms', 'regions_1ms']:
            assert np.array_equal(c[key][:8000], reference[key][:8000])
    paired = np.concatenate([inputs(folder, lo, hi) for folder, lo, hi in
        [(BASE, 0, 16.8), (SOURCE, 16.8, 26.8), (OUT/'runs'/NAME, 26.8, END)]])
    baseline_inputs = inputs(BASE, 0, END)
    assert np.array_equal(paired, baseline_inputs) and len(paired) == 568
    source_analysis = read(native.SOURCE/'analysis'/f'{native.NAME}.json')
    reference_core = np.asarray(source_analysis['absolute_Z_recovery']['reference_core_Z'])
    reference_features = source_analysis['recovery_window']['reference_features']
    pp = window.audit.temporal_audit(rr)
    events = window.parent.strict_events(rr, END)
    # Preserve the source high/exit tracker; replace only its event list with
    # the same stricter core-inclusive accounting used in the source analysis.
    pp['events'] = events
    tz, cores = z['slow_time_ms']/1000, z['Z'][:, [5, 6]]
    episodes = []
    for ex in pp['low_activity_exits']:
        lo = ex['start_s']
        hi = next((e['onset_s'] for e in pp['entries'] if e['onset_s'] > ex['confirmation_s']), END)
        ids = np.flatnonzero((tz >= lo) & (tz < hi) & (cores >= reference_core).all(1))
        recovered = float(tz[ids[0]]) if len(ids) else None
        event_window = window.audit.interval_events(events, max(ex['confirmation_s'], recovered), hi) if recovered is not None else None
        features = window.event_features(event_window['brief_events'], rr, f['field_5ms']) if event_window else {}
        ratios = {key: features.get(key)/reference_features[key] if features.get(key) is not None and reference_features[key] else None
                  for key in ['duration_ms', 'interval_ms', 'peak_Hz']}
        matched = all(v is not None and .5 <= v <= 2 for v in ratios.values())
        passed = bool(event_window and window.audit.qualifies(event_window, minimum_n=10, minimum_span=5.) and matched)
        spatial = [event_metrics(dict(rates=r5), e) for e in (event_window or {}).get('brief_events', [])]
        episodes.append(dict(exit=ex, window_end_s=hi, both_core_reference_s=recovered,
            after_reference_events=event_window, features=features, reference_ratios=ratios,
            sustained_native_return_screen=passed, spatial_event_metrics=spatial,
            spatial_summary=summarize(spatial) if spatial else None,
            window_right_censored=hi == END))
    R, K = m['global_E_rate_Hz'], k['sahp_mean_conductance_ratio']
    low = R[1:]*np.exp(.001/.015) <= 5
    low &= m['time_ms'][1:] > 16800
    error = abs(K[1:]-K[:-1]*np.exp(-.001/.5))
    assert low.any() and error[low].max() < 1e-9
    first_entry = pp['entries'][0]['onset_s'] if pp['entries'] else END
    pre = window.audit.interval_events(events, .5, first_entry)
    result = dict(status='COMPLETE_CONSTANT_TAU_RETURN_ANALYSIS', observed_s=END,
        source_prefix_s=[0, 16.8], diagnostic_stage_s=[16.8, 26.8], continuation_s=[26.8, END],
        prefix_invariance=prefix, first8s_reference_arrays_bitwise=True, paired_external_record_rows_bitwise=568,
        actual_constant_low_rate_K_tau_verified_s=.5, maximum_K_decay_error=float(error[low].max()),
        primary=pp, strict_pre=pre, episodes=episodes, reference_core_Z=reference_core,
        reference_features=reference_features, full_rate_censoring=activity_censor(rr),
        sustained_native_return_screen=any(e['sustained_native_return_screen'] for e in episodes),
        counts_as_independent_seed=False, statistical_unit='One paired parameter trajectory of seed9108405; no new independent seed.',
        native_spatial_review='PENDING', formal_bifurcation_allowed=False, human_review='PENDING', producer_sha256=sha(__file__))
    write(DEST/'result.json', normalize(result))
    np.savez_compressed(DEST/'readouts.npz', time5_s=f['field_time_ms']/1000, rates5_Hz=r5,
        field5_Hz=f['field_5ms']/geo['cell_e_counts']/.005, slow_time_s=tz, Z=z['Z'],
        time1_s=m['time_ms']/1000, causal_R_Hz=R, Graw=m['global_raw_conductance_ratio'], K_mean=K,
        budget_time_s=b['time_ms']/1000, Z_budget=b['values'])
    shutil.copy2(__file__, DEST/'producer.py')
    write(DEST/'progress.json', dict(status=result['status'], updated_epoch=time.time()))
    print(dict(sustained_native_return_screen=result['sustained_native_return_screen'],
        episodes=[{k: e[k] for k in ['both_core_reference_s', 'reference_ratios', 'sustained_native_return_screen']} for e in episodes]), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)

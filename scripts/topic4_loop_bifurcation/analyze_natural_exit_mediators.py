#!/usr/bin/env python3
"""Paired native interventions at the actual falling-rate state; no loop counts."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
import run_topic4_loop_zk_conditional as native
from run_topic4_recovery_window import assert_same_state
from analyze_native import original, activity_censor
from analyze_feedback_tail import first_run
from probe_natural_exit_mediators import OUT, REFERENCE, CONTROLS

DEST = OUT/'mediator_analysis'
START, END = 16.8, 26.8


def stream(folder, sub, clock, keys, right=False):
    """Read only overlapping files, retaining each stream's true sample clock."""
    pieces = {k: [] for k in [clock, *keys]}
    for path in sorted((folder/sub).glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        lo, hi = map(int, path.stem.split('_'))
        if hi <= round(START*10000) or lo >= round(END*10000):
            continue
        with np.load(path) as z:
            t = z[clock]/1000
            keep = ((t > START+1e-9) & (t <= END+1e-9) if right else
                    (t >= START-1e-9) & (t < END-1e-9))
            for k in pieces:
                pieces[k].append(z[k][keep])
    return {k: np.concatenate(v) for k, v in pieces.items()}


def first(t, mask, n=1):
    i = first_run(t, mask, n)
    return None if i is None else float(t[i])


def load_condition(folder, label, reference_core):
    m = stream(folder, 'mechanism_chunks', 'time_ms',
               ['global_E_rate_Hz', 'global_raw_conductance_ratio'])
    a = stream(folder, 'intrinsic_adaptation_chunks', 'time_ms', ['sahp_mean_conductance_ratio'])
    g = stream(folder, 'global_response_chunks', 'time_ms', ['q', 'G_raw'])
    z = stream(folder, 'chunks', 'slow_time_ms', ['Z'])
    b = stream(folder, 'z_budget_chunks', 'time_ms', ['values'], right=True)
    c = stream(folder, 'chunks', 'time_ms', ['spikes_1ms', 'regions_1ms'])
    assert all(np.array_equal(m['time_ms'], d['time_ms']) for d in [a, g])
    # Spike counts are centered bins; mechanism records are pre-step states.
    assert np.array_equal(c['time_ms'], m['time_ms']+.5)
    assert len(m['time_ms']) == 10000 and len(b['time_ms']) == len(z['slow_time_ms']) == 500
    t = m['time_ms']/1000
    assert np.allclose(t, START+np.arange(10000)*.001, rtol=0, atol=1e-12)
    R, G, K = m['global_E_rate_Hz'], m['global_raw_conductance_ratio'], a['sahp_mean_conductance_ratio']
    q = g['q']
    assert np.array_equal(G, g['G_raw'])
    assert np.allclose(q, np.clip((R-200)/300, 0, 1), rtol=0, atol=1e-14)
    tz, tb, Z, budget = z['slow_time_ms']/1000, b['time_ms']/1000, z['Z'][:, [0, 5, 6, 7]], b['values']
    assert np.max(abs(budget[:, :, 5])) < 1e-11
    assert np.max(abs(budget[1:, :, 0]-budget[:-1, :, 1])) < 1e-11
    assert np.max(abs(Z-budget[:, :, 0])) < 1e-11
    gain, loss = budget[:, :, 2].sum(0)*.02, budget[:, :, 3].sum(0)*.02
    delta = budget[-1, :, 1]-budget[0, :, 0]
    assert np.max(abs(delta-gain+loss)) < 1e-10
    # Verify the actual K kinetics on whole 1ms intervals certified to have
    # R<=5 throughout. Nonnegative spike increments give this upper bound.
    low = R[1:]*np.exp(.001/.015) <= 5
    tau = .5 if label == CONTROLS[1] else 5.
    error = abs(K[1:]-K[:-1]*np.exp(-.001/tau))
    if low.any():
        assert error[low].max() < 1e-9, (label, error[low].max())
    geo = np.load(OUT/'geometry.npz')
    counts = np.r_[32000, geo['region_counts'][:3]]
    raw = np.c_[c['spikes_1ms'][:, 0], c['regions_1ms'][:, :3]]
    assert np.array_equal(raw[:, 0], raw[:, 1:].sum(1))
    rates = raw.reshape(-1, 10, 4).sum(1)/counts/.01
    events = original.event_audit.events(rates[:, :3].max(1), end=END-START)
    events = [e for e in events if rates[round(e['start_s']*100):round(e['end_s']*100), 0].max() >= 20]
    brief = [e for e in events if .02-1e-9 <= e['duration_s'] <= .2+1e-9]
    rlow = first(t, R <= 5, 100)
    recovered = first(tz, (Z[:, 1:3] >= reference_core).all(1))
    after_low = (t >= rlow+.1) if rlow is not None else np.zeros(len(t), bool)
    job = read(folder/'result.json')['job']
    limit = job['threshold']/(18-job['global_reversal_mV'])
    row = dict(name=label, absolute_window_s=[START, END],
        first_R_at_or_below5_for100ms_s=rlow,
        first_R_at_or_above200_for100ms_after_low_s=first(t, after_low & (R >= 200), 100),
        first_G_below_global_recovery_block_s=first(t, G < limit),
        first_both_core_net_positive_1s_budget_endpoint_s=first(tb, (budget[:, 1:3, 4] > 0).all(1), 50),
        first_both_core_Z_reference_s=recovered,
        initial_Z_allE_A_B_other=Z[0].tolist(), final_Z_allE_A_B_other=budget[-1, :, 1].tolist(),
        maximum_core_Z=Z[:, 1:3].max(0).tolist(), accumulated_Z_recovery=gain.tolist(),
        accumulated_Z_consumption=loss.tolist(), actual_Z_delta=delta.tolist(),
        native_budget_max_abs_error=float(abs(budget[:, :, 5]).max()),
        minimum_causal_R_Hz=float(R.min()), sampled_R_at_or_below5_duration_s=float((R <= 5).sum()*.001),
        core_reference=reference_core.tolist(), G_resource_block_threshold=limit,
        expected_low_rate_K_tau_s=tau, actual_low_rate_K_tau_verified=bool(low.any()),
        certified_low_rate_1ms_intervals=int(low.sum()),
        low_rate_K_exponential_max_abs_error=float(error[low].max()) if low.any() else None,
        complete_population_events=events, complete_brief_count=len(brief),
        complete_brief_after_Z_reference=[e for e in brief if recovered is not None and e['start_s']+START >= recovered],
        rate_censoring=activity_censor(rates), final1s_rates_allE_A_B_other_Hz=rates[-100:].mean(0).tolist(),
        diagnostic_intervention=label != 'original_unmodified', counts_as_autonomous_loop=False,
        claim_limit='R thresholds are diagnostic markers, not a new onset/exit classifier. Z reference recovery and complete brief-event return are separate. Event times/censoring are relative to16.8s; no spatial propagation acceptance from counts.')
    arrays = dict(time_s=t, causal_R_Hz=R, Graw=G, K_mean=K, gate=q,
        slow_time_s=tz, Z_allE_A_B_other=Z, budget_time_s=tb, Z_budget=budget,
        rate_time_s=START+(np.arange(len(rates))+.5)*.01, rates_allE_A_B_other_Hz=rates)
    inputs = stream_inputs(folder)
    return row, arrays, inputs


def stream_inputs(folder):
    rows = []
    for path in sorted((folder/'chunks').glob('*.npz')):
        lo, hi = map(int, path.stem.split('_'))
        if hi <= round(START*10000) or lo >= round(END*10000):
            continue
        with np.load(path) as z:
            v = z['inputs'];t = v[:, 0]/1000
            rows.append(v[(t >= START-1e-9) & (t < END-1e-9)])
    result = np.concatenate(rows)
    assert len(result) == 100
    return result


def main(wait):
    DEST.mkdir(exist_ok=True)
    while True:
        statusfile = OUT/'repair_supervisor.json' if (OUT/'repair_supervisor.json').exists() else OUT/'supervisor.json'
        s = read(statusfile)
        if s['status'] in ['COMPLETE_TWO_NATIVE_MEDIATOR_CONTROLS_ANALYSIS_PENDING',
                           'COMPLETE_REPAIRED_TWO_NATIVE_MEDIATOR_CONTROLS_ANALYSIS_PENDING']:
            break
        if s['status'] == 'FAILED':
            write(DEST/'progress.json', dict(status='STOPPED_ON_NATIVE_FAILURE', native=s));return
        write(DEST/'progress.json', dict(status='WAITING_NATIVE_MEDIATOR_CONTROLS', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'full_state_gate.json')['status'] == 'PASS'
    source_analysis = read(native.SOURCE/'analysis'/f'{native.NAME}.json')
    reference = np.asarray(source_analysis['absolute_Z_recovery']['reference_core_Z'])
    rows, records, inputs = [], [], []
    for label, folder in [('original_unmodified', REFERENCE)]+[(n, OUT/'runs'/n) for n in CONTROLS]:
        assert read(folder/'result.json')['status'] == 'COMPLETE'
        row, data, drive = load_condition(folder, label, reference)
        rows.append(row);records.append(data);inputs.append(drive)
        np.savez_compressed(DEST/f'{label}.npz', **data)
    assert all(np.array_equal(inputs[0], x) for x in inputs[1:])
    states = [native.read_pickle(OUT/'runs'/n/'checkpoint.pkl')['engine'] for n in CONTROLS]
    exog = ['rng_state', 'xi', 'external_drive']
    assert_same_state({k: states[0][k] for k in exog}, {k: states[1][k] for k in exog})
    # The decay intervention must leave the measured trajectory bitwise
    # unchanged while the entire prefix is certified R>5 between samples.
    R = records[0]['causal_R_Hz'];lower = R[:-1]*np.exp(-.001/.015)
    prefix = int(np.flatnonzero(lower <= 5)[0])+1
    assert np.array_equal(records[0]['causal_R_Hz'][:prefix], records[2]['causal_R_Hz'][:prefix])
    assert np.array_equal(records[0]['K_mean'][:prefix], records[2]['K_mean'][:prefix])
    result = dict(status='COMPLETE_TWO_NATIVE_MEDIATOR_CONTROLS', rows=rows,
        original_source=str(REFERENCE), statistical_unit='One verified16.8s internal state and one common future noise path; two diagnostic interventions plus reused original. No added autonomous seed.',
        paired_recorded_future_inputs_bitwise=100, paired_final_full_exogenous_state_bitwise=True,
        K_intervention_certified_active_prefix_R_K_bitwise=True,
        K_intervention_certified_active_prefix_end_s=float(records[0]['time_s'][prefix-1]),
        source_state_gate=read(OUT/'full_state_gate.json'), initial_control_gate=read(OUT/'control_initial_state_qa.json'),
        counts_as_autonomous_loop=False, formal_bifurcation_allowed=False, producer_sha256=sha(__file__))
    write(DEST/'result.json', result);shutil.copy2(__file__, DEST/'producer.py')
    write(DEST/'progress.json', dict(status=result['status'], updated_epoch=time.time()))
    print([{k: row[k] for k in ['name', 'first_R_at_or_below5_for100ms_s', 'first_both_core_Z_reference_s', 'final_Z_allE_A_B_other']} for row in rows], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)

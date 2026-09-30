#!/usr/bin/env python3
"""Two exact short replays to audit current fields before electrode weighting."""
import json
import time
import numpy as np
from run_topic4_historical_manual_z import setup, RefilledZ, MZSlowVarsConfig, make_external_drive, simulate_kick
from checkpoint import load
from lfp import LFPRecorder
from analyze_topic4_fig5_early_spatial import ROOT, BASE, OUT, write, correlation


def main():
    started = time.time()
    jobs = [(8000., 250.), (10180., 550.)]
    write('native_current_replay_status.json', dict(status='BUILDING', jobs_ms=jobs,
        scope='Exact existing trajectory replay, 800 ms total; same parameters, fast state and future noise. Only new observers.'))
    original = np.load(BASE / 'runs/continuous_refill_release.npz')
    s, tr, frozen, identity = setup(9108401)
    expected_identity = json.loads((BASE/'runs/continuous_refill_release.json').read_text())['frozen_identity']
    assert identity == expected_identity
    cells = original['cell_e']; ncell = original['cell_e_counts']; ne = s.n_e; ni = s.n_i
    recorder = LFPRecorder(s.params, s.net['pos'], s.net['labels'], sites=s.contact_xy)
    output, checks = {}, []
    for start, duration in jobs:
        state = load(BASE.parent / 'historical_manual_hard_native_z_v1/checkpoints' / f't{start:g}ms.npz')
        s.params.T = duration
        slow = RefilledZ(ne+ni, s.params.V_th,
            MZSlowVarsConfig(use_z=True, use_m=False, tau_z=5000., I_th_EI=95.19851312666987), NE=ne)
        drive = make_external_drive(s, tr['spatial_ou'], 9108401)
        times, current_field, readout = [], [], []
        steps = round(duration / .1)
        rates = np.empty((steps, 2)); spikes = np.empty((steps, len(original['sample_ids'])), bool)
        def current_observer(tm, ie, ii, voltage):
            if round(tm / .1) % 5:
                return
            applied = ii.copy(); applied[:ne] *= slow.z[:ne]
            g = abs(ie[:ne]) + abs(applied[:ne])
            times.append(tm / 1000)
            current_field.append(np.bincount(cells, weights=g, minlength=400) / ncell)
            readout.append(recorder.sample(ie, applied))
        def spike_observer(tm, spk):
            k = round((tm-start)/.1)
            rates[k] = [spk[:ne].sum()/ne*10000, spk[ne:].sum()/ni*10000]
            spikes[k] = spk[original['sample_ids']]
            if k % 1000 == 0:
                write('native_current_replay_status.json', dict(status='RUNNING',
                    segment_start_ms=start, current_time_ms=tm, elapsed_s=time.time()-started))
        simulate_kick(s.params, s.net, KICK_BOOST=0., V_th_per_neuron=s.vtheta, slow=slow,
            external_e_rate_drive=drive, resume_state=state, time_offset_ms=start,
            early_stop_runaway=False, current_observer=current_observer, spike_observer=spike_observer,
            record_dense_spikes=False, fast_scatter=True, verbose=False)
        lo=round(start/.1);hi=lo+steps
        tt=np.array(times); rr=np.array(readout)
        refidx=np.rint(tt*2000).astype(int)
        qa=dict(segment_start_ms=start, duration_ms=duration,
            sampled_spikes_bitwise_equal=bool(np.array_equal(spikes,original['sample_spikes'][lo:hi])),
            E_counts_equal=bool(np.array_equal(np.rint(rates[:,0]*ne/10000), np.rint(original['rate_e_hz'][lo:hi]*ne/10000))),
            I_counts_equal=bool(np.array_equal(np.rint(rates[:,1]*ni/10000), np.rint(original['rate_i_hz'][lo:hi]*ni/10000))),
            virtual_SEEG_bitwise_equal=bool(np.array_equal(rr, original['lfp_effective'][refidx])))
        assert all(v for k,v in qa.items() if k.endswith('equal')), qa
        checks.append(qa)
        output[f't{start:g}_time_s']=tt;output[f't{start:g}_cell_mean_current']=np.array(current_field)
        output[f't{start:g}_readout']=rr
    baseline=output['t8000_cell_mean_current']; earlytime=output['t10180_time_s']
    early=output['t10180_cell_mean_current'][(earlytime>=10.48)&(earlytime<10.73)]
    delta=(early**2).mean(0)-(baseline**2).mean(0)
    output['native_current_delta_power']=delta
    output['native_current_early_power']=(early**2).mean(0)
    np.savez_compressed(OUT/'native_current_replay.npz', **output)
    analysis=np.load(OUT/'analysis_arrays.npz')
    values={family:correlation(-analysis['native_template_rank'][k],delta) for k,family in enumerate(['A','B'])}
    write('native_current_replay_summary.json', dict(status='COMPLETE', checks=checks,
        baseline_window_s=[8.,8.25],early_window_s=[10.48,10.73],
        observable='Temporal mean square of each native 1-mm cell mean applied AMPA+Z*GABA current, early minus baseline; no electrode spatial weighting.',
        baseline_scope='Additional exact 8.0-8.25s reference; not the full 0.5-8s baseline used for the original contact-power panel.',
        native_template_power_rho=values, min_delta=float(delta.min()),max_delta=float(delta.max()),
        elapsed_s=time.time()-started, frozen_identity=identity))
    write('native_current_replay_status.json',dict(status='COMPLETE', elapsed_s=time.time()-started,
        all_replay_checks_pass=True, native_template_power_rho=values))
    print(json.dumps(dict(status='COMPLETE',seconds=time.time()-started, native_template_power_rho=values)))


if __name__ == '__main__':
    main()

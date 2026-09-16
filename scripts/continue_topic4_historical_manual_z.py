#!/usr/bin/env python3
"""Bounded native-SNN continuations from verified autonomous-Z checkpoints."""
from run_topic4_historical_manual_z import ROOT, OUT, setup, read, write, make_external_drive, RefilledZ, MZSlowVarsConfig, simulate_kick
from topic4_historical_manual_z_common import TIMES_MS, ARM
REFERENCE = OUT
from checkpoint import load
import numpy as np
import time
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def run(job):
    started = time.time(); name = job['name']; tm = job['checkpoint_ms']
    s, tr, frozen, identity = setup(9108401)
    state = load(OUT / 'checkpoints' / f't{tm}ms.npz')
    if job.get('voltage_shift_E_mv'):
        state['V'][:s.net['NE']] += job['voltage_shift_E_mv']
    if job.get('future_seed'):
        seed = job['future_seed']
        state['rng_state'] = np.random.default_rng(seed).bit_generator.state
        state['external_drive']['rng_state'] = np.random.default_rng(seed+500000).bit_generator.state
    cfg = read(REFERENCE / 'protocol.json')
    slow = RefilledZ(s.net['NE'] + s.net['NI'], s.params.V_th,
                    MZSlowVarsConfig(use_z=True, use_m=False, tau_z=cfg['tau_z_ms'],
                                     I_th_EI=cfg['I_th_EI'], trace_stride_steps=100), NE=s.net['NE'])
    if job['mode'] == 'frozen':
        def keep_z(spk, labels, dt): slow._step_index += 1
        slow.step = keep_z
    p = s.params; p.T = job['duration_ms']; ne, ni = s.net['NE'], s.net['NI']
    drive = make_external_drive(s, tr['spatial_ou'], 9108401)
    if (REFERENCE / 'trajectory.npz').exists():
        reference = np.load(REFERENCE / 'trajectory.npz')
    else:
        from validate_topic4_fixed_rate_base import spatial_cell_index
        ref = np.load(OUT / 'reference_samples/runs/z_current_e_seed9108401.npz')
        cells = spatial_cell_index(s.positions_e, n_grid=20, sheet_l_mm=s.params.L)
        reference = {'sample_ids':ref['sample_ids'], 'cell_e':cells, 'cell_e_counts':np.bincount(cells,minlength=400)}
    samples = reference['sample_ids']; cells = reference['cell_e']
    steps = round(p.T / p.dt); frames = round(p.T)
    raster = np.empty((steps, len(samples)), bool); rates = np.empty((steps, 2))
    fields = np.zeros((frames, 400), np.uint16); ztrace = np.zeros((frames, 4))
    init_z = state['slow']['z'][:ne].copy()
    def observe(absolute_tm, spk):
        k = round((absolute_tm-tm) / p.dt); frame = k // 10
        raster[k] = spk[samples]; rates[k] = [spk[:ne].sum()/ne/p.dt*1000, spk[ne:].sum()/ni/p.dt*1000]
        fields[frame] += np.bincount(cells[spk[:ne]], minlength=400).astype(np.uint16)
        if k % 10 == 0:
            z = slow.z[:ne]; ztrace[frame] = [z.mean(), z.std(), z.min(), z.max()]
        if k % 1000 == 0:
            write(OUT / 'native_progress' / f'{name}.json', {'status': 'RUNNING', 'time_ms': k*p.dt,
                  'elapsed_s': time.time()-started})
    simulate_kick(p, s.net, KICK_BOOST=0., V_th_per_neuron=s.vtheta, slow=slow,
                  external_e_rate_drive=drive, resume_state=state, time_offset_ms=tm,
                  early_stop_runaway=False, spike_observer=observe, record_dense_spikes=False,
                  fast_scatter=True, verbose=False)
    if job['mode'] == 'frozen': assert np.array_equal(slow.z[:ne], init_z)
    if job.get('exact_resume_qa'):
        lo = round(tm/p.dt); hi = lo+steps
        assert np.array_equal(raster, reference['sample_spikes'][lo:hi])
        assert np.array_equal(rates[:, 0], reference['rate_e_hz'][lo:hi])
        assert np.array_equal(rates[:, 1], reference['rate_i_hz'][lo:hi])
    counts = np.rint(rates[:,0]*ne*p.dt/1000).astype(np.int64).reshape(-1,10).sum(1)
    assert np.array_equal(counts, fields.sum(1))
    folder = OUT / 'native'; folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f'{name}.npz', sample_spikes=raster, sample_ids=samples,
                        rate_e_hz=rates[:,0], rate_i_hz=rates[:,1], field_e_count_1ms=fields,
                        z_summary_1ms=ztrace, initial_z_e=init_z, final_z_e=slow.z[:ne],
                        cell_e=cells, cell_e_counts=reference['cell_e_counts'], dt_ms=p.dt)
    e5 = rates[:,0].reshape(-1,50).mean(1); tail=e5[len(e5)//2:]
    row = dict(job, status='COMPLETE', seconds=time.time()-started,
               late_E_mean_hz=float(tail.mean()), late_E_peak_5ms_hz=float(tail.max()),
               late_E_quiet_fraction=float(np.mean(tail<1)), late_E_cv=float(tail.std()/max(tail.mean(),1e-12)),
               initial_Z_mean=float(init_z.mean()), initial_Z_std=float(init_z.std()),
               final_Z_mean=float(slow.z[:ne].mean()), count_conservation=True,
               exact_resume_verified=bool(job.get('exact_resume_qa')), frozen_identity=identity)
    write(folder / f'{name}.json',row)
    write(OUT / 'native_progress' / f'{name}.json',row)
    return row


def main():
    jobs = [{'name':f'frozen_t{t}', 'checkpoint_ms':t, 'mode':'frozen', 'duration_ms':2000.} for t in TIMES_MS]
    write(OUT/'native_continuation_protocol.json', {'jobs':jobs, 'max_workers':3,
        'meaning':'Each actual neuron-wise Z field is frozen, with its native fast state and OU history carried forward.',
        'times':'Same absolute checkpoints as the supplied figure; phases are identified from this new trajectory.',
        'scope':'Finite two-second continuations, not asymptotic stability or bifurcation proof.'})
    rows=[];pending=list(jobs);active={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        while pending or active:
            status=read(OUT/'status.json')
            if status['status']=='FAILED':raise RuntimeError(status)
            ready={r['time_ms'] for r in read(OUT/'checkpoint_index.json')} if (OUT/'checkpoint_index.json').exists() else set()
            for job in list(pending):
                if len(active)>=3:break
                if job['checkpoint_ms'] in ready:
                    result_path=OUT/'native'/f"{job['name']}.json"
                    if result_path.exists() and read(result_path).get('status')=='COMPLETE':
                        rows.append(read(result_path));pending.remove(job);continue
                    active[pool.submit(run,job)]=job;pending.remove(job)
            for future in list(active):
                if future.done():
                    rows.append(future.result());del active[future]
            write(OUT/'native_batch_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows,
                  'active':[j['name'] for j in active.values()],'waiting_for_checkpoints':[j['name'] for j in pending]})
            if pending or active:time.sleep(5)
    write(OUT/'native_batch_status.json',{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'rows':rows})
    while not read(OUT/'status.json')['status'].startswith('COMPLETE'):
        if read(OUT/'status.json')['status']=='FAILED':raise RuntimeError(read(OUT/'status.json'))
        time.sleep(5)
    meta=read(OUT/'run.json');restore=meta['restore_start_ms']
    if restore is None or restore>8200:
        qa=run({'name':'exact_resume_t8000_200ms','checkpoint_ms':8000,'mode':'autonomous','duration_ms':200.,'exact_resume_qa':True})
        write(OUT/'native_resume_qa.json',qa)
    else:
        write(OUT/'native_resume_qa.json',{'status':'NOT_APPLICABLE','reason':'First saved checkpoint overlaps external refill.'})
    from plot_topic4_historical_manual_z import main as render
    render()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'native_batch_status.json',{'status':'FAILED','error':repr(exc)})
        raise

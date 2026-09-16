#!/usr/bin/env python3
"""Paired frozen-Z/history interventions with one common future input stream."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write, checkpoint_path, observables
from run_topic4_autonomous_z_manual_restore import setup, make_external_drive, RefilledZ, MZSlowVarsConfig, simulate_kick, spatial_cell_index
from checkpoint import load, save
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import time
import copy
import os


class Finished(Exception):
    pass


def paired_state(physical, anchor, z_payload):
    """Keep physical state and remaining-delay order; explicitly replace future input."""
    state = copy.deepcopy(physical)
    old_step = int(physical['step']); new_step = int(anchor['step'])
    for key in ('ring_sE', 'ring_sI'):
        a = physical[key]; n = len(a)
        state[key] = np.roll(a, (new_step-old_step) % n, axis=0)
        for delay in range(n):
            assert np.array_equal(a[(old_step+delay) % n], state[key][(new_step+delay) % n])
    for key in ('step', 'absolute_time_ms', 'xi', 'rng_state', 'external_drive', 'ras_keep'):
        state[key] = copy.deepcopy(anchor[key])
    state['slow']['step_index'] = int(anchor['slow']['step_index'])
    state['slow']['z'] = z_payload['slow']['z'].copy()
    for key in ('V', 'ref', 's_E', 'I_E', 's_I', 'I_I'):
        assert np.array_equal(state[key], physical[key])
    if old_step == new_step:
        assert np.array_equal(state['ring_sE'], physical['ring_sE'])
        assert np.array_equal(state['ring_sI'], physical['ring_sI'])
    return state


def run(job):
    started = time.time(); name = job['name']; seed = 9108401
    write(OUT/'native_progress'/f'{name}.json', {'status':'BUILDING'})
    s, tr, frozen, identity = setup(seed); cfg = read(REFERENCE/'protocol.json')
    assert cfg['frozen_identity'] == identity
    ne, ni = s.net['NE'], s.net['NI']; p = s.params
    if job.get('resume_endpoint'):
        state = load(job['resume_endpoint'])
    else:
        state = paired_state(load(checkpoint_path(job['history_ms'])), load(checkpoint_path(9400)),
                             load(checkpoint_path(job['z_profile_ms'])))
    tm = state['absolute_time_ms']; initial_z = state['slow']['z'][:ne].copy()
    if job.get('future_seed'):
        future_seed = job['future_seed']
        state['rng_state'] = np.random.default_rng(future_seed).bit_generator.state
        state['external_drive']['rng_state'] = np.random.default_rng(future_seed+500000).bit_generator.state
    slow = RefilledZ(ne+ni,p.V_th,MZSlowVarsConfig(use_z=True,use_m=False,
        tau_z=cfg['tau_z_ms'],I_th_EI=cfg['I_th_EI'],trace_stride_steps=100),NE=ne)
    def keep_z(spk, labels, dt): slow._step_index += 1
    slow.step = keep_z
    duration = job['duration_ms']; p.T = duration+p.dt
    drive = make_external_drive(s,tr['spatial_ou'],seed)
    reference = np.load(REFERENCE/'trajectory.npz'); samples = reference['sample_ids']; cells = reference['cell_e']
    icells = spatial_cell_index(s.positions_i,n_grid=20,sheet_l_mm=p.L)
    frames = round(duration); steps = round(duration/p.dt)
    raster = np.empty((steps,len(samples)),bool); rates = np.empty((steps,2))
    fields = np.zeros((frames,400),np.uint16); ifields = np.zeros_like(fields)
    seen = 0; endpoint_path = OUT/'native_endpoints'/f'{name}.npz'

    def observe(absolute_tm, spk):
        nonlocal seen
        k = round((absolute_tm-tm)/p.dt); frame = k//10
        raster[k] = spk[samples]
        rates[k] = [spk[:ne].sum()/ne/p.dt*1000,spk[ne:].sum()/ni/p.dt*1000]
        fields[frame] += np.bincount(cells[spk[:ne]],minlength=400).astype(np.uint16)
        ifields[frame] += np.bincount(icells[spk[ne:]],minlength=400).astype(np.uint16)
        seen = k+1
        if seen % 1000 == 0:
            write(OUT/'native_progress'/f'{name}.json', {'status':'RUNNING','time_ms':seen*p.dt,'elapsed_s':time.time()-started})

    def capture(step, payload):
        assert seen == steps
        save(payload,endpoint_path)
        raise Finished()

    try:
        simulate_kick(p,s.net,KICK_BOOST=0.,V_th_per_neuron=s.vtheta,slow=slow,
            external_e_rate_drive=drive,resume_state=state,time_offset_ms=tm,
            early_stop_runaway=False,spike_observer=observe,record_dense_spikes=False,fast_scatter=True,
            checkpoint_steps=[state['step']+steps],checkpoint_sink=capture,verbose=False)
    except Finished:
        pass
    assert seen == steps and np.array_equal(slow.z[:ne],initial_z)
    for k, n, field in [(0,ne,fields),(1,ni,ifields)]:
        count = np.rint(rates[:,k]*n*p.dt/1000).astype(np.int64).reshape(-1,10).sum(1)
        assert np.array_equal(count,field.sum(1))
    reference_qa = None
    if job.get('reference_qa'):
        old = np.load(OLD/'native/frozen_t9400.npz')
        assert np.array_equal(raster,old['sample_spikes'])
        assert np.array_equal(fields,old['field_e_count_1ms'])
        reference_qa = 'Bitwise raster and spatial count identity to preceding frozen-t9400 experiment.'
    folder = OUT/'native'; folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{name}.npz',sample_spikes=raster,sample_ids=samples,
        rate_e_hz=rates[:,0],rate_i_hz=rates[:,1],field_e_count_1ms=fields,field_i_count_1ms=ifields,
        initial_z_e=initial_z,final_z_e=slow.z[:ne],cell_e=cells,cell_i=icells,
        cell_e_counts=reference['cell_e_counts'],cell_i_counts=np.bincount(icells,minlength=400),dt_ms=p.dt,start_ms=tm)
    windows = [observables(fields[k:k+1000],reference['cell_e_counts']) for k in range(0,frames,1000)]
    row = dict(job,status='COMPLETE',seconds=time.time()-started,start_ms=tm,
        initial_mean_Z=float(initial_z.mean()),initial_std_Z=float(initial_z.std()),
        windows=windows,frozen_Z_bitwise=True,count_conservation=True,delay_rebasing_verified=True,
        reference_qa=reference_qa,endpoint_path=str(endpoint_path),frozen_identity=identity)
    write(folder/f'{name}.json',row); write(OUT/'native_progress'/f'{name}.json',row)
    return row


def main():
    while True:
        status = read(OUT/'replay_status.json')
        if status['status'] == 'COMPLETE': break
        if status['status'] == 'FAILED': raise RuntimeError(status)
        os.kill(read(OUT/'replay_process.json')['pid'],0); time.sleep(10)
    jobs = [{'name':f'z{z}_history{h}','z_profile_ms':z,'history_ms':h,'duration_ms':2000,
             'reference_qa':z==9400 and h==9400} for z in (9400,8000,8400,8800,9200) for h in (9400,8000)]
    write(OUT/'native_jobs.json',jobs)
    rows=[]
    # Queue at most one job per worker; errors stop dispatch of further jobs.
    with ProcessPoolExecutor(max_workers=2) as pool:
        iterator=iter(jobs); pending={pool.submit(run,next(iterator)) for _ in range(2)}
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done: rows.append(future.result())
            write(OUT/'native_batch_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
            for _ in done:
                job=next(iterator,None)
                if job is not None: pending.add(pool.submit(run,job))
    write(OUT/'native_batch_status.json',{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'rows':rows})


if __name__=='__main__':
    try: main()
    except Exception as exc:
        write(OUT/'native_batch_status.json',{'status':'FAILED','error':repr(exc)}); raise

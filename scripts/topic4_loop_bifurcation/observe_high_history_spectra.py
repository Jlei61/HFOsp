#!/usr/bin/env python3
"""One unchanged native high-history 72--74s source record for a second anchor."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,time
import numpy as np
from scipy import sparse
from campaign import ROOT,NATIVE,read,write,sha
import observe_source_aggregation as base
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'native_K9p35_high_history_source_spectra'
PARENT=ROOT/'native_K9p35_held_history/runs/exit_z0.21_k9.35_fields16p7_held_K9_history'
INITIAL=PARENT/'checkpoint.pkl'
NAME='unchanged_high_history_K9p35_72to74_source_observation'
LOW=ROOT/'native_K9p35_source_statistics'


def configure():
    base.OUT=OUT;base.PARENT=PARENT;base.INITIAL=INITIAL;base.NAME=NAME
    return base.configure()


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(PARENT/'result.json')['status']=='COMPLETE'
    initial=base.native.read_pickle(INITIAL)['engine'];assert initial['step']==720000
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_SECOND_HISTORY_SOURCE_OBSERVATION',created_epoch=time.time(),
        question='Does the spectral closure have relevant source-time information at the second native high-history state, where white density placed the global feedback on the wrong activation segment?',
        design='One unchanged native continuation72-74s from the completed30s high-history K9.35 intervention. Preserve entire internal/external engine and held fullZ/K. Readonly all-cell spike times0.1ms,39 selectedtarget current/V/M traces,4x0.5s individual input moments and expected external rates. No new parameter, seed, equation, or native feedback rule.',
        reference='The previous asymmetric42-44s observer began from a matched future external state after its own30s conditional protocol. Compare future external rate records here rather than assuming matching from filenames.',
        scope='Additional development anchor; a new2s observation on an existing conditionaltrajectory, not an independent seed, autonomous loop or attractor certification. No existing74s reference, so final-engine replay equality is not claimed.',
        checks='Initial wholeengine bitwise; complete20k observations; observer spikes equal native1ms totals; fixedZ/K retained; source code remains unchanged. Follow-on spectral assays require their own concrete bounded design.',
        source=str(INITIAL),source_sha256=sha(INITIAL),producer_sha256=sha(__file__),
        formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    p=copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='NATIVE_HIGH_HISTORY_SOURCE_SPECTRA',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',p);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz')
    configure();base.native.make_job(NAME,str(INITIAL),2.,True,.21,9.35,False)
    assert_same_state(initial,base.native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')['engine'])
    write(OUT/'initial_gate.json',dict(status='PASS',whole_engine_bitwise=True))
    shutil.copy2(__file__,OUT/'producer.py')


def worker(device,wait):
    if wait:
        while not (ROOT/'individual_source_spectral_damped_pilot/result.json').exists():
            state=read(ROOT/'individual_source_spectral_damped_pilot/supervisor.json')['status']
            if state=='FAILED':raise RuntimeError('Prior sampler failed; inspect before native dispatch')
            write(OUT/'observer_progress.json',dict(status='WAITING_BOUNDED_GPU_PILOT',pid=os.getpid(),updated_epoch=time.time()))
            time.sleep(10)
    configure();c=read(OUT/'contract.json');assert c['source_sha256']==sha(INITIAL) and c['producer_sha256']==sha(__file__)
    cells=np.load(LOW/'local_response_factorial/inputs.npz')['cells']
    backend=base.gpu.cuda_backend.wrap_simulator
    def wrap(original,device_index):
        fast=backend(original,device_index=device_index)
        def observed(params,net,*args,**kw):
            assert kw['resume_state']['step']==720000;slow=kw['slow']
            tr=np.empty((20000,4,len(cells)));global_trace=np.empty((20000,2));times=np.empty(20000)
            moments=np.zeros((4,7,40000));nu=np.zeros((4,40000));events=[];n=ni=0
            oldcur=kw.get('current_observer');oldsp=kw.get('spike_observer');oldin=kw.get('input_observer')
            def currents(tm,ie,ii,v):
                nonlocal n
                if oldcur is not None:oldcur(tm,ie,ii,v)
                tr[n]=np.stack([ie[cells],ii[cells],v[cells],slow.m[cells]])
                global_trace[n]=[slow.r_global,slow.global_state];times[n]=tm
                for j,a in enumerate([ie,ii,ie*ie,ii*ii,ie*ii,v,slow.m]):moments[n//5000,j]+=a
                n+=1
            def inputs(tm,rate,xi):
                nonlocal ni
                if oldin is not None:oldin(tm,rate,xi)
                nu[ni//5000]+=np.broadcast_to(rate,(40000,));ni+=1
            def spikes(tm,sp):
                if oldsp is not None:oldsp(tm,sp)
                events.append(np.flatnonzero(sp).astype('i4'))
            kw.update(current_observer=currents,input_observer=inputs,spike_observer=spikes)
            try:return fast(params,net,*args,**kw)
            finally:
                if n==ni==len(events)==20000:
                    assert np.allclose(times,72000+np.arange(20000)*.1,rtol=0,atol=1e-9)
                    lengths=np.array([len(a) for a in events]);indices=np.concatenate(events);pointers=np.r_[0,np.cumsum(lengths)]
                    sp=sparse.csc_matrix((np.ones(len(indices),dtype='u1'),indices,pointers),shape=(40000,20000))
                    sparse.save_npz(OUT/'source_spikes_0p1ms.npz',sp)
                    counts=np.stack([np.asarray(sp[:,i*5000:(i+1)*5000].sum(1)).ravel() for i in range(4)])
                    np.savez_compressed(OUT/'cell_statistics.npz',per_cell_spike_counts=counts,
                        per_cell_mean_moments=moments/5000,per_cell_mean_external_per_ms=nu/5000,
                        global_R_and_s=global_trace,time_ms=times)
                    np.savez_compressed(OUT/'target_traces.npz',cells=cells,IE_II_V_M=tr,
                        causal_R_and_s=global_trace,times_s=times/1000)
                else:write(OUT/'incomplete.json',dict(currents=n,inputs=ni,spikes=len(events)))
        return observed
    base.gpu.cuda_backend.wrap_simulator=wrap
    write(OUT/'observer_progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    try:base.gpu.worker(OUT,NAME,device)
    finally:base.gpu.cuda_backend.wrap_simulator=backend
    folder=OUT/'runs'/NAME;assert read(folder/'result.json')['status']=='COMPLETE'
    st=dict(np.load(OUT/'cell_statistics.npz'));sp=sparse.load_npz(OUT/'source_spikes_0p1ms.npz')
    recorded=np.concatenate([np.load(p)['spikes_1ms'] for p in sorted((folder/'chunks').glob('*.npz'))])
    totals=np.c_[np.asarray(sp[:32000].sum(0)).ravel(),np.asarray(sp[32000:].sum(0)).ravel()]
    assert np.array_equal(totals.reshape(2000,10,2).sum(1),recorded[:,:2])
    end=base.native.read_pickle(folder/'checkpoint.pkl')['engine'];assert end['step']==740000
    with np.load(PARENT/'held_fields.npz') as f:
        assert np.array_equal(end['slow']['z'][:32000],f['Z']) and np.array_equal(end['termination_mechanism']['sahp_g'],f['K'])
    with np.load(LOW/'cell_statistics.npz') as old:
        matched=np.array_equal(old['per_cell_mean_external_per_ms'],st['per_cell_mean_external_per_ms'])
        delta=float(abs(old['per_cell_mean_external_per_ms']-st['per_cell_mean_external_per_ms']).max())
    result=dict(status='PASS',initial_engine_bitwise=True,all_spike_counts_exact_native_record=True,
        fixed_Z_K_bitwise=True,samples=20000,matched_asymmetric_expected_external_blocks=matched,
        max_external_rate_block_difference=delta,no_existing_final_engine_reference=True,producer_sha256=sha(__file__))
    write(OUT/'observer_audit.json',result);write(OUT/'observer_progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--device',type=int,default=0)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    prepare() if a.command=='prepare' else worker(a.device,a.wait)

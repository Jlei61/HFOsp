#!/usr/bin/env python3
"""Read-only input observation at the existing native actual-field K9 high state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,time
import numpy as np
from campaign import ROOT,NATIVE,REPO,read,write,sha
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state
from coupled_density_exit import ADAPTED
from observe_native_exit_inputs import MOMENTS,EARLY

OUT=ROOT/'native_exit_branch_inputs'
NAME='actualfield_K9_high_42to44'
PARENT=ROOT/'exit_return_probes/runs/exit_z0.21_k9_fields16p7_high'
INITIAL=PARENT/'checkpoint.pkl'
STEPS=20000


def configure():
    protocol=read(OUT/'protocol.json');native.OUT=OUT;native.prepare=lambda:protocol
    with np.load(PARENT/'held_fields.npz') as z:zz,kk=z['Z'],z['K']
    native.fields=lambda zbar,kbar:(zz.copy(),kk.copy())
    return protocol


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    source=read(PARENT/'result.json');assert source['status']=='COMPLETE'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_READONLY_CONTINUATION',created_epoch=time.time(),
        question='In actual-exitfield K9 high state, does the low candidate surroundingrate arise from local input/response approximation or from free recurrentcoupling?',
        motivation='CompletedK9 candidate5-10s hasR~217/G~1.74 versus native243/4.3; nativeallE Zdrift-.042/s while candidateendpoint+.026/s, even though bothcores remainnegative. Similar high/quietclasses donotmatchthe recoverymechanism.',
        design='Exactlyone2s unchangednativeconditional continuation42-44s fromcomplete K9high finalstate. Same heldZ/K,allother physics/noisecontinuous. Record all3479groupcounts andthe same16 geometricallypreselected localgroups,0.1ms currents,state,externaldrive,R/G. Notanewparameterpoint orindependentloop.',
        downstream='Use actualupstreamcountandR/G forcing in the clampedlocaldensity process; this futurediagnostic will separate localresponsefromrecurrentcoupling. It cannot certify networkstability.',
        checks='Completeinitialengine identicalto savednative42s state; originalsourcehashesunchanged; everyobserver counts/time align andgroupcountssumto nativeE/I1msobservations. No44s pre-existingreference, so no bitwise44s replayclaim.',
        initial=str(INITIAL),initial_sha256=sha(INITIAL),held_fields_sha256=sha(PARENT/'held_fields.npz'),
        selected_groups=read(EARLY/'contract.json')['selected_groups'],producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    protocol=copy.deepcopy(read(NATIVE/'protocol.json'));protocol.update(stage='NATIVE_ACTUAL_EXIT_FIELD_LOCAL_INPUT_OBSERVER',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',protocol);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz')
    configure();native.make_job(NAME,str(INITIAL),2.,True,.21,9.,False)
    a=native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')['engine'];b=native.read_pickle(INITIAL)['engine']
    assert a['step']==420000;assert_same_state(a,b)
    write(OUT/'initial_gate.json',dict(status='PASS',whole42s_engine_bitwise=True))


def worker(device):
    configure();c=read(OUT/'contract.json');assert sha(__file__)==c['producer_sha256']
    assert sha(INITIAL)==c['initial_sha256'] and sha(PARENT/'held_fields.npz')==c['held_fields_sha256']
    assert read(OUT/'initial_gate.json')['status']=='PASS'
    assert not (OUT/'observer_progress.json').exists(),'No silentrestart'
    geo=dict(np.load(ADAPTED/'geometry.npz'));groups=geo['cell_group'];selected=np.array([r['group'] for r in c['selected_groups']]);G=len(selected)
    table=np.full(len(geo['group_size']),-1,int);table[selected]=np.arange(G)
    cells=np.flatnonzero(table[groups]>=0);ids=table[groups[cells]];sizes=geo['group_size'][selected]
    assert np.array_equal(np.bincount(ids,minlength=G),sizes)
    def avg(x):return np.bincount(ids,weights=x,minlength=G)/sizes
    backend=gpu.cuda_backend.wrap_simulator
    def observing_backend(original,device_index):
        fast=backend(original,device_index=device_index)
        def wrapped(params,net,*args,**kw):
            slow=kw['slow'];state=kw['resume_state'];assert state['step']==420000
            assert np.array_equal(net['pos'],geo['original_positions'])
            E=cells<32000;k=np.zeros(len(cells));k[E]=state['termination_mechanism']['sahp_g'][cells[E]]
            initial=np.stack([state['V'][cells],state['s_E'][cells],state['I_E'][cells],state['s_I'][cells],state['I_I'][cells],state['slow']['m'][cells],state['slow']['z'][cells],k],axis=1)
            thresholds=np.broadcast_to(kw['V_th_per_neuron'],(40000,))
            np.savez_compressed(OUT/'initial_local_state.npz',state=initial,ref=state['ref'][cells],native_cells=cells,selected_group_index=ids,
                selected_groups=selected,actual_threshold_mv=thresholds[cells],ring_sE=state['ring_sE'][:,cells],ring_sI=state['ring_sI'][:,cells],step=state['step'],
                global_R=state['termination_mechanism']['r_global'],global_state=state['global_feedback_response']['global_state'])
            counts=np.empty((STEPS,len(geo['group_size'])),dtype='u2');mom=np.empty((STEPS,len(MOMENTS),G));drive=np.empty((STEPS,G));glob=np.empty((STEPS,2));times=np.empty(STEPS)
            n=ni=ns=0;oldcur=kw.get('current_observer');oldin=kw.get('input_observer');oldsp=kw.get('spike_observer')
            def currents(tm,ie,ii,v):
                nonlocal n
                if oldcur is not None:oldcur(tm,ie,ii,v)
                a,b,vol=ie[cells],ii[cells],v[cells];z=slow.z[cells];m=slow.m[cells];k=np.zeros(len(cells));k[E]=slow.g_k[cells[E]]
                zi=z*b;current=a-zi-.0005*m;raw=30*slow.global_state;eligible=b+E*(18-slow.global_reversal)*raw<slow.cfg.I_th_EI
                values=[a,b,a*a,b*b,vol,vol*vol,z,z*z,m,m*m,k,k*k,zi,zi*zi,current,current*current,a*b,eligible]
                for j,value in enumerate(values):mom[n,j]=avg(value)
                glob[n]=[slow.r_global,slow.global_state];times[n]=tm;n+=1
            def inputs(tm,nu,xi):
                nonlocal ni
                if oldin is not None:oldin(tm,nu,xi)
                drive[ni]=avg(np.broadcast_to(nu,(40000,))[cells]);ni+=1
            def spikes(tm,sp):
                nonlocal ns
                if oldsp is not None:oldsp(tm,sp)
                counts[ns]=np.bincount(groups,weights=sp,minlength=len(geo['group_size'])).astype('u2');ns+=1
            kw.update(current_observer=currents,input_observer=inputs,spike_observer=spikes)
            try:return fast(params,net,*args,**kw)
            finally:
                if n==ni==ns==STEPS:
                    assert np.allclose(times,42000+np.arange(STEPS)*.1,rtol=0,atol=1e-9)
                    np.savez_compressed(OUT/'inputs.npz',time_ms=times,moments=mom,moment_names=MOMENTS,spikes=counts,external_rate_per_ms=drive,
                        global_R_and_s=glob,selected_groups=selected)
                else:write(OUT/'observer_incomplete.json',dict(current_samples=n,input_samples=ni,spike_samples=ns,expected=STEPS))
        return wrapped
    gpu.cuda_backend.wrap_simulator=observing_backend
    write(OUT/'observer_progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    try:gpu.worker(OUT,NAME,device)
    finally:gpu.cuda_backend.wrap_simulator=backend
    folder=OUT/'runs'/NAME;assert read(folder/'result.json')['status']=='COMPLETE'
    raw=[]
    for path in sorted((folder/'chunks').glob('*.npz')):
        with np.load(path) as z:raw.append(z['spikes_1ms'])
    raw=np.concatenate(raw)
    with np.load(OUT/'inputs.npz') as z:
        counts=z['spikes'];e=geo['population']==0
        expected=np.stack([counts[:,mask].sum(1).reshape(2000,10).sum(1) for mask in [e,~e]],axis=1)
        assert np.array_equal(expected,raw[:,:2]);assert np.isfinite(z['moments']).all()
    end=native.read_pickle(folder/'checkpoint.pkl')['engine'];assert end['step']==440000
    with np.load(PARENT/'held_fields.npz') as z:
        assert np.array_equal(end['slow']['z'][:32000],z['Z']) and np.array_equal(end['termination_mechanism']['sahp_g'],z['K'])
    result=dict(status='PASS',samples=STEPS,initial_engine_bitwise=True,held_Z_K_unchanged=True,E_I_group_counts_match_native1ms=True,
        scope='Read-only2s continuationwithsame frozenphysicalengine; noexisting44s reference forbitwisereplay, no newindependentseed orautonomousloop.',formal_bifurcation_allowed=False)
    write(OUT/'observer_audit.json',result);write(OUT/'observer_progress.json',dict(status='COMPLETE_OBSERVATION_QA_PASS',updated_epoch=time.time()))
    print('NATIVE ACTUAL FIELD INPUT OBSERVER PASS',result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','worker']);parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();prepare() if args.command=='prepare' else worker(args.device)

#!/usr/bin/env python3
"""One unchanged natural exit replay with additional local input observations."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,copy,shutil,time
import numpy as np
from campaign import ROOT,REPO,NATIVE,read,write,sha
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state
from native_campaign import STREAMS

OUT=ROOT/'native_exit_input_observation'
NAME='natural16p7_to20_observed'
INITIAL=ROOT/'exit_state_reconstruction/runs/source10_to16p70/checkpoint.pkl'
SOURCE=native.SOURCE/'runs'/native.NAME
GEO=REPO/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g40/geometry.npz'
EARLY=REPO/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/native_early_surround_inputs'
MOMENTS=['IE','II','IE2','II2','V','V2','Z','Z2','M','M2','K','K2','appliedII','appliedII2','net','net2','IE_II','eligible']


def prepare():
    assert read(ROOT/'exit_state_continuation_qa/gate.json')['status']=='PASS'
    assert read(ROOT/'conditional_density_inputs/result.json')['status']=='COMPLETE'
    assert not (OUT/'contract.json').exists();OUT.mkdir(exist_ok=True)
    c=dict(status='REGISTERED_BEFORE_REPLAY',created_epoch=time.time(),
        question='Record actual inputs and joint local states across the natural G/K-mediated exit so that the new density closure can be tested in its relevant conductance/transient domain.',
        scope='Exactlyone3.3s read-only continuation16.7to20s from independently verified complete originalstate; no interventions, newnoise, parameterchanges orautonomous replicate. Existing16 geometric g40targets reused independentlyofexitactivity.',
        observation='Every0.1ms all3479groupcounts;16target current/V/Z/M/K/eligibility moments,global causalR/G,actualmembermean externalrate; initialjointstatesandpendingdelayinput forselectednativecells.',
        gate='Original observationarrays slicedexactly tosameclock plus entire20s engine must matchbitwise; normalizeonlythedeclaredConditionalSlowmetadata classname. No downstreamresponse/scientific acceptance beforegate.',
        downstream='Conditional local test will prescribe actualG/R/incomingcounts. This cannot certifyautonomousGfeedback, spatialpropagation or derivatives. No automatic networkcampaign.',
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),producer_sha256=sha(__file__),
        selected_groups=read(EARLY/'contract.json')['selected_groups'],human_review='PENDING')
    write(OUT/'contract.json',c)
    p=copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='NATIVE_EXIT_READ_ONLY_OBSERVER',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',p);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz')
    native.OUT=OUT;native.prepare=lambda:p
    native.make_job(NAME,str(INITIAL),3.3,clamp=False,common_input=False)


def worker(device):
    c=read(OUT/'contract.json');assert sha(__file__)==c['producer_sha256'] and sha(INITIAL)==c['initial_sha256']
    folder=OUT/'runs'/NAME;assert not (OUT/'observer_progress.json').exists(),'No silent restart'
    geo=dict(np.load(GEO));groups=geo['cell_group'];selected=np.array([r['group'] for r in c['selected_groups']]);G=len(selected)
    table=np.full(len(geo['group_size']),-1,int);table[selected]=np.arange(G)
    cells=np.flatnonzero(table[groups]>=0);ids=table[groups[cells]];sizes=geo['group_size'][selected]
    assert np.array_equal(np.bincount(ids,minlength=G),sizes)
    def avg(v):return np.bincount(ids,weights=v,minlength=G)/sizes
    original_backend=gpu.cuda_backend.wrap_simulator
    def observing_backend(original,device_index):
        fast=original_backend(original,device_index=device_index)
        def wrapped(params,net,*args,**kw):
            slow=kw['slow'];resume=kw['resume_state'];assert resume['step']==167000
            assert np.array_equal(net['pos'],geo['original_positions'])
            thresholds=np.broadcast_to(kw['V_th_per_neuron'],(len(groups),))
            E=cells<slow.NE;K=np.zeros(len(cells));K[E]=resume['termination_mechanism']['sahp_g'][cells[E]]
            state=np.stack([resume['V'][cells],resume['s_E'][cells],resume['I_E'][cells],resume['s_I'][cells],
                resume['I_I'][cells],resume['slow']['m'][cells],resume['slow']['z'][cells],K],axis=1)
            np.savez_compressed(OUT/'initial_local_state.npz',state=state,ref=resume['ref'][cells],native_cells=cells,
                selected_group_index=ids,selected_groups=selected,actual_threshold_mv=thresholds[cells],
                ring_sE=resume['ring_sE'][:,cells],ring_sI=resume['ring_sI'][:,cells],step=resume['step'],
                global_R=resume['termination_mechanism']['r_global'],global_state=resume['global_feedback_response']['global_state'])
            count=np.empty((33000,len(geo['group_size'])),dtype='u2');mom=np.empty((33000,len(MOMENTS),G))
            drive=np.empty((33000,G));global_state=np.empty((33000,2));times=np.empty(33000);n=ns=ni=0
            prev_cur=kw.get('current_observer');prev_sp=kw.get('spike_observer');prev_in=kw.get('input_observer')
            def currents(tm,ie,ii,v):
                nonlocal n
                if prev_cur is not None:prev_cur(tm,ie,ii,v)
                a,b,vol=ie[cells],ii[cells],v[cells];z=slow.z[cells];m=slow.m[cells]
                k=np.zeros(len(cells));k[E]=slow.g_k[cells[E]]
                zi=z*b;net_current=a-zi-.0005*m;raw=30*slow.global_state
                eligible=(b+E*(18-slow.global_reversal)*raw<slow.cfg.I_th_EI)
                values=[a,b,a*a,b*b,vol,vol*vol,z,z*z,m,m*m,k,k*k,zi,zi*zi,net_current,net_current**2,a*b,eligible]
                for j,value in enumerate(values):mom[n,j]=avg(value)
                global_state[n]=[slow.r_global,slow.global_state];times[n]=tm;n+=1
            def inputs(tm,nu,xi):
                nonlocal ni
                if prev_in is not None:prev_in(tm,nu,xi)
                drive[ni]=avg(np.broadcast_to(nu,(len(groups),))[cells]);ni+=1
            def spikes(tm,sp):
                nonlocal ns
                if prev_sp is not None:prev_sp(tm,sp)
                count[ns]=np.bincount(groups,weights=sp,minlength=len(geo['group_size'])).astype('u2');ns+=1
            kw.update(current_observer=currents,input_observer=inputs,spike_observer=spikes)
            try:
                return fast(params,net,*args,**kw)
            finally:
                # Native horizon completion uses its normal Stop exception.
                if n==ns==ni==33000:
                    assert np.allclose(times,16700+np.arange(33000)*.1,atol=1e-9,rtol=0)
                    np.savez_compressed(OUT/'inputs.npz',time_ms=times,moments=mom,moment_names=MOMENTS,spikes=count,
                        external_rate_per_ms=drive,global_R_and_s=global_state,selected_groups=selected)
                else:
                    write(OUT/'observer_incomplete.json',dict(current_samples=n,spike_samples=ns,input_samples=ni,
                        expected=33000,downstream_allowed=False))
        return wrapped
    gpu.cuda_backend.wrap_simulator=observing_backend
    write(OUT/'observer_progress.json',dict(status='RUNNING',pid=os.getpid(),interval_s=[16.7,20.],updated_epoch=time.time()))
    try:gpu.worker(OUT,NAME,device)
    finally:gpu.cuda_backend.wrap_simulator=original_backend
    audit()


def audit():
    folder=OUT/'runs'/NAME;assert read(folder/'result.json')['status']=='COMPLETE'
    checks=[]
    for stream in STREAMS+['z_budget_chunks']:
        if not (SOURCE/stream).exists():continue
        for path in sorted((folder/stream).glob('*.npz')):
            start,end=map(int,path.stem.split('_'))
            matches=[q for q in (SOURCE/stream).glob('*.npz') if int(q.stem.split('_')[0])<=start and int(q.stem.split('_')[1])>=end]
            assert len(matches)==1,(stream,path.name)
            old=matches[0];a,b=map(int,old.stem.split('_'))
            with np.load(path) as x,np.load(old) as y:
                assert set(x.files)==set(y.files),(stream,path.name)
                for key in x.files:
                    if key in ['start_step','end_step']:
                        assert int(x[key])==(start if key=='start_step' else end);continue
                    if key in ['keys','region_names','variables']:expected=y[key]
                    else:
                        stride=(b-a)//len(y[key]);assert stride*len(y[key])==b-a
                        assert (start-a)%stride==0 and (end-a)%stride==0
                        expected=y[key][(start-a)//stride:(end-a)//stride]
                    assert np.array_equal(x[key],expected,equal_nan=x[key].dtype.kind in 'fc'),(stream,path.name,key)
                    checks.append(dict(stream=stream,chunk=path.name,key=key,bitwise=True))
    a=native.read_pickle(folder/'checkpoint.pkl')['engine'];b=native.read_pickle(SOURCE/'states/t20s.pkl')['engine']
    assert a['slow']['kind']=='ConditionalSlow' and b['slow']['kind']=='GlobalResponseSlow'
    a['slow']['kind']=b['slow']['kind'];assert_same_state(a,b)
    with np.load(OUT/'inputs.npz') as z:
        counts=z['spikes'];times=z['time_ms'];geo=dict(np.load(GEO));e=geo['population']==0
        pop=np.rint(counts[:,e].sum(1).reshape(3300,10).sum(1)).astype(int)
        raw=[]
        for path in sorted((folder/'chunks').glob('*.npz')):
            with np.load(path) as x:raw.append(x['spikes_1ms'])
        # The native recorder's first column is all E spike count per1ms.
        actual=np.concatenate(raw)
        assert np.array_equal(pop,actual[:,0]),(pop.shape,actual.shape)
        assert len(times)==33000 and np.isfinite(z['moments']).all()
    result=dict(status='PASS',samples=33000,original_arrays_bitwise=len(checks),checks=checks,
        full20s_engine_bitwise=True,only_metadata_normalization='slow.kind ConditionalSlow toGlobalResponseSlow withclampFalse',
        newly_recorded_group_counts_sum_to_original_E_counts=True,source=str(SOURCE),initial=str(INITIAL),
        observer_only=True,new_autonomous_replicate=False,formal_bifurcation_allowed=False)
    write(OUT/'replay_audit.json',result);write(OUT/'observer_progress.json',dict(status='COMPLETE_REPLAY_AUDIT_PASS',updated_epoch=time.time()))
    print('NATIVE EXIT INPUT REPLAY PASS',len(checks),'arrays, whole20s engine',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','audit']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':worker(a.device)
    else:audit()

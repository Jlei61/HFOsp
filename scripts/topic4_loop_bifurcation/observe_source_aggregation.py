#!/usr/bin/env python3
"""Two-second unchanged K9.35 continuation with individual source statistics.

All observers only read arrays. The completed asymmetric native state is used;
the concurrent held-K9-history experiment and physical engines are untouched.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,time
import numpy as np
from campaign import ROOT,NATIVE,read,write,sha
from coupled_density_exit import ADAPTED
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'native_K9p35_source_statistics'
NAME='unchanged_K9p35_42to44_source_observation'
PARENT=ROOT/'native_exit_K_bracket/runs/exit_z0.21_k9.35_fields16p7_high'
INITIAL=PARENT/'checkpoint.pkl'
STEPS=20000
MOMENTS=['IE','II','IE2','II2','IE_II','V','M']


def configure():
    p=read(OUT/'protocol.json');native.OUT=OUT;native.prepare=lambda:p
    with np.load(PARENT/'held_fields.npz') as z:zz,kk=z['Z'],z['K']
    native.fields=lambda zbar,kbar:(zz.copy(),kk.copy())
    return p


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(PARENT/'result.json')['status']=='COMPLETE'
    residual=read(ROOT/'density_exit_bracket_protocol/comparison/spatial_residual.json')
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_INDIVIDUAL_SOURCE_OBSERVATION',created_epoch=time.time(),
        question='Does source-group rate averaging distort the target inputs near the residual recruitment edge atK9.35, and how do the observed input fluctuations compare with the density assumptions?',
        selection='Development diagnostic at the completed asymmetricK9.35 state;11of400displaycells accountfor80percent of the protocolmatched squaredfield error. Selectedafter seeing thaterror, not independentvalidation.',
        design='Exactly one unchanged2s nativeconditional continuation42-44s. SameheldZ/K, completeinternal/externalstate andnativeequations. Record percellspikecounts and IE/II/currentsecondmoments/covariance/M means in four0.5sblocks, percellmeanexpectedexternalrate, plus causalR/G at0.1ms. No alteredinput orparameters, no newseed.',
        baseline='Compare individual-source weighted input reconstruction with the existing source-group averaged projection using the SAME observed spikecounts andoriginalweights. Finitewindow/filterboundary andexternalPoisson errors must remain separate; do not assume sourceaveraging is the cause.',
        unit='One nativeconditionaltrajectory; cells and0.5sblocks are numericaldiagnostic units, not independent biological replicates.',
        checks='Initialfullengine bitwise, unchangedsourcehashes, complete0.1ms samplecounts, observerE/Ispikes exactlymatch native1msrecord, fixedZ/K retained. There is no existing44s reference, so no whole-final-engine replay claim.',
        source=str(INITIAL),source_sha256=sha(INITIAL),held_fields_sha256=sha(PARENT/'held_fields.npz'),
        field_error_source='density_exit_bracket_protocol/comparison/spatial_residual.json',edge_cells=residual['top10cells'],
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    p=copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='NATIVE_INDIVIDUAL_SOURCE_READONLY_OBSERVER',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',p);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz')
    configure();native.make_job(NAME,str(INITIAL),2.,True,.21,9.35,False)
    a=native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')['engine'];b=native.read_pickle(INITIAL)['engine']
    assert a['step']==420000;assert_same_state(a,b)
    write(OUT/'initial_gate.json',dict(status='PASS',whole42s_engine_bitwise=True))


def worker(device):
    configure();c=read(OUT/'contract.json');assert c['producer_sha256']==sha(__file__)
    assert c['source_sha256']==sha(INITIAL) and c['held_fields_sha256']==sha(PARENT/'held_fields.npz')
    assert read(OUT/'initial_gate.json')['status']=='PASS';assert not (OUT/'observer_progress.json').exists()
    geo=dict(np.load(ADAPTED/'geometry.npz'));backend=gpu.cuda_backend.wrap_simulator
    def observing_backend(original,device_index):
        fast=backend(original,device_index=device_index)
        def wrapped(params,net,*args,**kw):
            slow=kw['slow'];state=kw['resume_state'];assert state['step']==420000
            assert np.array_equal(net['pos'],geo['original_positions'])
            counts=np.zeros((4,40000),dtype='u4');mom=np.zeros((4,len(MOMENTS),40000))
            ext=np.zeros((4,40000));global_state=np.empty((STEPS,2));times=np.empty(STEPS)
            spike_totals=np.empty((STEPS,2),dtype='u2')
            n=ni=ns=0;oldcur=kw.get('current_observer');oldin=kw.get('input_observer');oldsp=kw.get('spike_observer')
            def currents(tm,ie,ii,v):
                nonlocal n
                if oldcur is not None:oldcur(tm,ie,ii,v)
                b=n//5000
                for j,x in enumerate([ie,ii,ie*ie,ii*ii,ie*ii,v,slow.m]):mom[b,j]+=x
                global_state[n]=[slow.r_global,slow.global_state];times[n]=tm;n+=1
            def inputs(tm,nu,xi):
                nonlocal ni
                if oldin is not None:oldin(tm,nu,xi)
                ext[ni//5000]+=np.broadcast_to(nu,(40000,));ni+=1
            def spikes(tm,sp):
                nonlocal ns
                if oldsp is not None:oldsp(tm,sp)
                counts[ns//5000]+=sp
                spike_totals[ns]=[np.count_nonzero(sp[:32000]),np.count_nonzero(sp[32000:])];ns+=1
            kw.update(current_observer=currents,input_observer=inputs,spike_observer=spikes)
            try:return fast(params,net,*args,**kw)
            finally:
                if n==ni==ns==STEPS:
                    assert np.allclose(times,42000+np.arange(STEPS)*.1,rtol=0,atol=1e-9)
                    assert np.isfinite(mom).all() and np.isfinite(ext).all()
                    np.savez_compressed(OUT/'cell_statistics.npz',block_start_s=np.arange(4)*.5+42,
                        block_duration_s=.5,per_cell_spike_counts=counts,per_cell_mean_moments=mom/5000,
                        moment_names=MOMENTS,per_cell_mean_external_per_ms=ext/5000,
                        time_ms=times,global_R_and_s=global_state,spike_totals_0p1ms=spike_totals)
                else:write(OUT/'observer_incomplete.json',dict(current_samples=n,input_samples=ni,spike_samples=ns,expected=STEPS))
        return wrapped
    gpu.cuda_backend.wrap_simulator=observing_backend
    write(OUT/'observer_progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    try:gpu.worker(OUT,NAME,device)
    finally:gpu.cuda_backend.wrap_simulator=backend
    folder=OUT/'runs'/NAME;assert read(folder/'result.json')['status']=='COMPLETE'
    raw=[]
    for p in sorted((folder/'chunks').glob('*.npz')):
        with np.load(p) as z:raw.append(z['spikes_1ms'])
    raw=np.concatenate(raw)
    with np.load(OUT/'cell_statistics.npz') as z:
        assert np.array_equal(z['spike_totals_0p1ms'].reshape(2000,10,2).sum(1),raw[:,:2])
        counts=z['per_cell_spike_counts']
        assert counts[:,:32000].sum()==raw[:,0].sum() and counts[:,32000:].sum()==raw[:,1].sum()
    end=native.read_pickle(folder/'checkpoint.pkl')['engine'];assert end['step']==440000
    with np.load(PARENT/'held_fields.npz') as z:
        assert np.array_equal(end['slow']['z'][:32000],z['Z']) and np.array_equal(end['termination_mechanism']['sahp_g'],z['K'])
    result=dict(status='PASS',samples=STEPS,initial_engine_bitwise=True,held_Z_K_bitwise=True,
        E_I_counts_exact_native1ms=True,individual_counts_sum_exact_native=True,producer_sha256=sha(__file__),
        scope='Read-only sameequation continuation, noexisting44s reference forfinalengine replay. Newobservations, not an independentseed orcausalintervention.',formal_bifurcation_allowed=False)
    write(OUT/'observer_audit.json',result);write(OUT/'observer_progress.json',dict(status='COMPLETE_OBSERVATION_QA_PASS',updated_epoch=time.time()))
    print('INDIVIDUAL SOURCE OBSERVER PASS',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.device)

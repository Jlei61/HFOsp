#!/usr/bin/env python3
"""Exact exogenous mean-rate trajectory from native10s state, no neural replay.

Consume the same original external Poisson draws solely to advance the native
RNG. Their realized counts never enter the density candidate.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time,pickle
import numpy as np
from campaign import ROOT,REPO,read,write,sha
from conditional_density_inputs import OPS
import run_topic4_loop_zk_conditional as native
from checkpoint import restore_external_drive

OUT=ROOT/'coupled_density_exit'
SOURCE=native.SOURCE/'runs'/native.NAME
INITIAL=SOURCE/'states/t10s.pkl'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'forcing_contract.json').exists()
    write(OUT/'forcing_contract.json',dict(status='REGISTERED_BEFORE_RECONSTRUCTION',created_epoch=time.time(),
        question='Supply the exact original expected externalrates to all3479groups at0.1ms during10-20s, removing1msclockandfloat32globalinput approximations from the coupling diagnostic.',
        method='Restore original nativeglobalRNG,xi,andcompleteSpatialOUstate at10s. Advance exactoriginalOU andconsumeoriginalPoisson externaldraws inoriginalorder; no neuronspikes,currentorconnectivitysimulation. Recordmembergroupmeansfloat64.',
        checks='Intermediate16.7s andfinal20s nativeglobalRNG/xi andSpatialOUstate/cached/rng/clocks exactly; selected16groups every0.1ms matchalreadyvalidated16.7-20sobserver, originalsparseglobal/inputmeans throughout10-20s.',
        downstream='Numericalpopulationparticles useexpectedexternalrate, notrealizednativePoissoncounts. This is exogenous input matching, notneuralteacherforcing ornewnative seed.',
        interval_s=[10,20],sampling_ms=.1,initial=str(INITIAL),initial_sha256=sha(INITIAL),producer_sha256=sha(__file__)))
    start=time.time();s,tr,_,identity=native.base.old.setup(9108405);p=s.params
    assert identity==read(ROOT/'native_slices/protocol.json')['identity']
    geo=dict(np.load(OPS/'geometry.npz'));assert np.array_equal(geo['original_positions'],s.net['pos'])
    with INITIAL.open('rb') as f:initial=pickle.load(f)['engine']
    rng=np.random.default_rng();rng.bit_generator.state=initial['rng_state'];xi=initial['xi']
    spatial=native.base.old.make_external_drive(s,tr['spatial_ou'],9108405);restore_external_drive(initial,spatial)
    nu=p.nu_ext_ratio*native.base.old.simulate_kick.__globals__['compute_nu_theta'](p)[0]
    aa=np.exp(-p.dt/p.tau_n);sigma=p.sigma_n*1e-3*np.sqrt(p.tau_n/2.);bb=sigma*np.sqrt(1-aa*aa)
    trace=np.lib.format.open_memmap(OUT/'drive_0p1ms.npy',mode='w+',dtype='f8',shape=(100000,len(geo['group_size'])))
    sparse_reference={}
    for path in sorted((SOURCE/'chunks').glob('*.npz')):
        a,b=map(int,path.stem.split('_'))
        if b<=100000 or a>=200000:continue
        with np.load(path) as z:
            for row in z['inputs']:
                tick=round(row[0]/.1)
                if 100000<=tick<200000:sparse_reference[tick]=row.copy()
    with np.load(ROOT/'native_exit_input_observation/inputs.npz') as z:
        selected=z['selected_groups'];observed=z['external_rate_per_ms']
    references={167000:ROOT/'exit_state_reconstruction/runs/source10_to16p70/checkpoint.pkl',200000:SOURCE/'states/t20s.pkl'}
    checks=[];max_sparse_error=0.;max_selected_error=0.;sparse_count=0
    def verify(tick):
        with references[tick].open('rb') as f:target=pickle.load(f)['engine']
        assert xi==target['xi'] and rng.bit_generator.state==target['rng_state'],tick
        d=target['external_drive']
        assert np.array_equal(spatial._state,d['field_state']) and np.array_equal(spatial._cached,d['cached'])
        assert spatial._rng.bit_generator.state==d['rng_state']
        assert spatial._next_step==d['next_step'] and spatial._last_step==d['last_step']
        checks.append(dict(native_step=tick,global_xi_rng_bitwise=True,spatial_complete_state_bitwise=True))
    for index in range(100000):
        tick=100000+index
        if tick in references:verify(tick)
        tm=10000+index*.1
        xi=aa*xi+bb*rng.standard_normal();glob=max(0.,nu+xi)
        vec=np.full(40000,glob);vec[:32000]=np.maximum(vec[:32000]+spatial.step(tm),0.)
        rng.poisson(vec*.1,size=40000)
        trace[index]=np.bincount(geo['cell_group'],weights=vec,minlength=len(geo['group_size']))/geo['group_size']
        if tick in sparse_reference:
            expected=sparse_reference[tick]
            actual=np.array([tm,xi,vec[:32000].mean(),vec[32000:].mean()])
            err=float(abs(actual-expected).max());assert err==0.,(tick,err)
            max_sparse_error=max(max_sparse_error,err);sparse_count+=1
        if tick>=167000:
            err=float(abs(trace[index,selected]-observed[tick-167000]).max());assert err==0.,(tick,err)
            max_selected_error=max(max_selected_error,err)
        if (index+1)%5000==0:
            write(OUT/'forcing_progress.json',dict(status='RUNNING',pid=os.getpid(),time_s=(tick+1)*.0001,elapsed_s=time.time()-start))
            print('EXOGENOUS RECONSTRUCTION',(tick+1)*.0001,flush=True)
    verify(200000);trace.flush();assert sparse_count==100
    result=dict(status='PASS',interval_s=[10,20],sampling_ms=.1,groups=len(geo['group_size']),dtype='float64',
        endpoint_checks=checks,sparse_original_records_bitwise=sparse_count,selected16_original_steps_bitwise=33000,
        max_sparse_error=max_sparse_error,max_selected_error=max_selected_error,elapsed_s=time.time()-start,
        neural_simulations=0,source_sha256=sha(__file__),initial_sha256=sha(INITIAL))
    write(OUT/'forcing_qa.json',result);write(OUT/'forcing_progress.json',result);print('FORCING PASS',result,flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Validate the observer repair before accepting any root or new derivative."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from campaign import ROOT,read,write,sha
import phase_lif_mc as phase
from measure_target_direct_response import run as original_run
import measure_target_raw_G_response as rawG
import lif_mc

OUT=ROOT/'held_exit_phase_response_validation'
SOURCE=ROOT/'held_exit_stationarity_K9p35'
AUDIT=ROOT/'held_exit_response_window_audit'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    a=read(AUDIT/'result.json');assert a['status']=='COMPLETE_RESPONSE_WINDOW_AUDIT'
    with np.load(AUDIT/'inputs.npz') as z:cells=z['cells'];pars=z['pars']
    with np.load(SOURCE/'inputs.npz') as z:g=z['g'];physical=z['physical']
    R=2048;seed=929371
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_PHASE_REPAIR_COUNTS',created_epoch=time.time(),
        question='Does randomizing the counted starting phase remove the demonstrated high-rate finite-window bias, and do stationary DC gains depend on burn/count protocol?',
        prior_evidence='Four coretarget counts changed476.0→476.25Hz when4s counting start shifted0.5ms, despite countSEM0. All four16s starts gave476.1875Hz. Original fixed4s estimator is not adequate for subSEM stationary-root claims at these inputs.',
        design=dict(selected_targets=cells.tolist(),replicas=R,record_ms=[4000,16000],burn_ms=1000,
            extra_burn_uniform_discrete_ms=[0,1000],seed=seed,device=0,
            derivative_protocols=['original4s','phase4s_step_at_count_start','phase4s_modulated_burn','phase16s_modulated_burn']),
        implementation='Same local cell/synaptic/reset/refractory update. Add a reproducible perreplica extra burn, generated without consuming the physics RNG; same paths across amplitude pairs. Optional constant DC perturbation during burn approaches the perturbed stationary law before counting. No network-engine changes.',
        qa='With extra burn0 and original modulation timing, unmodulated and original mean/variance/shunt paired counts must reproduce prior kernels bitwise. Every recording still has exactly T/.1ms counted steps.',
        readouts='Full selected-cell4/16s count response, paired duration differences, four-channelfull/half-amplitude DC comparisons. Zero or weak gains remain unestimable; 10%/2SEM amplitude rule unchanged.',
        limits='Selected cells only; randomphase finite-record convergence is not proof of alltarget stationarity. Any failed case remains in outputs. No new branch/root gate relaxation or automatic alltarget replacement.',
        producer_sha256=sha(__file__),observer_sha256=sha(phase.__file__),formal_bifurcation_allowed=False))
    started=time.time()
    # Original static observers agree bitwise when new options are disabled.
    for mode in [0,1]:
        old=lif_mc.run(pars[:6],64,100,30,929379,crn=bool(mode),device=0,batch=6)
        new=phase.run(pars[:6],64,100,30,929379,phase_ms=0,stream_mode=mode)
        assert np.array_equal(old,new)
    conditions=[];rows=[]
    for j,cell in enumerate(cells):
        # All selected cells, preserving zero/weak response cases.
        channels=range(4 if cell<32000 else 3)
        steps=[.02*(pars[j,1]-11),.05*physical[cell,1],.1*physical[cell,2],.002*(1+g[cell])]
        for ch in channels:
            if steps[ch]<=0:continue
            for factor in [1.,.5]:
                p=pars[j].copy();p[20]=ch;p[22]=g[cell];p[23]=20. if cell<32000 else 10.
                amp=steps[ch]*factor;p[4]=amp/(physical[cell,ch] if ch in [1,2] else 1.)
                conditions.append(p);rows.append(dict(cell=int(cell),channel=ch,amplitude=amp,factor=factor))
    conditions=np.array(conditions)
    # Four physical channels at a selected E target, original timing and streams.
    qa=conditions[:8]
    old=original_run(rawG.kernel(),qa,64,100,30,929379)
    new=phase.run(qa,64,100,30,929379,phase_ms=0,stream_mode=1)
    assert np.array_equal(old,new)
    write(OUT/'implementation_qa.json',dict(status='PASS',static_stream_modes_bitwise=[0,1],physical_four_channels_bitwise=True,
        physics_kernel_changed=False,observer_only=True))
    # Static values: phase-randomized4s versus16s on identical replica paths.
    static=[]
    for T in [4000,16000]:
        write(OUT/'progress.json',dict(status='STATIC_PHASE_COUNTS',pid=os.getpid(),record_ms=T,updated_epoch=time.time()))
        out=phase.run(pars,R,T,1000,seed,phase_ms=1000,stream_mode=0)
        assert not out[:,:,3].any();static.append(out[:,:,2]/(T/1000))
    static=np.array(static);mean=static.mean(-1);sem=static.std(-1,ddof=1)/np.sqrt(R)
    np.savez_compressed(OUT/'static_response.npz',cells=cells,rates_Hz=static,mean_Hz=mean,SEM_Hz=sem)
    derivative=[];protocols=[(4000,0,False),(4000,1000,False),(4000,1000,True),(16000,1000,True)]
    for index,(T,jitter,warm) in enumerate(protocols):
        write(OUT/'progress.json',dict(status='DERIVATIVE_PROTOCOL_COUNTS',pid=os.getpid(),protocol=index,total=4,updated_epoch=time.time()))
        out=phase.run(conditions,R,T,1000,seed+1,phase_ms=jitter,dc_during_burn=warm,stream_mode=2)
        counts=out[:,:,2:];difference=counts[:,:,0]-counts[:,:,1]
        assert np.array_equal(out[:,:,0],difference*.5) and not out[:,:,1].any()
        gain=difference/(2*(T/1000)*np.array([q['amplitude'] for q in rows])[:,None])
        derivative.append(gain);np.savez_compressed(OUT/f'derivative_protocol_{index}.npz',counts=counts.astype('i4'),gain=gain)
    derivative=np.array(derivative);mu=derivative.mean(-1);se=derivative.std(-1,ddof=1)/np.sqrt(R)
    staticrows=[]
    for i,cell in enumerate(cells):
        delta=static[1,i]-static[0,i];tol=max(3*delta.std(ddof=1)/np.sqrt(R),1e-7)
        staticrows.append(dict(cell=int(cell),mean4s_Hz=float(mean[0,i]),mean16s_Hz=float(mean[1,i]),SEM4s_Hz=float(sem[0,i]),
            SEM16s_Hz=float(sem[1,i]),duration_delta_Hz=float(delta.mean()),paired_SEM_Hz=float(delta.std(ddof=1)/np.sqrt(R)),
            duration_agrees_at3pairedSEM=bool(abs(delta.mean())<=tol)))
    comparisons=[]
    for i,row in enumerate(rows):
        q=dict(row,gain_by_protocol=mu[:,i].tolist(),SEM_by_protocol=se[:,i].tolist())
        # Compare predeclared protocols, retain covariance from shared paths.
        tests=[]
        for before,after in [(0,1),(1,2),(2,3),(0,3)]:
            delta=derivative[after,i]-derivative[before,i];ds=float(delta.std(ddof=1)/np.sqrt(R));tol=max(.1*abs(mu[after,i]),2*ds,1e-7)
            tests.append(dict(before=before,after=after,difference=float(delta.mean()),paired_SEM=ds,tolerance=tol,
                estimable_reference=bool(abs(mu[after,i])>=10*max(se[after,i],1e-15)),within_tolerance=bool(abs(delta.mean())<=tol)))
        q['protocol_comparisons']=tests;comparisons.append(q)
    np.savez_compressed(OUT/'derivative_comparison.npz',gain=derivative,means=mu,SEM=se,
        cells=np.array([q['cell'] for q in rows]),channels=np.array([q['channel'] for q in rows]),
        amplitudes=np.array([q['amplitude'] for q in rows]))
    result=dict(status='COMPLETE_PHASE_OBSERVER_VALIDATION',static_rows=staticrows,derivative_rows=comparisons,
        static_duration_agreement_count=sum(q['duration_agrees_at3pairedSEM'] for q in staticrows),
        static_duration_total=len(staticrows),elapsed_s=time.time()-started,formal_bifurcation_allowed=False,
        root_gate_changed=False,interpretation='Protocol-dependent estimator effects are retained. This audit informs a measurement repair, not a new physical mechanism, root or stable branch.')
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status=result['status'],elapsed_s=result['elapsed_s']))
    print('PHASE VALIDATION COMPLETE',result['static_duration_agreement_count'],result['static_duration_total'],flush=True)


if __name__=='__main__':main()

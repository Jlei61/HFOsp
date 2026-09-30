#!/usr/bin/env python3
"""Bounded checks of the distribution-retaining local kernel, not closure QA."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from colored_population import LocalDensityParticles,parameters,cpu_supplied,lif_mc

OUT=ROOT/'colored_population_local'


def main(device):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_LOCAL_CHECKS',created_epoch=time.time(),
        question='Does explicitly retaining voltage, refractory state and both colored-current pairs give a faithful resumable local Markov solver, including changing conductance?',
        scope='No fitted response, no new native trajectory, no new network parameters. Exact reference Gaussian-current diffusion remains an approximation to native shot noise. These checks certify implementation only; no spatial closure or bifurcation promotion.',
        reason='Static v3/v4 pass independent mean rates but their independent gains fail, including DC variance signs. Stop generic history-MLP expansion. Retain the physical distribution before deciding whether the remaining network moment approximation is usable.',
        candidate='Per particle V,qA,IA,qG,IG,absolute refractory counter and independent numerical RNG. Current fluctuations stay raw when g changes. Population rate is threshold flux, not a fitted function of recent input means.',
        checks=['Eight heterogeneous E/I conditions,64 particles,400 supplied-noise steps versus independent CPU; g and raw mean/variance vary, including deterministic and zero-g endpoints.',
                'Exactly identical final states and counts for whole versus irregular chunked evolution, same initial state and RNG.',
                'E constant-g Gaussian-LIF counts at18 predefined workpoints against existing independent local solver; same random streams,8192 replicas,500ms burn and1000ms record. Compare per-replica counts, not a fitted population mean.'],
        source_sha256=sha(__file__),kernel_sha256=sha(__import__('colored_population').__file__),
        reference_sha256=sha(lif_mc.__file__),dt_ms=.1,device=device,
        boundary='Particles approximate a density. Their number is numerical resolution, not native network size or independent native seeds. No physical finite-population noise is inferred from particle sampling noise.'))
    start=time.time();rng=np.random.default_rng(927601);P,R,T=8,64,400
    pops=['E','I']*4;pars=parameters(np.linspace(14,22,P),pops)
    drive=np.empty((T,P,4));tt=np.arange(T)[:,None]
    drive[:,:,3]=(np.arange(P)[None,:]/2)*(1+np.sin(tt/31))
    drive[:,0,3]=0.;drive[:,:,0]=(1+drive[:,:,3])*(17+3*np.sin(tt/47+np.arange(P)))
    drive[:,:,1]=150*(1+.7*np.cos(tt/53+np.arange(P)));drive[:,:,2]=250*(1+.8*np.sin(tt/37+np.arange(P)))
    drive[:,0,1:3]=0.;drive[200:,:,3]*=2
    initial=rng.normal(0,2,(P,R,5));initial[:,:,0]=rng.uniform(11,13,(P,R));ref=rng.integers(0,20,(P,R),dtype=np.int32)
    noise=rng.normal(size=(T,P,R,4));cpu=cpu_supplied(pars,drive,noise,initial,ref)
    e=LocalDensityParticles(pars,R,drive,device=device,initial=initial,ref_initial=ref)
    cp=e.cp;sp=cp.zeros((T,P,R),dtype=cp.uint8)
    e.module.get_function('supplied_noise')(((P*R+127)//128,),(128,),(
        e.pars,e.drive,cp.asarray(noise),e.state,e.ref,sp,np.int32(P),np.int32(R),np.int32(T),.1))
    error=float(abs(e.state.get()-cpu[0]).max());assert error<1e-10,error
    assert np.array_equal(e.ref.get(),cpu[1]) and np.array_equal(sp.get(),cpu[2])
    parity=dict(maximum_physical_state_error=error,spikes_exact=True,refractory_exact=True,total_spikes=int(cpu[2].sum()))
    a=LocalDensityParticles(pars,R,drive,seed=927602,device=device)
    b=LocalDensityParticles(pars,R,drive,seed=927602,device=device)
    a.finish()
    for size in [1,7,31,113,3,245]:b.advance(size)
    for key in ['state','ref','rng','counts']:assert np.array_equal(getattr(a,key).get(),getattr(b,key).get()),key
    write(OUT/'implementation_check.json',dict(status='PASS',supplied_noise=parity,irregular_chunks_bitwise=True,
        rng_state_bytes=a.rng_bytes,independent_physical_check_only=True))
    del e,a,b;cp.get_default_memory_pool().free_all_blocks()
    rows=[]
    for g in [0.,2.,8.]:
        for mu in [7.5,14.5,19.4]:
            for se,si in [(2.,3.),(.2,.3)]:rows.append(dict(g=g,mu_eff=mu,sigma_E_eff=7*se,sigma_I_eff=7*si))
    write(OUT/'progress.json',dict(status='REFERENCE_CONSTANT_G',pid=os.getpid(),completed=0,total=len(rows)))
    for lo in range(0,len(rows),3):
        sub=rows[lo:lo+3];P=len(sub);pars=parameters([18.]*P,['E']*P)
        d=np.array([[r['mu_eff']*(1+r['g']),(r['sigma_E_eff']*(1+r['g']))**2,
                     (r['sigma_I_eff']*(1+r['g']))**2,r['g']] for r in sub])
        drive=np.broadcast_to(d,(15000,P,4)).copy()
        e=LocalDensityParticles(pars,8192,drive,seed=927603,device=device,bin_ms=100.,crn=True)
        observed=e.finish();ours=observed[:,:,5:].sum(2);reference=[]
        for row in sub:
            p=lif_mc.condition(row['mu_eff'],18.,row['sigma_E_eff']**2,row['sigma_I_eff']**2,'E')
            p[18]=np.exp(-.1*(1+row['g'])/20.);reference.append(p)
        target=lif_mc.run(reference,8192,1000.,500.,927603,device=device,batch=3)[:,:,2]
        for i,row in enumerate(sub):
            diff=ours[i]-target[i];row.update(maximum_count_difference=int(abs(diff).max()),
                nonidentical_replicas=int(np.count_nonzero(diff)),mean_Hz=float(ours[i].mean()),reference_mean_Hz=float(target[i].mean()))
            # Arithmetic is differently ordered in raw vs normalized currents.
            # No changed firing is accepted as bitwise state parity.
            row['counts_identical']=bool(np.array_equal(ours[i],target[i]))
        np.savez_compressed(OUT/f'constant_g_{lo:02d}.npz',counts=ours,reference_counts=target)
        write(OUT/'progress.json',dict(status='REFERENCE_CONSTANT_G',pid=os.getpid(),completed=lo+P,total=len(rows)))
        del e;cp.get_default_memory_pool().free_all_blocks()
    identical=all(r['counts_identical'] for r in rows)
    result=dict(status='LOCAL_IMPLEMENTATION_PASS' if identical else 'REFERENCE_COUNT_DIFFERENCE_REVIEW',
        independent_cpu_parity=parity,chunk_continuation_exact=True,constant_g_rows=rows,
        elapsed_s=time.time()-start,spatial_closure_validated=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status=result['status'],pid=os.getpid()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

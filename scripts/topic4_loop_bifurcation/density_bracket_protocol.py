#!/usr/bin/env python3
"""Two same-initial-history immediate-clamp density correspondence runs.

Uses the already checked individual-target physical kernel and the native
jobs' original12s state/held fields. No equation, tolerance, or branch extension.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
import target_density_exit as base
import target_density_field_family as family
import density_spatial as physical

OUT=ROOT/'density_exit_bracket_protocol'
SOURCE=ROOT/'native_exit_K_bracket'
NAMES=[f'exit_z0.21_k{k}_fields16p7_high' for k in ['9.35','9.5']]


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(base.OUT/'implementation_qa.json')['status']=='PASS'
    baseline=read(ROOT/'exit_return_probes/jobs/exit_z0.21_k9_fields16p7_high.json')
    for name in NAMES:
        job=read(SOURCE/'jobs'/f'{name}.json')
        for key in ['source_checkpoint','external_noise_source','branch_start_s','Z_template','K_template']:
            assert job[key]==baseline[key],key
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_SAMPLING',created_epoch=time.time(),
        question='Does the native K9.35 reduced-recruitment state appear when the density uses the same immediate clamp and original12s initial history, rather than its previous carried K9 ramp?',
        selection='Adaptive follow-up after completed native0-20s atK9.35 gives allE~98/coreA~392/coreB~132Hz, unlike carried-density~200/472/475. NativeK9.5 completed30s quiet. Full nativeK9.35 is still running; no final-prefix claim.',
        design='Exactly two10s40000targetx128particle runs at heldK9.35and9.5. Same original12s initial V/synapse/M/refractory/pending/R/G and actual16.7s Z/K fields as native jobs; same50-60s external expected-rate drive as previously validatedK9field checks; local Gaussian numericalseed928751. Only K differs between the two candidates. Compare0-5and5-10s, rates/core/400cell field/G/counterfactualZdrift.',
        interpretation='A match would support the relevance of approach history/input protocol, not prove history alone because the old density ramp also used constant expectedexternalmean. A mismatch retains a closure/finite-network-noise question. Same expecteddrive is not the same localPoisson orGaussian realization. No automatic newK, roots, or threshold changes.',
        stop='Exactly two10s runs; preserve all mismatch evidence. No extension or new parameter follows automatically.',
        jobs=[dict(name=n,job_sha256=sha(SOURCE/'jobs'/f'{n}.json')) for n in NAMES],
        producer_sha256=sha(__file__),base_sha256=sha(base.__file__),family_sha256=sha(family.__file__),physical_sha256=sha(physical.__file__),
        formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))


def worker(index,device):
    c=read(OUT/'contract.json')
    for key,path in [('producer_sha256',__file__),('base_sha256',base.__file__),('family_sha256',family.__file__),('physical_sha256',physical.__file__)]:
        assert c[key]==sha(path)
    row=c['jobs'][index];name=row['name'];folder=OUT/name;folder.mkdir(exist_ok=True)
    assert sha(SOURCE/'jobs'/f'{name}.json')==row['job_sha256']
    assert not (folder/'progress.json').exists(),'No silent restart'
    started=time.time();write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device))
    e=family.construct(SOURCE,name,device)
    assert np.array_equal(e.state.get(),e.initial_state.get())
    assert np.array_equal(e.ref.get(),e.initial_ref.get())
    assert np.array_equal(e.global_state.get(),e.initial_global)
    assert np.array_equal(e.pending.get(),e.pending_cpu)
    write(folder/'initial_qa.json',dict(status='PASS',same_declared_native_initial_cell_ref_pending_R_G=True,
        held_fields_exact=True,expected_drive_source='exit_branch_density/drive_0p1ms.npy',
        local_Gaussian_seed=928751,original_physics_unchanged=True))
    e.graph();outputs=[];globals=[]
    for offset in range(0,10000,10):
        outputs.append(e.chunk().astype('f4'));globals.append(e.global_output.get())
        if (offset+10)%500==0:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,
                simulation_s=(offset+10)/1000,elapsed_wall_s=time.time()-started,updated_epoch=time.time()))
            print(name,(offset+10)/1000,flush=True)
    assert int(e.clock.get()[0])==100000
    assert np.array_equal(e.state.get()[:,:,6:8],e.initial_state.get()[:,:,6:8])
    values=np.concatenate(outputs);glob=np.concatenate(globals)
    assert np.isfinite(values).all() and np.isfinite(glob).all()
    np.savez_compressed(folder/'trajectory.npz',elapsed_time_ms=np.arange(1,10001),group_output=values,
        channels=['rate_Hz','Z','M','K','IE','applied_II','V','abs_current','Z_eligible_fraction'],global_R_Hz=glob[:,0],global_s=glob[:,1])
    np.savez_compressed(folder/'final_state.npz',**{k:getattr(e,k).get() for k in ['state','ref','rng','history','clock','global_state']})
    result=dict(status='COMPLETE',name=name,duration_s=10.,physical_targets=40000,replicas=128,
        elapsed_s=time.time()-started,held_fields_bitwise=True,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--index',type=int);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.index,a.device)

#!/usr/bin/env python3
"""One frozen 8192-particle full-window numerical-resolution check."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time,argparse
import numpy as np
from campaign import ROOT,read,write,sha
from density_spatial import DensityNetwork

OUT=ROOT/'density_spatial_resolution'


def main(device):
    previous=read(ROOT/'density_spatial_onset/audit.json');assert previous['status']=='COMPLETE'
    assert all(r['n_original_A4_pass']==6 for r in previous['rows'])
    assert not (OUT/'contract.json').exists();OUT.mkdir(exist_ok=True)
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_NUMERICAL_REFINEMENT',created_epoch=time.time(),
        question='Do late resource depletion and entry persist when numerical particle noise is reduced fourfold in count, before treating this approximation as a deterministic population density?',
        evidence='Two2048-particle numerical streams pass originalA4six numerical criteria, but entry differs11.153vs11.924s and D9870=.20777/.21230 vsnative.25634. OriginalD tolerance.05 is nearly saturated.512/2048earlyprefixqualitativeagreement is insufficient forfullentryconvergence.',
        bounded_design='Exactlyone8192-particle/group,12.5s,seed927611 with nestedgroup/particleRNG streams. Same original recorded8401externalforcing and no addedG/K. No changed physical parameters, thresholds, groups, dt or acceptance limits.',
        interpretations='If morphology/lateZ/entry changes materially, particle-noise convergence is unresolved and continuation remains premature. If stable, still compare nativecontacts and G/K before promotion. This is numericalresolution, not a newnative seed or biologicalreplication.',
        next_gate='No automatic furtherresolutionorparametergrid. Report numerical variation versus originalnative tolerances, retaining the separate observed slowerZdepletion.',
        source_sha256=sha(__file__),engine_sha256=sha(__import__('density_spatial').__file__),device=device,
        replicas=8192,duration_ms=12500,seed=927611,formal_bifurcation_allowed=False))
    start=time.time();e=DensityNetwork(replicas=8192,seed=927611,device=device,duration_ms=12500,gain=0.);e.graph();output=[]
    for tick in range(0,12500,10):
        x=e.chunk();assert np.isfinite(x).all() and x[:,1].min()>=0 and x[:,1].max()<=1;output.append(x)
        if (tick+10)%250==0:write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=tick+10,elapsed_s=time.time()-start))
    data=np.concatenate(output);cell=e.geo['group_cell'];field=np.zeros((len(data),400));count=np.zeros(400)
    for g in np.flatnonzero(e.E):field[:,cell[g]]+=data[:,0,g]*e.sizes[g];count[cell[g]]+=e.sizes[g]
    field/=np.maximum(count,1)
    arrays=dict(time_ms=np.arange(12500)+1.,field_E_Hz=field.astype('f4'),cell_counts=count,group_sizes=e.sizes,population_E=e.E)
    for j,k in enumerate(['group_rate_Hz','group_Z','group_M','group_K','group_IE','group_applied_II','group_V','group_abs_current']):arrays[k]=data[:,j].astype('f4')
    np.savez_compressed(OUT/'trajectory.npz',**arrays)
    np.savez_compressed(OUT/'final_state.npz',state=e.state.get(),ref=e.ref.get(),history=e.history.get(),rng=e.rng.get(),clock=e.clock.get(),global_state=e.global_state.get(),accumulator=e.accumulator.get(),particle_count=8192,seed=927611)
    result=dict(status='COMPLETE',duration_ms=12500,elapsed_s=time.time()-start,numerical_particles_per_group=8192,native_correspondence_certified=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

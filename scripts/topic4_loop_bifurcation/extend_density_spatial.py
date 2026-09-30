#!/usr/bin/env python3
"""Bounded early-to-onset distribution-closure diagnostic after 3s review."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time,argparse
import numpy as np
from campaign import ROOT,read,write,sha
from density_spatial import DensityNetwork

BASE=ROOT/'density_spatial_baseline'
OUT=ROOT/'density_spatial_onset'


def register():
    if (OUT/'contract.json').exists():return read(OUT/'contract.json')
    d=read(BASE/'comparison.json');assert read(BASE/'runs.json')['status']=='COMPLETE'
    rows=[r for r in d['rows'] if r['kind']=='density_approximation']
    assert len(rows)==2 and all(r['primary_500_3000']['n']>0 and r['summary']['high_onset_ms'] is None for r in rows)
    OUT.mkdir(exist_ok=True)
    c=dict(status='FROZEN_AFTER_REVIEW_OF_3S_PREFIX',created_epoch=time.time(),
        question='Do retained local state distributions preserve late interictal propagation and predict the observed Z-driven entry without fitting its time?',
        evidence='512/2048 numerical particles/group yield12/11events in0.5–3s, median82/81ms,9bothcoreevents each,1forward/11or10reverse. Native8401/2 yield12/15events,95.5/86ms. Densityquietfraction.634/.639 exceeds native.552/.504, so this is promising but not accepted.',
        decision='Extend only2048-particle existing realization to12.5s using its complete saved state; addone2048-particle independentnumericalstream fromcold to12.5s. Same original recorded8401externaldrive andphysical model. No equation/parameter/response fitting.',
        jobs=[dict(name='R2048_num927611',seed=927611,resume=str(BASE/'R2048_num927611'),start_ms=3000),
              dict(name='R2048_num927612',seed=927612,resume=None,start_ms=0)],
        duration_ms=12500,replicas=2048,added_global_gain=0.,
        readouts='Original whole/core/surround rates, allqualifiedeventduration/quiet/area/direction,400cellfield,15contactcurrent envelopes, same-clock Z/M, entry. Keep original A4 rules with their original observationwindow; do not substitute3s counts for fullgate. Also retain stricter quiet-bounded complete-event subset.',
        statistical_unit='One physical external forcing, two numerical particle streams; not two native noise seeds.512/2048prefix checks numericalresolution only.',
        gate='This is a diagnostic, not promoted closure. Apply pre-existing full-window A4 criteria and native-noise comparisons; visual space/contact review required. If late depletion/entry/propagation fails, diagnose spatial/input/correlation closure before any continuation.',
        source_sha256=sha(__file__),engine_sha256=sha(__import__('density_spatial').__file__),
        original_prefix_sha256={n:sha(BASE/'R2048_num927611'/n) for n in ['trajectory.npz','final_state.npz']},
        resources='One extra GPU process, sequentialtwojobs; originalnativequeueunmodified.',formal_bifurcation_allowed=False)
    write(OUT/'contract.json',c);return c


def run(device):
    c=register();assert sha(__file__)==c['source_sha256'];assert sha(__import__('density_spatial').__file__)==c['engine_sha256']
    assert not (OUT/'status.json').exists();status=dict(status='RUNNING',pid=os.getpid(),completed=[],total=2);write(OUT/'status.json',status)
    for job in c['jobs']:
        folder=OUT/job['name'];folder.mkdir();start=time.time()
        e=DensityNetwork(replicas=2048,seed=job['seed'],device=device,duration_ms=12500,gain=0.);e.graph();output=[]
        if job['resume']:
            from pathlib import Path
            previous=Path(job['resume'])
            for n,h in c['original_prefix_sha256'].items():assert sha(previous/n)==h
            with np.load(previous/'final_state.npz') as z:
                assert z['particle_count']==2048 and z['seed']==job['seed'] and z['clock'][0]==30000
                for k in ['state','ref','history','rng','clock','global_state','accumulator']:getattr(e,k)[:]=e.cp.asarray(z[k])
            # Preserve the original stored observations exactly; no initial reset.
            with np.load(previous/'trajectory.npz') as z:
                keys=['group_rate_Hz','group_Z','group_M','group_K','group_IE','group_applied_II','group_V','group_abs_current']
                output.append(np.stack([z[k] for k in keys],axis=1).astype(float))
        for tick in range(job['start_ms'],12500,10):
            x=e.chunk();assert np.isfinite(x).all();assert x[:,1].min()>=0 and x[:,1].max()<=1.;output.append(x)
            if (tick+10)%250==0:write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=tick+10,elapsed_s=time.time()-start))
        data=np.concatenate(output);assert data.shape==(12500,8,e.P)
        cell=e.geo['group_cell'];field=np.zeros((len(data),400));count=np.zeros(400)
        for g in np.flatnonzero(e.E):field[:,cell[g]]+=data[:,0,g]*e.sizes[g];count[cell[g]]+=e.sizes[g]
        field/=np.maximum(count,1)
        arrays=dict(time_ms=np.arange(len(data))+1.,field_E_Hz=field.astype('f4'),cell_counts=count,group_sizes=e.sizes,population_E=e.E)
        for j,k in enumerate(['group_rate_Hz','group_Z','group_M','group_K','group_IE','group_applied_II','group_V','group_abs_current']):arrays[k]=data[:,j].astype('f4')
        np.savez_compressed(folder/'trajectory.npz',**arrays)
        if job['resume']:
            with np.load(previous/'trajectory.npz') as z:
                for k in ['group_rate_Hz','group_Z','group_M','group_K','group_IE','group_applied_II','group_V','group_abs_current']:
                    assert np.array_equal(z[k],arrays[k][:3000]),k
            write(folder/'prefix_verification.json',dict(status='PASS',all_stored_group_observations_unchanged=True,source=str(previous)))
        np.savez_compressed(folder/'final_state.npz',state=e.state.get(),ref=e.ref.get(),history=e.history.get(),rng=e.rng.get(),clock=e.clock.get(),global_state=e.global_state.get(),accumulator=e.accumulator.get(),particle_count=2048,seed=job['seed'])
        row=dict(status='COMPLETE',time_ms=12500.,elapsed_s=time.time()-start,job=job,native_correspondence_certified=False)
        write(folder/'result.json',row);write(folder/'progress.json',row);print(job['name'],row,flush=True)
        status['completed'].append(job['name']);write(OUT/'status.json',status);cp=e.cp;del e;cp.get_default_memory_pool().free_all_blocks()
    status['status']='COMPLETE';write(OUT/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);run(p.parse_args().device)

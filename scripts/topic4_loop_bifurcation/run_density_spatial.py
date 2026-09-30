#!/usr/bin/env python3
"""Only the two frozen 3s numerical-resolution prerequisites."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,write,read,sha
from density_spatial import DensityNetwork

OUT=ROOT/'density_spatial_baseline'


def main(device):
    contract=read(OUT/'contract.json');assert read(OUT/'implementation_check.json')['status']=='PASS'
    assert sha(__import__('density_spatial').__file__)==contract['engine_sha256']
    assert not (OUT/'runs.json').exists()
    progress=dict(status='RUNNING',pid=os.getpid(),completed=[],total=2);write(OUT/'runs.json',progress)
    for R in [512,2048]:
        name=f'R{R}_num927611';folder=OUT/name;folder.mkdir();start=time.time()
        e=DensityNetwork(replicas=R,seed=927611,device=device,duration_ms=3000,gain=0.);e.graph();output=[]
        write(folder/'identity.json',dict(graph_identity=e.prep['graph_identity'],numerical_particles_per_group=R,numerical_seed=927611,
            native_source_input_seed=9108401,counts_as_native_realization=False,physical_E_cells=int(e.sizes[e.E].sum()),physical_I_cells=int(e.sizes[~e.E].sum()),
            source_sha256=contract['engine_sha256'],groups=e.P))
        for k in range(300):
            x=e.chunk();assert np.isfinite(x).all();assert x[:,1].min()>=0 and x[:,1].max()<=1
            output.append(x)
            if (k+1)%25==0:
                write(folder/'progress.json',dict(status='RUNNING',time_ms=(k+1)*10,elapsed_s=time.time()-start,pid=os.getpid(),
                    last10ms_mean_E_Hz=float(np.average(x[:,0,e.E].mean(0),weights=e.sizes[e.E]))))
        data=np.concatenate(output);cell=e.geo['group_cell'];field=np.zeros((len(data),400));count=np.zeros(400)
        for g in np.flatnonzero(e.E):field[:,cell[g]]+=data[:,0,g]*e.sizes[g];count[cell[g]]+=e.sizes[g]
        field/=np.maximum(count,1)
        np.savez_compressed(folder/'trajectory.npz',time_ms=np.arange(len(data))+1.,group_rate_Hz=data[:,0].astype('f4'),
            group_Z=data[:,1].astype('f4'),group_M=data[:,2].astype('f4'),group_K=data[:,3].astype('f4'),group_IE=data[:,4].astype('f4'),
            group_applied_II=data[:,5].astype('f4'),group_V=data[:,6].astype('f4'),group_abs_current=data[:,7].astype('f4'),
            field_E_Hz=field.astype('f4'),cell_counts=count,group_sizes=e.sizes,population_E=e.E)
        np.savez_compressed(folder/'final_state.npz',state=e.state.get(),ref=e.ref.get(),history=e.history.get(),rng=e.rng.get(),clock=e.clock.get(),global_state=e.global_state.get(),
            accumulator=e.accumulator.get(),particle_count=R,seed=927611)
        row=dict(status='COMPLETE',duration_ms=3000.,elapsed_s=time.time()-start,numerical_particles_per_group=R,
            wholeE_mean_Hz=float(np.average(data[:,0,e.E].mean(0),weights=e.sizes[e.E])),native_correspondence_certified=False)
        write(folder/'result.json',row);write(folder/'progress.json',row);progress['completed'].append(name);write(OUT/'runs.json',progress)
        print(name,row,flush=True);cp=e.cp;del e;cp.get_default_memory_pool().free_all_blocks()
    progress['status']='COMPLETE';write(OUT/'runs.json',progress)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

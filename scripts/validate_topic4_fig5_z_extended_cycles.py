"""Off-grid residuals and spatially weighted readouts, without altering a cycle."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,argparse,time
import numpy as np
from scipy.signal import resample
from topic4_fig5_z_frozen_v1 import Orbit
from continue_topic4_fig5_z_periodic_arc import OUT,PRIOR


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');p.add_argument('--refined',action='store_true');a=p.parse_args()
    from topic4_fig5_z_frequency_parallel import install
    install(2)
    files=[]
    if a.physical:files=sorted((OUT/'periodic_physical').glob('s*_N512.npz'))
    elif a.refined:files=sorted((OUT/'arc_a_refined_N128').glob('point*.npz'))
    else:
        for folder in ['periodic_a_N64','periodic_b_N64']:
            files+=sorted((OUT/folder).glob('amp*.npz'))
        for folder,inds in [('arc_a_N64',[0,2,4,6,8,9,10,12,15,20,25]),('arc_b_N128',[0,1,2,3,4,5])]:
            files+=[OUT/folder/f'point{k:04d}.npz' for k in inds if (OUT/folder/f'point{k:04d}.npz').exists()]
    rows=[];o=None
    name='physical_resolution_qa' if a.physical else ('refined_periodic_resolution_qa' if a.refined else 'extended_periodic_resolution_qa')
    if (OUT/f'{name}.json').exists():rows=json.loads((OUT/f'{name}.json').read_text())
    done={r['file'] for r in rows}
    for path in files:
        if str(path) in done:continue
        z=np.load(path);N=len(z['r']);s=float(z['s']);T=float(z['T'])
        if float(z['residual_hz'])>1e-5:continue
        if o is None or o.N!=2*N:o=Orbit(s,2*N)
        o.s=s;o.z,o.z2=o.eq.z(s);m=o.m;r=resample(z['r'],2*N,axis=0)
        f=o.evaluate_fixed(r,T);dense=resample(z['r'],8*N,axis=0)
        macro=(dense[:,:3200]*m.w_u).reshape(8*N,400,8).sum(2)
        global_e=macro@m.count_e/m.count_e.sum()*1000
        row=dict(file=str(path),s=s,mean_z=1-s,N=N,period_ms=T,residual_hz=float(z['residual_hz']),off_grid_residual_hz=float(abs(f).max()*1000),off_grid_rms_hz=float(np.sqrt(np.mean(f*f))*1000),unit_minimum_hz=float(dense.min()*1000),unit_maximum_hz=float(dense.max()*1000),mean_hz=float(global_e.mean()),minimum_hz=float(global_e.min()),maximum_hz=float(global_e.max()))
        rows.append(row);print(row,flush=True);(OUT/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()

"""Controlled native-current replay: threshold before/after spatial averaging.

This is an offline diagnostic, not an additional network simulation or a fit.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
from run_topic4_spatial_kinetic_candidate import ROOT,SOURCE,OUT,write


def main():
    threshold=95.19851312666987;rows={}
    geometry={g:dict(np.load((SOURCE/f'approx/coarse_{g}' if g==20 else OUT/f'coarse_{g}')/'geometry.npz')) for g in (20,40)}
    for seed in (9108401,9108402):
        ts=[];zz=[];duties=[]
        for path in sorted((SOURCE/f'replay/runs/eta0.0005_s{seed}/fields').glob('*.npz')):
            with np.load(path) as x:
                steps=x['zm_step'];take=(steps>=80000)&(steps<=105000)&(steps%100==0)
                if not take.any():continue
                current=x['ii'][take];zs=x['z'][take]
                for step,ii,z in zip(steps[take],current,zs):
                    d=[float(np.mean(ii>=threshold))]
                    for g in (20,40):
                        cell=geometry[g]['cell_e'];count=geometry[g]['count_e']
                        mean=np.bincount(cell,weights=ii,minlength=g*g)/np.maximum(count,1)
                        d.append(float(np.sum(count*(mean>=threshold))/32000.))
                    ts.append(step*.0001);zz.append(float(z.mean()));duties.append(d)
        t=np.array(ts);z=np.array(zz);d=np.array(duties)
        assert len(t)==251 and np.allclose(np.diff(t),.01)
        reconstructed=np.zeros_like(d);reconstructed[0]=z[0]
        for k in range(1,len(t)):reconstructed[k]=reconstructed[k-1]+.01/5*(1-d[k-1]-reconstructed[k-1])
        summaries={}
        for name,lo,hi in [('pre_entry',8.,9.42),('transition',9.42,9.87),('entry_followup',9.87,10.37)]:
            ix=(t>=lo-1e-10)&(t<hi-1e-10)
            summaries[name]=dict(native_fraction_above_threshold=float(d[ix,0].mean()),
                 grid20_mean_current_fraction=float(d[ix,1].mean()),grid40_mean_current_fraction=float(d[ix,2].mean()),
                 grid20_minus_native=float((d[ix,1]-d[ix,0]).mean()),grid40_minus_native=float((d[ix,2]-d[ix,0]).mean()))
        idx=round((9.87-8)*100)
        rows[str(seed)]=dict(windows=summaries,at_native_entry=dict(actual_Z=float(z[idx]),
            sampled_neuron_current_Z=float(reconstructed[idx,0]),grid20_mean_current_Z=float(reconstructed[idx,1]),
            grid40_mean_current_Z=float(reconstructed[idx,2]),
            grid20_minus_sampled_neuron=float(reconstructed[idx,1]-reconstructed[idx,0]),
            grid40_minus_sampled_neuron=float(reconstructed[idx,2]-reconstructed[idx,0])),
            temporal_reconstruction_max_abs_error=float(np.max(abs(z-reconstructed[:,0]))))
        np.savez_compressed(OUT/f'z_threshold_averaging_s{seed}.npz',t_s=t,native_Z=z,duty=d,reconstructed_Z=reconstructed)
    write(OUT/'z_threshold_averaging.json',dict(results=rows,scope='Native currents held fixed; uniform10ms samples, identical temporal approximation for all reconstructions. Does not isolate feedback/correlation errors of the autonomous candidate.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

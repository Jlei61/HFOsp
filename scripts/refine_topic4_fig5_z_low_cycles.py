"""Refine the extended core-A family before using it in the wider figure."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,json
from scipy.signal import resample
from continue_topic4_fig5_z_periodic_arc import Arc,OUT
from topic4_fig5_z_frequency_parallel import install


def main():
    install(2);arc=Arc(128,0);dest=OUT/'arc_a_refined_N128';dest.mkdir(exist_ok=True);rows=[]
    for k in [2,4,6,8,9,10,12,15]:
        path=OUT/'arc_a_N64'/f'point{k:04d}.npz';z=np.load(path);r=resample(z['r'],128,axis=0);y=arc.pack(r,float(z['T']),float(z['s']));tan=np.zeros_like(y);tan[-1]=.1
        result,err,h=arc.correct(y,tan,y,maxiter=6);rr,T,s=arc.unpack(result)
        row=dict(index=k,s=s,period_ms=T,residual_hz=err,source=str(path),N=128);rows.append(row)
        np.savez_compressed(dest/f'point{k:04d}.npz',r=rr,T=T,s=s,residual_hz=err,N=128,source=str(path));print('REFINED LOW CYCLE',row,flush=True)
        (dest/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()

"""Actual orbit samples to display the extremely narrow folds smoothly."""
from common import *
from folds import Chart,metric
from scipy.signal import resample
import numpy as np

def segment(name,left,right,fractions):
    N=2048;scale=.01
    def get(path):
        z=np.load(path);return np.r_[(resample(z['r'],N,axis=0)/.01).ravel(),np.log(float(z['T'])),float(z['g'])/scale]
    a,b=get(left),get(right);t=b-a;t/=np.sqrt(metric(t,t,N));span=metric(b-a,t,N);chart=Chart(System(),a,t,N,scale)
    dest=OUT/'arcs'/name;dest.mkdir(parents=True,exist_ok=True)
    rows=read(dest/'progress.json') if (dest/'progress.json').exists() else []
    if rows:
        last=get(rows[-1]['source']);lastcoord=span*rows[-1]['fraction'];last,tangent,*_=chart.solve(last,lastcoord)
    else:last=a;tangent=t;lastcoord=0
    for i,f in enumerate(fractions):
        if i<len(rows):continue
        coord=span*f;guess=last+tangent*(coord-lastcoord)/(chart.normal@tangent)
        z,tangent,err,*_=chart.solve(guess,coord);last=z;lastcoord=coord
        r=z[:-2].reshape(N,6)*.01;path=dest/f'point{i:03d}.npz'
        np.savez_compressed(path,r=r,T=np.exp(z[-2]),g=z[-1]*scale,residual=err)
        rows.append(dict(fraction=float(f),g=float(z[-1]*scale),mean_hz=(r.mean(0)*1000).tolist(),source=str(path),residual=err))
        write(f'arcs/{name}/progress.json',rows);print(name,i,rows[-1]['g'],err,flush=True)

if __name__=='__main__':
    segment('between_low_folds',OUT/'folds/surround_first_fold_N4096.npz',OUT/'folds/surround_second_fold_N4096.npz',np.linspace(0,1,21))
    segment('between_fold_and_flip',OUT/'folds/surround_second_fold_N4096.npz',OUT/'flips/surround_micro_flip_N4096.npz',np.linspace(0,2,25))
    raw=read(V4/'arcs/recruitment_turn/progress.json')
    segment('approach_first_fold',raw[3]['source'],OUT/'folds/surround_first_fold_N4096.npz',np.linspace(0,1,21))

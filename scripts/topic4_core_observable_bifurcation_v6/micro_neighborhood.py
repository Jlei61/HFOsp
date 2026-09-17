"""Resolve the stable interval between LP0b and PD0 in arc coordinates."""
from common import *
from folds import Chart,metric
from scipy.signal import resample
import numpy as np
import subprocess,concurrent.futures

def main():
    N=2048;scale=.01
    def get(path):
        z=np.load(path);return np.r_[(resample(z['r'],N,axis=0)/.01).ravel(),np.log(float(z['T'])),float(z['g'])/scale]
    a=get(V5/'folds/surround_second_fold_N2048.npz');b=get(OUT/'flips/surround_micro_flip_N2048.npz')
    t=b-a;t/=np.sqrt(metric(t,t,N));span=metric(b-a,t,N);chart=Chart(System(),a,t,N,scale)
    dest=OUT/'arcs/micro_neighborhood';dest.mkdir(parents=True,exist_ok=True);rows=[]
    for i,f in enumerate([0,.25,.5,.75,1,1.25,1.5,2]):
        z,tan,err,*_=chart.solve(a+(b-a)*f,span*f);r=z[:-2].reshape(N,6)*.01;T=np.exp(z[-2]);g=z[-1]*scale
        path=dest/f'fraction{f:g}.npz';np.savez_compressed(path,r=r,T=T,g=g,residual=err)
        row=dict(fraction=f,g=float(g),T_ms=float(T),mean_hz=(r.mean(0)*1000).tolist(),residual=err,source=str(path));rows.append(row)
        write('arcs/micro_neighborhood/progress.json',rows);print('MICRO_POINT',json.dumps(row),flush=True)
    def run(row):
        with open('/tmp/core_v6_neighborhood_mu_'+str(row['fraction'])+'.log','w') as log:
            q=subprocess.run([sys.executable,str(ROOT/'scripts/topic4_core_observable_bifurcation_v6/stability.py'),row['source'],'--dt','.025'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        print('MICRO_MU',row['fraction'],q.returncode,flush=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(run,[rows[i] for i in [1,3,5]]))

if __name__=='__main__':main()

"""Use an observed group mean as a local continuation coordinate.

This crosses parameter folds without treating mean rate as a model parameter.
The unknown physical EE multiplier is solved together with the periodic orbit.
"""
from common import *
from folds import Chart
from scipy.sparse.linalg import gmres
import numpy as np,argparse

def run(path,name,group,target,step=5):
    z=np.load(path);r=z['r'];N=len(r);gscale=.01;s=System();q=np.r_[(r/.01).ravel(),np.log(float(z['T'])),float(z['g'])/gscale]
    normal=np.zeros((N,6));normal[:,group]=1;tan=np.r_[normal.ravel(),0,0]
    current=float(r[:,group].mean()*1000);direction=np.sign(target-current);targets=list(np.arange(current+direction*step,target,direction*step))+[target]
    dest=OUT/'means'/name;dest.mkdir(parents=True,exist_ok=True);rows=[]
    while abs(target-current)>1e-8:
        value=current+direction*min(step,abs(target-current))
        chart=Chart(s,q,tan,N,gscale);F,B,*_=chart.evaluate(q);rhs=np.zeros(len(q));rhs[-1]=1
        pred,info=gmres(B,rhs,rtol=1e-9,atol=1e-11,restart=150,maxiter=25)
        if info:raise RuntimeError(('mean predictor',info))
        coordinate=(value-current)/10;guess=q+pred*coordinate
        try:
            candidate,t,err,*_=chart.solve(guess,coordinate)
        except RuntimeError:
            step/=2
            print('REDUCE_MEAN_STEP',name,step,flush=True)
            if step<.01:raise
            continue
        q=candidate;r=q[:-2].reshape(N,6)*.01;g=q[-1]*gscale;T=np.exp(q[-2]);current=value
        file=dest/f'mean{value:.6f}.npz';np.savez_compressed(file,r=r,T=T,g=g,residual=err,N=N,tangent=t,gscale=gscale)
        row=dict(mean_coordinate_group=group,target_hz=value,g=float(g),T_ms=float(T),residual=err,mean_hz=(r.mean(0)*1000).tolist(),source=str(file));rows.append(row)
        (dest/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('MEAN_CONTINUE',name,json.dumps(row),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--from-orbit',required=True);ap.add_argument('--name',required=True);ap.add_argument('--group',type=int,required=True);ap.add_argument('--target',type=float,required=True);ap.add_argument('--step',type=float,default=5);a=ap.parse_args();run(a.from_orbit,a.name,a.group,a.target,a.step)

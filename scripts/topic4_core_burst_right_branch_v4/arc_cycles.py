"""Pseudo-arclength continuation of the full periodic boundary-value problem.

The physical parameter is an unknown, allowing true cycle folds to be crossed.
All points retain a phase condition and are saved individually in arc order.
"""
from common import *
from periodic import Orbit
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator,gmres
import numpy as np,argparse

def dot(a,b,N):return (a[:-2]@b[:-2])/N+a[-2:]@b[-2:]

def continue_arc(old_path,current_path,name,steps=40,ds=.04,N=2048,gscale=.002):
    s=System();dest=OUT/'arcs'/name;dest.mkdir(parents=True,exist_ok=True)
    def loadpoint(path):
        z=np.load(path);return np.r_[(resample(z['r'],N,axis=0)/.01).ravel(),np.log(float(z['T'])),float(z['g'])/gscale]
    old=loadpoint(old_path);y=loadpoint(current_path);tan=y-old;tan/=np.sqrt(dot(tan,tan,N));rows=[]
    for idx in range(steps):
        ref=y[:-2].reshape(N,6)*.01;k=np.arange(N//2+1)
        deriv=np.fft.irfft(2j*np.pi*k[:,None]*np.fft.rfft(ref,axis=0),n=N,axis=0)
        phase=deriv/np.sum(deriv*deriv)*.01
        prev=y.copy();oldtan=tan.copy();accepted=False
        for retry in range(8):
            pred=prev+ds*oldtan;trial=pred.copy();history=[]
            def evaluate(yv,with_jac=False):
                g=yv[-1]*gscale;orb=Orbit(s,g,N)
                if with_jac:F,J,_=orb.evaluate(yv[:-1],ref,phase,True)
                else:F=orb.evaluate(yv[:-1],ref,phase)
                F=np.r_[F,dot(yv-pred,oldtan,N)]
                if not with_jac:return F
                eps=1e-6
                plus=Orbit(s,g+eps,N).evaluate(yv[:-1],ref,phase)
                minus=Orbit(s,g-eps,N).evaluate(yv[:-1],ref,phase)
                dg=(plus-minus)/(2*eps)*gscale
                def mv(dy):return np.r_[J@dy[:-1]+dg*dy[-1],dot(dy,oldtan,N)]
                return F,LinearOperator((len(yv),len(yv)),matvec=mv)
            for it in range(14):
                F,J=evaluate(trial,True);err=float(abs(F).max());history.append(err)
                print('ARC_NEWTON',name,idx,retry,it,'g',trial[-1]*gscale,'T',np.exp(trial[-2]),'res',err,'ds',ds,flush=True)
                if err<1e-9:accepted=True;break
                step,info=gmres(J,-F,rtol=min(1e-5,max(1e-9,err*.005)),atol=1e-12,restart=140,maxiter=14)
                norm=np.linalg.norm(F);alpha=1.
                for back in range(14):
                    test=trial+alpha*step
                    if abs(test[-2]-trial[-2])<.4 and abs(test[-1]-trial[-1])*gscale<.03 and np.isfinite(test).all() and np.linalg.norm(evaluate(test))<norm:break
                    alpha*=.5
                else:break
                trial=test
            if accepted:break
            ds*=.5
            if ds<2e-5:break
        if not accepted:raise RuntimeError(('Arclength failed',name,idx,ds,history))
        y=trial
        # A bordered null solve gives the tangent; its parameter component
        # detects a turning point without relying on differences in g alone.
        F,J=evaluate(y,True);rhs=np.zeros(len(y));rhs[-1]=1
        tan,info=gmres(J,rhs,rtol=1e-9,atol=1e-11,restart=140,maxiter=16)
        if info:raise RuntimeError(('Tangent failed',name,idx,info))
        tan/=np.sqrt(dot(tan,tan,N));tan*=np.sign(dot(tan,oldtan,N))
        r=y[:-2].reshape(N,6)*.01;T=float(np.exp(y[-2]));g=float(y[-1]*gscale)
        path=dest/f'point{idx:03d}_g{g:.10f}.npz'
        np.savez_compressed(path,r=r,T=T,g=g,residual=float(abs(F).max()),history=history,tangent=tan,N=N,gscale=gscale)
        row=dict(index=idx,g=g,T_ms=T,N=N,residual=float(abs(F).max()),tangent_g=float(tan[-1]*gscale),mean_hz=(r.mean(0)*1000).tolist(),min_hz=(r.min(0)*1000).tolist(),max_hz=(r.max(0)*1000).tolist(),source=str(path))
        rows.append(row);(dest/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('ARC_ACCEPTED',json.dumps(row),flush=True)
        ds=min(.08,ds*(1.2 if len(history)<5 else .9))
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--current',required=True);ap.add_argument('--name',required=True);ap.add_argument('--steps',type=int,default=40);ap.add_argument('--ds',type=float,default=.04);a=ap.parse_args()
    continue_arc(a.old,a.current,a.name,a.steps,a.ds)

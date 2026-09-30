"""Parameter continuation of the old TP folds, with a minimally augmented solve."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from scipy.linalg import eig,solve
from topic4_fig5_D_physical_model import Equilibrium,OUT
from audit_topic4_fig5_D_folds import audit

def correct(eq,r,D):
    r=r.copy();history=[]
    for it in range(20):
        f,j=eq.evaluate(r,D,True);es,ls,vs=eig(j,left=True,right=True);idx=np.argmin(abs(es))
        if abs(es[idx].imag)>1e-7:raise RuntimeError('critical eigenvalue became complex')
        ev=es[idx].real;v=vs[:,idx].real;v/=np.linalg.norm(v);w=ls[:,idx].real;w/=w@v
        error=max(float(max(abs(f))),abs(ev));history.append(error);print('FOLD HOMOTOPY',eq.q_ie,it,D,error,flush=True)
        if max(abs(f))<1e-8 and abs(ev)<1e-9:return r,D,history
        h=.02;jp=eq.evaluate(r+h*v,D,True)[1];jm=eq.evaluate(r-h*v,D,True)[1];grad=w@(jp-jm)/(2*h)
        hd=1e-6;fp,jp=eq.evaluate(r,D+hd,True);fm,jm=eq.evaluate(r,D-hd,True);fd=(fp-fm)/(2*hd);gd=w@((jp-jm)@v)/(2*hd)
        B=np.empty((801,801));B[:800,:800]=j;B[:800,800]=fd;B[800,:800]=grad;B[800,800]=gd
        delta=solve(B,-np.r_[f,ev],check_finite=False);old=np.linalg.norm(f)/100+abs(ev)
        for back in range(10):
            rr=r+2.**(-back)*delta[:800];DD=D+2.**(-back)*delta[-1]
            if rr.min()<-1e-7 or abs(DD-D)>.03:continue
            ff,jj=eq.evaluate(rr,DD,True);ee=eig(jj,left=False,right=False);e=ee[np.argmin(abs(ee))]
            merit=np.linalg.norm(ff)/100+abs(e)
            if merit<old:r,D=rr,DD;break
        else:raise RuntimeError(('fold homotopy line search',eq.q_ie,it,error))
    raise RuntimeError(('no fold convergence',eq.q_ie,error))

def main():
    rows=[]
    for kind in ['TP','TP2']:
        initial=np.load(OUT/f'old_{kind}.npz');r=initial['r_hz'];D=float(initial['D']);q=1.;step=.025
        while q<1.25-1e-10:
            target=min(1.25,q+step);eq=Equilibrium(target)
            try:rr,DD,h=correct(eq,r,D)
            except RuntimeError as exc:
                print('RETRY',str(exc),flush=True);step/=2
                if step<.001:raise
                continue
            r,D,q=rr,DD,target;name=f'q{q:.6f}_tracked_{kind}';path=OUT/f'{name}.npz';np.savez_compressed(path,r_hz=r,D=D)
            rows.append(dict(kind=kind,q_ie=q,D=D,mean_e_hz=float(np.average(r[:400],weights=eq.m.count_e)),history=h,filename=str(path)))
            (OUT/'fold_working_point_homotopy.json').write_text(json.dumps(rows,indent=2)+'\n');step=min(step*1.3,.025)
        audit(f'q1.25_tracked_{kind}',path,1.25)

if __name__=='__main__':main()

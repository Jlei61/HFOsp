"""Refine a -1 Floquet crossing using anti-periodic variational collocation."""
from common import *
from folds import Chart,metric
from antiperiodic import build
from scipy.signal import resample
from scipy.optimize import brentq
from scipy.sparse.linalg import eigs
import numpy as np,argparse

def refine(left,right,name,N=2048,pair=False):
    scale=.01
    def load(path):
        z=np.load(path);return np.r_[(resample(z['r'],N,axis=0)/.01).ravel(),np.log(float(z['T'])),float(z['g'])/scale]
    a=load(left);b=load(right);t=b-a;t/=np.sqrt(metric(t,t,N));span=metric(b-a,t,N);s=System();chart=Chart(s,a,t,N,scale);cache={}
    def at(x):
        if x not in cache:
            z,*_=chart.solve(a+(b-a)*x/span,x);r=z[:-2].reshape(N,6)*.01;T=np.exp(z[-2]);g=z[-1]*scale;K=build(s,r,T,g)
            vals,vec=eigs(K,k=80,which='LM',ncv=170,tol=2e-10,maxiter=800);j=np.argmin(abs(vals-1))
            if abs(vals[j].imag)>1e-6 and not pair:raise RuntimeError(('not a real anti mode',vals[j]))
            test=float(np.prod(1-vals[np.argsort(abs(vals-1))[:2]]).real) if pair else vals[j].real-1
            cache[x]=(z,K,vals[j],vec[:,j].real,test)
            print('FLIP_REFINE',name,x,g,'K_eigenvalue',vals[j],flush=True)
        return cache[x]
    x=brentq(lambda x:at(x)[4],0,span,xtol=1e-9,rtol=1e-12);z,K,val,w,_=at(x)
    assert abs(val.imag)<1e-7
    vals,vec=eigs(K.T,k=80,which='LM',ncv=170,tol=2e-10,maxiter=800);j=np.argmin(abs(vals-1));leftvec=vec[:,j].real;leftvec/=np.linalg.norm(leftvec)
    w/=np.sqrt(np.sum(w*w)/N)
    eps=min(.003,span*.02);slope=(at(x+eps)[4]-at(x-eps)[4])/(2*eps)
    r=z[:-2].reshape(N,6)*.01;T=np.exp(z[-2]);g=z[-1]*scale
    row=dict(name=name,g=float(g),T_ms=float(T),N=N,anti_eigenvalue=float(val.real),anti_null_residual=float(abs(w-K@w).max()),left_null_residual=float(abs(leftvec-K.T@leftvec).max()),
        crossing_test=('two_mode_determinant' if pair else 'anti_eigenvalue_minus_one'),crossing_slope_per_arc=float(slope),mean_hz=(r.mean(0)*1000).tolist(),left_source=str(left),right_source=str(right))
    dest=OUT/'flips';dest.mkdir(exist_ok=True);path=dest/f'{name}_N{N}.npz'
    np.savez_compressed(path,r=r,T=T,g=g,N=N,residual=float(abs(chart.evaluate(z,x,False)).max()),mode=w.reshape(N,6),left_mode=leftvec.reshape(N,6))
    row['source']=str(path);(dest/f'{name}_N{N}.json').write_text(json.dumps(row,indent=2)+'\n');print('FLIP',json.dumps(row),flush=True)
    return path

def doubled(path,name,amplitudes):
    z=np.load(path);base=z['r'];T=float(z['T']);g=float(z['g']);mode=z['mode'];N=2*len(base);scale=1e-6
    rr=np.tile(base,(2,1));ww=np.r_[mode,-mode];t=np.r_[ww.ravel(),0,0];t/=np.sqrt(metric(t,t,N));a=np.r_[(rr/.01).ravel(),np.log(2*T),g/scale]
    chart=Chart(System(),a,t,N,scale);dest=OUT/'periodic'/name;dest.mkdir(parents=True,exist_ok=True);last=a.copy();oldamp=0
    for amp in amplitudes:
        guess=last+(amp-oldamp)*t;q,tan,err,*_=chart.solve(guess,amp);r=q[:-2].reshape(N,6)*.01;Tnew=float(np.exp(q[-2]));gnew=float(q[-1]*scale)
        path=dest/f'amp{amp:g}_N{N}.npz';np.savez_compressed(path,r=r,T=Tnew,g=gnew,N=N,residual=err,amplitude=amp,tangent=tan,gscale=scale)
        row=dict(g=gnew,T_ms=Tnew,N=N,residual=err,amplitude=amp,mean_hz=(r.mean(0)*1000).tolist(),half_period_difference=float(np.linalg.norm(r-np.roll(r,N//2,axis=0))/np.linalg.norm(r-r.mean(0))),source=str(path))
        path.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');print('DOUBLED',json.dumps(row),flush=True);last=q;oldamp=amp

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--left');ap.add_argument('--right');ap.add_argument('--name',required=True);ap.add_argument('--N',type=int,default=2048);ap.add_argument('--from-flip');ap.add_argument('--pair',action='store_true');ap.add_argument('--amplitudes',type=float,nargs='+',default=[.03,.06,.12,.24]);a=ap.parse_args()
    if a.from_flip:doubled(a.from_flip,a.name,a.amplitudes)
    else:refine(a.left,a.right,a.name,a.N,a.pair)

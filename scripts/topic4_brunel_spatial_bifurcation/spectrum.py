"""Verified temporal roots; stationary eigenvalues never stand in for them."""
from common import *
from model import SpatialBrunel
from response import characteristic
from scipy.sparse.linalg import eigs
from scipy.optimize import root
import argparse

def nearest(s,r,J,lam):
    M=characteristic(s,r,J,complex(lam));ev,vec=eigs(M,k=3,sigma=0,tol=1e-10)
    i=np.argmin(abs(ev));v=vec[:,i];v/=np.linalg.norm(v)
    return ev[i],v,float(np.linalg.norm(M@v))

def main(args):
    s=SpatialBrunel(args.grid);dest=OUT/f'g{args.grid}'/args.label;dest.mkdir(parents=True,exist_ok=True)
    r=None;records=[]
    for J in args.values:
        r,ok,trace=s.solve(J,r)
        if not ok:raise RuntimeError((J,trace[-1]))
        roots=[];sample=[]
        for freq in args.freqs:
            l=1e-7+2j*np.pi*freq/1000;ev,v,res=nearest(s,r,J,l)
            sample.append(dict(frequency_hz=freq,nearest=ev))
            def fun(x):
                lam=complex(*x)
                if abs(lam)>.7 or lam.real<-.08:return [1e3+abs(lam),1e3+abs(lam)]
                q,_,_=nearest(s,r,J,lam)
                return [q.real,q.imag]
            try:
                sol=root(fun,[l.real,l.imag],tol=1e-9)
                lam=complex(*sol.x);q,v,res=nearest(s,r,J,lam)
                if abs(q)<1e-7 and res<1e-7 and all(abs(lam-x['lambda_per_ms'])>1e-6 for x in roots):
                    if lam.imag<0:lam=lam.conjugate();v=v.conjugate()
                    roots.append(dict(lambda_per_ms=lam,residual=res,vector=v))
                    print('root',J,lam,'Hz',lam.imag*1000/(2*np.pi),'res',res,flush=True)
            except (ValueError,RuntimeError,OverflowError) as e: print('search',J,freq,str(e)[:100],flush=True)
        roots.sort(key=lambda q:-q['lambda_per_ms'].real)
        np.savez_compressed(dest/f'J{J:.6f}.npz',rates=r,**{f'v{k}':x['vector'] for k,x in enumerate(roots)})
        records.append(dict(J_EE_core=J,rates_hz=s.regional_rates(r),roots=[{k:v for k,v in q.items() if k!='vector'} for q in roots],samples=sample))
        write(dest/'result.json',dict(status='ROOT_SEARCH_COMPLETE' if J==args.values[-1] else 'RUNNING',rows=records,
            response='Eq36 with colored shifted bounds and matched static gain; Table2 extrapolation is not used',
            spectrum_complete=False,native_correspondence='NOT_VALIDATED'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--values',type=float,nargs='+',default=[.9,1.,1.02])
    p.add_argument('--freqs',type=float,nargs='+',default=[.05,1,3,6,10,20,40,80]);p.add_argument('--label',default='dynamic_roots');main(p.parse_args())

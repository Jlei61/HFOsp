"""Follow a spatial mode and solve its imaginary-axis crossing self-consistently."""
from common import *
from model import SpatialBrunel
from response import characteristic
from scipy.sparse.linalg import eigs
from scipy.optimize import root
import argparse
from functools import lru_cache

def main(args):
    s=SpatialBrunel(args.grid,response=args.response);suffix='_'+args.response if args.response!='shifted_white' else ''
    dest=OUT/f'g{args.grid}'/f'hopf_{args.core}{suffix}';dest.mkdir(parents=True,exist_ok=True)
    core={'A':0,'B':1}[args.core];mask=s.E&(s.geo['group_region']==core);weight=s.geo['group_size']
    r0,ok,_=s.solve(.94);assert ok
    calls=[]
    @lru_cache(maxsize=32)
    def stationary(J):
        r,ok,_=s.solve(float(J),r0)
        if not ok:raise ValueError('stationary root failed')
        return r
    def critical_mode(J,lam):
        r=stationary(J)
        M=characteristic(s,r,float(J),complex(lam));ev,vec=eigs(M,k=8,sigma=0,tol=1e-10)
        energy=weight[:,None]*abs(vec)**2;fractions=energy[mask].sum(0)/energy.sum(0)
        eligible=np.flatnonzero(fractions>.1)
        idx=eligible[np.argmin(abs(ev[eligible]))] if len(eligible) else int(np.argmax(fractions))
        return ev[idx],vec[:,idx],r
    def fun(x):
        J,omega=x;ev,v,r=critical_mode(J,1j*omega)
        calls.append(dict(J=J,frequency_hz=omega*1000/(2*np.pi),eigenvalue=ev))
        return [ev.real,ev.imag]
    sol=root(fun,[args.start,.032],tol=1e-9);J,omega=sol.x;ev,v,r=critical_mode(J,1j*omega)
    assert abs(ev)<1e-8,(sol.message,J,omega,ev)
    values=[]
    for j in [J-.002,J-.001,J,J+.001,J+.002]:
        def f(x):
            q,_,_=critical_mode(j,complex(*x));return [q.real,q.imag]
        q=root(f,[0,omega],tol=1e-9);lam=complex(*q.x);ee,vv,rr=critical_mode(j,lam);assert abs(ee)<1e-8
        values.append(dict(J_EE_core=j,lambda_per_ms=lam,rates_hz=s.regional_rates(rr)))
    energy=weight*abs(v)**2;energy/=energy.sum();region=s.geo['group_region']
    result=dict(J_EE_core=J,frequency_hz=omega*1000/(2*np.pi),rates_hz=s.regional_rates(r),
        equilibrium_residual=float(abs(s.residual(r,J)).max()),characteristic_residual=float(np.linalg.norm(characteristic(s,r,J,1j*omega)@v)),
        regional_energy=[float(energy[s.E&(region==k)].sum()) for k in range(3)],inhibitory_energy=float(energy[~s.E].sum()),
        nearby=values,grid=args.grid,rate_groups=s.P,
        model='Spatial diffusion mean-field with Eq36 dynamic response at shifted bounds and DC-matched gains',response=args.response,
        meaning='Imaginary-axis crossing of selected spatial mode; Hopf criticality (super/subcritical) not yet determined',calls=calls)
    write(dest/'result.json',result);np.savez_compressed(dest/'critical.npz',rates=r,J=J,omega=omega,vector=v)
    print({k:v for k,v in result.items() if k!='calls'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--core',choices=['A','B'],required=True)
    p.add_argument('--start',type=float,default=.95);p.add_argument('--response',default='shifted_white',choices=['shifted_white','calibrated','calibrated_full']);main(p.parse_args())

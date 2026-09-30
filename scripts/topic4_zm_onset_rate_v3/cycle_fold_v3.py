"""Fold of periodic orbits (LPC) refinement for the v3 model: bisection on dD/d(coordinate) along the cycle
branch with a core-mean (or global-mean) coordinate, using the phase-fixed bordered BVP (arclength row)."""
from periodic_v3 import *
from scipy.optimize import brentq
import argparse
def main(a):
    from cupyx.scipy.sparse.linalg import gmres
    s=load_model(a.device);o=PeriodicV3(s,a.N,a.device);cp=o.cp
    mask=s.E&(s.geo['group_region']==('AB'.index(a.core) if a.core in 'AB' else slice(None))) if a.core in 'AB' else s.E
    ww=s.sizes*mask;ww=ww/ww.sum();c=np.r_[np.tile(ww/a.N,a.N),0.,0.];cache=[]
    for path in [a.first,a.second]:
        z=np.load(path);r=resample(z['r'],a.N,axis=0);cache.append((float(r.mean(0)@ww*1000),r,float(z['T']),float(z['D'])))
    def solve(coord):
        lower=[q for q in cache if q[0]<=coord];upper=[q for q in cache if q[0]>=coord];_,r,T,D=min(cache,key=lambda q:abs(q[0]-coord))
        if lower and upper:
            L=max(lower,key=lambda q:q[0]);R=min(upper,key=lambda q:q[0])
            if R[0]>L[0]+1e-10:t=(coord-L[0])/(R[0]-L[0]);r=L[1]*(1-t)+R[1]*t;T=L[2]*(1-t)+R[2]*t;D=L[3]*(1-t)+R[3]*t
        pred=np.r_[(r*1000).ravel(),np.log(T),D*1000];pred+=c*(coord-c@pred)/(c@c);r=pred[:-2].reshape(a.N,s.P)/1000
        arc=(pred,c,np.ones_like(c));sol=o.solve(r,T,D,arc=arc,maxiter=24,tol=1e-9);assert sol['residual']<2e-8,sol['residual']
        y=cp.asarray(sol['y']);ref=cp.asarray(sol['r']);dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(ref,axis=0),n=a.N,axis=0);phase=dr/cp.sum(dr*dr)*RS
        F,A,_=o.evaluate(y,ref,phase,sol['D'],derivative=True,arc=tuple(cp.asarray(x) for x in arc))
        rhs=cp.zeros_like(y);rhs[-1]=1;dy,info=gmres(A,rhs,tol=1e-9,atol=1e-12,restart=120,maxiter=1500);err=float(cp.linalg.norm(A@dy-rhs));assert err<1e-6,err
        dJ=float(dy[-1]/1000);cache.append((coord,sol['r'],sol['T'],sol['D']));log('MEAN FOLD coord %.6f D %.9f T %.4f dD/dcoord %.3e'%(coord,sol['D'],sol['T'],dJ));return dJ,sol,dy.get()
    ca,cb=sorted([cache[0][0],cache[1][0]]);coord=brentq(lambda x:solve(x)[0],ca,cb,xtol=2e-10,rtol=2e-13)
    dJ,sol,tangent=solve(coord);h=min(.005,(cb-ca)*.002);curv=(solve(coord+h)[0]-solve(coord-h)[0])/(2*h)
    path=save_orbit(s,sol,f'{a.label}_N{a.N}');np.savez_compressed(PERIODIC_OUT/f'{a.label}_tangent_N{a.N}.npz',tangent=tangent)
    row=dict(label=a.label,D=sol['D'],T_ms=sol['T'],N=a.N,orbit=str(path),coordinate=f'{a.core}_mean_Hz',coordinate_value=coord,dD_dcoordinate=dJ,d2D_dcoordinate2=curv,residual_hz=sol['residual'],type='fold of periodic orbits',check='phase-fixed bordered BVP null tangent; nonzero parameter curvature')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row);log('CYCLE FOLD',row)
    for side,delta in [('before',-.02),('after',.02)]:
        dd,sl,_=solve(coord+delta);save_orbit(s,sl,f'{a.label}_{side}_N{a.N}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',required=True);p.add_argument('--N',type=int,default=128);p.add_argument('--core',default='G');p.add_argument('--device',type=int,default=0);main(p.parse_args())

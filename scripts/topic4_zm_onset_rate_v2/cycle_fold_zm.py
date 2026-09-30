"""Periodic fold refinement using a core mean as the local branch coordinate.

This remains regular when period and J both turn together (weakly coupled
core cycles). It uses the same phase-fixed bordered periodic BVP.
"""
from periodic_zm import *
from scipy.optimize import brentq


def main():
    from cupyx.scipy.sparse.linalg import gmres
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',required=True)
    p.add_argument('--N',type=int,default=256);p.add_argument('--core',choices=['A','B'],default='B');p.add_argument('--device',type=int,default=0)
    p.add_argument('--radius',type=float,help='Local mean-coordinate bracket about an already refined first orbit')
    a=p.parse_args();s=ZMSpatialRate();o=ZMPeriodic(s,a.N,a.device);cp=o.cp;core='AB'.index(a.core)
    mask=s.E&(s.geo['group_region']==core);ww=s.geo['group_size']*mask;ww=ww/ww.sum();c=np.r_[np.tile(ww/a.N,a.N),0.,0.]
    cache=[]
    for path in [a.first,a.second]:
        z=np.load(path);r=resample(z['r'],a.N,axis=0);coord=float(r.mean(0)@ww*1000);cache.append((coord,r,float(z['T']),float(z['D'])))
    if a.radius:
        coord,r,T,J=cache[0];cache=[(coord+d,r+mask[None,:]*d/1000,T,J) for d in [-a.radius,a.radius]]
    def solve(coord):
        _,r,T,J=min(cache,key=lambda q:abs(q[0]-coord))
        lower=[q for q in cache if q[0]<=coord];upper=[q for q in cache if q[0]>=coord]
        if lower and upper:
            left=max(lower,key=lambda q:q[0]);right=min(upper,key=lambda q:q[0])
            if right[0]>left[0]+1e-10:
                t=(coord-left[0])/(right[0]-left[0]);r=left[1]*(1-t)+right[1]*t;T=left[2]*(1-t)+right[2]*t;J=left[3]*(1-t)+right[3]*t
        pred=np.r_[(r*1000).ravel(),np.log(T),J*1000]
        pred+=c*(coord-c@pred)/(c@c);r=pred[:-2].reshape(a.N,s.P)/1000
        arc=(pred,c,np.ones_like(c));r,T,J,err,history=o.solve(r,T,J,arc=arc,maxiter=24,tol=1e-9);assert err<2e-8
        y=cp.asarray(np.r_[(r*1000).ravel(),np.log(T),J*1000]);ref=cp.asarray(r)
        dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(ref,axis=0),n=a.N,axis=0);phase=dr/cp.sum(dr*dr)*.001
        F,A,_=o.evaluate(y,ref,phase,J,derivative=True,arc=tuple(cp.asarray(x) for x in arc))
        rhs=cp.zeros_like(y);rhs[-1]=1;dy,info=gmres(A,rhs,tol=1e-9,atol=1e-12,restart=160 if a.N>=512 else 100,maxiter=1800 if a.N>=512 else 700)
        error=float(cp.linalg.norm(A@dy-rhs));assert error<1e-6
        dJ=float(dy[-1]/1000);cache.append((coord,r,T,J));print('MEAN FOLD',coord,J,T,dJ,flush=True)
        return dJ,r,T,J,dy.get(),err,history
    ca,cb=sorted([cache[0][0],cache[1][0]]);coord=brentq(lambda x:solve(x)[0],ca,cb,xtol=2e-10,rtol=2e-13)
    dJ,r,T,J,tangent,err,history=solve(coord);h=min(.005,(cb-ca)*.002);curv=(solve(coord+h)[0]-solve(coord-h)[0])/(2*h)
    path=save_orbit(s,r,T,J,err,history,f'{a.label}_N{a.N}');np.savez_compressed(PERIODIC_OUT/f'{a.label}_tangent_N{a.N}.npz',tangent=tangent)
    row=dict(label=a.label,D=J,J_EE_core=1.,T_ms=T,N=a.N,orbit=str(path),coordinate=f'core_{a.core}_mean_Hz',coordinate_value=coord,
        dD_dcoordinate=dJ,d2D_dcoordinate2=curv,residual_hz=err,type='fold of periodic orbits',
        check='phase-fixed bordered BVP null tangent; nonzero parameter curvature')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row);print('CYCLE FOLD',row,flush=True)
    for side,delta in [('before',-.02),('after',.02)]:
        dd,rr,tt,jj,tan,ee,hh=solve(coord+delta)
        save_orbit(s,rr,tt,jj,ee,hh,f'{a.label}_{side}_N{a.N}')


if __name__=='__main__':main()

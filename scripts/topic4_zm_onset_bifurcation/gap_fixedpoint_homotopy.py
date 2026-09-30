"""Track r = h Phi_D(r) + (1-h) r0 to the ORIGINAL fixed-D equation.

This is an algebraic root solver. h is never a biological parameter and all
h != 1 points are excluded from the bifurcation diagram and scientific states.
"""
from continue_D import *


def main(a):
    s=ZMRate();s.set_D(a.D);base=DEST/'g20';dest=base/a.label;dest.mkdir(exist_ok=False)
    pool=[]
    for name in ['D_gap_lower_guarded','D_gap_rate_slices']:
        for row in read(base/name/'result.json')['rows']:
            if row.get('converged',True):pool.append((row['global_E_hz'],base/name/f'point{row["index"]:04d}.npz'))
    lo=min(pool,key=lambda p:abs(p[0]-100));hi=min(pool,key=lambda p:abs(p[0]-172))
    with np.load(lo[1]) as z:rl=z['r'].copy()
    with np.load(hi[1]) as z:rh=z['r'].copy()
    seed=(rl+rh)/2;seed=np.maximum(seed,1e-12);eye=sparse.eye(s.P,format='csr')
    def residual(x):
        r=x[:-1]*RS;h=x[-1]
        return h*s.residual(r,a.D)+(1-h)*(seed-r)
    def jac(x):
        r=x[:-1]*RS;h=x[-1]
        J=(h*s.jacobian(r,a.D)-(1-h)*eye)*RS
        dh=s.residual(r,a.D)+r-seed
        return sparse.hstack([J,sparse.csr_matrix(dh[:,None])],format='csr')
    def tan(x,prev=None):
        A=jac(x)
        if prev is None:v=np.r_[spsolve(A[:,:-1].tocsc(),-A[:,-1].toarray().ravel()),1.]
        else:v=spsolve(sparse.vstack([A,sparse.csr_matrix(prev[None,:])]).tocsc(),np.r_[np.zeros(s.P),1.])
        v/=np.linalg.norm(v)
        if prev is not None and v@prev<0:v=-v
        return v
    x=np.r_[seed/RS,0.];t=tan(x);ds=.1;trace=[];success=False
    for k in range(a.steps):
        trace.append(dict(step=k,h=float(x[-1]),rate_hz=s.global_rate(x[:-1]*RS),ds=ds))
        if k%20==0:print(trace[-1],flush=True)
        if x[-1]>=.999:
            rr=np.clip(x[:-1]*RS,0,np.nextafter(1/s.ref,0));r,success,_=s.solve_D(a.D,rr)
            if success:break
        for attempt in range(22):
            pred=x+ds*t;xx=pred.copy();ok=False
            for it in range(22):
                if not -.001<xx[-1]<1.05:break
                f=np.r_[residual(xx),RS*t@(xx-pred)]
                if max(abs(f))<2e-11:ok=True;break
                B=sparse.vstack([jac(xx),sparse.csr_matrix((t*RS)[None,:])]).tocsc();dx=spsolve(B,-f)
                alpha=1.
                for back in range(22):
                    trial=xx+alpha*dx
                    if -.001<trial[-1]<1.05:
                        trial[:-1]=np.clip(trial[:-1],0,np.nextafter(1/s.ref/RS,0))
                        ff=np.r_[residual(trial),RS*t@(trial-pred)]
                        if np.all(np.isfinite(ff)) and np.linalg.norm(ff)<np.linalg.norm(f):xx=trial;break
                    alpha*=.5
                else:break
            if ok:
                tt=tan(xx,t);cos=float(tt@t);corr=float(np.linalg.norm(xx-pred)/ds)
                if cos>.94 and corr<.18:break
                ok=False
            ds*=.5
        if not ok:break
        x=xx;t=tt
        if it<5 and cos>.98 and corr<.1:ds=min(ds*1.2,6.)
    row=dict(index=0,D=a.D,converged=success,solver_only_homotopy=True,intermediate_states_excluded=True)
    if success:
        err=float(abs(s.residual(r,a.D)).max()*1000)
        assert err<2e-8 and r.min()>=0 and np.all(r<1/s.ref)
        row.update(global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),equilibrium_residual_hz=err)
        np.savez_compressed(dest/'point0000.npz',r=r,D=a.D,Z=s.Z,tangent=tangent(s,r,a.D))
    write(dest/'result.json',dict(status='COMPLETE',rows=[row],solver_trace=trace,
        scope='Only h=1 roots of the original fixed model are admissible scientific states'))
    print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--D',type=float,default=.205)
    p.add_argument('--label',default='D_gap_fixedpoint_homotopy_0p205');p.add_argument('--steps',type=int,default=1600)
    main(p.parse_args())

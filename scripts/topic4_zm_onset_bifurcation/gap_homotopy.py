"""Residual homotopy for a missing equilibrium, not a biological parameter scan.

Only h=1 states satisfying the original rate equations are accepted or plotted.
All h<1 states are solver intermediates and are explicitly excluded from data.
"""
from bridge_rate_slices import *


def main(a):
    s=ZMRate();base=DEST/'g20';dest=base/a.label;dest.mkdir(exist_ok=False)
    pool=[]
    for name in ['D_gap_lower_guarded','D_gap_rate_slices','D_gap_upper_v1']:
        for row in read(base/name/'result.json')['rows']:
            if row.get('converged',True):pool.append((row['global_E_hz'],row['D'],base/name/f'point{row["index"]:04d}.npz'))
    low=max((p for p in pool if p[0]<a.target),key=lambda q:q[0])
    high=min((p for p in pool if p[0]>a.target),key=lambda q:q[0])
    with np.load(low[2]) as z:rl=z['r'].copy()
    with np.load(high[2]) as z:rh=z['r'].copy()
    u=(a.target-low[0])/(high[0]-low[0]);r=(1-u)*rl+u*rh;D=(1-u)*low[1]+u*high[1]
    anchor=s.residual(r,D).copy();w=np.zeros(s.P);w[s.E]=s.mean_weights
    def residual(x):
        r=x[:-2]*RS;D=x[-2]*DS;h=x[-1]
        return np.r_[s.residual(r,D)-(1-h)*anchor,10*(w@r-a.target/1000)]
    def jac(x):
        r=x[:-2]*RS;D=x[-2]*DS
        return sparse.bmat([[s.jacobian(r,D)*RS,sparse.csr_matrix((s.parameter_derivative(r,D)*DS)[:,None]),sparse.csr_matrix(anchor[:,None])],
            [sparse.csr_matrix((10*w*RS)[None,:]),sparse.csr_matrix((1,1)),sparse.csr_matrix((1,1))]],format='csr')
    x=np.r_[r/RS,D/DS,0.]
    def tan(x,previous=None):
        A=jac(x)
        if previous is None:v=np.r_[spsolve(A[:,:-1].tocsc(),-np.r_[anchor,0.]),1.]
        else:v=spsolve(sparse.vstack([A,sparse.csr_matrix(previous[None,:])]).tocsc(),np.r_[np.zeros(s.P+1),1.])
        v/=np.linalg.norm(v)
        if previous is not None and v@previous<0:v=-v
        return v
    t=tan(x);ds=.1;trace=[];success=False
    for k in range(a.steps):
        trace.append(dict(step=k,h=float(x[-1]),D=float(x[-2]*DS),ds=ds))
        if k%20==0:print(trace[-1],flush=True)
        if x[-1]>=1:
            r,D,success,polish=solve_rate(s,a.target,x[:-2]*RS,float(x[-2]*DS),unbounded_trials=True)
            if success:break
        for attempt in range(22):
            pred=x+ds*t;xx=pred.copy();ok=False
            for it in range(22):
                if not 0<xx[-2]*DS<1:break
                f=np.r_[residual(xx),RS*t@(xx-pred)]
                if max(abs(f))<2e-11:ok=True;break
                B=sparse.vstack([jac(xx),sparse.csr_matrix((t*RS)[None,:])]).tocsc()
                dx=spsolve(B,-f);alpha=1.
                for back in range(24):
                    trial=xx+alpha*dx
                    if 0<trial[-2]*DS<1 and np.max(abs(trial[:-2]*RS))<3 and abs(trial[-1])<3:
                        ff=np.r_[residual(trial),RS*t@(trial-pred)]
                        if np.all(np.isfinite(ff)) and np.linalg.norm(ff)<np.linalg.norm(f):xx=trial;break
                    alpha*=.5
                else:break
            if ok:
                tt=tan(xx,t);cos=float(tt@t);corr=float(np.linalg.norm(xx-pred)/ds)
                if cos>.93 and corr<.2:break
                ok=False
            ds*=.5
        if not ok:break
        x=xx;t=tt
        if it<5 and cos>.98 and corr<.1:ds=min(ds*1.2,8.)
    row=dict(index=0,target_hz=a.target,converged=success,
        solver_only_homotopy=True,intermediate_states_excluded=True)
    if success:
        row.update(D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
            equilibrium_residual_hz=float(abs(s.residual(r,D)).max()*1000))
        np.savez_compressed(dest/'point0000.npz',r=r,D=D,Z=s.Z,tangent=tangent(s,r,D))
    write(dest/'result.json',dict(status='COMPLETE',rows=[row],homotopy_trace=trace,
        scope='Only original residual zero at h=1 may be used as an equilibrium'))
    print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=float,default=140)
    p.add_argument('--label',default='D_gap_homotopy_140');p.add_argument('--steps',type=int,default=600)
    main(p.parse_args())

"""Bound-constrained equilibrium searches in the remaining mean-rate gap."""
from bridge_rate_slices import *
from scipy.optimize import least_squares
import time


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
    t=(a.target-low[0])/(high[0]-low[0]);r0=(1-t)*rl+t*rh;D0=(1-t)*low[1]+t*high[1]
    w=np.zeros(s.P);w[s.E]=s.mean_weights
    calls=0;began=time.time()
    def fun(x):
        nonlocal calls
        calls+=1;r=x[:-1]*RS;D=x[-1]*DS
        f=np.r_[s.residual(r,D),10*(w@r-a.target/1000)]
        if calls%50==0:print('evaluation',calls,'max residual',max(abs(f)),'cost',sum(f*f),'seconds',time.time()-began,flush=True)
        return f
    def jac(x):
        r=x[:-1]*RS;D=x[-1]*DS
        return sparse.bmat([[s.jacobian(r,D)*RS,sparse.csr_matrix((s.parameter_derivative(r,D)*DS)[:,None])],
            [sparse.csr_matrix((10*w*RS)[None,:]),sparse.csr_matrix((1,1))]],format='csr')
    x0=np.r_[np.maximum(r0/RS,1e-13),D0/DS]
    upper=np.r_[1/s.ref/RS,1/DS]
    result=least_squares(fun,x0,jac=jac,bounds=(np.zeros(s.P+1),upper),
        ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=a.evaluations,
        tr_solver='lsmr',tr_options={'atol':1e-11,'btol':1e-11,'maxiter':2500})
    r=result.x[:-1]*RS;D=float(result.x[-1]*DS)
    # Polish the same exact equations after a basin of convergence is found.
    r,D,polished,trace=solve_rate(s,a.target,r,D,project_trials=True)
    residual=float(abs(s.residual(r,D)).max()*1000);rate=s.global_rate(r)
    good=bool(residual<2e-8 and abs(rate-a.target)<2e-8)
    row=dict(index=0,target_hz=a.target,D=D,global_E_hz=rate,converged=good,
        equilibrium_residual_hz=residual,rate_error_hz=rate-a.target,
        least_squares_status=result.message,evaluations=result.nfev,optimality=result.optimality,
        wall_s=time.time()-began,polished=polished,source_states=[str(low[2]),str(high[2])])
    write(dest/'result.json',dict(status='COMPLETE',rows=[row],connection='Independent equilibrium root; no branch connection asserted'))
    if good:np.savez_compressed(dest/'point0000.npz',r=r,D=D,Z=s.Z,tangent=tangent(s,r,D))
    print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=float,required=True)
    p.add_argument('--evaluations',type=int,default=500);p.add_argument('--label',required=True)
    main(p.parse_args())

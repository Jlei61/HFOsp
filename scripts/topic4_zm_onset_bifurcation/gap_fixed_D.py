"""Independent fixed-D root searches; no forced global rate or model changes."""
from continue_D import *
from scipy.optimize import root
import time


def main(a):
    s=ZMRate();base=DEST/'g20';dest=base/a.label;dest.mkdir(exist_ok=False)
    pool=[]
    for name in ['D_gap_lower_guarded','D_gap_rate_slices']:
        for row in read(base/name/'result.json')['rows']:
            if row.get('converged',True):pool.append((row['global_E_hz'],row['D'],base/name/f'point{row["index"]:04d}.npz'))
    low=min(pool,key=lambda p:abs(p[0]-100));high=min(pool,key=lambda p:abs(p[0]-172))
    with np.load(low[2]) as z:rl=z['r'].copy()
    with np.load(high[2]) as z:rh=z['r'].copy()
    rows=[];began=time.time()
    for method in ['hybr','krylov','anderson']:
        for seedname,rr in [('low',rl),('high',rh),('blend',(rl+rh)/2)]:
            def fun(x):return s.residual(x*RS,a.D)/RS
            def jac(x):return s.jacobian(x*RS,a.D).toarray()
            options={'xtol':1e-10,'maxfev':500,'factor':.2} if method=='hybr' else {'fatol':1e-9,'maxiter':120}
            try:res=root(fun,rr/RS,jac=jac if method=='hybr' else None,method=method,options=options)
            except (ValueError,OverflowError,ZeroDivisionError) as exc:
                row=dict(index=len(rows),method=method,seed=seedname,D=a.D,converged=False,error=str(exc))
            else:
                r=np.clip(res.x*RS,0,np.nextafter(1/s.ref,0));err=float(abs(s.residual(r,a.D)).max()*1000)
                row=dict(index=len(rows),method=method,seed=seedname,D=a.D,converged=err<2e-8,
                    global_E_hz=s.global_rate(r),equilibrium_residual_hz=err,evaluations=res.nfev,
                    message=res.message,wall_s=time.time()-began)
                if row['converged']:
                    np.savez_compressed(dest/f'point{len(rows):04d}.npz',r=r,D=a.D,Z=s.Z,tangent=tangent(s,r,a.D))
            rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows));print(row,flush=True)
            if row['converged']:break
        if rows[-1]['converged']:break
    write(dest/'result.json',dict(status='COMPLETE',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--D',type=float,default=.205)
    p.add_argument('--label',default='D_gap_fixed_0p205');main(p.parse_args())

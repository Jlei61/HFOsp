"""Newton search without positivity of intermediate iterates; only physical roots are retained."""
from native_path import *
from scipy.sparse.linalg import spsolve


def solve(s,r,maxiter=100):
    r=np.asarray(r,float).copy();trace=[]
    for i in range(maxiter):
        f=s.residual(r);err=float(abs(f).max());trace.append(err)
        if err<1e-10:
            physical=bool(np.min(r)>=-1e-9 and np.max(r*s.ref)<=1+1e-9)
            return r,physical,trace
        dr=spsolve(s.jacobian(r),-f)
        for j in range(25):
            rr=r+2.**-j*dr
            ff=s.residual(rr)
            if np.isfinite(ff).all() and np.linalg.norm(ff)<np.linalg.norm(f):
                r=rr;break
        else:return r,False,trace
    return r,False,trace


def main():
    s=model();path=attach_native_path(s);out=OUT/'equilibria/native_unconstrained';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for t,Z in zip(path['times_ms'][::-1],path['fields'][::-1]):
        z=np.load(BASE/f'stage_b/runs/c_relaxed_native_{t}.npz');s.set_Z(Z)
        candidates=[('tail_average',z['group_rate_hz'][-2000:].mean(0)/1000)]
        for folder in ['upper_cont','lower_cont']:
            info=read(BASE/f'equilibria/{folder}/result.json');q=min(info['rows'],key=lambda x:abs(x['D']-s.D))
            candidates.append((folder,np.load(BASE/f'equilibria/{folder}/point{q["index"]:04d}.npz')['r']))
        for name,r0 in candidates:
            r,ok,tr=solve(s,r0);q=dict(time_source_ms=t,D=s.D,seed=name,converged_physical=ok,trace=tr,
                global_E_hz=s.global_rate(r),minimum_rate=float(r.min()),max_fraction_refractory=float(max(r*s.ref)))
            if ok:
                np.savez_compressed(out/f't{t}_{name}.npz',r=r,Z=s.Z,D=s.D,state=s.equilibrium_state(r))
            rows.append(q);write(out/'result.json',rows);log(t,name,ok,tr[-1],s.global_rate(r))


if __name__=='__main__':main()

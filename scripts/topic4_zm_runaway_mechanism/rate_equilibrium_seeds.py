"""Equilibrium roots on the same local rate-Z path as the onset cycles."""
from equilibrium_unconstrained import *


def main():
    s=model();path=attach_rate_entry_path(s);dest=OUT/'equilibria/rate_path_seeds';dest.mkdir(parents=True,exist_ok=True)
    periodic=np.load(OUT/'periodic/rate_seed_N1024.npz')['r'].mean(0)
    high=np.load(OUT/'equilibria/native_unconstrained/t9870_tail_average.npz')['r']
    seeds=[('low',np.zeros(s.P)),('cycle_mean',periodic),('high',high)]
    rows=[]
    for D in [float(path['D'][0]),.14494,float(path['D'][1])]:
        s.set_D(D);roots=[]
        for name,initial in seeds:
            r,ok,tr=solve(s,initial,maxiter=60)
            duplicate=next((j for j,q in enumerate(roots) if abs(q-r).max()<1e-7),None) if ok else None
            row=dict(D=D,seed=name,converged_physical=ok,residual_per_ms=tr[-1],iterations=len(tr),
                     global_E_hz=s.global_rate(r),duplicate_of=duplicate,stability='NOT_COMPUTED')
            if ok and duplicate is None:
                roots.append(r);file=dest/f'D{D:.9f}_{name}.npz'
                np.savez_compressed(file,r=r,D=D,Z=s.Z,state=s.equilibrium_state(r));row['path']=str(file)
                f,rr=s.rhs(s.equilibrium_state(r),[A@r for A in s.matrices()],dynamic_z=False)
                row['full_rhs_max']=float(abs(f).max());assert abs(f).max()<1e-7
            rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows));log('RATE EQ',row)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,claim='Roots on the local spatial Z path; multistart does not prove branch completeness'))


if __name__=='__main__':main()

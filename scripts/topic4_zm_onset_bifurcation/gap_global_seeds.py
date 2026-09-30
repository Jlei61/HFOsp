"""Broader initial guesses for the unresolved stationary-rate interval."""
from bridge_rate_slices import *


def main(a):
    s=ZMRate();base=DEST/'g20';dest=base/a.label;dest.mkdir(exist_ok=False)
    pool=[]
    for name in ['D_gap_lower_guarded','D_gap_rate_slices','D_gap_upper_v1']:
        for row in read(base/name/'result.json')['rows']:
            if row.get('converged',True):pool.append((row['global_E_hz'],row['D'],base/name/f'point{row["index"]:04d}.npz'))
    lo=max((p for p in pool if p[0]<a.target),key=lambda q:q[0]);hi=min((p for p in pool if p[0]>a.target),key=lambda q:q[0])
    with np.load(lo[2]) as z:rl=z['r'].copy()
    with np.load(hi[2]) as z:rh=z['r'].copy()
    u=(a.target-lo[0])/(hi[0]-lo[0]);blend=(1-u)*rl+u*rh
    uniform=blend.copy();uniform[s.E]=a.target/1000;uniform[~s.E]=np.average(blend[~s.E],weights=s.sizes[~s.E])
    scaled_low=rl.copy();scaled_low[s.E]*=a.target/lo[0]
    scaled_high=rh.copy();scaled_high[s.E]*=a.target/hi[0]
    rows=[]
    for name,seed in [('uniform',uniform),('low_scaled',scaled_low),('high_scaled',scaled_high),('blend',blend)]:
        for D0 in [.1,.3,.2,.4,.05,.6]:
            r,D,ok,trace=solve_rate(s,a.target,seed,D0,unbounded_trials=True)
            row=dict(index=len(rows),seed=name,initial_D=D0,target_hz=a.target,converged=ok,
                final_solver_residual=float(trace[-1]),iterations=len(trace))
            if ok:
                row.update(D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
                    equilibrium_residual_hz=float(abs(s.residual(r,D)).max()*1000))
                np.savez_compressed(dest/f'point{len(rows):04d}.npz',r=r,D=D,Z=s.Z,tangent=tangent(s,r,D))
            rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows))
            print(row,flush=True)
            if ok:break
        if ok:break
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,
        scope='Original equations and final physical domain; only solver initial guesses vary'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=float,default=140)
    p.add_argument('--label',default='D_gap_global_seed_140');main(p.parse_args())

"""Search the missing stationary-rate range from spatial front initial guesses.

Only the nonlinear solver initial guess changes. All accepted states solve the
same frozen model, with physical Z and rates and dynamic-M equilibrium.
"""
from bridge_rate_slices import *
from scipy.special import expit
from scipy.optimize import brentq


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
    t=(a.target-low[0])/(high[0]-low[0]);D0=(1-t)*low[1]+t*high[1]
    pos=s.geo['positions'];rows=[]
    for angle in [0,45,90,135,180,225,270,315]:
        q=pos@np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        for width in [.5,2.]:
            def guess(bound):return rl+expit((q-bound)/width)*(rh-rl)
            bound=brentq(lambda b:s.global_rate(guess(b))-a.target,q.min()-40,q.max()+40)
            seed=guess(bound)
            r,D,ok,trace=solve_rate(s,a.target,seed,D0,unbounded_trials=True)
            row=dict(index=len(rows),angle=angle,width_mm=width,target_hz=a.target,converged=ok,
                final_solver_residual=float(trace[-1]),iterations=len(trace))
            if ok:
                row.update(D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
                    equilibrium_residual_hz=float(abs(s.residual(r,D)).max()*1000))
                np.savez_compressed(dest/f'point{len(rows):04d}.npz',r=r,D=D,Z=s.Z,tangent=tangent(s,r,D))
            rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows,
                connection='Independent equilibrium root from spatial initial guesses; no connection asserted'))
            print(row,flush=True)
            if ok:break
        if ok:break
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,
        connection='Independent equilibrium root from spatial initial guesses; no connection asserted'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=float,default=140)
    p.add_argument('--label',default='D_gap_spatial_seed_140');main(p.parse_args())

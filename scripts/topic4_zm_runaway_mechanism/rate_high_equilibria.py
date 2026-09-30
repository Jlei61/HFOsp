"""Search a high-rate equilibrium seed on the same physical local Z family.

This is an auxiliary equilibrium search, never a classification of the burst
attractor. Physical-domain checks are supplied by attach_rate_entry_path.
"""
from equilibrium_unconstrained import *


def main():
    s=model();attach_rate_entry_path(s)
    dest=OUT/'equilibria/rate_high_search';dest.mkdir(parents=True,exist_ok=True)
    rows=[];last=None
    for D in [.38,.36,.32,.28,.24,.20,.17,.15,.14769197,.14502,.14496]:
        s.set_D(D)
        seeds=[('previous',last)] if last is not None else []
        if last is None:seeds+=[('saturated',.99/s.ref),('intermediate',.25/s.ref)]
        for name,r0 in seeds:
            r,ok,tr=solve(s,r0,maxiter=80)
            q=dict(D=D,seed=name,converged_physical=ok,residual_per_ms=tr[-1],iterations=len(tr),
                   global_E_hz=s.global_rate(r),stability='NOT_COMPUTED')
            if ok:
                p=dest/f'D{D:.9f}.npz';np.savez_compressed(p,r=r,D=D,Z=s.Z,state=s.equilibrium_state(r))
                q['path']=str(p);last=r
            rows.append(q);write(dest/'result.json',dict(status='RUNNING',rows=rows));log('RATE HIGH EQ',q)
            if ok:break
        if not ok:last=None
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,claim='Auxiliary roots; no completeness or stability inferred'))


if __name__=='__main__':main()

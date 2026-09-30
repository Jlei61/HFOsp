"""Finite-time attraction map, kept distinct from continued bifurcation curves."""
from common import *
from integrate import simulate,describe,constant_state
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time


def worker(h,direction):
    s=System(h);folder=OUT/'state_scan_v2'/f'h{h:.5f}'/direction;folder.mkdir(parents=True,exist_ok=True)
    grid=np.unique(np.r_[[.5,.7,.9,1.,1.05,1.10,1.12,1.14,1.16,1.17],np.arange(1.18,1.6001,.02)])
    if direction=='down':grid=grid[::-1]
    state=None;rows=[];last_kind=None
    start_r=np.array([.3,.3,.01,.3,.3,.01]) if direction=='down' else np.array([.0002,.0002,0,0,0,0])
    for g in grid:
        dest=folder/('g'+f'{g:.5f}'.replace('.','p'))
        if dest.with_suffix('.json').exists() and dest.with_suffix('.npz').exists():
            row=read(dest.with_suffix('.json'));z=np.load(dest.with_suffix('.npz'))
            assert abs(float(z['g'])-g)<1e-10 and abs(float(z['h'])-h)<1e-10
            assert abs(row['g']-g)<1e-10 and row['direction']==direction
            state=(z['state'],z['history'],int(z['head']));rows.append(row);last_kind=row['kind'];continue
        start=time.monotonic()
        # Far below either known fold, solve and carry the low-rate equilibrium.
        if g<=1.1 and (direction=='up' or last_kind=='equilibrium_candidate'):
            r,err,ok=s.solve(g,np.array([.0002,.0002,0,0,0,0]))
            if not ok:raise RuntimeError(('low equilibrium',h,g,err))
            state=constant_state(s,r,.1);rate=np.tile(r,(12000,1));duration=0
        else:
            duration=6000 if state is None or last_kind=='unresolved' else 3000
            rate,state=simulate(s,g,duration_ms=duration,dt=.1,state=state,r0=start_r)
        row=describe(rate)
        if row['kind']=='unresolved':
            extra,state=simulate(s,g,duration_ms=6000,dt=.1,state=state)
            rate=np.r_[rate,extra];duration+=6000;row=describe(rate)
        if row['kind']=='equilibrium_candidate':
            rr,err,ok=s.solve(g,np.array(row['mean_hz'])/1000)
            row.update(equilibrium_residual=err,equilibrium_solve=bool(ok))
        row.update(h=float(h),g=float(g),direction=direction,dt_ms=.1,save_dt_ms=.5,duration_ms=duration,
                   elapsed_s=time.monotonic()-start,source=str(dest.with_suffix('.npz').relative_to(ROOT)))
        np.savez_compressed(dest.with_suffix('.npz'),r=rate[-12000:],state=state[0],history=state[1],head=state[2],g=g,h=h,sample_dt=.5)
        write(dest.with_suffix('.json'),row);rows.append(row);last_kind=row['kind']
        write(folder/'summary.json',rows)
        print('CELL',h,direction,g,row['kind'],row.get('pattern'),round(row['elapsed_s'],1),flush=True)
    return dict(h=h,direction=direction,cells=len(rows),unresolved=sum(r['kind']=='unresolved' for r in rows))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    levels=[0,.125,.25,.375,.5,.625,.75,.875,.95,.975,1.]
    jobs=[(h,d) for h in levels for d in ('up','down')]
    completed=[];failed=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,*job):job for job in jobs}
        for f in as_completed(futures):
            try:completed.append(f.result())
            except Exception as exc:failed.append(dict(job=futures[f],error=repr(exc)))
            write('state_scan_v2/status.json',dict(total_jobs=len(jobs),completed=completed,failed=failed,status='RUNNING'))
    write('state_scan_v2/status.json',dict(total_jobs=len(jobs),completed=completed,failed=failed,status='COMPLETED' if not failed else 'ERRORS'))

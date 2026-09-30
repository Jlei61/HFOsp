"""Refine only the sampling of a complete DDE initial history, by linear interpolation."""
from common import *
import argparse


def main(a):
    s=model();z=np.load(a.source);state,h=checkpoint_initial(a.source)
    old=float(z['dt_ms']) if 'dt_ms' in z else float(read(Path(a.source).parent/'contract.json')['dt_ms'])
    assert len(h)==round(s.delays[-1]/old)+1
    assert a.dt<old and abs(old/a.dt-round(old/a.dt))<1e-12
    oldlags=np.arange(len(h))*old;oldrates=h[(-np.arange(len(h)))%len(h)]
    depth=round(s.delays[-1]/a.dt)+1;newlags=np.arange(depth)*a.dt
    refined=np.array([np.interp(newlags,oldlags,oldrates[:,g]) for g in range(s.P)]).T
    factor=round(old/a.dt);assert np.array_equal(refined[::factor],oldrates)
    newhistory=np.empty_like(refined);newhistory[(-np.arange(depth))%depth]=refined
    p=Path(a.output);p.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(p,state=state,history=newhistory,tick=0,dt_ms=a.dt,source=a.source,source_dt_ms=old)
    write(p.with_suffix('.json'),dict(status='PASS',source=a.source,source_dt_ms=old,dt_ms=a.dt,
        method='linear interpolation of the labelled instantaneous rate history; all 14 endpoint states unchanged',
        exact_on_old_nodes=True,minimum_history_rate_per_ms=float(newhistory.min()),
        intended_comparison='Discard at least 2s before the finite-time perturbation estimate; history horizon is 35.8ms.'))
    log('HISTORY REGRID',p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--dt',type=float,required=True);main(p.parse_args())

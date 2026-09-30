"""Compare both numerical methods from one identical physical history.

The saved source flux bins define a piecewise-constant initial rate history.
Subdivision preserves every parent-bin integral, unlike point interpolation.
Outputs are restricted by bin integration to the common original lag grid.
"""
from common import OUT, np, read, write, log
from onset_exponential_midpoint import ExponentialMidpointEngine
from onset_state_continuation import build
from onset_relative_rate_recorder import RelativeRateRecorder
from fine_rate_frozen_Z_fields import capture, restore
from pathlib import Path
import argparse, os, gc, time

DEST=OUT/'core_a_bifurcation_type_20260924/numerical_checks/exponential_midpoint/short_spatial_convergence'
SOURCE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/three_burst_entry_continuation/native9584/whole_cycle_finish/uninterrupted_root_control/final_state.npz'


def conservative_history(source, e, original_dt=.05):
    base={k:v.copy() for k,v in source.items()}; factor=round(original_dt/e.dt)
    assert factor>=1 and abs(factor*e.dt-original_dt)<1e-12
    old=source['history']; tick=int(source['clock'][0]); depth=len(e.local.history)
    canonical=old[(tick-np.arange(len(old)))%len(old)]
    child=canonical[np.arange(depth)//factor].copy(); now=tick*factor
    history=np.empty_like(child); history[(now-np.arange(depth))%depth]=child
    nparent=min(len(old)-1,len(child)//factor)
    restored=child[:nparent*factor].reshape(nparent,factor,e.s.P).mean(1)
    error=float(abs(restored-canonical[:nparent]).max()); assert error<1e-14
    base['history']=history; base['clock']=np.array([now],dtype=source['clock'].dtype)
    qa=dict(factor=factor,parent_integral_error=error,initial_occupancy=[])
    for mask,ref in [(e.s.E,2.),(~e.s.E,1.)]:
        value=child[:round(ref/e.dt),mask].sum(0)*e.dt
        before=canonical[:round(ref/original_dt),mask].sum(0)*original_dt
        assert abs(value-before).max()<1e-12 and value.max()<=1+1e-10
        qa['initial_occupancy'].append(dict(maximum=float(value.max()),mass_error=float(abs(value-before).max())))
    return base, qa


def common_state(state, factor, parent_depth):
    h=state['history']; tick=int(state['clock'][0]); n=(parent_depth-1)*factor
    history=h[(tick-np.arange(n))%len(h)].reshape(parent_depth-1,factor,h.shape[1]).mean(1)
    return np.concatenate([state['syn'][:5],state['local'],history])


def main(device):
    parent=DEST.parent; assert read(parent/'spatial_identity.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True); assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(source=str(SOURCE),source_dt_ms=.05,duration_ms=120,
        methods=['old_endpoint','exponential_midpoint'],steps_ms=[.05,.025,.0125,.00625],
        initial_history='Same piecewise-constant original saved flux-bin history at every step size. Refine by exact conservative subdivision; no smoothing, interpolation of point values, or parameter change. Continuous synaptic/covariance/memory/M states identical.',
        readout='Common1ms integral rates; full synaptic/covariance/memory/M state and lag-bin integrals restricted to the original.05ms grid, omitting only its unused oldest padding bin. Compare both methods at20/60/120ms and adjacent-mesh error ratios before critical or long runs.',
        purpose='Test accuracy order and whether both discretizations approach the same original continuous physical equations. Not a claim that the120ms state or a bifurcation is converged.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'jobs.json',jobs)
    source=dict(np.load(SOURCE)); started=time.time()
    try:
        for method in ['old_endpoint','exponential_midpoint']:
            for dt in [.05,.025,.0125,.00625]:
                label=method+'_dt'+str(dt).replace('.','p');folder=DEST/label;folder.mkdir()
                e=build(device,dt) if method=='old_endpoint' else ExponentialMidpointEngine(dt=dt,device=device)
                if method!='old_endpoint':e.graph()
                base,qa=conservative_history(source,e);restore(e,base);rec=RelativeRateRecorder(e)
                rate=[]
                for k in range(12):
                    r=rec.chunk();assert np.isfinite(r).all() and r.min()>=0 and np.array_equal(r[:,0],r[:,1])
                    rate.append(r[:,0])
                    if (k+1)*10 in [20,60,120]:
                        state=capture(e);assert np.array_equal(state['syn'][5],source['syn'][5])
                        common=common_state(state,qa['factor'],len(source['history']))
                        np.savez_compressed(folder/f'common_state{(k+1)*10}.npz',state=common)
                np.savez_compressed(folder/'rates.npz',group_rate_hz=np.concatenate(rate),dt_ms=dt)
                write(folder/'result.json',dict(status='COMPLETE',dt_ms=dt,method=method,initial_history=qa))
                jobs['completed'].append(label);write(DEST/'jobs.json',jobs);log('SPATIAL MIDPOINT MESH',label)
                cp=e.cp;del rec,e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        jobs.update(status='COMPLETE',seconds=time.time()-started);write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)

"""Full delayed-state nonlinear differences for the matching new derivative."""
from common import OUT,np,read,write,log
from onset_exponential_midpoint import ExponentialMidpointEngine
from onset_midpoint_tangent import MidpointTangent
from fine_rate_frozen_Z_fields import capture,restore
import argparse,os,gc,time

BASE=OUT/'core_a_bifurcation_type_20260924'
DEST=BASE/'numerical_checks/exponential_midpoint/full_variational_check'


def main(device):
    assert read(DEST.parent/'short_spatial_convergence/analysis.json')['status']=='SHORT_WINDOW_ORDER_AND_COMMON_LIMIT_PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(duration_ms=40,dt_ms=[.05,.025],
        method='Differentiate every midpoint predictor/filter/readout, exponential renewal-bin update, full-step filters/M and delayed history insertion. Compare centered nonlinear full-flow differences over40ms, longer than the35.8ms maximum transmission delay.',
        acceptance='All nominal output and persistent/scratch state arrays bitwise identical with/without passive variational equations. Smallest-epsilon centered differences relative error<1e-4 in synaptic/M, covariance/memory and full ring history, at both steps. No clipping of physical perturbation states.',
        scope='Matching numerical derivative only; not a Floquet, spectrum, closed-orbit or bifurcation certificate.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(DEST/'jobs.json',jobs);rows=[];started=time.time()
    try:
        for dt,label in [(.05,'matched_natural_short_entry_step'),(.025,'matched_natural_short_entry_step_fine')]:
            e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
            path=BASE/'reference_stability_gap'/label/'held_short_field/checkpoint1000.npz'
            base=dict(np.load(path));restore(e,base);t=MidpointTangent(e);t.graph()
            e.chunk();expected=capture(e);restore(e,base);t.chunk();actual=capture(e)
            parity={k:bool(np.array_equal(v,actual[k])) for k,v in expected.items()};assert all(parity.values()),parity
            rng=np.random.default_rng(9250810)
            d={k:rng.uniform(-1,1,base[k].shape)*base[k] for k in ['syn','local','history']}
            d['syn'][5]=0;d['syn'][4,~e.s.E]=0
            clock=int(base['clock'][0]);slack=np.ones(e.s.P)
            for mask,ref in [(e.s.E,2.),(~e.s.E,1.)]:
                occupied=base['history'][(clock-np.arange(round(ref/dt)))%len(base['history'])][:,mask].sum(0)*dt
                slack[mask]=np.clip(1-occupied,1e-8,1.)
            d['history']*=slack
            restore(e,base);t.reset();t.syn[:]=e.cp.asarray(d['syn'][:5]);t.local[:]=e.cp.asarray(d['local']);t.history[:]=e.cp.asarray(d['history'])
            e.cp.cuda.get_current_stream().synchronize()
            for _ in range(4):t.chunk()
            derivative=dict(syn=t.syn.get(),local=t.local.get(),history=t.history.get());tests=[]
            for eps in [2e-5,1e-5,5e-6]:
                ends=[]
                for sign in [-1,1]:
                    state=dict(base)
                    for k in d:state[k]=base[k]+sign*eps*d[k]
                    assert state['history'].min()>=0 and state['syn'][:5].min()>=0
                    restore(e,state)
                    for _ in range(4):
                        output=e.chunk();assert np.isfinite(output).all() and output.min()>=0
                    ends.append(capture(e))
                errors={}
                for k,exact in derivative.items():
                    fd=(ends[1][k]-ends[0][k])/(2*eps)
                    if k=='syn':fd=fd[:5]
                    errors[k]=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-12))
                tests.append(dict(epsilon=eps,relative_errors=errors));log('MIDPOINT FULL FD',dt,tests[-1])
                write(DEST/f'progress_dt{dt}.json',dict(parity=parity,tests=tests))
            assert max(tests[-1]['relative_errors'].values())<1e-4,tests
            rows.append(dict(dt_ms=dt,source=str(path),nominal_full_state_bitwise=True,tests=tests))
            cp=e.cp;del t,e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        write(DEST/'result.json',dict(status='PASS',rows=rows,seconds=time.time()-started,model_promoted=False))
        jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)

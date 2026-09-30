"""Compare the new cache with the independently checked complete derivative."""
from common import OUT,np,read,write,log
from onset_exponential_midpoint import ExponentialMidpointEngine
from onset_midpoint_tangent import MidpointTangent
from onset_midpoint_cached_tangent import MidpointCachedTangent
from fine_rate_frozen_Z_fields import capture,restore
import argparse,os,time,gc

BASE=OUT/'core_a_bifurcation_type_20260924'
DEST=BASE/'numerical_checks/exponential_midpoint/cached_variational_check'


def main(device):
    assert read(DEST.parent/'full_variational_check/result.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(duration_ms=40,meshes_ms=[.05,.025],directions=['relative','absolute'],
        acceptance='Nominal terminal bitwise equal to original midpointflow; every declared directional derivative relative error<1e-10, across syn/M, covariance/memory and entire delayhistory. Two meshes and different source states, duration exceeds maxphysicaldelay.',
        scope='Exact derivative cache implementation only. Physical model and all periodic/stability/type gates unchanged.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(DEST/'jobs.json',jobs);rows=[]
    try:
        for dt,label in [(.05,'matched_natural_short_entry_step'),(.025,'matched_natural_short_entry_step_fine')]:
            e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
            source=BASE/'reference_stability_gap'/label/'held_short_field/checkpoint1000.npz'
            base=dict(np.load(source));restore(e,base)
            begin=time.time();C=MidpointCachedTangent(e,round(40/dt));C.graph();construction=time.time()-begin
            restore(e,base)
            for _ in range(4):e.chunk()
            final=capture(e);parity={k:bool(np.array_equal(v,C.nominal_terminal[k])) for k,v in final.items()}
            assert all(parity.values()),parity
            restore(e,base);J=MidpointTangent(e);J.graph();rng=np.random.default_rng(9250820)
            for kind in ['relative','absolute']:
                d={k:rng.uniform(-1,1,base[k].shape)*(base[k] if kind=='relative' else 1.) for k in ['syn','local','history']}
                d['syn'][5]=0;d['syn'][4,~e.s.E]=0;answers=[];times=[]
                for t in [J,C]:
                    restore(e,base);t.reset();t.syn[:]=e.cp.asarray(d['syn'][:5]);t.local[:]=e.cp.asarray(d['local']);t.history[:]=e.cp.asarray(d['history'])
                    e.cp.cuda.get_current_stream().synchronize();begin=time.time()
                    for _ in range(4):t.chunk()
                    answers.append(dict(syn=t.syn.get(),local=t.local.get(),history=t.history.get()));times.append(time.time()-begin)
                errors={k:float(np.linalg.norm(answers[1][k]-v)/max(np.linalg.norm(v),1e-12)) for k,v in answers[0].items()}
                row=dict(dt_ms=dt,direction=kind,relative_errors=errors,seconds_original_cached=times,
                    nominal_terminal_bitwise=True,cache_construction_seconds=construction)
                rows.append(row);write(DEST/'progress.json',rows);log('MIDPOINT CACHE CHECK',row)
                assert max(errors.values())<1e-10,errors
            cp=e.cp;del C,J,e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        write(DEST/'result.json',dict(status='PASS',rows=rows,model_promoted=False));jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)

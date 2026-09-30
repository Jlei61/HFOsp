"""Two short step halvings at the field whose long-event class changed.

This separates resolved early numerical convergence from later trajectory
separation. It does not turn200ms agreement into onset or attractor proof.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_period_return import dynamical_state,errors
from fine_rate_frozen_Z_fields import restore,capture
from pathlib import Path
import os,time,gc,argparse


def main(device):
    root=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
    out=root/'entry_short_mesh_convergence';out.mkdir(exist_ok=True)
    assert not (out/'jobs.json').exists()
    source=root/'actual_native_entry_midpoint/from_interictal_history/checkpoint10000.npz'
    assert read(root/'preentry_fine_growth/prefix5s_comparison.json')['status']=='INDEPENDENT_MATCHED_5S_PREFIX_READOUT_PASS'
    times=[10,20,50,100,200];steps=[.05,.025,.0125]
    write(out/'contract.json',dict(source=str(source),dt_ms=steps,duration_ms=200,state_times_ms=times,
        question='At Z_A=.712351, the fine5s prefix already has a1.679s activity while matched coarse has at most100ms. Does the same declared continuous-history initial-value problem show early convergence under two step halvings before long-time separation?',
        controls='Same full starting synapses,covariances,input memories,M andZ. Fine histories use the existing successive linear physical-lag interpolation, retaining every original knot and the physical starting clock. AllZ held, allE M dynamic. Constant original external mean and unchanged physical-delay privatevariance/graph/lockedresponse. Three200ms nominal continuations only.',
        readout='Original aligned1ms allgroup rates and full states at10,20,50,100,200ms. Compare every state on the shared .05ms physical-lag grid. Both error differences normalized by the same finest-state block RMS at each time; report ratio, all six physical blocks and actual rate-prefix parity.',
        coarse_replay='Compare completed coarse growth prefix at its recorded dtype; up to onefloat32ULP if the record isfloat32, or1e-14+1e-12*abs(reference)Hz iffloat64. Retain exact-bitwise flag and differences. No change to periodic root, derivative,phase,mesh or stability gates.',
        limits='Early error reduction does not prove long-time trajectory agreement, an asymptotic attractor, a critical parameter or a bifurcation type. Later divergence can reflect accumulated discretization, initial-history representation or instability; it does not identify one explanation by itself.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(out/'jobs.json',jobs)
    original=dict(np.load(source));base=original;previous=.05;private=None;start=time.time()
    expected=np.load(root/'preentry_growth_comparison/native_midpoint/block00.npz')['group_rate_hz'][:200]
    for dt in steps:
        e=build(device,dt);base=regrid_state(base,e,previous);previous=dt;restore(e,base)
        assert abs(float(base['clock'][0]*dt-original['clock'][0]*.05))<1e-10
        assert abs(float(base['clock'][0]*dt/10)-round(float(base['clock'][0]*dt/10)))<1e-8
        now_private=[e.corrected_private_data[k].get() for k in ['ampa','gaba']]
        if private is None:private=now_private
        else:assert all(np.array_equal(x,y) for x,y in zip(private,now_private))
        folder=out/f'dt{dt:g}';folder.mkdir();rates=[]
        for k in range(20):
            q=e.chunk();assert np.array_equal(q[:,0],q[:,1]) and np.isfinite(q).all()
            rates.extend(q[:,0]);tm=(k+1)*10
            if tm in times:
                state=capture(e)
                assert np.array_equal(state['syn'][5],original['syn'][5])
                assert np.all(state['parameters'][19]==0) and np.all(state['parameters'][20]==1)
                np.savez_compressed(folder/f'state{tm:03d}.npz',**state)
        rates=np.array(rates);np.savez_compressed(folder/'rates.npz',group_rate_hz=rates,dt_ms=dt)
        if dt==.05:
            cast=rates.astype(expected.dtype);diff=abs(cast.astype(float)-expected.astype(float))
            bound=np.abs(np.spacing(expected)).astype(float) if expected.dtype==np.float32 else 1e-14+1e-12*abs(expected)
            qa=dict(record_dtype=str(expected.dtype),all_bitwise=bool(np.array_equal(cast,expected)),maximum_absolute_difference_hz=float(diff.max()),within_recorded_bound=bool(np.all(diff<=bound)))
            write(out/'coarse_replay.json',qa);assert qa['within_recorded_bound'],qa
        jobs['completed'].append(dt);write(out/'jobs.json',jobs);log('ENTRY SHORT MESH',dt,'complete',round(time.time()-start,1))
        weights=e.s.sizes/e.s.sizes.sum();cp=e.cp;del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    rows=[]
    for tm in times:
        states=[]
        for dt in steps:
            q=dynamical_state(dict(np.load(out/f'dt{dt:g}'/f'state{tm:03d}.npz')))
            factor=round(.05/dt);q['history']=q['history'][::factor].copy();states.append(q)
        assert all(s['history'].shape==states[0]['history'].shape for s in states)
        ref=states[-1];a={k:ref[k]+states[0][k]-states[1][k] for k in ref}
        coarse_difference=errors(ref,a,weights);fine_difference=errors(ref,states[1],weights)
        ratio=coarse_difference['combined_relative_rms']/max(fine_difference['combined_relative_rms'],1e-30)
        rows.append(dict(time_ms=tm,coarse_minus_medium=coarse_difference,medium_minus_fine=fine_difference,reduction_ratio=ratio))
    write(out/'result.json',dict(status='SHORT_MESH_COMPARISON_COMPLETE',rows=rows,seconds=time.time()-start,
        onset_type='NOT_ESTABLISHED',model_promoted=False))
    jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('ENTRY SHORT MESH COMPARISON',[(r['time_ms'],r['reduction_ratio']) for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

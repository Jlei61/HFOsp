"""Independent unsegmented flow from a saved multiple-shooting candidate.

This checks the candidate's actual spatial activity and measures whole-cycle
and exact subperiod errors in the original model.
An approximate matching candidate is never relabeled as a periodic root.
"""
from common import np,read,write,log,OUT
from onset_state_continuation import build,regional_weights
from onset_poincare_corrector import SectionReturn
from onset_segment_flow import fixed_time,fixed_times
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import restore,capture
from onset_relative_rate_recorder import RelativeRateRecorder
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse,os


def audit(e,parent,single_pass=False):
    parent=Path(parent).resolve();out=parent/'independent_whole_flow';out.mkdir(exist_ok=True)
    assert not (out/'jobs.json').exists()
    assert e.dt==read(parent.parent/'contract.json')['dt_ms'], 'Audit must use the root integration mesh'
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    if (parent/'accepted_period.json').exists():
        meta=read(parent/'accepted_period.json');prefix='accepted_node'
    else:
        result=read(parent.parent/'result.json')
        assert result['status'] in ['NUMERICAL_MULTIPLE_SHOOTING_ROOT','NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT']
        row=result['iterations'][-1];assert parent.name==f"iteration{row['iteration']:02d}"
        meta=dict(period_ms=row['period_ms'],durations_ms=row.get('durations_ms'));prefix='node'
    T=meta['period_ms'];paths=sorted(parent.glob(prefix+'[0-9][0-9].npz'));K=len(paths)
    assert K>=2 and [p.name for p in paths]==[f'{prefix}{j:02d}.npz' for j in range(K)]
    h=np.array(meta.get('durations_ms') or [T/K]*K);assert len(h)==K and abs(h.sum()-T)<1e-8
    base=dict(np.load(paths[0]));Z=base['syn'][5].copy()
    A=SectionReturn(base,e,T);x=A.xref;w=e.s.sizes/e.s.sizes.sum();returns=[]
    divisors=[1,2,3,4,6,9];joint_times=np.cumsum(h)[:-1]
    times=[T/d for d in divisors]+joint_times.tolist()
    shared=fixed_times(A,x,times) if single_pass else None
    parity=[]
    if single_pass:
        for index in [0,3]:
            expected,slope=fixed_time(A,x,times[index]);actual,actual_slope=shared[index]
            error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-15))
            slope_error=float(np.linalg.norm(actual_slope-slope)/max(np.linalg.norm(slope),1e-15))
            parity.append(dict(time_ms=times[index],state_relative_error=error,slope_relative_error=slope_error))
            write(out/'shared_prefix_parity.json',parity)
            assert error<1e-10 and slope_error<1e-8,('Shared prefix parity failed',parity[-1])
    for index,divisor in enumerate(divisors):
        y,_=shared[index] if shared is not None else fixed_time(A,x,T/divisor)
        check=errors(dynamical_state(base),dynamical_state(A.state(y)),w)
        returns.append(dict(divisor=divisor,actual_time_ms=T/divisor,**check))
        log('MULTIPLE CANDIDATE EXACT RETURN',str(parent),divisor,check['combined_relative_rms'])
    joints=[]
    for j,t in enumerate(joint_times,1):
        y,_=shared[len(divisors)+j-1] if shared is not None else fixed_time(A,x,t)
        target=dict(np.load(paths[j]))
        assert np.array_equal(target['syn'][5],Z)
        joints.append(dict(node=j,time_ms=float(t),**errors(dynamical_state(target),dynamical_state(A.state(y)),w)))
    assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/relative_rate_recorder/result.json')['status']=='PASS'
    restore(e,base);W=regional_weights(e.s);recorder=RelativeRateRecorder(e)
    R,M,period_mean=recorder.read_period(T)
    assert np.isfinite(R).all() and R.min()>=0 and np.array_equal(period_mean[0],period_mean[1])
    time=np.arange(1,len(R)+1.)
    rr=R@W.T;sm=uniform_filter1d(rr,10,axis=0,mode='nearest');rows=[]
    for j,name in enumerate(['Global E','Core A','Core B','Surround']):
        edge=np.diff(np.r_[False,sm[:,j]<5,False].astype(int))
        quiet=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if b-a>=20]
        rows.append(dict(region=name,mean_hz=float(period_mean[0]@W[j]),complete_1ms_bins_mean_hz=float(rr[:,j].mean()),min_10ms_hz=float(sm[:,j].min()),
                         max_10ms_hz=float(sm[:,j].max()),quiet_intervals_relative_ms=quiet))
    terminal=capture(e);assert np.array_equal(terminal['syn'][5],Z)
    assert np.all(base['parameters'][19]==0) and np.all(base['parameters'][20]==1)
    np.savez_compressed(out/'profile.npz',time_ms=time,group_rate_hz=R.astype('f4'),regional_rate_hz=rr,
                         M_current=np.array(M),M_time_ms=np.arange(1,len(M)+1)*10.,Z=Z,period_seed_ms=T,
                         exact_T_group_mean_hz=period_mean[0])
    result=dict(status='UNSEGMENTED_CANDIDATE_FLOW_AUDITED_NOT_A_PERIODIC_CERTIFICATE',source=str(parent),
                period_seed_ms=T,dt_ms=e.dt,segments=K,exact_period_returns=returns,shooting_joint_errors=joints,rows=rows,
                shared_prefix=single_pass,shared_prefix_independent_parity=parity,
                definitions='One original uninterrupted full-network flow from accepted node0. All Z held and all E M dynamic. Exact T/divisor closure uses original cubic dense output, without rounding to milliseconds. Passive rate bins are1ms relative to the arbitrary starting clock, followed by10ms smoothing. Means integrate original per-step flux across exactT, including fractional last-step weight. Step quadrature error still needs mesh refinement. Edge quiet intervals can be censored.',
                scope='A saved solver iterate can be approximate. Whole-flow residual, even if small, still requires independent phase/mesh, fundamental period, Floquet and onset correspondence. A subperiod mismatch does not certify the full period.',model_promoted=False)
    write(out/'result.json',result);write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))
    log('MULTIPLE CANDIDATE PHYSICAL PROFILE',str(parent),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parents',nargs='+');p.add_argument('--device',type=int,default=1)
    p.add_argument('--single-pass',action='store_true')
    p.add_argument('--dt',type=float,default=.05)
    a=p.parse_args();e=build(a.device,a.dt)
    for parent in a.parents:audit(e,parent,a.single_pass)

"""Uninterrupted physical flow from a numerical return-branch candidate.

Measure whether alternating-mode growth changes burst termination or merely
the organization of bursts. No section resets during this record.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regional_weights,regrid_state
from onset_relative_rate_recorder import RelativeRateRecorder
from physical_delay_count_rate import projections
from fine_rate_frozen_Z_fields import capture,restore
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse,os,time


def main(a):
    source=Path(a.source).resolve();out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'jobs.json').exists();assert a.duration%10==0 and 100<=a.duration<=20000
    assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/relative_rate_recorder/result.json')['status']=='PASS'
    write(out/'contract.json',dict(source=str(source),dt_ms=a.dt,source_dt_ms=a.source_dt,duration_ms=a.duration,
        target_native_time_ms=a.target_native_time,
        question=('Does the exact natural short-event history change to long CoreA activity when only its withinCoreA Z field is depleted to the specified adjacent native field?' if a.target_native_time is not None else 'Does the supplied complete state generate long CoreA activity in the original uninterrupted network, or remain a succession of short self-terminating bursts?'),
        method='Restore the entire saved fast/M/delay state and run the unchanged original deterministic physical-delay model without section resets or interpolation. Record passive relative1ms group rates,10ms M and20x20 spatial rates.',
        resource='Every Z frozen, every E M dynamic; constant original external mean and private variance. No parameter or state clamp beyond the existing frozen-Z family.',
        definitions='For each region:10ms-smoothed E rate, quiet<5Hz foratleast20ms; intervals between qualified quiet intervals are activities, with edge censoring. Spatial persistent fraction: E-cell weight whose50Hz threshold duty>=90% inlast1s.',
        scope='Finite-time nonlinear-state readout, not a new periodic root, stability certificate or bifurcation type. A return to quiet refutes sustained activity over that observed interval only.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);start=time.time()
    try:
        e=build(a.device,a.dt);base=regrid_state(dict(np.load(source)),e,a.source_dt);assert base['history'].shape==e.local.history.shape
        assert np.all(base['parameters'][19]==0) and np.all(base['parameters'][20]==1)
        if a.target_native_time is not None:
            from core_a_parameter_path_audit import NativeTimeFamily
            family=NativeTimeFamily(e.s);z,d=family.field_at_time(a.target_native_time)
            assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
            original={key:value.copy() for key,value in base.items()};base['syn'][5]=z.copy()
            for key,value in original.items():
                if key=='syn':assert np.array_equal(value[:5],base[key][:5])
                else:assert np.array_equal(value,base[key])
            write(out/'Z_intervention.json',dict(only_within_CoreA_Z_changed=True,outsideA_Z_bitwise_unchanged=True,
                all_initial_fast_M_history_bitwise_unchanged=True,theta_ms=a.target_native_time,D_A=d,Z_A=1-d))
        restore(e,base);rec=RelativeRateRecorder(e);W=regional_weights(e.s);P,count=projections(e.s,e.coarse,e.parent)[20]
        R=[];M=[];Z=base['syn'][5].copy()
        for k in range(a.duration//10):
            x=rec.chunk();assert np.array_equal(x[:,0],x[:,1]) and np.isfinite(x).all() and x.min()>=0
            R.append(x[:,0]);M.append(e.syn[4].get())
            if (k+1)%100==0:
                now=(k+1)*10;state=capture(e);assert np.array_equal(state['syn'][5],Z)
                np.savez_compressed(out/f'checkpoint{now}.npz',**state)
                jobs.update(completed_ms=now);write(out/'jobs.json',jobs);log('CANDIDATE UNINTERRUPTED FLOW',now)
        r=np.concatenate(R);m=np.array(M);regional=r@W.T;field=(P@r.T).T
        sm=uniform_filter1d(regional,10,axis=0,mode='nearest');rows=[]
        for j,name in enumerate(['Global E','Core A','Core B','Surround']):
            edge=np.diff(np.r_[False,sm[:,j]<5,False].astype(int))
            quiet=[(int(u),int(v)) for u,v in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if v-u>=20]
            activities=[];last=0
            for u,v in quiet:
                if u>last:activities.append(dict(start_ms=last,end_ms=u,duration_ms=u-last,left_censored=last==0,right_censored=False))
                last=v
            if last<len(r):activities.append(dict(start_ms=last,end_ms=len(r),duration_ms=len(r)-last,left_censored=last==0,right_censored=True))
            rows.append(dict(region=name,mean_hz=float(regional[:,j].mean()),quiet_intervals_ms=quiet,activities=activities))
        final=capture(e);assert int(final['clock'][0])-int(base['clock'][0])==round(a.duration/a.dt)
        assert np.array_equal(final['syn'][5],Z)
        persistent=float(count[((field[-min(1000,len(field)):]>=50).mean(0)>=.9)].sum()/count.sum())
        np.savez_compressed(out/'trajectory.npz',time_ms=np.arange(1,len(r)+1),group_rate_hz=r.astype('f4'),
            regional_rate_hz=regional,field_E_hz=field.astype('f4'),cell_counts=count,
            M_time_ms=np.arange(1,len(m)+1)*10,M_current=m,Z=Z)
        np.savez_compressed(out/'final_state.npz',**final)
        result=dict(status='UNINTERRUPTED_CANDIDATE_FLOW_COMPLETE',rows=rows,spatial_persistent_fraction_last1s=persistent,
            seconds=time.time()-start,source=str(source),onset_bifurcation_type='NOT_ESTABLISHED',model_promoted=False)
        write(out/'result.json',result);jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('CANDIDATE FLOW PROFILE',rows)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--duration',type=int,default=4000)
    p.add_argument('--source-dt',type=float,default=.05);p.add_argument('--target-native-time',type=float)
    p.add_argument('--device',type=int,default=0);main(p.parse_args())

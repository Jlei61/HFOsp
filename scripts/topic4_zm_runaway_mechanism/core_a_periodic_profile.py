"""Read actual spatial activity along an already closed full-state orbit."""
from common import np,read,write,log
from onset_state_continuation import build,regional_weights
from onset_variational_return import Coordinates
from fine_rate_frozen_Z_fields import capture,restore
from physical_delay_count_rate import projections
from pathlib import Path
import argparse,os


def main(parent,device):
    parent=Path(parent).resolve();meta=read(parent/'result.json')
    assert meta['status']=='NUMERICAL_PERIODIC_ROOT'
    out=parent/'actual_orbit_profile';out.mkdir(exist_ok=True)
    assert not(out/'jobs.json').exists();write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    contract=read(parent/'contract.json');dt=contract['dt_ms'];T=meta.get('period_ms')
    if T is None:T=meta.get('rows',meta.get('iterations'))[-1]['period_ms']
    source=parent/('root_state.npz' if (parent/'root_state.npz').exists() else 'latest_state.npz')
    numerical_method=contract.get('numerical_method','old_endpoint')
    if numerical_method=='exponential_midpoint':
        from onset_exponential_midpoint import ExponentialMidpointEngine
        e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
    else:
        assert numerical_method=='old_endpoint'
        e=build(device,dt)
    base=dict(np.load(source));c=Coordinates(base,e.s);x=c.pack(base)
    with e.stream:
        e.stream.begin_capture()
        for _ in range(round(1/dt)):e.step()
        one=e.stream.end_capture()
    restore(e,base);start=int(base['clock'][0]);n=round(1/dt)
    W=regional_weights(e.s);P,count=projections(e.s,e.coarse,e.parent)[20]
    R=[base['history'][start%len(base['history'])]*1000];M=[base['syn'][4]];Mtime=[0.]
    partial=[];targets={round(T/j):j for j in range(2,9)}
    for ms in range(1,int(np.ceil(T))+1):
        one.launch(e.stream);e.stream.synchronize();tick=int(e.local.clock.get()[0])
        index=(tick-np.arange(n-1,-1,-1))%len(base['history'])
        r=e.local.history[e.cp.asarray(index)].get()*1000
        R.extend(r);M.append(e.syn[4].get());Mtime.append(float(ms))
        if ms in targets:
            state=capture(e);q=c.pack(state)
            partial.append(dict(divisor=targets[ms],elapsed_ms=ms,
                full_state_coordinate_relative=float(np.linalg.norm(q-x)/np.linalg.norm(x))))
    r=np.array(R);t=np.arange(len(r))*dt;keep=t<=T
    # Integral over one period, linearly interpolating only the final fraction.
    final=np.array([np.interp(T,t,r[:,j]) for j in range(e.s.P)])
    integration_t=np.r_[t[keep],T];integration_r=np.vstack([r[keep],final])
    regional=integration_r@W.T
    mean=np.trapz(regional,integration_t,axis=0)/T
    groupmean=np.trapz(integration_r,integration_t,axis=0)/T
    np.savez_compressed(out/'profile.npz',time_ms=t[keep],group_rate_hz=r[keep].astype('f4'),
        regional_rate_hz=r[keep]@W.T,mean_group_hz=groupmean,
        M_time_ms=Mtime,M_current=np.array(M),Z=base['syn'][5],cell_counts=count,
        mean_field_hz=P@groupmean,period_ms=T,dt_ms=dt)
    rows=[]
    for j,label in enumerate(['Global E','Core A','Core B','Surround']):
        rows.append(dict(region=label,mean_hz=float(mean[j]),min_hz=float(regional[:,j].min()),
            max_hz=float(regional[:,j].max()),time_fraction_below5=float(np.mean(regional[:,j]<5))))
    write(out/'result.json',dict(status='ACTUAL_CLOSED_ROOT_PROFILE_RECORDED',period_ms=T,dt_ms=dt,
        numerical_method=numerical_method,
        rows=rows,partial_period_returns=partial,
        fundamental_scope='Excludes tested subperiods only; full phase/mesh checks and Floquet certificate remain separate.',
        model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()));log('CORE A ORBIT PROFILE',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0)
    a=p.parse_args();main(a.parent,a.device)

"""Replay a selected observed recurrence in the same midpoint physical flow."""
from common import np,read,write,log
from onset_exponential_midpoint import ExponentialMidpointEngine
from onset_relative_rate_recorder import RelativeRateRecorder
from onset_period_return import dynamical_state,errors
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,os,time


def main(a):
    source=Path(a.source).resolve();out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'jobs.json').exists()
    audit=read(source/'independent_audit.json');assert audit['status']=='INDEPENDENT_READOUT_PASS'
    contract=read(source/'numerical_contract.json');assert contract['method']=='exponential_midpoint'
    dt=contract['dt_ms'];candidate=next(q for q in audit['best_by_burst_count'] if q['burst_count']==a.burst_count)
    first=round(candidate['time1_ms']/dt)*dt;last=round(candidate['time2_ms']/dt)*dt
    origin=int(first//1000)*1000;terminal=int(np.ceil(last/1000))*1000
    assert 0<origin<first<last<=terminal<=read(source/'contract.json')['duration_ms']
    write(out/'contract.json',dict(source=str(source),dt_ms=dt,numerical_method='exponential_midpoint',candidate=candidate,
        times_ms=[first,last],period_seed_ms=last-first,checkpoint_interval_ms=[origin,terminal],
        acceptance='All original1ms group rates within one float32 ULP+1e-12Hz; full original float64 terminalcheckpoint combinederror<1e-10 and everyblock<1e-9. Report bitwise identity separately. This checks seed provenance, not periodic closure.',
        scope='Complete retained spatial/M/history state, actual uninterrupted flow. A near return only nominates shooting; no bifurcation claim.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);begin=time.time()
    try:
        e=ExponentialMidpointEngine(dt=dt,device=a.device);e.graph()
        initial=dict(np.load(source/f'checkpoint{origin}.npz'));restore(e,initial);rec=RelativeRateRecorder(e)
        tick=int(initial['clock'][0]);targets={tick+round((t-origin)/dt):i for i,t in enumerate([first,last])}
        with np.load(source/'trajectory.npz') as z:expected=z['group_rate_hz'][origin:terminal]
        states={};bitwise=True;different=0;maxdifference=0.
        for elapsed in range(origin+10,terminal+1,10):
            now=tick+round((elapsed-origin-10)/dt);end=now+round(10/dt)
            if any(now<t<=end for t in targets):
                for t in range(now+1,end+1):
                    rec.step()
                    if t in targets:
                        state=capture(e);assert int(state['clock'][0])==t
                        states[targets[t]]=state;np.savez_compressed(out/f'node{targets[t]:02d}.npz',**state)
                actual=rec.output.get()
            else:actual=rec.chunk()
            offset=elapsed-origin;ref=expected[offset-10:offset];q=actual[:,0].astype('f4')
            delta=abs(q.astype(float)-ref.astype(float));bound=abs(np.spacing(ref).astype(float))+1e-12
            assert np.all(delta<=bound),('Recorded source mismatch',elapsed,float(delta.max()))
            bitwise=bitwise and bool(np.array_equal(q,ref));different+=int(np.count_nonzero(q!=ref));maxdifference=max(maxdifference,float(delta.max()))
            if elapsed%1000==0:log('MIDPOINT SEED REPLAY',elapsed)
        assert len(states)==2
        w=e.s.sizes/e.s.sizes.sum();final=capture(e);reference=dict(np.load(source/f'checkpoint{terminal}.npz'))
        checkpoint=errors(dynamical_state(reference),dynamical_state(final),w)
        assert checkpoint['combined_relative_rms']<1e-10 and max(v['relative_rms'] for v in checkpoint['blocks'].values())<1e-9
        assert np.array_equal(final['syn'][5],reference['syn'][5])
        closure=errors(dynamical_state(states[0]),dynamical_state(states[1]),w)
        write(out/'result.json',dict(status='SOURCE_REPLAY_PASS_RECURRENCE_ONLY',period_ms=last-first,dt_ms=dt,
            all_recorded_rates_bitwise=bitwise,recorded_differences=different,maximum_rate_difference_hz=maxdifference,
            source_checkpoint=checkpoint,full_state_recurrence=closure,seconds=time.time()-begin,model_promoted=False))
        jobs['status']='COMPLETE';write(out/'jobs.json',jobs);log('MIDPOINT FULL RECURRENCE',last-first,closure)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--burst-count',type=int,required=True);p.add_argument('--device',type=int,default=0);main(p.parse_args())

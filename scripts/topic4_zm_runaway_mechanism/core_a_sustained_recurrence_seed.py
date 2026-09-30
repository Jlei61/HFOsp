"""Exact full-state seed from the sustained-Core-A trajectory, not short bursts."""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from fine_rate_frozen_Z_fields import capture,restore
from onset_variational_return import Coordinates
from onset_period_return import dynamical_state,errors
import argparse,os,time

ROOT=OUT/'core_a_bifurcation_type_20260924'


def main(device):
    parent=ROOT/'near_returns/below_high_A';screen=read(parent/'result.json')
    choices=[r for r in screen['candidates'] if r['period_ms']<=150]
    row=min(choices,key=lambda r:r['score']);times=[int(row['time1_ms']),int(row['time2_ms'])]
    out=parent/'exact_short';out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    source=ROOT/'fold_attractor_contrast/below'
    assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
    start_time=((times[0]-1)//5000)*5000;assert start_time>0
    statefile=source/f'checkpoint{start_time}.npz';expected={}
    for t in range(start_time,times[-1]+1,5000):
        expected[t//5000]=np.load(source/f'block{t//5000:02d}.npz')['group_rate_hz']
    write(out/'contract.json',dict(question='Does the sustained-Core-A side contain a full periodic skeleton distinct from the short self-limited burst cycle?',
        candidate=row,selection='Lowest full-group recurrence score among periods<=150ms during continuously high Core A. Longer B-burst candidates remain separate; this short recurrence is not presumed periodic.',
        source=str(statefile),source_time_ms=start_time,
        equations='Same entire spatial delayed model, native Core-A-only resource field at D_A=.3431576413, all Z held and all M dynamic. Exact replay from actual trajectory checkpoint, no rate, M or history reset.',
        acceptance='Every replayed1ms original group-rate record must match bitwise at its storedfloat32precision. Full-state recurrence residual is reported, not called a periodic orbit.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    e=build(device);base=dict(np.load(statefile));restore(e,base)
    assert np.all(base['parameters'][19]==0) and np.all(base['parameters'][20]==1)
    with e.stream:
        e.stream.begin_capture()
        for _ in range(round(1/e.dt)):e.step()
        one=e.stream.end_capture()
    states=[];begin=time.time();stop=((times[-1]+9)//10)*10
    for tm in range(start_time+10,stop+1,10):
        if any(tm-10<t<=tm for t in times):
            for tick in range(tm-9,tm+1):
                one.launch(e.stream);e.stream.synchronize()
                if tick in times:
                    state=capture(e);states.append(state)
                    np.savez_compressed(out/f'state{tick}.npz',**state)
            rate=e.output.get()
        else:rate=e.chunk()
        block=(tm-1)//5000;j=tm-block*5000
        assert np.array_equal(rate[:,0].astype('f4'),expected[block][j-10:j])
        assert np.array_equal(e.syn[5].get(),base['syn'][5])
    assert len(states)==2
    c=Coordinates(states[0],e.s);x=c.pack(states[0]);y=c.pack(states[1])
    error=errors(dynamical_state(states[0]),dynamical_state(states[1]),e.s.sizes/e.s.sizes.sum())
    write(out/'result.json',dict(status='EXACT_REPLAY_PASS_RECURRENCE_ONLY',times_ms=times,period_ms=times[1]-times[0],
        full_state_errors=error,weighted_coordinate_relative=float(np.linalg.norm(y-x)/np.linalg.norm(x)),
        all_original_1ms_group_rates_bitwise=True,seconds=time.time()-begin,model_promoted=False))
    jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('SUSTAINED A EXACT SEED',times,error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args();main(a.device)

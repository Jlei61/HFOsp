"""Recover complete states of one individually recurrent burst, exactly."""
from common import OUT,np,read,write,log
import onset_state_continuation as flow
from fine_rate_frozen_Z_fields import capture,restore
from onset_variational_return import Coordinates
from onset_period_return import dynamical_state,errors
import argparse,os,time

LOCAL=OUT/'core_a_transition_continuation_20260924'
DEST=OUT/'core_a_bifurcation_type_20260924/near_returns/mid_lower/exact_seed'


def main(device):
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    row=read(DEST.parent/'result.json')['candidates'][0];times=[int(row['time1_ms']),int(row['time2_ms'])]
    assert times[-1]<5000
    c=read(LOCAL/'conditions.json')['mid1_from_lower'];flow.DEST=LOCAL
    expected=np.load(LOCAL/'mid1_from_lower/block00.npz')['group_rate_hz']
    write(DEST/'contract.json',dict(question='Can the closest individual recurrence of the self-terminating side seed a complete-state periodic solution?',
        candidate=row,source_condition=c,controls='Exact replay of unchanged full spatial equations with all M dynamic and the original local frozen Z field; require1ms recorded rates to match. Recurrence is not a closed orbit.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(DEST/'jobs.json',jobs)
    e=flow.build(device);Z=flow.initialize(e,c)
    with e.stream:
        e.stream.begin_capture()
        for _ in range(round(1/e.dt)):e.step()
        one=e.stream.end_capture()
    states=[];start=time.time();stop=((times[-1]+9)//10)*10
    for tm in range(10,stop+1,10):
        if any(tm-10<t<=tm for t in times):
            for st in range(tm-9,tm+1):
                one.launch(e.stream);e.stream.synchronize()
                if st in times:
                    state=capture(e);states.append(state)
                    np.savez_compressed(DEST/f'state{st}.npz',**state)
            rates=e.output.get()
        else:rates=e.chunk()
        assert np.array_equal(rates[:,0].astype('f4'),expected[tm-10:tm])
        if tm%500==0:jobs['elapsed_ms']=tm;write(DEST/'jobs.json',jobs);log('EXACT LOCAL RECURRENCE',tm,time.time()-start)
    assert len(states)==2;C=Coordinates(states[0],e.s)
    x=C.pack(states[0]);y=C.pack(states[1]);err=errors(dynamical_state(states[0]),dynamical_state(states[1]),e.s.sizes/e.s.sizes.sum())
    write(DEST/'result.json',dict(status='EXACT_REPLAY_PASS_RECURRENCE_ONLY',times_ms=times,period_ms=times[1]-times[0],
        weighted_coordinate_relative=float(np.linalg.norm(y-x)/np.linalg.norm(x)),full_state_errors=err,
        all_original1ms_group_rates_bitwise=True,model_promoted=False))
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs);log('EXACT LOCAL RECURRENCE RESULT',err)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

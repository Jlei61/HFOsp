"""Replay the observed six-burst reference recurrence in the original flow."""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regional_weights
from fine_rate_frozen_Z_fields import capture,restore
from onset_period_return import dynamical_state,errors
import argparse,os,time

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
DEST=BASE/'actual_reference_recurrence/exact_six_burst_seed'


def main(device):
    screen=read(BASE/'actual_reference_recurrence/result.json')
    assert screen['status']=='PASSIVE_FULL_GROUP_RECURRENCE_SCREEN'
    candidate=min(screen['candidates'],key=lambda r:r['score'])
    assert candidate['burst_count']==6
    source=OUT/'onset_state_continuation_20260923/native9000_from_lower'
    dt=.05;first=round(candidate['time1_ms']/dt)*dt;last=round(candidate['time2_ms']/dt)*dt
    assert 5000<first<last<10000
    # Saved nodes are actual states on the integration grid. Quarter-node
    # offsets up to half a step belong to the initial guess only; the
    # subsequent nonlinear solver enforces its own exact segment durations.
    times=np.rint(np.linspace(first,last,5)/dt)*dt
    sixth=np.rint(np.linspace(first,last,7)/dt)*dt
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(candidate=candidate,dt_ms=dt,source=str(source/'checkpoint5000.npz'),
        question='Does the actual interictal multi-burst recurrence give a full-state periodic candidate distinct from the already unstable single-burst root?',
        times_ms=times.tolist(),six_node_times_ms=sixth.tolist(),
        node_rounding='Full actual states at nearest0.05ms grid points; no interpolated history or M. Quarter-node offset is only an initial-guess error, to be removed by exact segment matching.',
        equations='Unchanged full3479-group spatial drift, native9s full Z held, every E M dynamic, original constant mean and private variance. No future SNN input or physical parameter adjustment.',
        acceptance='All replayed1ms group rates match original stored float32 bitwise; complete state endpoint error reported. Recurrence alone does not establish a periodic attractor or a bifurcation.',model_promoted=False))
    write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    e=build(device);base=dict(np.load(source/'checkpoint5000.npz'));restore(e,base)
    clock0=int(base['clock'][0]);targets={}
    for prefix,tt in [('node',times),('sixnode',sixth)]:
        for j,tm in enumerate(tt):targets.setdefault(clock0+round((tm-5000)/dt),[]).append((prefix,j))
    expected=np.load(source/'block01.npz')['group_rate_hz'];states={};begin=time.time()
    stop=int(np.ceil(last/10))*10
    for tm in range(5010,stop+1,10):
        now=int(e.local.clock.get()[0]);end=now+round(10/dt)
        if any(now<tick<=end for tick in targets):
            for tick in range(now+1,end+1):
                e.step()
                if tick in targets:
                    e.cp.cuda.get_current_stream().synchronize();state=capture(e)
                    assert int(state['clock'][0])==tick and np.array_equal(state['syn'][5],base['syn'][5])
                    for prefix,j in targets[tick]:
                        np.savez_compressed(DEST/f'{prefix}{j:02d}.npz',**state)
                        if prefix=='node':states[j]=state
            e.cp.cuda.get_current_stream().synchronize();r=e.output.get()
        else:r=e.chunk()
        offset=tm-5000
        assert np.array_equal(r[:,0].astype('f4'),expected[offset-10:offset])
        if tm%1000==0:log('REFERENCE EXACT MULTIBURST REPLAY',tm,round(time.time()-begin,1))
    assert len(states)==5
    er=errors(dynamical_state(states[0]),dynamical_state(states[4]),e.s.sizes/e.s.sizes.sum())
    rate=expected[int(first-5000):int(last-5000)].astype(float)@regional_weights(e.s).T
    write(DEST/'result.json',dict(status='EXACT_REPLAY_PASS_RECURRENCE_ONLY',period_ms=last-first,
        times_ms=times.tolist(),six_node_times_ms=sixth.tolist(),full_state_errors=er,
        all_original_1ms_group_rates_bitwise=True,all_Z_held=True,all_M_dynamic=True,
        mean_global_A_B_surround_hz=rate.mean(0).tolist(),seconds=time.time()-begin,model_promoted=False))
    write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid()));log('REFERENCE EXACT SEED COMPLETE',er)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

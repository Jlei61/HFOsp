"""Replay a complete A-sustained/B-burst recurrence with intermediate states.

Every state comes from the original physical trajectory. This is a seed for
full-state multiple shooting, not a periodicity or bifurcation certificate.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regional_weights
from fine_rate_frozen_Z_fields import capture,restore
from onset_period_return import dynamical_state,errors
import argparse,os,time


def main(device):
    root=OUT/'core_a_bifurcation_type_20260924'
    parent=root/'near_returns/below_high_A'
    candidate=min((r for r in read(parent/'result.json')['candidates'] if r['period_ms']>=200),key=lambda r:r['score'])
    first,last=candidate['time1_ms'],candidate['time2_ms']
    times=np.linspace(first,last,5)
    out=parent/'exact_long';out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    source=root/'fold_attractor_contrast/below'
    start=int((first-1)//5000)*5000
    write(out/'contract.json',dict(source=str(source/f'checkpoint{start}.npz'),candidate=candidate,
        selection='Lowest original all-group recurrence score with period>=200ms and continuously high Core A. Its actual B burst and quiet structure must be checked; no stable cycle is assumed.',
        times_ms=times.tolist(),number_segments=4,dt_ms=.05,
        equations='Original3479-group spatial field, same physical delays and locked response. All M dynamic. Entire Z held, only native Core-A resource pattern differs from native9s.',
        validation='Every replayed1ms all-group rate must match the original float32 record bitwise; all stored Z fields must match exactly. Intermediate snapshots are exact actual steps, with no interpolated physical state.',
        interpretation='A multi-segment shooting seed only; full closure, nontrivial amplitude, independent phase/mesh and stability remain required.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    e=build(device);base=dict(np.load(source/f'checkpoint{start}.npz'));restore(e,base)
    ticks=np.rint(times/e.dt).astype(int);assert np.max(abs(ticks*e.dt-times))<1e-10
    # Absolute model clocks need not equal the experiment's elapsed clocks.
    clock0=int(base['clock'][0]);targets={clock0+int(t-round(start/e.dt)):j for j,t in enumerate(ticks)}
    stop=int(np.ceil(last/10))*10;expected={i:np.load(source/f'block{i:02d}.npz')['group_rate_hz']
        for i in range(start//5000,(stop-1)//5000+1)}
    states={};begin=time.time();W=regional_weights(e.s)
    for tm in range(start+10,stop+1,10):
        now=int(e.local.clock.get()[0]);end=now+round(10/e.dt)
        if any(now<t<=end for t in targets):
            for tick in range(now+1,end+1):
                e.step()
                if tick in targets:
                    e.cp.cuda.get_current_stream().synchronize();state=capture(e);j=targets[tick]
                    assert int(state['clock'][0])==tick
                    assert np.array_equal(state['syn'][5],base['syn'][5])
                    states[j]=state;np.savez_compressed(out/f'node{j:02d}.npz',**state)
            e.cp.cuda.get_current_stream().synchronize();rate=e.output.get()
        else:rate=e.chunk()
        block=(tm-1)//5000;j=tm-block*5000
        assert np.array_equal(rate[:,0].astype('f4'),expected[block][j-10:j])
        if tm%1000==0:log('LONG CORE A RECURRENCE REPLAY',tm,round(time.time()-begin,1))
    assert len(states)==5
    er=errors(dynamical_state(states[0]),dynamical_state(states[4]),e.s.sizes/e.s.sizes.sum())
    R=np.concatenate([expected[k] for k in sorted(expected)]).astype(float)@W.T
    selected=R[int(first-start):int(last-start)]
    readout=[dict(region=name,mean_hz=float(selected[:,j].mean()),min_hz=float(selected[:,j].min()),
        max_hz=float(selected[:,j].max()),fraction_below5=float(np.mean(selected[:,j]<5)))
        for j,name in enumerate(['Global E','Core A','Core B','Surround'])]
    write(out/'result.json',dict(status='EXACT_REPLAY_PASS_RECURRENCE_ONLY',times_ms=times.tolist(),
        period_ms=last-first,full_state_errors=er,actual_state_structure=readout,
        original_1ms_group_rates_bitwise=True,all_Z_held=True,all_M_dynamic=True,
        seconds=time.time()-begin,model_promoted=False))
    jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('LONG CORE A SEED COMPLETE',er,readout)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args();main(a.device)

"""Change only the numerical shooting partition of an actual flow seed.

Every new node comes from the unchanged full flow. All original saved nodes
are revisited and compared in full physical coordinates before accepting
the repartition. No extra physical modes or changed model parameters.
"""
from common import np,read,write,log
from onset_state_continuation import build
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,os,time


def main(a):
    source=Path(a.source).resolve();out=Path(a.destination).resolve();r=read(source/'result.json')
    assert r['status'] in ['EXACT_REPLAY_PASS_RECURRENCE_ONLY','REPLAY_AGREEMENT_PASS_RECURRENCE_ONLY']
    K=a.segments;T=r['period_ms'];oldK=r.get('segments',4);dt=.05
    initial=dict(np.load(source/'node00.npz'));tick0=int(initial['clock'][0]);old={}
    for j in range(oldK+1):
        state=dict(np.load(source/f'node{j:02d}.npz'));offset=int(state['clock'][0])-tick0
        assert np.array_equal(state['syn'][5],initial['syn'][5]);old[offset]=(j,state)
    steps=round(T/dt);assert abs(steps*dt-T)<1e-8 and max(old)==steps
    targets={int(n):j for j,n in enumerate(np.rint(np.linspace(0,steps,K+1)).astype(int))}
    out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'contract.json',dict(source=str(source),segments=K,old_segments=oldK,period_ms=T,dt_ms=dt,
        reason='The reference foursegment Newton operator has a projected singular-value ratio about7.6e6, and all bounded actual proposals fail despite the full derivative test. Shorter physical flow segments limit segmentwise expansion; this changes the boundary-value numerical partition only.',
        physical_scope='Same exact complete source node0, unchangedfull3479-group drift, allZheld/allE Mdynamic, no futureinput or response/network adjustment.',
        verification='Visit every original saved node at its original actual grid time. Original six-block comparison combined<1e-10 and eachblock<1e-9; clock/Z/parameters exact. Report bit equality separately. Original periodic closure, phase, mesh and stability gates unchanged.',
        new_nodes='Actual original-flow states at nearest0.05ms uniform fraction times; roundoff in node spacing belongs only to the initial seed, not the matching equation.',model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()));e=build(a.device);restore(e,initial)
    w=e.s.sizes/e.s.sizes.sum();now=0;checks=[];all_bits=True;first=None;last=None;start=time.time()
    for tick in sorted(set(old)|set(targets)):
        whole,tail=divmod(tick-now,round(10/dt))
        for _ in range(whole):e.chunk()
        for _ in range(tail):e.step()
        e.cp.cuda.get_current_stream().synchronize();state=capture(e);now=tick
        assert int(state['clock'][0])==tick0+tick and np.array_equal(state['syn'][5],initial['syn'][5])
        if tick in old:
            j,reference=old[tick];err=errors(dynamical_state(reference),dynamical_state(state),w)
            exact=all(v.tobytes()==state[k].tobytes() for k,v in reference.items());all_bits &= exact
            assert np.array_equal(state['parameters'],reference['parameters'])
            checks.append(dict(old_node=j,relative_time_ms=tick*dt,full_bits=bool(exact),**err));write(out/'source_node_checks.json',checks)
            assert err['combined_relative_rms']<1e-10 and max(v['relative_rms'] for v in err['blocks'].values())<1e-9,checks[-1]
        if tick in targets:
            j=targets[tick];np.savez_compressed(out/f'node{j:02d}.npz',**state)
            if j==0:first=state
            if j==K:last=state
            log('REPARTITION ACTUAL NODE',j,K,tick*dt)
    closure=errors(dynamical_state(first),dynamical_state(last),w)
    write(out/'result.json',dict(status='EXACT_REPLAY_PASS_RECURRENCE_ONLY' if all_bits else 'REPLAY_AGREEMENT_PASS_RECURRENCE_ONLY',
        segments=K,period_ms=T,times_ms=[float(t*dt) for t in targets],source_node_checks=checks,
        original_checkpoint_comparison=checks[-1],all_original_saved_nodes_bitwise=bool(all_bits),
        full_state_errors=closure,all_Z_held=True,all_M_dynamic=True,seconds=time.time()-start,model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()));log('REPARTITION COMPLETE',closure['combined_relative_rms'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--segments',type=int,required=True,choices=[6,9,12,18]);p.add_argument('--device',type=int,default=1)
    main(p.parse_args())

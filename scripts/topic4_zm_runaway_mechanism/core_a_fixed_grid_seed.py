"""Repartition a saved recurrence using one uninterrupted original flow.

Intermediate nodes lie on exact original time steps; only the final return
uses fractional dense output. This repairs an observed interpolation floor
without changing any physical equation, resource field or root gate.
"""
from common import np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn
from onset_segment_flow import fixed_time
from onset_period_return import dynamical_state,errors
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,os,time


def main(a):
    parent=Path(a.parent).resolve();out=Path(a.destination).resolve()
    out.mkdir(parents=True,exist_ok=True);assert not (out/'jobs.json').exists()
    state_name=getattr(a,'state_name','accepted_node00.npz')
    period_name=getattr(a,'period_name','accepted_period.json')
    meta=read(parent/period_name);T=meta['period_ms'];K=a.segments
    source_contract=read(parent.parent/'contract.json')
    dt=getattr(a,'dt',.05)
    assert source_contract['dt_ms']==dt,'Seed must retain the saved state time mesh'
    write(out/'contract.json',dict(source=str(parent),source_state=str(parent/state_name),
        period_source=str(parent/period_name),segments=K,dt_ms=dt,
        question='Remove independently established negative-history interpolation floor at quiet shooting boundaries.',
        method='Starting from accepted node0, run the original full spatial model uninterrupted. First K-1 boundaries are fixed equal integer-step durations; final duration is the remaining period. Other accepted nodes are not interpolated or clipped. Each new node is an actual state, with Z unchanged and every E M dynamic. An independent fixed_time run checks one intermediate node.',
        interpretation='Numerical seed only. Its closure and spectral stability must be calculated anew; not a native trajectory replay claim or a periodic certificate.',model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()));start=time.time()
    e=build(a.device,dt);base=dict(np.load(parent/state_name))
    assert base['history'].shape==e.local.history.shape
    h=round(T/K/e.dt)*e.dt;steps=round(h/e.dt)
    assert T-(K-1)*h>2*e.dt
    A=SectionReturn(base,e,T);restore(e,base);nodes=[base]
    np.savez_compressed(out/'node00.npz',**base)
    for j in range(1,K):
        whole,tail=divmod(steps,round(10/e.dt))
        for _ in range(whole):e.chunk()
        for _ in range(tail):e.step()
        e.cp.cuda.get_current_stream().synchronize();state=capture(e)
        assert np.array_equal(state['syn'][5],base['syn'][5])
        assert np.all(state['parameters'][19]==0) and np.all(state['parameters'][20]==1)
        assert A.admissible(A.c.pack(state))
        nodes.append(state);np.savez_compressed(out/f'node{j:02d}.npz',**state)
        log('FIXED GRID ACTUAL NODE',j,j*h)
    expected,_=fixed_time(A,A.xref,h);actual=A.c.pack(nodes[1])
    parity=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-15))
    assert parity<1e-10,('Original intermediate-state parity',parity)
    final,_=fixed_time(A,A.xref,T)
    closure=errors(dynamical_state(base),dynamical_state(A.state(final)),e.s.sizes/e.s.sizes.sum())
    np.savez_compressed(out/'terminal.npz',**A.state(final))
    write(out/'result.json',dict(status='UNINTERRUPTED_FIXED_GRID_SEED',segments=K,period_ms=T,dt_ms=dt,
        fixed_grid_segment_ms=h,durations_ms=[h]*(K-1)+[T-(K-1)*h],
        all_nodes_on_actual_steps=True,all_nodes_admissible=True,single_pass_parity_relative=parity,
        full_state_errors=closure,final_cubic_endpoint_admissible=bool(A.admissible(final)),
        all_Z_held=True,all_M_dynamic=True,seconds=time.time()-start,model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))
    log('FIXED GRID SEED',T,h,closure['combined_relative_rms'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--destination',required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--segments',type=int,default=6)
    p.add_argument('--dt',type=float,choices=[.05,.025,.0125],default=.05)
    p.add_argument('--state-name',choices=['accepted_node00.npz','accepted_state.npz'],default='accepted_node00.npz')
    p.add_argument('--period-name',choices=['accepted_period.json','accepted_update.json'],default='accepted_period.json')
    main(p.parse_args())

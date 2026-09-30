"""Compare scheduling forms of identical full-state nominal evolution.

No physical coefficient, initial bit, precision or acceptance gate changes.
"""
from common import OUT,np,write,log
from onset_state_continuation import build
from fine_rate_frozen_Z_fields import capture,restore
from onset_tangent_cuda import Tangent
from onset_variational_return import Coordinates
from onset_period_return import errors,dynamical_state
import argparse,os,time


def main(device):
    p=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_native_entry_midpoint/late_recurrence'
    d=p/'execution_order_diagnostic';d.mkdir(exist_ok=True);assert not (d/'jobs.json').exists()
    write(d/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    e=build(device);base=dict(np.load(p/'multiple_three_burst/iteration00/node01.npz'))
    c=Coordinates(base,e.s);v=np.random.default_rng(92503).normal(size=c.size);v/=np.linalg.norm(v)
    restore(e,base);t=Tangent(e);t.graph();steps=2636;rows=[];ref=None;start=time.time()
    modes=['nominal_graph','nominal_graph','nominal_sync_steps','tangent_graph','tangent_graph','tangent_sync_steps']
    write(d/'contract.json',dict(source=str(p/'multiple_three_burst/iteration00/node01.npz'),steps=steps,dt_ms=e.dt,modes=modes,
        question='Does the nominal discrepancy depend on first use,10ms CUDA graphs, per-step synchronization or the presence of a passive tangent?',
        scope='Diagnosis only; failed source and derivative checks are retained, no acceptance criterion is amended. Every mode starts from identical complete physical bytes.'))
    for j,mode in enumerate(modes):
        restore(e,base)
        if mode.startswith('tangent'):c.set_tangent(t,v)
        actual=capture(e);assert all(a.tobytes()==actual[k].tobytes() for k,a in base.items())
        if mode.endswith('graph'):
            for _ in range(steps//200):(t.chunk() if mode.startswith('tangent') else e.chunk())
            for _ in range(steps%200):(t.step() if mode.startswith('tangent') else e.step())
            e.cp.cuda.get_current_stream().synchronize()
        else:
            for _ in range(steps):
                (t.step() if mode.startswith('tangent') else e.step())
                e.cp.cuda.get_current_stream().synchronize()
        state=capture(e);assert int(state['clock'][0])==int(base['clock'][0])+steps
        if ref is None:ref=state
        row=dict(mode=mode,repeat_index=j,initial_bitwise=True,
            terminal_bitwise={k:bool(np.array_equal(a,ref[k])) for k,a in state.items()},
            errors=errors(dynamical_state(ref),dynamical_state(state),e.s.sizes/e.s.sizes.sum()),seconds=time.time()-start)
        rows.append(row);write(d/'progress.json',rows);np.savez_compressed(d/f'terminal{j:02d}.npz',**state)
        log('EXECUTION ORDER DIAGNOSTIC',mode,row['errors']['combined_relative_rms'],row['terminal_bitwise'])
    write(d/'result.json',dict(status='DIAGNOSTIC_COMPLETE',rows=rows,model_promoted=False));write(d/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

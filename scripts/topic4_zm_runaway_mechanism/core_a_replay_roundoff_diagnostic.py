"""Localize the retained machine-precision replay discrepancy, without waiver."""
from common import OUT,np,read,write,log
from onset_state_continuation import build,initialize
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import restore,capture
import core_a_native_entry_midpoint as point
import argparse


def main(device):
    dest=point.DEST/'replay_diagnosis';dest.mkdir(exist_ok=True)
    e=build(device);c=read(point.DEST/'conditions.json')[point.LABEL]
    initialize(e,c);base=capture(e);w=e.s.sizes/e.s.sizes.sum();rows=[];states=[]
    # Always verify the actual restored bits, including signed zeros.
    def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    for mode in ['graph','graph','manual','manual','graph_sync']:
        restore(e,base);actual=capture(e)
        assert all(bits(v,actual[k]) for k,v in base.items())
        if mode=='manual':
            for _ in range(round(10/e.dt)):e.step()
            e.cp.cuda.get_current_stream().synchronize();output=e.output.get()
        else:
            if mode=='graph_sync':e.cp.cuda.runtime.deviceSynchronize()
            output=e.chunk()
        state=capture(e);states.append(state)
        row=dict(mode=mode,full_initial_bits=True,
            terminal_bits_vs_first={k:bits(v,states[0][k]) for k,v in state.items()},
            original_blocks_vs_first=errors(dynamical_state(states[0]),dynamical_state(state),w),
            expected_emitted_bitwise=bool(np.array_equal(output[:,0],output[:,1])),
            output_max_difference_vs_first=float(np.max(abs(state['output']-states[0]['output']))))
        rows.append(row);write(dest/f'device{device}.json',dict(status='DIAGNOSTIC_IN_PROGRESS',rows=rows))
        log('RESTORED REPLAY MODE',mode,row['output_max_difference_vs_first'],row['original_blocks_vs_first']['combined_relative_rms'])
    for j,state in enumerate(states):np.savez_compressed(dest/f'device{device}_terminal{j}.npz',**state)
    write(dest/f'device{device}.json',dict(status='REPLAY_DIAGNOSTIC_COMPLETE',rows=rows,
        scope='Diagnostic only. The prior failed exact replay remains failed; no implementation gate or bifurcation certificate is changed.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

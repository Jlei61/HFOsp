"""Retain failed bit equality and test the measured FP64 replay floor.

This changes only the implementation replay criterion for the new midpoint.
No periodic closure, derivative, phase, mesh or bifurcation gate is changed.
"""
from common import np,read,write,log
from fine_rate_frozen_Z_fields import capture,restore
from onset_period_return import errors,dynamical_state
import core_a_native_entry_midpoint as point
import argparse


def main(device):
    dest=point.DEST
    prior=read(dest/'ordinary_replay_device1.json');assert prior['status']=='FAIL'
    diagnosis=read(dest/'replay_diagnosis/device1.json');assert diagnosis['status']=='REPLAY_DIAGNOSTIC_COMPLETE'
    assert max(r['original_blocks_vs_first']['combined_relative_rms'] for r in diagnosis['rows'])<1e-12
    assert all(r['full_initial_bits'] and r['expected_emitted_bitwise'] for r in diagnosis['rows'])
    amendment=dict(reason='Exact restored initial bits produce FP64 terminal variation about1.6--2.8e-15 in the original block norm, for both graph and manual integration; extra device synchronization does not remove it. Original failures and arrays are retained. Cause is not attributed. Test an explicit componentwise floating-point floor rather than repeatedly retrying until a bitwise pass.',
        scope='Only the new finite-trajectory implementation replay check. Original full initial-state identity, frozenZ, emitted/expected equality and model coefficients remain exact. All periodic closure, actual derivative, independent phase/mesh and bifurcation criteria unchanged.',
        gate='Three10ms restored repeats, initial state bits identical; each component absdifference<=1e-14+1e-12*abs(reference); six-block combined<1e-12 and everyblock<1e-11; clocks, parameter rows and Z exact.',
        scientific_effect='The measured replay floor is many orders below the unchanged periodic-root and phase gates. It neither establishes periodicity nor licenses a bifurcation label.',model_promoted=False)
    write(dest/'numerical_replay_contract.json',amendment)
    e=point.point.run.flow.build(device);c=read(dest/'conditions.json')[point.LABEL]
    z=point.point.run.flow.initialize(e,c);base=capture(e);w=e.s.sizes/e.s.sizes.sum()
    initial=dict(np.load(c['initial']));A=e.s.E&(e.s.geo['group_region']==0)
    bits=lambda a,b:a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    assert all(bits(v[:5],initial[k][:5]) if k=='syn' else bits(v,initial[k]) for k,v in base.items())
    assert bits(z[~A],initial['syn'][5,~A])
    rows=[];reference=None
    for j in range(3):
        if j:restore(e,base)
        current=capture(e);assert all(bits(v,current[k]) for k,v in base.items())
        q=e.chunk();terminal=capture(e)
        if reference is None:reference=terminal
        bound={k:bool(np.all(abs(v.astype(float)-reference[k].astype(float))<=1e-14+1e-12*abs(reference[k].astype(float)))) for k,v in terminal.items()}
        err=errors(dynamical_state(reference),dynamical_state(terminal),w)
        row=dict(repeat=j,initial_bits=True,terminal_bits=all(bits(v,reference[k]) for k,v in terminal.items()),componentwise_pass=bound,full_state_errors=err)
        rows.append(row);write(dest/'numerical_replay_checks.json',rows)
        assert np.isfinite(q).all() and np.array_equal(q[:,0],q[:,1]) and bits(terminal['syn'][5],z)
        assert bits(terminal['clock'],reference['clock']) and bits(terminal['parameters'],reference['parameters'])
        assert all(bound.values()) and err['combined_relative_rms']<1e-12 and max(v['relative_rms'] for v in err['blocks'].values())<1e-11,row
    write(dest/'implementation_check.json',dict(status='PASS',criterion='EXPLICIT_FP64_REPLAY_BOUND',prior_exact_equality='FAIL_RETAINED',rows=rows,
        only_core_A_Z_changed=True,full_initial_state_bitwise_except_core_A_Z=True,Z_held=True,M_dynamic=True,
        constant_mean_external=True,count_innovations_disabled=True,private_Q_retained=True,amendment=str(dest/'numerical_replay_contract.json'),model_promoted=False))
    log('MIDPOINT NUMERICAL REPLAY PASS',max(r['full_state_errors']['combined_relative_rms'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)

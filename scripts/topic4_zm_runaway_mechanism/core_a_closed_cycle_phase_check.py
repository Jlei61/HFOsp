"""Check one closed whole-state cycle from actual shifted flow states.

Starting phases are original integer-step states, never interpolated quiet
histories. A failed shifted closure remains a numerical qualification failure.
"""
from common import np, read, write, log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn
from onset_segment_flow import fixed_time
from onset_period_return import errors, dynamical_state
from fine_rate_frozen_Z_fields import restore, capture
from pathlib import Path
import argparse, os, time


def main(a):
    parent=Path(a.parent).resolve(); out=parent/'shifted_phase_check'
    out.mkdir(exist_ok=True); assert not (out/'jobs.json').exists()
    result=read(parent/'result.json'); contract=read(parent/'contract.json')
    assert result['status'] in ['NUMERICAL_MULTIPLE_SHOOTING_ROOT', 'NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT']
    row=result['iterations'][-1]; T=row['period_ms']; dt=contract['dt_ms']
    source=parent/f"iteration{row['iteration']:02d}"/'node00.npz'
    phases=[50.,150.,300.]; assert max(phases)<T
    write(out/'contract.json',dict(source=str(source),dt_ms=dt,period_ms=T,
        shifts_ms=phases,
        question='Does the same whole-cycle root return at its original period from three independently generated physical phases, including deep quiet and subsequent activity?',
        method='Generate the shifted states by one uninterrupted original integer-step flow. Preserve their actual physical clocks and every fast, covariance, input memory, M and lag-history value. From each, run the unchanged original fixed-time return at T with its existing cubic endpoint readout. No shifted root correction or state clipping.',
        acceptance='Existing independent-phase gates from core_a_periodic_validate.py: combined six-block closure<1e-6 and each block<1e-5. The tighter root-correction stopping tolerances1e-7/1e-6 are an additional diagnostic, not a newly tightened phase gate. Record negative endpoint history count and magnitude separately.',
        limits='Phase check at one dt only; independent corrected mesh, Floquet and critical crossing remain required. These phase choices are numerical checks, not three SNN samples.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid()); write(out/'jobs.json',jobs); started=time.time()
    try:
        e=build(a.device,dt); base=dict(np.load(source)); restore(e,base)
        w=e.s.sizes/e.s.sizes.sum(); previous=0.; states=[]
        for phase in phases:
            whole,tail=divmod(round((phase-previous)/dt),round(10/dt))
            for _ in range(whole): e.chunk()
            for _ in range(tail): e.step()
            e.cp.cuda.get_current_stream().synchronize(); state=capture(e)
            assert np.array_equal(state['syn'][5],base['syn'][5])
            assert int(state['clock'][0])-int(base['clock'][0])==round(phase/dt)
            assert np.all(state['parameters'][19]==0) and np.all(state['parameters'][20]==1)
            np.savez_compressed(out/f'shift{int(phase):03d}.npz',**state)
            states.append(state); previous=phase
        rows=[]
        for phase,state in zip(phases,states):
            A=SectionReturn(state,e,T); assert A.admissible(A.xref)
            y,_=fixed_time(A,A.xref,T); terminal=A.state(y)
            check=errors(dynamical_state(state),dynamical_state(terminal),w)
            maximum=max(q['relative_rms'] for q in check['blocks'].values())
            passed=check['combined_relative_rms']<1e-6 and maximum<1e-5
            rows.append(dict(phase_ms=phase,**check,pass_gate=bool(passed),
                passes_root_correction_tolerance=bool(check['combined_relative_rms']<1e-7 and maximum<1e-6),
                endpoint_negative_history_count=int((terminal['history']<0).sum()),
                endpoint_minimum_history_per_ms=float(terminal['history'].min())))
            write(out/'progress.json',rows); log('CLOSED CYCLE SHIFT',phase,rows[-1])
        status='SHIFTED_PHASE_CLOSURE_PASS' if all(q['pass_gate'] for q in rows) else 'SHIFTED_PHASE_CLOSURE_FAIL'
        write(out/'result.json',dict(status=status,rows=rows,seconds=time.time()-started,
            period_ms=T,dt_ms=dt,bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status); write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc)); write(out/'jobs.json',jobs); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('parent'); p.add_argument('--device',type=int,default=1)
    main(p.parse_args())

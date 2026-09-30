"""Taylor test in the exact saved Newton direction, using original flow.

Random-direction checks do not bound nonlinear remainders in a strongly
amplified solver direction. No parameter, physical state, or gate is changed.
"""
from common import np, read, write, log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn, CubicSectionDerivative
from onset_segment_flow import fixed_time
from pathlib import Path
import argparse, os, time


def main(a):
    parent=Path(a.parent).resolve(); out=parent/'saved_newton_direction_check'
    out.mkdir(exist_ok=True); assert not (out/'jobs.json').exists()
    contract=read(parent/'contract.json'); meta=read(parent/'iterations.json')[0]
    saved=np.load(parent/'iteration00/first_proposal_direction.npz')
    base=dict(np.load(parent/'latest_state.npz'))
    write(out/'contract.json',dict(source=str(parent),dt_ms=contract['dt_ms'],
        method='Original uncached full-state implicit Poincare derivative and actual nonlinear returns along the exact saved first Newton proposal, at successively smaller amplitudes. Compare fixed-time and section-return directional Taylor errors. No projection or clipping.',
        question='Does failure in this saved Newton direction come from the linearization, nonlinear transient amplification, or section-time selection?',
        model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid()); write(out/'jobs.json',jobs); start=time.time()
    try:
        e=build(a.device,contract['dt_ms'])
        A=CubicSectionReturn(base,e,meta['period_ms'],contract['return_halfwidth_ms'])
        A.c.scale=saved['coordinate_scale'].copy(); A.c.weight=saved['coordinate_weight'].copy()
        A.xref=saved['source'].copy(); A.normal=saved['normal'].copy()
        x=A.c.pack(base); delta=saved['delta']; assert np.linalg.norm(x-A.xref)<1e-12
        y,info=A(x); slope=A.last_time_slope.copy(); T=info['period_ms']
        J=CubicSectionDerivative(A,x,T,slope); dy=J(delta)
        dT=J.last_return_time_derivative; fixed_derivative=dy-slope*dT
        baseline,_=fixed_time(A,x,T)
        parity=float(np.linalg.norm(baseline-y)/np.linalg.norm(y)); assert parity<1e-10
        head=dict(period_ms=T,section_speed=info['section_speed'],step_norm=float(np.linalg.norm(delta)),
            section_derivative_norm=float(np.linalg.norm(dy)),fixed_derivative_norm=float(np.linalg.norm(fixed_derivative)),
            predicted_period_change_ms=dT,fixed_vs_section_parity=parity)
        write(out/'baseline.json',head); log('NEWTON DIRECTION BASELINE',head)
        rows=[]
        for eps in [1e-4,1e-3,1e-2,.1,1.]:
            z=x+eps*delta; assert A.admissible(z)
            q,returned=A(z); fixed,_=fixed_time(A,z,T)
            row=dict(epsilon=eps,period_ms=returned['period_ms'],section_speed=returned['section_speed'],
                predicted_period_ms=T+eps*dT,
                section_Taylor_error=float(np.linalg.norm((q-y)/eps-dy)/np.linalg.norm(dy)),
                fixed_time_Taylor_error=float(np.linalg.norm((fixed-baseline)/eps-fixed_derivative)/np.linalg.norm(fixed_derivative)),
                return_residual_ratio=float(np.linalg.norm(q-z)/np.linalg.norm(y-x)))
            rows.append(row); write(out/'progress.json',rows); log('NEWTON DIRECTION CHECK',row)
        np.savez_compressed(out/'direction.npz',delta=delta,section_derivative=dy,fixed_derivative=fixed_derivative)
        write(out/'result.json',dict(status='DIRECTIONAL_TAYLOR_DIAGNOSTIC_COMPLETE',baseline=head,rows=rows,
            seconds=time.time()-start,bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE'); write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc)); write(out/'jobs.json',jobs); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('parent'); p.add_argument('--device',type=int,required=True)
    main(p.parse_args())

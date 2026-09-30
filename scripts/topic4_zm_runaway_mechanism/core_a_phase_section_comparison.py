"""Compare two phase sections at one unchanged full-state iterate."""
from common import np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import regional_M_section
from onset_cubic_section import CubicSectionReturn
from onset_period_return import errors,dynamical_state
from pathlib import Path
import argparse,os


def main(a):
    source=Path(a.source).resolve();out=Path(a.destination).resolve()
    out.mkdir(parents=True,exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'contract.json',dict(source=str(source),dt_ms=a.dt,period_seed_ms=a.period,
        question='Does a phase marker based on the evolving CoreA mean M give a useful local return from the same nonconverged fine-grid cycle seed?',
        method='Two original nonlinear cubic section returns from an identical complete state: full-flow phase plane and cell-weighted CoreA mean-M plane. No state correction, M clamp, coefficient change, or different resource field. Positive orientation is chosen from the actual original initial flow.',
        maximum_halfwidth_ms=20,model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    try:
        e=build(a.device,a.dt);base=dict(np.load(source));A=CubicSectionReturn(base,e,a.period,20.)
        x=A.xref;rows=[]
        assert np.all(base['parameters'][19]==0) and np.all(base['parameters'][20]==1)
        for section in ['flow','CoreA_dynamic_M']:
            if section!='flow':regional_M_section(A,0)
            try:
                y,meta=A(x)
                check=errors(dynamical_state(base),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
                row=dict(section=section,status='RETURN_MEASURED_NOT_ROOT',**meta,**check)
                np.savez_compressed(out/(section+'_endpoint.npz'),**A.state(y))
            except RuntimeError as exc:row=dict(section=section,status='LOCAL_SECTION_WINDOW_MISS',error=str(exc))
            rows.append(row);write(out/'progress.json',rows);log('PHASE SECTION COMPARISON',row)
        write(out/'result.json',dict(status='PHASE_SECTION_DIAGNOSTIC_COMPLETE',rows=rows,
            M_dynamic=True,bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--dt',type=float,required=True);p.add_argument('--period',type=float,required=True)
    p.add_argument('--device',type=int,required=True);main(p.parse_args())

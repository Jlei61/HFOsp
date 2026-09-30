"""Test a new positive solver coordinate using an already computed direction."""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_poincare_corrector import SectionReturn
from onset_period_return import errors,dynamical_state
from core_a_positive_newton_coordinates import retract
import argparse,os
from pathlib import Path

PARENT=OUT/'core_a_bifurcation_type_20260924/near_returns/mid_lower/periodic_newton_core_A'


def main(device,iteration,parent=None):
    global PARENT
    if parent:PARENT=Path(parent).resolve()
    source=PARENT/f'iteration{iteration:02d}';data=np.load(source/'newton_direction.npz')
    out=source/'positive_coordinate_probe';out.mkdir(exist_ok=True);assert not (out/'jobs.json').exists()
    contract=read(PARENT/'contract.json');initial=read(PARENT/'iterations.json')[iteration]
    write(out/'contract.json',dict(question='Do positivity-preserving Newton coordinates improve the actual unchanged return residual when linear proposals hit physical bounds?',
        numerical_only='Negative increments of positive coordinates use x/(1-d/x), positive increments are linear. The first derivative at zero step equals the ordinary Newton direction. The exact same phase plane is restored multiplicatively in its participating current-rate coordinates. Every candidate passes original full-state physical constraints, then the actual unmodified flow is evaluated.',
        source=str(source/'newton_direction.npz'),trials=[1.,.5,.25,.125,.0625],model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    Return=SectionReturn
    if contract.get('interpolation')=='cubic':
        from onset_cubic_section import CubicSectionReturn
        Return=CubicSectionReturn
    dt=contract.get('dt_ms',.05);e=build(device,dt)
    base=regrid_state(np.load(contract['source']),e,contract.get('source_dt_ms',.05))
    if contract.get('target_D_A') is not None:
        from core_a_equilibrium_branch import Family
        family=Family(e.s);z,_=family.field(contract['target_D_A'])
        assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
        base['syn'][5]=z
    A=Return(base,e,initial['period_ms'],20.)
    A.normal=data['normal'];x=data['source'];delta=data['delta'];rows=[]
    # First-order consistency check; only solver algebra, no altered flow.
    algebra=[]
    for eps in [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
        candidate=retract(A,x,delta,eps)
        algebra.append(dict(epsilon=eps,relative_direction_error=float(np.linalg.norm((candidate-x)/eps-delta)/np.linalg.norm(delta)) if candidate is not None else float('inf')))
        if algebra[-1]['relative_direction_error']<1e-3:break
    write(out/'coordinate_check.json',algebra)
    assert algebra[-1]['relative_direction_error']<1e-3
    reference_norm=initial['coordinate_residual']*np.linalg.norm(x)
    for alpha in [1.,.5,.25,.125,.0625]:
        candidate=retract(A,x,delta,alpha)
        if candidate is None:rows.append(dict(alpha=alpha,status='INADMISSIBLE'));continue
        try:value,meta=A(candidate)
        except RuntimeError as exc:
            rows.append(dict(alpha=alpha,status='SECTION_WINDOW_MISS',error=str(exc)));continue
        err=errors(dynamical_state(A.state(candidate)),dynamical_state(A.state(value)),e.s.sizes/e.s.sizes.sum())
        ratio=float(np.linalg.norm(value-candidate)/reference_norm)
        row=dict(alpha=alpha,status='EVALUATED',residual_ratio=ratio,**meta,**err);rows.append(row);log('POSITIVE NEWTON PROBE',row)
        if ratio<1:
            np.savez_compressed(out/('candidate'+str(alpha).replace('.','p')+'.npz'),**A.state(candidate))
            if ratio<.75:break
    write(out/'result.json',dict(status='PROBE_COMPLETE_NOT_AN_ORBIT',rows=rows,model_promoted=False));write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--iteration',type=int,default=0);p.add_argument('--parent');a=p.parse_args();main(a.device,a.iteration,a.parent)

"""Under-relaxed, accelerated numerical solution of the full return equation.

The integration itself is unchanged. Convex under-relaxation is a solver for
P(X)=X, not feedback added to the network. Its fixed points are identical.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_poincare_corrector import SectionReturn,regional_rate_section
from onset_period_return import dynamical_state,errors
from pathlib import Path
import argparse,os,time

ROOT=OUT/'core_a_bifurcation_type_20260924/relaxed_return_corrector'


def main(source,period,device,name,maximum,dt=.05,source_dt=.05,section='core_A',relax=.15,m_relax=.75,halfwidth=20.,target_D=None,interpolation='linear',positive_anderson=False,orientation=None,anderson_ridge=1e-12,anderson_clip=False):
    out=ROOT/name;out.mkdir(parents=True,exist_ok=True);assert not (out/'jobs.json').exists()
    base={k:v.copy() for k,v in np.load(source).items()};np.savez_compressed(out/'source_snapshot.npz',**base)
    write(out/'contract.json',dict(question='Can a positivity-preserving numerical return iteration close the actual local bursting orbit more effectively than damped Newton?',
        source=str(Path(source).resolve()),period_seed_ms=period,dt_ms=dt,source_dt_ms=source_dt,target_D_A=target_D,
        section=section,section_orientation=orientation,interpolation=interpolation,positive_anderson_retraction=positive_anderson,relaxation=relax,M_relaxation=m_relax,return_halfwidth_ms=halfwidth,anderson_ridge=anderson_ridge,anderson_clip=anderson_clip,
        equations='Same full spatial conditional drift, all Z held at the same original Core-A-only field, all M dynamic. Every P evaluation integrates original dynamics.',
        solver='G(X)=X+W*(P(X)-X), with declared positive fast/history and M relaxation. This is a numerical fixed-point iteration; P=identity iff G=identity. A Core A rate section preserves phase under block-wise convex mixing; a flow-normal section requires uniform relaxation. Anderson memory6 is allowed only if full original physical admissibility holds and coefficientL1<=20. Otherwise use the convex G. Reject a proposal that increases actual residual by more than2x, fall back and clear memory.',
        reason='Current Newton proposals hit nonnegative M/history bounds. Residual-seeded linear diagnostics suggest a large negative return direction, which motivates damping as a solver choice, not a certified Floquet multiplier.',
        maximum_actual_returns=maximum,closure_gate='combined<1e-7 and allsixblocks<1e-6; independent phase/mesh and Floquet still required',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),returns=0);write(out/'jobs.json',jobs);start=time.time()
    e=build(device,dt=dt);base=regrid_state(base,e,source_dt)
    if target_D is not None:
        from core_a_equilibrium_branch import Family
        family=Family(e.s);z,tm=family.field(target_D)
        assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
        base['syn'][5]=z
        write(out/'target_field.json',dict(D_A=target_D,native_field_coordinate_ms=tm,
            outside_A_unchanged=True,all_M_dynamic=True))
    Return=SectionReturn
    if interpolation=='cubic':
        from onset_cubic_section import CubicSectionReturn
        Return=CubicSectionReturn
    A=Return(base,e,period,halfwidth)
    if section in ['core_A','core_B']:
        regional_rate_section(A,0 if section=='core_A' else 1,orientation)
    else:assert section=='flow' and m_relax==relax
    x=A.xref.copy();history=[];rows=[];fallback=None;oldnorm=None;best=np.inf;status='RETURN_BUDGET_NOT_AN_ORBIT'
    try:
        for k in range(maximum):
            try:y,meta=A(x)
            except RuntimeError as exc:
                if fallback is None:raise
                rows.append(dict(return_index=k,status='SECTION_MISS_FALLBACK',error=str(exc)));write(out/'iterations.json',rows)
                x=fallback;fallback=None;history=[];continue
            f=y-x;norm=float(np.linalg.norm(f));err=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
            row=dict(return_index=k,**meta,**err,coordinate_residual=norm/np.linalg.norm(A.xref));rows.append(row)
            log('RELAXED CORE A RETURN',k,meta['period_ms'],err['combined_relative_rms'],row['coordinate_residual'])
            jobs['returns']=k+1;jobs['last_residual']=err['combined_relative_rms'];write(out/'jobs.json',jobs)
            if norm<best:
                best=norm;np.savez_compressed(out/'best_state.npz',**A.state(x));write(out/'best_coordinates.json',dict(period_ms=meta['period_ms'],**err,coordinate_residual=row['coordinate_residual']))
            if err['combined_relative_rms']<1e-7 and max(v['relative_rms'] for v in err['blocks'].values())<1e-6:
                status='NUMERICAL_PERIODIC_ROOT';np.savez_compressed(out/'root_state.npz',**A.state(x));break
            if fallback is not None and oldnorm is not None and norm>2*oldnorm:
                row['next_method']='REJECT_ANDERSON_USE_PREVIOUS_CONVEX';x=fallback;fallback=None;history=[]
                write(out/'iterations.json',rows);continue
            A.period=meta['period_ms'];change=relax*f
            change.reshape(-1,e.s.P)[4]=m_relax*f.reshape(-1,e.s.P)[4]
            g=x+change;assert A.admissible(g) and abs(A.normal@(g-A.xref))<1e-9
            history.append((g.copy(),change.copy()));history=history[-6:]
            proposal=g;method='CONVEX_RELAXATION';coeff=[1.]
            if len(history)>1:
                F=[v[1] for v in history];gram=np.array([[float(a@b) for b in F] for a in F]);m=len(F)
                gram+=np.eye(m)*max(float(np.trace(gram)),1e-30)*anderson_ridge
                matrix=np.zeros((m+1,m+1));matrix[:m,:m]=gram;matrix[m,:m]=1;matrix[:m,m]=1
                rhs=np.r_[np.zeros(m),1.];coef=np.linalg.solve(matrix,rhs)[:m]
                row['unconstrained_coefficient_l1']=float(np.sum(abs(coef)))
                if anderson_clip and np.sum(abs(coef))>20:
                    theta=19./(np.sum(abs(coef))-1.)
                    coef*=theta;coef[-1]+=1.-theta
                    assert abs(coef.sum()-1)<1e-8 and np.sum(abs(coef))<=20+1e-8
                    row['coefficient_blend_to_current_convex']=float(theta)
                candidate=sum(c*h[0] for c,h in zip(coef,history))
                retracted=False
                row['anderson_admissible_before_retraction']=bool(A.admissible(candidate))
                if positive_anderson and np.sum(abs(coef))<=20+1e-8 and not A.admissible(candidate):
                    from core_a_positive_newton_coordinates import retract
                    candidate=retract(A,x,candidate-x,1.)
                    retracted=True
                row['anderson_admissible_after_retraction']=bool(candidate is not None and A.admissible(candidate))
                if candidate is not None and np.sum(abs(coef))<=20+1e-8 and A.admissible(candidate) and abs(A.normal@(candidate-A.xref))<1e-9:
                    proposal=candidate;method='ANDERSON6_ON_RELAXED_MAP'+('_POSITIVE_RETRACTION' if retracted else '');coeff=coef.tolist()
            row['next_method']=method;row['mixing_coefficients']=coeff;write(out/'iterations.json',rows)
            fallback=g.copy() if method.startswith('ANDERSON') else None;oldnorm=norm;x=proposal
        write(out/'iterations.json',rows);write(out/'result.json',dict(status=status,rows=rows,dt_ms=dt,elapsed_seconds=time.time()-start,
            best=read(out/'best_coordinates.json'),scope='Numerical solver only; its mixing has no biological interpretation or effect on the original flow.',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--period',type=float,required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--name',required=True);p.add_argument('--returns',type=int,default=40)
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--source-dt',type=float,default=.05)
    p.add_argument('--section',choices=['core_A','core_B','flow'],default='core_A');p.add_argument('--relax',type=float,default=.15);p.add_argument('--m-relax',type=float,default=.75);p.add_argument('--halfwidth',type=float,default=20.)
    p.add_argument('--target-D',type=float)
    p.add_argument('--interpolation',choices=['linear','cubic'],default='linear')
    p.add_argument('--positive-anderson',action='store_true')
    p.add_argument('--orientation',type=int,choices=[-1,1])
    p.add_argument('--anderson-ridge',type=float,default=1e-12);p.add_argument('--anderson-clip',action='store_true')
    a=p.parse_args();main(a.source,a.period,a.device,a.name,a.returns,a.dt,a.source_dt,a.section,a.relax,a.m_relax,a.halfwidth,a.target_D,a.interpolation,a.positive_anderson,a.orientation,a.anderson_ridge,a.anderson_clip)

"""Independent phase and time-step check of a solved local periodic root."""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_poincare_corrector import SectionReturn
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,os

PARENT=OUT/'core_a_bifurcation_type_20260924/near_returns/mid_lower/periodic_newton'


def main(device,dt,parent=None):
    global PARENT
    if parent:PARENT=Path(parent).resolve()
    result=read(PARENT/'result.json')
    numerical_method=read(PARENT/'contract.json').get('numerical_method','old_endpoint')
    whole=result['status'] in ['NUMERICAL_MULTIPLE_SHOOTING_ROOT','NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT']
    assert whole or result['status']=='NUMERICAL_PERIODIC_ROOT'
    Return=SectionReturn
    if whole or read(PARENT/'contract.json').get('interpolation')=='cubic':
        from onset_cubic_section import CubicSectionReturn
        Return=CubicSectionReturn
    DEST=PARENT/('independent_dt'+str(dt).replace('.','p'));DEST.mkdir(exist_ok=True)
    assert not (DEST/'jobs.json').exists()
    write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    if whole:
        row=result['iterations'][-1]
        source=PARENT/f"iteration{row['iteration']:02d}"/'node00.npz'
        T=row['period_ms']
    else:
        source=PARENT/('root_state.npz' if (PARENT/'root_state.npz').exists() else 'latest_state.npz')
        T=result.get('period_ms')
        if T is None:T=result.get('rows',result.get('iterations'))[-1]['period_ms']
    source_dt=result.get('dt_ms',read(PARENT/'contract.json')['dt_ms'])
    write(DEST/'contract.json',dict(source=str(source),source_dt_ms=source_dt,dt_ms=dt,
        numerical_method=numerical_method,
        seed_period_ms=T,phases=[0.,.25,.5],
        method='Existing core_a_periodic_validate procedure. Generate each phase by original integer-step flow, then find its positively oriented full-state Poincare return with the existing cubic actual-step interpolant. Report return-period drift as well as whole-state closure. This is distinct from an additional fixed-T interpolation diagnostic.',
        acceptance='Historical independent-phase gates: combined<1e-6 and each physical block<1e-5. Root-correction tolerances and neutral/stability gates are unchanged. Uncorrected refined-mesh returns are diagnostics, not absence of a cycle.',model_promoted=False))
    if numerical_method=='exponential_midpoint':
        from onset_exponential_midpoint import ExponentialMidpointEngine
        from check_spatial_midpoint_convergence import conservative_history
        e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
        base,qa=conservative_history(dict(np.load(source)),e,source_dt)
        write(DEST/'initial_history_qa.json',qa)
    else:
        assert numerical_method=='old_endpoint'
        e=build(device,dt=dt);base=regrid_state(np.load(source),e,source_dt)
    rows=[]
    for phase in [0.,.25,.5]:
        restore(e,base);steps=round(phase*T/e.dt);chunks,tail=divmod(steps,round(10/e.dt))
        for _ in range(chunks):e.chunk()
        for _ in range(tail):e.step()
        e.cp.cuda.get_current_stream().synchronize();seed=capture(e)
        A=Return(seed,e,T,5.);x=A.xref;y,meta=A(x)
        err=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
        row=dict(phase=phase,actual_shift_ms=steps*e.dt,**meta,**err,
                 period_difference_ms=meta['period_ms']-T);rows.append(row)
        log('CORE A INDEPENDENT PERIOD',dt,row);write(DEST/'progress.json',rows)
    strict=all(r['combined_relative_rms']<1e-6 and max(x['relative_rms'] for x in r['blocks'].values())<1e-5 for r in rows)
    write(DEST/'result.json',dict(status='PHASE_CLOSURE_PASS' if strict else 'PHASE_OR_MESH_CLOSURE_NOT_YET_PASSED',dt_ms=dt,rows=rows,
        scope='Same numerical root at multiple phases; refinement begins with physical-lag-preserving history interpolation. An uncorrected refined residual alone is not absence of an orbit. No Floquet or bifurcation certificate.',model_promoted=False))
    write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--dt',type=float,default=.05);p.add_argument('--parent');a=p.parse_args();main(a.device,a.dt,a.parent)

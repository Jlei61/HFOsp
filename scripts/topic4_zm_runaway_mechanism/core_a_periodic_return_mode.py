"""Verify one real eigenmode of a numerically closed full-state section map.

An unstable eigenmode suffices to disprove stability of this numerical root;
a subunit power iterate does not certify absence of other unstable modes.
Physical periodic/Floquet attribution remains gated by phase and mesh checks.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn,regional_rate_section
from onset_shooting_newton import SectionDerivative
from onset_period_return import dynamical_state,errors
from pathlib import Path
import argparse,os,time


def main(parent,device,maximum,cached=False):
    parent=Path(parent).resolve();r=read(parent/'result.json');assert r['status']=='NUMERICAL_PERIODIC_ROOT'
    out=parent/('return_mode_cached' if cached else 'return_mode');out.mkdir(exist_ok=True);assert not (out/'jobs.json').exists()
    contract=read(parent/'contract.json');dt=contract['dt_ms'];T=r.get('rows',r.get('iterations'))[-1]['period_ms']
    Return,Derivative=SectionReturn,SectionDerivative
    if contract.get('interpolation')=='cubic':
        from onset_cubic_section import CubicSectionReturn,CubicSectionDerivative
        Return,Derivative=CubicSectionReturn,CubicSectionDerivative
        if cached:
            from onset_cached_tangent import CachedCubicSectionDerivative
            qa=OUT/'core_a_bifurcation_type_20260924/numerical_checks'/('exact_cached_tangent_fine' if dt==.025 else 'exact_cached_tangent')/'result.json'
            assert read(qa)['status']=='PASS' and read(qa)['dt_ms']==dt
            Derivative=CachedCubicSectionDerivative
    else:assert not cached,'Exact cached implementation requires cubic dense output'
    source=parent/('root_state.npz' if (parent/'root_state.npz').exists() else 'latest_state.npz')
    write(out/'contract.json',dict(source=str(source),dt_ms=dt,period_ms=T,cached_derivative=cached,
        method='Power iteration of actual full delayed-state Poincare derivative, with the derivative of return time. Signed Rayleigh value and direct full-space eigen-residual checked at each step; require two consecutive residuals<1e-7. This verifies at most one real eigenpair, not an entire spectrum.',
        phase_and_mesh='Retain independent validation status. No physical Floquet/bifurcation promotion from this fixed-step section map alone.',maximum_returns=maximum,model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);start=time.time()
    e=build(device,dt=dt);base={k:v.copy() for k,v in np.load(source).items()};A=Return(base,e,T,5.)
    if contract.get('section','core_A') in ['core_A','core_B']:
        regional_rate_section(A,0 if contract.get('section','core_A')=='core_A' else 1,contract.get('section_orientation'))
    x=A.xref;y,meta=A(x);er=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
    assert er['combined_relative_rms']<1e-6
    J=Derivative(A,x,meta['period_ms'],A.last_time_slope)
    rng=np.random.default_rng(92407);v=rng.normal(size=A.c.size);v.reshape(-1,e.s.P)[4,~e.s.E]=0
    v-=A.normal*(A.normal@v);v/=np.linalg.norm(v);rows=[];passed=0;status='NO_REAL_EIGENPAIR_CERTIFIED'
    for k in range(maximum):
        q=J(v);value=float(v@q);res=float(np.linalg.norm(q-value*v));rel=res/max(abs(value),1.)
        row=dict(iteration=k+1,real_multiplier=value,eigen_residual=res,relative_eigen_residual=rel,
            phase_leakage=float(abs(A.normal@q)),norm_gain=float(np.linalg.norm(q)))
        rows.append(row);write(out/'progress.json',rows);log('CORE A RETURN MODE',row)
        passed=passed+1 if rel<1e-7 else 0
        if passed>=2:
            # The last product is independent of the preceding eigenvalue
            # estimate; retain its vector and full actual derivative product.
            np.savez_compressed(out/'verified_mode.npz',vector=v,product=q,normal=A.normal,multiplier=value,
                coordinate_scale=A.c.scale,coordinate_weight=A.c.weight)
            status='ONE_REAL_SECTION_MAP_EIGENPAIR_VERIFIED';break
        v=q/np.linalg.norm(q)
    write(out/'result.json',dict(status=status,dt_ms=dt,period_ms=meta['period_ms'],rows=rows,
        numerical_root_unstable=bool(status.startswith('ONE_REAL') and abs(value)>1+1e-6),
        physical_Floquet_status='PENDING_INDEPENDENT_PHASE_AND_MESH',seconds=time.time()-start,model_promoted=False))
    jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0);p.add_argument('--returns',type=int,default=20)
    p.add_argument('--cached-derivative',action='store_true')
    a=p.parse_args();main(a.parent,a.device,a.returns,a.cached_derivative)

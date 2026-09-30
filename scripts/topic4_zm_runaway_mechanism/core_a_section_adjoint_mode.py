"""Numerical left/right return mode, not a global basin or crisis certificate."""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_poincare_corrector import regional_rate_section
from onset_cached_tangent import CachedCubicSectionDerivative
from onset_cached_adjoint import CachedCubicSectionAdjoint
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import capture
from pathlib import Path
import argparse,os,time


def main(parent,device):
    base=OUT/'core_a_bifurcation_type_20260924'
    assert read(base/'numerical_checks/full_cached_adjoint/result.json')['status']=='PASS'
    parent=Path(parent).resolve();result=read(parent/'result.json');assert result['status']=='NUMERICAL_PERIODIC_ROOT'
    contract=read(parent/'contract.json');assert contract['interpolation']=='cubic'
    out=parent/'section_adjoint_mode';out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'contract.json',dict(source=str(parent),
        method='Exact transpose of full cached cubic Poincare derivative, including return time and all four dense-output samples. Two whole-period bilinear identities precede left-eigenvector power iteration. Independently verify both right and left full-operator residuals.',
        scope='Numerical local unstable coordinate at an existing shooting root. Not proof of a global stable manifold, separatrix, collision, crisis, spectral completeness or native onset. Existing independent phase/mesh gates remain unchanged.',model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()));begin=time.time()
    T=result.get('period_ms')
    if T is None:T=result.get('rows',result.get('iterations'))[-1]['period_ms']
    e=build(device,contract['dt_ms'])
    state=parent/('root_state.npz' if (parent/'root_state.npz').exists() else 'latest_state.npz')
    A=CubicSectionReturn(dict(np.load(state)),e,T,5.)
    if contract.get('section','core_A') in ['core_A','core_B']:
        regional_rate_section(A,0 if contract.get('section','core_A')=='core_A' else 1,contract.get('section_orientation'))
    x=A.xref;y,meta=A(x)
    closure=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
    assert closure['combined_relative_rms']<1e-6
    J=CachedCubicSectionDerivative(A,x,meta['period_ms'],A.last_time_slope);JT=CachedCubicSectionAdjoint(J)
    rng=np.random.default_rng(92499);checks=[]
    for j in range(2):
        v=rng.normal(size=x.size);w=rng.normal(size=x.size)
        for q in [v,w]:q.reshape(-1,e.s.P)[4,~e.s.E]=0;q/=np.linalg.norm(q)
        jv=J(v);before=capture(e);jtw=JT(w);after=capture(e)
        assert all(np.array_equal(a,after[k]) for k,a in before.items())
        lhs=float(w@jv);rhs=float(v@jtw);error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
        checks.append(dict(direction=j,relative_bilinear_error=error,nominal_unchanged=True));write(out/'bilinear_checks.json',checks)
        assert error<1e-8,checks;log('SECTION ADJOINT IDENTITY',checks[-1])
    known=[]
    for path in (parent/'section_spectrum_cached').glob('mode*.npz'):
        data=np.load(path);mu=complex(data['multiplier'])
        if abs(mu.imag)<1e-8:known.append((abs(mu),mu.real,path))
    assert known,'Requires a prior independently verified real right mode'
    _,mu,path=max(known);mode=np.load(path)
    u=(mode['vector'].real.reshape(-1,e.s.P)*mode['coordinate_scale']/mode['coordinate_weight']*A.c.weight/A.c.scale).ravel()
    u/=np.linalg.norm(u);right_error=float(np.linalg.norm(J(u)-mu*u)/max(1.,abs(mu)))
    assert right_error<1e-6,right_error
    w=rng.normal(size=x.size);w.reshape(-1,e.s.P)[4,~e.s.E]=0;w/=np.linalg.norm(w);rows=[]
    for k in range(10):
        q=JT(w);err=float(np.linalg.norm(q-mu*w)/max(1.,abs(mu)))
        row=dict(iteration=k,relative_left_residual=err,rayleigh=float(w@q));rows.append(row)
        write(out/'left_progress.json',rows);log('CORE A LEFT RETURN MODE',row)
        if err<1e-8:break
        w=q/np.linalg.norm(q)
    passed=err<1e-8
    if passed:
        overlap=float(w@u);assert abs(overlap)>1e-10;w/=overlap
        np.savez_compressed(out/'local_unstable_coordinate.npz',left=w,right=u,multiplier=mu,
            coordinate_scale=A.c.scale,coordinate_weight=A.c.weight,base_state_coordinate=x,section_normal=A.normal)
    write(out/'result.json',dict(status='VERIFIED_NUMERICAL_LEFT_RIGHT_MODE' if passed else 'LEFT_MODE_NOT_CONVERGED',
        multiplier=mu,right_residual=right_error,left_relative_residual=err,checks=checks,closure=closure,
        seconds=time.time()-begin,physical_Floquet='PENDING_EXISTING_PHASE_AND_MESH_GATES',
        meaning='left dot delta_state is a local linear unstable coordinate only. No global stable manifold or collision has been computed.',model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0)
    a=p.parse_args();main(a.parent,a.device)

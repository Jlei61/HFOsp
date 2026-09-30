"""Bounded full-state Arnoldi spectrum at a numerically closed section root.

Ritz values are candidates until checked by the actual derivative in the full
state space. A finite Krylov search cannot certify completeness or stability.
Independent phase and mesh certificates remain separate.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn,regional_rate_section
from onset_shooting_newton import SectionDerivative
from onset_period_return import dynamical_state,errors
from pathlib import Path
import argparse,os,time,math


def main(parent,device,maximum,cached=False,cache_segments=None,measure_neutral=False,initial_mode=None,mode_random_fraction=.01):
    parent=Path(parent).resolve();result=read(parent/'result.json')
    whole=result['status'] in ['NUMERICAL_MULTIPLE_SHOOTING_ROOT','NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT']
    assert whole or result['status']=='NUMERICAL_PERIODIC_ROOT'
    assert not (cached and cache_segments)
    assert not measure_neutral or cache_segments
    assert 0<=mode_random_fraction<=1
    contract=read(parent/'contract.json');dt=contract['dt_ms']
    numerical_method=contract.get('numerical_method','old_endpoint')
    T=result.get('period_ms')
    if T is None:T=result.get('rows',result.get('iterations'))[-1]['period_ms']
    out=parent/('section_spectrum_segmented' if cache_segments else ('section_spectrum_cached' if cached else 'section_spectrum'));out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    prior_gate=parent/'multiple_section_spectrum/result.json'
    prior_failed=prior_gate.exists() and read(prior_gate)['status']=='NEUTRAL_MODE_GATE_FAILED_NO_PHYSICAL_FLOQUET'
    write(out/'contract.json',dict(source=str(parent),maximum_Krylov=maximum,dt_ms=dt,cached_derivative=cached,
        numerical_method=numerical_method,
        question='Which numerical section-return eigenmodes can guide the next mesh-qualified continuation step?',
        cache_segments=cache_segments,prior_physical_neutral_gate_failed=prior_failed,
        measure_neutral=measure_neutral,initial_mode=str(Path(initial_mode).resolve()) if initial_mode else None,
        mode_random_fraction=mode_random_fraction if initial_mode else None,
        neutral_scope='When requested, use the same original order2/order4 phase velocities and1e-3 neutral gate on the full fixed-time variational product. Record failures without reinterpreting the numerical Poincare spectrum as physical Floquet. This run is declared numerical before either result is observed.',
        prior_gate_result=str(prior_gate) if prior_gate.exists() else None,
        method='Two-pass Arnoldi of the full original delayed-state section derivative, including return time. Independently apply the actual derivative to at most four converged Ritz vectors. No reduced physical dynamics or low-dimensional fitted map.',
        scope='Existing numerical Poincare-map diagnostic, not a physical Floquet certificate. A previous failed physical neutral-mode gate is retained and is not waived by this calculation. Its sole use is to nominate which parameter interval/mode to resolve on a finer mesh. No complete spectrum, physical stable branch, crossing or bifurcation label without existing independent phase/mesh and correspondence checks.',model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()));start=time.time()
    Return,Derivative=SectionReturn,SectionDerivative
    if whole or contract.get('interpolation')=='cubic':
        from onset_cubic_section import CubicSectionReturn,CubicSectionDerivative
        Return,Derivative=CubicSectionReturn,CubicSectionDerivative
        if cached:
            from onset_cached_tangent import CachedCubicSectionDerivative
            qa=OUT/'core_a_bifurcation_type_20260924/numerical_checks'/('exact_cached_tangent_fine' if dt==.025 else 'exact_cached_tangent')/'result.json'
            assert read(qa)['status']=='PASS' and read(qa)['dt_ms']==dt
            Derivative=CachedCubicSectionDerivative
        if cache_segments:
            from onset_segmented_poincare import SegmentedPoincareDerivative
            Derivative=lambda A,x,T,slope:SegmentedPoincareDerivative(A,x,T,slope,cache_segments)
    else:assert not cached,'Exact cached implementation requires cubic dense output'
    if numerical_method=='exponential_midpoint':
        from onset_exponential_midpoint import ExponentialMidpointEngine
        from onset_midpoint_tangent import MidpointTangent
        from onset_midpoint_cached_tangent import MidpointCachedTangent
        assert not cached and contract.get('interpolation')=='cubic'
        qa=OUT/'core_a_bifurcation_type_20260924/numerical_checks/exponential_midpoint'
        assert read(qa/'section_variational_check_offgrid/result.json')['status']=='PASS'
        if cache_segments:
            Derivative=lambda A,x,T,slope:SegmentedPoincareDerivative(A,x,T,slope,cache_segments,
                tangent_class=MidpointTangent,cached_tangent_class=MidpointCachedTangent)
        else:
            Derivative=lambda A,x,T,slope:CubicSectionDerivative(A,x,T,slope,tangent_class=MidpointTangent)
        e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
    else:
        assert numerical_method=='old_endpoint'
        e=build(device,dt)
    source=parent/f"iteration{result['iterations'][-1]['iteration']:02d}"/'node00.npz' if whole else parent/('root_state.npz' if (parent/'root_state.npz').exists() else 'latest_state.npz')
    A=Return(dict(np.load(source)),e,T,5.)
    if cache_segments:
        from onset_segmented_poincare import cycle_coordinates
        cycle_coordinates(A,T,cache_segments)
    if contract.get('section','flow' if whole else 'core_A') in ['core_A','core_B']:
        regional_rate_section(A,0 if contract.get('section','core_A')=='core_A' else 1,contract.get('section_orientation'))
    x=A.xref;y,meta=A(x)
    closure=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
    assert closure['combined_relative_rms']<1e-6
    J=Derivative(A,x,meta['period_ms'],A.last_time_slope)
    if cache_segments:write(out/'partition_derivative_check.json',J.partition_check)
    neutral=[]
    if measure_neutral:
        from fine_rate_frozen_Z_fields import restore,capture
        restore(e,A.state(x));diff=[]
        for _ in range(4):
            e.step();e.cp.cuda.get_current_stream().synchronize()
            diff.append(A.c.pack(capture(e))-x)
        for order in [2,4]:
            velocity=sum(((-1)**(j+1)*math.comb(order,j)/j)*diff[j-1] for j in range(1,order+1))/dt
            err=float(np.linalg.norm(J.fixed_time(velocity)-velocity)/np.linalg.norm(velocity))
            neutral.append(dict(order=order,relative_error=err,pass_gate=err<1e-3))
            write(out/'neutral_checks.json',neutral);log('NUMERICAL SECTION PHYSICAL NEUTRAL CHECK',neutral[-1])
    rng=np.random.default_rng(92431);v=rng.normal(size=x.size)
    if initial_mode:
        # Transport an existing physical perturbation into this orbit's
        # numerical coordinates. This is an Arnoldi start, not a reduced
        # dynamics or an assumption that the old mode remains an eigenmode.
        with np.load(initial_mode) as z:
            old=z['vector'];scale=z['coordinate_scale'];weight=z['coordinate_weight']
        assert old.size==x.size, 'Mode transfer requires the same history mesh'
        assert np.linalg.norm(old.imag)<1e-8*np.linalg.norm(old.real), 'Real seed mode required'
        v=(old.real.reshape(-1,e.s.P)/weight*scale/A.c.scale*A.c.weight).ravel()
        if mode_random_fraction:
            random=rng.normal(size=x.size);random/=np.linalg.norm(random)
            v=v/np.linalg.norm(v)+mode_random_fraction*random
    v.reshape(-1,e.s.P)[4,~e.s.E]=0;v-=A.normal*(A.normal@v);v/=np.linalg.norm(v)
    V=[v];H=np.zeros((maximum+1,maximum));rows=[]
    for k in range(maximum):
        q=J(V[k])
        for _ in range(2):
            for j in range(k+1):
                h=float(V[j]@q);H[j,k]+=h;q-=h*V[j]
        H[k+1,k]=np.linalg.norm(q)
        values,vectors=np.linalg.eig(H[:k+1,:k+1]);res=abs(H[k+1,k]*vectors[-1])
        order=np.argsort(-abs(values))[:8]
        row=dict(dimension=k+1,candidates=[dict(real=float(values[j].real),imag=float(values[j].imag),
            modulus=float(abs(values[j])),estimated_residual=float(res[j])) for j in order])
        rows.append(row);write(out/'progress.json',rows);log('CORE A SECTION SPECTRUM',row)
        if H[k+1,k]<1e-13:break
        if k+1<maximum:V.append(q/H[k+1,k])
    np.savez_compressed(out/'hessenberg.npz',H=H[:k+2,:k+1],values=values,estimated_residuals=res)
    # Interleave largest modulus and proximity to both real crossings. A
    # complex leading pair must not be crowded out by several slow real M
    # modes that happen to lie closer to +1.
    rankings=[np.argsort(-abs(values)),np.argsort(abs(values-1)),np.argsort(abs(values+1))]
    ordering=[int(r[j]) for j in range(len(values)) for r in rankings]
    selected=[]
    for j in ordering:
        if j in selected or values[j].imag< -1e-8 or res[j]>1e-5*max(1,abs(values[j])):continue
        selected.append(int(j))
        if len(selected)==4:break
    verified=[]
    for index,j in enumerate(selected):
        z=sum(c*v for c,v in zip(vectors[:,j],V));z/=np.linalg.norm(z)
        real=J(z.real);imag=J(z.imag) if np.linalg.norm(z.imag)>1e-14 else np.zeros_like(real)
        product=real+1j*imag;error=float(np.linalg.norm(product-values[j]*z)/max(1,abs(values[j])))
        entry=dict(real=float(values[j].real),imag=float(values[j].imag),relative_residual=error,
            status='VERIFIED_NUMERICAL_EIGENPAIR' if error<1e-6 else 'FAILED_FULL_RESIDUAL')
        verified.append(entry);write(out/'verified.json',verified)
        np.savez_compressed(out/f'mode{index:02d}.npz',vector=z,product=product,multiplier=values[j],
            coordinate_scale=A.c.scale,coordinate_weight=A.c.weight)
    write(out/'result.json',dict(status='BOUNDED_NUMERICAL_SPECTRUM_COMPLETE',closure=closure,
        verified=verified,seconds=time.time()-start,completeness='NOT_ESTABLISHED',
        physical_Floquet=('NOT_QUALIFIED_PRIOR_NEUTRAL_GATE_FAILED' if prior_failed else
            'NOT_QUALIFIED_CURRENT_NEUTRAL_GATE_FAILED' if neutral and not neutral[-1]['pass_gate'] else 'PENDING_PHASE_AND_MESH'),
        neutral=neutral,
        prior_physical_neutral_gate_failed=prior_failed,model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0)
    p.add_argument('--krylov',type=int,default=16);p.add_argument('--cached-derivative',action='store_true')
    p.add_argument('--cache-segments',type=int)
    p.add_argument('--measure-neutral',action='store_true')
    p.add_argument('--initial-mode',help='Same-mesh real full-state numerical mode used only as an Arnoldi starting direction')
    p.add_argument('--mode-random-fraction',type=float,default=.01,
                   help='Add a reproducible full-state random component to a transferred mode to retain sensitivity to other modes')
    a=p.parse_args();main(a.parent,a.device,a.krylov,a.cached_derivative,a.cache_segments,a.measure_neutral,a.initial_mode,a.mode_random_fraction)

"""Full-state Newton-GMRES hookstep with admissible physical-state proposals.

The trust-region least-squares construction follows Chandler & Kerswell,
arXiv:1207.4682, section3.3. Our positive retraction and original-flow residual
checks are explicit additions for the nonnegative rate/history coordinates.
No low-rank physical model is substituted: Krylov vectors only solve Newton's
linear equation in the entire retained delayed state.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_cubic_section import CubicSectionReturn,CubicSectionDerivative
from onset_period_return import errors,dynamical_state
from core_a_positive_newton_coordinates import retract
from scipy.optimize import brentq
from scipy.linalg import solve_triangular
from pathlib import Path
import argparse,os,time


def physical_gram(vectors,block_size=32768):
    """Actual vector Gram matrix, using cache-friendly BLAS blocks.

    This evaluates the same full-state dot products, without replacing them
    by an orthogonality assumption or dropping any delayed coordinate.
    """
    G=np.zeros((len(vectors),len(vectors)))
    for first in range(0,len(vectors[0]),block_size):
        block=np.stack([v[first:first+block_size] for v in vectors])
        G+=block@block.T
    return G


def krylov(J,b,maximum,folder,P,m_factor,relative_scale=None,right_override=None,operator_description=None,linear_tolerance=1e-3):
    assert 0<linear_tolerance<=.1
    beta=np.linalg.norm(b);V=[b/beta];U=[];H=np.zeros((maximum+1,maximum));rows=[]
    def right(v):
        if right_override is not None:return right_override(v)
        u=v.copy()
        if relative_scale is not None:u*=relative_scale
        u.reshape(-1,P)[4]*=m_factor
        u-=J.normal*(J.normal@u)
        return u
    for k in range(maximum):
        u=right(V[k]);U.append(u);q=u-J(u)
        for _ in range(2):
            for j in range(k+1):
                h=float(V[j]@q);H[j,k]+=h;q-=h*V[j]
        H[k+1,k]=np.linalg.norm(q)
        rhs=np.r_[beta,np.zeros(k+1)];coef=np.linalg.lstsq(H[:k+2,:k+1],rhs,rcond=None)[0]
        residual=float(np.linalg.norm(H[:k+2,:k+1]@coef-rhs)/beta)
        rows.append(dict(dimension=k+1,relative_linear_residual=residual))
        write(folder/'linear_progress.json',rows);log('CORE A HOOK KRYLOV',k+1,residual)
        if residual<linear_tolerance or H[k+1,k]<1e-14:break
        if k+1<maximum:V.append(q/H[k+1,k])
    H=H[:k+2,:k+1];rhs=np.r_[beta,np.zeros(k+1)]
    # U need not be orthonormal after physical M preconditioning. Transform
    # the actual full-state update norm into Euclidean small coordinates.
    G=physical_gram(U);L=np.linalg.cholesky(G)
    transform=solve_triangular(L.T,np.eye(len(U)),lower=False)
    A=H@transform;left,singular,right_t=np.linalg.svd(A,full_matrices=False)
    projected=left.T@rhs
    np.savez_compressed(folder/'hook_system.npz',H=H,G=G,rhs=rhs,singular_values=singular,
        M_right_precondition=m_factor,linear_tolerance=linear_tolerance,
        operator_description=operator_description or '(I-DP) times phase-projected M preconditioner; not a monodromy matrix')
    def proposal(radius):
        z=np.divide(projected,singular,out=np.zeros_like(projected),where=singular>1e-13*singular.max())
        lam=0.
        if np.linalg.norm(z)>radius:
            def norm(mu):return np.linalg.norm(singular*projected/(singular*singular+mu))-radius
            hi=max(singular.max()**2,1e-12)
            while norm(hi)>0:hi*=4
            lam=brentq(norm,0.,hi,xtol=1e-14)
            z=singular*projected/(singular*singular+lam)
        small=transform@(right_t.T@z);delta=sum(a*u for a,u in zip(small,U))
        norm_actual=float(np.linalg.norm(delta))
        assert norm_actual<=radius*(1+1e-6)+1e-12
        return delta,dict(radius=float(radius),step_norm=norm_actual,lagrange_multiplier=float(lam),
            predicted_relative_residual=float(np.linalg.norm(H@small-rhs)/beta))
    return proposal,rows


def main(a):
    numerical_method=getattr(a,'numerical_method','old_endpoint')
    target_native_time=getattr(a,'target_native_time',None)
    closure_tolerance=getattr(a,'closure_tolerance',1e-7)
    partition_check=getattr(a,'partition_check','every')
    linear_tolerance=getattr(a,'linear_tolerance',1e-3)
    assert partition_check in ['every','first'] and 0<linear_tolerance<=.1
    assert 0<closure_tolerance<=1e-7, 'Only tightening the original closure gate is supported'
    cache_segments=getattr(a,'cache_segments',None)
    cycle_rms=getattr(a,'cycle_rms',False)
    refractory_projection=getattr(a,'refractory_projection',False)
    assert cache_segments is None or 2<=cache_segments<=32
    assert not refractory_projection or a.orthogonal_phase
    if refractory_projection:
        assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/refractory_projection/result.json')['status']=='PASS'
    assert not (cache_segments and a.cached_derivative)
    assert not cycle_rms or cache_segments
    assert not a.orthogonal_phase or a.projected_proposal
    halfwidth=min(8.,a.period*.24) if a.halfwidth is None else a.halfwidth
    out=Path(a.destination).resolve();out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    source=Path(a.source).resolve()
    write(out/'contract.json',dict(source=str(source),period_seed_ms=a.period,dt_ms=a.dt,source_dt_ms=a.source_dt,
        numerical_method=numerical_method,target_native_path_time_ms=target_native_time,
        all_Z_held=True,all_M_dynamic=True,section='flow',interpolation='cubic',
        method='Full-state Newton-GMRES hookstep. Actual update norm is bounded, including any right preconditioning. Nonnegative solver proposals use a rational retraction at positive coordinates and project outward steps at exact zero onto the coordinate boundary. This only constrains numerical proposals; the physical flow is unchanged and never clipped. The predicted residual is recomputed from the actual retracted step with the full variational derivative. Only an actual residual decrease with adequate prediction agreement is accepted.',
        reference='https://arxiv.org/pdf/1207.4682 section3.3; positivity and physical M preconditioning are explicit solver additions.',
        M_right_precondition=a.m_precondition,initial_radius_fraction=a.radius,
        relative_positive_precondition=a.relative_positive,
        cached_derivative=a.cached_derivative,
        cache_segments=cache_segments,cycle_rms_coordinates=cycle_rms,
        partition_check_policy=partition_check,linear_solve_relative_tolerance=linear_tolerance,
        segmented_derivative='Compose full fixed-time segment derivatives, then apply the exact implicit section-return-time derivative. Compare against the original unsegmented uncached return on every iteration, or on the first iteration of this root solve as explicitly recorded by partition_check_policy. First-iterate nonlinear Poincare finite differences, actual physical residual checks for every accepted update, and separate final phase/spectral qualification remain required.' if cache_segments else None,
        positive_proposal_method=('exact Euclidean projection onto nonnegative bounds and the full phase plane; independently checked KKT solver' if a.orthogonal_phase else ('nonnegative coordinate projection with signed-memory phase restoration' if a.projected_proposal else 'rational positive retraction; unmodified Newton if admissible')),
        orthogonal_phase_projection=a.orthogonal_phase,
        refractory_occupancy_projection=refractory_projection,
        return_halfwidth_ms=halfwidth,
        relative_positive_precondition_definition='Right-coordinate scale for positive variables = max(current physical value / source row RMS scale,1e-4); signed input-memory variables unscaled. This changes only the numerical linear-system coordinates. Actual full-state norm, derivative and residual remain unchanged.',
        maximum_iterations=a.iterations,Krylov_per_iteration=a.krylov,maximum_trial_returns=5,
        closure_gate=f'combined<{closure_tolerance:g} and every block<{10*closure_tolerance:g}; nontrivial amplitude, independent phase/mesh, stability and bifurcation checks remain separate.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);start=time.time()
    Derivative=CubicSectionDerivative
    if cache_segments:
        from onset_segmented_poincare import SegmentedPoincareDerivative
        Derivative=lambda A,x,T,slope:SegmentedPoincareDerivative(A,x,T,slope,cache_segments,
            verify_partition=(partition_check=='every' or iteration==0))
    if a.cached_derivative:
        qa=read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/exact_cached_tangent/result.json')
        assert qa['status']=='PASS' and qa['dt_ms']==a.dt
        from onset_cached_tangent import CachedCubicSectionDerivative
        Derivative=CachedCubicSectionDerivative
    if numerical_method=='exponential_midpoint':
        from onset_exponential_midpoint import ExponentialMidpointEngine
        from onset_midpoint_tangent import MidpointTangent
        from onset_midpoint_cached_tangent import MidpointCachedTangent
        from check_spatial_midpoint_convergence import conservative_history
        qa=OUT/'core_a_bifurcation_type_20260924/numerical_checks/exponential_midpoint'
        assert read(qa/'full_variational_check/result.json')['status']=='PASS'
        assert read(qa/'section_variational_check_offgrid/result.json')['status']=='PASS'
        assert not a.cached_derivative,'Use explicitly selected midpoint cache segments'
        if cache_segments:
            assert read(qa/'cached_variational_check/result.json')['status']=='PASS'
            Derivative=lambda A,x,T,slope:SegmentedPoincareDerivative(A,x,T,slope,cache_segments,
                tangent_class=MidpointTangent,cached_tangent_class=MidpointCachedTangent,
                verify_partition=(partition_check=='every' or iteration==0))
        else:
            Derivative=lambda A,x,T,slope:CubicSectionDerivative(A,x,T,slope,tangent_class=MidpointTangent)
        e=ExponentialMidpointEngine(dt=a.dt,device=a.device);e.graph()
        base,history_qa=conservative_history(dict(np.load(source)),e,a.source_dt)
        write(out/'initial_history_qa.json',history_qa)
    else:
        assert numerical_method=='old_endpoint'
        e=build(a.device,a.dt);base=regrid_state(np.load(source),e,a.source_dt)
    if target_native_time is not None:
        from core_a_parameter_path_audit import NativeTimeFamily
        family=NativeTimeFamily(e.s);z,d=family.field_at_time(target_native_time)
        assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
        base['syn'][5]=z
        write(out/'target_field.json',dict(native_time_ms=target_native_time,D_A=d,Z_A=1-d,
            outside_A_unchanged=True,all_M_dynamic=True))
    A=CubicSectionReturn(base,e,a.period,halfwidth)
    if cycle_rms:
        from onset_segmented_poincare import cycle_coordinates
        cycle_coordinates(A,a.period,cache_segments)
    x=A.xref.copy()
    radius=a.radius*np.linalg.norm(x);rows=[];status='ITERATION_LIMIT_NOT_AN_ORBIT'
    try:
        for iteration in range(a.iterations):
            folder=out/f'iteration{iteration:02d}';folder.mkdir(exist_ok=True)
            y,meta=A(x);f=y-x;norm=float(np.linalg.norm(f));slope=A.last_time_slope.copy()
            err=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
            row=dict(iteration=iteration,**meta,**err,coordinate_residual=norm/np.linalg.norm(x));rows.append(row)
            write(out/'iterations.json',rows);np.savez_compressed(out/'latest_state.npz',**A.state(x))
            jobs.update(iteration=iteration,last_residual=err['combined_relative_rms']);write(out/'jobs.json',jobs)
            log('CORE A HOOK ROOT',iteration,meta['period_ms'],err['combined_relative_rms'],radius)
            if err['combined_relative_rms']<closure_tolerance and max(v['relative_rms'] for v in err['blocks'].values())<10*closure_tolerance:
                status='NUMERICAL_PERIODIC_ROOT';break
            if iteration:
                del J
                import gc
                gc.collect();e.cp.get_default_memory_pool().free_all_blocks()
            if cache_segments and partition_check=='first' and iteration:
                assert read(out/'iteration00/partition_derivative_check.json')['status']=='PASS'
            J=Derivative(A,x,meta['period_ms'],slope)
            if cache_segments:write(folder/'partition_derivative_check.json',J.partition_check)
            if iteration==0:
                # A cubic return can overshoot a zero rate between actual
                # samples, making the residual an inadmissible FD direction
                # even though x itself is physical. Test a reproducible full
                # state direction instead; do not clip the flow or waive FD.
                rng=np.random.default_rng(241355)
                direction=x*rng.standard_normal(x.size)
                free=np.zeros_like(A.normal)
                free.reshape(-1,e.s.P)[11:47]=A.normal.reshape(-1,e.s.P)[11:47]
                direction-=free*float(A.normal@direction)/float(A.normal@free)
                direction*=np.linalg.norm(f)/np.linalg.norm(direction)
                derivative=J(direction);checks=[];error=float('inf')
                for eps in [1e-3,1e-4,1e-5,1e-6,1e-7]:
                    if not A.admissible(x+eps*direction):
                        checks.append(dict(epsilon=eps,status='INADMISSIBLE_TEST_DIRECTION'))
                        write(out/'derivative_check.json',checks);continue
                    try:
                        q,_=A(x+eps*direction)
                    except RuntimeError as exc:
                        # A finite test perturbation can move a section
                        # crossing outside its declared local window.
                        # Reduce its size; neither waive the derivative
                        # gate nor mistake a missed section for a bifurcation.
                        checks.append(dict(epsilon=eps,status='SECTION_WINDOW_MISS',error=str(exc)))
                        write(out/'derivative_check.json',checks)
                        continue
                    error=float(np.linalg.norm((q-y)/eps-derivative)/np.linalg.norm(derivative))
                    checks.append(dict(epsilon=eps,relative_error=error,
                        direction='Fixed-seed full-state multiplicative perturbation, projected to the section through signed input-memory coordinates; actual admissibility checked without clipping.',
                        phase_residual=float(A.normal@direction)))
                    write(out/'derivative_check.json',checks)
                    if error<1e-3:break
                assert error<1e-3,'Actual full-state derivative gate failed'
            factor=1./(-np.expm1(-meta['period_ms']/float(e.transport.consts[6].get()))) if a.m_precondition else 1.
            relative_scale=None
            if a.relative_positive:
                shape=(-1,e.s.P);positive=np.ones_like(x.reshape(shape));relative=x.reshape(shape)/A.c.weight
                positive[:11]=np.maximum(relative[:11],1e-4)
                positive[47:]=np.maximum(relative[47:],1e-4)
                positive[4,~e.s.E]=1.
                relative_scale=positive.ravel()
            propose,linear=krylov(J,f,a.krylov,folder,e.s.P,factor,relative_scale,linear_tolerance=linear_tolerance)
            trials=[];accepted=False
            for attempt in range(9):
                delta,small=propose(radius);diagnostic={}
                candidate=retract(A,x,delta,1.,project_zero=True,diagnostics=diagnostic,
                    project_all=a.projected_proposal,orthogonal_phase=a.orthogonal_phase,
                    refractory_projection=refractory_projection)
                small['admissibility_diagnostic']=diagnostic
                if attempt==0:
                    np.savez_compressed(folder/'first_proposal_direction.npz',delta=delta,source=x,
                        normal=A.normal,coordinate_scale=A.c.scale,coordinate_weight=A.c.weight)
                if candidate is None:
                    trials.append(dict(status='INADMISSIBLE_NO_FLOW',**small))
                    write(folder/'trials.json',trials);radius*=.5;continue
                effective=candidate-x;actual_norm=float(np.linalg.norm(effective))
                if actual_norm>1.05*radius:
                    trials.append(dict(status='RETRACTION_EXCEEDS_RADIUS',actual_norm=actual_norm,**small));write(folder/'trials.json',trials);radius*=.5;continue
                if sum(t['status'] in ['EVALUATED','SECTION_WINDOW_MISS'] for t in trials)>=5:break
                try:value,info=A(candidate)
                except RuntimeError as exc:
                    trials.append(dict(status='SECTION_WINDOW_MISS',error=str(exc),**small));radius*=.5;continue
                predicted=f-effective+J(effective)
                pred_gain=1-float(predicted@predicted)/(norm*norm)
                newnorm=float(np.linalg.norm(value-candidate));actual_gain=1-(newnorm/norm)**2
                agreement=actual_gain/pred_gain if pred_gain>0 else -float('inf')
                trial=dict(status='EVALUATED',**small,actual_step_norm=actual_norm,
                    retraction_relative_change=float(np.linalg.norm(effective-delta)/max(np.linalg.norm(delta),1e-30)),
                    residual_ratio=newnorm/norm,predicted_gain=pred_gain,agreement=agreement,period_ms=info['period_ms'])
                trials.append(trial);write(folder/'trials.json',trials);log('CORE A HOOK TRIAL',trial)
                if actual_gain>0 and agreement>.1:
                    x=candidate;A.period=info['period_ms'];accepted=True
                    np.savez_compressed(folder/'accepted_state.npz',**A.state(x))
                    write(folder/'accepted_update.json',dict(period_ms=A.period,residual_ratio=newnorm/norm))
                    if agreement>.75:radius=min(2*radius,.2*np.linalg.norm(x))
                    break
                radius*=.5
            row.update(trials=trials,linear_last=linear[-1]);write(out/'iterations.json',rows)
            if not accepted:status='NO_ACCEPTED_HOOKSTEP_NOT_A_BIFURCATION';break
        write(out/'result.json',dict(status=status,period_ms=rows[-1]['period_ms'],dt_ms=a.dt,iterations=rows,
            seconds=time.time()-start,physical_Floquet='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--period',type=float,required=True);p.add_argument('--dt',type=float,default=.05);p.add_argument('--source-dt',type=float,default=.05)
    p.add_argument('--device',type=int,default=0);p.add_argument('--krylov',type=int,default=24);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--radius',type=float,default=.02);p.add_argument('--m-precondition',action='store_true')
    p.add_argument('--relative-positive',action='store_true');p.add_argument('--cached-derivative',action='store_true')
    p.add_argument('--cache-segments',type=int);p.add_argument('--cycle-rms',action='store_true')
    p.add_argument('--refractory-projection',action='store_true')
    p.add_argument('--numerical-method',choices=['old_endpoint','exponential_midpoint'],default='old_endpoint')
    p.add_argument('--target-native-time',type=float)
    p.add_argument('--closure-tolerance',type=float,default=1e-7)
    p.add_argument('--partition-check',choices=['every','first'],default='every',
                   help='Keep first-iterate original-flow derivative comparison; optionally omit repeats within this same root solve')
    p.add_argument('--linear-tolerance',type=float,default=1e-3,
                   help='Krylov solve tolerance only; accepted physical residual and all root gates are unchanged')
    p.add_argument('--projected-proposal',action='store_true');p.add_argument('--orthogonal-phase',action='store_true');p.add_argument('--halfwidth',type=float);main(p.parse_args())

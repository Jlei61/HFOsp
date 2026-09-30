"""Full-state multiple shooting of the observed A-sustained/B-burst seed.

Four physical trajectory segments and their common period are solved jointly.
No averaged waveform, clamped M, external future activity or smaller network
is substituted for a segment. The full cyclic matching residual decides
acceptance; solver convergence alone cannot identify an onset bifurcation.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn
from onset_segment_flow import fixed_time,SegmentDerivative
from onset_period_return import dynamical_state,errors
from core_a_periodic_hookstep import krylov
from core_a_positive_newton_coordinates import retract
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,os,time,gc


def cyclic_slow_inverse(rhs,segment_over_tau):
    """Invert x[j+1]-exp(-h/tau)*x[j]=rhs[j], including cycle closure.

    This is a right preconditioner for the known intrinsic M decay only.
    The full variational operator still retains every M-to-rate feedback.
    All shooting nodes use the same physical coordinate scaling.
    """
    count=rhs.shape[0]
    u=np.broadcast_to(np.asarray(segment_over_tau,float),(count,))
    assert count>=2 and np.all(u>0)
    a=np.exp(-u)
    x=np.empty_like(rhs)
    x[0]=sum(np.exp(-u[j+1:].sum())*rhs[j] for j in range(count))/(-np.expm1(-u.sum()))
    for j in range(count-1):x[j+1]=a[j]*x[j]+rhs[j]
    return x


def segment_partition(period,count,fixed_grid_ms=None):
    if fixed_grid_ms is None:return np.full(count,period/count),np.full(count,1/count)
    h=np.r_[np.full(count-1,fixed_grid_ms),period-(count-1)*fixed_grid_ms]
    assert h.min()>0
    derivative=np.zeros(count);derivative[-1]=1.
    return h,derivative


def setup(e,source,resume=None,target_D=None,target_native_time=None,common_scale='source',segments=4):
    assert target_D is None or target_native_time is None
    assert common_scale in ['source','cycle_rms']
    result=read(source/'result.json');assert result['status'] in ['EXACT_REPLAY_PASS_RECURRENCE_ONLY','REPLAY_AGREEMENT_PASS_RECURRENCE_ONLY','UNINTERRUPTED_FIXED_GRID_SEED']
    assert result.get('dt_ms',.05)==e.dt,'Use the same mesh as the stored full-state seed'
    if resume:assert read(resume.parent/'contract.json')['dt_ms']==e.dt
    if result['status']=='UNINTERRUPTED_FIXED_GRID_SEED':
        assert result['all_nodes_on_actual_steps'] and result['all_nodes_admissible']
        assert result['single_pass_parity_relative']<1e-10
    if result['status']=='REPLAY_AGREEMENT_PASS_RECURRENCE_ONLY':
        qa=result['original_checkpoint_comparison']
        assert qa['combined_relative_rms']<1e-10 and max(v['relative_rms'] for v in qa['blocks'].values())<1e-9
    T=result['period_ms'];parts=[];scale=None
    family=None
    if target_native_time is not None:
        from core_a_parameter_path_audit import NativeTimeFamily
        family=NativeTimeFamily(e.s);target_Z,_=family.field_at_time(target_native_time)
    elif target_D is not None:
        from core_a_equilibrium_branch import Family
        family=Family(e.s);target_Z,_=family.field(target_D)
    if resume:T=read(resume/'accepted_period.json')['period_ms']
    assert segments>=2
    assert result.get('segments',4)==segments,'Seed nodes must match the declared segment count'
    for j in range(segments):
        path=resume/f'accepted_node{j:02d}.npz' if resume else source/f'node{j:02d}.npz'
        base=dict(np.load(path))
        assert base['history'].shape==e.local.history.shape
        if family is not None:
            assert np.array_equal(base['syn'][5,~family.A],target_Z[~family.A])
            base['syn'][5]=target_Z.copy()
        A=SectionReturn(base,e,T/segments)
        if scale is None:scale=A.c.scale.copy()
        A.c.scale=scale.copy();A.xref=A.c.pack(base)
        # Recompute the phase normal in the common physical coordinate scale.
        restore(e,base);points=[A.xref]
        for _ in range(2):
            e.step();e.cp.cuda.get_current_stream().synchronize();points.append(A.c.pack(capture(e)))
        normal=(-3*points[0]+4*points[1]-points[2])/(2*e.dt)
        A.normal=normal/np.linalg.norm(normal);parts.append(A)
    if common_scale=='cycle_rms':
        # The reference phase can be quiet while other nodes are active.
        # A common row scale from all nodes prevents their histories from
        # being measured in the tiny quiet-phase units. This is a full,
        # invertible numerical coordinate change, not a reduced model.
        power=sum(((A.c.raw(A.base)**2)*(A.c.weight**2)).sum(1) for A in parts)/len(parts)
        scale=np.maximum(np.sqrt(power)[:,None],parts[0].c.floor)
        for A in parts:
            A.c.scale=scale.copy();A.xref=A.c.pack(A.base)
            restore(e,A.base);points=[A.xref]
            for _ in range(2):
                e.step();e.cp.cuda.get_current_stream().synchronize();points.append(A.c.pack(capture(e)))
            normal=(-3*points[0]+4*points[1]-points[2])/(2*e.dt)
            A.normal=normal/np.linalg.norm(normal)
    for A in parts:
        assert np.array_equal(A.base['syn'][5],parts[0].base['syn'][5])
        assert np.array_equal(A.c.scale,parts[0].c.scale)
        assert np.linalg.norm(A.c.pack(A.state(A.xref))-A.xref)<1e-12
    return parts,T


def main(a):
    K=getattr(a,'segments',4)
    dt=getattr(a,'dt',.05)
    condensed=getattr(a,'condensed_linear',False)
    if condensed:
        assert not a.cyclic_right and not a.cyclic_M
        assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/condensed_multiple_shooting/result.json')['status']=='PASS'
    assert not a.cyclic_M or a.cyclic_right
    assert not a.refractory_projection or a.orthogonal_phase
    if a.refractory_projection:
        assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/refractory_projection/result.json')['status']=='PASS'
    source=Path(a.source).resolve();out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'jobs.json').exists();begin=time.time()
    source_result=read(source/'result.json')
    fixed_grid_ms=source_result.get('fixed_grid_segment_ms')
    write(out/'contract.json',dict(source=str(source),dt_ms=dt,segments=K,
        condensed_linear_solve=condensed,
        condensed_linear_method='Exact block elimination of the K-node Newton equations only; all nonlinear nodes remain independent. Solve the node-zero/period bordered Schur system, back-substitute every node, independently check the original full linear residual, and damp the entire recovered update within the original full-state radius. Original nonlinear matching, derivative and physical gates unchanged.' if condensed else None,
        fixed_grid_segment_ms=fixed_grid_ms,
        numerical_restart=str(Path(a.resume).resolve()) if a.resume else None,
        target_D_A=a.target_D,target_native_path_time_ms=getattr(a,'target_native_time',None),
        common_coordinate_scale=getattr(a,'common_scale','source'),
        coordinate_scale_definition='source uses the first node row RMS; cycle_rms uses cell-weighted row RMS across all complete nodes with the same existing row floors. Shared invertible coordinates only; original per-node physical closure gates unchanged.',
        parameter_continuation='Only original native within-Core-A Z changes. target_native_path_time_ms follows the continuous recorded field path, with meanD_A only a display coordinate; target_D_A is the historical first-upcrossing lookup and must not be presumed continuous across mean-recovery loops. Outside-Core-A Z must match bitwise; other initial states including every M are preserved as numerical seeds. Each physical segment holds all Z and evolves all M.',
        cyclic_right_precondition=a.cyclic_right,
        cyclic_M_precondition=a.cyclic_M,
        cyclic_M_definition='Exact inverse of deltaM[j+1]-exp(-T/(K*tau_M))*deltaM[j] across all K nodes; rate feedback remains in the full actual derivative. Numerical right coordinates only.' if a.cyclic_M else None,
        orthogonal_phase_bound_projection=a.orthogonal_phase,
        refractory_occupancy_projection=a.refractory_projection,
        equations='Unchanged3479-group spatial delayed rate model; entireZheld and allMdynamic. Exact observed recurrent trajectory supplies K complete initial nodes.',
        method='Solve K full-state matching equations Phi(h_j,X_j)-X_(j+1)=0 and one autonomous phase condition. Default h_j=T/K. If fixed_grid_segment_ms is present, first K-1 h_j are anchored integer-step durations and only last h_j=T-sum(h_0..h_K-2) varies. Intermediate endpoints then use actual steps without fractional-history interpolation; final endpoint retains original cubic output. Exact full variational equations and all physical root gates unchanged.',
        gates='Independent cached-versus-original variational products at every node and a full bordered actual-flow finite difference. Original nonlinear matching residual and actual-step hookstep prediction gate every accepted update. Full closure combined<1e-7 and every physical block<1e-6 at every node, phase<1e-9. Independent phase/mesh and Floquet remain separate.',
        period_interval_ms=[.8*read(source/'result.json')['period_ms'],1.2*read(source/'result.json')['period_ms']],
        maximum_iterations=a.iterations,Krylov_per_iteration=a.krylov,
        linear_solve_tolerance_cap=a.linear_tolerance,
        linear_solve_rule='min(cap,max(1e-6,0.1*sqrt(maximum physical matching residual))). Only the inexact Newton solve tolerance changes; physical root gates and actual nonlinear/prediction checks remain unchanged.',
        interpretation='A periodic root is only a candidate invariant skeleton. Must preserve actual A-sustained/B-burst structure, continue a relevant critical crossing and check stability/nondegeneracy before naming the onset bifurcation.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    e=build(a.device,dt);parts,T=setup(e,source,Path(a.resume).resolve() if a.resume else None,a.target_D,getattr(a,'target_native_time',None),getattr(a,'common_scale','source'),K)
    seedT=read(source/'result.json')['period_ms'];N=parts[0].c.size;wT=.01
    x=np.concatenate([A.xref for A in parts]);radius=a.radius*np.linalg.norm(x);rows=[]
    status='ITERATION_LIMIT_NOT_A_PERIODIC_ROOT'
    def evaluate(xx,period):
        values=[];slopes=[];checks=[]
        durations,_=segment_partition(period,K,fixed_grid_ms)
        for j,A in enumerate(parts):
            y,slope=fixed_time(A,xx[j*N:(j+1)*N],durations[j])
            nxt=xx[((j+1)%K)*N:((j+1)%K+1)*N]
            values.append(y-nxt);slopes.append(slope)
            checks.append(errors(dynamical_state(A.state(nxt)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum()))
        phase=float(parts[0].normal@(xx[:N]-parts[0].xref))
        return np.r_[np.concatenate(values),-phase],slopes,checks,phase
    try:
        for iteration in range(a.iterations):
            step=out/f'iteration{iteration:02d}';step.mkdir(exist_ok=True)
            f,slopes,checks,phase=evaluate(x,T);norm=float(np.linalg.norm(f))
            durations,time_derivative=segment_partition(T,K,fixed_grid_ms)
            row=dict(iteration=iteration,period_ms=T,durations_ms=durations.tolist(),matching=checks,phase=phase,
                max_combined_relative_rms=max(z['combined_relative_rms'] for z in checks),coordinate_residual=norm/np.linalg.norm(x))
            rows.append(row);write(out/'iterations.json',rows)
            for j,A in enumerate(parts):np.savez_compressed(step/f'node{j:02d}.npz',**A.state(x[j*N:(j+1)*N]))
            jobs.update(iteration=iteration,stage='MATCHING',last_residual=row['max_combined_relative_rms']);write(out/'jobs.json',jobs)
            log('MULTIPLE SHOOTING MATCHING',iteration,T,row['max_combined_relative_rms'])
            if row['max_combined_relative_rms']<1e-7 and max(v['relative_rms'] for z in checks for v in z['blocks'].values())<1e-6 and abs(phase)<1e-9:
                status='NUMERICAL_MULTIPLE_SHOOTING_ROOT';break
            if iteration:
                del derivative,J,shared,propose
                gc.collect();e.cp.get_default_memory_pool().free_all_blocks()
            derivative=[None]*K;shared=None
            for j in np.argsort(-durations):
                A=parts[j]
                J=SegmentDerivative(A,x[j*N:(j+1)*N],durations[j],shared=shared)
                shared=J.shared;derivative[j]=J
                if iteration==0:
                    rng=np.random.default_rng(92471+j);v=x[j*N:(j+1)*N]*rng.normal(size=N);v/=np.linalg.norm(v)
                    actual=J(v);old=SegmentDerivative(A,x[j*N:(j+1)*N],durations[j],cached=False)
                    expected=old(v);err=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                    write(step/f'segment{j}_cache_check.json',dict(status='PASS' if err<1e-10 else 'FAIL',relative_error=err))
                    assert err<1e-10,('Segment derivative cache mismatch',j,err)
                    del old
                log('MULTIPLE SHOOTING CACHE',iteration,j)
            def linear(v):
                vv=v[:-1].reshape(K,N);dt=v[-1]/wT;result=[]
                for j,J in enumerate(derivative):result.append(vv[(j+1)%K]-J(vv[j])-slopes[j]*dt*time_derivative[j])
                return np.r_[np.concatenate(result),parts[0].normal@vv[0]]
            if iteration==0:
                rng=np.random.default_rng(92475);v=x*rng.normal(size=x.size);v*=norm/np.linalg.norm(v)
                direction=np.r_[v,wT*.01];actual=linear(direction);fd=[];err=np.inf
                for epsilon in [1e-4,1e-5,1e-6]:
                    xx=x+epsilon*v
                    if not all(A.admissible(xx[j*N:(j+1)*N]) for j,A in enumerate(parts)):continue
                    ff,_,_,_=evaluate(xx,T+epsilon*.01)
                    err=float(np.linalg.norm((ff-f)/epsilon+actual)/np.linalg.norm(actual))
                    fd.append(dict(epsilon=epsilon,relative_error=err));write(out/'bordered_derivative_check.json',fd)
                    if err<1e-3:break
                assert err<1e-3,('Multiple shooting derivative FD gate',fd)
                if a.check_only:
                    write(out/'result.json',dict(status='MULTIPLE_SHOOTING_DERIVATIVE_CHECKS_PASS',matching=checks,finite_difference=fd,model_promoted=False))
                    jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);return
            factor=1./(-np.expm1(-T/(K*float(e.transport.consts[6].get()))))
            def right(v):
                u=v.copy()
                if a.cyclic_right:
                    # The leading matching operator is the cyclic shift
                    # S(delta)_j=delta_(j+1). Its exact inverse transforms
                    # S-DPhi to I-DPhi*S^-1, without modifying any flow.
                    u[:-1]=np.roll(v[:-1].reshape(K,N),1,axis=0).ravel()
                if a.cyclic_M:
                    m=cyclic_slow_inverse(v[:-1].reshape(K,-1,e.s.P)[:,4],durations/float(e.transport.consts[6].get()))
                    u[:-1].reshape(K,-1,e.s.P)[:,4]=m
                else:
                    for j in range(K):u[j*N:(j+1)*N].reshape(-1,e.s.P)[4]*=factor
                # Phase is an explicit bordered equation. Projecting this
                # coordinate here would prevent Krylov from solving period.
                return u
            jobs.update(stage='KRYLOV');write(out/'jobs.json',jobs)
            tolerance=min(a.linear_tolerance,max(1e-6,.1*np.sqrt(row['max_combined_relative_rms'])))
            if condensed:
                from onset_condensed_shooting import condensed_proposal
                full_M_factor=1./(-np.expm1(-T/float(e.transport.consts[6].get())))
                propose,progress=condensed_proposal(derivative,slopes,time_derivative,parts[0].normal,
                    f,wT,e.s.P,full_M_factor,a.krylov,step,tolerance,linear)
            else:
                propose,progress=krylov(lambda v:v-linear(v),f,a.krylov,step,e.s.P,1. if a.cyclic_M else factor,
                    right_override=right,operator_description='Full K-segment cyclic [next-DPhi, -dPhi/dT;phase] Newton matching operator; not Floquet',
                    linear_tolerance=tolerance)
            trials=[];accepted=False
            for attempt in range(9):
                delta,small=propose(radius);tt=T+delta[-1]/wT;xx=[];diagnostics=[]
                if attempt==0:
                    np.save(step/'first_proposal_delta.npy',delta)
                    np.save(step/'phase_normal.npy',parts[0].normal)
                for j,A in enumerate(parts):
                    d={};q=retract(A,x[j*N:(j+1)*N],delta[j*N:(j+1)*N],1.,
                        project_zero=True,project_all=True,restore_phase=j==0,diagnostics=d,orthogonal_phase=a.orthogonal_phase,
                        refractory_projection=a.refractory_projection)
                    xx.append(q);diagnostics.append(d)
                if not .8*seedT<tt<1.2*seedT or (fixed_grid_ms is not None and tt-(K-1)*fixed_grid_ms<2*e.dt) or any(q is None for q in xx):
                    trials.append(dict(status='INADMISSIBLE_NUMERICAL_PROPOSAL',diagnostics=diagnostics,**small));write(step/'trials.json',trials);radius*=.5;continue
                xx=np.concatenate(xx);effective=np.r_[xx-x,wT*(tt-T)]
                if np.linalg.norm(effective)>1.05*radius:
                    trials.append(dict(status='PROJECTION_EXCEEDS_RADIUS',actual_norm=float(np.linalg.norm(effective)),diagnostics=diagnostics,**small));write(step/'trials.json',trials);radius*=.5;continue
                if sum(t['status']=='EVALUATED' for t in trials)>=4:break
                ff,ss,cc,pp=evaluate(xx,tt);pred=f-linear(effective)
                gain=1-float(ff@ff)/(norm*norm);pg=1-float(pred@pred)/(norm*norm);agreement=gain/pg if pg>0 else -np.inf
                trial=dict(status='EVALUATED',period_ms=tt,max_combined_relative_rms=max(z['combined_relative_rms'] for z in cc),
                    residual_ratio=float(np.linalg.norm(ff)/norm),predicted_gain=pg,agreement=agreement,diagnostics=diagnostics,**small)
                trials.append(trial);write(step/'trials.json',trials);log('MULTIPLE SHOOTING TRIAL',trial)
                if gain>0 and agreement>.1:
                    x=xx;T=float(tt);accepted=True
                    for j,A in enumerate(parts):np.savez_compressed(step/f'accepted_node{j:02d}.npz',**A.state(x[j*N:(j+1)*N]))
                    write(step/'accepted_period.json',dict(period_ms=T,durations_ms=segment_partition(T,K,fixed_grid_ms)[0].tolist(),fixed_grid_segment_ms=fixed_grid_ms))
                    if agreement>.75:radius=min(2*radius,.2*np.linalg.norm(x))
                    break
                radius*=.5
            row.update(trials=trials,linear_last=progress[-1]);write(out/'iterations.json',rows)
            if not accepted:status='NO_ACCEPTED_MULTIPLE_SHOOTING_STEP_NOT_A_BIFURCATION';break
        write(out/'result.json',dict(status=status,iterations=rows,seconds=time.time()-begin,physical_Floquet='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--dt',type=float,choices=[.05,.025,.0125],default=.05)
    p.add_argument('--krylov',type=int,default=80);p.add_argument('--radius',type=float,default=.03)
    p.add_argument('--linear-tolerance',type=float,default=1e-3)
    p.add_argument('--resume');p.add_argument('--cyclic-right',action='store_true')
    p.add_argument('--target-D',type=float)
    p.add_argument('--target-native-time',type=float)
    p.add_argument('--common-scale',choices=['source','cycle_rms'],default='source')
    p.add_argument('--segments',type=int,choices=[4,6,9,12,18],default=4)
    p.add_argument('--cyclic-M',action='store_true')
    p.add_argument('--condensed-linear',action='store_true')
    p.add_argument('--orthogonal-phase',action='store_true')
    p.add_argument('--refractory-projection',action='store_true')
    p.add_argument('--check-only',action='store_true');main(p.parse_args())

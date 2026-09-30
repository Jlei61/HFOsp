"""Four independent segment times with four phase gauges, same periodic flow.

The extra three segment times are numerical placement freedoms for the same
closed orbit. Three additional node phase conditions remove these freedoms.
No physical state, spatial connection, Z pattern or dynamic M is eliminated.
"""
from common import OUT,np,read,write,log
from core_a_multiple_shooting import setup
from onset_state_continuation import build
from onset_segment_flow import fixed_time,SegmentDerivative
from onset_period_return import dynamical_state,errors
from core_a_periodic_hookstep import krylov
from core_a_positive_newton_coordinates import retract
from pathlib import Path
import argparse,os,time,gc


def cyclic_decay_inverse(rhs,h_over_tau):
    a=np.exp(-np.asarray(h_over_tau));assert a.shape==(4,) and np.all(a<1)
    x=np.empty_like(rhs)
    x[0]=(rhs[3]+a[3]*rhs[2]+a[3]*a[2]*rhs[1]+a[3]*a[2]*a[1]*rhs[0])/(-np.expm1(-sum(h_over_tau)))
    for j in range(3):x[j+1]=a[j]*x[j]+rhs[j]
    return x


def main(a):
    qa=OUT/'core_a_bifurcation_type_20260924/numerical_checks'
    assert read(qa/'refractory_projection/result.json')['status']=='PASS'
    assert read(qa/'variable_segment_M_inverse/result.json')['status']=='PASS'
    if a.border_precondition:
        assert read(qa/'cyclic_phase_time_inverse/result.json')['status']=='PASS'
    source=Path(a.source).resolve();resume=Path(a.resume).resolve();out=Path(a.destination).resolve()
    out.mkdir(parents=True,exist_ok=True);assert not(out/'jobs.json').exists();start=time.time()
    original_T=read(source/'result.json')['period_ms'];rmeta=read(resume/'accepted_period.json')
    write(out/'contract.json',dict(source=str(source),resume=str(resume),dt_ms=.05,
        target_D_A=a.target_D,segments=4,independent_segment_times=True,phase_conditions=4,
        cyclic_phase_time_precondition=a.border_precondition,
        equations='Same3479-group physical delayed spatial rate field, entire Z held, all M dynamic. Optional target_D_A changes only the previously audited native within-Core-A field; outside Z unchanged.',
        numerical_parameterization='Solve Phi(h_j,X_j)-X_(j+1)=0 and normal_j dot(X_j-Xref_j)=0 for four full states and four positive durations. The added three durations are node placement freedoms removed by three extra phase gauges; total physical period=sum(h_j). No new physical parameters.',
        period_interval_ms=[.8*original_T,1.2*original_T],segment_interval_ms=[.5*original_T/4,1.5*original_T/4],
        method='Exact original-flow fixed-time cubic endpoints and full variational products. Shared float64 cache capacity accommodates unequal segment lengths. Cyclic intrinsic-M inverse is exact for unequal durations; all rate feedback remains in actual operator. All four node proposals respect positivity, refractory occupancy and their phase planes.',
        derivative_gate='Independent cached/original products at every node; actual full-state/four-time finite difference. FD history direction points inward at saturated refractory constraints.',
        acceptance='Original four physical matching errors all<1e-7, every physical block<1e-6, all phases<1e-9. Actual nonlinear residual reduction and actual-step variational prediction gate each proposal. Independent complete-cycle/phase/mesh/Floquet and correspondence remain separate.',
        iterations=a.iterations,Krylov=a.krylov,linear_tolerance_cap=a.linear_tolerance,model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    e=build(a.device);parts,T=setup(e,source,resume,a.target_D);N=parts[0].c.size
    h=np.array(rmeta.get('durations_ms',[T/4]*4),dtype=float);assert abs(h.sum()-T)<1e-8
    x=np.concatenate([A.xref for A in parts]);wt=.02;radius=a.radius*np.linalg.norm(x)
    rows=[];status='ITERATION_LIMIT_NOT_A_PERIODIC_ROOT';last_accepted=None
    def evaluate(xx,hh):
        f=[];slopes=[];checks=[];phases=[]
        for j,A in enumerate(parts):
            y,slope=fixed_time(A,xx[j*N:(j+1)*N],hh[j]);nxt=xx[((j+1)%4)*N:((j+1)%4+1)*N]
            f.append(y-nxt);slopes.append(slope)
            checks.append(errors(dynamical_state(A.state(nxt)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum()))
            phases.append(float(A.normal@(xx[j*N:(j+1)*N]-A.xref)))
        return np.r_[np.concatenate(f),-np.array(phases)],slopes,checks,phases
    try:
        for iteration in range(a.iterations):
            step=out/f'iteration{iteration:02d}';step.mkdir(exist_ok=True)
            f,slopes,checks,phases=evaluate(x,h);norm=float(np.linalg.norm(f))
            row=dict(iteration=iteration,period_ms=float(h.sum()),durations_ms=h.tolist(),matching=checks,
                phases=phases,max_combined_relative_rms=max(z['combined_relative_rms'] for z in checks),
                coordinate_residual=norm/np.linalg.norm(x));rows.append(row);write(out/'iterations.json',rows)
            for j,A in enumerate(parts):np.savez_compressed(step/f'node{j:02d}.npz',**A.state(x[j*N:(j+1)*N]))
            jobs.update(iteration=iteration,stage='MATCHING',last_residual=row['max_combined_relative_rms']);write(out/'jobs.json',jobs)
            log('MULTIPHASE MATCHING',iteration,h.tolist(),row['max_combined_relative_rms'])
            if row['max_combined_relative_rms']<1e-7 and max(v['relative_rms'] for z in checks for v in z['blocks'].values())<1e-6 and max(abs(np.array(phases)))<1e-9:
                status='NUMERICAL_MULTIPHASE_SHOOTING_ROOT';break
            if iteration:
                del derivatives,J,shared,propose,right;gc.collect();e.cp.get_default_memory_pool().free_all_blocks()
            capacity=int(np.floor(h.max()/e.dt))+2;required=capacity*44*e.s.P*8
            free,_=e.cp.cuda.runtime.memGetInfo();assert required<.8*free,('Derivative cache memory',required,free)
            shared=e.cp.empty((capacity,44,e.s.P),dtype='f8');derivatives=[]
            for j,A in enumerate(parts):
                size=int(np.floor(h[j]/e.dt))+2
                J=SegmentDerivative(A,x[j*N:(j+1)*N],h[j],shared=shared[:size]);derivatives.append(J)
                if iteration==0:
                    rng=np.random.default_rng(92500+j);v=x[j*N:(j+1)*N]*rng.normal(size=N);v/=np.linalg.norm(v)
                    actual=J(v);old=SegmentDerivative(A,x[j*N:(j+1)*N],h[j],cached=False);expected=old(v)
                    err=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));assert err<1e-10
                    write(step/f'segment{j}_cache_check.json',dict(status='PASS',relative_error=err));del old
                log('MULTIPHASE CACHE',iteration,j)
            def linear(v):
                vv=v[:-4].reshape(4,N);dh=v[-4:]/wt
                states=[vv[(j+1)%4]-derivatives[j](vv[j])-slopes[j]*dh[j] for j in range(4)]
                return np.r_[np.concatenate(states),[A.normal@vv[j] for j,A in enumerate(parts)]]
            if iteration==0:
                rng=np.random.default_rng(92505);v=x*rng.normal(size=x.size)
                hh=v.reshape(4,-1,e.s.P);hh[:,47:]=-abs(hh[:,47:]);v*=norm/np.linalg.norm(v)
                dh=np.array([.001,-.002,.003,-.004]);direction=np.r_[v,wt*dh];actual=linear(direction);fd=[]
                err=np.inf
                for eps in [1e-4,1e-5,1e-6]:
                    xx=x+eps*v
                    if not all(A.admissible(xx[j*N:(j+1)*N]) for j,A in enumerate(parts)):continue
                    ff,_,_,_=evaluate(xx,h+eps*dh)
                    err=float(np.linalg.norm((ff-f)/eps+actual)/np.linalg.norm(actual))
                    fd.append(dict(epsilon=eps,relative_error=err));write(out/'bordered_derivative_check.json',fd)
                    if err<1e-3:break
                assert err<1e-3,fd
                if a.check_only:
                    write(out/'result.json',dict(status='MULTIPHASE_DERIVATIVE_CHECKS_PASS',checks=fd,model_promoted=False))
                    jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);return
            def simple_right(v):
                u=v.copy();u[:-4]=np.roll(v[:-4].reshape(4,N),1,axis=0).ravel()
                u[:-4].reshape(4,-1,e.s.P)[:,4]=cyclic_decay_inverse(v[:-4].reshape(4,-1,e.s.P)[:,4],h/float(e.transport.consts[6].get()))
                return u
            right=simple_right
            if a.border_precondition:
                from core_a_multiple_border_precondition import CyclicBorderInverse
                right=CyclicBorderInverse(slopes,[A.normal for A in parts],e.s.P,h/float(e.transport.consts[6].get()),wt)
                write(step/'border_precondition_check.json',right.verify())
            jobs.update(stage='KRYLOV');write(out/'jobs.json',jobs)
            propose,progress=krylov(lambda v:v-linear(v),f,a.krylov,step,e.s.P,1.,right_override=right,
                operator_description='Four-time four-phase full periodic matching, not Floquet',
                linear_tolerance=min(a.linear_tolerance,max(1e-6,.1*np.sqrt(row['max_combined_relative_rms']))))
            jobs.update(stage='ACTUAL_TRIALS');write(out/'jobs.json',jobs);trials=[];accepted=False
            for attempt in range(9):
                delta,small=propose(radius);hh=h+delta[-4:]/wt
                if not (.8*original_T<hh.sum()<1.2*original_T and np.all(hh>.5*original_T/4) and np.all(hh<1.5*original_T/4)):
                    trials.append(dict(status='OUTSIDE_SEGMENT_TIME_INTERVAL',durations_ms=hh.tolist(),**small));write(step/'trials.json',trials);radius*=.5;continue
                candidates=[];diagnostics=[]
                for j,A in enumerate(parts):
                    d={};q=retract(A,x[j*N:(j+1)*N],delta[j*N:(j+1)*N],1.,project_zero=True,project_all=True,
                        restore_phase=True,orthogonal_phase=True,refractory_projection=True,diagnostics=d)
                    candidates.append(q);diagnostics.append(d)
                if any(q is None for q in candidates):
                    trials.append(dict(status='PROJECTION_NOT_CONVERGED',diagnostics=diagnostics,**small));write(step/'trials.json',trials);radius*=.5;continue
                xx=np.concatenate(candidates);effective=np.r_[xx-x,wt*(hh-h)]
                assert np.linalg.norm(effective)<=radius*1.00001+1e-10
                if sum(t['status']=='EVALUATED' for t in trials)>=4:break
                ff,_,cc,pp=evaluate(xx,hh);pred=f-linear(effective)
                gain=1-float(ff@ff)/(norm*norm);pg=1-float(pred@pred)/(norm*norm);agreement=gain/pg if pg>0 else -np.inf
                trial=dict(status='EVALUATED',period_ms=float(hh.sum()),durations_ms=hh.tolist(),
                    max_combined_relative_rms=max(z['combined_relative_rms'] for z in cc),residual_ratio=float(np.linalg.norm(ff)/norm),
                    predicted_gain=pg,agreement=agreement,diagnostics=diagnostics,**small)
                trials.append(trial);write(step/'trials.json',trials);log('MULTIPHASE TRIAL',trial)
                if gain>0 and agreement>.1:
                    x=xx;h=hh;accepted=True
                    for j,A in enumerate(parts):np.savez_compressed(step/f'accepted_node{j:02d}.npz',**A.state(x[j*N:(j+1)*N]))
                    last_accepted=dict(period_ms=float(h.sum()),durations_ms=h.tolist(),max_combined_relative_rms=trial['max_combined_relative_rms'],directory=str(step))
                    write(step/'accepted_period.json',last_accepted)
                    jobs['last_accepted']=last_accepted;write(out/'jobs.json',jobs)
                    if agreement>.75 and attempt==0:radius=min(2*radius,.2*np.linalg.norm(x))
                    break
                radius*=.5
            row.update(trials=trials,linear_last=progress[-1]);write(out/'iterations.json',rows)
            if not accepted:status='NO_ACCEPTED_MULTIPHASE_STEP_NOT_A_BIFURCATION';break
        write(out/'result.json',dict(status=status,iterations=rows,last_accepted=last_accepted,seconds=time.time()-start,physical_Floquet='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--resume',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--target-D',type=float)
    p.add_argument('--iterations',type=int,default=6);p.add_argument('--krylov',type=int,default=64)
    p.add_argument('--radius',type=float,default=.03);p.add_argument('--linear-tolerance',type=float,default=.01)
    p.add_argument('--border-precondition',action='store_true')
    p.add_argument('--check-only',action='store_true');main(p.parse_args())

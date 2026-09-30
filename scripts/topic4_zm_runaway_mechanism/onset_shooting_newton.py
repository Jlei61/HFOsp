"""Bounded matrix-free Newton correction of the actual section return map.

The time-of-return derivative is included. Full-state variational equations
are differentiated at the same trajectory and mesh; there is no low-rank
replacement of the network. Failed linear/nonlinear solves are not bifurcations.
"""
from common import np,read,write,log
from onset_state_continuation import DEST,build
from onset_poincare_corrector import SectionReturn
from onset_period_return import errors,dynamical_state
from onset_tangent_cuda import Tangent
from fine_rate_frozen_Z_fields import restore
from datetime import datetime
import argparse,os,time


class SectionDerivative:
    def __init__(self,A,x,T,slope):
        self.A=A;self.x=x.copy();self.T=T;self.slope=slope.copy()
        self.normal=A.normal;self.den=float(self.normal@self.slope)
        assert self.den>0
        e=A.e;restore(e,A.state(x));self.t=Tangent(e);self.t.graph()
        self.n=int(np.floor(T/e.dt));self.alpha=T/e.dt-self.n
        self.whole=self.n//round(10/e.dt);tail=self.n%round(10/e.dt)
        with self.t.stream:
            self.t.stream.begin_capture()
            for _ in range(tail):self.t.step()
            self.tail=self.t.stream.end_capture()
        self.calls=0

    def __call__(self,v):
        A=self.A;e=A.e;t=self.t;c=A.c
        restore(e,A.state(self.x));c.set_tangent(t,v)
        for _ in range(self.whole):t.chunk()
        self.tail.launch(t.stream);t.stream.synchronize();left=c.tangent(t)
        t.step();e.cp.cuda.get_current_stream().synchronize();right=c.tangent(t)
        fixed=(1-self.alpha)*left+self.alpha*right
        self.last_return_time_derivative=-float(self.normal@fixed)/self.den
        result=fixed+self.slope*self.last_return_time_derivative
        self.calls+=1
        return result


def gmres_step(J,b,maximum,folder):
    """Small unrestarted full-state Arnoldi solve, with a persisted residual."""
    beta=float(np.linalg.norm(b));V=[b/beta];H=np.zeros((maximum+1,maximum));rows=[]
    for k in range(maximum):
        v=V[k]-J(V[k])
        for _ in range(2):
            for j in range(k+1):
                h=float(V[j]@v);H[j,k]+=h;v-=h*V[j]
        H[k+1,k]=np.linalg.norm(v)
        rhs=np.zeros(k+2);rhs[0]=beta
        coefficient=np.linalg.lstsq(H[:k+2,:k+1],rhs,rcond=None)[0]
        rel=float(np.linalg.norm(H[:k+2,:k+1]@coefficient-rhs)/beta)
        rows.append(dict(dimension=k+1,relative_linear_residual=rel));write(folder/'linear_progress.json',rows)
        log('ONSET NEWTON KRYLOV',k+1,rel)
        if rel<1e-3 or H[k+1,k]<1e-14:break
        if k+1<maximum:V.append(v/H[k+1,k])
    delta=sum(a*v for a,v in zip(coefficient,V))
    np.savez_compressed(folder/'linear_system.npz',H=H,coefficients=coefficient)
    return delta,rows


def run(device,maximum):
    parent=DEST/'native9420_from_lower';source=parent/'shooting_corrector_dt0p05'
    assert read(source/'independent_validation/result.json')['status']=='SAME_PHASE_ROOT_VERIFIED'
    assert read(DEST/'tangent_implementation/full_state_check.json')['status']=='PASS'
    folder=parent/'newton_native9425_dt0p05';folder.mkdir(exist_ok=True);assert not (folder/'jobs.json').exists()
    write(folder/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can an actual full-state Newton correction improve the adjacent native9425 periodic shooting equation when bounded Picard/Anderson correction did not?',
        source=str(source/'latest_state.npz'),target_field='native9425',dt_ms=.05,M_dynamic=True,
        derivative='DP = A_T - slope*(normal dot A_T)/(normal dot slope), including the derivative of return time for the exact linearly interpolated section crossing. A_T uses verified actual variational equations.',
        check='One-sided derivatives along residual P(x)-x at epsilon1e-3 and5e-4 preserve physical admissibility by convex interpolation. RelativeJVPerror<1e-3 and phase leakage<1e-9 required before linear solve.',
        budget=f'One nonlinear Newton correction only. Two finite-difference section returns, one derivative verification, at most{maximum}Krylovproducts and3physicallyadmissible line-searchreturns. No branchcampaign.',
        acceptance='A nonlinear residual decrease is solver evidence only. A new orbit requires the existing full residual and independent checks; a solver failure is not a bifurcation.',model_promoted=False))
    status=dict(status='RUNNING',pid=os.getpid(),stage='setup');write(folder/'jobs.json',status);start=time.time()
    try:
        e=build(device);base={k:v for k,v in np.load(source/'latest_state.npz').items()}
        base['syn'][5]=np.load(DEST/'fields.npz')['native9425']
        T=read(source/'result.json')['period_ms'];A=SectionReturn(base,e,T,20.)
        x=A.xref.copy();y,info=A(x);f=y-x;slope=A.last_time_slope.copy()
        initial=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
        J=SectionDerivative(A,x,info['period_ms'],slope);v=f;derivative=J(v);checks=[]
        status.update(stage='section_derivative_check');write(folder/'jobs.json',status)
        for eps in [1e-3,5e-4]:
            pert=x+eps*v;assert A.admissible(pert)
            q,meta=A(pert);fd=(q-y)/eps
            row=dict(epsilon=eps,relative_error=float(np.linalg.norm(fd-derivative)/max(np.linalg.norm(derivative),1e-30)),
                     phase_leakage=float(abs(A.normal@derivative)),perturbed_return_ms=meta['period_ms'])
            checks.append(row);write(folder/'derivative_check.json',checks);log('ONSET SECTION DERIVATIVE',row)
        passed=checks[-1]['relative_error']<1e-3 and checks[-1]['phase_leakage']<1e-9
        if not passed:
            write(folder/'result.json',dict(status='SECTION_DERIVATIVE_GATE_FAILED',checks=checks,model_promoted=False))
            status.update(status='COMPLETE_NEGATIVE');write(folder/'jobs.json',status);return
        status.update(stage='Newton_GMRES');write(folder/'jobs.json',status)
        delta,linear=gmres_step(J,f,maximum,folder)
        np.savez_compressed(folder/'newton_step.npz',delta=delta,source=x,normal=A.normal)
        direction_phase=float(A.normal@delta);assert abs(direction_phase)<1e-8*max(np.linalg.norm(delta),1.)
        trials=[];accepted=False;attempts=0
        # Admissibility checks do not integrate and do not change the model.
        for alpha in [1.,.5,.25,.125,.0625,.03125,.015625]:
            candidate=x+alpha*delta
            if not A.admissible(candidate):
                trials.append(dict(alpha=alpha,status='INADMISSIBLE_NO_FLOW'));continue
            if attempts>=3:break
            attempts+=1;value,meta=A(candidate)
            err=errors(dynamical_state(A.state(candidate)),dynamical_state(A.state(value)),e.s.sizes/e.s.sizes.sum())
            norm=float(np.linalg.norm(value-candidate));ratio=norm/np.linalg.norm(f)
            row=dict(alpha=alpha,status='EVALUATED',weighted_residual_ratio=ratio,**meta,**err);trials.append(row)
            write(folder/'line_search.json',trials);log('ONSET NEWTON LINE',alpha,ratio,err['combined_relative_rms'])
            if ratio<1 and err['combined_relative_rms']<initial['combined_relative_rms']:
                accepted=True;np.savez_compressed(folder/'corrected_state.npz',**A.state(candidate));break
        write(folder/'result.json',dict(status='ONE_NEWTON_STEP_IMPROVED_NOT_AN_ORBIT' if accepted else 'BOUNDED_NEWTON_STEP_NOT_ACCEPTED',
            source_error=initial,source_return=info,derivative_checks=checks,linear_progress=linear,line_search=trials,
            derivative_calls=J.calls,return_calls=A.calls,seconds=time.time()-start,
            scope='One bounded solver diagnostic; no branch disappearance, periodic stability or bifurcation certification.',model_promoted=False))
        status.update(status='COMPLETE');write(folder/'jobs.json',status)
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc));write(folder/'jobs.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--krylov',type=int,default=6)
    a=p.parse_args();assert 2<=a.krylov<=12;run(a.device,a.krylov)

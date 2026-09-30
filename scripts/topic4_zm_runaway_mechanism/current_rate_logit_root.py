"""Bound-preserving numerical coordinates for the SAME equilibrium equation.

No network fitting: q=log(r*tref/(1-r*tref)) is solely a solver coordinate.
The failed direct-rate Newton result is retained unchanged.
"""
from common import OUT, np, write, read, log
from current_rate_characteristic import CurrentRateCharacteristic, DT0
from fine_rate_frozen_Z_fields import native_field
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logit
from datetime import datetime
import os

DEST=OUT/'current_rate_analysis_interface'


def value_and_jacobian(s,q,jacobian=True):
    r=expit(q)/s.ref;op=s.local_operating(*s.moments(r))
    F=op['log_hazard']+np.log(s.ref/DT0)-q
    if not jacobian:return r,F,op
    grad=op['base_gradient']+op['feature_gradient'][:,:3]*op['input_normalization_gradient']
    L=sum(sparse.diags(grad[:,ch])@mat for ch,mat in enumerate(s.temporal_components(0.)))
    J=L@sparse.diags(r*(1-r*s.ref))-sparse.eye(s.P)
    return r,F,op,J.tocsc()


def main():
    assert read(DEST/'single_equilibrium_root.json')['status']=='ROOT_FAIL'
    path=DEST/'logit_solver_contract.json';assert not path.exists()
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
        reason='Direct-rate Newton reduces residual only111to110Hz in40iterations; diagnose positivity boundary restriction then use an invertible coordinate, not more iterations of the same stalled solve.',
        unchanged='SamefullQequations, originalnative9870Zfield, constantmeaninput, dynamicM, sameoriginaltailmeaninitialguess. No target fitting or physicalparameterchange.',
        solver='q=logit(r*tref); atmost60Newtonsteps,per-step infinity radius4,30backtracks; originalrate-residual1e-11perms required. Analytic q-Jacobian checked against independent finite differences before solving.',
        output='Separate logit result, no branch search or stability promotion. One numerical initial-guess floor1e-8 in r*tref is recorded, never a change to model response or accepted equilibrium.'))
    write(DEST/'logit_jobs.json',dict(status='RUNNING',pid=os.getpid()))
    s=CurrentRateCharacteristic(40);s.set_Z(native_field(s,9870))
    z=np.load(DEST/'native9870_fullQ_equilibrium.npz');initial=z['initial_guess']
    f=s.residual(initial);delta=spsolve(s.jacobian(initial),-f)
    allowed=np.full(s.P,np.inf);negative=delta<0;allowed[negative]=(initial[negative]+1e-11)/(-delta[negative])
    i=int(np.argmin(allowed));diagnostic=dict(limiting_group=i,group_population='E' if s.E[i] else 'I',
        initial_rate_per_ms=float(initial[i]),newton_direction_per_ms=float(delta[i]),positivity_step_bound=float(allowed[i]))
    q=logit(np.clip(initial*s.ref,1e-8,1-1e-8));rng=np.random.default_rng(920092)
    r,F,op,J=value_and_jacobian(s,q);checks=[]
    for h in [1e-4,5e-5]:
        v=rng.normal(size=s.P);num=(value_and_jacobian(s,q+h*v,False)[1]-value_and_jacobian(s,q-h*v,False)[1])/(2*h)
        err=float(np.linalg.norm(J@v-num)/np.linalg.norm(num));assert err<2e-5,err;checks.append(err)
    trace=[];ok=False
    for it in range(60):
        r,F,op,J=value_and_jacobian(s,q);rate_error=float(np.max(abs(op['rate']-r)));err=float(np.max(abs(F)))
        trace.append(dict(iteration=it,rate_residual_per_ms=rate_error,logit_residual=err));log('CURRENT LOGIT NEWTON',it,rate_error,err)
        if rate_error<1e-11:ok=True;break
        step=spsolve(J,-F);alpha=min(1.,4/max(np.max(abs(step)),1e-12))
        for back in range(30):
            qt=q+alpha*step;rr,ff,_=value_and_jacobian(s,qt,False)
            if np.max(abs(ff))<err:q=qt;break
            alpha*=.5
        else:break
    r,F,op=value_and_jacobian(s,q,False);error=float(np.max(abs(s.residual(r))));ok=bool(error<1e-11)
    np.savez_compressed(DEST/'native9870_fullQ_equilibrium_logit.npz',r=r,Z=s.Z,q=q,initial_guess=initial)
    result=dict(status='ROOT_PASS' if ok else 'ROOT_FAIL',residual_per_ms=error,trace=trace,
        direct_Newton_boundary_diagnostic=diagnostic,coordinate_Jacobian_errors=checks,
        global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r),D=s.D,
        numerical_coordinates_only=True,stability='NOT_COMPUTED',model_promoted=False)
    write(DEST/'single_equilibrium_logit_root.json',result);write(DEST/'logit_jobs.json',dict(status=result['status'],pid=os.getpid()))
    log('CURRENT LOGIT ROOT',result['status'],error,s.global_rate(r))


if __name__=='__main__':main()

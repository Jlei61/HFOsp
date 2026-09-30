"""Bounded trust-region root correction, unchanged current full-Q equations."""
from common import OUT,np,read,write,log,time
from current_rate_characteristic import CurrentRateCharacteristic
from fine_rate_frozen_Z_fields import native_field
from scipy.optimize import least_squares
from datetime import datetime
import os

DEST=OUT/'current_rate_analysis_interface'


def main():
    assert read(DEST/'single_equilibrium_logit_root.json')['status']=='ROOT_FAIL'
    contract=DEST/'trust_solver_contract.json';assert not contract.exists()
    write(contract,dict(created_local=datetime.now().astimezone().isoformat(),
        purpose='Numerical root ofsameonefixedfield; line-search Newtonfailure is not a bifurcation. This does not fit data or parameters.',
        unchanged='Full diffusion, native9870ms Z field, constant originalmeanexternalinput, dynamicM, frozenconditioned39 response, g40 graph.',
        method='scipy least_squares trust-region reflective onoriginalrate residual, sparseanalyticJacobian, bounds0to1/tref, x_scale=jac; max50functionevaluations and500LSMRiterations perlinear solve. Original1e-11permsrootresidual required independently.',
        limits='No parameter/initial-field scan, no addedresponsefit, branch or eigenvalues; retainnonzero minimum as failure. Priorfailedroots remain unchanged.'))
    s=CurrentRateCharacteristic(40);s.set_Z(native_field(s,9870));z=np.load(DEST/'native9870_fullQ_equilibrium.npz')
    initial=np.minimum(np.maximum(z['initial_guess'],1e-10),1/s.ref-1e-10);trace=[];start=time.time()
    def function(r):
        f=s.residual(r);entry=dict(evaluation=len(trace),residual_per_ms=float(np.max(abs(f))),L2=float(np.linalg.norm(f)),elapsed_seconds=time.time()-start)
        trace.append(entry);write(DEST/'trust_jobs.json',dict(status='RUNNING',pid=os.getpid(),latest=entry));log('CURRENT TRUST ROOT',entry)
        return f
    result=least_squares(function,initial,jac=s.jacobian,bounds=(np.zeros(s.P),1/s.ref),x_scale='jac',
        ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=50,tr_solver='lsmr',
        tr_options=dict(maxiter=500,atol=1e-10,btol=1e-10))
    r=result.x;error=float(np.max(abs(s.residual(r))));ok=bool(error<1e-11)
    np.savez_compressed(DEST/'native9870_fullQ_equilibrium_trust.npz',r=r,Z=s.Z,initial_guess=initial)
    status='ROOT_PASS' if ok else 'ROOT_FAIL'
    write(DEST/'single_equilibrium_trust_root.json',dict(status=status,residual_per_ms=error,nfev=result.nfev,trace=trace,
        optimizer_status=int(result.status),optimizer_message=result.message,global_rate_hz=s.global_rate(r),
        regional_rates_hz=s.regional_rates(r),D=s.D,stability='NOT_COMPUTED',model_promoted=False))
    write(DEST/'trust_jobs.json',dict(status=status,pid=os.getpid()));log('CURRENT TRUST COMPLETE',status,error)


if __name__=='__main__':main()

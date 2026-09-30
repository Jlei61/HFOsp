"""Trust-region correction of the exact local-family static equation."""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy.optimize import least_squares
import os,time

DEST=OUT/'core_a_bifurcation_type_20260924/static_trust'
SOURCE=OUT/'core_a_bifurcation_type_20260924/static_candidates'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    assert read(SOURCE/'mean.json')['status']=='ROOT_FAIL'
    s=PhysicalDelayConditionalDrift();z=np.load(SOURCE/'mean.npz');s.set_Z(z['Z'])
    write(DEST/'contract.json',dict(question='Can a trust region resolve the stalled stationary candidate at the same Core A endpoint?',
        unchanged='Exact same static equations and Z/M constraints. Only numerical solver changes; no network parameter or target response fitting.',
        solver='Logit-equation least_squares TRF with analytic sparse Jacobian, x_scale=jac, LSMR at most1000linear iterations, at most180function evaluations. Original rate residual<1e-11/ms required; nonzero minima remain failures.',
        initial=str(SOURCE/'mean.npz'),model_promoted=False))
    start=time.time();trace=[];cache={}
    def evaluate(q):
        if 'q' not in cache or not np.array_equal(q,cache['q']):
            cache['q']=q.copy();cache['value']=value_and_jacobian(s,q)
        return cache['value']
    def function(q):
        r,F,op,J=evaluate(q)
        row=dict(evaluation=len(trace),L2_logit_residual=float(np.linalg.norm(F)),rate_residual_per_ms=float(abs(op['rate']-r).max()),seconds=time.time()-start)
        trace.append(row);write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid(),latest=row));log('CORE A STATIC TRUST',row)
        return F
    def jac(q):return evaluate(q)[3]
    result=least_squares(function,z['q'],jac=jac,x_scale='jac',tr_solver='lsmr',
        tr_options=dict(maxiter=1000,atol=1e-10,btol=1e-10),max_nfev=180,
        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    r,F,op,J=evaluate(result.x);error=float(abs(s.residual(r)).max());status='ROOT_PASS' if error<1e-11 else 'ROOT_FAIL'
    np.savez_compressed(DEST/'candidate.npz',r=r,Z=s.Z,q=result.x)
    write(DEST/'result.json',dict(status=status,residual_per_ms=error,trace=trace,global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r),
        optimizer_message=result.message,nfev=result.nfev,stability='NOT_COMPUTED',model_promoted=False))
    write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid(),scientific_result=status))


if __name__=='__main__':main()

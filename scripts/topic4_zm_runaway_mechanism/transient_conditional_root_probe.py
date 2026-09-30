"""Four bounded equilibrium solves of the current conditional drift.

The history-quadratic transient correction is zero, with zero derivative,
at stationary histories. Thus the existing exact static interface applies.
Neither a converged root nor failure establishes an onset bifurcation.
"""
from common import OUT, np, read, write, log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy.special import logit
from scipy.sparse.linalg import spsolve
from datetime import datetime
import os

SOURCE = OUT/'transient_autonomous_Z_probe_20260923'
DEST = OUT/'transient_conditional_root_probe_20260923'


def main():
    assert read(SOURCE/'independent_audit.json')['status'] == 'READOUT_AUDIT_PASS'
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json', dict(created_local=datetime.now().astimezone().isoformat(),
        question='Are actual static roots near the native preentry/entry Z fields accessible from low and time-mean high guesses, rather than mistaking an oscillatory mean for an equilibrium?',
        rationale='Both autonomous five-second probes remain spatially time dependent; their mean static residuals exceed200Hz. Independent fixed-point solving is required before drawing any equilibrium branch.',
        equations='Same g40 physical-delay private-Q conditional drift, original constant external mean, full native spatial Z held, M dynamic with equilibrium M=.5*E*r. Locked transient correction and its first derivative vanish at a stationary state.',
        seeds='At native9420 and9870 fullZ fields: previously verified Z1 low root, and the corresponding final2s conditional-drift mean. Four independent solves; neither initial guess is assumed a root.',
        solver='Invertible logit rate coordinates; maximum40Newton iterations, infinity-step radius4,30backtracks. Require original rate residual <1e-11perms. Analytic coordinate Jacobian checked by two centered finite-difference steps.',
        scope='Root existence and numerical identity only. Native/local correspondence failures retained. No named bifurcation, stability, fitted response or physical parameter change.',
        budget='Four bounded solves plus finite-difference audits. If successful, actual corrected-engine stationary flow identity is a required separate numerical verification.',
        model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[])
    write(DEST/'jobs.json',jobs)
    s=PhysicalDelayConditionalDrift()
    baseline=np.load(OUT/'physical_delay_conditional_drift_interface/baseline_equilibrium.npz')['r']
    rows=[]
    for tm in [9420,9870]:
        z=np.load(SOURCE/str(tm)/'trajectory.npz');s.set_Z(z['Z'])
        for name,initial in [('low',baseline),('tail_mean',z['mean_tail_rate_per_ms'])]:
            label=f'{tm}_{name}';q=logit(np.clip(initial*s.ref,1e-8,1-1e-8))
            rng=np.random.default_rng(92317);v=rng.normal(size=s.P)
            r,F,op,J=value_and_jacobian(s,q);checks=[]
            for h in [1e-4,5e-5]:
                fd=(value_and_jacobian(s,q+h*v,False)[1]-value_and_jacobian(s,q-h*v,False)[1])/(2*h)
                error=float(np.linalg.norm(J@v-fd)/np.linalg.norm(fd));assert error<2e-5
                checks.append(error)
            trace=[]
            for it in range(40):
                r,F,op,J=value_and_jacobian(s,q);residual=float(abs(op['rate']-r).max());err=float(abs(F).max())
                trace.append(dict(iteration=it,rate_residual_per_ms=residual,logit_residual=err))
                log('CONDITIONAL ROOT',label,it,residual,err)
                if residual<1e-11:break
                step=spsolve(J,-F);alpha=min(1.,4/max(float(abs(step).max()),1e-12))
                for back in range(30):
                    qt=q+alpha*step;_,ff,_=value_and_jacobian(s,qt,False)
                    if abs(ff).max()<err:q=qt;break
                    alpha*=.5
                else:break
            r,F,op=value_and_jacobian(s,q,False);residual=float(abs(s.residual(r)).max())
            row=dict(label=label,status='ROOT_PASS' if residual<1e-11 else 'ROOT_FAIL',D=s.D,
                global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r),
                residual_per_ms=residual,trace=trace,coordinate_Jacobian_relative_errors=checks,
                stability='NOT_COMPUTED',model_promoted=False)
            np.savez_compressed(DEST/f'{label}.npz',r=r,Z=s.Z,initial_guess=initial,q=q)
            write(DEST/f'{label}.json',row);rows.append(row)
            jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
            log('CONDITIONAL ROOT RESULT',label,row['status'],residual,row['global_rate_hz'])
    write(DEST/'result.json',dict(status='COMPLETE',rows=rows,model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    main()

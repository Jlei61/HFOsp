"""Find actual stationary states at the local-resource endpoint.

The solver changes numerical coordinates/merit only. A stationary temporal
mean is not presumed, and unsuccessful candidates are never drawn as roots.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy.sparse.linalg import spsolve,lsmr
from scipy.special import logit
from pathlib import Path
import argparse,os,time

DEST=OUT/'core_a_bifurcation_type_20260924/static_candidates'
SOURCE=OUT/'core_a_resource_bifurcation_20260923'


def solve(s,initial,label):
    q=logit(np.clip(initial*s.ref,1e-10,1-1e-10));trace=[];start=time.time()
    for it in range(70):
        r,F,op,J=value_and_jacobian(s,q);norm=float(np.linalg.norm(F));error=float(abs(op['rate']-r).max())
        trace.append(dict(iteration=it,L2_logit_residual=norm,rate_residual_per_ms=error))
        write(DEST/f'{label}_progress.json',dict(pid=os.getpid(),trace=trace,seconds=time.time()-start))
        log('CORE A STATIC',label,it,error,norm)
        if error<1e-11:break
        if len(trace)>10 and norm>trace[-10]['L2_logit_residual']*(1-1e-7):break
        newton=spsolve(J,-F)
        directions=[newton]
        accepted=False
        for attempt in range(2):
            if attempt==1:
                # Numerical residual descent, not a dynamical-network change.
                directions=[lsmr(J,-F,damp=.1,atol=1e-8,btol=1e-8,maxiter=250)[0]]
            for step in directions:
                alpha=min(1.,3/max(float(abs(step).max()),1e-12))
                for back in range(22):
                    trial=q+alpha*step;_,ff,_=value_and_jacobian(s,trial,False)
                    if np.linalg.norm(ff)<norm*(1-1e-4*alpha):
                        q=trial;accepted=True;break
                    alpha*=.5
                if accepted:break
            if accepted:break
        if not accepted:break
    r,F,op=value_and_jacobian(s,q,False);error=float(abs(s.residual(r)).max())
    np.savez_compressed(DEST/f'{label}.npz',r=r,Z=s.Z,q=q,initial_guess=initial)
    row=dict(label=label,status='ROOT_PASS' if error<1e-11 else 'ROOT_FAIL',
        residual_per_ms=error,global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r),
        trace=trace,seconds=time.time()-start,stability='NOT_COMPUTED',model_promoted=False)
    write(DEST/f'{label}.json',row);return row


def main():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(question='Does the observed locally high state have a stationary counterpart in the same full network, needed before testing an equilibrium SN/Hopf explanation?',
        equations='Same physical-private-Q static interface, original graph and thresholds, only native CoreA10370 Z changed against native9s outside Z. Dynamic M stationary constraint m=.5*E*r. Locked transient correction and first derivative vanish at stationary histories.',
        solver='Logit coordinates with L2 residual Armijo merit, exact sparse Jacobian, numerical step radius3. Damped LSMR descent only if Newton line search fails. At most70iterations per seed; stop on sustained residual stagnation. Accept only original rate residual<1e-11/ms.',
        seeds='Observed last5s mean, final instantaneous group rate, and regional flattened mean (numerical guesses only). No claim that a trajectory mean is a root.',
        scope='No root failure implies nonexistence, and no root existence implies the observed onset transition. Roots require separate actual-engine and temporal-stability checks.'))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'jobs.json',jobs)
    s=PhysicalDelayConditionalDrift();z=np.load(SOURCE/'fields.npz')['coreA10370_background9000'];s.set_Z(z)
    block=np.load(SOURCE/'coreA_depleted/block01.npz');mean=block['group_rate_hz'].mean(0).astype(float)/1000
    instant=np.load(SOURCE/'coreA_depleted/final_state.npz')['rate']
    flat=mean.copy()
    for pop in [s.E,~s.E]:
        for region in [0,1,2]:
            m=pop&(s.geo['group_region']==region)
            if m.any():flat[m]=np.average(mean[m],weights=s.sizes[m])
    rows=[]
    for label,initial in [('mean',mean),('instant',instant),('regional',flat)]:
        row=solve(s,initial,label);rows.append(row);jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
        if row['status']=='ROOT_PASS':break
    write(DEST/'result.json',dict(status='COMPLETE',rows=rows,bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


if __name__=='__main__':main()

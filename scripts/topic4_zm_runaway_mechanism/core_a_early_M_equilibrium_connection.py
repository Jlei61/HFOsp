"""Connect a known root to the actual pre-long-event frozen-M fast system.

The auxiliary interpolation is only an equilibrium solver homotopy. Its
turns are not physical Z bifurcations and are never plotted as such.
"""
from common import OUT,np,read,write,log
from core_a_frozen_M_static import FrozenMDrift
from current_rate_logit_root import value_and_jacobian
from scipy.sparse.linalg import spsolve
import os,time

TYPE=OUT/'core_a_bifurcation_type_20260924'
REF=TYPE/'reference_stability_gap'
DEST=REF/'early_M_fast_equilibrium_connection'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    source=TYPE/'equilibrium_branch/point000.npz'
    target=REF/'first_long_M_intervention_from2800/replayed_initial.npz'
    assert read(REF/'first_long_frozen_M_stationary/implementation_check.json')['status']=='PASS'
    assert read(REF/'first_long_M_intervention_from2800/independent_audit.json')['status']=='PASS'
    z=np.load(source);base=np.load(target)
    # The known dynamic-M equilibrium is also an exact frozen-M equilibrium
    # when its M is prescribed at .5*E*r. No physical coefficients change.
    probe=FrozenMDrift(base['syn'][4]);M0=.5*probe.E*z['r'];Z0=z['Z'].copy()
    M1=base['syn'][4].copy();Z1=base['syn'][5].copy();s=probe
    mask=s.E&(s.geo['group_region']==0);assert np.array_equal(Z0[~mask],Z1[~mask])
    write(DEST/'contract.json',dict(source=str(source),target_slow_field=str(target),
        question='At the actual spatial M just before the first prolonged activity, is there a fast-system equilibrium connected to an already verified root? This can nominate a relevant fast fold/Hopf for a slow-passage explanation, distinct from full-dynamic-M periodic bifurcations.',
        model='The same3479-group spatial graph, response, physical private variance, constant external mean and synaptic/delay equations. This diagnostic alone holds all M and Z. Its target is exactly the existing2.8s held-M intervention field, not a scalar M substitution. Main trajectories and periodic branches keep M dynamic.',
        numerical='Start at a verified full-dynamic-M equilibrium, hold M at its exact equilibrium value .5*E*r, and interpolate prescribed M and CoreA Z to the actual target. Predictor and logit Newton solve the unchanged fast stationary equations. The homotopy coordinate is numerical, not a physical bifurcation axis.',
        acceptance='Original fast-system rate residual<1e-11/ms at every accepted point. Existing independently checked frozen-M Jacobian is reused. Root at the target still needs original-engine, stability and actual-path checks; numerical stalls do not prove nonexistence or a fold.',
        budget='At most40homotopy trials,12Newton steps each; step below.002 stops. No automatic physical branch or stability scan from auxiliary turns.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(DEST/'jobs.json',jobs);started=time.time()
    def set_point(a):
        s.held_M=(1-a)*M0+a*M1;s.set_Z((1-a)*Z0+a*Z1)
    def evaluate(q,a,jac=True):
        set_point(a);return value_and_jacobian(s,q,jac)
    def save(q,a):
        r,F,op=evaluate(q,a,False);err=float(abs(s.residual(r)).max());assert err<1e-11,err
        row=dict(index=len(accepted),auxiliary_coordinate=float(a),residual_per_ms=err,
            global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r))
        np.savez_compressed(DEST/f'point{len(accepted):03d}.npz',q=q,r=r,Z=s.Z,M=s.held_M,auxiliary_coordinate=a)
        accepted.append(row);write(DEST/'accepted.json',accepted);log('EARLY M FAST ROOT',row)
    try:
        q=z['q'].copy();a=0.;step=.05;accepted=[];trials=[];save(q,a)
        status='NUMERICAL_TRIAL_LIMIT'
        for k in range(40):
            b=min(1.,a+step);r,F,op,J=evaluate(q,a)
            h=1e-5;lo=max(0.,a-h);hi=min(1.,a+h)
            Fp=evaluate(q,hi,False)[1];Fm=evaluate(q,lo,False)[1]
            dq=spsolve(J,-(Fp-Fm)/(hi-lo));trial=q+(b-a)*dq
            trace=[];passed=False
            for it in range(12):
                rr,ff,oo,jj=evaluate(trial,b);norm=float(np.linalg.norm(ff));err=float(abs(oo['rate']-rr).max())
                trace.append(dict(iteration=it,rate_residual_per_ms=err,logit_L2=norm))
                if err<1e-11:passed=True;break
                delta=spsolve(jj,-ff);alpha=min(1.,3./max(abs(delta).max(),1e-12));ok=False
                for back in range(18):
                    candidate=trial+alpha*delta;fc=evaluate(candidate,b,False)[1]
                    if np.linalg.norm(fc)<norm*(1-1e-4*alpha):trial=candidate;ok=True;break
                    alpha*=.5
                if not ok:break
            trials.append(dict(trial=k,target_auxiliary_coordinate=b,step=step,accepted=passed,trace=trace))
            write(DEST/'trials.json',trials)
            if passed:
                q=trial;a=b;save(q,a)
                if a==1.:status='TARGET_FROZEN_M_FAST_EQUILIBRIUM_ROOT';break
                step=min(.15,step*(1.3 if len(trace)<=5 else .9))
            else:
                step*=.5
                if step<.002:status='NUMERICAL_STEP_STALL_NOT_A_BIFURCATION';break
        write(DEST/'result.json',dict(status=status,accepted_points=len(accepted),trials=len(trials),
            final_auxiliary_coordinate=a,seconds=time.time()-started,
            fast_system_bifurcation='NOT_ESTABLISHED',full_system_onset_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':main()

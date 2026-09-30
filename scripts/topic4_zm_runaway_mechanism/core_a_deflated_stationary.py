"""Check for a missed asymmetric equilibrium without changing the model.

Shifted norm deflation excludes an already known stationary root from the
numerical merit. Only distinct roots of the original unmodified residual
are accepted; auxiliary merit turns never count as network bifurcations.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy.sparse.linalg import spsolve
from scipy.special import logit
import os,time

DEST=OUT/'core_a_bifurcation_type_20260924/asymmetric_stationary_deflation'


def factor(q,known):
    d=q-known;r2=float(d@d/len(d));assert r2>1e-24
    return 1.+1./r2,-2*d/(len(d)*r2*(1.+r2)),np.sqrt(r2)


def main():
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists();start=time.time()
    root=DEST.parent/'numerical_root_homotopy/physical_candidate.npz';old=np.load(root)
    s=PhysicalDelayConditionalDrift();s.set_Z(old['Z']);known=old['q']
    assert np.max(abs(s.residual(old['r'])))<1e-11
    source=OUT/'core_a_resource_bifurcation_20260923/coreA_depleted/block01.npz'
    rates=np.load(source)['group_rate_hz'].astype(float)/1000
    A=s.E&(s.geo['group_region']==0);B=s.E&(s.geo['group_region']==1)
    ar=rates[:,A]@(s.sizes[A]/s.sizes[A].sum());br=rates[:,B]@(s.sizes[B]/s.sizes[B].sum())
    possible=np.flatnonzero((ar>.25)&(br<.005));assert len(possible)
    j=int(possible[np.argmin(br[possible])]);instant=rates[j].copy();mean=rates.mean(0);mean[B]=.0001
    write(DEST/'contract.json',dict(
        question='Have previous searches missed an asymmetric high-A/low-B equilibrium because they all converged to the same high-both root?',
        equations='Same original stationary physical-private-Q interface and native Core-A-only Z field at the existing stronger endpoint. All dynamical M impose their stationary law; no M clamp or new physical parameter. Original full temporal dynamics remain required for stability.',
        known_root=str(root),source=str(source),instant_source_bin=j,
        numerical_seeds=['Actual high-A/B-quiet spatial instantaneous rates','Actual spatial mean with only initial B guess100mHz'],
        method='Shifted norm deflation G=(1+1/rho^2)*F, rho=RMS(q-q_known), in original logit-rate coordinates. Sherman-Morrison gives delta=delta_Newton/(1-grad(log factor) dot delta_Newton). Numerical merit descent and full actual derivative checks; no deflated object is a new physical branch.',
        reference='https://arxiv.org/abs/1410.5620',
        gates='Original physical residual<1e-11/ms and RMS logit separation from known root>1e-3. New roots with A>200Hz and B<50Hz are relevant asymmetric candidates, not automatic onset certificates. Do not continue additional high-both peripheral roots.',
        budget='Two numerical initial guesses at one already studied Z field, maximum80iterations each; no physical parameter sweep. Failure does not prove absence or uniqueness.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'jobs.json',jobs);results=[]
    for label,initial in [('actual_B_quiet',instant),('mean_lowB',mean)]:
        q=logit(np.clip(initial*s.ref,1e-10,1-1e-10));rows=[];status='NO_DISTINCT_ROOT_ESTABLISHED'
        for iteration in range(80):
            r,F,op,J=value_and_jacobian(s,q);m,g,distance=factor(q,known)
            norm=np.linalg.norm(F);merit=m*norm;error=float(np.max(abs(s.residual(r))))
            row=dict(iteration=iteration,original_rate_residual_per_ms=error,deflated_merit=float(merit),distance_from_known=float(distance))
            rows.append(row);write(DEST/f'{label}_iterations.json',rows);log('ASYMMETRIC DEFLATED ROOT',label,row)
            if error<1e-11 and distance>1e-3:status='DISTINCT_ORIGINAL_EQUILIBRIUM_ROOT';break
            if iteration==0:
                rng=np.random.default_rng(92496);v=rng.normal(size=s.P);v/=np.linalg.norm(v)
                derivative=m*(J@v+F*(g@v));checks=[]
                for eps in [1e-4,1e-5,1e-6]:
                    values=[]
                    for sign in [-1,1]:
                        qq=q+sign*eps*v;values.append(factor(qq,known)[0]*value_and_jacobian(s,qq,False)[1])
                    relative=float(np.linalg.norm((values[1]-values[0])/(2*eps)-derivative)/np.linalg.norm(derivative))
                    checks.append(dict(epsilon=eps,relative_error=relative))
                    if relative<1e-5:break
                write(DEST/f'{label}_deflated_derivative_check.json',checks);assert relative<1e-5
            newton=spsolve(J,-F);den=float(1-g@newton);directions=[]
            if abs(den)>1e-12:directions.append(newton/den)
            directions.append(-(J.T@F+g*(F@F)))
            accepted=False
            for delta in directions:
                if not np.isfinite(delta).all():continue
                alpha=min(1.,3./max(float(abs(delta).max()),1e-30))
                for _ in range(24):
                    qq=q+alpha*delta
                    if np.max(abs(qq))<80 and np.linalg.norm(qq-known)/np.sqrt(s.P)>1e-10:
                        ff=value_and_jacobian(s,qq,False)[1];next_merit=factor(qq,known)[0]*np.linalg.norm(ff)
                        if next_merit<merit*(1-1e-4*alpha):q=qq;accepted=True;break
                    alpha*=.5
                if accepted:break
            if not accepted:break
            if len(rows)>12 and merit>rows[-12]['deflated_merit']*(1-1e-7):break
        r,F,op=value_and_jacobian(s,q,False);error=float(np.max(abs(s.residual(r))))
        distance=factor(q,known)[2]
        if error<1e-11 and distance>1e-3:status='DISTINCT_ORIGINAL_EQUILIBRIUM_ROOT'
        ra=float(np.average(r[A],weights=s.sizes[A])*1000);rb=float(np.average(r[B],weights=s.sizes[B])*1000)
        np.savez_compressed(DEST/f'{label}.npz',r=r,q=q,Z=s.Z,initial=initial)
        row=dict(label=label,status=status,original_rate_residual_per_ms=error,distance_from_known=float(distance),
            CoreA_hz=ra,CoreB_hz=rb,regional_rates_hz=s.regional_rates(r),
            relevant_asymmetric_candidate=bool(status.startswith('DISTINCT') and ra>200 and rb<50),
            dynamical_stability='NOT_COMPUTED',bifurcation_type='NOT_ESTABLISHED')
        results.append(row);write(DEST/f'{label}_result.json',row);jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
        if row['relevant_asymmetric_candidate']:break
    write(DEST/'result.json',dict(status='BOUNDED_DISTINCT_ROOT_SEARCH_COMPLETE',rows=results,seconds=time.time()-start,
        interpretation='Only distinct original equilibria count; no failed solve proves nonexistence and no root alone certifies onset.',model_promoted=False))
    jobs.update(status='COMPLETE');write(DEST/'jobs.json',jobs)


if __name__=='__main__':main()

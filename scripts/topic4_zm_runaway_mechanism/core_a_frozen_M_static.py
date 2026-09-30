"""Fast-subsystem stationary diagnostic at an actual long-event M field.

M is an explicitly frozen slow parameter here only. Main full-system
periodic calculations keep every M dynamic and have separate certificates.
"""
from common import OUT, np, read, write, log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy import sparse
from scipy.special import logit
import core_a_static_candidates as root
from datetime import datetime
import os

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
SOURCE=BASE/'actual_D0300/from_interictal_history'
DEST=BASE/'first_long_frozen_M_stationary'


class FrozenMDrift(PhysicalDelayConditionalDrift):
    def __init__(self, held_M):
        super().__init__()
        self.held_M=np.array(held_M,float,copy=True)
        assert self.held_M.shape==(self.P,) and np.isfinite(self.held_M).all()
        assert np.all(self.held_M[~self.E]==0) and self.held_M.min()>=0

    def moments(self,r,m=None):
        assert m is None
        return super().moments(r,self.held_M)

    def temporal_components(self,lam,dt=None):
        U,VE,VI=super().temporal_components(lam,dt)
        if dt is None:
            M=.5*self.E/(1+1000*lam)
        else:
            z=np.exp(-lam*dt);decay=np.exp(-dt/1000)
            M=.5*self.E*z*(1-decay)/(1-decay*z)
        return U+sparse.diags(M),VE,VI

    def jacobian(self,r):
        # Keep the inherited direct-rate interface consistent with the
        # explicitly frozen-M logit interface used by this diagnostic.
        gradient=self.local_operating(*self.moments(r))['gradient']
        return (sum(sparse.diags(gradient[:,j])@v for j,v in
                    enumerate(self.temporal_components(0.)))-sparse.eye(self.P)).tocsc()


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    assert read(SOURCE/'whole_record_audit.json')['status']=='AUDIT_PASS'
    base=dict(np.load(SOURCE/'checkpoint5000.npz'))
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the first long activity have a stationary fast-network counterpart at its actual5s spatial M, which could nominate a slow-M-controlled fast branch for the matched termination intervention?',
        source=str(SOURCE/'checkpoint5000.npz'),
        changed_only='Explicit fast-subsystem diagnostic: hold every M at its actual5s value. Same held original Z and full3479group physical network. No response fit, scalar M replacement or isolated-core model.',
        equation='r=Phi(mu(r)-m_5s,ve(r),vi(r)); local covariance/filter steady values are eliminated exactly. In the static and temporal gain remove the dynamic-M feedback .5E/(1+1000lambda), because M is prescribed. This is not the full-system stationary law M=.5Er.',
        seeds='Actual instantaneous5s group rates and preceding100ms group mean. Guesses only; original residual<1e-11/ms required.',
        verification='Independent centered logit-Jacobian directional differences at two step sizes, threshold2e-5, and direct original static moment comparison. Original-engine stationary replay and temporal stability required separately if root exists.',
        budget='Two existing70-step Armijo/LSMR solves with stagnation stops; no automatic branch continuation. Failure is not nonexistence.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'jobs.json',jobs)
    s=FrozenMDrift(base['syn'][4]);s.set_Z(base['syn'][5]);root.DEST=DEST
    initial=base['rate'];q=logit(np.clip(initial*s.ref,1e-10,1-1e-10))
    r,F,op,J=value_and_jacobian(s,q);rng=np.random.default_rng(925003);v=rng.normal(size=s.P);checks=[]
    for h in [1e-4,5e-5]:
        fd=(value_and_jacobian(s,q+h*v,False)[1]-value_and_jacobian(s,q-h*v,False)[1])/(2*h)
        checks.append(dict(step=h,relative_error=float(np.linalg.norm(J@v-fd)/np.linalg.norm(fd))))
    standard=PhysicalDelayConditionalDrift();standard.set_Z(s.Z)
    difference=max(float(abs(a-b).max()) for a,b in zip(s.moments(r),standard.moments(r,base['syn'][4])))
    passed=max(x['relative_error'] for x in checks)<2e-5 and difference==0
    write(DEST/'implementation_check.json',dict(status='PASS' if passed else 'FAIL',directional_Jacobian=checks,original_moment_max_difference=difference))
    assert passed,checks
    block=np.load(SOURCE/'block00.npz');mean=block['group_rate_hz'][-100:].astype(float).mean(0)/1000
    rows=[]
    try:
        for label,guess in [('instantaneous5s',initial),('preceding100ms',mean)]:
            row=root.solve(s,guess,label);rows.append(row);jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
            if row['status']=='ROOT_PASS':break
        np.savez_compressed(DEST/'held_slow_fields.npz',Z=s.Z,M=s.held_M)
        write(DEST/'result.json',dict(status='COMPLETE',rows=rows,
            scope='Frozen-M fast-subsystem root screen only; no full-system bifurcation classification.',model_promoted=False))
        jobs.update(status='COMPLETE');write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':main()

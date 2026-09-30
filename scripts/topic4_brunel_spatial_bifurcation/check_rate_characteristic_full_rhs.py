"""Check the characteristic operator against the nonlinear nine-state RHS.

Finite differences act on real/imaginary physical states and delayed arrivals.
This checks the analytic elimination and units, not native-SNN equivalence.
"""
from rate_field import *


def main():
    s=RateField();rows=[];rng=np.random.default_rng(187121)
    for core in 'AB':
        z=np.load(RATE_OUT/f'hopf_{core}.npz');r=z['rates'];J=float(z['J'])
        y=s.equilibrium_state(r,J);arr=np.array([m@r for m in s.matrices(J)])
        for kind,lam,v in [
            ('critical',1j*float(z['omega']),z['vector']),
            ('random_2Hz',.0001+2j*np.pi*2/1000,rng.normal(size=s.P)+1j*rng.normal(size=s.P)),
            ('random_80Hz',-.0001+2j*np.pi*80/1000,rng.normal(size=s.P)+1j*rng.normal(size=s.P))]:
            v=v/abs(v).max()*.001
            state=s.eigenstate(r,J,lam,v)
            arrivals=np.array([m@v for m in s.matrices(J,lam)])
            characteristic=s.characteristic(r,J,lam)@v
            expected=lam*state
            expected[0]-=characteristic/s.tf
            expected[1]-=characteristic/s.ts
            checks=[]
            for eps in [1e-3,3e-4,1e-4]:
                actual=np.zeros_like(state,dtype=complex)
                for part,coefficient in [(np.real,1),(np.imag,1j)]:
                    derivative=(s.rhs(y+eps*part(state),arr+eps*part(arrivals))-
                                s.rhs(y-eps*part(state),arr-eps*part(arrivals)))/(2*eps)
                    actual+=coefficient*derivative
                checks.append(dict(epsilon=eps,
                    full_state_relative_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
                    rate_state_relative_error=float(np.linalg.norm(actual[:2]-expected[:2])/np.linalg.norm(expected[:2])),
                    critical_mode_relative_error=float(np.linalg.norm(actual-lam*state)/np.linalg.norm(lam*state)) if kind=='critical' else None))
            passed=all(q['full_state_relative_error']<1e-6 and q['rate_state_relative_error']<1e-6 for q in checks[-2:])
            if kind=='critical':passed=passed and all(q['critical_mode_relative_error']<1e-6 for q in checks[-2:])
            rows.append(dict(core=core,J_EE_core=J,direction=kind,checks=checks,passed=passed))
    result=dict(status='PASS' if all(q['passed'] for q in rows) else 'CHECK_FAILED',rows=rows,
        equations_changed=False,
        scope='CPU centered differences of the nonlinear nine-state physical RHS at both first Hopf equilibria; critical eigenvectors and random complex directions at 2 and 80Hz. Checks analytic elimination, adaptation, filters and physical-delay factors. No exhaustive spectrum or SNN correspondence claim.')
    write(RATE_OUT/'model_audit_20260918/characteristic_full_rhs_check.json',result)
    print(result,flush=True)
    assert result['status']=='PASS'


if __name__=='__main__':main()

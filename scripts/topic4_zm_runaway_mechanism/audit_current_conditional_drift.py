"""Verify private-Q static/temporal operators, without a branch search."""
from common import OUT,model,np,read,write,log
from current_conditional_drift import CurrentConditionalDrift
from fine_rate_frozen_Z_fields import native_field
from shared_variance_network_sensitivity import split

DEST=OUT/'conditional_drift_analysis_interface'


def main():
    DEST.mkdir(exist_ok=True);s=CurrentConditionalDrift(40);physical=model(40)
    rng=np.random.default_rng(920094);v=rng.normal(size=s.P)+1j*rng.normal(size=s.P);operators=[]
    for lam in [0.,.002+.03j,-.0001+.1j]:
        mats=s.matrices(lam);full=physical.matrices(lam)
        for ch in [0,1]:
            difference=mats[ch]-full[ch]
            assert not difference.nnz or np.max(abs(difference.data))<1e-12
        expected_history=(np.exp(-lam*s.delays[:,None])*v).ravel()
        for k,kind in enumerate(['ampa','gaba']):
            direct=s.private_operators[kind]@expected_history
            projected=mats[k+2]@v
            err=float(np.linalg.norm(direct-projected)/max(np.linalg.norm(direct),1e-15));assert err<1e-12
            operators.append(dict(lambda_real=float(np.real(lam)),lambda_imag=float(np.imag(lam)),synapse=kind,relative_error=err))
    finer,_=split(physical,.025)
    for kind in ['ampa','gaba']:
        assert np.array_equal(finer[kind].indptr,s.private_operators[kind].indptr)
        assert np.array_equal(finer[kind].indices,s.private_operators[kind].indices)
        assert np.array_equal(finer[kind].data,s.private_operators[kind].data)
    rows=[]
    for tm,level in [(9000,.0002),(9420,.02),(9870,.2)]:
        s.set_Z(native_field(s,tm));r=level*rng.uniform(.8,1.2,s.P);J=s.jacobian(r)
        checks=[]
        for h in [1e-4,5e-5]:
            direction=r*rng.normal(size=s.P)
            fd=(s.residual(r+h*direction)-s.residual(r-h*direction))/(2*h)
            err=float(np.linalg.norm(J@direction-fd)/np.linalg.norm(fd));assert err<2e-5
            checks.append(dict(h=h,relative_error=err))
        dc=[]
        for dt in [None,.05,.025]:
            difference=s.characteristic(r,0.,dt)+J
            err=float(np.max(abs(difference.data))) if difference.nnz else 0.
            rel=err/max(float(np.max(abs(J.data))),1.)
            assert rel<1e-10;dc.append(dict(dt_ms=dt,relative_error=rel))
        rows.append(dict(native_Z_time_ms=tm,D=s.D,Jacobian=checks,temporal_DC=dc))
    write(DEST/'implementation_check.json',dict(status='PASS',groups=s.P,
        mean_operators_unchanged=True,private_operator_checks=operators,private_operator_bitwise_same_at_two_steps=True,
        response_and_parameters_unchanged=True,rows=rows,
        object='Conditional drift of current private-Q count model. NOT previous full-Q expected-rate model; separate results.',
        scope='Operator/derivative implementation only; no equilibrium solving or native/dynamical acceptance.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('PRIVATE DRIFT ANALYSIS INTERFACE PASS')


if __name__=='__main__':main()

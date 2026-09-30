"""Verify that changing coefficient-reading state alters nonlinear flow, not equilibrium linearization."""
from common import *
from closure_network_sensitivity import SensitivityModel
from response_voltage_units import VoltageScaledResponseTable


def main():
    source=OUT/'equilibria/native_unconstrained/t9870_tail_average.npz'
    eq=np.load(source);s=model();s.__class__=SensitivityModel
    s.resp.tables={p:VoltageScaledResponseTable(tab) for p,tab in s.resp.tables.items()}
    s.set_Z(eq['Z'],source=str(source));y=s.equilibrium_state(eq['r'])
    arrivals=np.array([a@eq['r'] for a in s.matrices()])
    def evaluate(y,arr,history,dynamic_z=False):
        s.history_weights=history
        return s.rhs(y,arr,dynamic_z=dynamic_z)
    fixed,rate0=evaluate(y,arrivals,False)
    other,rate1=evaluate(y,arrivals,True)
    equilibrium_res=float(np.max(abs(fixed)))
    assert equilibrium_res<1e-8,equilibrium_res
    equality=float(np.max(abs(fixed-other)))
    rateequality=float(np.max(abs(rate0-rate1)))
    assert equality<1e-11 and rateequality<1e-12,(equality,rateequality)
    rng=np.random.default_rng(919789);rows=[]
    scales=np.maximum(abs(y),1.);scales[11]=1.
    for trial in range(3):
        dy=rng.normal(size=y.shape)*scales
        da=rng.normal(size=arrivals.shape)*np.maximum(abs(arrivals),.01)
        for dynamic_z in [False,True]:
            for eps in [1e-5,5e-6]:
                derivatives=[];rate_derivatives=[]
                for history in [False,True]:
                    fp,rp=evaluate(y+eps*dy,arrivals+eps*da,history,dynamic_z)
                    fm,rm=evaluate(y-eps*dy,arrivals-eps*da,history,dynamic_z)
                    derivatives.append((fp-fm)/(2*eps))
                    rate_derivatives.append((rp-rm)/(2*eps))
                relative=float(np.linalg.norm(derivatives[0]-derivatives[1])/max(np.linalg.norm(derivatives[0]),1e-12))
                rate_relative=float(np.linalg.norm(rate_derivatives[0]-rate_derivatives[1])/max(np.linalg.norm(rate_derivatives[0]),1e-12))
                assert relative<1e-5,relative
                assert rate_relative<1e-5,rate_relative
                rows.append(dict(direction=trial,dynamic_Z_linearization=dynamic_z,step=eps,relative_difference=relative,rate_derivative_relative_difference=rate_relative))
    q=dict(status='CONDITIONAL_EQUILIBRIUM_AND_LOCAL_JACOBIAN_IDENTITY_PASS',source=str(source),
           units='Both models include the same voltage-unit correction',
           conditional_equilibrium_residual_per_ms=equilibrium_res,
           equilibrium_RHS_difference=equality,rate_difference_per_ms=rateequality,
           finite_difference_checks=rows,
           reason='At steady inputs mu=mus,ve=vEf=vEv,vi=vIf=vIv. Derivatives of coefficient lookup multiply zero filter differences, so changing instantaneous to filtered lookup leaves first-order response unchanged.',
           scope='The Z field is held at this equilibrium. Dynamic-Z derivatives are checked at the same state, not claimed as a full-ZM equilibrium. This is not periodic-orbit or onset-type equivalence.')
    write(OUT/'closure_network_sensitivity/linearization_identity.json',q);log('LINEARIZATION IDENTITY',q)


if __name__=='__main__':main()

"""Check the corrected conditional-drift analytic operators at fixed probes."""
from common import OUT,model,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from fine_rate_frozen_Z_fields import native_field

DEST=OUT/'physical_delay_conditional_drift_interface'


def main():
    assert read(OUT/'physical_delay_variance_split/result.json')['status']=='PHYSICAL_DELAY_UNIT_ERROR_CONFIRMED'
    DEST.mkdir(exist_ok=True);s=PhysicalDelayConditionalDrift();full=model(40)
    rng=np.random.default_rng(920096);v=rng.normal(size=s.P)+1j*rng.normal(size=s.P);operators=[]
    for lam in [0.,.002+.03j,-.0001+.1j]:
        mats=s.matrices(lam);base=full.matrices(lam)
        for k in [0,1]:
            difference=mats[k]-base[k];assert not difference.nnz or np.max(abs(difference.data))<1e-12
        history=(np.exp(-lam*s.delays[:,None])*v).ravel()
        for k,kind in enumerate(['ampa','gaba']):
            direct=s.private_operators[kind]@history;actual=mats[k+2]@v
            error=float(np.linalg.norm(direct-actual)/max(np.linalg.norm(direct),1e-15));assert error<1e-12
            operators.append(dict(lam=[float(np.real(lam)),float(np.imag(lam))],kind=kind,relative_error=error))
    rows=[]
    for tm,level in [(9000,.0002),(9420,.02),(9870,.2)]:
        s.set_Z(native_field(s,tm));r=level*rng.uniform(.8,1.2,s.P);J=s.jacobian(r);fd_checks=[];dc=[]
        for h in [1e-4,5e-5]:
            v=r*rng.normal(size=s.P);fd=(s.residual(r+h*v)-s.residual(r-h*v))/(2*h)
            error=float(np.linalg.norm(fd-J@v)/np.linalg.norm(fd));assert error<2e-5
            fd_checks.append(dict(h=h,relative_error=error))
        for dt in [None,.05,.025]:
            difference=s.characteristic(r,0.,dt)+J
            error=(float(np.max(abs(difference.data))) if difference.nnz else 0.)/max(float(np.max(abs(J.data))),1.)
            assert error<1e-10;dc.append(dict(dt_ms=dt,relative_error=error))
        rows.append(dict(native_Z_time_ms=tm,D=s.D,Jacobian=fd_checks,temporal_DC=dc))
    write(DEST/'implementation_check.json',dict(status='PASS',operators=operators,rows=rows,
        object='Corrected physical-delay private-Q conditional drift,3479groups,constantmeanexternalinput,Zfieldheld,Mdynamic.',
        mean_graph_response_and_slow_laws_unchanged=True,scope='Analytic operator checks only. Actual engine two-step check and originalnativecorrespondence remain required.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('PHYSICAL DELAY RATE ANALYSIS INTERFACE PASS')


if __name__=='__main__':main()

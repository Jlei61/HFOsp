"""Independent identities for the refractory population-rate candidate."""
from refractory_rate_response import *
from nonlinear_rate_response import physical_from_features
from scipy.special import expit
from pathlib import Path

def check():
    torch.set_num_threads(2);rows=[]
    for pop in 'EI':
        base=BaseLogit(pop);ref=base.ref
        physical=np.array([[15.,30.,50.],[25.,500.,300.],[100.,20.,10.]])
        for theta in [14.2,18.]:
            ell,grad=base.evaluate(physical,theta,True)
            old=base.maximum*expit(base.base.evaluate(physical,theta))
            new=base.maximum*expit(ell+np.log(ref/DT0))
            assert np.max(abs(new-old))<1e-8
            for j in range(3):
                eps=1e-5*max(1.,float(physical[:,j].max()));p=physical.copy();m=p.copy();p[:,j]+=eps;m[:,j]-=eps
                fd=(base.evaluate(p,theta)-base.evaluate(m,theta))/(2*eps)
                assert np.max(abs(fd-grad[:,j]))<1e-6
        for dt in [.1,.05,.025]:
            A,b,c,G,Q=covariance_matrices(pop,dt)
            for j in range(2):
                assert np.max(abs(expm(G[j]*dt)-A[j]))<1e-12
                assert np.max(abs(np.linalg.solve(G[j],(A[j]-np.eye(3))@Q[j])-b[j]))<1e-10
                assert abs(c[j]*np.linalg.solve(np.eye(3)-A[j],b[j])[2]-1)<1e-10
            for ell in [-10.,-3.,0.,3.]:
                target=1000/(ref+DT0*np.exp(-ell))
                rate,mass=implicit_flux(np.full(round(100/dt),ell),dt,ref,target)
                assert np.max(abs(rate-target))<1e-6 and mass>=0
                transient,mass=implicit_flux(np.full(round(100/dt),ell),dt,ref)
                assert mass>=-1e-10 and np.isfinite(transient).all()
        # Exact discrete linear response of the refractory integral, without a fitted network.
        for dt in [.1,.05]:
            T=100.;t=(np.arange(round(3000/dt))+1)*dt
            for f in [10.,80.]:
                ell=-3.;amp=1e-5;rr,_=implicit_flux(ell+amp*np.sin(2*np.pi*f*t/1000),dt,ref)
                keep=t>1000.;phase=2*np.pi*f*t[keep]/1000
                measured=2*np.mean(rr[keep]*(np.sin(phase)+1j*np.cos(phase)))/amp
                _,_,K=transfer_factors(pop,[f],dt)
                rho=np.exp(ell)/DT0;r=1000*rho/(1+ref*rho)
                predicted=r/(1+rho*K[0])
                assert abs(measured-predicted)/abs(predicted)<1e-6,(pop,dt,f,measured,predicted)
        # The same burn/state trajectory is returned by both observation interfaces.
        wave=np.tile(np.array([20.,40.,60.])[:,None],(1,32))
        short=features(wave,100.,.1,100,500,pop)
        full=features(wave,100.,.1,100,500,pop,include_burn=True)
        assert np.array_equal(full[100:],short)
        rows.append(dict(pop=pop,static_identity=True,gradient_finite_difference=True,
            covariance_exact_generator=True,refractory_mass=True,discrete_frequency_response=True,burn_identity=True))
    DEST.mkdir(exist_ok=True)
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,
        scope='Equation and discretization identities only; no fitted local or spatial accuracy claim.'))
    log('REFRACTORY RATE IMPLEMENTATION PASS')

if __name__=='__main__':check()

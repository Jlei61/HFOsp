"""Check temporal gain against an independently differentiated state update."""
from common import OUT, np, write, log
from current_rate_characteristic import CurrentRateCharacteristic, local_factors
from refractory_rate_response import covariance_matrices, DT0, TAUS
from nonlinear_rate_response import normalized_input, SCALE
from fine_rate_frozen_Z_fields import native_field
from scipy.special import expit
from scipy.linalg import solve
import torch

DEST=OUT/'current_rate_analysis_interface'


def local_state_test(s, index, x, dt):
    pop='E' if s.E[index] else 'I'; theta=s.theta[index]; ref=s.ref[index]
    A,B,C,_,_=covariance_matrices(pop,dt); nref=round(ref/dt)
    u=normalized_input(x,theta)/SCALE
    features=np.r_[u,np.zeros(36)]
    with torch.no_grad():
        ell=float(s.nets[pop].logits(torch.tensor(features),torch.tensor(s.bases[pop].evaluate(x,theta))).numpy())
    r=expit(ell+np.log(ref/DT0))/ref
    cov=np.array([np.linalg.solve(np.eye(3)-A[ch],B[ch]*x[ch+1]) for ch in range(2)])
    initial=np.r_[cov.ravel(),np.repeat(u,12),np.full(nref-1,r)]

    def step(states, inputs):
        states=np.atleast_2d(states);inputs=np.atleast_2d(inputs)
        after=states.copy(); f=np.zeros((len(states),39))
        for ch in range(2):after[:,3*ch:3*ch+3]=states[:,3*ch:3*ch+3]@A[ch].T+inputs[:,ch+1,None]*B[ch]
        physical=np.column_stack([inputs[:,0],C[0]*after[:,2],C[1]*after[:,5]])
        u=normalized_input(physical,theta)/SCALE; f[:,:3]=u
        for ch in range(3):
            for j,tau in enumerate(TAUS):
                b=dt/tau;e=np.exp(-b);k=6+12*ch+3*j
                h1,h2,h3=states[:,k:k+3].T
                after[:,k]=e*h1+(1-e)*u[:,ch]
                after[:,k+1]=e*(h2+b*h1)+(1-e*(1+b))*u[:,ch]
                after[:,k+2]=e*(h3+b*h2+.5*b*b*h1)+(1-e*(1+b+.5*b*b))*u[:,ch]
                f[:,3+12*ch+3*j:3+12*ch+3*j+3]=after[:,k:k+3]-u[:,ch,None]
        with torch.no_grad():
            ell=s.nets[pop].logits(torch.tensor(f),torch.tensor(s.bases[pop].evaluate(physical,theta))).numpy()
        rate=(1-dt*states[:,42:].sum(1))*expit(ell+np.log(dt/DT0))/dt
        after[:,42]=rate;after[:,43:]=states[:,42:-1]
        return after,rate

    constant, rr=step(initial,x)
    assert np.max(abs(constant[0]-initial))<1e-10 and abs(rr[0]-r)<1e-12
    # Numerical state matrix and forcing/output derivatives, with two steps.
    checks=[];n=len(initial)
    for epsilon in [2e-5,1e-5]:
        scale=epsilon*np.maximum(abs(initial),1e-3)
        perturb=np.diag(scale);xp=np.broadcast_to(x,(n,3))
        yp,rp=step(initial+perturb,xp);ym,rm=step(initial-perturb,xp)
        matrix=((yp-ym)/(2*scale[:,None])).T;out=(rp-rm)/(2*scale)
        h=epsilon*np.maximum(abs(x),1.)
        yp,rp=step(np.broadcast_to(initial,(3,n)),x+np.diag(h))
        ym,rm=step(np.broadcast_to(initial,(3,n)),x-np.diag(h))
        forcing=((yp-ym)/(2*h[:,None])).T;direct=(rp-rm)/(2*h)
        for frequency in [0.,1.,5.,15.,40.,100.]:
            lam=2j*np.pi*frequency/1000;z=np.exp(-lam*dt)
            gain=direct+z*out@solve(np.eye(n)-z*matrix,forcing)
            # Independent closed-form implementation on the full spatial object.
            full=np.broadcast_to(x,(s.P,3)).copy()
            op=s.local_operating(*full.T)
            analytic=s.local_gain(op,lam,dt)[index]
            error=float(np.linalg.norm(gain-analytic)/max(np.linalg.norm(analytic),1e-10))
            assert error<2e-4,(index,dt,epsilon,frequency,error,gain,analytic)
            checks.append(dict(epsilon=epsilon,frequency_hz=frequency,relative_error=error))
    return dict(index=int(index),population=pop,theta=float(theta),input=x.tolist(),dt_ms=dt,checks=checks)


def main():
    s=CurrentRateCharacteristic(40);rows=[]
    # Two threshold/operating conditions per E/I, not fitted or solved states.
    indices=[np.flatnonzero(s.E)[0],np.flatnonzero(s.E)[-1],np.flatnonzero(~s.E)[0],np.flatnonzero(~s.E)[-1]]
    for k,i in enumerate(indices):
        sc=s.theta[i]-11.;x=np.array([11.+(.3 if k%2==0 else 1.3)*sc,sc**2*(1+k*.1),sc**2*.6])
        for dt in [.05,.025]:rows.append(local_state_test(s,i,x,dt))
        log('CURRENT TEMPORAL LOCAL STATE MATRIX PASS',i)
    dc=[]
    for tm,level in [(9000,.0002),(9420,.02),(9870,.2)]:
        s.set_Z(native_field(s,tm));r=np.full(s.P,level);J=s.jacobian(r)
        for dt in [None,.05,.025]:
            C=s.characteristic(r,0.,dt);difference=C+J
            err=float(np.max(abs(difference.data))) if difference.nnz else 0.
            rel=err/max(float(np.max(abs(J.data))),1.)
            assert rel<1e-10,(tm,dt,err,rel)
            dc.append(dict(native_Z_time_ms=tm,dt_ms=dt,relative_DC_error=rel))
    write(DEST/'temporal_implementation_check.json',dict(status='PASS',local_rows=rows,DC_network_checks=dc,
        object='Current conditioned39 full-diffusion expected-rate with full Z held, M dynamic, constant external mean.',
        meaning='Local analytic frequency response matches independent finite differences of the entire covariance/history/refractory state update at two steps; spatial zero-frequency characteristic matches negative static residual Jacobian.',
        limits='No equilibrium root, complete time-domain spatial tangent test, contour root count or stability classification yet. This interface check cannot waive native/local correspondence failures.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))


if __name__=='__main__':main()

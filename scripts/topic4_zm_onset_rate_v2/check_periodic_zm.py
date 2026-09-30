"""Independent checks of fixed-J, spatial-Z periodic operator and derivatives."""
from periodic_zm import *


def main():
    s=ZMSpatialRate();o=ZMPeriodic(s,32);cp=o.cp;D=.18;T=268.5
    rng=np.random.default_rng(831);q=rng.normal(size=s.P)+1j*rng.normal(size=s.P)
    q*=.0001/abs(q).max();phase=np.exp(2j*np.pi*np.arange(32)/32)[:,None]
    r=2*np.real(phase*q);s.set_D(D);lam=2j*np.pi/T
    a,b,qa,qb=s.matrices(1.,lam)
    expected_m=np.array([
        s.tm*(s.area[0]*(a@q)/((1+lam*s.rise[0])*(1+lam*s.decay[0]))-
              s.Z*s.area[1]*(b@q)/((1+lam*s.rise[1])*(1+lam*s.decay[1])))-.5*s.E*q/(1+1000*lam),
        s.tm*s.area[0]**2*(qa@q)/(1+lam*s.tau[0]/2),
        s.tm*(s.Z*s.area[1])**2*(qb@q)/(1+lam*s.tau[1]/2)])
    expected=2*np.real(phase[None,:,:]*expected_m[:,None,:])
    actual=o.moments(cp.asarray(r),o.kernels(T,D)).get()
    errors=dict(harmonic_moment_relative_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
    r0=np.load(OLD/'g20/D_gap_lower_guarded/verified_outer_fold/fold31.npz')['r']
    gr=cp.asarray(np.broadcast_to(r0,(32,s.P)).copy())
    dc=o.moments(gr,o.kernels(T,D)).get()+o.private.get()[:,None,:]
    errors['dc_moment_max_error']=float(np.max(abs(dc[:,0]-np.array(s.moments(r0)))))
    r=cp.asarray(r0+r);reference=r.copy()
    dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
    phasecond=dr/cp.sum(dr*dr)*.001
    y=cp.r_[(r*1000).ravel(),np.log(T),D*1000]
    weight=cp.ones(len(y));tangent=cp.asarray(rng.normal(size=len(y)));tangent/=cp.linalg.norm(tangent)
    arc=(y.copy(),tangent,weight)
    F,A,_=o.evaluate(y,reference,phasecond,D,derivative=True,arc=arc)
    # Test rate, log-period and D columns separately to detect scale mistakes.
    for kind in ['rate','logT','D','mixed']:
        direction=cp.asarray(rng.normal(size=len(y)))
        if kind=='rate':direction[-2:]=0
        elif kind=='logT':direction[:]=0;direction[-2]=1
        elif kind=='D':direction[:]=0;direction[-1]=10
        else:direction[-2:]=cp.asarray([.3,20.])
        analytic=A@direction;eps=1e-5
        fp=o.evaluate(y+eps*direction,reference,phasecond,D,arc=arc)
        fm=o.evaluate(y-eps*direction,reference,phasecond,D,arc=arc)
        errors[kind+'_jacobian_relative_error']=float(cp.linalg.norm((fp-fm)/(2*eps)-analytic)/cp.linalg.norm(analytic))
    print(errors,flush=True)
    assert errors['harmonic_moment_relative_error']<1e-10
    assert errors['dc_moment_max_error']<1e-8
    assert max(v for k,v in errors.items() if 'jacobian' in k)<2e-6
    errors.update(status='PASS',J_EE_core=1.,M='dynamic with tau=1000 ms',Z='held spatial field on native 9.420 s power path')
    write(PERIODIC_OUT/'operator_checks.json',errors)


if __name__=='__main__':main()

"""Floquet Fourier moments against physical-delay matrices, and lambda column."""
from spectral_floquet_zm import *


def main():
    s=ZMSpatialRate();o=SpectralFloquet(s,PERIODIC_OUT/'orbits/burstUp_0043_N512.npz',32,0);cp=o.cp
    rng=np.random.default_rng(991);q=rng.normal(size=s.P)+1j*rng.normal(size=s.P)
    k=3;phase=np.exp(2j*np.pi*k*np.arange(o.N)/o.N)[:,None];u=cp.asarray(phase*q)
    lam=.037+.004j;l=lam+2j*np.pi*k/o.T;a,b,qa,qb=s.matrices(1.,l)
    expected=np.array([s.tm*(s.area[0]*(a@q)/((1+l*s.rise[0])*(1+l*s.decay[0]))-
        s.Z*s.area[1]*(b@q)/((1+l*s.rise[1])*(1+l*s.decay[1])))-.5*s.E*q/(1+l*1000),
        s.tm*s.area[0]**2*(qa@q)/(1+l*s.tau[0]/2),s.tm*(s.Z*s.area[1])**2*(qb@q)/(1+l*s.tau[1]/2)])[:,None,:]*phase[None,:,:]
    actual=o.moments(u,o.kernels(lam)).get();err=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    kernels=o.kernels(lam);H,Hp=kernels[-2:]
    derivative=-o.filt(cp.sum(o.gains*o.moments(u,kernels),axis=0),Hp)-o.filt(cp.sum(o.gains*o.moments(u,kernels,True),axis=0),H)
    def apply(ll):
        ker=o.kernels(ll);return u-o.filt(cp.sum(o.gains*o.moments(u,ker),axis=0),ker[-2])
    eps=1e-6;fd=(apply(lam+eps)-apply(lam-eps))/(2*eps)
    de=float(cp.linalg.norm(fd-derivative)/cp.linalg.norm(derivative))
    assert err<1e-10 and de<1e-5,(err,de)
    result=dict(status='PASS',complex_harmonic_moment_relative_error=float(err),lambda_derivative_relative_error=de)
    write(PERIODIC_OUT/'spectral_operator_checks.json',result);print(result)


if __name__=='__main__':main()

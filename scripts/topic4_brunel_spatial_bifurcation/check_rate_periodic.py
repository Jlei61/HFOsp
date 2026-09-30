"""Independent operator, Jacobian, and local normal-form/BVP checks."""
from rate_periodic import *
from rate_hopf_normal_form import moment_mode


def main():
    s=RateField();o=Periodic(s,32);cp=o.cp;z=np.load(PERIODIC_OUT/'normal_form_A.npz');J=float(z['J']);T=2*np.pi/float(z['w'])
    q=z['q'];phase=np.exp(2j*np.pi*np.arange(32)/32)[:,None];r=2*np.real(phase*q)
    expected=2*np.real(phase[None,:,:]*moment_mode(s,J,2j*np.pi/T,q)[:,None,:])
    actual=o.moments(cp.asarray(r),o.kernels(T,J)).get();moment_error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    r=z['r']+r*.1;df=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(cp.asarray(r),axis=0),n=o.N,axis=0)
    phasecondition=df/cp.sum(df*df)*.001;y=cp.r_[(cp.asarray(r)*1000).ravel(),np.log(T),J*1000]
    ref=cp.asarray(r);amplitude=(cp.asarray(q),.1);F,A,_=o.evaluate(y,ref,phasecondition,J,amplitude,True)
    direction=cp.asarray(np.random.default_rng(813).normal(size=len(y)));direction[-2:]=cp.asarray([.3,2.])
    analytic=A@direction;eps=1e-5
    fp=o.evaluate(y+eps*direction,ref,phasecondition,J,amplitude);fm=o.evaluate(y-eps*direction,ref,phasecondition,J,amplitude)
    jac_error=float(cp.linalg.norm((fp-fm)/(2*eps)-analytic)/cp.linalg.norm(analytic))
    nfrows=[]
    for core in 'AB':
        nf=read(PERIODIC_OUT/f'normal_form_{core}.json');sl=nf['J_shift_per_amplitude_squared']
        for amp in [.1,.2,.4]:
            path=PERIODIC_OUT/f'orbits/H{core}_a{amp:.5f}_N32.npz';orbit=np.load(path)
            estimate=(float(orbit['J'])-nf['J_EE_core'])/amp**2
            nfrows.append(dict(core=core,amplitude=amp,normal_form_J_slope=sl,BVP_J_slope=estimate,relative_difference=(estimate-sl)/sl))
    result=dict(moment_operator_relative_error=moment_error,BVP_Jacobian_relative_error=jac_error,normal_form_BVP_comparison=nfrows)
    assert moment_error<1e-10;assert jac_error<1e-6
    write(PERIODIC_OUT/'operator_checks.json',result);print(result,flush=True)


if __name__=='__main__':main()

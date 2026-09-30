"""Check the two-angle operator against the existing one-angle DDE BVP."""
from rate_invariant_torus import *


def main(device=1):
    s=RateField();q0=read(PERIODIC_OUT/'TR_A_B_N128.json');z=np.load(q0['orbit']);mode=np.load(PERIODIC_OUT/'TR_A_B_mode_N128.npz')
    nt,np_=64,8;T=float(z['T']);J=float(z['J']);lam=complex(mode['lam']);omega=2*np.pi/T
    shift=round(lam.imag/omega);nu=lam.imag-shift*omega
    q=resample(mode['u'],nt,axis=0)*np.exp(2j*np.pi*shift*np.arange(nt)/nt)[:,None];q/=abs(q).max()
    r=resample(z['r'],nt,axis=0);o=Torus(s,nt,np_,device);cp=o.cp;one=Periodic(s,nt,device);one.low_memory=True
    tor=cp.asarray(np.repeat(r[:,None,:],np_,axis=1));k=o.kernels(T,nu,J);kp=one.kernels(T,J)
    actual=o.moments(tor,k).reshape(3,nt,np_,s.P);expected=one.moments(cp.asarray(r),kp)[:,:,None,:]
    momerr=float(cp.max(cp.abs(actual-expected)))
    filt=o.filt(tor,k[-1]);expectedf=one.filt(cp.asarray(r),kp[-2])[:,None,:]
    filtererr=float(cp.max(cp.abs(filt-expectedf)))
    amp=.005/1000;tor+=2*amp*cp.real(cp.asarray(q)[:,None,:]*cp.exp(2j*cp.pi*cp.arange(np_)/np_)[None,:,None])
    ref=tor.copy();dr=cp.fft.ifft(1j*o.kt[:,None,None]*cp.fft.fft(ref,axis=0),axis=0).real
    phase=dr/cp.sum(dr*dr)*.001;qq=cp.asarray(q)
    y=cp.r_[(tor/.001).ravel(),cp.asarray([np.log(T),1.,J/.001])]
    F,A=o.evaluate(y,ref,phase,qq,amp,nu,True)
    d=cp.asarray(np.random.default_rng(4701).normal(size=len(y)));d[:-3]*=.01;d[-3:]*=.01
    h=1e-5;fd=(o.evaluate(y+h*d,ref,phase,qq,amp,nu)-o.evaluate(y-h*d,ref,phase,qq,amp,nu))/(2*h)
    pred=A@d;rateerr=float(cp.linalg.norm(fd[:-3]-pred[:-3])/cp.linalg.norm(fd[:-3]));bordererr=float(cp.max(cp.abs(fd[-3:]-pred[-3:])))
    # The actual torus critical Floquet mode, shifted into the small modulation
    # frequency, must annihilate the linearized two-angle physical residual.
    base=np.repeat(r[:,None,:],np_,axis=1)
    direction=2*np.real(q[:,None,:]*np.exp(2j*np.pi*np.arange(np_)/np_)[None,:,None])
    tiny=1e-7;yb=y.copy();yb[:-3]=cp.asarray(base.ravel()/.001)
    plus=yb.copy();minus=yb.copy();plus[:-3]+=cp.asarray(direction.ravel()*tiny/.001);minus[:-3]-=cp.asarray(direction.ravel()*tiny/.001)
    linear=(o.evaluate(plus,ref,phase,qq,amp,nu)-o.evaluate(minus,ref,phase,qq,amp,nu))[:-3]/(2*tiny/.001)
    modeerr=float(cp.linalg.norm(linear)/np.linalg.norm(direction))
    weight=cp.r_[cp.full(len(y)-3,10/np.sqrt(len(y)-3)),cp.asarray([50.,1.,1.])]
    tangent=d/cp.linalg.norm(d*weight);arc=(y.copy(),tangent,weight)
    fa,aa=o.evaluate(y,ref,phase,qq,amp,nu,True,arc=arc)
    fda=(o.evaluate(y+h*d,ref,phase,qq,amp,nu,arc=arc)-o.evaluate(y-h*d,ref,phase,qq,amp,nu,arc=arc))/(2*h)
    arate=float(cp.linalg.norm((aa@d-fda)[:-3])/cp.linalg.norm(fda[:-3]))
    aborder=float(cp.max(cp.abs((aa@d-fda)[-3:])))
    same=float(cp.max(cp.abs(F[:-1]-fa[:-1])))
    assert arate<2e-5 and aborder<1e-6 and same<1e-12
    write(PERIODIC_OUT/'torus_arclength_operator_checks.json',dict(status='PASS',
        full_Jacobian_rate_relative_error=arate,border_max_error=aborder,
        unchanged_physical_and_phase_residual_max_error=same,
        scope='Directional derivative includes all three border coordinates; pseudo-arclength changes only the family parameterization.'))
    result=dict(status='PASS' if momerr<1e-10 and filtererr<1e-10 and rateerr<2e-5 and bordererr<1e-6 and modeerr<2e-5 else 'FAIL',
        one_angle_moment_max_error=momerr,one_angle_filter_max_error=filtererr,
        full_Jacobian_rate_relative_error=rateerr,border_max_error=bordererr,critical_Floquet_mode_relative_residual=modeerr,
        modulation_frequency_per_ms=nu,tests='Same-DDE one-angle limit, full directional derivative, actual critical Floquet mode after frequency-gauge shift.')
    write(PERIODIC_OUT/'torus_operator_checks.json',result);print('TORUS OPERATOR CHECKS',result,flush=True)
    assert result['status']=='PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args();main(a.device)

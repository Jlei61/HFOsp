"""Identical LTI orbit reconstruction with the final inverse FFT on the CPU.

The result is needed on the host anyway. Avoiding a large prime-length batched
CUDA FFT workspace allows the same 2049-mode orbit to fit beside running jobs.
"""
from common import np
from floquet_v3 import resample,TAU_M


def orbit_states_and_derivative(o,sol,n,derivative_indices=None,include_rate=True):
    """Same Fourier orbit, padded directly to its evaluation grid.

    Analytic differentiation of the original harmonics avoids a second FFT of
    the much larger sampled state. CPU FFTs run on contiguous temporal rows.
    """
    from scipy.fft import irfft
    cp=o.cp;s=o.s;N=len(sol['r']);K=N//2+1;T=sol['T']
    assert n>N
    s.set_D(sol['D']);Z=s.Z
    ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam=o.kernels(T)
    rf=cp.fft.rfft(cp.asarray(sol['r']),axis=0);v=rf.ravel()
    a,b,qa,qb=[(x@v).reshape(K,s.P) for x in ops];tm=o.gp[0];E=o.gp[2]
    qa_h=tm*s.area[0]*a/(1+lam*s.rise[0]);ia_h=qa_h/(1+lam*s.decay[0])
    qg_h=tm*s.area[1]*b/(1+lam*s.rise[1]);ig_h=qg_h/(1+lam*s.decay[1])
    va_h=tm*s.area[0]**2*qa/(1+lam*s.tau[0]/2);vg_h=tm*s.area[1]**2*qb/(1+lam*s.tau[1]/2)
    m_h=.5*E*hm*rf;muh=ia_h-cp.asarray(Z)*ig_h-m_h
    vEh=va_h;vIh=cp.asarray(Z)**2*vg_h
    # Materialize one state component at a time. The 14-component list used
    # several additional GB on fine orbits, despite all FFT results living on
    # the host. This retains the same arithmetic and Fourier coefficients.
    harmonics=[lambda:muh/(1+lam*cp.asarray(s.poles[0])[None,:]),
               lambda:muh*fs,lambda:vEh*fE,lambda:vIh*fI,
               lambda:qa_h,lambda:ia_h,lambda:qg_h,lambda:ig_h,
               lambda:va_h,lambda:vg_h,lambda:m_h,lambda:None,
               lambda:vEh*fvE,lambda:vIh*fvI]
    full=np.empty((n+1,14,s.P))
    derivative=np.empty_like(full) if derivative_indices is None else np.empty((len(derivative_indices),14,s.P))
    frequency=2j*np.pi*np.arange(K)/T
    for k,make_harmonic in enumerate(harmonics):
        h=make_harmonic()
        if h is None:
            full[:,k]=Z;derivative[:,k]=0.;continue
        hh=np.ascontiguousarray(h.get().T)
        del h
        cp.get_default_memory_pool().free_all_blocks()
        if N%2==0:
            # The source Nyquist mode represents a real cosine once resampled.
            hh[:,-1]=hh[:,-1].real*.5
        full[:-1,k]=irfft(hh,n=n,axis=-1,workers=2).T*(n/N)
        dd=irfft(hh*frequency,n=n,axis=-1,workers=2)
        if derivative_indices is None:derivative[:-1,k]=dd.T*(n/N)
        else:derivative[:,k]=dd[:,np.asarray(derivative_indices)%n].T*(n/N)
        del dd
    for k in [0,1]:full[:-1,k]+=s.private_mu
    for k in [2,12]:full[:-1,k]+=s.private_ve
    full[-1]=full[0]
    if derivative_indices is None:derivative[-1]=derivative[0]
    rr=None
    if include_rate:
        rr=resample(sol['r'],n,axis=0);rr=np.concatenate([rr,rr[:1]],axis=0)
    return full,derivative,rr


def orbit_states(o,sol,n):
    cp=o.cp;s=o.s;N=len(sol['r']);K=N//2+1;T=sol['T'];s.set_D(sol['D']);Z=s.Z
    r=cp.asarray(sol['r']);ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam=o.kernels(T)
    rf=cp.fft.rfft(r,axis=0);v=rf.ravel()
    a,b,qa,qb=[(x@v).reshape(K,s.P) for x in ops];tm=o.gp[0];E=o.gp[2];tf=cp.asarray(s.poles[0])
    qa_h=tm*s.area[0]*a/(1+lam*s.rise[0]);ia_h=qa_h/(1+lam*s.decay[0])
    qg_h=tm*s.area[1]*b/(1+lam*s.rise[1]);ig_h=qg_h/(1+lam*s.decay[1])
    va_h=tm*s.area[0]**2*qa/(1+lam*s.tau[0]/2);vg_h=tm*s.area[1]**2*qb/(1+lam*s.tau[1]/2)
    m_h=.5*E*rf/(1+lam*TAU_M)
    muh=tm*(s.area[0]*ha*a-cp.asarray(Z)*s.area[1]*hg*b)-.5*E*hm*rf
    vEh=tm*s.area[0]**2*hva*qa;vIh=cp.asarray(Z)**2*tm*s.area[1]**2*hvg*qb
    harmonics=[muh/(1+lam*tf[None,:]),muh*fs,vEh*fE,vIh*fI,
               qa_h,ia_h,qg_h,ig_h,va_h,vg_h,m_h,vEh*fvE,vIh*fvI]
    st=np.array([np.fft.irfft(x.get(),n=N,axis=0) for x in harmonics])
    st[0]+=s.private_mu;st[1]+=s.private_mu;st[2]+=s.private_ve;st[11]+=s.private_ve
    Y=np.concatenate([st[:11],np.broadcast_to(Z,(1,N,s.P)),st[11:]],axis=0).transpose(1,0,2)
    full=resample(Y,n,axis=0);full=np.concatenate([full,full[:1]],axis=0);full[:,11]=Z
    rr=resample(sol['r'],n,axis=0);rr=np.concatenate([rr,rr[:1]],axis=0)
    return full,rr


if __name__=='__main__':
    from native_path import *
    from streaming_periodic import StreamPeriodic
    from floquet_v3 import orbit_states as original
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args()
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    o=StreamPeriodic(s,512,a.device);o.cache_mean_operators=False
    old,ro=original(o,sol,1024);new,rn=orbit_states(o,sol,1024)
    rel=float(np.linalg.norm(old-new)/np.linalg.norm(old));err=float(abs(old-new).max())
    assert rel<1e-12 and np.array_equal(ro,rn)
    write(OUT/'orbit_reconstruction_parity.json',dict(status='PASS',relative_error=rel,max_error=err,
          rate_history_bitwise=True,change='Only inverse FFT device; same LTI operators and resampling'))
    log('ORBIT RECONSTRUCTION PARITY',rel,err)

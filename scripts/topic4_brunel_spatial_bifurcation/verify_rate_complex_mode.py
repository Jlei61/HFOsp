"""Recover and independently propagate a complex full-state Floquet mode."""
from rate_floquet import *
from rate_floquet_spectral import SpectralFloquet


def recover_complex(f,u,lam,D,dt):
    cp=f.cp;s=f.s;N=f.N;k=f.kernels(lam);ops,_,fil,_,H,_=k
    uf=cp.fft.fft(cp.asarray(u),axis=0)/N
    zz=lam+2j*np.pi*cp.fft.fftfreq(N)*N/f.T;z=zz[:,None]
    a,b,qa,qb=[(o@uf.ravel()).reshape(N,s.P) for o in ops]
    tm,ref,th,alpha,tf,ts,E=f.o.gpars;drive=uf/H
    xa=drive/(1+z*tf);xb=drive/(1+z*ts)
    qav=tm*s.area[0]*a/(1+z*s.rise[0]);iav=qav/(1+z*s.decay[0])
    qgv=tm*s.area[1]*b/(1+z*s.rise[1]);igv=qgv/(1+z*s.decay[1])
    va=tm*s.area[0]**2*qa/(1+z*s.tau[0]/2);vg=tm*s.area[1]**2*qb/(1+z*s.tau[1]/2)
    m=.5*E*uf/(1+z*1000)
    local=cp.stack([v.sum(0) for v in [xa,xb,qav,iav,qgv,igv,va,vg,m]])
    history=cp.exp(-cp.arange(1,D+1)[:,None]*dt*zz[None,:])@uf
    check=float(cp.max(cp.abs(alpha*local[0]+(1-alpha)*local[1]-cp.asarray(u)[0])))
    assert check<1e-9
    return cp.r_[local.ravel(),history.ravel()].get(),check


def main(a):
    s=RateField();critical=read(PERIODIC_OUT/f'{a.label}_N{a.N}.json')
    z=np.load(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz');u=z['u'];lam=complex(z['lam'])
    f=SpectralFloquet(s,critical['orbit'],a.N,a.device)
    n=int(np.ceil(float(z['T'])/a.dt));dt=float(z['T'])/n;D=int(np.ceil(s.delays[-1]/dt))+1
    v,error=recover_complex(f,u,lam,D,dt)
    del f
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    m=Monodromy(s,critical['orbit'],a.dt,a.device);assert len(v)==m.dim
    mu=np.exp(lam*m.T);mv=m.matvec(v.real)+1j*m.matvec(v.imag)
    residual=float(np.linalg.norm(mv-mu*v)/np.linalg.norm(v))
    rayleigh=complex(np.vdot(v,mv)/np.vdot(v,v))
    out=dict(label=a.label,N=a.N,dt_ms=m.dt,J_EE_core=m.J,T_ms=m.T,
        complex_multiplier=mu,independent_rayleigh_multiplier=rayleigh,
        full_state_mode_relative_defect=residual,output_reconstruction_error=error,
        method='Complex Floquet harmonic reconstruction of all nine local states and entire physical delay history, independently advanced by variational Heun.',
        scope='Validation of this eigenpair only; not all other multipliers or torus nonlinear stability.')
    write(PERIODIC_OUT/f'{a.label}_monodromy_check_N{a.N}_dt{a.dt:g}.json',out)
    print('COMPLEX MODE CHECK',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--N',type=int,default=128)
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--device',type=int,default=1)
    main(p.parse_args())

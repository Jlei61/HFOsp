"""Independent full-state propagation of the known antiperiodic null mode.

Recover all nine local states and the complete delay history from Fourier
coefficients, then check M v = -v and M phase = phase without an Arnoldi
search through the many passive adaptation modes. No temporal/spatial modes
are removed. The monodromy time step is independently refined.
"""
from rate_floquet import *
import gc


def recover(o,kernels,cf,D,dt):
    cp=o.cp;s=o.s;K=o.K;lam=2j*np.pi*cp.arange(K)[:,None]/o.recovery_period
    tm,ref,th,alpha,tf,ts,E=o.gpars;ops=kernels[0];H=kernels[-2]
    a,b,qa,qb=[(op@cf.ravel()).reshape(K,s.P) for op in ops]
    target=cf/H;xa=target/(1+lam*tf);xb=target/(1+lam*ts)
    qav=tm*s.area[0]*a/(1+lam*s.rise[0]);iav=qav/(1+lam*s.decay[0])
    qgv=tm*s.area[1]*b/(1+lam*s.rise[1]);igv=qgv/(1+lam*s.decay[1])
    va=tm*s.area[0]**2*qa/(1+lam*s.tau[0]/2);vg=tm*s.area[1]**2*qb/(1+lam*s.tau[1]/2)
    m=.5*E*cf/(1+lam*1000);factor=cp.full((K,1),2.);factor[0]=1;factor[-1]=1
    local=cp.stack([cp.sum(v*factor,axis=0).real for v in [xa,xb,qav,iav,qgv,igv,va,vg,m]])
    phase=cp.exp(-cp.arange(1,D+1)[:,None]*dt*lam[:,0][None,:])
    history=(phase@(cf*factor)).real
    return cp.r_[local.ravel(),history.ravel()].get()


def main():
    p=argparse.ArgumentParser();p.add_argument('critical_json');p.add_argument('mode')
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--device',type=int,default=1)
    p.add_argument('--label',default='PD',help='Output prefix; keep distinct PD critical points separate')
    p.add_argument('--stream-harmonics',action='store_true')
    p.add_argument('--bounded-recovery',action='store_true',help='Reconstruct the identical full state in bounded frequency blocks')
    a=p.parse_args();s=RateField();q=read(Path(a.critical_json));z=np.load(q['orbit']);u=np.load(a.mode)['u']
    m=Monodromy(s,q['orbit'],a.dt,a.device,stream_harmonics=a.stream_harmonics);cp=m.cp;cp.get_default_memory_pool().free_all_blocks()
    N=len(u);bounded=a.bounded_recovery or N>=4096
    if bounded:
        from scipy import fft
        from rate_periodic_state import recover_bounded
        cf=fft.rfft(np.r_[u,-u],axis=0,workers=4)/(2*N)
        v=recover_bounded(s,cf,2*m.T,m.J,m.D,m.dt,a.device)
        r=resample(z['r'],N,axis=0)
        cf=fft.rfft(np.r_[r,r],axis=0,workers=4)/(2*N)
        cf*=2j*np.pi*np.arange(len(cf))[:,None]/(2*m.T)
        phase=recover_bounded(s,cf,2*m.T,m.J,m.D,m.dt,a.device)
        del cf
    else:
        o=Periodic(s,2*N,a.device);o.low_memory=True;o.recovery_period=2*m.T
        o.stream_harmonics=a.stream_harmonics
        o.harmonic_chunk_size=64;o.derivative_chunk_size=64;k=o.kernels(2*m.T,m.J)
        anti=cp.asarray(np.r_[u,-u]);cf=cp.fft.rfft(anti,axis=0)/(2*N)
        v=recover(o,k,cf,m.D,m.dt)
        r=resample(z['r'],N,axis=0);cf=cp.fft.rfft(cp.asarray(np.r_[r,r]),axis=0)/(2*N)
        cf*=2j*np.pi*cp.arange(o.K)[:,None]/(2*m.T);phase=recover(o,k,cf,m.D,m.dt)
        o.cache=None;del o,k,cf,anti
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
    mv=m.matvec(v);mp=m.matvec(phase);mu=float(v@mv/(v@v))
    row=dict(J_EE_core=m.J,T_ms=m.T,orbit=q['orbit'],mode=a.mode,dt_ms=m.dt,
             recovered_multiplier=mu,minus_one_relative_defect=float(np.linalg.norm(mv+v)/np.linalg.norm(v)),
             rayleigh_eigen_residual=float(np.linalg.norm(mv-mu*v)/np.linalg.norm(v)),
             phase_relative_defect=float(np.linalg.norm(mp-phase)/np.linalg.norm(phase)),
             state_reconstruction='exact harmonic blocks' if bounded else 'full harmonic bank',
             meaning='Full nine-state plus delay-history variational propagation of the independently computed antiperiodic null mode; this is not a complete spectrum')
    write(PERIODIC_OUT/f'{a.label}_monodromy_check_N{N}_dt{a.dt:g}.json',row)
    save_periodic_array(PERIODIC_OUT/f'{a.label}_monodromy_seed_N{N}_dt{a.dt:g}.npz',local=v[:9*s.P],history=v[9*s.P:].reshape(m.D,s.P),dt=m.dt)
    print('PD DIRECT MONODROMY',row,flush=True)


if __name__=='__main__':main()

"""Full-DDE initial state displaced along a verified unstable cycle mode."""
from verify_rate_complex_mode import *


def main(a):
    s=RateField();z=np.load(a.orbit);J=a.J;o=Periodic(s,128,a.device)
    r,T,J,e,h=o.solve(resample(z['r'],128,axis=0),float(z['T']),J,tol=1e-11);assert e<1e-11
    path=save_orbit(s,r,T,J,e,h,f'weak_unstable_J{J:.7f}_N128')
    mode=np.load(a.mode);sf=SpectralFloquet(s,path,128,a.device,4,5e-5)
    lam,u,hist=sf.refine(complex(mode['lam']),resample(mode['u'],128,axis=0),tol=2e-11)
    assert hist[-1]<2e-11 and lam.real>0
    D=s.prep['max_delay_steps']*round(.1/a.dt)
    from rate_periodic_state import recover
    o.recovery_period=T;k=o.kernels(T,J)
    base=recover(o,k,o.cp.fft.rfft(o.cp.asarray(r),axis=0)/len(r),D,a.dt)
    direction,error=recover_complex(sf,u,lam,D,a.dt)
    factor=a.amplitude/1000/abs(u).max();v=base+factor*direction.real
    y=v[:9*s.P].reshape(9,s.P);history=np.empty((D+1,s.P));history[0]=s.output(y)
    history[(-np.arange(1,D+1))%(D+1)]=v[9*s.P:].reshape(D,s.P)
    assert y[:2].min()>0 and history.min()>0
    dest=PERIODIC_OUT/'unstable_cycle_departure'/f'J{J:.7f}_dt{a.dt:g}_a{a.amplitude:g}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'initial.npz',final_state=y,final_history=history,dt_ms=a.dt,final_history_tick_modulo_depth=0)
    np.savez_compressed(dest/'mode.npz',u=u,lam=lam,J=J,T=T,full_mode=direction)
    row=dict(J_EE_core=J,T_ms=T,dt_ms=a.dt,orbit=str(path),spectral_mode_seed=str(a.mode),
        exponent_per_ms=lam,predicted_linear_efolding_ms=1/lam.real,mode_residual=hist[-1],
        rate_mode_amplitude_hz=a.amplitude,amplitude_definition='Coefficient after max absolute complex rate eigenfunction normalization; initial phase takes the real component.',
        initial_checkpoint=str(dest/'initial.npz'),
        purpose='Follow the unstable periodic orbit into its nonlinear outcome at fixed J. An initial-condition experiment, not a proof of global invariant-manifold connection.',
        no_noise=True,no_native_spikes=True)
    write(dest/'contract.json',row);print('UNSTABLE CYCLE SEED',row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('mode');p.add_argument('--J',type=float,default=.946)
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--amplitude',type=float,default=.01)
    p.add_argument('--device',type=int,default=1);main(p.parse_args())

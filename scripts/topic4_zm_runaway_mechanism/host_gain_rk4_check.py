"""Compare host gain scheduling and partial phase storage to original operator."""
from native_path import *
from streaming_periodic import StreamPeriodic
from orbit_reconstruction import orbit_states_and_derivative
from spectral_grid_sampler import SpectralGridSampler
from rk4_monodromy import RK4Monodromy
import argparse,gc


def main(a):
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    o=StreamPeriodic(s,len(z['r']),a.device);o.cache_mean_operators=False
    n=int(np.ceil(sol['T']/.05));dt=sol['T']/n
    ids=np.unique(np.r_[0,(-2*np.arange(1,int(np.ceil(s.delays[-1]/dt))+3))%(2*n)])
    full,dd,_=orbit_states_and_derivative(o,sol,2*n,include_rate=False)
    partial,dpart,_=orbit_states_and_derivative(o,sol,2*n,derivative_indices=ids,include_rate=False)
    state_error=float(np.max(abs(full-partial)));derivative_error=float(np.max(abs(dd[ids]-dpart)))
    derivative_relative=float(np.linalg.norm(dd[ids]-dpart)/np.linalg.norm(dd[ids]))
    log('PARTIAL STORAGE',state_error,derivative_error,derivative_relative)
    assert state_error<1e-10 and derivative_relative<1e-12
    sample=SpectralGridSampler(partial,sol['T'],derivative=dpart,derivative_indices=ids)
    assert np.array_equal(sample(ids*sol['T']/(2*n),1),dpart)
    del partial,dd,dpart;gc.collect()
    o.sample_state=sample
    old=RK4Monodromy(s,o,sol,dtmax=.05,device=a.device)
    x=np.random.default_rng(198).normal(size=old.dim);x[11*s.P:12*s.P]=0.
    old_gains=old.gains.get();start=time.time();expected=old.matvec(x);old_seconds=time.time()-start
    del old;gc.collect();o.cp.get_default_memory_pool().free_all_blocks()
    new=RK4Monodromy(s,o,sol,dtmax=.05,device=a.device,host_gain_cache=True)
    gain_error=float(np.max(abs(new.host_gains-old_gains)))
    start=time.time();actual=new.matvec(x);new_seconds=time.time()-start
    rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert gain_error==0 and rel<1e-12,(gain_error,rel)
    result=dict(status='PASS',gain_max_error=gain_error,map_relative_error=rel,
                map_bitwise_equal=bool(np.array_equal(actual,expected)),
                state_error=state_error,partial_derivative_error=derivative_error,
                partial_derivative_relative_error=derivative_relative,
                seconds=[old_seconds,new_seconds],model_change=False,
                scope='Only memory scheduling changes; identical float64 coefficients and RK4 stages')
    write(OUT/'host_gain_rk4_check.json',result);log('HOST GAIN RK4 CHECK',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())

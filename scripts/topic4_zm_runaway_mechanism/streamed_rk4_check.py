"""Bounded-memory RK4 has exactly the same stages and coefficients."""
from native_path import *
from streaming_periodic import StreamPeriodic
from orbit_reconstruction import orbit_states
from spectral_grid_sampler import SpectralGridSampler
import rk4_monodromy
import argparse,gc


def main(a):
    s=model();attach_native_path(s)
    z=np.load(OUT/'periodic/seed_N512.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    o=StreamPeriodic(s,len(z['r']),a.device);o.cache_mean_operators=False
    rk4_monodromy.orbit_states=orbit_states
    m=rk4_monodromy.RK4Monodromy(s,o,sol,dtmax=.05,device=a.device)
    x=np.random.default_rng(196).normal(size=m.dim);x[11*s.P:12*s.P]=0
    expected=m.matvec(x);gains=m.gains.get();n=m.n
    del m;gc.collect();o.cp.get_default_memory_pool().free_all_blocks()
    full,_=orbit_states(o,sol,2*n)
    o.sample_state=SpectralGridSampler(full,sol['T'])
    m=rk4_monodromy.RK4Monodromy(s,o,sol,dtmax=.05,device=a.device)
    actual=m.matvec(x)
    rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    gain=float(abs(m.gains.get()-gains).max())
    assert rel<1e-11 and gain<1e-10,(rel,gain)
    q=dict(status='PASS',map_relative_error=rel,gain_max_difference=gain,
           change='Stage coefficient scheduling only; same RK4, cubic delays and original model')
    write(OUT/'streamed_rk4_monodromy_check.json',q);log('STREAMED RK4 PARITY',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())

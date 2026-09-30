"""Whole-period zero-parameter parity with the existing homogeneous map."""
from common import *
from native_path import attach_native_path
from streaming_periodic import StreamPeriodic
from orbit_reconstruction import orbit_states_and_derivative
from spectral_grid_sampler import SpectralGridSampler
from preinterpolated_rk4 import PreinterpolatedRK4
from parameter_forced_monodromy import ParameterForcedRK4
from scipy.fft import next_fast_len
import argparse,gc


def main(device):
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    o=StreamPeriodic(s,len(sol['r']),device);o.cache_mean_operators=False
    n=next_fast_len(int(np.ceil(sol['T']/.05)),real=True);dt=np.nextafter(sol['T']/n,np.inf)
    full,deriv,_=orbit_states_and_derivative(o,sol,2*n,derivative_indices=np.array([0]),include_rate=False)
    o.sample_state=SpectralGridSampler(full,sol['T'],derivative=deriv,derivative_indices=np.array([0]))
    m=PreinterpolatedRK4(s,o,sol,dtmax=dt,device=device,host_gain_cache=True)
    x=np.random.default_rng(919803).normal(size=m.dim);x[11*s.P:12*s.P]=0.
    expected=m.matvec(x);del m;gc.collect();o.cp.get_default_memory_pool().free_all_blocks()
    f=ParameterForcedRK4(s,o,sol,dtmax=dt,device=device,host_gain_cache=True)
    for lo in range(0,2*n+1,512):
        hi=min(lo+512,2*n+1)
        sampled=o.sample_state(np.arange(lo,hi)*f.dt/2)
        expected_factors=np.stack([sampled[:,7],2*sampled[:,11]*sampled[:,9]],axis=1)
        assert np.array_equal(f.host_factors[lo:hi],expected_factors)
    observed=f.matvec(x);relative=float(np.linalg.norm(observed-expected)/np.linalg.norm(expected))
    maximum=float(np.max(abs(observed-expected)));assert relative<1e-12,(relative,maximum)
    q=dict(status='ZERO_PARAMETER_WHOLE_MAP_PARITY_PASS',relative=relative,maximum=maximum,
           exact_grid_factor_cache_bitwise=True,
           dt_ms=f.dt,steps=f.n,scope='Same orbit and timestep; adding a retained constant parameter direction leaves the zero-direction homogeneous map unchanged.')
    write(OUT/'parameter_forced_map_check.json',q);log('PARAMETER MAP PARITY',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)

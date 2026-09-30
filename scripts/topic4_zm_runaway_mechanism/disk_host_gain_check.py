"""Verify complete monodromy parity for RAM versus disk-backed gain arrays."""
from native_path import *
from streaming_periodic import StreamPeriodic
from orbit_reconstruction import orbit_states_and_derivative
from spectral_grid_sampler import SpectralGridSampler
from preinterpolated_rk4 import PreinterpolatedRK4
from host_array_storage import sanity
import argparse, gc


def main(a):
    storage_check = sanity()
    s = model(); attach_native_path(s)
    z = np.load(OUT/'periodic/seed_N512.npz')
    sol = dict(r=z['r'], T=float(z['T']), D=float(z['D']))
    o = StreamPeriodic(s, len(z['r']), a.device); o.cache_mean_operators = False
    n = int(np.ceil(sol['T']/.05))
    state, derivative, _ = orbit_states_and_derivative(o, sol, 2*n,
        derivative_indices=np.array([0]), include_rate=False)
    o.sample_state = SpectralGridSampler(state, sol['T'], derivative,
                                         derivative_indices=np.array([0]))
    old = PreinterpolatedRK4(s, o, sol, dtmax=.05, device=a.device,
                            host_gain_cache=True, host_gain_storage='memory')
    x = np.random.default_rng(91903).normal(size=old.dim)
    x[11*s.P:12*s.P] = 0.
    expected = old.matvec(x)
    gain = old.host_gains.copy()
    del old; gc.collect(); o.cp.get_default_memory_pool().free_all_blocks()
    new = PreinterpolatedRK4(s, o, sol, dtmax=.05, device=a.device,
                            host_gain_cache=True, host_gain_storage='disk')
    assert isinstance(new.host_gains, np.memmap)
    assert np.array_equal(gain, new.host_gains)
    actual = new.matvec(x)
    error = float(np.max(abs(actual-expected)))
    assert np.array_equal(actual, expected), error
    result = dict(status='PASS', allocation_check=storage_check,
        gain_blocks_bitwise_equal=True, complete_map_bitwise_equal=True,
        complete_map_max_error=error, storage=new.host_gain_storage,
        scope='Only temporary cache backing changes. Float64 gains, delayed interpolation, RK4, frozen equations, timestep, and acceptance gates are identical.')
    write(OUT/'disk_host_gain_check.json', result)
    log('DISK HOST GAIN CHECK', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', type=int, default=0)
    main(parser.parse_args())

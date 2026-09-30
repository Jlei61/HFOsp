"""Same Fourier reconstruction with bounded work arrays and disk-backed states.

Population blocks contain independent FFT rows. Component-major storage avoids
keeping the complete half-step trajectory as an anonymous host allocation.
No interpolation, retained mode, parameter or physical equation is changed.
"""
from common import np
from floquet_v3 import TAU_M
from host_array_storage import allocate_host_array


def orbit_states_and_derivative_bounded(o, sol, n, derivative_indices=None,
                                        include_rate=False, population_block=16):
    from scipy.fft import irfft
    from common import log
    assert not include_rate, 'This bounded diagnostic reconstructs states and their derivatives only'
    assert derivative_indices is not None
    cp = o.cp
    s = o.s
    N = len(sol['r'])
    K = N // 2 + 1
    T = sol['T']
    assert n > N
    s.set_D(sol['D'])
    Z = s.Z
    ops, (_, _, _, _, hm), (fs, fE, fI, fvE, fvI), lam = o.kernels(T)
    rf = cp.fft.rfft(cp.asarray(sol['r']), axis=0)
    a, b, qa, qb = [(x @ rf.ravel()).reshape(K, s.P) for x in ops]
    tm = o.gp[0]
    E = o.gp[2]
    qa_h = tm * s.area[0] * a / (1 + lam * s.rise[0])
    ia_h = qa_h / (1 + lam * s.decay[0])
    qg_h = tm * s.area[1] * b / (1 + lam * s.rise[1])
    ig_h = qg_h / (1 + lam * s.decay[1])
    va_h = tm * s.area[0] ** 2 * qa / (1 + lam * s.tau[0] / 2)
    vg_h = tm * s.area[1] ** 2 * qb / (1 + lam * s.tau[1] / 2)
    m_h = .5 * E * hm * rf
    mu_h = ia_h - cp.asarray(Z) * ig_h - m_h
    vi_h = cp.asarray(Z) ** 2 * vg_h
    harmonics = [
        lambda: mu_h / (1 + lam * cp.asarray(s.poles[0])[None, :]),
        lambda: mu_h * fs,
        lambda: va_h * fE,
        lambda: vi_h * fI,
        lambda: qa_h, lambda: ia_h, lambda: qg_h, lambda: ig_h,
        lambda: va_h, lambda: vg_h, lambda: m_h, lambda: None,
        lambda: va_h * fvE, lambda: vi_h * fvI,
    ]
    backing, storage = allocate_host_array((14, n + 1, s.P), 'disk')
    full = backing.transpose(1, 0, 2)
    ids = np.asarray(derivative_indices) % n
    derivative = np.empty((len(ids), 14, s.P))
    frequency = 2j * np.pi * np.arange(K) / T
    log('BOUNDED ORBIT STORAGE', storage, 'population_block', population_block)
    for k, make_harmonic in enumerate(harmonics):
        h = make_harmonic()
        if h is None:
            # A bounded temporal block prevents a full-size broadcast copy.
            for lo in range(0, n + 1, 4096):
                full[lo:lo + 4096, k] = Z
            derivative[:, k] = 0.
        else:
            for lo in range(0, s.P, population_block):
                hi = min(lo + population_block, s.P)
                hh = np.ascontiguousarray(h[:, lo:hi].get().T)
                if N % 2 == 0:
                    hh[:, -1] = hh[:, -1].real * .5
                sampled = irfft(hh, n=n, axis=-1, workers=2)
                # Preserve the original per-value scaling before assignment.
                sampled *= n / N
                full[:-1, k, lo:hi] = sampled.T
                del sampled
                dd = irfft(hh * frequency, n=n, axis=-1, workers=2)
                derivative[:, k, lo:hi] = dd[:, ids].T * (n / N)
                del dd, hh
            del h
            cp.get_default_memory_pool().free_all_blocks()
            if k in (0, 1):
                full[:-1, k] += s.private_mu
            if k in (2, 12):
                full[:-1, k] += s.private_ve
            full[-1, k] = full[0, k]
        backing.flush()
        log('BOUNDED ORBIT COMPONENT COMPLETE', k)
    return full, derivative, None


def check(device):
    from common import model, OUT, write, log
    from native_path import attach_native_path
    from streaming_periodic import StreamPeriodic
    from orbit_reconstruction import orbit_states_and_derivative
    from spectral_grid_sampler import SpectralGridSampler
    from scipy.signal import resample
    s = model()
    attach_native_path(s)
    z = np.load(OUT / 'periodic/seed_N512.npz')
    n = 2048
    indices = np.unique(np.r_[0, (-2 * np.arange(1, 127)) % n])
    cases = []
    for N in (len(z['r']), len(z['r']) + 1):
        r = z['r'] if N == len(z['r']) else resample(z['r'], N, axis=0)
        sol = dict(r=r, T=float(z['T']), D=float(z['D']))
        o = StreamPeriodic(s, N, device)
        o.cache_mean_operators = False
        old, old_d, _ = orbit_states_and_derivative(o, sol, n,
                            derivative_indices=indices, include_rate=False)
        new, new_d, _ = orbit_states_and_derivative_bounded(o, sol, n,
                            derivative_indices=indices, population_block=17)
        scale = max(np.linalg.norm(old), 1e-30)
        err = float(np.linalg.norm(old - new) / scale)
        derr = float(np.linalg.norm(old_d - new_d) / max(np.linalg.norm(old_d), 1e-30))
        maximum = float(np.max(abs(old - new)))
        assert err < 1e-12 and derr < 1e-12 and maximum < 1e-10
        first = SpectralGridSampler(old, sol['T'], old_d, indices)
        second = SpectralGridSampler(new, sol['T'], new_d, indices)
        slots = np.r_[0, 1, 51, 1003, n]
        sampled = np.max(abs(first(slots * sol['T'] / n) - second(slots * sol['T'] / n)))
        dsampled = np.max(abs(first(indices * sol['T'] / n, 1) - second(indices * sol['T'] / n, 1)))
        assert sampled < 1e-10 and dsampled < 1e-9
        cases.append(dict(N=N, n=n, relative_state_error=err, relative_derivative_error=derr,
                     max_state_error=maximum, sample_error=float(sampled), derivative_sample_error=float(dsampled),
                     bitwise_state=bool(np.array_equal(old, new)), bitwise_derivative=bool(np.array_equal(old_d, new_d))))
        del first, second, old, new, old_d, new_d, o
    q = dict(status='BOUNDED_RECONSTRUCTION_PARITY_PASS', cases=cases,
             source='periodic/seed_N512.npz', models_changed=False,
             scope='Same 14 states, original harmonics, exact-grid sampler and phase derivatives. Layout/storage and independent FFT-row batching only; not a bifurcation result.')
    write(OUT / 'bounded_reconstruction_parity.json', q)
    log('BOUNDED RECONSTRUCTION CHECK', q)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=0)
    check(p.parse_args().device)

"""M4 empirical linear-mode estimator.

Fit an affine discrete map x[t+1] ~= A x[t] + b to a low-dimensional observable trajectory
[rate_E_mean, q_I_mean, S_G], then convert eig(A) to continuous rates lambda = log(mu)/dt.
The leading complex pair gives a descriptive growth rate and frequency.

This is a data-driven low-dimensional mode estimate. It is not a full network Jacobian and
not a proof of a Hopf bifurcation; it only tests whether the measured operating-window
trajectory contains a decaying or growing oscillatory component.
"""
import numpy as np


def estimate_linear_modes(X, dt):
    """X: (T, d) observable trajectory (rows = time), dt in ms. Fit affine discrete map x[t+1] = A x[t] + b,
    return continuous eigenvalues lambda = log(eig(A))/dt and the leading complex pair as (sigma [1/ms],
    omega [rad/ms], f [Hz]). f_hz = omega/(2*pi)*1e3 (rad/ms -> cycles/s)."""
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError(f"X must be (T,d), got {X.shape}")
    X0, X1 = X[:-1], X[1:]
    Z = np.hstack([X0, np.ones((len(X0), 1))])           # affine design: [x | 1]
    M, *_ = np.linalg.lstsq(Z, X1, rcond=None)           # (d+1, d): rows [A^T ; b^T]
    A = M[:-1].T                                          # (d,d) discrete transition
    mu = np.linalg.eigvals(A)                             # discrete multipliers
    lam = np.log(mu.astype(complex)) / dt                # continuous eigenvalues [1/ms]
    lam = lam[np.argsort(-lam.real)]                     # most-unstable first
    comp = lam[np.abs(lam.imag) > 1e-9]                  # complex modes (oscillatory)
    lead = comp[0] if len(comp) else lam[0]
    omega = float(abs(lead.imag))
    return dict(eigs_per_ms=lam, sigma_per_ms=float(lead.real), omega_rad_per_ms=omega,
                f_hz=omega / (2.0 * np.pi) * 1e3, is_complex=bool(len(comp)), n_dim=X.shape[1])


def _ground_truth_traj(sigma, f_hz, dt, T_ms, x0=(1.0, 0.0), noise=1e-4, seed=0):
    """Exact discrete rotation-scaling ground truth: x[t+1] = exp(sigma*dt) R(omega*dt) x[t] + noise.
    True continuous eigenvalues are exactly sigma +/- i*omega, so the estimator must recover them up to noise."""
    omega = 2.0 * np.pi * f_hz / 1e3                      # rad/ms
    g = np.exp(sigma * dt)
    c, s = np.cos(omega * dt), np.sin(omega * dt)
    M = g * np.array([[c, -s], [s, c]])                  # exact one-step propagator
    n = int(round(T_ms / dt))
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2)); X[0] = x0
    for t in range(n - 1):
        X[t + 1] = M @ X[t] + noise * rng.standard_normal(2)
    return X


def _synthetic_selftest():
    dt = 0.1                                              # ms, matches the sim
    cases = [(-0.002, 5.0), (+0.0015, 5.0), (-0.006, 8.0)]  # (sigma 1/ms, f Hz): stable-osc / growing-osc / faster-decay
    for sigma_true, f_true in cases:
        X = _ground_truth_traj(sigma_true, f_true, dt, T_ms=8000.0)
        r = estimate_linear_modes(X, dt)
        assert r["is_complex"], f"expected complex pair, got {r['eigs_per_ms']}"
        assert abs(r["f_hz"] - f_true) < 0.5, f"freq {r['f_hz']:.3f} != {f_true} (+/-0.5Hz)"
        assert np.sign(r["sigma_per_ms"]) == np.sign(sigma_true), \
            f"sigma sign wrong: {r['sigma_per_ms']:+.5f} vs true {sigma_true:+.5f}"
        assert abs(r["sigma_per_ms"] - sigma_true) < 5e-4, \
            f"sigma {r['sigma_per_ms']:+.5f} != {sigma_true:+.5f} (+/-5e-4)"
        print(f"  OK true(sigma={sigma_true:+.4f}/ms f={f_true}Hz) -> "
              f"est(sigma={r['sigma_per_ms']:+.5f}/ms f={r['f_hz']:.3f}Hz complex={r['is_complex']})")
    print("SYNTHETIC SELFTEST PASSED")


def load_observables(npz_path, label):
    """Low-D observable trajectory [rate_E_mean(Hz), q_I_mean, S_G] from a dynamic_qi_traces.npz arm."""
    with np.load(npz_path, allow_pickle=True) as z:
        return np.column_stack([np.asarray(z[f"{label}__{s}"], float)
                                for s in ("rate", "trace_qI_mean", "trace_SG")])  # (T,3)


def fit_operating_point(npz_path, label, win_ms, dt=0.1, smooth_ms=10.0, ds=50):
    """Fit empirical linear modes on a quasi-stationary window of the operating point. Smooth (average out
    fast HFO-band noise) + downsample to the envelope timescale (dt_eff = dt*ds) + z-score columns (rate/qI/SG
    live on very different scales; standardize so the fit is not rate-variance-dominated), then estimate modes."""
    X = load_observables(npz_path, label)
    lo, hi = win_ms
    X = X[int(lo / dt):min(int(hi / dt), len(X))]
    w = max(1, int(smooth_ms / dt)); k = np.ones(w) / w
    Xs = np.column_stack([np.convolve(X[:, i], k, mode="same") for i in range(X.shape[1])])[::ds]
    Xz = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    r = estimate_linear_modes(Xz, dt * ds)
    r["window_ms"] = win_ms; r["n_samples"] = len(Xs); r["dt_eff_ms"] = dt * ds
    return r


def _real_data_smoke_test():
    """SMOKE TEST ONLY (one seed, tool shakedown) -- NOT the step-⑤ result. Confirms the real-data path runs
    and the bounded vs runaway operating points give a sane sigma ordering (bounded sigma<~0, runaway sigma>0)."""
    d = "results/topic4_m4_dynamic_finescan_seed2/dynamic_qi_traces.npz"
    print("\n=== REAL-DATA SMOKE TEST (seed2, NOT the ⑤ result) ===")
    for lab, win, tag in [("kq0.10_aG14.0", (3000, 15000), "BOUNDED"),
                          ("kq0.10_aG16.0", (2000, 5500), "RUNAWAY@5781 pre-escape")]:
        r = fit_operating_point(d, lab, win)
        print(f"  {lab} [{tag}] win={r['window_ms']}ms n={r['n_samples']} "
              f"-> sigma={r['sigma_per_ms']:+.5f}/ms f={r['f_hz']:.2f}Hz complex={r['is_complex']}")


if __name__ == "__main__":
    import sys
    _synthetic_selftest()
    if "--smoke" in sys.argv:
        _real_data_smoke_test()

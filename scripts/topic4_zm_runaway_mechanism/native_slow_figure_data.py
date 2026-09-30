"""Actual native slow observations for comparison figures, not fitted targets."""
from common import OUT, ROOT, BASE, np, read, log


def main():
    source = ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401'
    times, Z, M = [], [], []
    for p in sorted((source/'fields').glob('*.npz')):
        with np.load(p) as z:
            times.extend(z['zm_step']*.1)
            Z.extend(z['z'].mean(1, dtype=float))
            M.extend(z['m'].mean(1, dtype=float)*.0005)
    with np.load(source/'checkpoints/t12500ms.npz') as z:
        times.append(12500.)
        Z.append(z['slow__z'][:32000].mean())
        M.append(z['slow__m'][:32000].mean()*.0005)
    t, Z, M = np.array(times), np.array(Z), np.array(M)
    checkpoints = read(BASE/'native_reference/checkpoint_projections.json')
    for tm in [8000, 9000, 9420, 9870, 10370, 12500]:
        i = np.flatnonzero(t==tm)[0]
        assert abs(1-Z[i]-checkpoints[str(tm)]['D']) < 1e-8
        assert abs(M[i]-checkpoints[str(tm)]['mean_M']*.0005) < 1e-7
    p = OUT/'transient_native_Z_path_20260923/native_slow_observations.npz'
    if p.exists():
        old = np.load(p)
        assert np.array_equal(t, old['time_ms'])
        assert np.array_equal(Z, old['Z'])
        assert np.array_equal(M, old['M_feedback_mV'])
    else:
        np.savez_compressed(p, time_ms=t, Z=Z, M_feedback_mV=M, original_observations=True)
    log('NATIVE SLOW OBSERVATIONS CHECKED', len(t))


if __name__ == '__main__':
    main()

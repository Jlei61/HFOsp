"""Bounded continuation beyond J_EE,core=1.15; reuse the unchanged v2 equations.

No native SNN runs are added. Newton residual and full-state recurrence are
checked before any orbit is admitted to the new figure.
"""
from pathlib import Path
import sys, json, os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/topic4_core_bifurcation_v2'))
from model import System, OUT as SOURCE
from periodic import Orbit
from scipy.signal import resample
import numpy as np

OUT = ROOT / 'results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915'

def main():
    dest = OUT / 'periodic'
    dest.mkdir(parents=True, exist_ok=True)
    s = System()
    z = np.load(SOURCE / 'periodic/g1.15000000_N2048.npz')
    r, T = z['r'], float(z['T'])
    rows = []
    # The accepted extension is bounded at 1.175. Attempts at 1.18 and
    # 1.1775 did not converge; their residuals are retained separately and
    # are not treated as physical branch endpoints or new bifurcations.
    for g in (1.155, 1.16, 1.17, 1.1725, 1.175):
        path = dest / f'g{g:.8f}_N2048.npz'
        if path.exists():
            z = np.load(path)
            r, T, err, hist = z['r'], float(z['T']), float(z['residual']), z['history'].tolist()
        else:
            r, T, err, hist = Orbit(s, g, 2048).solve(r, T, maxiter=24)
            if not np.isfinite(err) or err > 1e-8:
                raise RuntimeError(f'Periodic continuation failed at J={g}: {err}')
            np.savez_compressed(path, g=g, r=r, T=T, residual=err, history=hist)
        rr = resample(r, 4096, axis=0)
        defect = Orbit(s, g, 4096).evaluate(np.r_[(rr/.01).ravel(), np.log(T)], rr, np.zeros_like(rr))
        row = dict(J_EE_core=g, T_ms=T, N=len(r), residual=err,
                   offgrid_defect_hz=float(np.max(abs(defect[:-1]))*10),
                   minimum_hz=(r.min(0)*1000).tolist(), mean_hz=(r.mean(0)*1000).tolist(),
                   maximum_hz=(r.max(0)*1000).tolist(), source=str(path))
        rows.append(row)
        print('ACCEPTED_ORBIT', json.dumps(row), flush=True)
        (OUT / 'extended_periodic_branch.json').write_text(json.dumps(rows, indent=2)+'\n')

if __name__ == '__main__':
    main()

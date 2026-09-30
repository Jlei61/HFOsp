"""Bounded stationary searches seeded by already computed physical-D trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import argparse
from pathlib import Path
import numpy as np
from topic4_fig5_z_bifurcation_preview import Equilibrium, ROOT

OLD = ROOT / 'results/topic4_sef_hfo/fig5_z_branch_extension_20260915'
OUT = ROOT / 'results/topic4_sef_hfo/fig5_D_fast_slow_20260916'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['newton', 'hybrid'], default='newton')
    method = parser.parse_args().method
    OUT.mkdir(exist_ok=True)
    eq = Equilibrium(); m = eq.m
    source = np.load(OLD / 'deterministic/s0.228845.npz')['r_hz']
    seeds = [source[-1000:].mean(0), source[-1], source[-500]]
    result = OUT / 'target_root_search.json'
    rows = json.loads(result.read_text()) if result.exists() else []
    for k, seed in enumerate(seeds):
        if any(r.get('method', 'newton') == method and r['trial'] == k for r in rows):
            continue
        macro = np.r_[(seed[:3200].reshape(400, 8) * m.w_u.reshape(400, 8)).sum(1), seed[3200:]]
        D = .228844760565
        saved = OUT / f'target_root_{method}_trial{k}.npz'
        if saved.exists():
            a = np.load(saved); rate = a['r_hz']; residual = float(np.max(np.abs(eq.evaluate(rate, D))))
            ok = bool(residual < 2e-6 and rate.min() > -1e-7)
        else:
            rate, residual, ok = eq.solve(macro, D) if method == 'hybrid' else eq.newton(macro, D, iterations=30)
        row = dict(trial=k, method=method, D=D, seed='last-second mean' if k == 0 else f'late snapshot {k}',
                   valid=bool(ok), residual_hz=float(residual), global_E_hz=float(np.average(rate[:400], weights=m.count_e)))
        rows.append(row)
        np.savez_compressed(OUT / f'target_root_{method}_trial{k}.npz', D=D, r_hz=rate, residual_hz=residual)
        (OUT / 'target_root_search.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(row, flush=True)
        if ok:
            # Only successful roots may seed continuation. Both directions bounded.
            import topic4_fig5_z_bifurcation_preview as cont
            cont.OUT = OUT
            for direction in (-1, 1):
                cont.trace_branch(eq, rate, D, direction, f'target_equilibria_{direction:+d}',
                                  max_points=90, lower_bound=.14, upper_bound=.31,
                                  step_max=.02, trust_curvature=True)
            break


if __name__ == '__main__':
    main()

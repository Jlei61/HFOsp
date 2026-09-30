"""Positive real characteristic-root certificate for new physical-D equilibria."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import argparse
import numpy as np
from scipy.optimize import brentq
from topic4_fig5_z_frozen_v1 import Characteristic
from analyze_topic4_fig5_D_target_roots import OUT


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--complex', action='store_true'); args = parser.parse_args()
    c = Characteristic()
    a = np.load(OUT / 'target_root_hybrid_trial0.npz')
    r, D = a['r_hz'], float(a['D']); c.at(r, D)
    if args.complex:
        from topic4_fig5_z_characteristic_root_newton import root
        result = json.loads((OUT/'target_root_stability.json').read_text())
        result['complex_search'] = []
        for guess in (5+65j, 20+150j, 70+100j):
            lam, mode, err, history = root(c, guess, maxiter=14)
            found = err < 1e-7 and lam.real > 1e-5
            row = dict(guess_per_s=[guess.real,guess.imag], lambda_per_s=[lam.real,lam.imag],
                       residual=float(err), status='UNSTABLE_COMPLEX_MULTIPLIER_CERTIFIED' if found else 'UNCLASSIFIED')
            result['complex_search'].append(row); print('COMPLEX',row,flush=True)
            if found:
                reg = [np.bincount(c.m.cell_e[c.m.g175 == k], minlength=400) for k in range(3)]
                energy = np.array([np.dot(w, abs(mode[:400])**2) for w in reg]); energy /= energy.sum()
                result.update(status=row['status'], lambda_per_s=row['lambda_per_s'], frequency_hz=abs(lam.imag)/2/np.pi,
                              weighted_macro_E_mode_energy_A_B_surround=energy.tolist(),
                              exact_map_multiplier_modulus=float(np.exp(lam.real*c.m.dt/1000)))
                np.savez_compressed(OUT/'target_root_unstable_mode.npz', D=D, rate_hz=r, mode=mode, growth_per_s=lam)
            (OUT/'target_root_stability.json').write_text(json.dumps(result,indent=2)+'\n')
            if found: break
        return
    grid = [0., .1, .3, 1., 3., 10., 30., 100., 300., 1000., 3000., 10000.]
    signs = []
    bracket = None
    for lam in grid:
        sign, logdet = np.linalg.slogdet(c.matrix(lam).real)
        signs.append([lam, float(sign), float(logdet)])
        print('CHARACTERISTIC', signs[-1], flush=True)
        if len(signs) > 1 and sign*signs[-2][1] < 0:
            bracket = [signs[-2][0], lam]
            break
    result = dict(D=D, global_E_hz=float(np.average(r[:400], weights=c.m.count_e)),
                  fixed_point_residual_hz=float(np.max(np.abs(c.eq.evaluate(r,D)))),
                  local_M_elimination_residual_hz=c.eq.last['local_error_hz'],
                  determinant_scan=signs, positive_growth_bracket_per_s=bracket,
                  status='UNSTABLE_REAL_MULTIPLIER_CERTIFIED' if bracket else 'UNCLASSIFIED')
    if bracket:
        lo, hi = bracket
        # Determinant sign bisection is insensitive to determinant magnitude.
        sign_lo = np.linalg.slogdet(c.matrix(lo).real)[0]
        for _ in range(25):
            mid = (lo+hi)/2
            if np.linalg.slogdet(c.matrix(mid).real)[0] == sign_lo: lo = mid
            else: hi = mid
        lam = (lo+hi)/2
        mat = c.matrix(lam)
        _, singular, vh = np.linalg.svd(mat)
        mode = vh[-1].conj()
        reg = [np.bincount(c.m.cell_e[c.m.g175 == k], minlength=400) for k in range(3)]
        energy = np.array([np.dot(w, abs(mode[:400])**2) for w in reg]); energy /= energy.sum()
        result.update(positive_growth_per_s=lam, refined_growth_bracket_per_s=[lo, hi],
                      smallest_characteristic_singular_value=float(singular[-1]),
                      weighted_macro_E_mode_energy_A_B_surround=energy.tolist(),
                      exact_map_multiplier=float(np.exp(lam*c.m.dt/1000)))
        np.savez_compressed(OUT/'target_root_unstable_mode.npz', D=D, rate_hz=r, mode=mode, growth_per_s=lam)
    (OUT/'target_root_stability.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result, flush=True)


if __name__ == '__main__': main()

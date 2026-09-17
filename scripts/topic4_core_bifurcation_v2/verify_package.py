"""Bounded checks of the published equations, frozen input and saved orbits.

This does not launch native SNN simulations or a new continuation campaign.
Run from any working directory with Python, NumPy, SciPy and Numba available.
"""
from pathlib import Path
import json

from model import System, OUT
from branches import fold
from periodic import Orbit
from rate_paths import ROOT, saved_path
import numpy as np


def main():
    assert OUT.is_relative_to(ROOT), 'Model input must belong to this checkout'
    system = System()
    first, second = system.weights(1.)
    scaled_first, scaled_second = system.weights(1.23)
    selected = np.zeros((6, 6), dtype=bool)
    selected[0, 0] = selected[1, 1] = True
    np.testing.assert_allclose(scaled_first[:, selected], first[:, selected] * 1.23)
    np.testing.assert_allclose(scaled_second[selected], second[selected] * 1.23**2)
    assert np.array_equal(first[:, ~selected], scaled_first[:, ~selected])
    assert np.array_equal(second[~selected], scaled_second[~selected])
    for a in (0, 3):
        for b in (1, 4):
            assert not first[:, a, b].any() and not first[:, b, a].any()

    reference = json.loads((OUT/'fold.json').read_text())
    computed = fold(system)
    g_error = abs(computed['g'] - reference['g'])
    rate_error = float(np.max(abs(np.array(computed['r_hz']) - reference['r_hz'])))
    assert computed['residual'] < 1e-7 and g_error < 1e-9 and rate_error < 1e-5
    assert computed['transversality'] != 0 and computed['quadratic'] != 0
    rate = np.array(computed['r_hz']) / 1000
    zero_mode_error = float(np.max(abs(system.characteristic(0., rate, computed['g']) @ np.array(computed['v']))))
    assert zero_mode_error < 1e-7

    v7 = ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916'
    rows = json.loads((v7/'reduced_condition_coordinates.json').read_text())
    checks = []
    for number in ('20b', '12a', '15a', '15b'):
        row = next(r for r in rows if r['number'] == number)
        path = saved_path(row['path'])
        assert path.is_relative_to(ROOT)
        with np.load(path) as data:
            r = data['r']; period = float(data['T']); g = float(data['g'])
        orbit = Orbit(system, g, len(r))
        derivative = np.fft.irfft(2j*np.pi*orbit.k[:, None]*np.fft.rfft(r, axis=0), n=len(r), axis=0)
        phase = derivative / np.sum(derivative**2) * .01
        residual = float(np.max(abs(orbit.evaluate(np.r_[(r/.01).ravel(), np.log(period)], r, phase))))
        mean_error = float(np.max(abs(r.mean(0)*1000 - row['mean'])))
        assert residual < 1e-7 and mean_error < 1e-8
        checks.append(dict(condition=number, file=str(path.relative_to(ROOT)),
                           period_ms=period, residual=residual, mean_error_hz=mean_error))
    result = dict(status='PASS', source_root=str(ROOT), model_source=str(OUT.relative_to(ROOT)),
                  coupling_scaling='Only AA/BB EE first moments scale by g; second moments by g^2',
                  no_direct_core_AB_edges=True, fold_g_error=g_error,
                  fold_rate_error_hz=rate_error, fold_residual=computed['residual'],
                  fold_zero_mode_error=zero_mode_error, periodic_orbit_checks=checks,
                  scope='Published equations and four saved periodic solutions re-evaluated. Historical Floquet classifications are not all recomputed; native-SNN correspondence remains unvalidated.')
    destination = ROOT/'results/topic4_sef_hfo/six_rate_model_publication_20260917'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'equation_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

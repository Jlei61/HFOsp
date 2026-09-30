"""Read-only bandwidth audit of the four previously selected cycle inputs.

The variance arrays are *unfiltered intensity drives* used by the local LIF
assay, whereas mean current already contains the synaptic mean and imposed M.
Input power is not an error estimate or a bifurcation/validation certificate.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def spectrum(x, period_ms):
    """One-sided AC mean-square contributions, including an even Nyquist."""
    n = len(x)
    y = np.asarray(x, dtype=float) - np.mean(x)
    f = np.fft.rfftfreq(n, d=period_ms / n / 1000.)
    h = np.fft.rfft(y) / n
    power = abs(h) ** 2
    power[1:] *= 2
    if n % 2 == 0:
        power[-1] *= .5
    power[0] = 0
    variance = np.mean(y ** 2)
    error = abs(power.sum() - variance) / max(variance, 1e-30)
    assert error < 1e-12, error
    return f, power, error


def main():
    contract = json.loads((OUT / 'cycle_input_bandwidth_contract.json').read_text())
    data = np.load(OUT / contract['source'])
    info = json.loads((OUT / 'native_cycle_waveform_response/preparation.json').read_text())
    closure = json.loads((OUT / 'frozen_data/response_closure/closure.json').read_text())
    original_band = closure['fit_band_hz']
    repaired = json.loads((OUT / 'response_oscillatory_capacity_contract.json').read_text())['training_frequencies_hz']
    assert original_band == contract['original_fit_frequencies_hz']
    assert repaired == contract['later_capacity_fit_frequencies_hz']
    # Independent analytic check includes odd/even lengths and a Nyquist mode.
    checks = []
    for n in [1023, 1024]:
        t = np.arange(n) / n
        x = 7 + 2 * np.cos(2 * np.pi * 10 * t) + 3 * np.sin(2 * np.pi * 120 * t)
        f, power, error = spectrum(x, 1000.)
        assert abs(power[f > 40].sum() / power.sum() - 9 / 13) < 1e-12
        checks.append(error)
    f, power, error = spectrum((-1.) ** np.arange(1024), 1000.)
    assert abs(power[-1] - 1) < 1e-12
    checks.append(error)
    rows, powers = [], []
    for group, wave in zip(info['groups'], data['wave']):
        for name, unit, x in zip(contract['channels'], ['mV', 'mV^2', 'mV^2'], wave):
            f, power, error = spectrum(x, float(data['T_ms']))
            total = power.sum()
            cumulative = np.cumsum(power) / total
            rows.append(dict(
                label=group['label'], group=group['group'], channel=name, unit=unit,
                mean=float(np.mean(x)), minimum=float(np.min(x)), maximum=float(np.max(x)),
                fluctuation_rms=float(np.sqrt(total)),
                fraction_AC_power_above_Hz={str(c): float(power[f > c].sum() / total)
                                           for c in contract['cutoffs_hz']},
                frequency_containing_95_percent_AC_power_hz=float(f[np.searchsorted(cumulative, .95)]),
                frequency_containing_99_percent_AC_power_hz=float(f[np.searchsorted(cumulative, .99)]),
                parseval_relative_error=float(error)))
            powers.append(power)
    dest = OUT / 'cycle_input_bandwidth'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest / 'spectra.npz', frequency_hz=f, AC_power=np.array(powers))
    result = dict(status='READ_ONLY_BANDWIDTH_COMPLETE', rows=rows,
                  source=contract['source'], D=float(data['D']), T_ms=float(data['T_ms']),
                  original_fit_frequencies_hz=original_band,
                  later_capacity_fit_frequencies_hz=repaired,
                  analytic_check_max_error=max(checks),
                  input_definition=info['drive'], statistical_unit=contract['statistical_unit'],
                  scope=contract['scope'], model_changed=False, new_simulations=0)
    (dest / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))
    for row in rows:
        print(row['label'], row['channel'], row['fraction_AC_power_above_Hz'],
              'f95', row['frequency_containing_95_percent_AC_power_hz'])


if __name__ == '__main__':
    main()

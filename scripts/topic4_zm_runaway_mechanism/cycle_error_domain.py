"""Locate existing local waveform errors relative to response-table support."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def read(path):
    return json.loads(path.read_text())


def main():
    contract = read(OUT / 'cycle_error_domain_contract.json')
    data = np.load(OUT / contract['source'])
    predictions = np.load(OUT / contract['prediction_source'])
    corrected = np.load(OUT / contract['voltage_corrected_prediction_source'])
    observations = np.load(OUT / contract['reference_source'])
    base = OUT / 'native_cycle_waveform_response'
    groups = read(base / 'preparation.json')['groups']
    primary = read(base / 'result.json')
    closure = read(OUT / 'frozen_data/response_closure/closure.json')
    table = np.load(OUT / 'frozen_data/response_closure/closure.npz')
    params = read(ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    taus = [params['tau_r_' + k] + params['tau_d_' + k] for k in ['AMPA', 'GABA']]
    w = data['wave'].shape[-1]
    period = float(data['T_ms'])
    lam = 2j * np.pi * np.arange(w // 2 + 1) / period

    def filt(x, tau):
        return np.fft.irfft(np.fft.rfft(x) / (1 + lam * tau), n=w)

    assert np.max(abs(filt(np.full(w, 3.), 7.) - 3.)) < 1e-12
    t = np.arange(w) * period / w
    freq = 2 * np.pi / period
    expected = np.real(np.exp(1j * freq * t) / (1 + 1j * freq * 7.))
    assert np.max(abs(filt(np.cos(freq * t), 7.) - expected)) < 1e-12
    phase = ((np.arange(primary['steps']) + 1) * primary['dt_ms'] / period) % 1
    b = observations['measured_hz'].shape[1]
    bins = np.minimum((phase * b).astype(int), b - 1)
    exposure = np.bincount(bins, minlength=b)
    assert np.max(abs(exposure * primary['dt_ms'] - observations['occupancy_ms'])) < 1e-10
    pos = phase * w
    lo = np.floor(pos).astype(int) % w
    hi = (lo + 1) % w
    alpha = pos - np.floor(pos)

    def sample(x):
        return (1 - alpha) * x[lo] + alpha * x[hi]

    rows = []
    for j, group in enumerate(groups):
        pop = group['pop']
        mu, raw_e, raw_i = data['wave'][j]
        ve, vi = filt(raw_e, taus[0] / 2), filt(raw_i, taus[1] / 2)
        poles = closure['poles'][pop]
        mus = filt(mu, poles['tau_s'])
        ves, vis = filt(ve, poles['tau_vE']), filt(vi, poles['tau_vI'])
        domains = [(mu, ve, vi), (mus, ves, vis)]
        scale = group['theta_mv'] - 11
        xlo, xhi = table['x_' + pop][[0, -1]]
        emax, imax = table['sE_' + pop][-1], table['sI_' + pop][-1]
        variants = [(name, moment, predictions, False) for name, moment in zip(contract['variants'], domains)]
        variants += [(name, moment, corrected, True) for name, moment in zip(contract['voltage_corrected_variants'], domains)]
        for name, (m, e, i), curves, corrected_units in variants:
            k = list(curves['variants']).index(name)
            x = (sample(m) - 11) / scale
            se = np.sqrt(np.maximum(sample(e), 0)) / scale
            si = np.sqrt(np.maximum(sample(i), 0)) / scale
            outside = (x < xlo) | (x > xhi) | (se > emax) | (si > imax)
            fraction = np.bincount(bins, weights=outside.astype(float), minlength=b) / exposure
            assert np.array_equal(curves['measured_hz'], observations['measured_hz'])
            error = (curves['predicted_hz'][j, k] - observations['measured_hz'][j]) ** 2
            # Equal-bin weighting reproduces the original waveform L2 gate.
            masks = dict(wholly_inside=fraction == 0, wholly_outside=fraction == 1,
                         mixed=(fraction > 0) & (fraction < 1))
            allocation = {key: float(error[mask].sum() / error.sum()) for key, mask in masks.items()}
            assert abs(sum(allocation.values()) - 1) < 1e-12
            if corrected_units:
                original = read(OUT / 'nonlinear_mean_readout/result.json')['rows'][j]['rows'][k]
                expected_l2 = original['waveform_L2']
            else:
                original = read(base / 'history_weights_result.json')['groups'][j]['rows'][k]
                expected_l2 = original['relative_waveform_L2']
            l2 = np.sqrt(error.sum()) / max(np.linalg.norm(observations['measured_hz'][j]), np.sqrt(b))
            assert abs(l2 - expected_l2) < 1e-12
            rows.append(dict(label=group['label'], group=group['group'], variant=name,
                             corrected_voltage_units=corrected_units,
                             relative_waveform_L2=float(l2), outside_time_fraction=float(outside.mean()),
                             binned_error_energy_fraction=allocation,
                             proportional_outside_error_fraction=float(error @ fraction / error.sum()),
                             bins={key: int(mask.sum()) for key, mask in masks.items()},
                             lookup_limits=dict(x=[float(xlo), float(xhi)], sigma_E_max=float(emax), sigma_I_max=float(imax))))
    result = dict(status='READ_ONLY_ERROR_DOMAIN_COMPLETE', rows=rows,
                  scope=contract['scope'], statistical_unit=contract['statistical_unit'],
                  allocation_note='Mixed bins are retained separately; proportional allocation is descriptive, not an exact within-bin error decomposition.',
                  model_changed=False, new_simulations=0)
    dest = OUT / 'cycle_error_domain'
    dest.mkdir(exist_ok=True)
    (dest / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print(row['label'], row['variant'], row['outside_time_fraction'], row['binned_error_energy_fraction'])


if __name__ == '__main__':
    main()

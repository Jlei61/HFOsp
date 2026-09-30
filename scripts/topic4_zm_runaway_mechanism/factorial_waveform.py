"""Local mean/variance waveform counterfactuals at original strong amplitudes."""
from in_domain_waveform import acquire
from native_cycle_waveform_response import condition
from common import OUT, ROOT, np, read, write, log, ResponseParams
from transfer_spline import TransferSpline
from response_voltage_units import VoltageScaledResponseTable
import argparse

DEST = OUT / 'factorial_waveform'


def prepare():
    c = read(OUT / 'factorial_waveform_contract.json')
    z = np.load(OUT / c['source'])
    groups = read(OUT / 'native_cycle_waveform_response/preparation.json')['groups']
    response = ResponseParams(OUT / 'frozen_data/response_closure/closure.json')
    transfer = {p: TransferSpline(OUT / f'frozen_data/transfer_table/table_{p}.npz') for p in 'EI'}
    period = float(z['T_ms'])
    w = z['wave'].shape[-1]
    lam = 2j * np.pi * np.arange(w // 2 + 1) / period
    params = read(ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    taus = [params['tau_r_' + k] + params['tau_d_' + k] for k in ['AMPA', 'GABA']]
    def filt(x, tau):
        return np.fft.irfft(np.fft.rfft(x) / (1 + lam * tau), n=w)
    old = np.load(OUT / 'nonlinear_mean_readout/response.npz')
    primary = read(OUT / 'native_cycle_waveform_response/result.json')
    phase = ((np.arange(primary['steps']) + 1) * primary['dt_ms'] / period) % 1
    b = c['phase_bins']
    bins = np.minimum((phase * b).astype(int), b - 1)
    exposure = np.bincount(bins, minlength=b)
    pos = phase * w
    lo = np.floor(pos).astype(int) % w
    hi = (lo + 1) % w
    alpha = pos - np.floor(pos)
    def aggregate(x):
        return np.bincount(bins, weights=(1 - alpha) * x[lo] + alpha * x[hi], minlength=b) / exposure
    waves, predictions, rows, pars, parity = [], [], [], [], []
    for j, (group, source) in enumerate(zip(groups, z['wave'])):
        pop, theta = group['pop'], group['theta_mv']
        tab = VoltageScaledResponseTable(response.tables[pop])
        pole = response.poles[pop]
        th = np.full(w, theta)
        for name in c['conditions']:
            wave = source.copy()
            if name == 'mean_only':
                wave[1:] = source[1:].mean(axis=1, keepdims=True)
                assert np.array_equal(wave[0], source[0])
            elif name == 'variances_only':
                wave[0] = source[0].mean()
                assert np.array_equal(wave[1:], source[1:])
            else:
                assert name == 'full' and np.array_equal(wave, source)
            assert np.min(wave[1:]) > 0
            mu, raw_e, raw_i = wave
            ve, vi = filt(raw_e, taus[0] / 2), filt(raw_i, taus[1] / 2)
            mus = filt(mu, pole['tau_s'])
            vef, vif = filt(ve, pole['tau_cE']), filt(vi, pole['tau_cI'])
            ves, vis = filt(ve, pole['tau_vE']), filt(vi, pole['tau_vI'])
            curves = []
            for lookup in [(mu, ve, vi), (mus, ves, vis)]:
                (a, ae, ai, ee, ei), _ = tab.evaluate(*lookup, th)
                m = a * mu + (1 - a) * mus + ee * (ve - vef) + ei * (vi - vif)
                e = np.maximum(ae * ve + (1 - ae) * ves, 0)
                i = np.maximum(ai * vi + (1 - ai) * vis, 0)
                curves.append(transfer[pop].evaluate(m, e, i, th)['rate'] * 1000)
            if name == 'full':
                for k in range(2):
                    reference = old['predicted_hz'][j, k]
                    error = float(np.linalg.norm(aggregate(curves[k]) - reference) / max(np.linalg.norm(reference), 1))
                    assert error < 1e-9, error
                    parity.append(error)
            waves.append(wave)
            predictions.append(curves)
            pars.append(condition(0, theta, 1, 1, pop, dt=c['dt_ms']))
            rows.append(dict(source_label=group['label'], source_group=group['group'],
                             pop=pop, condition=name, theta_mv=theta))
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST / 'prepared.npz', wave=waves, predictions_hz=predictions,
                        pars=pars, T_ms=period)
    write(DEST / 'preparation.json', dict(status='PREDICTIONS_LOCKED_BEFORE_ACQUISITION', rows=rows,
                                        variants=c['variants'], T_ms=period, cases=len(rows),
                                        full_input_prediction_parity=parity, new_coefficients=0))
    log('FACTORIAL PREDICTIONS LOCKED', len(rows), 'parity', max(parity))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare()
    else:
        acquire('factorial_waveform_contract.json', DEST)

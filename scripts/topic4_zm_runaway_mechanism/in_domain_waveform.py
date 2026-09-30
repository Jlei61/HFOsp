"""Fresh bounded-input local response assay; no spatial network or fitting."""
from native_cycle_waveform_response import simulate, condition
from common import OUT, ROOT, np, read, write, log, ResponseParams
from transfer_spline import TransferSpline
from response_voltage_units import VoltageScaledResponseTable
import argparse

DEST = OUT / 'in_domain_waveform'


def prepare():
    c = read(OUT / 'in_domain_waveform_contract.json')
    z = np.load(OUT / c['source'])
    info = read(OUT / 'native_cycle_waveform_response/preparation.json')['groups']
    response = ResponseParams(OUT / 'frozen_data/response_closure/closure.json')
    transfer = {p: TransferSpline(OUT / f'frozen_data/transfer_table/table_{p}.npz') for p in 'EI'}
    theta, reset = c['threshold_mv'], c['reset_mv']
    scale = theta - reset
    period = float(z['T_ms'])
    w = z['wave'].shape[-1]
    lam = 2j * np.pi * np.arange(w // 2 + 1) / period
    params = read(ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    taus = [params['tau_r_' + k] + params['tau_d_' + k] for k in ['AMPA', 'GABA']]
    def filt(x, tau):
        return np.fft.irfft(np.fft.rfft(x) / (1 + lam * tau), n=w)
    waves, predictions, rows, pars = [], [], [], []
    for group, source in zip(info, z['wave']):
        assert group['label'] in c['shape_groups']
        ac = source - source.mean(axis=1, keepdims=True)
        ac /= np.max(abs(ac), axis=1, keepdims=True)
        assert np.max(abs(ac)) <= 1 + 1e-14
        pop = group['pop']
        tab = VoltageScaledResponseTable(response.tables[pop])
        pole = response.poles[pop]
        th = np.full(w, theta)
        for amplitude in c['amplitudes']:
            mu = reset + scale * (c['baseline_normalized_mean'] + amplitude * ac[0])
            raw_e = (scale * c['baseline_sigma_E']) ** 2 * (1 + amplitude * ac[1])
            raw_i = (scale * c['baseline_sigma_I']) ** 2 * (1 + amplitude * ac[2])
            wave = np.array([mu, raw_e, raw_i])
            assert np.min(wave[1:]) > 0
            ve, vi = filt(raw_e, taus[0] / 2), filt(raw_i, taus[1] / 2)
            mus = filt(mu, pole['tau_s'])
            vef, vif = filt(ve, pole['tau_cE']), filt(vi, pole['tau_cI'])
            ves, vis = filt(ve, pole['tau_vE']), filt(vi, pole['tau_vI'])
            curves = []
            for lookup in [(mu, ve, vi), (mus, ves, vis)]:
                m, e, i = lookup
                # Check the complete periodic history, not only selected bins.
                x = (m - reset) / scale
                se = np.sqrt(e) / scale
                si = np.sqrt(i) / scale
                assert x.min() > tab.x[0] and x.max() < tab.x[-1]
                assert se.min() > 0 and se.max() < tab.sEmax
                assert si.min() > 0 and si.max() < tab.sImax
                (alpha, ae, ai, eta_e, eta_i), _ = tab.evaluate(m, e, i, th)
                effective_mu = alpha * mu + (1 - alpha) * mus + eta_e * (ve - vef) + eta_i * (vi - vif)
                effective_e = np.maximum(ae * ve + (1 - ae) * ves, 0)
                effective_i = np.maximum(ai * vi + (1 - ai) * vis, 0)
                curves.append(transfer[pop].evaluate(effective_mu, effective_e, effective_i, th)['rate'] * 1000)
            predictions.append(curves)
            waves.append(wave)
            pars.append(condition(0, theta, 1, 1, pop, dt=c['dt_ms']))
            rows.append(dict(source_label=group['label'], pop=pop, amplitude=amplitude,
                             theta_mv=theta, all_lookup_times_in_domain=True))
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST / 'prepared.npz', wave=waves, predictions_hz=predictions,
                        pars=pars, T_ms=period)
    write(DEST / 'preparation.json', dict(status='PREDICTIONS_LOCKED_BEFORE_ACQUISITION', rows=rows,
                                        variants=c['variants'], T_ms=period, cases=len(rows),
                                        source=c['source'], new_coefficients=0))
    log('IN-DOMAIN PREDICTIONS LOCKED', len(rows))


def acquire(contract_name='in_domain_waveform_contract.json', dest=DEST):
    c = read(OUT / contract_name)
    assert read(OUT / 'native_cycle_waveform_response/implementation_check.json')['status'] == 'PASS'
    info = read(dest / 'preparation.json')
    assert info['status'] == 'PREDICTIONS_LOCKED_BEFORE_ACQUISITION'
    z = np.load(dest / 'prepared.npz')
    r, dt, b = c['replicates'], c['dt_ms'], c['phase_bins']
    period = float(z['T_ms'])
    steps, burn = round(c['record_cycles'] * period / dt), round(c['burn_cycles'] * period / dt)
    label = c.get('log_label', 'IN-DOMAIN')
    log(label + ' ACQUISITION', len(info['rows']), r, steps, burn, dt)
    counts = simulate(z['pars'], z['wave'], r, period, dt, burn, steps, c['seed'], b, c['device'])
    phase = ((np.arange(steps) + 1) * dt / period) % 1
    bins = np.minimum((phase * b).astype(int), b - 1)
    exposure = np.bincount(bins, minlength=b)
    occupancy = exposure * dt
    rates = counts / occupancy[None, None, :] * 1000
    measured = rates.mean(axis=1)
    sem = rates.std(axis=1, ddof=1) / np.sqrt(r)
    w = z['wave'].shape[-1]
    pos = phase * w
    lo = np.floor(pos).astype(int) % w
    hi = (lo + 1) % w
    alpha = pos - np.floor(pos)
    def aggregate(x):
        return np.bincount(bins, weights=(1 - alpha) * x[lo] + alpha * x[hi], minlength=b) / exposure
    predictions = np.array([[aggregate(x) for x in group] for group in z['predictions_hz']])
    rows = []
    for j, group in enumerate(info['rows']):
        target = measured[j]
        denom = max(np.linalg.norm(target), np.sqrt(b))
        mc_means = counts[j].sum(axis=1) / (steps * dt) * 1000
        mean = float(mc_means.mean())
        group_rows = []
        for name, pred in zip(c['variants'], predictions[j]):
            error = float(np.linalg.norm(pred - target) / denom)
            bias = float(abs(np.average(pred, weights=occupancy) - mean) / max(mean, 1))
            ac_error = float(np.linalg.norm((pred - pred.mean()) - (target - target.mean())) /
                             max(np.linalg.norm(target - target.mean()), np.sqrt(b)))
            gate = c['acceptance']
            group_rows.append(dict(variant=name, waveform_L2=error, relative_mean_error=bias,
                                   AC_waveform_L2_descriptive=ac_error,
                                   passed=error <= gate['normalized_waveform_RMSE_max'] and bias <= gate['relative_cycle_mean_error_max']))
        split = float(np.linalg.norm(rates[j, :r//2].mean(0) - rates[j, r//2:].mean(0)) / denom)
        rows.append(dict(**group, MC_mean_hz=mean, MC_mean_SEM_hz=float(mc_means.std(ddof=1) / np.sqrt(r)),
                         MC_split_half_relative_difference=split, rows=group_rows))
    np.savez_compressed(dest / 'response.npz', counts=counts, predicted_hz=predictions,
                        measured_hz=measured, sem_hz=sem, occupancy_ms=occupancy,
                        phase_centres=(np.arange(b) + .5) / b, T_ms=period)
    result = dict(status=c.get('result_status', 'IN_DOMAIN_WAVEFORM_ASSAY_COMPLETE'), rows=rows, scope=c['scope'],
                  replicates=r, dt_ms=dt, steps=steps, burn_steps=burn,
                  all_cases_pass={name: all(group['rows'][k]['passed'] for group in rows) for k, name in enumerate(c['variants'])},
                  statistical_unit=c['statistical_unit'], model_promoted=False)
    write(dest / 'result.json', result)
    log(label + ' WAVEFORM RESULT', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        acquire()

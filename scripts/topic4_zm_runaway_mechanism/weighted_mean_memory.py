"""Weight each mean-input increment before, rather than after, causal memory."""
from common import OUT, ROOT, np, read, write, log, ResponseParams
from transfer_spline import TransferSpline
from response_voltage_units import VoltageScaledResponseTable


def main(all_channels=False):
    contract_name = 'weighted_all_memory_contract.json' if all_channels else 'weighted_mean_memory_contract.json'
    c = read(OUT / contract_name)
    response = ResponseParams(OUT / 'frozen_data/response_closure/closure.json')
    transfer = {p: TransferSpline(OUT / f'frozen_data/transfer_table/table_{p}.npz') for p in 'EI'}
    params = read(ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    syn = [(params['tau_r_' + k] + params['tau_d_' + k]) / 2 for k in ['AMPA', 'GABA']]
    dest = OUT / c.get('output_directory', 'weighted_mean_memory')
    dest.mkdir(exist_ok=True)
    all_rows, identities, constant_checks = [], [], []
    for dataset in c['datasets']:
        source = OUT / dataset
        assert read(source / 'independent_audit.json')['status'] == 'COUNT_LEVEL_AUDIT_PASS'
        info = read(source / 'preparation.json')
        result = read(source / 'result.json')
        data = np.load(source / 'prepared.npz')
        obs = np.load(source / 'response.npz')
        w = data['wave'].shape[-1]
        period = float(data['T_ms'])
        lam = 2j * np.pi * np.arange(w // 2 + 1) / period
        def filt(x, tau):
            return np.fft.irfft(np.fft.rfft(x) / (1 + lam * tau), n=w)
        def diff(x):
            return np.fft.irfft(np.fft.rfft(x) * lam, n=w)
        phase = ((np.arange(result['steps']) + 1) * result['dt_ms'] / period) % 1
        b = obs['measured_hz'].shape[1]
        bins = np.minimum((phase * b).astype(int), b - 1)
        exposure = np.bincount(bins, minlength=b)
        pos = phase * w
        lo = np.floor(pos).astype(int) % w
        hi = (lo + 1) % w
        fraction = pos - np.floor(pos)
        def aggregate(x):
            return np.bincount(bins, weights=(1 - fraction) * x[lo] + fraction * x[hi], minlength=b) / exposure
        curves = []
        for j, group in enumerate(info['rows']):
            pop, theta = group['pop'], group['theta_mv']
            pole = response.poles[pop]
            tab = VoltageScaledResponseTable(response.tables[pop])
            th = np.full(w, theta)
            def evaluate(wave, history=False, weighted=True):
                mu, raw_e, raw_i = wave
                ve, vi = filt(raw_e, syn[0]), filt(raw_i, syn[1])
                mus = filt(mu, pole['tau_s'])
                vef, vif = filt(ve, pole['tau_cE']), filt(vi, pole['tau_cI'])
                ves, vis = filt(ve, pole['tau_vE']), filt(vi, pole['tau_vI'])
                coord = (mus, ves, vis) if history else (mu, ve, vi)
                (alpha, ae, ai, ee, ei), _ = tab.evaluate(*coord, th)
                if weighted:
                    memory = filt(pole['tau_s'] * (1 - alpha) * diff(mu), pole['tau_s'])
                    effective = mu - memory
                else:
                    effective = alpha * mu + (1 - alpha) * mus
                if weighted and all_channels:
                    effective += filt(pole['tau_cE'] * ee * diff(ve), pole['tau_cE'])
                    effective += filt(pole['tau_cI'] * ei * diff(vi), pole['tau_cI'])
                    e = np.maximum(ve - filt(pole['tau_vE'] * (1 - ae) * diff(ve), pole['tau_vE']), 0)
                    i = np.maximum(vi - filt(pole['tau_vI'] * (1 - ai) * diff(vi), pole['tau_vI']), 0)
                else:
                    effective += ee * (ve - vef) + ei * (vi - vif)
                    e = np.maximum(ae * ve + (1 - ae) * ves, 0)
                    i = np.maximum(ai * vi + (1 - ai) * vis, 0)
                return transfer[pop].evaluate(effective, e, i, th)['rate'] * 1000
            group_rows, group_curves = [], []
            for k, name in enumerate(c['variants']):
                raw = evaluate(data['wave'][j], bool(k))
                pred = aggregate(raw)
                target = obs['measured_hz'][j]
                error = float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), np.sqrt(b)))
                mean = float(np.average(pred, weights=obs['occupancy_ms']))
                bias = abs(mean - result['rows'][j]['MC_mean_hz']) / max(result['rows'][j]['MC_mean_hz'], 1.)
                physical = bool(raw.min() >= -1e-10 and raw.max() <= 1000 / params['tau_ref_' + pop] + 1e-10)
                group_rows.append(dict(variant=name, waveform_L2=error, relative_mean_error=bias,
                                       mean_hz=mean, minimum_raw_hz=float(raw.min()), maximum_raw_hz=float(raw.max()),
                                       physical_bounds_pass=physical,
                                       passed=physical and error <= c['acceptance']['normalized_waveform_RMSE_max'] and bias <= c['acceptance']['relative_cycle_mean_error_max']))
                group_curves.append(pred)
            all_rows.append(dict(dataset=dataset, **group, rows=group_rows))
            curves.append(group_curves)
            # Independent analytic identities at one E and one I workpoint.
            if dataset == c['datasets'][0] and j in (0, 9):
                t = np.arange(w) * period / w
                test = 18 + np.sin(2 * np.pi * t / period) + .1 * np.cos(6 * np.pi * t / period)
                a = .37
                actual = test - filt(pole['tau_s'] * (1 - a) * diff(test), pole['tau_s'])
                expected = a * test + (1 - a) * filt(test, pole['tau_s'])
                parity = float(np.max(abs(actual - expected)))
                assert parity < 1e-10
                constant_checks.append(dict(pop=pop, maximum_identity_error=parity))
                span = theta - 11
                base = np.array([np.full(w, 11 + .6 * span), np.full(w, (span * .7) ** 2), np.full(w, (span * 1.4) ** 2)])
                direction = np.array([span * np.sin(2 * np.pi * t / period), span ** 2 * .1 * np.cos(2 * np.pi * t / period), span ** 2 * .2 * np.sin(2 * np.pi * t / period + .7)])
                for history in (False, True):
                    errors = []
                    for eps in (1e-4, 5e-5):
                        dp = (evaluate(base + eps * direction, history, True) - evaluate(base - eps * direction, history, True)) / (2 * eps)
                        dq = (evaluate(base + eps * direction, history, False) - evaluate(base - eps * direction, history, False)) / (2 * eps)
                        errors.append(float(np.linalg.norm(dp - dq) / max(np.linalg.norm(dq), 1.)))
                    assert errors[-1] < 1e-6, errors
                    identities.append(dict(pop=pop, history=history, first_derivative_errors=errors))
        np.savez_compressed(dest / f'{dataset}.npz', predicted_hz=curves, measured_hz=obs['measured_hz'],
                            variants=c['variants'], phase_centres=obs['phase_centres'], T_ms=period)
    outcome = dict(status='WEIGHTED_ALL_MEMORY_DIAGNOSTIC_COMPLETE' if all_channels else 'WEIGHTED_MEAN_MEMORY_DIAGNOSTIC_COMPLETE', rows=all_rows,
                   constant_weight_identity=constant_checks, first_order_identity=identities,
                   all_cases_pass={name: all(r['rows'][k]['passed'] for r in all_rows) for k, name in enumerate(c['variants'])},
                   scope=c['scope'], model_promoted=False)
    write(dest / 'result.json', outcome)
    log('WEIGHTED MEAN MEMORY', outcome['all_cases_pass'])
    for row in all_rows:
        log(row['dataset'], row['source_label'], row.get('condition', row.get('amplitude')),
            [(x['variant'], round(x['waveform_L2'], 4), round(x['relative_mean_error'], 4), x['passed']) for x in row['rows']])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--all-channels', action='store_true')
    main(p.parse_args().all_channels)

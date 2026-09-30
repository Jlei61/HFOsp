"""Calibrated local real-pole versus damped-pair response, no network changes.

The pair has an explicit real two-state realization:
 h1' = a(q-h1)+b h2, h2' = b(q-h1)-a h2.
Its outputs q-h1 and h2 have transfer s(s+a)/den and s*b/den,
den=(s+a)^2+b^2. Both vanish at DC, and local eigenvalues are -a+-ib.
No midpoint observations are fitted. Poles are channel-local diagnostics, not
yet a shared-pole neuronal population model or a spatial response table.
"""
from common import *
from scipy.optimize import least_squares

DEST = OUT/'response_oscillatory_capacity'


def design(frequency, taus, decay=None, oscillation=None):
    w = 2j*np.pi*np.asarray(frequency)/1000
    z = w[:, None]*np.asarray(taus)
    A = z/(1+z)
    if decay is not None:
        den = (w+decay)**2+oscillation**2
        A = np.column_stack([A, w*(w+decay)/den, w*oscillation/den])
    return A


def fit(f, response, sem, base, gain, stationary_hz, kind, contract):
    if kind == 'real5':
        taus = contract['real5_times_ms']
    else:
        taus = contract['pair_real_times_ms']
    def coefficients(decay=None, oscillation=None):
        A = gain*design(f, taus, decay, oscillation)/sem[:, None]
        y = (response-base)/sem
        B = np.r_[A.real, A.imag, np.sqrt(contract['ridge'])*np.eye(A.shape[1])]
        target = np.r_[y.real, y.imag, np.zeros(A.shape[1])]
        coef = np.linalg.lstsq(B, target, rcond=None)[0]
        return coef, B@coef-target
    if kind == 'real5':
        coef, residual = coefficients()
        return dict(kind=kind, real_times_ms=taus, coefficients=coef.tolist(),
                    objective=float(residual@residual), success=True)
    bounds = np.log(np.array([[1/contract['decay_time_bounds_ms'][1], 2*np.pi*contract['oscillation_bounds_hz'][0]/1000],
                             [1/contract['decay_time_bounds_ms'][0], 2*np.pi*contract['oscillation_bounds_hz'][1]/1000]]))
    trials = []
    for decay_ms in contract['initial_decay_times_ms']:
        for factor in contract['initial_frequency_factors']:
            hz = np.clip(stationary_hz*factor, *contract['oscillation_bounds_hz'])
            initial = np.log([1/decay_ms, 2*np.pi*hz/1000])
            initial = np.clip(initial, bounds[0]+1e-8, bounds[1]-1e-8)
            solver = least_squares(lambda v: coefficients(*np.exp(v))[1], initial,
                                   bounds=bounds, max_nfev=contract['max_nfev'],
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            a, b = np.exp(solver.x)
            coef, residual = coefficients(a, b)
            trials.append(dict(kind=kind, real_times_ms=taus, decay_per_ms=float(a),
                               oscillation_per_ms=float(b), coefficients=coef.tolist(),
                               objective=float(residual@residual), success=bool(solver.success),
                               nfev=solver.nfev, optimality=float(solver.optimality)))
    # Prespecified multistart selection uses only fitting frequencies.
    best = min(trials, key=lambda r: r['objective'])
    return dict(**best, trials=trials)


def predict(f, base, gain, fitted, synaptic_ms):
    A = design(f, fitted['real_times_ms'], fitted.get('decay_per_ms'), fitted.get('oscillation_per_ms'))
    return (base+gain*A@np.asarray(fitted['coefficients']))/(1+2j*np.pi*np.asarray(f)/1000*synaptic_ms/2)


def main():
    c = read(OUT/'response_oscillatory_capacity_contract.json')
    source = read(OUT/'response_midpoint_assay/result.json')
    meta = read(OUT/'response_midpoint_assay/metadata.json')
    s = model(); DEST.mkdir(exist_ok=True)
    measurements = {(r['line'], r['point'], r['channel'], r['frequency_hz']): r for r in source['measurements']}
    frequencies = np.array(c['training_frequencies_hz'])
    fit_mask = np.isin(frequencies, c['selection_fit_frequencies_hz'])
    rows = []
    for line_id, line in enumerate(meta['lines']):
        pop, ch = line['pop'], line['channel']
        for sig in line['knots']:
            k = line['sigmas'].index(sig)
            zero = measurements[(line_id, k, ch, 0.)]
            mean = measurements[(line_id, k, 0, 0.)]
            se = sig if ch==1 else c['fixed_sigma_E']; si = sig if ch==2 else c['fixed_sigma_I']
            mu, ve, vi = 11+7*line['x'], (7*se)**2, (7*si)**2
            static = s.spline[pop].evaluate(np.array([mu]), np.array([ve]), np.array([vi]), np.array([18.]))
            base = static['d_ve' if ch==1 else 'd_vi'][0]*1000
            gain = static['d_mu'][0]*1000/7
            stationary = static['rate'][0]*1000
            observations = [measurements[(line_id, k, ch, f)] for f in frequencies]
            raw = np.array([complex(*r['measured']) for r in observations])
            raw_sem = np.array([r['sem'] for r in observations])
            syn = 1+2j*np.pi*frequencies/1000*s.tau[ch-1]/2
            noise = np.maximum(raw_sem*abs(syn), c['SEM_floor_fraction']*max(abs(base), .01*abs(gain), 1e-8))
            dc = zero['measured'][0]
            eligible = abs(dc)/max(zero['sem'], 1e-12)>=10 and abs(mean['measured'][0])/max(mean['sem'], 1e-12)>=10
            variants = []
            for kind in ['real5', 'real3_pair']:
                partial = fit(frequencies[fit_mask], (raw*syn)[fit_mask], noise[fit_mask], base, gain, stationary, kind, c)
                held = predict(frequencies[~fit_mask], base, gain, partial, s.tau[ch-1])
                held_error = abs(held-raw[~fit_mask])/max(abs(dc), 1e-12)
                full = fit(frequencies, raw*syn, noise, base, gain, stationary, kind, c)
                prediction = predict(frequencies, base, gain, full, s.tau[ch-1])
                future = predict(c['fresh_frequencies_hz'], base, gain, full, s.tau[ch-1])
                variants.append(dict(kind=kind, selection_fit=partial,
                    frequency_holdout_errors=held_error.tolist(), full_fit=full,
                    training_errors=(abs(prediction-raw)/max(abs(dc), 1e-12)).tolist(),
                    locked_fresh_predictions=[[v.real, v.imag] for v in future]))
            row = dict(line=line_id, point=k, pop=pop, channel=ch, x=line['x'], sigma=sig,
                       workpoint=dict(pop=pop, theta=18., mu=mu, ve=ve, vi=vi),
                       eligible=bool(eligible), DC=dc, stationary_rate_hz=float(stationary),
                       static_DC=float(base), gain_factor=float(gain), variants=variants)
            rows.append(row)
            log('OSCILLATORY FIT', len(rows), pop, ch, line['x'], sig,
                [(v['kind'], v['frequency_holdout_errors']) for v in variants])
    summary = []
    tol = np.where(frequencies[~fit_mask]<=25, .1, .15)
    for pop in 'EI':
        for ch in [1, 2]:
            for x in c['x_values']:
                selected = [r for r in rows if r['pop']==pop and r['channel']==ch and r['x']==x and r['eligible']]
                methods = {}
                for idx, kind in enumerate(['real5', 'real3_pair']):
                    errors = np.array([r['variants'][idx]['frequency_holdout_errors'] for r in selected])
                    tr = np.array([r['variants'][idx]['training_errors'] for r in selected])
                    methods[kind] = dict(heldout_failures=int((errors>tol).sum()), heldout_count=int(errors.size),
                         median_heldout_error=float(np.median(errors)) if errors.size else None,
                         median_training_error=float(np.median(tr)) if tr.size else None)
                summary.append(dict(pop=pop, channel=ch, x=x, methods=methods))
    result = dict(status='FITS_AND_FRESH_PREDICTIONS_LOCKED', rows=rows, summaries=summary,
                  fresh_frequencies_hz=c['fresh_frequencies_hz'], fresh_observations='NOT_YET_ACQUIRED',
                  scope=c['scope'])
    write(DEST/'fit_result.json', result)
    log('OSCILLATORY CAPACITY', summary)


if __name__ == '__main__':
    main()

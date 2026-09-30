"""Independent count aggregation and calibration-source audit for midpoint assay."""
from common import *
from scipy.interpolate import CubicSpline, PchipInterpolator

DEST = OUT/'response_midpoint_assay'


def main():
    c = read(OUT/'response_midpoint_assay_contract.json')
    metadata = read(DEST/'metadata.json'); assert metadata['status'] == 'COMPLETE'
    result = read(DEST/'result.json'); assert result['status'] == 'MIDPOINT_ASSAY_COMPLETE'
    z = np.load(DEST/'checkpoint.npz'); assert z['done'].all()
    observations = np.load(DEST/'observations.npy', mmap_mode='r')
    parameters = z['parameters']
    assert len(parameters) == len(metadata['meta']) == len(result['measurements'])
    lookup = {}; max_count_error = 0.; max_sem_error = 0.
    for i, m in enumerate(metadata['meta']):
        p = parameters[i]; ch = m['channel']
        amplitude = p[4]*(1 if ch == 0 else p[2 if ch == 1 else 3])
        pair = observations[i, :, :2]
        est = pair*1000/(c['duration_ms']*amplitude)
        value = est.mean(axis=0)
        sem = np.sqrt(np.sum(np.var(est, axis=0, ddof=0))/c['replicates'])
        original = result['measurements'][i]
        max_count_error = max(max_count_error, float(np.max(abs(value-original['measured']))))
        max_sem_error = max(max_sem_error, abs(float(sem)-original['sem']))
        lookup[(m['line'], m['point'], ch, m['frequency_hz'])] = (complex(*value), float(sem))
    assert max_count_error < 1e-10 and max_sem_error < 1e-10
    s = model(); frequency = np.array(c['frequencies_hz'][1:])
    w = 2j*np.pi*frequency/1000; t = np.array(c['filter_times_ms'])
    basis = w[:, None]*t/(1+w[:, None]*t)
    max_prediction_error = 0.; counts = {}; checked_points = 0; calibration_rows = []
    for line_id, line in enumerate(metadata['lines']):
        ch = line['channel']; pop = line['pop']; sigmas = np.array(line['sigmas']); knots = np.array(line['knots'])
        ids = [int(np.flatnonzero(sigmas == v)[0]) for v in knots]
        coeff = []
        for k in ids:
            sigma = sigmas[k]; se = sigma if ch==1 else c['fixed_sigma_E']; si = sigma if ch==2 else c['fixed_sigma_I']
            g = s.spline[pop].evaluate(np.array([11+7*line['x']]), np.array([(7*se)**2]),
                                      np.array([(7*si)**2]), np.array([18.]))
            gain = g['d_mu'][0]*1000/7; base = g['d_ve' if ch==1 else 'd_vi'][0]*1000
            response = np.array([lookup[(line_id, k, ch, f)][0] for f in frequency])
            sem = np.array([lookup[(line_id, k, ch, f)][1] for f in frequency])
            filt = 1+w*s.tau[ch-1]/2
            scale = np.maximum(sem*abs(filt), c['SEM_floor_fraction']*max(abs(base), .01*abs(gain), 1e-8))
            A = gain*basis/scale[:, None]; y = (response*filt-base)/scale
            lhs = (A.conj().T@A).real+c['ridge']*np.eye(len(t))
            rhs = (A.conj().T@y).real
            coef = np.linalg.solve(lhs, rhs)
            assert np.linalg.norm(lhs@coef-rhs) < 1e-7*max(np.linalg.norm(rhs), 1)
            coeff.append(coef)
            own_dc, own_sem = lookup[(line_id, k, ch, 0.)]
            mean_dc, mean_sem = lookup[(line_id, k, 0, 0.)]
            own_snr = abs(own_dc.real)/max(own_sem, 1e-12)
            fit_prediction = (base+gain*basis@coef)/filt
            fit_errors = abs(fit_prediction-response)/max(abs(own_dc.real), 1e-12)
            calibration_rows.append(dict(line=line_id, pop=pop, channel=ch, x=line['x'], sigma=float(sigma),
                own_DC_SNR=float(own_snr), eligible=bool(own_snr>=10 and abs(mean_dc.real)/max(mean_sem, 1e-12)>=10),
                measured=[[v.real, v.imag] for v in response], SEM=sem.tolist(),
                predicted=[[v.real, v.imag] for v in fit_prediction], errors=fit_errors.tolist(),
                failures=int((fit_errors>np.where(frequency<=25, .1, .15)).sum()),
                scope='In-sample fit adequacy at calibration knots only; no midpoint fitting.'))
        coeff = np.array(coeff)
        cubic = CubicSpline(knots, coeff, axis=0)
        pchip = PchipInterpolator(knots, coeff, axis=0)
        log_pchip = PchipInterpolator(np.log(knots), coeff, axis=0)
        for r in [q for q in result['rows'] if q['line'] == line_id]:
            v = r['sigma']; assert v not in knots
            k = int(np.flatnonzero(sigmas == v)[0]); assert k not in ids
            left = np.searchsorted(knots, v)-1
            assert abs(v-(knots[left]+knots[left+1])/2) < 1e-12
            assert all(source != k for source in ids)
            # Midpoint measurements are accessed only after all coefficients are fitted.
            dc, dcsem = lookup[(line_id, k, ch, 0.)]
            dmu, dmusem = lookup[(line_id, k, 0, 0.)]
            eligible = abs(dc.real)/max(dcsem, 1e-12)>=10 and abs(dmu.real)/max(dmusem, 1e-12)>=10
            assert bool(eligible) == r['eligible']
            se = v if ch==1 else c['fixed_sigma_E']; si = v if ch==2 else c['fixed_sigma_I']
            g = s.spline[pop].evaluate(np.array([11+7*line['x']]), np.array([(7*se)**2]),
                                      np.array([(7*si)**2]), np.array([18.]))
            gain = g['d_mu'][0]*1000/7; base = g['d_ve' if ch==1 else 'd_vi'][0]*1000
            fields = {'linear_sigma': (coeff[left]+coeff[left+1])/2, 'cubic_sigma': cubic(v),
                      'pchip_sigma': pchip(v), 'pchip_log_sigma': log_pchip(np.log(v))}
            target = np.array([lookup[(line_id, k, ch, f)][0] for f in frequency])
            for name, value in fields.items():
                pred = (base+gain*basis@value)/(1+w*s.tau[ch-1]/2)
                old = np.array([complex(*z) for z in r['methods'][name]['predicted']])
                max_prediction_error = max(max_prediction_error, float(np.max(abs(pred-old))))
                error = abs(pred-target)/max(abs(dc.real), 1e-12)
                assert np.allclose(error, r['methods'][name]['errors'], atol=1e-8, rtol=1e-8)
                failed = int((error>np.where(frequency<=25, .1, .15)).sum())
                assert failed == r['methods'][name]['failures']
                if eligible:
                    key = f'{pop}_{ch}_{name}'
                    pair = counts.setdefault(key, [0, 0]); pair[0]+=failed; pair[1]+=len(frequency)
            checked_points += 1
    assert max_prediction_error < 1e-8
    write(DEST/'independent_audit.json', dict(status='COUNT_AND_NO_MIDPOINT_FIT_AUDIT_PASS',
          conditions=len(parameters), checked_midpoints=checked_points,
          max_count_mean_error=max_count_error, max_sem_error=max_sem_error,
          max_independent_prediction_error=max_prediction_error, frequency_failure_counts=counts,
          calibration_fit_rows=calibration_rows,
          scope='Numerical and source-separation audit, not a scientific accuracy pass or accepted network.'))
    log('MIDPOINT AUDIT PASS', counts)


if __name__ == '__main__':
    main()

"""Acquire fresh frequencies after local response predictions have been locked."""
from response_oscillatory_capacity import DEST, predict, design
from common import *
from lif_mc import condition, run
import argparse
import hashlib


def main(device):
    contract = read(OUT/'response_oscillatory_capacity_contract.json')
    settings = contract['fresh_assay']
    fitpath = DEST/'fit_result.json'
    fit = read(fitpath); assert fit['status'] == 'FITS_AND_FRESH_PREDICTIONS_LOCKED'
    digest = hashlib.sha256(fitpath.read_bytes()).hexdigest()
    s = model()
    # Realization and locked-prediction checks precede data acquisition.
    realization_error = 0.; positive_poles = 0; unfinished_optimizations = 0
    for item in fit['rows']:
        for method in item['variants']:
            f = method['full_fit']
            predicted = predict(contract['fresh_frequencies_hz'], item['static_DC'], item['gain_factor'], f, s.tau[item['channel']-1])
            assert np.allclose(predicted, [complex(*v) for v in method['locked_fresh_predictions']], atol=1e-12, rtol=1e-12)
            if f['kind'] == 'real3_pair':
                a, b = f['decay_per_ms'], f['oscillation_per_ms']
                A = np.array([[-a, b], [-b, -a]])
                assert np.max(np.linalg.eigvals(A).real) < 0
                assert np.allclose(A@np.array([1., 0.])+np.array([a, b]), 0, atol=1e-14)
                positive_poles += 1
                unfinished_optimizations += not f['success']
                for hz in [0., 15., 60., 120., 200.]:
                    w = 2j*np.pi*hz/1000
                    state = np.linalg.solve(w*np.eye(2)-A, np.array([a, b]))
                    outputs = np.array([1-state[0], state[1]])
                    analytic = design([hz], [], a, b)[0]
                    realization_error = max(realization_error, float(np.max(abs(outputs-analytic))))
    assert realization_error < 1e-12
    write(DEST/'realization_audit.json', dict(status='STABLE_REAL_REALIZATION_AND_LOCK_PASS',
          max_transfer_error=realization_error, checked_pairs=positive_poles,
          selected_fits_at_optimizer_budget=unfinished_optimizations,
          fit_sha256=digest, scope='Realization, DC and prediction identity only; not response accuracy.'))
    meta = []; pars = []
    for j, item in enumerate(fit['rows']):
        q = item['workpoint']; ch = item['channel']
        for channel, hz in [(0, 0.), (ch, 0.)]+[(ch, f) for f in contract['fresh_frequencies_hz']]:
            amp = settings['mean_amplitude_mv'] if channel == 0 else settings['variance_amplitude_relative']
            pars.append(condition(q['mu'], q['theta'], q['ve'], q['vi'], q['pop'], amplitude=amp, freq_hz=hz, channel=channel))
            meta.append(dict(point=j, channel=channel, frequency_hz=hz))
    pars = np.asarray(pars)
    folder = DEST/'fresh'; folder.mkdir(exist_ok=True)
    ck = folder/'checkpoint.npz'; rawpath = folder/'observations.npy'
    if ck.exists():
        z = np.load(ck); assert np.array_equal(z['parameters'], pars)
        assert read(folder/'metadata.json')['fit_sha256'] == digest
        done = z['done']; observed = np.load(rawpath, mmap_mode='r+')
    else:
        done = np.zeros(len(pars), bool)
        observed = np.lib.format.open_memmap(rawpath, mode='w+', dtype=np.float64, shape=(len(pars), settings['replicates'], 4))
        np.savez(ck, parameters=pars, done=done)
    write(folder/'metadata.json', dict(status='RUNNING', fit_sha256=digest, conditions=len(pars), meta=meta))
    for start in range(0, len(pars), settings['batch']):
        idx = np.arange(start, min(start+settings['batch'], len(pars)))
        if done[idx].all(): continue
        observed[idx] = run(pars[idx], settings['replicates'], settings['duration_ms'], settings['burn_ms'], settings['seed'], device=device)
        observed.flush(); done[idx] = True
        np.savez(ck, parameters=pars, done=done)
        log('OSCILLATORY FRESH', int(done.sum()), '/', len(pars))
    assert hashlib.sha256(fitpath.read_bytes()).hexdigest() == digest
    measurements = []; lookup = {}
    for i, m in enumerate(meta):
        p = pars[i]; amp = p[4] if m['channel'] == 0 else p[4]*p[2 if m['channel'] == 1 else 3]
        samples = (observed[i, :, 0]+1j*observed[i, :, 1])*1000/(settings['duration_ms']*amp)
        avg = samples.mean(); sem = float(np.sqrt(np.mean(abs(samples-avg)**2)/settings['replicates']))
        row = dict(**m, measured=[avg.real, avg.imag], sem=sem)
        measurements.append(row); lookup[(m['point'], m['channel'], m['frequency_hz'])] = row
    rows = []
    for j, item in enumerate(fit['rows']):
        dc = lookup[(j, item['channel'], 0.)]; mean = lookup[(j, 0, 0.)]
        for k, hz in enumerate(contract['fresh_frequencies_hz']):
            target = lookup[(j, item['channel'], hz)]
            methods = {}
            for m in item['variants']:
                pred = complex(*m['locked_fresh_predictions'][k])
                error = abs(pred-complex(*target['measured']))/max(abs(dc['measured'][0]), 1e-12)
                methods[m['kind']] = dict(predicted=[pred.real, pred.imag], error=float(error),
                                         failed=bool(error>(.1 if hz<=25 else .15)))
            rows.append(dict(point=j, pop=item['pop'], channel=item['channel'], x=item['x'], sigma=item['sigma'],
                             workpoint=item['workpoint'], frequency_hz=hz, eligible=item['eligible'],
                             old_DC=item['DC'], DC=dc['measured'][0], DC_SEM=dc['sem'],
                             DC_SNR=float(abs(dc['measured'][0])/max(dc['sem'], 1e-12)),
                             mean_DC_SNR=float(abs(mean['measured'][0])/max(mean['sem'], 1e-12)),
                             measured=target['measured'], SEM=target['sem'], methods=methods))
    summary = []
    for pop in 'EI':
        for ch in [1, 2]:
            for x in contract['x_values']:
                rr = [r for r in rows if r['pop']==pop and r['channel']==ch and r['x']==x and r['eligible']]
                for band in ['within_training_band', 'above_training_band']:
                    selected = [r for r in rr if (r['frequency_hz']<=80) == (band=='within_training_band')]
                    methods = {name:dict(failures=sum(r['methods'][name]['failed'] for r in selected), n=len(selected),
                                        median_error=float(np.median([r['methods'][name]['error'] for r in selected])) if selected else None)
                               for name in ['real5', 'real3_pair']}
                    summary.append(dict(pop=pop, channel=ch, x=x, band=band, methods=methods))
    write(folder/'result.json', dict(status='FRESH_FREQUENCY_VALIDATION_COMPLETE', fit_sha256=digest,
          measurements=measurements, rows=rows, summaries=summary, replacement_promoted=False,
          scope='Fresh frequencies at calibration workpoints only. Whole-workpoint transfer and strong nonlinear response remain unvalidated.'))
    write(folder/'metadata.json', dict(status='COMPLETE', fit_sha256=digest, conditions=len(pars), meta=meta))
    log('OSCILLATORY FRESH COMPLETE', summary)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device', type=int, default=0)
    main(p.parse_args().device)

"""Fresh, prespecified operating-point interpolation assay on 12 input lines.

Coarse knots provide calibration; all arithmetic midpoints are held out as
entire workpoints. The six frequency responses at a midpoint are never used
to fit its interpolated prediction. This is a local diagnostic, not a model.
"""
from common import *
from lif_mc import condition, run
from scipy.interpolate import CubicSpline, PchipInterpolator
import argparse

DEST = OUT / 'response_midpoint_assay'


def conditions(c):
    meta = []
    pars = []
    lines = []
    for pop in 'EI':
        for x in c['x_values']:
            for ch in [1, 2]:
                knots = np.array(c['sigma_knots'][pop][str(ch)])
                mid = (knots[:-1]+knots[1:])/2
                sigmas = np.sort(np.r_[knots, mid])
                line = len(lines)
                lines.append(dict(pop=pop, x=x, channel=ch, knots=knots.tolist(), sigmas=sigmas.tolist()))
                for k, sig in enumerate(sigmas):
                    se = sig if ch == 1 else c['fixed_sigma_E']
                    si = sig if ch == 2 else c['fixed_sigma_I']
                    for channel, freq in [(0, 0.)]+[(ch, f) for f in c['frequencies_hz']]:
                        amplitude = c['mean_amplitude_mv'] if channel == 0 else c['variance_amplitude_relative']
                        pars.append(condition(11+7*x, 18., (7*se)**2, (7*si)**2, pop,
                                              amplitude=amplitude, freq_hz=freq, channel=channel))
                        meta.append(dict(line=line, point=k, sigma=float(sig), channel=channel,
                                         frequency_hz=freq, calibration=bool(np.any(sig == knots))))
    return np.array(pars), meta, lines


def analyse(c, pars, meta, lines, observed):
    s = model()
    measured = []
    for i, r in enumerate(meta):
        p = pars[i]
        amp = p[4] if r['channel'] == 0 else p[4]*p[2 if r['channel'] == 1 else 3]
        samples = (observed[i, :, 0]+1j*observed[i, :, 1])*1000/(c['duration_ms']*amp)
        m = samples.mean()
        measured.append(dict(**r, measured=[float(m.real), float(m.imag)],
                             sem=float(np.sqrt(np.mean(abs(samples-m)**2)/c['replicates']))))
    freq = np.array(c['frequencies_hz'][1:])
    lam = 2j*np.pi*freq/1000
    taus = np.array(c['filter_times_ms'])
    basis = lam[:, None]*taus/(1+lam[:, None]*taus)
    predictions = []
    for line_id, line in enumerate(lines):
        pop, ch = line['pop'], line['channel']
        sigmas = np.array(line['sigmas'])
        knot = np.isin(sigmas, line['knots'])
        mu = np.full(len(sigmas), 11+7*line['x'])
        se = sigmas if ch == 1 else np.full(len(sigmas), c['fixed_sigma_E'])
        si = sigmas if ch == 2 else np.full(len(sigmas), c['fixed_sigma_I'])
        g = s.spline[pop].evaluate(mu, (7*se)**2, (7*si)**2, np.full(len(sigmas), 18.))
        base = g['d_ve' if ch == 1 else 'd_vi']*1000
        gain = g['d_mu']*1000/7
        filt = 1+lam*s.tau[ch-1]/2
        coef = np.zeros((len(sigmas), len(taus)))
        dc = np.zeros(len(sigmas)); semdc = dc.copy(); eligible = np.zeros(len(sigmas), bool)
        obs = np.zeros((len(sigmas), len(freq)), complex); sem = np.zeros(obs.shape, float)
        for k in range(len(sigmas)):
            rows = [r for r in measured if r['line'] == line_id and r['point'] == k]
            zero = next(r for r in rows if r['channel'] == ch and r['frequency_hz'] == 0)
            mean = next(r for r in rows if r['channel'] == 0)
            dc[k] = zero['measured'][0]; semdc[k] = zero['sem']
            ac = [next(r for r in rows if r['channel'] == ch and r['frequency_hz'] == f) for f in freq]
            obs[k] = [complex(*r['measured']) for r in ac]; sem[k] = [r['sem'] for r in ac]
            eligible[k] = (abs(mean['measured'][0])/max(mean['sem'], 1e-12) >= 10
                           and abs(dc[k])/max(semdc[k], 1e-12) >= 10)
            # Only calibration knots are fitted. No midpoint oracle is computed.
            if knot[k]:
                floor = c['SEM_floor_fraction']*max(abs(base[k]), .01*abs(gain[k]), 1e-8)
                noise = np.maximum(sem[k]*abs(filt), floor)
                A = gain[k]*basis/noise[:, None]
                b = (obs[k]*filt-base[k])/noise
                coef[k] = np.linalg.lstsq(np.r_[A.real, A.imag, np.sqrt(c['ridge'])*np.eye(len(taus))],
                                         np.r_[b.real, b.imag, np.zeros(len(taus))], rcond=None)[0]
        xk = sigmas[knot]; ck = coef[knot]
        interpolators = {'cubic_sigma': CubicSpline(xk, ck, axis=0),
                         'pchip_sigma': PchipInterpolator(xk, ck, axis=0),
                         'pchip_log_sigma': PchipInterpolator(np.log(xk), ck, axis=0)}
        for k in np.flatnonzero(~knot):
            left = np.searchsorted(xk, sigmas[k])-1
            q = (sigmas[k]-xk[left])/(xk[left+1]-xk[left])
            values = {'linear_sigma': (1-q)*ck[left]+q*ck[left+1]}
            for name, interp in interpolators.items():
                values[name] = interp(np.log(sigmas[k]) if name.endswith('log_sigma') else sigmas[k])
            stats = {}
            for name, value in values.items():
                pred = (base[k]+gain[k]*(basis@value))/filt
                error = abs(pred-obs[k])/max(abs(dc[k]), 1e-12)
                stats[name] = dict(predicted=[[z.real, z.imag] for z in pred],
                                   errors=error.tolist(),
                                   failures=int((error>np.where(freq<=25, .1, .15)).sum()))
            predictions.append(dict(line=line_id, pop=pop, channel=ch, x=line['x'], sigma=float(sigmas[k]),
                                    fixed_other_sigma=float(c['fixed_sigma_I' if ch==1 else 'fixed_sigma_E']),
                                    eligible=bool(eligible[k]), DC=float(dc[k]), DC_SEM=float(semdc[k]),
                                    adjacent_calibration_eligible=eligible[k-1:k+2:2].tolist(),
                                    all_calibration_eligible=eligible[knot].tolist(),
                                    measured=[[z.real, z.imag] for z in obs[k]], SEM=sem[k].tolist(), methods=stats))
    summary = []
    for pop in 'EI':
        for ch in [1, 2]:
            selected = [r for r in predictions if r['pop']==pop and r['channel']==ch and r['eligible']]
            stats = {}
            for name in ['linear_sigma', 'cubic_sigma', 'pchip_sigma', 'pchip_log_sigma']:
                error = np.array([r['methods'][name]['errors'] for r in selected])
                stats[name] = dict(failures=sum(r['methods'][name]['failures'] for r in selected),
                                   tested_frequencies=int(error.size),
                                   median_workpoint_RMS=float(np.median(np.sqrt(np.mean(error**2, axis=1)))) if len(selected) else None)
            summary.append(dict(pop=pop, channel=ch, eligible_midpoints=len(selected),
                                total_midpoints=sum(r['pop']==pop and r['channel']==ch for r in predictions), methods=stats))
    result = dict(status='MIDPOINT_ASSAY_COMPLETE', measurements=measured, rows=predictions, summaries=summary,
                  scope=c['scope'], replacement_promoted=False, original_validation_reused=False)
    write(DEST/'result.json', result)
    log('MIDPOINT RESULTS', summary)


def main(device):
    c = read(OUT/'response_midpoint_assay_contract.json')
    DEST.mkdir(exist_ok=True)
    pars, meta, lines = conditions(c)
    path = DEST/'observations.npy'
    checkpoint = DEST/'checkpoint.npz'
    if checkpoint.exists():
        z = np.load(checkpoint); assert np.array_equal(z['parameters'], pars)
        done = z['done']; observed = np.load(path, mmap_mode='r+')
    else:
        done = np.zeros(len(pars), bool)
        observed = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                             shape=(len(pars), c['replicates'], 4))
        np.savez(checkpoint, parameters=pars, done=done)
    write(DEST/'metadata.json', dict(status='RUNNING', conditions=len(pars), meta=meta, lines=lines))
    for start in range(0, len(pars), c['batch']):
        ids = np.arange(start, min(start+c['batch'], len(pars)))
        if done[ids].all():
            continue
        observed[ids] = run(pars[ids], c['replicates'], c['duration_ms'], c['burn_ms'], c['seed'], device=device)
        observed.flush(); done[ids] = True
        np.savez(checkpoint, parameters=pars, done=done)
        log('MIDPOINT ASSAY', int(done.sum()), '/', len(pars))
    analyse(c, pars, meta, lines, observed)
    write(DEST/'metadata.json', dict(status='COMPLETE', conditions=len(pars), meta=meta, lines=lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', type=int, default=0)
    main(p.parse_args().device)

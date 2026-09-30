"""Training-only, leave-one-coordinate-plane-out response transfer diagnosis.

No independent validation, error-selected assay or SNN trajectory is loaded.
Direct local fits are an oracle/control, not cross-validation. Each spatial
prediction excludes every frequency at the target operating coordinate plane.
"""
from common import *
from scipy.interpolate import CubicSpline

DEST = OUT / 'response_workpoint_transfer'


def main():
    contract = read(OUT / 'response_workpoint_transfer_contract.json')
    old = read(OUT / 'response_bank_candidate_contract.json')
    source = read(BASE / 'dynamic_assay/rows.json')['rows']
    s = model()
    frequencies = np.array(old['all_training_frequencies_hz'])
    lam = 2j * np.pi * frequencies / 1000
    taus = np.array(old['filter_times_ms'])
    basis = lam[:, None] * taus / (1 + lam[:, None] * taus)
    summaries = []
    details = []
    saved = {}
    for pop in 'EI':
        rows = [r for r in source if r['pop'] == pop]
        grids = [np.array(sorted({r[k] for r in rows})) for k in ['x', 'sigma_E', 'sigma_I']]
        coords = [np.arcsinh(grids[0]), grids[1], grids[2]]
        shape = tuple(map(len, grids))
        n = int(np.prod(shape))
        keys = list(np.ndindex(shape))
        pindex = {tuple(grids[a][ix[a]] for a in range(3)): j for j, ix in enumerate(keys)}
        observations = [dict() for _ in keys]
        for r in rows:
            j = pindex[(r['x'], r['sigma_E'], r['sigma_I'])]
            observations[j][(r['channel'], r['frequency_hz'])] = r
        xyz = np.array(list(pindex))
        gains = s.spline[pop].evaluate(11+7*xyz[:, 0], (7*xyz[:, 1])**2,
                                     (7*xyz[:, 2])**2, np.full(n, 18.))
        gains = np.array([gains['d_mu']*7000, gains['d_ve']*49000, gains['d_vi']*49000])
        for ch in range(3):
            observed = np.zeros((n, len(frequencies)), complex)
            sem = np.ones(observed.shape)
            dc = np.zeros(n)
            eligible = np.zeros(n, bool)
            scale = 7. if ch == 0 else 49.
            filt = np.ones(len(frequencies), complex) if ch == 0 else 1 + lam*s.tau[ch-1]/2
            for j, point in enumerate(observations):
                if (ch, 0.) not in point:
                    continue
                mean_dc = point[(0, 0.)]
                rr = np.array([complex(*point[(ch, float(f))]['response']) for f in frequencies])*scale
                ss = np.array([point[(ch, float(f))]['sem'] for f in frequencies])*scale
                eligible[j] = (abs(complex(*mean_dc['response']))/max(mean_dc['sem'], 1e-12) >= 10
                               and (ch == 0 or max(abs(rr))/max(np.median(ss), 1e-12) >= 10))
                observed[j] = rr
                sem[j] = ss
                dc[j] = complex(*point[(ch, 0.)]['response']).real*scale
            corrected = observed*filt
            floor = old['SEM_floor_fraction']*np.maximum.reduce([
                abs(gains[ch]), .01*abs(gains[0]), np.full(n, 1e-5)])
            noise = np.maximum(sem*abs(filt), floor[:, None])
            coefficients = np.zeros((n, 5))
            for j in np.flatnonzero(eligible):
                A = gains[0, j]*basis/noise[j, :, None]
                target = (corrected[j]-gains[ch, j])/noise[j]
                coefficients[j] = np.linalg.lstsq(
                    np.r_[A.real, A.imag, np.sqrt(old['ridge'])*np.eye(5)],
                    np.r_[target.real, target.imag, np.zeros(5)], rcond=None)[0]
            absolute = coefficients*gains[0, :, None]
            local = (gains[ch, :, None]+absolute@basis.T)/filt
            saved[f'{pop}_{ch}_coefficients'] = coefficients
            saved[f'{pop}_{ch}_eligible'] = eligible
            by_axis = {a: [] for a in range(3)}
            eligible_interior = np.zeros(3, int)
            for j, ix in enumerate(keys):
                if not eligible[j]:
                    continue
                for axis in range(3):
                    k = ix[axis]
                    if k == 0 or k == shape[axis]-1:
                        continue
                    eligible_interior[axis] += 1
                    candidates = np.array([q for q in range(shape[axis]) if q != k])
                    near = sorted(candidates, key=lambda q: abs(coords[axis][q]-coords[axis][k]))[:4]
                    if len(near) != 4:
                        continue
                    near = np.array(sorted(near))
                    near_ids = []
                    for q in near:
                        other = list(ix)
                        other[axis] = q
                        near_ids.append(np.ravel_multi_index(tuple(other), shape))
                    if not np.all(eligible[near_ids]):
                        continue
                    # All four training sources lie outside the target plane.
                    assert all(keys[q][axis] != k for q in near_ids)
                    assert k-1 in near and k+1 in near
                    close_ids = [near_ids[list(near).index(q)] for q in [k-1, k+1]]
                    frac = (coords[axis][k]-coords[axis][k-1])/(coords[axis][k+1]-coords[axis][k-1])
                    predictions = {'local_fit_control': local[j]}
                    for name, field in [('ratio', coefficients), ('absolute_gain', absolute)]:
                        multiplier = gains[0, j] if name == 'ratio' else 1.
                        cubic = CubicSpline(coords[axis][near], field[near_ids], axis=0)(coords[axis][k])
                        linear = (1-frac)*field[close_ids[0]]+frac*field[close_ids[1]]
                        for method, c in [('cubic', cubic), ('linear', linear)]:
                            predictions[f'{name}_{method}'] = (gains[ch, j]+multiplier*(basis@c))/filt
                    errs = {name: (abs(value-observed[j])/max(abs(dc[j]), 1e-8)).tolist()
                            for name, value in predictions.items()}
                    item = dict(pop=pop, channel=ch, point=j, coordinate_index=list(ix), axis=axis,
                                source_points=near_ids, DC_measured=float(dc[j]),
                                errors=errs, mean_gain=float(gains[0, j]),
                                min_source_mean_gain=float(gains[0, near_ids].min()),
                                max_source_mean_gain=float(gains[0, near_ids].max()))
                    details.append(item)
                    by_axis[axis].append(item)
            tol = np.where(frequencies <= 25, .1, .15)
            for axis, cases in by_axis.items():
                stats = {}
                if cases:
                    for name in cases[0]['errors']:
                        e = np.array([q['errors'][name] for q in cases])
                        stats[name] = dict(median_DC_normalized_error=float(np.median(e)),
                                           p90_error=float(np.percentile(e, 90)),
                                           failed_frequencies=int((e>tol).sum()),
                                           tested_frequencies=int(e.size),
                                           median_workpoint_RMS=float(np.median(np.sqrt(np.mean(e**2, axis=1)))))
                summary = dict(pop=pop, channel=ch, axis=['asinh_x', 'sigma_E', 'sigma_I'][axis],
                               eligible_workpoints=int(eligible.sum()), eligible_interior=int(eligible_interior[axis]),
                               evaluated_four_source_workpoints=len(cases), metrics=stats)
                summaries.append(summary)
                log('WORKPOINT TRANSFER', summary)
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST/'local_fits.npz', **saved)
    write(DEST/'result.json', dict(status='TRAINING_TRANSFER_DIAGNOSTIC_COMPLETE', summaries=summaries,
                                 rows=details, scope=contract['scope'], source=str(BASE/'dynamic_assay/rows.json')))


if __name__ == '__main__':
    main()

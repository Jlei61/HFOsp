"""Bound integer-period encounters for newly checked B-leading return points.

Use every cached target within the declared J/period windows, preserving
all populations and continuous-profile norms. This is a finite catalogue
screen, not proof that global branch connections are absent.
"""
from pathlib import Path
import argparse
import time
import numpy as np
from scipy.signal import resample
from compare_rate_torus_periodic_targets import distances

from bound_rate_period_multiple_encounters import (
    DEST, RateField, families, read, write, statistics, controls, fingerprint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-prefix', type=Path, required=True)
    parser.add_argument('--current-prefix', type=Path, required=True)
    args = parser.parse_args()
    before, after = [read(p) for p in [args.previous_prefix, args.current_prefix]]
    assert before['status'] == after['status'] == 'SAMPLED_PASS'
    old, new = before['included_orbits'], after['included_orbits']
    assert new[:len(old)] == old and len(new) > len(old)
    checks = {str(Path(q['orbit']).resolve()): q for q in after['checks']}
    sources = []
    for index in range(len(old), len(new)):
        key = str(Path(new[index]).resolve())
        check = checks[key]
        assert check['filter_state_check']['positive']
        assert check['minimum_rate_Hz'] >= -1e-9
        assert check['maximum_group_defect_Hz'] < .001
        sources.append(dict(index=index, orbit=new[index], canonical_orbit=key,
            J_EE_core=check['J_EE_core'], T_ms=check['T_ms']))
    fs = families()
    catalogue = {family: [dict(index=i, orbit=q['path'],
        canonical_orbit=str(Path(q['path']).resolve()),
        J_EE_core=q['J_EE_core'], T_ms=q['T_ms']) for i, q in enumerate(fs[family])]
        for family in ['A', 'B', 'single', 'double']}
    ratios = [(1, 1)]+[(m, 1) for m in range(2, 9)]+[(1, m) for m in range(2, 9)]
    candidates = []
    needed = {q['canonical_orbit']: q for q in sources}
    for source in sources:
        for family, targets in catalogue.items():
            for target in targets:
                if abs(source['J_EE_core']-target['J_EE_core']) > .01:
                    continue
                for source_multiple, target_multiple in ratios:
                    ratio = source_multiple*source['T_ms']/(target_multiple*target['T_ms'])
                    if abs(np.log(ratio)) <= np.log(1.10):
                        candidates.append(dict(source=source, target=target, target_family=family,
                            source_period_multiple=source_multiple, target_period_multiple=target_multiple))
                        needed[target['canonical_orbit']] = target
    cache_path = DEST/'period_multiple_bounds/period_multiple_1_to_8_all_pair_bounds_moments.json'
    cached = {str(Path(q['orbit']).resolve()): q for q in read(cache_path)['rows']}
    model = RateField()
    weights = model.geo['group_size']/model.geo['group_size'].sum()
    assert model.P == len(weights) == 935
    moments, records = {}, []
    for key, item in needed.items():
        path = Path(item['orbit'])
        identity = fingerprint(path)
        previous = cached.get(key)
        if previous is not None and previous['profile_fingerprint'] == identity:
            row = dict(previous, reused_from=str(cache_path))
        else:
            with np.load(path) as z:
                assert z['r'].shape[1] == 935
                assert abs(float(z['J'])-item['J_EE_core']) < 1e-10
                assert abs(float(z['T'])-item['T_ms']) < 1e-6
                mean, sigma, correction = statistics(z['r']*1000, weights)
                row = dict(orbit=str(path), profile_fingerprint=identity,
                    J_EE_core=float(z['J']), T_ms=float(z['T']), temporal_mesh=len(z['r']),
                    population_mean_Hz=mean, temporal_RMS_Hz=sigma,
                    Nyquist_variance_correction_Hz_squared=correction)
        assert abs(row['J_EE_core']-item['J_EE_core']) < 1e-10
        assert abs(row['T_ms']-item['T_ms']) < 1e-6
        assert fingerprint(path) == identity
        moments[key] = (np.asarray(row['population_mean_Hz']), row['temporal_RMS_Hz'])
        records.append(row)
    rows = []
    for candidate in candidates:
        x, y = candidate['source'], candidate['target']
        mx, sx = moments[x['canonical_orbit']]
        my, sy = moments[y['canonical_orbit']]
        assert max(sx, sy) > 0
        bound = float(np.sqrt((mx-my)**2@weights+(sx-sy)**2)/max(sx, sy))
        rows.append(dict(**candidate, normalized_distance_lower_bound=bound))
    rows.sort(key=lambda q: q['normalized_distance_lower_bound'])
    phase_checks = []
    for family in catalogue:
        subset = [q for q in rows if q['target_family'] == family]
        if not subset:
            continue
        candidate = subset[0]
        paths = [Path(candidate[key]['orbit']) for key in ['source', 'target']]
        snapshots = [fingerprint(p) for p in paths]
        profiles = []
        for p in paths:
            with np.load(p) as z:
                profiles.append(z['r']*1000)
        sm, tm = candidate['source_period_multiple'], candidate['target_period_multiple']
        divisor = int(np.lcm(sm, tm))
        N = max(8192, 2*len(profiles[0])*sm, 2*len(profiles[1])*tm)
        N = ((N+divisor-1)//divisor)*divisor
        x, y = [np.tile(resample(r, N//m, axis=0), (m, 1))
                for r, m in zip(profiles, [sm, tm])]
        distance, phase = distances(x[:, None, :], y, weights)
        sx, sy = [moments[candidate[key]['canonical_orbit']][1] for key in ['source', 'target']]
        normalized = float(distance[0]/max(sx, sy))
        assert normalized >= candidate['normalized_distance_lower_bound']-1e-8
        assert [fingerprint(p) for p in paths] == snapshots
        phase_checks.append(dict(pair=candidate, phase_mesh=N,
            normalized_common_phase_distance=normalized, common_phase_cycles=float(phase[0]),
            profile_fingerprints=snapshots))
    summaries = []
    for family in catalogue:
        for sm, tm in ratios:
            subset = [q for q in rows if q['target_family'] == family and
                      q['source_period_multiple'] == sm and q['target_period_multiple'] == tm]
            summaries.append(dict(target_family=family, source_period_multiple=sm,
                target_period_multiple=tm, window_pairs=len(subset),
                minimum_normalized_lower_bound=subset[0]['normalized_distance_lower_bound'] if subset else None,
                unexcluded_candidates=[q for q in subset if q['normalized_distance_lower_bound'] <= .05],
                weakest_bounds=subset[:3]))
    folder = DEST/'Bleading_extension'
    name = f'prefix{len(new)}_integer_encounter_bounds'
    moments_path = folder/(name+'_moments.json')
    write(moments_path, dict(rows=records))
    result = dict(status='NEW_PREFIX_FINITE_WINDOW_PAIRS_BOUNDED', timestamp=time.time(),
        previous_prefix=str(args.previous_prefix), current_prefix=str(args.current_prefix),
        source_points=sources, target_catalogue=catalogue, moments_source=str(moments_path),
        rows=summaries, total_window_pairs=len(rows), candidate_threshold=.05,
        nearest_bound_pair_phase_checks=phase_checks,
        unexcluded_candidates=[q for q in rows if q['normalized_distance_lower_bound'] <= .05],
        parameter_window=.01, relative_period_window=1.10, analytic_controls=controls(),
        formula='D_phase_min / max(sigma_x,sigma_y) >= sqrt(norm_weighted(mean_x-mean_y)^2 + (sigma_x-sigma_y)^2) / max(sigma_x,sigma_y).',
        norm='Neuron-count weighted, all 935 populations, continuous real trigonometric profiles with Nyquist correction. Integer repetition preserves means and variances; no spatial or temporal downsampling.',
        target_scope='Cached target waveforms at their original J; their physical validity and stability are not certified by this screen.',
        global_connection_established=False, interval_completeness=False,
        scope='Only the newly accepted return points against every cached H1, H2, A-leading and alternating-burst target in the stated windows; test integer repetitions 1 through 8 in both period directions. Negative bounds exclude close identity of these finite waveform pairs, not unseen branches, unsampled connections or other period ratios.')
    write(folder/(name+'.json'), result)
    print('NEW RETURN BOUNDS', len(sources), len(needed), len(rows),
          len(result['unexcluded_candidates']), flush=True)
    for summary in summaries:
        if summary['window_pairs']:
            print(summary['target_family'], summary['source_period_multiple'],
                  summary['target_period_multiple'], summary['window_pairs'],
                  summary['minimum_normalized_lower_bound'], flush=True)


if __name__ == '__main__':
    main()

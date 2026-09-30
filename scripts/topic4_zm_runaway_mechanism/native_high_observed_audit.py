"""Check connected high-equilibrium coverage and selected observed-path roots.

Selection is by nearest D on the completed continuation chain, before any
spectrum is calculated. Positive roots establish instability only; failure to
find one does not establish stability or completeness of the spectrum.
"""
from native_path import *
from equilibrium_spectrum import cache_characteristic
from root_count_v3 import refine_root


def same_state(a, b):
    x, y = np.load(a), np.load(b)
    return bool(float(x['D']) == float(y['D']) and
                np.array_equal(x['r'], y['r']) and np.array_equal(x['Z'], y['Z']))


def main():
    folder = OUT/'equilibria'
    labels = ['native_high_descent', 'native_high_continued01',
              'native_high_continued02', 'native_high_continued03',
              'native_high_continued04', 'native_high_continued05',
              'native_high_continued07']
    out = folder/'native_high_observed_audit'
    out.mkdir(exist_ok=True)
    chain, allrows, joins = [], [], []
    for i, label in enumerate(labels):
        p = folder/label
        data, contract = read(p/'result.json'), read(p/'contract.json')
        assert data['status'] != 'RUNNING', (label, data['status'])
        first, last = data['rows'][0], data['rows'][-1]
        if i:
            assert same_state(contract['resume'], first['path'])
            previous = chain[-1]['last_source']
            if not same_state(previous, first['path']):
                # Separate audited corners provide the only inter-run gaps.
                audit = read(Path(contract['resume']).parent/'result.json')
                assert audit['status'] == 'JOIN_CONTINUITY_AND_ONE_SIDED_CHECK_PASS'
                assert same_state(previous, audit['source'])
                assert same_state(contract['resume'], audit['continuation_seed'])
                joins.append(str(Path(contract['resume']).parent/'result.json'))
        for join in data.get('path_joins', []):
            audit = read(join['audit'])
            assert audit['status'] == 'JOIN_CONTINUITY_AND_ONE_SIDED_CHECK_PASS'
            index = join['after_index']
            assert same_state(data['rows'][index]['path'], audit['source'])
            assert same_state(data['rows'][index+1]['path'], audit['continuation_seed'])
            joins.append(join['audit'])
        chain.append(dict(label=label, status=data['status'], n_rows=len(data['rows']),
            D_min=min(x['D'] for x in data['rows']), D_max=max(x['D'] for x in data['rows']),
            first_source=first['path'], last_source=last['path'],
            turn_candidates=len(data['turn_brackets']),
            max_recorded_residual_hz=max(x['residual_hz'] for x in data['rows'])))
        allrows.extend(data['rows'])
    targets = [.21, .21933565218, .228844760565, .256344425345, .30]
    selected = [min(allrows, key=lambda r: abs(r['D']-D)) for D in targets]
    contract = dict(status='REGISTERED_BEFORE_SPECTRUM', chain=chain,
        audited_joins=joins, target_D=targets, selected_sources=[r['path'] for r in selected],
        guesses_per_ms=[[.005, w] for w in [0., .02, .05, .1, .2, .4]],
        scope='Coverage of one connected conditional equilibrium family, not all equilibria. No stable label unless independently certified.')
    write(out/'contract.json', contract)
    rows = []
    for k, (target, row) in enumerate(zip(targets, selected)):
        z = np.load(row['path']); r = z['r']
        s = model(); attach_native_path(s); s.set_D(float(z['D']))
        assert np.max(abs(s.Z-z['Z'])) < 1e-12
        residual = float(abs(s.residual(r)).max()*1000)
        assert residual < 2e-8 and np.all(r >= 0) and np.all(r < 1/s.ref)
        checks = cache_characteristic(s, r)
        roots, attempts = [], []
        for guess in contract['guesses_per_ms']:
            lam = complex(*guess)
            try:
                ans = refine_root(s, r, lam, tol=1e-10)
            except Exception as exc:
                attempts.append(dict(guess=guess, status='FAILED', error=repr(exc)))
                continue
            if ans is None:
                attempts.append(dict(guess=guess, status='NOT_CONVERGED'))
                continue
            value, v, error = ans
            attempts.append(dict(guess=guess, status='CONVERGED',
                                 lambda_per_ms=[value.real, value.imag], residual=error))
            if any(abs(value-complex(*q['lambda_per_ms'])) < 1e-7 for q in roots):
                continue
            energy = s.E*s.sizes*abs(v)**2; energy /= energy.sum()
            field = np.bincount(s.geo['group_cell'], weights=energy, minlength=400)
            cell = int(np.argmax(field))
            q = dict(lambda_per_ms=[value.real, value.imag], residual=error,
                frequency_hz=abs(value.imag)*1000/(2*np.pi),
                mode_energy_A_B_surround=[float(energy[s.geo['group_region'] == j].sum()) for j in range(3)],
                maximal_E_energy_cell_mm=[cell % 20+.5, cell//20+.5])
            np.savez_compressed(out/f'point{k:02d}_root{len(roots):02d}.npz',
                               r=r, Z=s.Z, D=s.D, v=v, lambda_per_ms=value)
            roots.append(q)
        result = dict(target_D=target, source=row['path'], D=s.D, global_Z=1-s.D,
            global_E_hz=s.global_rate(r), regional_hz=s.regional_rates(r),
            equilibrium_residual_hz=residual, roots=roots, attempts=attempts,
            characteristic_cache_checks=checks,
            status='UNSTABLE_BY_POSITIVE_ROOT' if any(q['lambda_per_ms'][0] > 1e-7 for q in roots) else 'STABILITY_NOT_ESTABLISHED')
        rows.append(result)
        write(out/'result.json', dict(status='RUNNING', chain=chain, audited_joins=joins, rows=rows))
        log('OBSERVED HIGH EQUILIBRIUM', result)
    write(out/'result.json', dict(status='COMPLETE', chain=chain, audited_joins=joins, rows=rows,
        Z='held native spatial path', M='dynamic',
        scope='All selected equilibria have physical rates and the frozen equations. Roots prove only sampled-point instability; neither Hopf crossings, total unstable counts nor branch completeness follow.'))


if __name__ == '__main__':
    main()

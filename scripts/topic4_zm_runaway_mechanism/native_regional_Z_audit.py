"""Read native regional interventions independently with unchanged criteria."""
from native_late_Z_clamp_audit import N, R, OUT, PAIR, np, innovations, describe
import csv

DEST = OUT / 'native_regional_Z_feedback'


def main():
    c = N.read(OUT / 'native_regional_Z_feedback_contract.json')
    assert c['arms'] == ['cores_dynamic', 'surround_dynamic']
    assert N.read(DEST / 'application_check.json')['status'] == 'PASS'
    assert N.read(PAIR / 'dynamic_replay_qa.json')['status'] == 'PASS'
    N.check_reference_sources()
    geo = dict(np.load(DEST / 'geometry.npz'))
    original_geo = dict(np.load(PAIR / 'geometry.npz'))
    for key in ['cell_e_counts', 'region_counts', 'centers_mm']:
        assert np.array_equal(geo[key], original_geo[key])
    ref = PAIR / 'runs/native_t9000_Zdynamic'
    ref_xi = innovations(ref)
    initial = N.replay_checkpoint(9000)
    ref_final = N.load_pickle(ref / 'checkpoint.pkl')['engine']
    ref_origin = N.read(ref / 'continuation.json')
    checks_keys = ['rng_state', 'external_drive', 'xi']
    arms = [('all_dynamic', PAIR, 'native_t9000_Zdynamic'),
            ('all_held', PAIR, 'native_t9000_Zheld'),
            ('cores_dynamic', DEST, 'native_t9000_cores_dynamic'),
            ('surround_dynamic', DEST, 'native_t9000_surround_dynamic')]
    arrays = {}; rows = []; table = []
    for label, parent, name in arms:
        folder = parent / 'runs' / name
        assert N.read(folder / 'result.json')['status'] == 'COMPLETE'
        applied = N.read(folder / 'applied_configuration.json')
        origin = N.read(folder / 'continuation.json')
        assert not applied['frozen_M_state_update'] and applied['M_effective_feedback']
        assert origin['initial_state_bitwise_identical']
        assert not origin['clock_rebased'] and not origin['random_streams_replaced']
        assert origin['source_sha256'] == ref_origin['source_sha256']
        final = N.load_pickle(folder / 'checkpoint.pkl')['engine']
        assert np.isfinite(final['slow']['z']).all() and np.all((final['slow']['z'] >= 0) & (final['slow']['z'] <= 1))
        assert np.all(final['slow']['z'][N.NE:] == 1.) and np.all(final['slow']['m'][N.NE:] == 0.)
        assert not np.array_equal(final['slow']['m'], initial['slow']['m'])
        checks = dict(same_future_xi=np.array_equal(innovations(folder), ref_xi),
                      input_state_difference_keys=N.compare_states(
                          {k: final[k] for k in checks_keys}, {k: ref_final[k] for k in checks_keys}))
        assert checks['same_future_xi'] and checks['input_state_difference_keys'] == []
        if parent == DEST:
            applied_region = N.read(folder / 'regional_application.json')
            assert applied_region['status'] == 'PASS' and applied_region['steps_this_execution'] == 35000
            assert applied_region['held_Z_exact_every_step'] and applied_region['unheld_E_Z_changed']
            assert N.read(folder / 'geometry_check.json')['status'] == 'PASS'
            masks = np.load(folder / 'regional_masks.npz'); held = masks['held_Z']; region = masks['E_region']
            assert np.array_equal(final['slow']['z'][held], initial['slow']['z'][held])
            assert np.array_equal(np.bincount(region, minlength=3), geo['region_counts'][:3])
            checks['held_Z_exact'] = True
        else:
            maskfile = DEST / 'runs/native_t9000_cores_dynamic/regional_masks.npz'
            region = np.load(maskfile)['E_region']
        d = R.load_chunks(folder, keys=('spikes_1ms', 'field_1ms', 'regions_1ms'))
        assert np.array_equal(d['regions_1ms'][:, :3].sum(1), d['spikes_1ms'][:, 0])
        observed, field = describe(d, 9000, geo['cell_e_counts'])
        regional = d['regions_1ms'][:, :3] / geo['region_counts'][:3] * 1000
        region_stats = []
        for k in range(3):
            m = region == k
            region_stats.append(dict(region=['Core A', 'Core B', 'Surround'][k], cells=int(m.sum()),
                Z_initial=float(initial['slow']['z'][:N.NE][m].mean()),
                Z_final=float(final['slow']['z'][:N.NE][m].mean()),
                tail_rate_Hz=float(regional[2500:, k].mean()),
                M_feedback_final_mv=float(.0005 * final['slow']['m'][:N.NE][m].mean())))
        row = dict(arm=label, source=str(folder), checks=checks, regional=region_stats,
                   global_Z_final=float(final['slow']['z'][:N.NE].mean()), **observed)
        rows.append(row)
        arrays[label + '_field_Hz'] = field.astype(np.float32)
        arrays[label + '_regional_Hz'] = regional.astype(np.float32)
        summary = dict(arm=label, high_entry_s=row['high_entry_s'], broad_entry_s=row['broad_entry_s'],
                       complete_events=len(row['complete_events']), tail_mean_Hz=row['tail']['mean_rate_hz'],
                       tail_persistent_fraction=row['tail']['persistent_fraction']['50Hz_duty80'],
                       global_Z_final=row['global_Z_final'])
        table.append(summary); print(summary, flush=True)
    N.write(DEST / 'result.json', dict(status='COMPLETE', rows=rows, same_initial_state_and_future_input=True,
            application_and_geometry_checks='PASS', contract=str(OUT / 'native_regional_Z_feedback_contract.json'),
            statistical_unit=c['scope'], bifurcation_type='NOT_INFERRED', M='Dynamic in all four conditions.'))
    np.savez_compressed(DEST / 'fields.npz', **arrays, cell_counts=geo['cell_e_counts'], centers_mm=geo['centers_mm'])
    with (DEST / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys()); writer.writeheader(); writer.writerows(table)


if __name__ == '__main__': main()

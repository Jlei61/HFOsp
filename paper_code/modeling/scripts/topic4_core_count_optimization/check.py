"""Scientific execution gates: original dynamics, added core drives, frozen loss."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import pickle
from modeling.scripts.topic4_core_count_optimization.common import *
from modeling.scripts.topic4_core_count_optimization.physics import CoreCountOU, CoreOUMixture, core_field, v2
REFERENCE = Path(str(INPUT_ROOT / 'modeling/topic4_sef_hfo/label_free_dense_parameter_search_20260915/search/units/dense_axis3_12/2511_847401/workers/trajectory.npz'))

def unit_checks():
    args = (1.0, 0.95, 0.0, 0.1, 20.0, 3.0, 847401)
    old = CoreOUMixture(np.array([0, 0, 1, -1]), *args)
    two = CoreCountOU(np.array([0, 0, 1, -1]), *args)
    one = CoreCountOU(np.array([0, 0, -1]), *args)
    three = CoreCountOU(np.array([0, 0, 1, -1, 2]), *args)
    four = CoreCountOU(np.array([0, 0, 1, -1, 2, 3]), *args)
    third, fourth = ([], [])
    for i in range(1000):
        a = old.step(i * 0.1).copy()
        b = two.step(i * 0.1)
        c = one.step(i * 0.1)
        d = three.step(i * 0.1)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:2], c[:2])
        np.testing.assert_array_equal(a, d[:4])
        third.append(d[4])
        assert c[-1] == d[3] == 0.0
        e = four.step(i * 0.1)
        np.testing.assert_array_equal(d, e[:5])
        fourth.append(e[5])
    assert np.std(third) > 0 and (not np.array_equal(third, np.zeros(1000)))
    assert np.std(fourth) > 0 and (not np.array_equal(third, fourth))
    pos = np.array([[3.0, 3.0], [10.0, 10.0], [17.0, 17.0], [1.0, 19.0]])
    f = core_field(pos, pos, pos[:3], [1.0] * 3, [1.0] * 3, n_total=8, quantile_seed=11, core_mean=14.0, core_std=0.01, v_base=18.0, sheet_mm=20.0)
    np.testing.assert_array_equal(f['core_index'], [0, 1, 2, -1])
    assert f['vtheta'][2] < 18 and f['vtheta'][3] == 18
    masks = v2._block_masks(np.array([2, 3, 1]), np.array([2, 2, 2]), 4, f['core_index'], f['i_core_index'], 'ampa')
    np.testing.assert_array_equal(masks['EE_same_core_scale'], [True, False, False])
    np.testing.assert_array_equal(masks['EE_core_to_out_scale'], [False, True, False])
    pos4 = np.array([[3.0, 3.0], [10.0, 3.0], [3.0, 10.0], [10.0, 10.0], [19.0, 19.0]])
    f4 = core_field(pos4, pos4, pos4[:4], [1.0] * 4, [1.0] * 4, n_total=10, quantile_seed=11, core_mean=14.0, core_std=0.01, v_base=18.0, sheet_mm=20.0)
    np.testing.assert_array_equal(f4['core_index'], [0, 1, 2, 3, -1])
    assert np.all(f4['vtheta'][:4] < 18.0) and np.all(f4['vtheta'][4:] == 18.0)
    masks4 = v2._block_masks(np.array([3, 4, 0]), np.array([3, 3, 3]), 5, f4['core_index'], f4['i_core_index'], 'ampa')
    np.testing.assert_array_equal(masks4['EE_same_core_scale'], [True, False, False])
    np.testing.assert_array_equal(masks4['EE_core_to_out_scale'], [False, True, False])
    with (OUT / 'objective.pkl').open('rb') as stream:
        obj = pickle.load(stream)
    r = read(REFERENCE.with_suffix('.json'))
    with np.load(REFERENCE) as z:
        ix = [i for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0] >= 1500 and r['events'][i]['window_ms'][1] <= r['actual_duration_ms']]
        assert z['contact_names'].tolist() == obj.names
        score = obj.score(z['centroid_ms'][ix])
    oldscore = read(REFERENCE.parents[5] / 'scores/dense_axis3_12.json')
    np.testing.assert_allclose(score['J'], oldscore['J'], rtol=0, atol=1e-09)
    for g in GROUPS:
        np.testing.assert_allclose(score['groups'][g]['D_off'], oldscore['groups'][g]['D_off'], rtol=0, atol=1e-09)
    write(OUT / 'unit_validation.json', dict(status='PASS', original_two_core_OU_exact=True, core_streams_preserved=True, third_core_threshold_and_pathways=True, fourth_core_threshold_and_pathways=True, fourth_core_OU_independent_existing_streams_exact=True, historical_J_reproduced=score['J'], objective_sha256=sha(OUT / 'objective.pkl')))

def canary_checks(cores=None):
    verify_sources()
    unit_checks()
    records = []
    plan = read(OUT / 'plan.json')
    cores = plan['core_counts'] if cores is None else cores
    assert set(cores) <= set(plan['core_counts'])
    for k in cores:
        u = OUT / 'canary' / f'core{k}'
        a = read(u / 'applied_physics.json')
        r = read(u / 'result.json')
        assert r['physical_status'] == 'COMPLETE_NO_RUNAWAY' and r['actual_duration_ms'] == read(OUT / 'plan.json')['canary_ms']
        assert a['threshold']['n_raised'] == 0 and len(a['threshold']['members']) == k and (min(a['threshold']['members']) > 0)
        assert len(r['input_audit']['maximum_core_rate_error']) == k and max(r['input_audit']['maximum_core_rate_error'].values()) < 1e-12
        assert r['input_audit']['maximum_outside_rate_error'] == r['input_audit']['maximum_I_rate_error'] == 0
        with np.load(u / 'trajectory.npz') as z:
            assert set(np.unique(z['core_index_E'])) == set(range(k)) | {-1}
            if k == 2:
                with np.load(REFERENCE) as old:
                    for name in ('sheet_activity_counts', 'trace_E_spikes', 'trace_I_spikes', 'trace_coreAE_mean_V', 'trace_coreBE_mean_V'):
                        np.testing.assert_array_equal(z[name], old[name][:len(z[name])])
            if k >= 3:
                for j in range(2, k):
                    letter = chr(65 + j)
                    assert len(z[f'group_core{letter}E']) > 0 and f'trace_core{letter}E_spikes' in z
                    assert any((f'E{letter}' in name for name in a['graph']['block_summary']))
        records.append(dict(core_count=k, actual_duration_ms=r['actual_duration_ms'], counts=a['threshold']['members'], wall_seconds=r['wall_seconds'], peak_rss_gib=r['peak_rss_gib']))
    old = read(OUT / 'canary_validation.json') if (OUT / 'canary_validation.json').exists() else {}
    previous = [r for r in old.get('checks', []) if r['core_count'] not in cores] if old.get('status') == 'PASS' else []
    merged = sorted(previous + records, key=lambda r: r['core_count'])
    write(OUT / 'canary_validation.json', dict(status='PASS', checks=merged, historical_two_core_1000ms_prefix_exact=any((r['core_count'] == 2 for r in merged)), conclusion='Physical implementation gate only; no loss or propagation acceptance.', updated_unix=time.time()))
    for record in records:
        write(OUT / 'canary_validation' / f"core{record['core_count']}.json", dict(status='PASS', check=record, updated_unix=time.time()))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--canaries', action='store_true')
    p.add_argument('--cores', nargs='+', type=int)
    args = p.parse_args()
    canary_checks(args.cores) if args.canaries else unit_checks()

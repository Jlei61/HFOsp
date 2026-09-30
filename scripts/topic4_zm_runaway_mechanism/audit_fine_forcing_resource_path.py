"""Compare spatial slow states without replacing the original A4 gates.

The question is whether similar scalar D coordinates hide different core and
surround resources. Actual saved states are compared on the common clock;
the two samples around each model's own high-rate entry are descriptive only.
"""
from common import OUT, BASE, ROOT, model, np, read, write, log
import argparse


DEST = OUT / 'conditioned_refractory_fine_forcing'
PARENT = OUT / 'conditioned_refractory_spatial_resolution'
NATIVE = ROOT / 'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints'
TIMES = [8000, 9000, 9420, 9870, 10370]


def main(partial=False):
    jobs = read(DEST / 'jobs.json')
    if not partial:
        assert jobs['status'] == 'COMPLETE' and len(jobs['completed']) == 2
    s = model(40)
    members = s.geo['cell_group'][:32000]
    counts = np.bincount(members, minlength=s.P)
    assert np.array_equal(counts[s.E], s.sizes[s.E])
    native = {}
    reference = read(BASE / 'native_reference/checkpoint_projections.json')

    def project(x):
        return np.bincount(members, weights=x, minlength=s.P) / np.maximum(counts, 1)

    def describe(label, tm, z, m, target=None):
        z = z.astype(float); m = m.astype(float)
        assert np.isfinite(z).all() and np.isfinite(m).all()
        assert z[s.E].min() >= 0 and z[s.E].max() <= 1
        regions = []
        for reg, name in enumerate(['Core A', 'Core B', 'Surround']):
            mask = s.E & (s.geo['group_region'] == reg)
            regions.append(dict(region=name, neurons=int(s.sizes[mask].sum()),
                Z=float(np.average(z[mask], weights=s.sizes[mask])),
                M_current_mV=float(np.average(m[mask], weights=s.sizes[mask]))))
        row = dict(label=label, time_ms=float(tm), D=float(1-z[s.E]@s.mean_weights), regions=regions)
        if target is not None:
            zn, mn = target
            row.update(Z_field_RMS=float(np.sqrt(((z[s.E]-zn[s.E])**2)@s.mean_weights)),
                Z_field_mean_bias=float((z[s.E]-zn[s.E])@s.mean_weights),
                M_field_RMS_mV=float(np.sqrt(((m[s.E]-mn[s.E])**2)@s.mean_weights)))
        return row

    rows = []
    for tm in TIMES:
        with np.load(NATIVE / f't{tm}ms.npz') as state:
            zn = project(state['slow__z'][:32000]); zn[~s.E] = 1
            mn = .0005*project(state['slow__m'][:32000])
            assert abs(zn[s.E]@s.mean_weights-state['slow__z'][:32000].mean(dtype=float)) < 1e-12
        assert abs(1-zn[s.E]@s.mean_weights-reference[str(tm)]['D']) < 1e-12
        native[tm] = (zn, mn)
        rows.append(describe('native', tm, zn, mn))

    labels = ['recorded_drive_expected', 'recorded_drive_binomial_seed1']
    sources = [(label+'_parent', PARENT/label) for label in labels]
    sources += [(label+'_fine_forcing', DEST/label) for label in jobs['completed']]
    entries = []
    for label, folder in sources:
        with np.load(folder/'trajectory.npz') as data:
            ts = data['state_time_ms']; z = data['Z']; m = data['M_current']
            for tm in TIMES:
                where = np.flatnonzero(ts == tm); assert len(where) == 1
                k = int(where[0]); row = describe(label, tm, z[k], m[k], native[tm])
                assert abs(row['D']-data['D'][k]) < 6e-8  # saved Z is float32; D is float64
                rows.append(row)
            entry = read(folder/'result.json')['high_onset_ms']
            if entry is not None:
                k = int(np.searchsorted(ts, entry)); assert 0 < k < len(ts)
                assert ts[k-1] <= entry <= ts[k]
                for j in [k-1, k]:
                    row = describe(label, ts[j], z[j], m[j], native[9870])
                    row.update(high_entry_ms=entry, reference_clock_ms=9870,
                        meaning='Saved bracket around own operational entry; not common-clock acceptance or a bifurcation point.')
                    entries.append(row)
    name = 'resource_path_partial.json' if partial else 'resource_path_comparison.json'
    result = dict(status='PARTIAL_RESOURCE_PATH_AUDIT_COMPLETE' if partial else 'RESOURCE_PATH_AUDIT_COMPLETE',
        completed_new_runs=jobs['completed'], common_clock_ms=TIMES, rows=rows, own_entry_brackets=entries,
        observable='Cell-weighted Z and eta_M*M by actual physical core membership; group-field RMS at the same clock.',
        gates_unchanged=True, model_promoted=False, bifurcation_type='NOT_ESTABLISHED',
        scope='One original history. Spatial slow-state diagnostic only; no invented scalar critical Z or alignment-based acceptance.')
    write(DEST/name, result)
    log('RESOURCE PATH', [(r['label'], r['D'], [x['Z'] for x in r['regions']], r.get('Z_field_RMS')) for r in rows if r['time_ms']==9870])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--partial', action='store_true')
    main(p.parse_args().partial)

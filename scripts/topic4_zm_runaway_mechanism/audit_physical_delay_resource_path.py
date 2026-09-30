"""Compare physical core resources and the slow-law budget on the same clock.

The interval target is inferred from the exact linear Z equation. It is an
exponentially weighted time average, not a new firing-rate or onset threshold.
No simulation, fitting or acceptance criteria are changed here.
"""
from common import OUT, BASE, model, np, read, write, log
from physical_delay_count_rate import DEST, BASELINE
from audit_fine_forcing_resource_path import NATIVE


TIMES = [0, 8000, 9000, 9420, 9870, 10370, 12500]


def main():
    assert read(DEST/'jobs.json')['status'] == 'COMPLETE'
    s = model(40)
    members = s.geo['cell_group'][:32000]
    counts = np.bincount(members, minlength=s.P)
    assert np.array_equal(counts[s.E], s.sizes[s.E])

    def project(x):
        return np.bincount(members, weights=x, minlength=s.P)/np.maximum(counts, 1)

    masks = [('All E', s.E)] + [(name, s.E & (s.geo['group_region'] == k))
        for k, name in enumerate(['Core A', 'Core B', 'Surround'])]
    states = {}; rows = []; budgets = []; own_entries = []
    for label, folder in [('native', None), ('legacy_delay_split', BASELINE),
                          ('physical_delay_split', DEST/'recorded_drive_binomial_seed1')]:
        state = {0: (np.ones(s.P), np.zeros(s.P))}
        if folder is None:
            for tm in TIMES[1:]:
                with np.load(NATIVE/f't{tm}ms.npz') as data:
                    z = project(data['slow__z'][:32000]); z[~s.E] = 1
                    m = .0005*project(data['slow__m'][:32000])
                    assert abs(np.average(z[s.E], weights=s.sizes[s.E]) -
                               data['slow__z'][:32000].mean(dtype=float)) < 1e-12
                    state[tm] = z, m
        else:
            with np.load(folder/'trajectory.npz') as data:
                zall = data['Z'].astype(float); mall = data['M_current'].astype(float)
                for tm in TIMES[1:]:
                    where = np.flatnonzero(data['state_time_ms'] == tm)
                    assert len(where) == 1
                    i = where[0]; state[tm] = zall[i], mall[i]
                    assert abs(1-zall[i, s.E]@s.mean_weights-data['D'][i]) < 6e-8
                entry=read(folder/'result.json')['high_onset_ms']
                if entry is not None:
                    k=int(np.searchsorted(data['state_time_ms'],entry)); assert 0<k<len(zall)
                    zn,mn=states['native'][9870]
                    for j in [k-1,k]:
                        for name,mask in masks:
                            weights=s.sizes[mask]/s.sizes[mask].sum()
                            own_entries.append(dict(label=label,region=name,
                                state_time_ms=float(data['state_time_ms'][j]),high_entry_ms=float(entry),
                                Z=float(zall[j,mask]@weights),M_current_mV=float(mall[j,mask]@weights),
                                native9870_Z_RMS=float(np.sqrt((zall[j,mask]-zn[mask])**2@weights)),
                                reference_clock_ms=9870,
                                scope='Own operational high-rate entry bracket; not same-clock acceptance or a bifurcation coordinate.'))
        states[label] = state
        for tm in TIMES:
            z, m = state[tm]
            for name, mask in masks:
                weights = s.sizes[mask]/s.sizes[mask].sum()
                zn, mn = states['native'][tm]
                rows.append(dict(label=label, time_ms=tm, region=name,
                    neurons=int(s.sizes[mask].sum()), Z=float(z[mask]@weights),
                    D=float(1-z[mask]@weights), M_current_mV=float(m[mask]@weights),
                    native_Z_RMS=float(np.sqrt((z[mask]-zn[mask])**2@weights)),
                    native_M_RMS_mV=float(np.sqrt((m[mask]-mn[mask])**2@weights))))
        for a, b in zip(TIMES[:-1], TIMES[1:]):
            # Native replay uses Euler at 0.1 ms; the rate engine uses exp.
            decay = (np.exp(np.log1p(-.1/5000.)*((b-a)/.1)) if label == 'native'
                     else np.exp(-(b-a)/5000.))
            d0 = 1-state[a][0]; d1 = 1-state[b][0]
            # dD/dt = (q-D)/tau_Z, q=P(unscaled GABA >= native threshold).
            q = (d1-decay*d0)/(1-decay)
            assert q[s.E].min() >= -3e-6 and q[s.E].max() <= 1+3e-6
            for name, mask in masks:
                weights = s.sizes[mask]/s.sizes[mask].sum()
                qbar = float(q[mask]@weights)
                reconstruction = decay*float(d0[mask]@weights)+(1-decay)*qbar
                assert abs(reconstruction-float(d1[mask]@weights)) < 1e-12
                budgets.append(dict(label=label, region=name, interval_ms=[a,b],
                    exponentially_weighted_depletion_target=qbar,
                    retained_initial_fraction=float(decay),
                    D_change=float((d1[mask]-d0[mask])@weights),
                    target_min=float(q[mask].min()), target_max=float(q[mask].max())))
    nativeD = read(BASE/'native_reference/checkpoint_projections.json')['9870']['D']
    assert abs(next(r['D'] for r in rows if r['label']=='native' and
                   r['time_ms']==9870 and r['region']=='All E')-nativeD) < 1e-12
    write(DEST/'resource_path_comparison.json', dict(status='RESOURCE_PATH_AUDIT_PASS',
        times_ms=TIMES, rows=rows, slow_budget=budgets,own_entry_brackets=own_entries,
        equation='dD/dt=(q-D)/5000ms; q is the probability of unscaled GABA current >= the native threshold.',
        budget_definition='qbar=(D(t1)-w*D(t0))/(1-w); w=(1-0.1/5000)^((t1-t0)/0.1) for native Euler, exp(-(t1-t0)/5000) for rate. A normalized geometric/exponential target average, not an ordinary mean or independent current measurement.',
        scope='Single matched count realization; exact source clocks and physical core members. Values inferred from saved Z endpoints, so cannot distinguish erroneous current prediction from errors in target closure. Native target Gaussian audit is separate.',
        original_gates_unchanged=True, model_promoted=False, bifurcation_type='NOT_ESTABLISHED'))
    log('PHYSICAL DELAY RESOURCE PATH', [(r['label'],r['D'],r['native_Z_RMS'])
        for r in rows if r['time_ms']==9870 and r['region']=='All E'])


if __name__ == '__main__':
    main()

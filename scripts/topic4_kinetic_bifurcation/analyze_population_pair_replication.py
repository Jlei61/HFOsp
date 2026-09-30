"""Read the prespecified second-seed control, retaining each paired trajectory.

Events are nested observations, not independent replicates. This diagnoses a
real late-window correspondence discrepancy and makes no bifurcation claim.
"""
from compare_density_spatial import *


def main():
    folder = OUT / 'population_pair_replication'
    contract = json.load(open(folder / 'contract.json'))
    base = OUT / 'particle_controls/selected_g40'
    density = OUT / 'recurrence_searches/D225_5to15s'
    density_stats = summarize(density, 8000, 12000)
    density_spatial = extract(density, .225, 8000, 12000)
    rows = []
    comparisons = []
    for seed in (1901, contract['seed']):
        paired = {}
        for name, suffix in [('grouped', ''), ('individual', '_microscopic')]:
            source = base / f'D0.225000_Nscale16_seed{seed}_12000ms{suffix}'
            status = json.load(open(source / 'status.json'))
            assert status['status'] == 'COMPLETE', source
            cfg = json.load(open(source / 'config.json'))
            assert cfg['D'] == contract['D'] and cfg['population_multiplier'] == 16
            windows = [summarize(source, *w) for w in contract['windows_ms']]
            late = windows[-1]
            spatial = extract(source, .225, 8000, 12000) if late['finite_events'] else None
            row = dict(seed=seed, model=name, source=str(source.resolve()),
                windows=windows,
                late_spatial_vs_density=compare(density_spatial, spatial) if spatial else None,
                late_core_B_minus_A_crossing_ms=spatial['core_B_minus_A_crossing_ms'] if spatial else [],
                M_mean_identity_error=status['individual_and_population_M_mean_error'])
            rows.append(row)
            paired[name] = late
        g, i = paired['grouped'], paired['individual']
        comparisons.append(dict(seed=seed,
            grouped_late_category=g['category'], individual_late_category=i['category'],
            grouped_minus_individual_mean_rate_hz=g['mean_E_hz']-i['mean_E_hz'],
            grouped_minus_individual_quiet_fraction=g['quiet_fraction']-i['quiet_fraction'],
            grouped_complete_events=g['finite_events'], individual_complete_events=i['finite_events']))
    result = dict(status='PAIRED_REPLICATION_COMPLETE', contract=str((folder/'contract.json').resolve()),
        D=.225, rows=rows, paired_late_differences=comparisons,
        density_late_statistics=density_stats,
        statistical_unit='Two paired independent private-input realizations at one D and one population size; events are nested descriptors',
        interpretation='The late grouped-versus-individual discrepancy repeats across the two seeds. Finite-population/history effects and within-group closure remain unresolved; this is not an asymptotic-limit test.',
        model_correspondence_acceptance='NOT_ESTABLISHED for the density/native transition boundary',
        bifurcation_claim='None; the observed finite-window categories do not determine a critical point type')
    (folder/'result.json').write_text(json.dumps(safe(result), indent=2)+'\n')
    for row in comparisons:
        print(row)
    for row in rows:
        late=row['windows'][-1]
        print(row['seed'], row['model'], '8--12s mean Hz', late['mean_E_hz'],
              'quiet fraction', late['quiet_fraction'], 'events', late['finite_events'],
              'spatial', row['late_spatial_vs_density'])


if __name__ == '__main__':
    main()

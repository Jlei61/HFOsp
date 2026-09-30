"""Keep the complete resource-resolution comparison, including negative results."""
from compare_density_spatial import *
import csv


def main():
    base=OUT/'particle_controls/selected_g40';folder=OUT/'population_pair_replication'
    cases=[(1901,'parent_mean',''),(1901,'two_strata','_Zquadrature2'),
        (1901,'four_strata','_Zquadrature4'),(1901,'eight_strata','_Zquadrature8'),(1901,'exact_Z','_individual_flags2'),
        (1901,'all_individual','_microscopic'),(1902,'parent_mean',''),
        (1902,'four_strata','_Zquadrature4'),
        (1902,'eight_strata','_Zquadrature8'),
        (1902,'exact_Z','_individual_flags2'),(1902,'all_individual','_microscopic')]
    rows=[];pending=[];maps={}
    for seed,name,suffix in cases:
        source=base/f'D0.225000_Nscale16_seed{seed}_12000ms{suffix}'
        status=source/'status.json'
        if not status.exists() or json.load(open(status))['status']!='COMPLETE':
            pending.append(dict(seed=seed,condition=name,source=str(source)));continue
        windows=[summarize(source,*w) for w in ((1000,4000),(4000,8000),(8000,12000))]
        late=windows[-1];starts=np.array([e['start_ms'] for e in late['events']]);iei=np.diff(starts)
        temporal=dict(complete_intervals=len(iei),inter_event_intervals_ms=iei,
            inter_event_interval_cv=float(iei.std(ddof=1)/iei.mean()) if len(iei)>1 else None,
            interpretation='Within-trajectory description; insufficient by itself to classify periodicity or an event-generation mechanism')
        spatial=extract(source,.225,8000,12000) if late['finite_events'] else None
        if spatial is not None:maps[(seed,name)]=spatial
        rows.append(dict(seed=seed,condition=name,source=str(source.resolve()),windows=windows,
            late_temporal_description=temporal,
            late_B_minus_A_ms=spatial['core_B_minus_A_crossing_ms'] if spatial else []))
    comparisons=[]
    for key,spatial in maps.items():
        seed,name=key;ref=(seed,'exact_Z')
        if ref in maps and key!=ref:
            comparisons.append(dict(seed=seed,condition=name,reference='exact_Z',**compare(maps[ref],spatial)))
    between=compare(maps[(1901,'exact_Z')],maps[(1902,'exact_Z')]) if all((seed,'exact_Z') in maps for seed in (1901,1902)) else None
    result=dict(status='DECLARED_RESOURCE_COMPARISON_COMPLETE' if not pending else 'INTERIM_WAITING_FOR_DECLARED_RUNS',
        D=.225,Nscale=16,rows=rows,pending=pending,spatial_vs_paired_exact_Z=comparisons,
        between_exact_Z_inputs=between,
        statistical_unit='Paired private-input trajectories; event and pixel observations are nested',
        acceptance='Diagnostic of one resource-path parameter and the declared resolutions. All levels are retained. No full density/SNN equivalence or critical bifurcation type is accepted here.')
    (folder/'resource_resolution_comparison.json').write_text(json.dumps(safe(result),indent=2)+'\n')
    summary=[]
    for row in rows:
        late=row['windows'][-1]
        summary.append(dict(seed=row['seed'],condition=row['condition'],mean_E_hz=late['mean_E_hz'],
            quiet_fraction=late['quiet_fraction'],complete_events=late['finite_events'],
            event_duration_median_ms=late['event_duration_median_ms'],
            interval_cv=row['late_temporal_description']['inter_event_interval_cv']))
    with open(folder/'resource_resolution_late_summary.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(result['status'])
    for row in summary:print(row)


if __name__=='__main__':
    main()

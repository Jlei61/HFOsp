"""Unchanged propagation observer, native variability, and spatial diagnostics."""
from shared_source import *
import evaluate as previous
from analyze import dynamics,metric_distance,coverage
import csv,itertools

def summarize(name,z,key,positions,sizes):
    ob,ids,mu,q=observations(z[key]);space,onset=previous.spatial_events(z,ob,ids,positions,sizes)
    out=dict(name=name,N=len(ids),summary=q,coverage=coverage(q),event_ids=ids,centroid_ms=mu[ids],
        dynamics=dynamics(z['six_counts'],sizes),spatial_events=space,mean_spatial_onset_map=onset)
    write(OUT/'summaries'/f'{name}.json',out);write(OUT/'observations'/f'{name}.json',ob)
    return out

def main():
    zmodel=model();sizes=np.bincount(zmodel['region'],minlength=6);native={}
    for seed in (848101,848102,848103):
        native[seed]=read(BASE/'summaries'/f'native_s{seed}.json')
        write(OUT/'summaries'/f'native_s{seed}.json',native[seed])
    spread=np.max([metric_distance(native[a]['summary'],native[b]['summary']) for a,b in itertools.combinations(native,2)],axis=0)
    rows=[];detail={};comparisons=[]
    for folder in sorted((OUT/'runs').glob('*')):
        if not (folder/'result.json').exists():continue
        config=read(folder/'result.json');z=np.load(folder/'trajectory.npz');seed=config['seed']
        assert np.array_equal(z['nu_core'],np.load(PRIOR/f'native/{seed}/trajectory.npz')['nu_core'])
        readouts=[('contact_envelope','neuron'),('group_contact_envelope','population')]
        if 'common_group_contact_envelope' in z.files:readouts.append(('common_group_contact_envelope','coarse_population'))
        for key,readout in readouts:
            name=folder.name+'_'+readout;q=summarize(name,z,key,zmodel['positions'],sizes)
            errors=np.array([metric_distance(q['summary'],native[s]['summary']) for s in native]);complete=q['coverage']==dict(contacts=15,pairs={'SCL':6,'ICL':55})
            status='NO_VALID_EVENTS' if q['N']==0 else ('INSUFFICIENT_COVERAGE' if not complete else ('PROPAGATION_MISMATCH' if np.any(np.all(errors>spread[None,:],axis=0)) else 'NO_CLEAR_THREE_METRIC_FAILURE_NOT_ACCEPTED'))
            q.update(config=config,readout=readout,errors_to_native=errors,native_pair_max=spread,status=status,robust_exceedance=np.all(errors>spread[None,:],axis=0))
            write(OUT/'summaries'/f'{name}.json',q);detail[name]=q
            own=list(native).index(seed);d=q['dynamics'];contacts=q['summary']['contacts'];space=q['spatial_events']
            rows.append(dict(run=folder.name,seed=seed,readout=readout,N=q['N'],AE_mean_hz=d['AE']['mean_hz'],BE_mean_hz=d['BE']['mean_hz'],
                AE_low_fraction=d['AE']['low_rate_fraction'],BE_low_fraction=d['BE']['low_rate_fraction'],
                AE_interval_ms=d['AE']['median_peak_interval_ms'],BE_interval_ms=d['BE']['median_peak_interval_ms'],
                AE_interval_CV=d['AE']['peak_interval_CV'],BE_interval_CV=d['BE']['peak_interval_CV'],
                SCL9_participation=contacts['SCL9']['participation'],ICL10_participation=contacts['ICL10']['participation'],
                field_area_median=float(np.median([s['area_fraction'] for s in space])) if space else None,
                field_onset_span_median_ms=float(np.median([s['field_onset_span_ms'] for s in space])) if space else None,
                rank_error=errors[own,0],order_error=errors[own,1],participation_error=errors[own,2],status=status))
        a=detail[folder.name+'_neuron'];b=detail[folder.name+'_population']
        compare=dict(run=folder.name,neuron_vs_group_errors=metric_distance(a['summary'],b['summary']),N_neuron=a['N'],N_group=b['N'])
        if 'common_group_contact_envelope' in z.files:
            coarse=detail[folder.name+'_coarse_population'];compare['neuron_vs_common_coarse_errors']=metric_distance(a['summary'],coarse['summary'])
            compare['N_common_coarse']=coarse['N']
        comparisons.append(compare)
    if rows:
        with (OUT/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(safe(rows))
    write(OUT/'comparison.json',dict(rows=rows,native_pair_max=spread,readout_comparison=comparisons,
        status='EXPLORATORY_NOT_ACCEPTED',bifurcation_allowed=False,statistical_unit='one topology; noise repeats, events nested in trajectories',
        metric_provenance='Unchanged observer and shaft-balanced mean rank, conditional within-shaft order, participation errors from previous two rounds',
        spatial_provenance='Unchanged exploratory 1 mm / 2 ms E field, 5 ms Gaussian smoothing, first crossing above 20 Hz in valid observer event windows'))
    print(json.dumps(safe(rows),indent=2),flush=True)

if __name__=='__main__':main()

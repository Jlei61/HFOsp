"""Seed-level correspondence, with event and censoring summaries kept nested."""
from compare_density_spatial import *
from itertools import product


def paired_summary(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a;n=len(a)
    # Exact enumeration of the empirical paired bootstrap at n=6. These are
    # descriptive small-sample intervals, not an equivalence acceptance test.
    draws=np.asarray(list(product(range(n),repeat=n)),dtype=np.int16)
    means=delta[draws].mean(1)
    signs=np.asarray(list(product((-1.,1.),repeat=n)))
    return dict(individual_values=a,grouped_values=b,paired_grouped_minus_individual=delta,
        mean_paired_difference=float(delta.mean()),paired_bootstrap_percentile_95=np.quantile(means,[.025,.975]),
        individual_between_seed_SD=float(a.std(ddof=1)),grouped_between_seed_SD=float(b.std(ddof=1)),
        exact_sign_flip_p=float(np.mean(abs(signs@delta/n)>=abs(delta.mean())-1e-12)),
        inference='Descriptive paired seed-level uncertainty; no equivalence conclusion from a nonsignificant difference and no independent-event/pixel replication.')


def main():
    base=OUT/'particle_controls/selected_g40';folder=OUT/'population_pair_replication/target_N1_six_seed_extension'
    assert json.load(open(folder/'status.json'))['status']=='EXECUTION_COMPLETE'
    rows=[];event_maps={};mean_maps={}
    for seed in range(1901,1907):
        for name,suffix in [('grouped',''),('individual','_microscopic')]:
            source=base/f'D0.225000_Nscale1_seed{seed}_12000ms{suffix}'
            assert json.load(open(source/'status.json'))['status']=='COMPLETE'
            windows=[summarize(source,*w) for w in ((1000,4000),(4000,8000),(8000,12000))]
            with np.load(source/'trajectory.npz') as z:
                rates=z['rate_1ms'];field=z['field_1ms'];count=z['count_e']
            late=rates[8000:12000].reshape(-1,10,4).mean(1)
            intervals=stretches(late[:,0]>=5.)
            activity=[dict(start_ms=8000+s*10,end_ms=8000+t*10,duration_ms=(t-s)*10,
                           left_censored=s==0,right_censored=t==len(late)) for s,t in intervals]
            mean_field=(field[8000:12000]*count).reshape(-1,20,2,20,2).sum((2,4)).reshape(-1,400).mean(0)
            coarse_count=count.reshape(20,2,20,2).sum((1,3)).ravel()
            mean_field/=np.maximum(coarse_count,1)
            mean_maps[(seed,name)]=dict(image=mean_field,count=coarse_count)
            spatial=extract(source,.225,8000,12000) if windows[-1]['finite_events'] else None
            if spatial is not None:event_maps[(seed,name)]=spatial
            rows.append(dict(seed=seed,model=name,source=str(source.resolve()),windows=windows,
                late_active_segments_including_censored=activity,
                late_maximum_active_segment_ms=max([x['duration_ms'] for x in activity],default=0),
                late_B_minus_A_ms=spatial['core_B_minus_A_crossing_ms'] if spatial else [],
                spatial_event_selection='Complete events only; all-time mean field also compared separately to retain sustained/censored activity.'))
    metrics={}
    for k in ('mean_E_hz','quiet_fraction','finite_events'):
        values={name:[next(r for r in rows if r['seed']==seed and r['model']==name)['windows'][-1][k]
                      for seed in range(1901,1907)] for name in ('individual','grouped')}
        metrics[k]=paired_summary(values['individual'],values['grouped'])
    pairwise=[];between=[]
    for kind,maps in [('all_time_mean',mean_maps),('complete_event_median',event_maps)]:
        for seed in range(1901,1907):
            if all((seed,n) in maps for n in ('individual','grouped')):
                pairwise.append(dict(kind=kind,seed=seed,**compare(maps[(seed,'individual')],maps[(seed,'grouped')])))
        for s in range(1901,1907):
            for t in range(s+1,1907):
                if all((v,'individual') in maps for v in (s,t)):
                    between.append(dict(kind=kind,seeds=[s,t],**compare(maps[(s,'individual')],maps[(t,'individual')])))
    report=dict(status='SIX_PAIRED_TARGET_INPUTS_ANALYZED',D=.225,Nscale=1,rows=rows,
        seed_level_primary_metrics=metrics,paired_spatial=pairwise,between_individual_spatial=between,
        scope='Same selected-g40 communication, frozen Z, dynamic M and common OU at zero. It is not acceptance of the complete Fig5 stochastic trajectory or of the infinite-population density.',
        statistical_unit='Six independent paired private-input realizations. Events, pixels and pairwise spatial comparisons are dependent nested descriptions.',
        decision='Requires scientific review of all primary and propagation evidence; no automated bifurcation or model-equivalence acceptance.')
    (folder/'analysis.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    np.savez_compressed(folder/'spatial_arrays.npz',**{f'{kind}_{s}_{name}':v['image'] for kind,maps in [('all_time',mean_maps),('event',event_maps)] for (s,name),v in maps.items()})
    for k,v in metrics.items():print(k,json.dumps(safe(v)),flush=True)


if __name__=='__main__':main()

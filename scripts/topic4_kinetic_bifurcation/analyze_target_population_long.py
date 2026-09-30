"""Direct N1 target-size comparison, separated from the N16 limit diagnostic."""
from compare_density_spatial import *


def main():
    folder=OUT/'population_pair_replication';base=OUT/'particle_controls/selected_g40'
    rows=[];prefix=[];maps={};pairs=[]
    for seed in (1901,1902):
        pair={}
        for name,suffix in [('grouped',''),('individual','_microscopic')]:
            source=base/f'D0.225000_Nscale1_seed{seed}_12000ms{suffix}'
            short=base/f'D0.225000_Nscale1_seed{seed}_4000ms{suffix}'
            assert json.load(open(source/'status.json'))['status']=='COMPLETE',source
            with np.load(short/'trajectory.npz') as a,np.load(source/'trajectory.npz') as b:
                identical={};error={}
                for k in a.files:
                    v=b[k] if k=='count_e' else b[k][:len(a[k])]
                    identical[k]=bool(np.array_equal(a[k],v));error[k]=float(np.max(abs(a[k]-v)))
            passed=all(v for k,v in identical.items() if k!='field_1ms') and error['field_1ms']<1e-10
            assert passed,(seed,name,identical,error)
            prefix.append(dict(seed=seed,model=name,pass_replay=passed,arrays_bitwise=identical,maximum_absolute_errors=error))
            windows=[summarize(source,*w) for w in ((1000,4000),(4000,8000),(8000,12000))]
            late=windows[-1];pair[name]=late
            spatial=extract(source,.225,8000,12000) if late['finite_events'] else None
            if spatial is not None:maps[(seed,name)]=spatial
            rows.append(dict(seed=seed,model=name,source=str(source.resolve()),windows=windows,
                late_B_minus_A_ms=spatial['core_B_minus_A_crossing_ms'] if spatial else []))
        a,b=pair['individual'],pair['grouped']
        pairs.append(dict(seed=seed,grouped_minus_individual_mean_rate_hz=b['mean_E_hz']-a['mean_E_hz'],
            grouped_minus_individual_quiet_fraction=b['quiet_fraction']-a['quiet_fraction'],
            grouped_minus_individual_complete_events=b['finite_events']-a['finite_events']))
    spatial_pairs=[dict(seed=seed,**compare(maps[(seed,'individual')],maps[(seed,'grouped')]))
        for seed in (1901,1902) if all((seed,name) in maps for name in ('individual','grouped'))]
    between=compare(maps[(1901,'individual')],maps[(1902,'individual')]) if all((seed,'individual') in maps for seed in (1901,1902)) else None
    report=dict(status='DIRECT_TARGET_N1_LONG_COMPARISON_COMPLETE',D=.225,Nscale=1,rows=rows,
        prefix_replay=prefix,paired_late_differences=pairs,paired_spatial_comparisons=spatial_pairs,
        between_individual_inputs_spatial_comparison=between,
        statistical_unit='Two paired private-input realizations at the actual40000-neuron target size; events and pixels are nested.',
        scope='Same selected g40 communication and frozen resource path, with common OU zero. N16 and deterministic-density correspondence remain separate.',
        acceptance='Direct target-size evidence; not automatic full time-dependent Fig5, population-limit, numerical-mesh or bifurcation acceptance.')
    (folder/'target_population_long_result.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    for row in rows:
        x=row['windows'][-1];print(row['seed'],row['model'],x['mean_E_hz'],x['quiet_fraction'],x['finite_events'])
    print(pairs)


if __name__=='__main__':
    main()

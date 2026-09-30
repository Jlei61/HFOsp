"""Compare paired same-law Z assignments, without counting relabelings as seeds."""
from compare_density_spatial import *


def main():
    folder=OUT/'population_pair_replication';base=OUT/'particle_controls/selected_g40'
    assert json.load(open(folder/'resource_exchangeability_replay_qa.json'))['status']=='PASS'
    rows=[];pairs=[]
    for seed in (1901,1902):
        paired={}
        for name,suffix in [('original',''),('permuted','_Zperm884')]:
            source=base/f'D0.225000_Nscale16_seed{seed}_12000ms_individual_flags2{suffix}'
            assert json.load(open(source/'status.json'))['status']=='COMPLETE',source
            windows=[summarize(source,*w) for w in ((1000,4000),(4000,8000),(8000,12000))]
            late=windows[-1];paired[name]=late
            spatial=extract(source,.225,8000,12000) if late['finite_events'] else None
            rows.append(dict(seed=seed,assignment=name,source=str(source.resolve()),windows=windows,
                late_B_minus_A_ms=spatial['core_B_minus_A_crossing_ms'] if spatial else []))
        a,b=paired['original'],paired['permuted']
        pairs.append(dict(seed=seed,mean_rate_change_hz=b['mean_E_hz']-a['mean_E_hz'],
            quiet_fraction_change=b['quiet_fraction']-a['quiet_fraction'],
            complete_event_change=b['finite_events']-a['finite_events']))
    report=dict(status='PAIRED_EXCHANGEABILITY_DIAGNOSTIC_COMPLETE',D=.225,Nscale=16,rows=rows,pairs=pairs,
        law_invariance='Exact within-parent Z multisets, shared parent thresholds/M, mean communication and group/core/spatial-bin observables are unchanged; private inputs are iid within each parent.',
        statistical_unit='Two private-input realizations; original and permuted assignments are paired correlated observations, not four independent seeds.',
        interpretation='These observed same-law changes calibrate stochastic-history sensitivity. They neither establish absence of resource-resolution bias nor certify the density population limit.',
        bifurcation_acceptance='NOT_INFERRED')
    (folder/'resource_exchangeability_result.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    for row in rows:
        x=row['windows'][-1];print(row['seed'],row['assignment'],x['mean_E_hz'],x['quiet_fraction'],x['finite_events'])
    print(pairs)


if __name__=='__main__':
    main()

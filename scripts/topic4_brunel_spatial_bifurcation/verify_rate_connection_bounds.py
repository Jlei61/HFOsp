"""Check the unified finite-catalog bounds against actual phase comparisons."""
from bound_rate_period_multiple_encounters import DEST, controls, statistics
from complete_rate_positive_stability import RateField, np, read, write, distances
from audit_rate_survey_filter_states import fingerprint
from scipy.signal import resample
from pathlib import Path
import time


def main():
    source=DEST/'period_multiple_1_to_8_all_pair_bounds.json';q=read(source)
    assert q['status']=='ALL_FINITE_WINDOW_PAIRS_BOUNDED'
    assert q['multiples']==list(range(1,9))
    moments=read(q['moments_source'])['rows']
    for row in moments:assert fingerprint(row['orbit'])==row['profile_fingerprint']
    expected={(tuple([a,b]),m) for a in ['A','B'] for b in ['single','Bleading','double'] for m in range(1,9)}
    assert {(tuple(row['families']),row['period_multiple']) for row in q['rows']}==expected
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    checked=[]
    for row in q['rows']:
        if not row['window_pairs']:continue
        pair=row['weakest_bounds'][0];repeat=row['period_multiple']
        with np.load(pair['first_orbit']) as z:x=z['r']*1000
        with np.load(pair['second_orbit']) as z:y=z['r']*1000
        mx,sx,_=statistics(x,weights);my,sy,_=statistics(y,weights)
        lower=float(np.sqrt((mx-my)**2@weights+(sx-sy)**2))
        scale=max(sx,sy)
        assert abs(lower-pair['waveform_RMS_lower_bound_Hz'])<1e-9
        assert abs(scale-pair['normalization_RMS_Hz'])<1e-9
        # Upsample past both original bandwidths, including the repeated
        # short profile. These comparisons need no new BVP or spatial model.
        mesh=max(8192,2*repeat*len(x),2*len(y))
        rx=resample(np.tile(x,(repeat,1)),mesh,axis=0)
        ry=resample(y,mesh,axis=0)
        sx_grid=float(np.sqrt(np.mean((rx-rx.mean(0))**2,axis=0)@weights))
        sy_grid=float(np.sqrt(np.mean((ry-ry.mean(0))**2,axis=0)@weights))
        assert abs(sx-sx_grid)<1e-9 and abs(sy-sy_grid)<1e-9
        distance,phase=distances(rx[:,None,:],ry,weights)
        assert distance[0]>=lower-1e-7
        checked.append(dict(families=row['families'],period_multiple=repeat,
            first_orbit=pair['first_orbit'],second_orbit=pair['second_orbit'],
            phase_mesh=mesh,normalized_lower_bound=lower/scale,
            normalized_phase_minimized_distance=float(distance[0]/scale),
            common_phase_cycles=float(phase[0])))
        print('BOUND CONTROL',row['families'],repeat,lower/scale,distance[0]/scale,flush=True)
    total=sum(row['window_pairs'] for row in q['rows'])
    excluded=sum(row['excluded_pairs'] for row in q['rows'])
    unexcluded=[dict(families=row['families'],period_multiple=row['period_multiple'],
        candidates=row['unexcluded_candidates']) for row in q['rows'] if row['unexcluded_candidates']]
    result=dict(status='UNIFIED_FINITE_CATALOG_CONNECTION_SCREEN_CHECKED',timestamp=time.time(),
        source=str(source),catalog=q['catalog_source'],moments=q['moments_source'],
        full_population_profiles=len(moments),period_multiples=list(range(1,9)),
        window_pair_count=total,pairs_excluded_by_bound=excluded,
        minimum_normalized_lower_bound=min(row['minimum_normalized_lower_bound'] for row in q['rows'] if row['window_pairs']),
        candidate_threshold=q['candidate_threshold'],unexcluded_candidates=unexcluded,
        analytic_controls=controls(),weakest_pair_phase_controls=checked,
        scope='Every pairing in one frozen catalog within |delta J|<=0.01 and a 10 percent integer-period-ratio window was bounded in all 935 populations. The weakest-bound pair in each nonempty family/ratio window was also compared after a common phase shift at sufficient temporal resolution. These are cached waveform identity checks, not physical-profile validation of all catalog entries, identical-J continuation, basin boundary location, or proof of global disconnection.',
        interval_completeness=False,global_disconnection_proved=False)
    write(DEST/'period_multiple_1_to_8_connection_evidence.json',result)
    print('UNIFIED SCREEN',total,'excluded',excluded,'remaining groups',len(unexcluded),flush=True)


if __name__=='__main__':
    main()

"""Resolve the temporal bandwidth of a finite B-leading encounter shortlist.

These are cached-waveform comparisons, not an all-pair search, physical
validation of the target profiles, or an identical-J connecting-orbit solve.
"""
from bound_rate_period_multiple_encounters import DEST, statistics, controls
from complete_rate_positive_stability import RateField, np, read, write, distances
from audit_rate_survey_filter_states import fingerprint
from scipy.signal import resample
from pathlib import Path
import argparse
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--output-name',required=True)
    parser.add_argument('--nearest',type=int,default=3)
    a=parser.parse_args()
    assert Path(a.output_name).name==a.output_name and a.nearest>0
    q=read(a.source);assert q['status']=='SCREEN_COMPLETE'
    accepted=q['accepted_source_segment'];assert accepted['family']=='Bleading'
    model=RateField();assert model.P==935
    weights=model.geo['group_size']/model.geo['group_size'].sum()
    rows=[]
    for group in q['rows']:
        for pair in group['nearest'][:a.nearest]:
            paths=[Path(pair[k]) for k in ['first_orbit','second_orbit']]
            before=[fingerprint(p) for p in paths]
            profiles=[]
            for p,key in zip(paths,['first','second']):
                with np.load(p) as z:
                    assert z['r'].shape[1]==935
                    assert abs(float(z['J'])-pair[key+'_J'])<1e-12
                    assert abs(float(z['T'])-pair[key+'_T_ms'])<1e-8
                    profiles.append(z['r']*1000)
            x,y=profiles
            mx,sx,_=statistics(x,weights);my,sy,_=statistics(y,weights)
            scale=max(sx,sy);assert scale>0
            lower=float(np.sqrt((mx-my)**2@weights+(sx-sy)**2)/scale)
            N=max(8192,2*len(x),2*len(y))
            rx,ry=[resample(r,N,axis=0) for r in profiles]
            for r,expected in [(rx,sx),(ry,sy)]:
                measured=float(np.sqrt(np.mean((r-r.mean(0))**2,axis=0)@weights))
                assert abs(measured-expected)<1e-9
            distance,phase=distances(rx[:,None,:],ry,weights)
            fine=float(distance[0]/scale)
            assert fine>=lower-1e-8
            assert [fingerprint(p) for p in paths]==before
            row=dict(families=group['families'],pair=pair,
                source_profile_fingerprints=before,phase_mesh=N,
                normalized_mean_and_amplitude_lower_bound=lower,
                normalized_phase_aligned_distance=fine,
                change_from_original_screen=fine-pair['relative_waveform_difference'],
                common_phase_cycles=float(phase[0]))
            rows.append(row)
            print('FINE ENCOUNTER',group['families'],N,lower,fine,flush=True)
    result=dict(status='FINITE_SHORTLIST_TEMPORAL_CHECK_COMPLETE',timestamp=time.time(),
        source=str(a.source.resolve()),accepted_source_segment=accepted,
        nearest_pairs_per_family=a.nearest,rows=rows,analytic_controls=controls(),
        candidate_threshold=.05,candidates=[r for r in rows if r['normalized_phase_aligned_distance']<.05],
        global_connection_established=False,
        scope='Rechecks only the recorded nearest cached pairs in all 935 rate groups, preserving means and using one common phase on at least twice both temporal meshes. Targets remain at their original, nearby J; their physical validity is not certified here. A negative result excludes neither unseen pairs nor unsampled or untraced branch connections.')
    write(DEST/'Bleading_extension'/a.output_name,result)


if __name__=='__main__':main()

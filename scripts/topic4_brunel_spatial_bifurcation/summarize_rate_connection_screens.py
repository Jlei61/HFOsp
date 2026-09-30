"""Collect finite period-ratio screens with their different coverage scopes."""
from pathlib import Path
from complete_rate_positive_stability import read, write
import time


DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def main():
    sources=[DEST/'period_multiple_connection_screen.json',
             DEST/'period_multiple_5_7_connection_screen.json']
    bounds_source=DEST/'period_multiple_5_7_all_pair_bounds.json'
    bounds=read(bounds_source)
    assert bounds['status']=='ALL_FINITE_WINDOW_PAIRS_BOUNDED'
    lookup={(tuple(q['families']),q['period_multiple']):q for q in bounds['rows']}
    rows=[];seen=set()
    for source in sources:
        scan=read(source);assert scan['status']=='SCREEN_COMPLETE'
        assert scan['identity_phase_control_Hz']<1e-7
        for row in scan['rows']:
            key=tuple(row['families']),row['period_multiple']
            assert key not in seen;seen.add(key)
            record=dict(**row,screen_source=str(source),phase_samples=scan['phase_samples'],
                coverage='Selected phase-resolved pairs within each parameter/period window')
            if key in lookup:
                bound=lookup[key]
                assert bound['window_pairs']==row['window_pairs']
                assert not bound['unexcluded_candidates']
                assert bound['excluded_pairs']==bound['window_pairs']
                record.update(all_pair_bound_source=str(bounds_source),
                    minimum_normalized_distance_lower_bound=bound['minimum_normalized_lower_bound'],
                    excluded_pairs_by_bound=bound['excluded_pairs'],
                    coverage='Every finite window pair bounded; selected pairs also compared after a common phase shift')
            rows.append(record)
    expected={(tuple([a,b]),m) for a in ['A','B'] for b in ['single','Bleading','double'] for m in range(1,9)}
    assert seen==expected
    candidates=[dict(families=q['families'],period_multiple=q['period_multiple'],
        candidates=q['same_J_correction_candidates']) for q in rows if q['same_J_correction_candidates']]
    result=dict(status='FINITE_PERIOD_RATIOS_ONE_TO_EIGHT_SCREENED',timestamp=time.time(),
        rows=rows,sources=list(map(str,sources)),all_pair_bounds_source=str(bounds_source),
        all_pair_bound_count=sum(q['window_pairs'] for q in bounds['rows']),
        same_J_followup_candidates=candidates,interval_completeness=False,
        scope='Union of separate finite catalog screens. The 5- and 7-fold windows additionally have full-population phase-independent bounds for every candidate pair. Other ratios retain their original shortlist scope. No waveform coincidence is promoted to a global connection, and absence of candidates does not exclude unsampled global connections or bifurcations.')
    write(DEST/'period_multiple_connection_evidence_1_to_8.json',result)
    print('PERIOD-RATIO SCREENS',len(rows),'all-pair bounds',result['all_pair_bound_count'],
          'same-J follow-up groups',len(candidates),flush=True)


if __name__=='__main__':
    main()

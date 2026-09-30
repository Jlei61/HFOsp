"""Compare prespecified matched trajectories at two time steps.

Categorical agreement is reported directly. No post-hoc error tolerance,
infinite-time stability or bifurcation acceptance is invented here.
"""
from common import *
import csv


def main():
    assert read(OUT/'native_near_turn_matched_history_refinement.json')['status']=='PASS'
    sources=[OUT/f'native_near_turn_matched_audit_dt{dt}.json' for dt in [.05,.025]]
    rows=[]
    for source in sources:
        q=read(source);assert q['status']=='COMPLETE'
        for r in q['rows']:
            c=r['canonical'];peaks=r['spatial']['spatial_recurrence_peaks']
            rows.append(dict(dt_ms=q['dt_ms'],D=r['D'],global_Z=r['global_Z'],
                category=c['category'],tail_mean_hz=c['tail']['mean_rate_hz'],
                tail_quiet_fraction=c['tail']['quiet_fraction'],
                complete_events=r['complete_events_whole_30s'],
                max_event_ms=r['maximum_complete_event_ms'],
                best_recurrence_correlation=peaks[0]['correlation'] if peaks else None,
                best_recurrence_lag_ms=peaks[0]['lag_ms'] if peaks else None,
                best_recurrence_return_error=peaks[0]['normalized_return_error'] if peaks else None))
    low=[r for r in rows if r['D']==.21932];high=[r for r in rows if r['D']==.21934]
    assert len(low)==len(high)==2
    q=dict(status='DESCRIPTIVE_STEP_COMPARISON_COMPLETE',rows=rows,
        source_audits=[str(p) for p in sources],
        cross_step_categories_agree=all(a['category']==b['category'] for a,b in [low,high]),
        original_low_selflimited_high_unresolved_pattern_replicated=(
            all(r['category']=='SELF_LIMITED' for r in low) and
            all(r['category']=='UNRESOLVED' for r in high)),
        scope='Two matched finite30s conditional trajectories at each timestep. Full state and coincident physical delay-history samples checked. Categorical agreement is not quantitative event-distribution convergence or Floquet/bifurcation validation.',
        recurrence='Peak over5-to1000ms lags can be a multiple of the fundamental period; do not interpret the best lag itself as the period.')
    write(OUT/'native_near_turn_step_comparison.json',q)
    with (OUT/'native_near_turn_step_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    log('NEAR TURN TIME COMPARISON',q)


if __name__=='__main__':main()

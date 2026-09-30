"""Finite-time state audit along the two continuation histories.

This table is never used as a substitute for corrected invariant branches.
All rows retain the history, physical D, analysis window and map precision.
"""
from analyze_qualification import *
from scipy.signal import find_peaks
import csv


def main():
    cases=[('burst','recurrence_searches/D225_5to15s'),
        ('burst','recurrence_searches/D2375_burst_fp64_confirmation'),
        ('persistent','recurrence_searches/D2375_from_persistent_fp64'),
        ('burst','recurrence_searches/D240_from_burst_fp64'),
        ('burst','recurrence_searches/D240_extended21s_fp64'),
        ('burst','recurrence_searches/D245_from_burst_fp64'),
        ('burst','recurrence_searches/D250_from_burst_fp64'),
        ('reset','qualification/selected_g40/D0.250000_degree6_dv0.125_4000ms')]
    rows=[];pending=[]
    for history,rel in cases:
        folder=OUT/rel
        if not (folder/'status.json').exists() or json.load(open(folder/'status.json'))['status']!='COMPLETE':
            pending.append(rel);continue
        cfg=json.load(open(folder/'config.json'))
        with np.load(folder/'trajectory.npz') as z:r=z['rate_1ms']
        end=cfg.get('initial_ms',0.)+len(r);start=end-2000
        s=summarize(folder,start,end)
        peaks,_=find_peaks(r[-2000:,0],height=20,prominence=10,distance=100)
        row=dict(D=cfg['D'],history=history,source=rel,precision=cfg.get('precision','FP64'),
            interval_definition='Global E peaks in 1 ms bins, at least 100 ms apart; this is a rate readout, not proof of periodicity',
            statistics=s,peak_times_ms=start+peaks,peak_intervals_ms=np.diff(peaks),
            global_1ms_minimum_hz=float(r[-2000:,0].min()),global_1ms_maximum_hz=float(r[-2000:,0].max()),
            invariant_branch_acceptance='NOT_ESTABLISHED')
        rows.append(row)
    report=dict(status='COMPLETE_FOR_AVAILABLE_RUNS',rows=rows,pending=pending,
        interpretation='Finite-time history-dependent autonomous responses. Z frozen on the specified spatial path; M dynamic. No saddle-node, Hopf, period-doubling, or separatrix inferred from this table.')
    (OUT/'continued_regime_audit.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    with (OUT/'continued_regime_audit.csv').open('w') as f:
        keys=['D','history','window_start_ms','window_end_ms','mean_E_hz','finite_events','quiet_fraction','category','source','precision']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:
            s=r['statistics'];w.writerow(dict(D=r['D'],history=r['history'],window_start_ms=s['window_ms'][0],window_end_ms=s['window_ms'][1],
                **{k:s[k] for k in ('mean_E_hz','finite_events','quiet_fraction','category')},source=r['source'],precision=r['precision']))
    for r in rows:print(r['D'],r['history'],r['statistics']['mean_E_hz'],r['statistics']['category'])


if __name__=='__main__':main()

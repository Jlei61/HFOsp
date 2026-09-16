#!/usr/bin/env python3
"""Native event and resource audit beyond the operational high/return marker.

Quiet-high-quiet-high is distinguished from recovery of recurrent finite events.
All definitions are observations; none enters the simulator's input or dynamics.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,time
from pathlib import Path
import numpy as np
import analyze_topic4_autonomous_recovery as summary

def finite_events(rate,regions,nr):
    quiet=rate<5
    edges=np.diff(np.r_[False,quiet,False].astype(int))
    runs=[(a,b) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=3]
    events=[]
    for left,right in zip(runs[:-1],runs[1:]):
        lo,hi=left[1],right[0]
        if hi<=lo or hi-lo>30 or rate[lo:hi].max()<20:continue
        peak=lo+int(np.argmax(rate[lo:hi]));rr=regions[lo:hi]/nr/.01
        events.append(dict(start_s=lo*.01,end_s=hi*.01,peak_s=(peak+.5)*.01,
            duration_s=(hi-lo)*.01,peak_all_E_Hz=float(rate[peak]),
            core_A_B_surround_peak_Hz=rr[:,:3].max(0).tolist(),
            core_A_B_peak_time_s=[(lo+int(np.argmax(rr[:,i]))+.5)*.01 for i in [0,1]]))
    return events

def temporal_audit(rate,regional_rate,entries,recoveries):
    """Check the actual intervals behind rolling-window state labels.

    A rolling low-window confirmation can fall inside the next rising episode.
    Preserve the original marker but expose that temporal overlap for review.
    """
    def intervals(mask,min_bins=1):
        edge=np.diff(np.r_[False,mask,False].astype(int))
        return [dict(start_s=float(lo*.01),end_s=float(hi*.01),duration_s=float((hi-lo)*.01))
            for lo,hi in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if hi-lo>=min_bins]
    high=intervals(rate>=200,20)
    quiet=intervals((rate<5)&(regional_rate[:,:2]<5).all(1),3)
    checks=[]
    for i,rec in enumerate(recoveries):
        previous=entries[i] if i<len(entries) else None
        following=entries[i+1] if i+1<len(entries) else None
        lo=previous['confirmation_s'] if previous else 0.
        hi=following['onset_s'] if following else len(rate)*.01
        gaps=[dict(start_s=max(v['start_s'],lo),end_s=min(v['end_s'],hi),
                   duration_s=max(0.,min(v['end_s'],hi)-max(v['start_s'],lo)))
              for v in quiet if v['end_s']>lo and v['start_s']<hi]
        inside=[v for v in high if v['start_s']<=rec['confirmation_s']<v['end_s']]
        checks.append(dict(recovery_index=i,confirmation_s=rec['confirmation_s'],
            confirmation_inside_sustained_high=inside,
            next_onset_precedes_recovery_confirmation=bool(following and following['onset_s']<rec['confirmation_s']),
            strict_core_and_global_quiet_intervals_between_entries=gaps,
            longest_strict_quiet_gap_s=max([v['duration_s'] for v in gaps],default=0.)))
    return dict(sustained_high_intervals=high,recovery_timing_checks=checks,
        interpretation='Read-only temporal audit of the original state marker. High is the unchanged all-E >=200Hz for200ms; strict quiet is all-E and both cores <5Hz for at least30ms. Rolling low-window confirmation may overlap a new burst; inspect actual intervals and finite events before accepting a recurrence figure.')

def main(root):
    summary.OUT=root;geometry=np.load(root/'geometry.npz');nr=geometry['region_counts'];rows=[]
    for folder in sorted((root/'runs').iterdir()):
        if folder.name.startswith('qa_') or not (folder/'chunks').exists():continue
        a=summary.load(folder,['spikes_1ms','regions_1ms','slow_time_ms','Z','field_5ms'])
        if a is None:continue
        s=summary.classify(folder);n=len(a['spikes_1ms'])//10
        rate=a['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/32000/.01
        regs=a['regions_1ms'][:n*10].reshape(n,10,6).sum(1)
        events=finite_events(rate,regs,nr)
        timing=temporal_audit(rate,regs[:,:3]/nr[:3]/.01,s['entries'],s['recoveries'])
        first=s['entries'][0]['onset_s'] if s['entries'] else n*.01
        pre=[e for e in events if e['start_s']>=.2 and e['end_s']<first]
        post_high=[e for e in events if s['entries'] and e['start_s']>=s['entries'][0]['confirmation_s']]
        returns=[];ts=a['slow_time_ms']/1000;zz=a['Z'][:,[0,5,6]]
        for i,rec in enumerate(s['recoveries']):
            end=s['entries'][i+1]['onset_s'] if len(s['entries'])>i+1 else n*.01
            start=rec['start_s']
            post=[e for e in events if e['start_s']>=start and e['end_s']<end]
            after=[e for e in post if e['start_s']>=rec['confirmation_s']]
            sel=(ts>=start)&(ts<end)
            zreturn=None
            if sel.sum()>1:
                x=zz[sel];minimum=np.minimum.accumulate(x,axis=0)
                # Ordered upward excursion, not an unordered min/max range.
                zreturn=np.max(x-minimum,axis=0).tolist()
            returns.append(dict(low_window_start_s=start,low_state_confirmation_s=rec['confirmation_s'],
                next_entry_s=end if len(s['entries'])>i+1 else None,
                finite_events_in_return_interval=post,finite_events_after_confirmation=after,
                Z_ordered_upward_excursion_mean_A_B=zreturn,
                distinction='Return to quiescence alone is not evidence that the previous repertoire of recurrent self-limited propagation has returned. Positive Z excursion is descriptive, not proof of a stable limit cycle.'))
        # Display-readiness is intentionally more demanding than the rate marker.
        full=False
        if len(s['entries'])>=2 and returns:
            zup=returns[0]['Z_ordered_upward_excursion_mean_A_B']
            full=len(pre)>=3 and len(returns[0]['finite_events_in_return_interval'])>=2
        record=dict(name=folder.name,observed_s=n*.01,complete=s['complete'],operational_mode=s['mode'],
            preentry_finite_event_count=len(pre),preentry_events=pre,returns=returns,temporal_audit=timing,
            post_first_entry_finite_event_count=len(post_high),post_first_entry_finite_events=post_high,
            post_high_event_interpretation='Finite episodes are retained even when the original2s low-mean return marker is not met. Their presence alone does not establish recovery of the preentry repertoire; compare durations, rates and native spatial recruitment.',
            ready_for_full_recurrence_figure_review=bool(full),
            readiness_rule='At least3 finite preentry events, operational high-return-high and at least2 finite events during the return interval. Z excursions are reported separately: recovery of activity is not assumed to require a preset size of Z recovery. This is a review aid, not a biological success threshold or proof of patient compatibility, an autonomous attractor, or independent confirmation.',
            confirmation='One development noise realization unless the job explicitly names a separate seed. Human raster and native-field review still required.')
        summary.write(folder/'finite_event_resource_audit.json',record);rows.append(record)
    summary.write(root/'finite_event_resource_audit.json',dict(updated_at=time.time(),
        event_definition='All-E10ms rate: a peak>=20Hz, duration<=300ms, bounded on both sides by at least30ms continuously<5Hz. Events not already returned before first onset are excluded from preentry count. These are model events, not labels of patient TA/TB.',rows=rows))
    print([(r['name'],r['preentry_finite_event_count'],r['operational_mode'],r['ready_for_full_recurrence_figure_review']) for r in rows])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=summary.OUT);args=ap.parse_args();main(args.root)

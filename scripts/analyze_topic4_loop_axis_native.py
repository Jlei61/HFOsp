#!/usr/bin/env python3
"""Native axis controls: same observers, explicit structural and censoring limits."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import run_topic4_loop_axis_native as axis
import analyze_topic4_interictal_recurrence as audit
import run_topic4_rhythm_preserving_feedback as rhythm
from run_topic4_recovery_window import event_features
from zoom_topic4_return_core_propagation import event_metrics, summarize


def load_chunks(folder, keys):
    parts={k:[] for k in keys};end=0
    for path in sorted(folder.glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as z:
            if 'start_step' in z:
                assert int(z['start_step'])==end
                end=int(z['end_step'])
            for key in keys:parts[key].append(z[key])
    return {k:np.concatenate(v) for k,v in parts.items()} if parts[keys[0]] else None


def one(condition, source=False):
    root=axis.native.SOURCE if source else axis.OUT/condition
    name=axis.native.NAME if source else f'{condition}_s9108405'
    folder=root/'runs'/name
    data=load_chunks(folder/'chunks',['spikes_1ms','regions_1ms','field_5ms','slow_time_ms','Z','inputs'])
    if data is None:return None
    if source:
        # Compare the same120s observation window, although the source contains240s.
        for key,rate in [('spikes_1ms',1000),('regions_1ms',1000),('field_5ms',200),
                         ('slow_time_ms',50),('Z',50),('inputs',10)]:
            data[key]=data[key][:120*rate]
    end=len(data['spikes_1ms'])/1000.
    with np.load(root/'geometry.npz') as geo:
        counts=np.r_[32000,geo['region_counts'][:3]]
    raw=np.c_[data['spikes_1ms'][:,0],data['regions_1ms'][:,:3]]
    rates=raw.reshape(-1,10,4).sum(1)/counts/.01
    r5=raw.reshape(-1,5,4).sum(1)/counts/.005
    assert np.array_equal(data['field_5ms'].sum(1),data['spikes_1ms'][:,0].reshape(-1,5).sum(1))
    assert np.array_equal(data['regions_1ms'][:,:3].sum(1),data['spikes_1ms'][:,0])
    primary=audit.temporal_audit(rates)
    events=rhythm.strict_events(rates,end)
    # The established event observer requires two qualifying quiet intervals.
    # Expose an unfinished final activity episode instead of silently reading
    # its absence from complete-event counts as absence of population activity.
    quiet_spans=[(lo,hi) for lo,hi in audit.old.event_audit.f.spans(rates[:,:3].max(1)<5.)
                 if (hi-lo)*.01>=.02-1e-9]
    unfinished=None
    tail_start=quiet_spans[-1][1] if quiet_spans else 0
    if len(rates)-tail_start>=2 and rates[tail_start:,0].max()>=20.:
        unfinished=dict(start_s=tail_start*.01,observed_end_s=end,
            observed_duration_lower_bound_s=(len(rates)-tail_start)*.01,
            right_censored=True,left_boundary_confirmed=bool(quiet_spans),
            mean_Hz_allE_A_B_other=rates[tail_start:].mean(0).tolist(),
            peak_Hz_allE_A_B_other=rates[tail_start:].max(0).tolist(),
            interpretation='Activity since the last qualifying20ms jointquiet interval; no following qualifying quiet interval observed. Not counted as a complete brief event.')
    first=primary['entries'][0]['onset_s'] if primary['entries'] else end
    initial=audit.interval_events(events,.5,min(first,8.))
    baseline=event_features(initial['brief_events'],rates,data['field_5ms'])
    comparison_keys=['duration_ms','interval_ms','peak_Hz']
    baseline_estimable=all(baseline.get(k) is not None and np.isfinite(baseline[k]) and baseline[k]>0
                           for k in comparison_keys)
    initial_spatial=[event_metrics(dict(rates=r5),e) for e in initial['brief_events']]
    # Current-graph 8s Z reference is common across structural controls. Also
    # report each control's own pre-entry reference, rather than mixing them.
    with np.load(axis.native.SOURCE/'references/native_s9108405.npz') as reference:
        ir=np.searchsorted(reference['slow_time_ms'],8000.,side='right')-1
        zref=reference['Z'][ir,[5,6]]
    ts=data['slow_time_ms']/1000.;zcores=data['Z'][:,[5,6]]
    own_ref_time=min(8.,first)
    own_index=np.searchsorted(ts,own_ref_time,side='left')-1
    own_zref=zcores[max(0,own_index)]
    episodes=[]
    for ex in primary['low_activity_exits']:
        lo=ex['start_s'];hi=next((e['onset_s'] for e in primary['entries'] if e['onset_s']>ex['confirmation_s']),end)
        ix=np.flatnonzero((ts>=lo)&(ts<hi))
        if not len(ix):continue
        at_reference=ix[np.all(zcores[ix]>=zref,axis=1)]
        own_hits=ix[np.all(zcores[ix]>=own_zref,axis=1)]
        hit=float(ts[at_reference[0]]) if len(at_reference) else None
        returned=audit.interval_events(events,max(ex['confirmation_s'],hit),hi) if hit is not None else None
        features=event_features(returned['brief_events'],rates,data['field_5ms']) if returned else {}
        ratios={k:features.get(k)/baseline[k] if features.get(k) is not None and baseline[k] else None
                for k in comparison_keys}
        recurrent=bool(returned and audit.qualifies(returned,minimum_n=10,minimum_span=5.))
        temporal=(bool(recurrent and all(v is not None and .5<=v<=2 for v in ratios.values()))
                  if baseline_estimable else None)
        spatial=[event_metrics(dict(rates=r5),e) for e in returned['brief_events']] if returned else []
        episodes.append(dict(exit=ex,window_end_s=hi,peak_joint_core_Z=float(zcores[ix].min(1).max()),
            first_recovered_current_graph_reference_s=hit,
            first_recovered_own_preentry_reference_s=float(ts[own_hits[0]]) if len(own_hits) else None,
            short_events_after_recovery=returned,features=features,ratios_to_own_initial=ratios,
            brief_recurrence_after_common_Z_screen=recurrent,
            temporal_return_screen=temporal,
            own_initial_comparison_status='ESTIMABLE' if baseline_estimable else 'NOT_ESTIMABLE_OWN_INITIAL_EVENTS',
            core_recruitment=summarize(spatial) if spatial else None,
            event_metrics=spatial,next_entry_censored=hi==end))
    result_path=folder/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    # Full exogenous stream is independent of network spikes in this engine.
    input_source=load_chunks(axis.native.SOURCE/'runs'/axis.native.NAME/'chunks',['inputs'])['inputs']
    expected=input_source[:len(data['inputs'])]
    assert np.array_equal(data['inputs'],expected), 'Paired external input records changed'
    return dict(condition=condition,name=name,observed_s=end,
        run_status=result.get('status','RUNNING_PARTIAL'),full120s=end>=120,
        entries=primary['entries'],exits=primary['low_activity_exits'],
        complete_activity_episodes=events,right_censored_activity_episode=unfinished,
        initial_short_events=initial,initial_features=baseline,
        own_initial_comparison_estimable=bool(baseline_estimable),
        initial_core_recruitment=summarize(initial_spatial) if initial_spatial else None,
        initial_event_metrics=initial_spatial,
        common_current_graph_Z_reference=zref.tolist(),own_preentry_Z_reference=own_zref.tolist(),
        own_reference_time_s=float(ts[max(0,own_index)]),episodes=episodes,
        temporal_returns=sum(e['temporal_return_screen'] is True for e in episodes) if baseline_estimable else None,
        temporal_return_comparison_status='ESTIMABLE' if baseline_estimable else 'NOT_ESTIMABLE_OWN_INITIAL_EVENTS',
        brief_recurrence_episodes_after_common_Z=sum(e['brief_recurrence_after_common_Z_screen'] for e in episodes),
        return_readout_definition='Brief recurrence requires at least10 complete20-200ms events spanning5s and80percent brief, after both cores regain the common current-graph Z reference. Comparison to own initial events additionally requires duration, interval and peak ratios within0.5-2. Missing own initial features makes that comparison null, not failure; no substitute baseline is silently used. All counts refer only to the observed window; propagation reviewed separately.',
        paired_exogenous_records_exact=True,spike_field_integrity='PASS',
        autonomous=True,externally_clamped=False,certified_bifurcation=False,
        scientific_unit='One trajectory per fixed graph, common seed9108405; events nested within trajectory.',
        human_spatial_review='PENDING')


def main():
    rows=[one('current',True)]
    rows += [one(c) for c in ['rotated','isotropic']]
    rows=[r for r in rows if r is not None]
    axis.native.write(axis.OUT/'analysis.json',dict(rows=rows,
        status='COMPLETE' if len(rows)==3 and all(r['full120s'] for r in rows) else 'RUNNING_PARTIAL',
        observer='Same high-entry/low-activity observer plus jointquiet-bounded20-200ms short events. Z recovery, brief recurrence and comparison with own initial events assessed separately; the last is not estimable when own initial features are missing.',
        limits='One matched-noise trajectory per graph; source outdegree changes and rotation anisotropy attenuates. Do not call differences pure orientation effects or infer certified bifurcations.'))
    print(json.dumps([dict(condition=r['condition'],seconds=r['observed_s'],entries=len(r['entries']),
                           exits=len(r['exits']),temporal_returns=r['temporal_returns'],
                           brief_recurrence_after_Z=r['brief_recurrence_episodes_after_common_Z'],
                           own_initial_comparison=r['temporal_return_comparison_status']) for r in rows]),flush=True)


if __name__=='__main__':main()

"""Read-only audit: can a window centroid fall between distinct local bursts?

Replay the frozen observer exactly, preserving events, masks, scores and labels.
These model-only diagnostics are not new patient-derived training penalties.
"""
import argparse,json
import numpy as np
from scripts import analyze_topic4_propagation_recovery_night as review
from src import topic4_observation_repaired as observer
an=review.an;rt=review.rt


def main(phase):
    old,plan,spec,cases=review.stage_cases(phase)
    out=review.night.OUT/('local_burst_audit_'+phase);out.mkdir(exist_ok=True)
    contract=rt.load_observation_contract(rt.read(an.run.PARENT));events=[];contacts=[];checks=[]
    for c in cases:
      seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==c['base_id'] and int(t)==c['topology']}) or old['seeds']
      for seed in seeds:
        path=an.run.result_path(c['output_stage'],c['base_id'],c['topology'],seed);unit=an.load_unit(path,old['analysis']['burnin_ms'])
        if unit is None:continue
        r,a,primary=unit;env=a['contact_envelope'].T;dt=float(a['contact_envelope_dt_ms'])
        replay=observer.observe(env,dt,contract)
        # Saved envelopes are float32, whereas the worker observes before casting.
        # Require identical events/masks and sub-microsecond time agreement.
        assert np.array_equal(np.isfinite(replay['centroid_ms']),np.isfinite(a['centroid_ms']))
        assert np.allclose(replay['centroid_ms'],a['centroid_ms'],equal_nan=True,rtol=0,atol=1e-4)
        assert np.array_equal(replay['primary_event_indices'],a['primary_event_indices'])
        checks.append(dict(candidate=c['id'],seed=seed,event_and_mask_identity=True,maximum_centroid_delta_ms=float(np.nanmax(abs(replay['centroid_ms']-a['centroid_ms']))),arrays_sha256=r['arrays_sha256']))
        bar=np.asarray(replay['threshold']);base=np.asarray(contract['baseline'])
        for i in an.all_detected_ids(r,old['analysis']['burnin_ms']):
          e=replay['events'][i];lo,hi=e['window_ms'];start=round(lo/dt);stop=round(hi/dt)
          segment=env[:,start:stop];baseline=np.maximum(base,np.quantile(segment,.1,axis=1));mass=np.maximum(segment-baseline[:,None],0)
          common=dict(candidate=c['id'],base_id=c['base_id'],display_name=review.display(c),topology_seed=c['topology'],seed=seed,event=int(i),
              mode='TA' if a['event_mode'][i]==1 else 'TB',primary=bool(i in primary),window_start_ms=lo,window_end_ms=hi)
          gap=[];multib=[];outside_group=[]
          for ci in np.flatnonzero(np.isfinite(a['centroid_ms'][i])):
            detected=segment[ci]>bar[ci]
            minimum=max(1,int(np.ceil(contract['minimum_detection_ms']/dt)))
            for aa,bb in observer.runs(detected):
                if bb-aa<minimum:detected[aa:bb]=False
            rr=observer.runs(detected);center=float(a['centroid_ms'][i,ci]);t=(np.arange(start,stop)+.5)*dt
            value=float(np.interp(center,t,mass[ci]));fraction=value/max(float(mass[ci].max()),1e-20)
            gap.append(fraction);multib.append(len(rr)>1)
            group_lo,group_hi=e['qualifying_interval_ms'];padding=contract['extension_ms']
            contributes=any((start+aa)*dt-padding<group_hi and (start+bb)*dt+padding>group_lo for aa,bb in rr)
            outside_group.append(not contributes)
            contacts.append(dict(common,contact=str(a['contact_names'][ci]),n_local_detections=len(rr),centroid_ms=center,
                envelope_at_centroid_over_contact_peak=fraction,total_baseline_subtracted_mass=float(mass[ci].sum()),
                detection_with_30ms_extension_contributes_to_qualifying_interval=contributes,
                detection_spans_ms=json.dumps([[(start+aa)*dt,(start+bb)*dt] for aa,bb in rr])))
          events.append(dict(common,n_participants=len(gap),multiple_local_bursts_contact_fraction=e['multiple_local_bursts_contact_fraction'],
              median_mass_outside_detection=e['median_mass_outside_detection'],centroid_at_less_than_10pct_peak_fraction=float(np.mean(np.asarray(gap)<.1)),
              centroid_at_less_than_25pct_peak_fraction=float(np.mean(np.asarray(gap)<.25)),participant_without_qualifying_interval_contribution_fraction=float(np.mean(outside_group))))
    an.writecsv(out/'events.csv',events);an.writecsv(out/'contacts.csv',contacts)
    summaries=[]
    for key in sorted(set((e['candidate'],e['seed'],e['mode']) for e in events)):
      for layer in ['primary','all_detected']:
        ee=[e for e in events if (e['candidate'],e['seed'],e['mode'])==key and (layer=='all_detected' or e['primary'])]
        if not ee:continue
        z=dict(candidate=key[0],seed=key[1],mode=key[2],layer=layer,n=len(ee))
        for field in ['multiple_local_bursts_contact_fraction','median_mass_outside_detection','centroid_at_less_than_10pct_peak_fraction','centroid_at_less_than_25pct_peak_fraction','participant_without_qualifying_interval_contribution_fraction']:
            x=np.array([e[field] for e in ee]);z.update({field+'_median':float(np.median(x)),field+'_mean':float(x.mean())})
        summaries.append(z)
    an.writecsv(out/'summary.csv',summaries)
    rt.write(out/'audit.json',dict(status='COMPLETE_READ_ONLY',phase=phase,checks=checks,events=len(events),contact_observations=len(contacts),
        inference='Window isolation does not guarantee a single local burst. Values at 10/25 percent peak are descriptive sensitivity cutoffs, not patient-calibrated gates. No mask/event/score changed.',
        producer=__file__,producer_sha256=rt.sha(__file__)))
    print(json.dumps(dict(output=str(out),runs=len(checks),events=len(events)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='wave1');args=parser.parse_args();main(args.phase)

"""Frozen patient-neighborhood diagnostics, without changing selection or scores."""
import argparse
import numpy as np
from scripts import analyze_topic4_propagation_recovery_night as review
an=review.an;rt=review.rt

def main(phase):
    old,plan,spec,cases=review.stage_cases(phase)
    ev=rt.load_evaluator(rt.read(an.run.PARENT));radius=np.asarray(ev.radii)
    dest=review.night.OUT/('patient_support_'+phase);dest.mkdir(parents=True,exist_ok=True)
    lookup={(c['base_id'],c['topology']):c for c in cases};rows=[];missing=[]
    for cid,topo,seed in spec['units']:
        c=lookup[(cid,int(topo))];path=an.run.result_path(spec['stage'],cid,topo,seed)
        if not an.run.complete(path):missing.append([cid,topo,seed]);continue
        r=rt.read(path)
        with np.load(path.with_suffix('.npz')) as z:
            a={k:z[k] for k in ['event_mode','event_support','event_distance_modes','event_time_ms','primary_event_indices']}
        ids=an.all_detected_ids(r,1500);primary=ids[np.isin(ids,a['primary_event_indices'])]
        for layer,ii in [('primary',primary),('all_detected',ids)]:
          for mode,label in [('ALL',None),('TA',1),('TB',0)]:
            selected=ii if label is None else ii[a['event_mode'][ii]==label]
            d=a['event_distance_modes'][selected];states=a['event_support'][selected];labels=a['event_mode'][selected]
            valid=np.isfinite(d).all(1)&(labels>=0)
            d=d[valid];states=states[valid];labels=labels[valid]
            supported=(d<=radius[:,0]).any(1);outside=(d>radius[:,1]).all(1)
            row=dict(candidate=cid,display_name=review.display(c),topology_seed=int(topo),dynamics_seed=int(seed),layer=layer,mode=mode,
                selected_events=len(selected),readable_events=len(d),
                assigned_mode_supported_n=int((states==1).sum()),assigned_mode_outside_n=int((states==-1).sum()),
                within_either_q90_n=int(supported.sum()),outside_both_q99_n=int(outside.sum()),
                other_mode_q90_only_n=int((supported&(states!=1)).sum()))
            for key in ['assigned_mode_supported','assigned_mode_outside','within_either_q90','outside_both_q99','other_mode_q90_only']:
                row[key+'_fraction']=row[key+'_n']/len(d) if len(d) else None
            for j,name in [(1,'TA'),(0,'TB')]:
                row['distance_to_'+name+'_median']=float(np.median(d[:,j])) if len(d) else None
            rows.append(row)
    an.writecsv(dest/'support.csv',rows)
    rt.write(dest/'manifest.json',dict(status='COMPLETE' if not missing else 'PARTIAL',phase=phase,missing=missing,
        model_units=len(spec['units'])-len(missing),source_spec=str(review.night.OUT/f'{phase}_units.json'),
        radii_q90_q99=radius.tolist(),producer=__file__,producer_sha256=rt.sha(__file__),
        frozen_rule='Mean joint-feature distance to five FIT events in each patient mode; radii are 90th/99th percentiles of assigned-mode CAL distances. No refitting or new threshold.',
        either_mode='Within either fixed q90 neighborhood, or outside both fixed q99 neighborhoods. These are descriptive union/intersection checks; no newly calibrated global false-positive rate.',
        interpretation='Patient FIT/CAL are previously reused development blocks, not independent external validation. A classifier label does not imply support; support in frozen features does not prove pathway or HFO waveform recovery.',
        no_change='No training loss, primary eligibility, patient geometry or model selection modified. Within-run events are not treated as independent networks.'))
    print(dict(output=str(dest),complete_units=len(spec['units'])-len(missing),missing=len(missing)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='long');a=p.parse_args();main(a.phase)

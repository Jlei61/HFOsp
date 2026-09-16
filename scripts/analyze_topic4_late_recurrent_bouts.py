#!/usr/bin/env python3
"""Keep all quiet-bounded bouts, including those longer than the IED cutoff."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json,time
import numpy as np
import analyze_topic4_autonomous_recovery as common
import analyze_topic4_event_extent as extent
from analyze_topic4_long_recovery_contrast import PAIRS,BASE

KEYS=['duration_s','peak_all_E_Hz','native_recruited_area_union',
      'native_peak_simultaneous_area','preceding_quiet_s']

def summarize(rows):
    return dict(n=len(rows),metrics={key:dict(min=float(np.min([r[key] for r in rows])),
        median=float(np.median([r[key] for r in rows])),max=float(np.max([r[key] for r in rows])))
        for key in KEYS} if rows else {})

def main():
    output=[]
    for subdir,name,label in PAIRS:
        root=BASE/subdir;folder=root/'runs'/name
        assert (root/'continuation_complete.json').exists(),'Wait for actual saved endpoint'
        # Re-read native fields to avoid using a supervisor's earlier prefix.
        extent.main(root,name)
        d=common.read(folder/'event_extent_audit.json');r=common.read(folder/'result.json')
        assert d['observed_s']==r['end_s']
        bouts=d['episodes'];first=r['tracker']['entries'][0]
        groups={
            'preentry': [v for v in bouts if v['start_s']>=.2 and v['end_s']<first['onset_s'] and v['left_bounded'] and v['right_bounded']],
            'postfirst_before30': [v for v in bouts if v['start_s']>=first['confirmation_s'] and v['end_s']<=30 and v['left_bounded'] and v['right_bounded']],
            'late30_to_endpoint': [v for v in bouts if v['start_s']>=30 and v['left_bounded'] and v['right_bounded']]}
        censored=[v for v in bouts if v['start_s']>=30 and not v['right_bounded']]
        row=dict(label=label,name=name,observed_s=d['observed_s'],groups={key:summarize(value) for key,value in groups.items()},
            late_complete_bouts=groups['late30_to_endpoint'],late_right_censored_activity=censored,
            cutoff_s=.3,late_bouts_longer_than_finite_event_cutoff=sum(v['duration_s']>.3 for v in groups['late30_to_endpoint']))
        output.append(row);print(label,row['groups'],'censored',[(v['start_s'],v['duration_s']) for v in censored],flush=True)
    common.write(BASE/'overnight_review/late_bout_review.json',dict(updated_at=time.time(),rows=output,
        selection='All all-E>=20Hz-peak activity bouts separated by at least30ms<5Hz. Complete bouts require quiet on both sides. No upper duration cutoff is applied here; long and right-censored activity are reported rather than silently excluded.',
        spatial_observable='Direct1mm neuron-count grid,20ms sliding windows every5ms; active grid>=20Hz. Area union and peak simultaneous area are separate.',
        statistical_unit='Descriptive within-trajectory bouts; no event-level p-values or independent-replicate claim.',
        scientific_boundary='Repeated terminated bouts can constitute a broad bursting state without restoring the initial localized interictal-event repertoire. A right-censored high segment is not an observed finite burst.'))

if __name__=='__main__':main()

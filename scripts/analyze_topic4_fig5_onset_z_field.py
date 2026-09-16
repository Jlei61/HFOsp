#!/usr/bin/env python3
"""Fig3C broadband robust-z on a fixed model window and E1146 seizure fields."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import csv
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import analyze_topic4_fig5_preentry_events as audit
# The historical model loader prepends archived worktrees to sys.path.
# Patient readout must use the current, locked Fig3C implementation.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
src.__path__.insert(0,str(ROOT/'src'))
from scripts.paper_figures import plot_fig3b_interictal_ictal_shared_field as clinical
from src.topic5_template_axis_field import scorers_from_interictal_record, _smooth_from_weights
assert Path(clinical.__file__).resolve()==ROOT/'scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py'

OUT=audit.OUT/'onset_broadband_z_20260915'
PATIENT_CACHE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/patient_readout_v1/per_seizure'
BASELINE=(.5,3.5)
EARLY_DURATION=1.


def write(path,data):
    audit.f.write(path,audit.f.safe(data))


def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    value=dict(question='Compare onset-related broadband enhancement fields in the displayed model trajectory and one explicitly selected seizure from E1146.',
        model_baseline_s=list(BASELINE),model_early_duration_s=EARLY_DURATION,
        model_onset='Start of the first all-E >=200Hz episode sustained for200ms; not the earlier state3 pre-ictal frame or confirmation time.',
        model_signal='Recorded pre-Z synaptic-current virtual-contact proxy, CAR across all15 available virtual contacts before PSD.',
        estimator='Canonical Fig3C _extract_log_band_power: SciPy spectrogram Tukey0.25, constant detrend,1s window,0.5s hop; natural log of summed1-150Hz PSD.',
        normalization='Canonical distal_baseline_robust_z: subtract baseline log-power median, divide by1.4826*MAD, then subtract baseline robust-z median.',
        short_baseline_adaptation='Model baseline0.5-3.5s supplies5 complete1s spectral cells. Explicit min_frames=5 replaces clinical min_frames=50 for this model-only descriptive adapter.',
        time_interpretation='The1s model early window and3s model baseline are phase-matched display choices; there is no validated10-fold model/patient time mapping and no rescaling of frequencies.',
        patient='All25 complete exact1-150Hz E1146 seizures, clinical0-10s; each retains its own EEG-onset[-120,-90]s distal baseline and canonical CAR.',
        selection='Require positive mean patient robust-z across all15 contacts to represent overall enhancement; maximize signed Pearson correlation between the two fields evaluated at the15 contacts using the frozen shared-A identity kernel. No mirror/sign flip or contact exclusion. Ties use lower seizure index.',
        secondary='Retain signed raw-contact Pearson/Spearman and the unrestricted all25 ranking. The positive-mean gate is fixed before scoring this model.',
        display='Both fields use Fig3C right-panel painter: frozen shared-TA geometry/support,6mm display kernel,Blues,continuous min-max for painting with colorbars restored to actual robust-z. No rank transform.',
        statistical_unit='One selected patient seizure per model realization;15 contacts and5 overlapping baseline spectral cells are not independent subjects.',
        boundary='Selected illustration only. Model proxy and clinical EEG observation operators and baseline durations differ; matched field shape is not held-out validation, absolute amplitude equivalence, or seizure-mechanism proof.')
    path=OUT/'analysis_contract.json'
    if path.exists():assert audit.f.read(path)==value,'Model window/selection contract changed; use a new analysis version.'
    else:write(path,value)
    return value


@lru_cache(maxsize=1)
def patient_cases():
    record,path=clinical._load_record('epilepsiae_1146')
    fz=clinical.load_frozen('epilepsiae_1146')
    names=list(record['interictal_field']['contact_order'])
    candidates=clinical._checkpoint_rows('epilepsiae_1146',len(names))
    cases=[]
    for candidate in candidates:
        index=candidate['seizure_idx'];source=PATIENT_CACHE/f'seizure_{index:03d}.json'
        if source.exists():
            case=audit.f.read(source)
            assert case['clinical_contact_names']==names
            values=np.asarray(case['activation_clinical_order'],float);meta=case['metadata']
        else:
            values,meta=clinical._extract_clinical_activation('epilepsiae_1146',index,record)
            source=OUT/'per_seizure'/f'seizure_{index:03d}.json'
            write(source,dict(seizure_idx=index,activation_clinical_order=values,clinical_contact_names=names,metadata=meta))
        assert meta['band_hz']==[1.,150.] and meta['reference']=='car'
        assert meta['spectral_window_sec']==1 and meta['spectral_hop_sec']==.5
        assert meta['baseline_eeg_sec']==[-120.,-90.] and meta['clinical_window_sec']==[0.,10.]
        assert meta['n_baseline_frames']>=50 and len(values)==15 and np.isfinite(values).all()
        checkpoint,cp_path=clinical._checkpoint_event('epilepsiae_1146',index)
        parity=clinical._score_audit(record,values,checkpoint)
        error=max(v.get('abs_error',0.) for v in parity['checkpoint_comparison'].values())
        assert error<=1e-12
        cases.append(dict(seizure_idx=index,values=values,extraction=meta,source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint=str(cp_path),checkpoint_max_error=error))
    assert len(cases)==25
    canon=audit.f.read(ROOT/'results/paper-ready-figure/fig3/fig3_panelc_metadata.json')
    assert canon['frozen_fingerprint']==fz['fingerprint']
    locked=next(c for c in cases if c['seizure_idx']==canon['seizure_idx'])
    assert np.allclose(locked['values'],canon['raw_ictal_robust_z_mean'],rtol=0,atol=1e-12)
    return record,fz,cases


def segment_power(signal,times,fs,window):
    lo,hi=window;start=round((lo-times[0])*fs);n=round((hi-lo)*fs)
    assert start>=0 and start+n<=len(times)
    assert np.isclose(times[start],lo,atol=1e-9)
    data=signal[start:start+n].T
    powers,centers=clinical._extract_log_band_power(data,fs,[clinical.BAND],band_hz_override={clinical.BAND:(1.,150.)})
    return powers[clinical.BAND],centers+lo


def compute(a,onset,name):
    contract=protocol()
    record,fz,cases=patient_cases()
    names=[str(n) for n in a['contact_names']];target=list(fz['names'])
    assert len(names)==len(set(names))==15 and set(names)==set(target)
    index=[names.index(n) for n in target]
    times=np.asarray(a['lfp_time_ms'],float)/1000
    measured_fs=1/float(np.median(np.diff(times)));assert np.isclose(measured_fs,2000.)
    fs=2000.  # Exact recorder rate keeps the inclusive150Hz FFT bin in-band.
    assert np.allclose(np.diff(times),1/fs,rtol=0,atol=1e-10)
    raw=np.asarray(a['lfp_raw'],float)
    signal=(raw-raw.mean(axis=1,keepdims=True))[:,index]
    early=(float(onset),float(onset)+EARLY_DURATION)
    bp,bt=segment_power(signal,times,fs,BASELINE)
    ep,et=segment_power(signal,times,fs,early)
    assert bp.shape==(15,5) and ep.shape==(15,1)
    powers=np.c_[bp,ep];centers=np.r_[bt,et]
    robust=clinical.distal_baseline_robust_z(powers,centers,BASELINE,min_frames=5)
    rows,complete=clinical.aggregate_complete_windows(robust['delta'],centers,
        np.asarray([[*early,np.mean(early)]]),spectral_window_sec=1.)
    assert complete[0]
    model=rows[0];assert np.isfinite(model).all(),'Insufficient nonzero model baseline MAD.'
    manual=(ep[:,0]-np.median(bp,axis=1))/(1.4826*np.median(abs(bp-np.median(bp,axis=1)[:,None]),axis=1))
    assert np.allclose(model,manual,rtol=1e-12,atol=1e-12)
    weights=scorers_from_interictal_record(record)['shared_a']['weight_id']
    model_field=_smooth_from_weights(model,weights)
    comparisons=[]
    for case in cases:
        patient=case['values'];field=_smooth_from_weights(patient,weights)
        comparisons.append(dict(seizure_idx=case['seizure_idx'],public_seizure=f'SZ{case["seizure_idx"]+1}',
            field_r=float(np.corrcoef(model_field,field)[0,1]),
            contact_r=float(np.corrcoef(model,patient)[0,1]),contact_rho=float(spearmanr(model,patient).statistic),
            patient_mean_z=float(patient.mean()),patient_min_z=float(patient.min()),
            positive_contacts=int((patient>0).sum()),enhancement_eligible=bool(patient.mean()>0)))
    comparisons.sort(key=lambda row:(-row['field_r'],row['seizure_idx']))
    eligible=[row for row in comparisons if row['enhancement_eligible']]
    assert eligible,'No patient case meets the predeclared enhancement criterion.'
    chosen=eligible[0];patient=next(c for c in cases if c['seizure_idx']==chosen['seizure_idx'])
    result=dict(name=name,contract=contract,contact_names=target,model_robust_z=model,
        model_baseline_s=list(BASELINE),model_early_s=list(early),model_entry_s=float(onset),
        model_baseline_frames=int(robust['n_baseline_frames']),model_early_frames=ep.shape[1],
        model_baseline_log_power_median=robust['baseline_log_power_median'],
        model_baseline_log_power_mad=robust['baseline_log_power_mad'],
        model_baseline_log_power=bp,model_early_log_power=ep,model_baseline_z_median=robust['baseline_z_center'],
        model_mean_z=float(model.mean()),model_positive_contacts=int((model>0).sum()),
        model_signal_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),model_sampling_hz=float(fs),
        selected=chosen,patient_robust_z=patient['values'],patient_extraction=patient['extraction'],
        patient_source=patient['source'],patient_source_sha256=patient['source_sha256'],
        patient_checkpoint_max_error=patient['checkpoint_max_error'],
        n_candidates=len(cases),n_enhancement_candidates=len(eligible),all_candidates=comparisons,unrestricted_best=comparisons[0],
        frozen_fingerprint=fz['fingerprint'],field_scoring='Signed Pearson at15 contact-evaluated fields with frozen shared-A identity weights; no mirror or sign flip.',
        display_geometry='Frozen E1146 shared TA plane, exact name-matched model values; this is an observation-space projection, not native SNN XY.',
        human_review='PENDING',producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    write(dest/'comparison.json',result)
    with (dest/'all_seizure_comparisons.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    np.savez_compressed(dest/'comparison_arrays.npz',contact_names=np.asarray(target),model_robust_z=model,
        patient_robust_z=patient['values'],baseline_log_power=bp,early_log_power=ep,
        baseline_frame_centers_s=bt,early_frame_centers_s=et,frozen_weight_id=weights)
    return result


if __name__=='__main__':
    for row in audit.sources():
        if row['eta_m']!=.0005:continue
        data,result=audit.load_small(row['source'],keys=('lfp_time_ms','lfp_raw'))
        out=compute(data,result['tracker']['entries'][0]['onset_s'],row['name'])
        print(row['name'],json.dumps(audit.f.safe(dict(model_z=out['model_robust_z'],selected=out['selected'],baseline_frames=out['model_baseline_frames']))),flush=True)

#!/usr/bin/env python3
"""Fixed-window model readout vs all complete E1146 clinical-onset cases.

This is an explicitly selected illustration, not an independent validation.
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
from scipy.signal import spectrogram
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import plot_fig3b_interictal_ictal_shared_field as clinical

BASE = ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT = BASE/'patient_readout_v1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def power(signal, fs):
    f, t, psd = spectrogram(np.asarray(signal, float), fs=fs, nperseg=round(fs),
                           noverlap=round(fs/2), scaling='density', mode='psd', axis=0)
    return psd[(f >= 1) & (f <= 150)].sum(axis=0).T, t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'per_seizure').mkdir(exist_ok=True)
    write(OUT/'selection_protocol.json', dict(
        model_window_s=[10.,11.], model_window_reason='Fixed one-second window containing states 2 and 3, before manual refill; not selected by patient fit.',
        band_hz=[1,150], spectral_window_s=1, spectral_hop_s=.5,
        model_baseline_s=[.5,8.], model_value='10 log10(early power / mean baseline power)',
        clinical_value='Canonical distal-baseline robust z of log power, clinical onset 0-10 s',
        initial_diagnostic='Rank all complete exact-band cases by signed Spearman at all 15 exact-name contacts.',
        selection='Among cases with positive enhancement at all 15 contacts, maximize signed Spearman; no sign flip or removal of contacts.',
        revision_reason='The initial unrestricted maximum was SZ16 with all 15 powers below baseline. After seeing that result, an explicit positive-enhancement gate was added to match the user request; the full ranking is retained.',
        statistical_unit='One explicitly selected seizure; 15 spatial contacts are not independent subjects.',
        display_geometry='Patient contacts at their fixed model embedding, no registration to energy values.',
        inference='Best-case illustration, not a held-out test or a population claim.'))
    a=np.load(BASE/'runs/continuous_refill_release.npz')
    names=[str(v) for v in a['contact_names']]
    p,t=power(a['lfp_effective'],2000.)
    base=(t>=1.) & (t<=7.5)
    early=np.isclose(t,10.5)
    assert base.sum()==14 and early.sum()==1
    model_db=10*np.log10(p[early][0]/p[base].mean(axis=0))
    record,record_path=clinical._load_record('epilepsiae_1146')
    clinical_names=record['interictal_field']['contact_order']
    index=np.array([clinical_names.index(name) for name in names])
    candidates=clinical._checkpoint_rows('epilepsiae_1146',len(names))
    rows=[];vectors=[]
    print('model_db',model_db.tolist(),flush=True)
    for row in candidates:
        idx=row['seizure_idx'];cache=OUT/'per_seizure'/f'seizure_{idx:03d}.json'
        if cache.exists():
            result=json.loads(cache.read_text())
        else:
            activation,meta=clinical._extract_clinical_activation('epilepsiae_1146',idx,record)
            checkpoint,path=clinical._checkpoint_event('epilepsiae_1146',idx)
            audit=clinical._score_audit(record,activation,checkpoint)
            error=max(v.get('abs_error',0.) for v in audit['checkpoint_comparison'].values())
            assert error<1e-12,(idx,error)
            result=dict(seizure_idx=idx,activation_clinical_order=activation.tolist(),
                        clinical_contact_names=clinical_names,metadata=meta,
                        checkpoint_path=str(path),checkpoint_max_error=error)
            write(cache,result)
        y=np.asarray(result['activation_clinical_order'])[index]
        assert np.isfinite(y).all()
        rho=float(spearmanr(model_db,y).statistic)
        out=dict(seizure_idx=idx,public_seizure=f'SZ{idx+1}',rho=rho,
                 min_patient_z=float(y.min()),positive_contacts=int((y>0).sum()),
                 checkpoint_max_error=result['checkpoint_max_error'])
        rows.append(out);vectors.append(y)
        print(json.dumps(out),flush=True)
    unrestricted=int(np.argmax([r['rho'] for r in rows]))
    eligible=[i for i,r in enumerate(rows) if r['positive_contacts']==len(names)]
    best=max(eligible,key=lambda i:rows[i]['rho'])
    selected=rows[best]
    with (OUT/'all_seizure_comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows(sorted(rows,key=lambda r:-r['rho']))
    np.savez_compressed(OUT/'comparison_arrays.npz',contact_names=np.array(names),
        contact_xy=a['contact_xy'],centers_mm=a['centers_mm'],model_dB=model_db,
        patient_robust_z=vectors[best],all_patient_robust_z=np.array(vectors),
        seizure_idx=np.array([r['seizure_idx'] for r in rows]),
        baseline_model_power=p[base].mean(axis=0),early_model_power=p[early][0])
    summary=dict(selected=selected,n_candidates=len(rows),n_positive_enhancement_candidates=len(eligible),
        unrestricted_best=rows[unrestricted],
        correlation_range=[min(r['rho'] for r in rows),max(r['rho'] for r in rows)],
        correlation_median=float(np.median([r['rho'] for r in rows])),
        model_positive_contacts=int((model_db>0).sum()),model_dB=model_db.tolist(),
        model_window_s=[10.,11.],clinical_window_s=[0.,10.],
        clinical_source_record=str(record_path),selection_bias='Best among fully positive enhancement cases after auditing all 25; post-result eligibility clarification, not independent validation',
        exact_name_alignment_pass=True,checkpoint_reproduction_pass=True,
        model_estimator='SciPy spectrogram default Tukey(0.25), constant detrend, 1 s window, 1-150 Hz inclusive',
        model_baseline_frames=int(base.sum()),clinical_min_baseline_frames=50,
        amplitudes_comparable=False)
    write(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

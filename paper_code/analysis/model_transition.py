"""New Z/M trajectories -> historical display entry and onset broadband field."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram
from analysis.src.topic5_tspectral_field_concordance import distal_baseline_robust_z


def display_entry(run):
    """Historical t120: centered20ms rate; >=80% of100ms at>=120Hz.

    Kept separate from the native 200-Hz/200-ms safeguard/entry readout.
    This is a high-rate transition descriptor, not a seizure diagnosis.
    """
    dt=float(run['parameters']['dt']);rate=np.convolve(run['rate_E'],np.ones(round(20/dt))/round(20/dt),mode='same')
    width=round(100/dt);hits=(rate>=120).astype(int);count=np.convolve(hits,np.ones(width,dtype=int),mode='valid')
    ix=np.flatnonzero(count>=.8*width)
    return None if not len(ix) else float(run['time_ms'][ix[0]])/1000.


def onset_broadband_field(samples,sfreq,onset_sec,*,baseline=(.5,3.5),duration_sec=1.,minimum_baseline_frames=5):
    """CAR -> Tukey0.25 1s PSD/0.5s hop -> log sum1–150Hz -> robust z.

    Model-only short baseline uses >=5 complete spectral cells. A clinical
    adapter must use its actual distal baseline and >=50 frames instead.
    The onset window must be fully observed; missing coverage is NOT_ESTIMABLE.
    """
    x=np.asarray(samples,float)
    if x.ndim!=2 or not np.isfinite(x).all() or sfreq<=300:raise ValueError('Finite contact x samples, in V/proxy units, with sampling >300Hz required')
    length=x.shape[1]/sfreq
    if onset_sec is None:return dict(status='NOT_ESTIMABLE',reason='No historical high-rate entry')
    if onset_sec+duration_sec>length+1e-9:return dict(status='NOT_ESTIMABLE',reason='Incomplete post-entry spectral window',available_duration_sec=length-onset_sec)
    x=x-x.mean(axis=0,keepdims=True)
    f,t,p=spectrogram(x,fs=sfreq,window=('tukey',.25),nperseg=round(sfreq),noverlap=round(.5*sfreq),detrend='constant',scaling='density',mode='psd',axis=-1)
    logp=np.log(np.maximum(p[:,(f>=1)&(f<=150)].sum(1),np.finfo(float).tiny))
    # Complete windows, matching the declared .5–3.5-s model baseline.
    base=(t-.5>=baseline[0]-1e-9)&(t+.5<=baseline[1]+1e-9)
    if base.sum()<minimum_baseline_frames:return dict(status='NOT_ESTIMABLE',reason='Insufficient complete baseline cells')
    # Recenter the spectral origin so the onset window itself is computed,
    # rather than choosing a neighboring global-grid cell after seeing power.
    start=round(onset_sec*sfreq);stop=start+round(duration_sec*sfreq)
    ff,tt,pp=spectrogram(x[:,start:stop],fs=sfreq,window=('tukey',.25),nperseg=round(sfreq),noverlap=round(.5*sfreq),detrend='constant',scaling='density',mode='psd',axis=-1)
    early=np.log(np.maximum(pp[:,(ff>=1)&(ff<=150)].sum(1),np.finfo(float).tiny)).mean(1)
    location=np.median(logp[:,base],axis=1);scale=1.4826*np.median(abs(logp[:,base]-location[:,None]),axis=1)
    if not np.all(np.isfinite(scale)&(scale>1e-12)):return dict(status='NOT_ESTIMABLE',reason='Zero baseline variance or invalid contact')
    value=np.divide(early-location,scale,out=np.full(len(x),np.nan),where=scale>1e-12)
    value-=np.nanmedian(np.divide(logp[:,base]-location[:,None],scale[:,None],out=np.full_like(logp[:,base],np.nan),where=scale[:,None]>1e-12),axis=1)
    if not np.isfinite(value).all():return dict(status='NOT_ESTIMABLE',reason='Zero baseline variance or invalid contact')
    return dict(status='OK',robust_z=value,baseline_sec=baseline,onset_sec=onset_sec,duration_sec=duration_sec,
                n_baseline_frames=int(base.sum()),band_hz=[1,150],reference='CAR across all supplied contacts',unit='one new model trajectory')


def compare_raw_patient_fields(model_field,raw_seizures,contact_names,scorer):
    """Recompute each candidate from raw EEG, then the paper's descriptive selection.

    Each item supplies samples_V, sfreq, channel_names, onset_sec, baseline_sec
    (a real distal interval in this excerpt's time coordinates), seizure_index.
    Raw inputs must include the full canonical CAR montage. Never feed cached
    onset fields here. Selection is positive mean z, maximum signed identity-
    kernel correlation, then lower seizure index. This is not a validation p.
    """
    from scipy.stats import pearsonr
    from analysis.src.topic5_template_axis_field import _smooth_from_weights
    if model_field['status']!='OK':return dict(status='NOT_ESTIMABLE',reason=model_field['reason'])
    if np.asarray(scorer['weight_id']).shape!=(len(contact_names),len(contact_names)):raise ValueError('Patient comparison requires complete fixed target support')
    model=_smooth_from_weights(model_field['robust_z'],np.asarray(scorer['weight_id']))
    rows=[]
    for raw in raw_seizures:
        field=onset_broadband_field(raw['samples_V'],raw['sfreq'],raw['onset_sec'],baseline=raw['baseline_sec'],duration_sec=10.,minimum_baseline_frames=50)
        row=dict(seizure_index=raw['seizure_index'],status=field['status'])
        if field['status']=='OK':
            names=list(raw['channel_names'])
            if len(set(names))!=len(names):raise ValueError('Duplicate clinical contact names')
            if not all(n in names for n in contact_names):raise ValueError('Clinical comparison needs every fixed target contact')
            value=field['robust_z'][[names.index(n) for n in contact_names]]
            patient=_smooth_from_weights(value,np.asarray(scorer['weight_id']))
            valid=np.isfinite(patient)&np.isfinite(model)
            r=float(pearsonr(model[valid],patient[valid]).statistic) if valid.sum()==len(contact_names) else np.nan
            row.update(robust_z=value,mean_z=float(np.mean(value)),field_r=r,eligible=bool(np.isfinite(r) and np.mean(value)>0))
        rows.append(row)
    eligible=[r for r in rows if r.get('eligible',False)]
    selected=sorted(eligible,key=lambda r:(-r['field_r'],r['seizure_index']))[0] if eligible else None
    return dict(status='OK' if selected else 'NOT_ESTIMABLE',selected=selected,candidates=rows,interpretation='selected descriptive illustration; not held-out validation')

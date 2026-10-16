"""Raw clinical samples and new model currents -> Fig5E/F readouts."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import spectrogram,periodogram,iirnotch,filtfilt
from analysis import paper_clinical as paper
from analysis.spectral_fields import native_entry
from analysis.src.topic5_template_axis_field import _smooth_matrix_from_weights as smooth


def clinical_candidates(directory,contact_names=None):
    """Recompute spectra and the six-band phase rule; baseline annotations are inputs.

    The declared candidate recordings and quiet baseline intervals are fixed
    independently of this model. No stored power, phase, field or match is read.
    """
    directory=Path(directory);rows=[]
    for item in json.loads((directory/'clinical_examples.json').read_text()):
        order=list(item['names'] if contact_names is None else contact_names)
        if set(order)!=set(item['names']):raise ValueError('Clinical and model target contacts differ')
        with np.load(directory/item['file'],allow_pickle=False) as z:
            x=z['samples_V'].astype(float);fs=float(z['sfreq']);names=z['channel_names'].tolist()
        ref=names.index('CAR_REFERENCE');ix=[names.index(n) for n in order]
        x=x[ix]-x[ref]
        for hz in item['notch_hz']:
            b,a=iirnotch(hz,30,fs);x=filtfilt(b,a,x,axis=-1)
        f,t,p=spectrogram(x,fs=fs,nperseg=round(fs),noverlap=round(.5*fs),scaling='density',mode='psd',axis=-1)
        t+=item['start_rel_clinical_s']
        arrays=dict(time_s=t,names=np.array(order),broadband=np.log(np.maximum(p[:,(f>=1)&(f<=150)].sum(1),1e-30)))
        line=paper.line_noise_bin_mask(f,[50,100,150,200,250],2.)
        for band,(lo,hi) in paper.BANDS.items():
            take,_,_=paper.band_bin_selection(f,lo,hi,line,half_open=True)
            arrays['gate_'+band]=np.log(np.maximum(p[:,take].sum(1),1e-30))
        baseline=item['baseline_s'];keep=(t-.5>=baseline[0]-1e-8)&(t+.5<=baseline[1]+1e-8)
        event=dict(item,baseline=dict(indices=np.flatnonzero(keep)),baseline_clinical_s=baseline)
        phase,features=paper.locate_phase(event,arrays)
        row=dict(seizure_number=item['seizure_number'],seizure_id=item['seizure_id'],baseline_s=baseline,names=order,
                 phase=phase,eligible=False,readouts=None,example=None)
        if phase is not None:
            lo,hi=phase['window_s'];qualification=paper.qualify(arrays,event,[lo,hi])
            on=(t-.5>=lo-1e-8)&(t+.5<=hi+1e-8)
            joint=max(((b-a)*.5 for a,b in paper._run_lengths(features['joint']&on)),default=0.)
            value=paper.contrast(arrays['broadband'],t,keep,lo,hi)
            row.update(qualification=qualification,joint_support_s=joint,readouts=value,
                       eligible=bool(qualification['eligible'] and joint>=2 and value is not None and value['n_complete_frames']>=4))
        # Panel E retains its separate distal-baseline / clinical0-10s contract.
        distal=(item['eeg_onset_s']-120,item['eeg_onset_s']-90)
        example_base=(t-.5>=distal[0]-1e-8)&(t+.5<=distal[1]+1e-8)
        if example_base.sum()>=50:
            row['example']=paper.contrast(arrays['broadband'],t,example_base,0.,10.)
        rows.append(row)
    return rows


def descriptive_example(model_field,rows,weights):
    """Panel E: positive mean enhancement, then maximum signed correlation."""
    if model_field['status']!='OK':return dict(status='NOT_ESTIMABLE',reason=model_field['reason'])
    model=paper.unit_rows(smooth(np.atleast_2d(model_field['robust_z']),weights))[0]
    candidates=[]
    for row in rows:
        value=row['example']
        if value is None or np.mean(value['z'])<=0:continue
        field=paper.unit_rows(smooth(np.atleast_2d(value['z']),weights))[0]
        candidates.append(dict(seizure_number=row['seizure_number'],robust_z=value['z'],signed_r=float(model@field)))
    if not candidates:return dict(status='NOT_ESTIMABLE',reason='No positive, complete clinical example')
    return dict(status='OK',selected=sorted(candidates,key=lambda r:(-r['signed_r'],r['seizure_number']))[0],
                interpretation='Selected descriptive illustration; independent of the panel F statistic')


def model_power(car,starts):
    """Original 100-ms Tukey PSD, 1-Hz padded grid, 1-150 Hz sum (true resolution 10 Hz)."""
    frames=np.stack([car[round(s*2000):round((s+.1)*2000)].T for s in starts])
    f,p=periodogram(frames,fs=2000,axis=-1,window=('tukey',.25),detrend='constant',nfft=2000,scaling='density')
    return np.log(np.maximum(p[:,:,(f>=1)&(f<=150)].sum(-1)*(f[1]-f[0]),1e-30)).T


def model_contrast(run):
    """Original quiet-window baseline and final-recruitment-to-power-peak+200ms.

    All times are measured from the new simulation, before clinical matching.
    Missing high entry or stable plateau produces an explicit unavailable result.
    """
    dt=run['parameters']['dt'];entry=native_entry(run['time_ms'],run['rate_E'],dt_ms=dt)['onset_sec']
    if entry is None:return dict(status='NOT_ESTIMABLE',reason='No native high-rate entry')
    ne=len(run['ec']);width=round(10/dt);n=len(run['rate_E'])//width
    all_counts=np.rint(run['rate_E'][:n*width].reshape(n,width).sum(1)*ne*dt/1000).astype(int)
    core=[];counts=[ne]
    for k in range(2):
        ids=np.flatnonzero(run['ec']==k);counts.append(len(ids));mask=np.isin(run['raster_neuron'],ids)
        bins=np.floor(run['raster_time_ms'][mask]/10+1e-9).astype(int)
        core.append(np.bincount(bins[bins<n],minlength=n))
    counts.append(ne-sum(counts[1:]));spikes=np.column_stack([all_counts,*core,all_counts-core[0]-core[1]])
    if np.min(spikes)<0:raise ValueError('Inconsistent core and all-E spike bins')
    times=np.arange(1,n+1)*.01;rates=spikes/np.asarray(counts)[None]/.01
    raw=run['contact_current'][:,::5].T;car=raw-raw.mean(1,keepdims=True)
    try:
        phase=paper.phase_windows(times,rates,entry,entry)
        start,_,_=paper.anchor(times,rates[:,0],entry,phase['plateau_rate_hz'])
        # Reproduce the original spectral grid and final-ramp search interval.
        # Later high-rate power peaks are outside the onset-field comparison.
        origin=max(0.,phase['final_ramp_start_s']-1.)
        stop=min(len(car)/2000-.1,entry+3.)
        starts=round(origin*2000)/2000+np.arange(int(np.floor((stop-origin)/.05))+1)*.05
        pooled=np.exp(model_power(car,starts)).mean(0);sustained=np.convolve(pooled,np.ones(3)/3,mode='valid')
        search=[phase['final_ramp_start_s']-.1,entry+.6]
        eligible=np.flatnonzero((starts[:len(sustained)]>=search[0])&(starts[:len(sustained)]+.2<=search[1]))
        if not len(eligible):raise ValueError('No complete final-ramp power windows')
        peak=int(eligible[np.argmax(sustained[eligible])]);end=float(starts[peak]+.2+.2)
        if end>len(car)/2000:raise ValueError('Incomplete power-peak plus200ms endpoint')
        baseline=paper.baseline_windows(dict(spikes_10ms=spikes,time_s=times),np.asarray(counts),entry)
        if len(baseline)<5:raise ValueError('Fewer than five quiet 100-ms baseline windows')
        early_starts=round(start*2000)/2000+np.arange(int(np.floor((end-start-.1+1e-8)/.05))+1)*.05
        rest=model_power(car,baseline);early=model_power(car,early_starts);location=np.median(rest,1)
        delta=early.mean(1)-location;mad=1.4826*np.median(abs(rest-location[:,None]),1)
        if not np.all(mad>1e-12):raise ValueError('Invalid quiet-baseline spectral variance')
        return dict(status='OK',db=delta*10/np.log(10),robust_z=delta/mad,contact_names=run['contact_names'],
                    baseline_starts_s=baseline,interval_s=[start,end],entry_s=entry,power_search_s=search,
                    spectral_support_s=[float(early_starts[0]),float(early_starts[-1]+.1)])
    except (AssertionError,ValueError) as error:
        return dict(status='NOT_ESTIMABLE',reason=str(error) or 'Unresolved model phase',entry_s=entry)


def patient_statistic(model,rows,weights,*,subject='E1146',permutations=99999,seed=20261016):
    """Full-contact shuffles precede smoothing; reselect the best seizure each draw."""
    if model['status']!='OK':return dict(status='NOT_ESTIMABLE',reason=model['reason'])
    eligible=[r for r in rows if r['eligible']]
    if not eligible:return dict(status='NOT_ESTIMABLE',reason='No qualified clinical windows')
    value=np.asarray(model['db']);clinical=np.array([r['readouts']['db'] for r in eligible])
    if np.asarray(weights).shape!=(len(value),len(value)):raise ValueError('Complete fixed contact support required')
    if 'contact_names' in model and any(r['names']!=model['contact_names'] for r in eligible):
        raise ValueError('Clinical and model contact orders differ')
    fields=paper.unit_rows(smooth(clinical,weights));model_field=paper.unit_rows(smooth(value[None],weights))[0]
    r=fields@model_field;best=int(np.argmax(abs(r)))
    # Separate deterministic streams for different patients.
    rng=np.random.default_rng(np.random.SeedSequence([seed,*subject.encode('utf-8')]))
    perm=np.array([rng.permutation(len(value)) for _ in range(permutations)])
    null=abs(paper.unit_rows(smooth(value[perm],weights))@fields.T).max(1)
    match=dict(subject=subject,patient=subject,best_abs_r=float(abs(r[best])),signed_r=float(r[best]),
               null_median=float(np.median(null)),selected_seizure_number=eligible[best]['seizure_number'],
               n_seizures=len(eligible),n_channels=len(value))
    return dict(status='OK',match=match,null=null,permutations=permutations,
                model_window=model,observable='dB without MAD division',unit='patient')


def cohort_statistic(patients):
    """Use one selected match per patient and the median across patients in each draw."""
    rows=[p for p in patients if p['status']=='OK']
    if not rows:return dict(status='NOT_ESTIMABLE',reason='; '.join(p.get('reason','Unavailable') for p in patients))
    if len({p['match']['subject'] for p in rows})!=len(rows):raise ValueError('Repeated patient')
    matches=[p['match'] for p in rows];observed=np.array([r['best_abs_r'] for r in matches])
    null=np.median(np.stack([p['null'] for p in rows]),axis=0)
    return dict(status='OK',matches=matches,n_patients=len(rows),n_seizures=sum(r['n_seizures'] for r in matches),
                observed_median=float(np.median(observed)),p=float((1+np.sum(null>=np.median(observed)-1e-12))/(len(null)+1)),
                permutations=len(null),unit='patient',observable='dB without MAD division')

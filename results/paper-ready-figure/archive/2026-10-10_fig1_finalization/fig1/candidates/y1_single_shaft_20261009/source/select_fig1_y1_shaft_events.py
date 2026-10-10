"""Find illustrative Y1 events on one contiguous A-shaft montage.

Selection is for the waveform/spectrum illustration only; C-F are untouched.
Uses participation masks and physical channel order, then the unchanged Figure 1
spectrogram/centroid kernel on real EDF snippets. No channel-specific time shifts.
"""
from pathlib import Path
import argparse
import sys, json, re
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures.fig1_spectrogram_utils import compute_group_event_spectrogram_stack, centroid_alignment_audit

OUT = ROOT / 'results/paper-ready-figure/fig1/candidates/y1_single_shaft_20261009'
DATA = Path('/mnt/yuquan_data/yuquan_24h_edf/zhangkexuan')


def compute_independent_event_spectra(signals, fs=1000., duration=.32):
    """Apply the existing kernel independently to discontinuous EDF excerpts.

    Joining unrelated excerpts before STFT introduces synthetic boundary power
    into each event's normalization. Keep each excerpt's spectrum and centroid
    intact; concatenate only their display coordinates.
    """
    samples = int(round(fs * duration))
    assert signals.shape[1] == 3 * samples
    spectra, times, centers = [], [], []
    for event in range(3):
        values = signals[:, event*samples:(event+1)*samples]
        spec, t, freqs, c = compute_group_event_spectrogram_stack(values, fs, np.array([duration]))
        spectra.append(spec)
        times.append(t + event*duration)
        c[:, :, 0] += event*duration
        centers.append(c)
    return np.concatenate(spectra, axis=1), np.concatenate(times), freqs, np.concatenate(centers, axis=1)


def read_snippet(raw, channels, center):
    fs = float(raw.info['sfreq'])
    names = [re.sub(r'-(Ref|REF)$', '', re.sub(r'^(EEG|POL)\s+', '', n.strip())).replace(' ', '') for n in raw.ch_names]
    pairs = [(n, f'A{int(n[1:])+1}') for n in channels]
    picks = sorted({names.index(n) for pair in pairs for n in pair})
    mapping = {pick: row for row,pick in enumerate(picks)}
    first = int(round((center-.16)*fs)); last = first+int(round(.32*fs))
    data = raw.get_data(picks=picks, start=first, stop=last)
    values = np.array([data[mapping[names.index(a)]]-data[mapping[names.index(b)]] for a,b in pairs])
    if fs != 1000: values = resample_poly(values, 1000, int(round(fs)), axis=-1)
    for hz in (50,100,150,200,250):
        b,a = iirnotch(hz, Q=30, fs=1000); values = filtfilt(b,a,values,axis=-1)
    b,a = butter(3, [.16,.50], btype='bandpass'); values=filtfilt(b,a,values,axis=-1)
    return values, [first,last]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-examples', action='store_true',
                        help='Reproduce the three visually suitable A3-A9 examples from the initial screen')
    parser.add_argument('--cached-assessment', action='store_true',
                        help='Rebuild the selected display from saved real-EDF snippets')
    args = parser.parse_args()
    rows=json.loads((OUT/'source/event_scan.json').read_text())
    short=[]
    for first in range(2,7):
        group=[r for r in rows if r['channels'][0]==f'A{first}' and len(r['channels'])==7]
        short += group[:20]
    if args.selected_examples:
        examples = [('FA134AX6', 1562), ('FA134AXJ', 292), ('FA134AXF', 1475)]
        short = [next(r for r in rows if (r['record'], r['event_index']) == key
                      and r['channels'][0] == 'A3' and len(r['channels']) == 7) for key in examples]
    raw_cache={}; assessed=[]; snippets={}
    if args.cached_assessment:
        assessed=json.loads((OUT/'source/raw_event_assessment.json').read_text())
        with np.load(OUT/'source/candidate_snippets.npz') as cache:
            snippets={key:cache[key] for key in cache.files}
        short=[]
    try:
        for index,row in enumerate(short):
            record=row['record']
            if record not in raw_cache:
                raw_cache[record]=mne.io.read_raw_edf(DATA/f'{record}.edf', preload=False, encoding='latin1', verbose='ERROR')
            raw=raw_cache[record]
            center=float(np.mean(row['window']))
            values,bounds=read_snippet(raw,row['channels'],center)
            spec,t,f,c=compute_group_event_spectrogram_stack(values,1000,np.array([.32]))
            # One common recentering of the entire event, never per-channel shifts.
            shift=float((c[:,0,0].min()+c[:,0,0].max())/2-.16)
            center += shift
            values,bounds=read_snippet(raw,row['channels'],center)
            spec,t,f,c=compute_group_event_spectrogram_stack(values,1000,np.array([.32]))
            times=c[:,0,0]-.16
            span=float(np.ptp(times)*1000)
            rho=float(spearmanr(np.arange(len(times)),times).statistic)
            active_strength=np.max(np.abs(values),axis=1)/np.maximum(np.median(np.abs(values),axis=1),1e-12)
            eligible=bool(np.max(np.abs(times))<.038 and 20<span<70 and abs(rho)>.70)
            try:
                audit=centroid_alignment_audit(spec,t,f,c,.32,.70)
            except ValueError as exc:
                audit=dict(all_centroids_pass=False, rejection_reason=str(exc))
                eligible=False
            row={**row,'crop_center_sec':center,'crop_sample_bounds':bounds,'sample_rate_in':float(raw.info['sfreq']),
                 'display_centroid_ms':(times*1000).tolist(),'display_span_ms':span,'display_rho':rho,
                 'min_peak_to_median_abs':float(active_strength.min()),'centroid_audit_pass':audit['all_centroids_pass'],
                 'eligible_for_display':eligible,'display_score':span*abs(rho)}
            assessed.append(row)
            if eligible and audit['all_centroids_pass']:
                key=f"{record}_{row['event_index']}_{row['channels'][0]}"
                snippets[key]=values;row['cache_key']=key
            (OUT/'source/raw_event_assessment.json').write_text(json.dumps(assessed,indent=2))
            np.savez_compressed(OUT/'source/candidate_snippets.npz',**snippets)
            print(index+1,len(short),record,row['event_index'],row['channels'][0],round(span,1),round(rho,2),eligible,flush=True)
    finally:
        for raw in raw_cache.values():raw.close()
    (OUT/'source/raw_event_assessment.json').write_text(json.dumps(assessed,indent=2))
    np.savez_compressed(OUT/'source/candidate_snippets.npz',**snippets)
    eligible=[r for r in assessed if r.get('cache_key')]
    groups=[]
    for first in range(2,7):
        for sign in (-1,1):
            group=sorted([r for r in eligible if r['channels'][0]==f'A{first}' and np.sign(r['display_rho'])==sign],key=lambda r:-r['display_score'])
            if len(group)>=3: groups.append((sum(r['display_score'] for r in group[:3]),group[:3]))
    groups.sort(key=lambda item:-item[0])
    assert groups,'No three clear events on one montage'
    selected=groups[0][1]
    signals=np.concatenate([snippets[r['cache_key']] for r in selected],axis=1)
    spec,t,f,c=compute_independent_event_spectra(signals)
    audit=centroid_alignment_audit(spec,t,f,c,.32,.70)
    assert audit['all_centroids_pass']
    offsets=c[:,:,0]-np.array([.16,.48,.80])[None,:]
    assert np.max(np.abs(offsets))<.04
    selection=dict(patient='Y1',subject='zhangkexuan',shaft='A',channels=selected[0]['channels'],
      bipolar_channels=[f'{n}-A{int(n[1:])+1}' for n in selected[0]['channels']],events=selected,
      method='Participation-masked contiguous montage; real-EDF spectrogram screening; common event recentering only',
      spectrum_rule='Unchanged spectrum kernel applied independently to each discontinuous excerpt; no synthetic-join power in normalization',
      selection_scope='Three frozen illustrative examples reproduced after the exploratory screen' if args.selected_examples else 'Top 20 artifact candidates per montage screened',
      evaluated_candidates=len(assessed),eligible_candidates=len(eligible),centroid_audit=audit,
      concatenated_centroid_offsets_ms=(offsets*1000).tolist(),purpose='Illustration only; no cohort inference or C-F event changes')
    (OUT/'source/selection.json').write_text(json.dumps(selection,indent=2))
    np.savez_compressed(OUT/'source/selected_signals.npz',signals_V=signals,specs=spec,times=t,freqs=f,centers=c,
                        channels=np.array(selection['bipolar_channels']))
    print('SELECTED',selection['channels'],[(r['record'],r['event_index'],r['display_span_ms']) for r in selected],flush=True)


if __name__=='__main__':main()

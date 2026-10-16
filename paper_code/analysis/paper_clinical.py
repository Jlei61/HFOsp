"""Numerical functions copied from the paper Fig5 phase/statistic producers."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d
BANDS={'delta':(1.,4.),'theta':(4.,8.),'alpha':(8.,13.),'beta':(13.,30.),'gamma':(30.,80.),'high_gamma':(80.,150.)}

@dataclass(frozen=True)
class TimingResult:
    detected: bool
    rise_sec: float
    peak_sec: float
    peak_value: float
    threshold: float
    baseline_median: float
    baseline_q: float
    total_above_sec: float
    longest_above_sec: float
    n_baseline_frames: int
    n_search_frames: int

def smooth_trace(trace: np.ndarray, rel_t: np.ndarray, *, smooth_sec: float = 2.0) -> np.ndarray:
    """NaN-aware centered moving average on the native time grid."""
    y = np.asarray(trace, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.size != t.size:
        raise ValueError("trace and rel_t must be matching 1D arrays")
    if y.size < 2 or float(smooth_sec) <= 0.0:
        return y.copy()
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rel_t must be strictly increasing")
    width = max(1, int(round(float(smooth_sec) / dt)))
    if width <= 1:
        return y.copy()
    kernel = np.ones(width, dtype=float)
    finite = np.isfinite(y)
    num = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(num, den, out=np.full_like(y, np.nan), where=den > 0)

def _run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive runs of True values."""
    x = np.asarray(mask, dtype=bool)
    padded = np.r_[False, x, False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]

def detect_sustained_enhancement(
    trace: np.ndarray,
    rel_t: np.ndarray,
    *,
    baseline: tuple[float, float] = (-120.0, -90.0),
    search: tuple[float, float] = (-60.0, 20.0),
    baseline_quantile: float = 0.99,
    sustain_sec: float = 2.0,
) -> TimingResult:
    """Detect the first sustained search-window excursion above baseline.

    ``trace`` is expected to be the already-smoothed, distal-baseline-referenced
    spatial summary.  The threshold is the empirical quantile of the same
    trace in the distal baseline.  A crossing is accepted only when it remains
    above threshold for at least ``sustain_sec`` on the native grid.
    """
    y = np.asarray(trace, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.size != t.size:
        raise ValueError("trace and rel_t must be matching 1D arrays")
    if y.size < 2:
        raise ValueError("at least two time frames are required")
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rel_t must be strictly increasing")
    bmask = (t >= float(baseline[0])) & (t < float(baseline[1])) & np.isfinite(y)
    smask = (t >= float(search[0])) & (t <= float(search[1])) & np.isfinite(y)
    if not np.any(bmask):
        raise ValueError(f"no finite frames in distal baseline {baseline}")
    if not np.any(smask):
        raise ValueError(f"no finite frames in search window {search}")

    bvals = y[bmask]
    threshold = float(np.nanquantile(bvals, float(baseline_quantile)))
    baseline_median = float(np.nanmedian(bvals))
    search_idx = np.flatnonzero(smask)
    above_local = y[search_idx] > threshold
    min_frames = max(1, int(np.ceil(float(sustain_sec) / dt)))
    runs = _run_lengths(above_local)
    accepted = [(a, b) for a, b in runs if (b - a) >= min_frames]
    detected = bool(accepted)
    rise_sec = float(t[search_idx[accepted[0][0]]]) if detected else float("nan")

    search_y = y[search_idx]
    peak_local = int(np.nanargmax(search_y))
    peak_sec = float(t[search_idx[peak_local]])
    peak_value = float(search_y[peak_local])
    total_above_sec = float(sum(b - a for a, b in runs) * dt)
    longest_above_sec = float(max((b - a for a, b in runs), default=0) * dt)
    return TimingResult(
        detected=detected,
        rise_sec=rise_sec,
        peak_sec=peak_sec,
        peak_value=peak_value,
        threshold=threshold,
        baseline_median=baseline_median,
        baseline_q=float(baseline_quantile),
        total_above_sec=total_above_sec,
        longest_above_sec=longest_above_sec,
        n_baseline_frames=int(np.sum(bmask)),
        n_search_frames=int(np.sum(smask)),
    )

def line_noise_bin_mask(freqs, harmonics_hz, halfwidth_hz):
    freqs = np.asarray(freqs, float); m = np.zeros(freqs.shape, bool)
    for h in harmonics_hz: m |= np.abs(freqs - float(h)) <= float(halfwidth_hz)
    return m

def band_bin_selection(freqs, lo, hi, line_mask, half_open=False):
    freqs = np.asarray(freqs, float)
    in_band = (freqs >= float(lo)) & ((freqs < float(hi)) if half_open else (freqs <= float(hi)))
    n_band = int(in_band.sum())
    band_mask = in_band & ~np.asarray(line_mask, bool)
    return band_mask, float(band_mask.sum()) / max(n_band, 1), n_band

def features(e, a):
    t = a['time_s']; base = e['baseline_s']
    keep = (t - .5 >= base[0] - 1e-8) & (t + .5 <= base[1] + 1e-8)
    names = a['names'].tolist()
    gate = e['gate_contact_names']
    ix = [names.index(n) for n in gate]
    lp = np.stack([a['gate_' + b][ix] for b in BANDS])
    med = np.median(lp[:, :, keep], axis=2)
    mad = 1.4826 * np.median(abs(lp[:, :, keep] - med[:, :, None]), axis=2)
    z = (lp - med[:, :, None]) / np.maximum(mad[:, :, None], 1e-15)
    db = (lp - med[:, :, None]) * 10 / np.log(10)
    # Smooth baseline independently so surrounding activity cannot inflate it.
    zq = np.quantile(z, .75, axis=1)
    smooth = np.stack([smooth_trace(v, t, smooth_sec=2.) for v in zq])
    bs = np.stack([smooth_trace(np.where(keep, v, np.nan), t, smooth_sec=2.) for v in zq])
    threshold = np.nanquantile(bs[:, keep], .99, axis=1)
    smooth[:,keep]=bs[:,keep]
    above = smooth > threshold[:, None]
    joint = (above.sum(0) >= 5) & (above[:3].sum(0) >= 2) & (above[3:].sum(0) >= 2)
    dbq = np.quantile(db, .75, axis=1)
    return dict(t=t, keep=keep, gate=gate, lp=lp, db=db, dbq=dbq,
                smooth=smooth, threshold=threshold, above=above, joint=joint)

def locate_phase(e,a):
    f = features(e, a); t = f['t']
    lo = max(t[0] + .5, e['eeg_onset_s'] - 15.)
    hi = min(t[-1] - .5, e['eeg_onset_s'] + 60., e['eeg_offset_s'])
    use = (t >= lo) & (t <= hi)
    bouts = []
    for left, right in _run_lengths(f['joint'] & use):
        if (right - left) * .5 < 2.: continue
        # Report the first ten seconds or the complete shorter joint bout.
        end = min(right, left + 20)
        vals = np.median(f['dbq'][:, left:end], axis=1)
        strength = min(np.sort(vals)[1], np.sort(vals[:3])[1], np.sort(vals[3:])[1])
        bouts.append(dict(start_s=float(t[left]-.5), end_s=float(t[right-1]+.5),
                          duration_s=float((right-left)*.5), band_db=vals.tolist(),
                          five_band_strength_db=float(strength),
                          evaluation_s=[float(t[left]-.5), float(t[end-1]+.5)]))
    # An explanatory strongest visible bout is NOT the selected statistical event.
    best = max(bouts, key=lambda b: b['five_band_strength_db']) if bouts else None
    # Date the rise with a five-band consensus, never summed1–150Hz power.
    # Keep actual dB amplitudes: normalized profiles do not prove that a weak
    # or noisy band supplies valid physiological recruitment evidence.
    dbsm=uniform_filter1d(f['dbq'],4,axis=1,mode='nearest')
    highuse=use&(t>=e['eeg_onset_s'])
    high=np.quantile(dbsm[:,highuse],.9,axis=1)
    unit=np.clip(dbsm/np.maximum(high[:,None],1e-9),0,2)
    consensus=np.minimum.reduce([np.sort(unit,axis=0)[1],np.sort(unit[:3],axis=0)[1],np.sort(unit[3:],axis=0)[1]])
    eligible_high=highuse&f['joint']
    if eligible_high.any():consensus=consensus/max(float(np.quantile(consensus[eligible_high],.9)),1e-9)
    highruns=[(a,b) for a,b in _run_lengths((consensus>=.9)&use&f['joint']) if (b-a)*.5>=2 and np.any(consensus[a:b]>=1.)]
    multiband_phase=None
    if highruns:
        h0,h1=highruns[0]
        before=np.flatnonzero(use&(t<t[h0])&(consensus<.1))
        full=np.flatnonzero((np.arange(len(t))>=h0)&(np.arange(len(t))<h1)&(consensus>=1))
        if len(before) and len(full):
            start_i=int(before[-1]+1);full_i=int(full[0]);end=min(float(t[full_i]+2.5),e['eeg_offset_s'],float(t[-1]+.5))
            start=float(t[start_i]-.5);capture=(t-.5>=start-1e-8)&(t+.5<=end+1e-8)
            flags=[]
            if any(b['end_s']<start-2 for b in bouts):flags.append('EARLIER_MULTIBAND_BOUT')
            if start<=lo+1:flags.append('RISE_NEAR_SEARCH_BOUNDARY')
            if np.any(high<3):flags.append('AT_LEAST_ONE_BAND_HAS_LESS_THAN_3DB_HIGH_REFERENCE_DESCRIPTIVE_ONLY')
            multiband_phase=dict(window_s=[start,end],high_start_s=float(t[h0]),full_high_s=float(t[full_i]),
                                 flags=flags,band_db=np.median(f['dbq'][:,capture],axis=1).tolist(),
                                 high_band_db=high.tolist(),status='PROPOSAL_REQUIRES_MORPHOLOGY_REVIEW')
    return multiband_phase, f


def qualify(a,e,window):
    t=a['time_s'];keep=np.zeros(len(t),bool);keep[e['baseline']['indices']]=True
    lo,hi=window;on=(t-.5>=lo-1e-8)&(t+.5<=hi+1e-8)
    gate=e['gate_contact_names'];ix=[list(a['names']).index(n) for n in gate]
    hits=[]
    for band in BANDS:
        p=a['gate_'+band][ix];med=np.median(p[:,keep],1);mad=1.4826*np.median(abs(p[:,keep]-med[:,None]),1)
        z=(p-med[:,None])/np.maximum(mad,1e-15)[:,None];trace=np.quantile(z,.75,axis=0)
        sm=smooth_trace(np.where(keep|on,trace,np.nan),t,smooth_sec=2.)
        sm=np.where(keep|on,sm,np.nan)
        hit=detect_sustained_enhancement(sm,t,baseline=tuple(e['baseline_clinical_s']),search=(lo+.5,hi-.5),baseline_quantile=.99,sustain_sec=2.)
        hits.append(dict(band=band,hit=bool(hit.detected)))
    low=sum(h['hit'] for h in hits[:3]);fast=sum(h['hit'] for h in hits[3:])
    return dict(eligible=low+fast>=5 and low>=2 and fast>=2,low_hits=low,fast_hits=fast,total_hits=low+fast,bands=hits)

def contrast(p,t,keep,lo,hi):
    mask=(t-.5>=lo-1e-9)&(t+.5<=hi+1e-9)
    if not mask.any():return None
    med=np.median(p[:,keep],axis=1);mad=1.4826*np.median(abs(p[:,keep]-med[:,None]),axis=1)
    assert (mad>1e-12).all()
    mean=p[:,mask].mean(1)
    return dict(z=(mean-med)/mad,db=(mean-med)*10/np.log(10),start_s=float(lo),end_s=float(hi),
                n_complete_frames=int(mask.sum()),first_center=float(t[mask][0]),last_center=float(t[mask][-1]))

def unit_rows(values):
    values = np.atleast_2d(np.asarray(values, float))
    centered = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 1e-12).any():
        raise ValueError('Non-finite or constant field')
    return centered / norm

def phase_windows(t, rates, high_onset, old_onset):
    """Times denote ends of 10-ms rate bins. No clinical values are inputs."""
    np.testing.assert_allclose(np.diff(t), .01, atol=1e-8)
    quiet = [(a, b) for a, b in _run_lengths(rates[:, 0] < 5)
             if b - a >= 2 and t[b - 1] < high_onset - 1e-8]
    assert quiet, "No connected-episode anchor"
    episode = float(t[quiet[-1][1] - 1])
    sm = uniform_filter1d(rates[:, 0], 5, mode="nearest")
    # A later observed plateau, not the theoretical refractory maximum.
    tail = (t >= high_onset + 1.5 - 1e-8) & (t < high_onset + 2.5 - 1e-8)
    assert tail.sum() >= 99, "Incomplete plateau reference"
    plateau = float(np.median(sm[tail]))
    high = [(a, b) for a, b in _run_lengths((sm >= .9 * plateau) & (t > high_onset)) if b - a >= 10]
    assert high, "No sustained90%-plateau crossing"
    t90 = float(t[high[0][0]] - .01)
    low = [(a, b) for a, b in _run_lengths((sm < .1 * plateau) & (t < high_onset)) if b - a >= 5]
    assert low, "No final10%-plateau crossing"
    t10 = max(episode, float(t[low[-1][1] - 1]))
    windows = dict(old=[old_onset, old_onset + 1],
        episode_to_high=[episode, high_onset],
        episode_to_plateau=[episode, t90 + .1],
        final_ramp=[t10, t90 + .1], episode_first100ms=[episode, episode + .1],
        figure5_state2_to5=[episode, high_onset + .5],
        episode_to_high_plus100ms=[episode, high_onset + .1],
        terminal_rise_early=[t10, high_onset + .1])
    assert all(hi - lo >= .1 - 1e-8 for lo, hi in windows.values())
    return dict(episode_start_s=episode, final_ramp_start_s=t10, plateau90_s=t90,
        plateau_rate_hz=plateau, plateau_reference_s=[high_onset + 1.5, high_onset + 2.5],
        high_onset_s=high_onset, old_onset_s=old_onset, windows=windows)

def anchor(t,rate,entry,plateau):
    # Last20%-level crossing before sustained high entry: earlier independent
    # excursions are excluded, and there is no pre-rise padding.
    smooth=uniform_filter1d(rate,5,mode='nearest')
    below=np.flatnonzero((t<entry)&(smooth<.2*plateau))
    assert len(below)>0
    idx=int(below[-1]+1);start=float(t[idx])
    assert start<=entry and np.all(smooth[idx:np.searchsorted(t,entry)]>=.2*plateau)
    reached=np.flatnonzero((t>entry)&(smooth>=plateau))
    stable=[int(i) for i in reached if len(smooth[i:i+10])==10 and np.all(abs(smooth[i:i+10]/plateau-1)<=.05)]
    assert stable
    full=float(t[stable[0]])
    return start,full,smooth

def baseline_windows(a,counts,onset_s,window_s=.1):
    rates=a['spikes_10ms']/counts[None]/.01;t=a['time_s']
    quiet=(rates[:,0]<5.)&(rates[:,1:3].max(1)<20.)&(rates[:,3]<5.)
    starts=[];limit=int(round(3/window_s))
    for s in np.arange(.5,onset_s-window_s-.15,window_s):
        mask=(t>s-.02)&(t<=s+window_s+.02)
        if mask.sum()>=round(window_s/.01) and np.all(quiet[mask]):starts.append(float(s))
        if len(starts)>=limit:break
    return starts

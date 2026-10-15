"""Native SEEG -> referenced/filtered signals -> detected events -> masked ranks.

All arrays are computed from samples. No event/rank/label cache is read.
Times are seconds until the explicit centroid_ms export for the SNN objective.
"""
from pathlib import Path
import json
from fractions import Fraction
import numpy as np
from scipy import signal, ndimage
from data_processing.src.hfo_detector import HFODetectionConfig, HFODetector
from data_processing.src.group_event_analysis import (
    build_windows_from_detections, compute_stitched_spectrogram_centroids_legacy,
    lag_rank_from_centroids,
)


def read_samples(path, *, start_sec=0, duration_sec=200, channels=None):
    """NPZ stores unreferenced samples in V, sfreq and channel_names; EDF via MNE.

    Epilepsiae .data/.head: use the copied load_epilepsiae_block adapter with
    its explicit CAR/reference choice, then call preprocess with reference="none" and notch_freqs=[].
    """
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            x, fs, names = z['samples_V'].astype(float), float(z['sfreq']), z['channel_names'].astype(str)
        i = int(round(start_sec * fs)); j = min(len(x.T), i + int(round(duration_sec * fs)))
        x = x[:, i:j]
        if channels is not None:
            index = [list(names).index(n) for n in channels]; x, names = x[index], names[index]
    else:
        import mne
        with mne.io.read_raw_edf(path, preload=False, encoding='latin1', verbose='ERROR') as raw:
            fs = float(raw.info['sfreq']); names = np.asarray(raw.ch_names)
            import re
            names = np.array([re.sub(r'-(Ref|REF)$', '', re.sub(r'^(EEG|POL)\s+', '', n.strip())).replace(' ', '') for n in names])
            picks = list(range(len(names))) if channels is None else [list(names).index(n) for n in channels]
            i = int(round(start_sec * fs)); j = min(raw.n_times, i + int(round(duration_sec * fs)))
            x, names = raw.get_data(picks=picks, start=i, stop=j), names[picks]
    if x.ndim != 2 or x.shape[1] < fs or not np.isfinite(x).all() or len(set(names)) != len(names):
        raise ValueError('Need finite, uniquely named channel x sample raw data with >=1 second')
    return x, fs, names


def preprocess(x, fs, names, config):
    """Explicit reference, polyphase resample, Q=30 IIR notch, order-3 bandpass.

    bipolar_pairs maps each output alias to two native contacts. Geometry in
    this tutorial follows the legacy left-contact alias, not the midpoint.
    CAR here is global over the supplied retained channels (specify these).
    """
    cfg = config['preprocessing']; names = list(names)
    drop=set(cfg.get('drop_channels',[]))
    if drop:
        keep=[i for i,name in enumerate(names) if name not in drop]
        if not keep:raise ValueError('No raw contacts remain after explicit exclusions')
        x=x[keep];names=[names[i] for i in keep]
    if cfg['reference'] == 'bipolar':
        pairs = cfg['bipolar_pairs']; out_names = list(pairs)
        x = np.stack([x[names.index(a)] - x[names.index(b)] for a, b in pairs.values()])
    elif cfg['reference'] == 'car':
        x = x - np.mean(x, axis=0, keepdims=True); out_names = names
    elif cfg['reference'] == 'none':
        out_names = names
    else:
        raise ValueError('reference must be bipolar/car/none')
    target = float(cfg['target_sfreq'])
    if target <= 2 * cfg['band_hz'][1]:
        raise ValueError('Target rate must exceed twice the upper filter frequency')
    ratio = Fraction(target / fs).limit_denominator(10000)
    x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
    for hz in cfg['notch_freqs']:
        if hz < target / 2:
            b, a = signal.iirnotch(hz, 30, target); x = signal.filtfilt(b, a, x, axis=1)
    broadband = x.copy()
    b, a = signal.butter(3, cfg['band_hz'], btype='bandpass', fs=target)
    x = signal.filtfilt(b, a, x, axis=1)
    return dict(band_V=x, broadband_V=broadband, sfreq=target, names=np.asarray(out_names))


def detect_groups(pre, config):
    detected = HFODetector(HFODetectionConfig(**config['detector'])).detect(
        pre['band_V'], sfreq=pre['sfreq'], ch_names=list(pre['names']))
    detections = dict(zip(detected.ch_names, detected.events_by_channel))
    counts = np.array([len(detections[n]) for n in pre['names']])
    # Legacy pack eligibility is patient-specific mean + k SD channel count.
    threshold = counts.mean() + config['packing']['pick_k'] * counts.std()
    selected = [n for n, c in zip(pre['names'], counts) if c > threshold]
    if not selected:
        raise ValueError('No pack-eligible channel; inspect raw data/configuration, do not invent events')
    opts = {k: v for k, v in config['packing'].items() if k != 'pick_k'}
    windows = build_windows_from_detections({n: detections[n] for n in selected},
        t_max_sec=pre['band_V'].shape[1]/pre['sfreq'], **opts)
    windows = [w for w in windows if w.start >= 0 and w.end <= pre['band_V'].shape[1]/pre['sfreq']]
    return dict(detections=detections, counts=counts, packing_channels=selected, windows=windows)


def event_features(pre, detected):
    """Concatenate newly detected windows, then full S^3 spectrogram centroid.

    The spectrogram is computed over the concatenation (50 ms Hamming,
    40 ms overlap, Gaussian sigma=1.5 before 50<f<300 cropping), as in Fig1B.
    Participation is detection overlap, never isfinite(legacy rank).
    """
    windows = detected['windows']; fs = pre['sfreq']
    if not windows:
        raise ValueError('No group events were detected')
    segments = [pre['band_V'][:, round(w.start*fs):round(w.end*fs)] for w in windows]
    lengths = np.array([s.shape[1] for s in segments]); borders = np.cumsum(lengths)/fs
    stitched = np.concatenate(segments, axis=1)
    centers = compute_stitched_spectrogram_centroids_legacy(stitched, borders, sfreq=fs)
    participating = np.zeros(centers.shape, bool)
    for ci, name in enumerate(pre['names']):
        times = np.asarray(detected['detections'][name]).reshape(-1, 2)
        for ei, w in enumerate(windows):
            participating[ci, ei] = np.any((times[:, 0] < w.end) & (times[:, 1] > w.start))
    participating &= np.isfinite(centers)
    centers[~participating] = np.nan
    lag, ranks = lag_rank_from_centroids(centers, participating)
    return dict(ranks=ranks, participating=participating, centroid_ms=lag*1000,
                channel_names=pre['names'], windows_sec=np.array([[w.start,w.end] for w in windows]),
                stitched_V=stitched, borders_sec=borders, stitched_centroid_sec=centers)


def event_spectrogram(pre, features, index):
    fs = pre['sfreq']; x = features['stitched_V']
    f, t, s = signal.spectrogram(x, fs=fs, window='hamming', nperseg=round(.05*fs),
                                noverlap=round(.04*fs), mode='magnitude', axis=-1)
    # Each channel independently; never smooth across channels.
    s = np.stack([ndimage.gaussian_filter(row, 1.5) for row in s])
    lo = 0 if index == 0 else features['borders_sec'][index-1]
    hi = features['borders_sec'][index]
    keep_f = (f > 50) & (f < 300); keep_t = (t > lo) & (t < hi)
    s = s[:, keep_f][:, :, keep_t]
    s /= np.maximum(s.max(axis=(1, 2), keepdims=True), 1e-30)
    return (t[keep_t]-lo)*1000, f[keep_f], s, (features['stitched_centroid_sec'][:,index]-lo)*1000


def process_record(record, config, input_root):
    samples, fs, names = read_samples(Path(input_root)/record['file'],
        start_sec=record.get('start_sec', 0), duration_sec=record.get('duration_sec', 200),channels=record.get('channels'))
    pre = preprocess(samples, fs, names, config)
    detected = detect_groups(pre, config); features = event_features(pre, detected)
    features['block_ids'] = np.full(features['ranks'].shape[1], record['block_id'])
    return pre, detected, features


def join_blocks(blocks):
    """Align by name before concatenation; absent contacts remain nonparticipants."""
    names = sorted(set().union(*(set(b['channel_names']) for b in blocks)))
    n = sum(b['ranks'].shape[1] for b in blocks)
    out = dict(channel_names=np.asarray(names), ranks=np.full((len(names), n), -1),
               participating=np.zeros((len(names), n), bool), centroid_ms=np.full((len(names), n), np.nan),
               block_ids=np.concatenate([b['block_ids'] for b in blocks]))
    col = 0
    for block in blocks:
        index = [names.index(k) for k in block['channel_names']]; m = block['ranks'].shape[1]
        for key in ['ranks', 'participating', 'centroid_ms']:
            out[key][np.ix_(index, np.arange(col, col+m))] = block[key]
        col += m
    return out

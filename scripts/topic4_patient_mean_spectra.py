"""Display-only mean of frozen real patient spectra using the Fig2C algorithm.

Build in a separate process so the main checkout's patient producer cannot alter
the running SNN executor's imports. This packet is not a frequency estimator.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings
import numpy as np

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
PACKET = MAIN / '.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/multievent_distribution_search_v2_1/patient_time_packet'
OUT = Path('/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915/analysis/patient_mean_spectra')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    sys.path.insert(0, str(MAIN))
    from scripts import plot_topic5_interictal_event_envelope_field as canonical
    from scripts.paper_figures.fig1_spectrogram_utils import dominant_enhancement_centroids
    packet = json.loads((PACKET / 'packet_manifest.json').read_text())
    inventory = canonical._inventory('1146')
    OUT.mkdir(parents=True, exist_ok=True)
    events = []; names = None; freq = None
    for number, row in enumerate(packet['readable_events']):
        with np.load(row['arrays_path']) as old:
            cn = old['contact_names'].astype(str).tolist()
            part = old['participation_mask'].astype(bool)
            expected = old['fig1a_centroid_ms'].copy()
        if names is None: names = cn
        assert cn == names
        record = inventory[row['record_name']]
        lo, hi = row['packed_window_sec']
        packed_path = canonical.LAGPAT_ROOT / '1146/all_recs' / f"{row['record_name']}_packedTimes.npy"
        packed = np.load(packed_path)
        index = int(np.argmin(np.abs(packed[:, 0] - lo)))
        assert np.allclose(packed[index], [lo, hi], atol=1e-7, rtol=0)
        crop = max(lo - canonical.PAD_SEC, 0.)
        raw = canonical.load_epilepsiae_block(record['data_path'], record['head_path'], reference='car',
            segment_sec=None, crop_start_sec=crop, crop_duration_sec=hi-lo+2*canonical.PAD_SEC)
        x = raw.data[[raw.ch_names.index(n) for n in names]]
        t = np.arange(raw.data.shape[1]) / raw.sfreq + crop
        clean = canonical.notch_filt(x, raw.sfreq, [f for f in canonical.NOTCH_HZ if f < raw.sfreq/2])
        band = canonical.band_filt(clean, raw.sfreq, list(canonical.BAND))
        tile = band[:, (t >= lo) & (t < hi)]
        duration = tile.shape[1] / raw.sfreq
        stack, st, sf, centers = canonical.compute_group_event_spectrogram_stack(tile, raw.sfreq,
            np.asarray([duration]), spec_window='hamming', spec_win_sec=.05, spec_overlap_sec=.04,
            spec_freq_range=(50., 300.), gaussian_sigma=1.5, enhancement_threshold=.70)
        zero = float(np.nanmin(centers[part, 0, 0]))
        times = (centers[:, 0, 0] - zero) * 1000
        error = float(np.nanmax(np.abs(times[part] - expected[part])))
        assert error < 1e-3, (row['raw_global_event_index'], error)
        if freq is None: freq = sf
        assert np.array_equal(freq, sf)
        event = dict(row=row, spec=stack.reshape(len(names), len(sf), -1), time=(st-zero)*1000,
            lo=-zero*1000, hi=(hi-lo-zero)*1000, part=part, centers=times,
            frequencies=centers[:, 0, 1], centroid_replay_error_ms=error,
            raw_data_path=record['data_path'], raw_head_path=record['head_path'], sfreq=float(raw.sfreq))
        events.append(event)
        print(f"real spectrum {number+1}/{len(packet['readable_events'])}: {row['raw_global_event_index']}", flush=True)
    # Display interpolation only. Native STFT spacing is retained in provenance;
    # a 2 ms grid does not imply 2 ms spectral temporal resolution.
    grid = np.arange(np.floor(min(e['lo'] for e in events)/2)*2,
                     np.ceil(max(e['hi'] for e in events)/2)*2+1, 2.)
    common = (grid >= max(e['lo'] for e in events)) & (grid < min(e['hi'] for e in events))
    arrays = dict(contact_names=np.asarray(names), spec_t_ms=grid, spec_freq_hz=freq)
    meta = dict(schema='real_spectral_mean_v1', patient='E10 / Epilepsiae 1146',
        packet_path=str(PACKET / 'packet_manifest.json'), packet_sha256=sha(PACKET/'packet_manifest.json'),
        producer_path=str(Path(canonical.__file__)), producer_sha256=sha(canonical.__file__),
        builder_path=str(Path(__file__).resolve()), builder_sha256=sha(__file__),
        algorithm='Fig2C: CAR, original notch and bandpass, magnitude STFT 50 ms Hamming / 40 ms overlap, Gaussian sigma 1.5, per-event per-contact maximum normalization',
        aggregation='equal event weight among participating events; display only the common observed time domain of all 64 windows, so each contact denominator stays fixed; no post-average normalization',
        common_complete_domain_ms=grid[common][[0,-1]].tolist(),
        alignment='one common time translation per event by its earliest participating Fig2C spectral centroid; no channel-wise alignment or time warping',
        marker='dominant >=70%-of-peak connected-component centroid of the displayed mean spectrum; not mean of event centroids and not training lagPat centroid',
        display_grid_ms=2., native_stft_step_ms=float(np.diff(events[0]['time'])[0]),
        boundary='same outer-cell extension as Fig2C inside the actual packed window; outside the window is missing, never zero-filled',
        labels_used_for_training=False, balanced_packet_estimates_natural_frequency=False,
        full_FIT_template=False, modes={}, events=[])
    for mode, k in [('TA', 1), ('TB', 0)]:
        selected = [e for e in events if e['row']['mode'] == k]
        interpolated = []
        for e in selected:
            v = np.full((len(names), len(freq), len(grid)), np.nan)
            mask = (grid >= e['lo']) & (grid < e['hi'])
            for c in np.flatnonzero(e['part']):
                for f in range(len(freq)):
                    v[c, f, mask] = np.interp(grid[mask], e['time'], e['spec'][c, f])
            interpolated.append(v)
        values = np.stack(interpolated)
        counts = np.isfinite(values[:, :, 0]).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(values, axis=0)
        mean[:, :, ~common] = np.nan
        ct = []; cf = []
        for channel in mean:
            # Translate the template domain to the helper's positive tile axis.
            shifted = (grid-grid[0]+1)/1000
            center = dominant_enhancement_centroids(channel, shifted, np.array([shifted[-1]+.001]),
                edge_guard_sec=0., enhancement_threshold=.70)[0]
            ct.append(center[0]*1000+grid[0]-1); cf.append(center[1])
        arrays.update({f'{mode}_spec': mean.astype(np.float32), f'{mode}_counts': counts.astype(np.int16),
            f'{mode}_centroid_ms': np.asarray(ct), f'{mode}_centroid_freq_index': np.asarray(cf)})
        meta['modes'][mode] = dict(N=len(selected), participating_events_per_contact=np.sum([e['part'] for e in selected], axis=0).tolist(),
            peak_of_mean=float(np.nanmax(mean)))
    for e in events:
        meta['events'].append(dict(**e['row'], raw_data_path=e['raw_data_path'], raw_head_path=e['raw_head_path'],
            centroid_replay_error_ms=e['centroid_replay_error_ms'], sfreq=e['sfreq']))
    tmp = OUT/'mean_spectra.tmp.npz'; np.savez_compressed(tmp, **arrays); tmp.replace(OUT/'mean_spectra.npz')
    meta['arrays_sha256'] = sha(OUT/'mean_spectra.npz')
    (OUT/'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(meta['modes'], ensure_ascii=False), flush=True)


def build_envelope():
    """Explicit fallback from cached real HFO envelopes, never a fake spectrum."""
    packet = json.loads((PACKET/'packet_manifest.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []; names = None
    for row in packet['readable_events']:
        with np.load(row['arrays_path']) as z:
            cn = z['contact_names'].astype(str).tolist()
            if names is None: names = cn
            assert names == cn
            mask = z['packed_window_mask'].astype(bool)
            t = z['time_ms'][mask].astype(float)
            v = z['positive_envelope_mass'][:, mask].astype(float)
            v /= np.maximum(v.max(axis=1, keepdims=True), 1e-20)
            part = z['participation_mask'].astype(bool)
        samples.append(dict(time=t, value=v, part=part, row=row))
    grid = np.arange(np.floor(min(e['time'][0] for e in samples)/2)*2,
                     np.ceil(max(e['time'][-1] for e in samples)/2)*2+1, 2.)
    common = (grid >= max(e['time'][0] for e in samples)) & (grid <= min(e['time'][-1] for e in samples))
    arrays = dict(contact_names=np.asarray(names), time_ms=grid)
    meta = dict(schema='real_envelope_mean_v2_common_domain', kind='cached HFO envelope, not STFT',
        N_total=len(samples), packet_sha256=sha(PACKET/'packet_manifest.json'),
        alignment='original packet: one common earliest participating Fig2C spectral-centroid translation per event; no channel-wise shifts or time warping',
        normalization='positive robust-z HFO envelope / within-contact-event maximum, before averaging; no post-average renormalization',
        averaging='equal event weight among participating events; display only the common observed time domain of all 64 windows, so each contact denominator stays fixed; gray elsewhere; not natural mode frequency',
        common_complete_domain_ms=grid[common][[0,-1]].tolist(),
        marker='magnitude-weighted centroid of connected >=70%-peak interval containing maximum of the mean envelope; not the mean of individual centroids',
        full_FIT_template=False, modes={}, source_events=packet['readable_events'],
        raw_spectrum_status='PENDING_RAW_DATA_ACCESS', builder_sha256=sha(__file__))
    for mode,k in [('TA',1),('TB',0)]:
        selected=[e for e in samples if e['row']['mode']==k]; values=[]
        for e in selected:
            v=np.full((len(names),len(grid)),np.nan)
            for c in np.flatnonzero(e['part']):
                v[c]=np.interp(grid,e['time'],e['value'][c],left=np.nan,right=np.nan)
            values.append(v)
        values=np.stack(values)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning); mean=np.nanmean(values,axis=0)
        mean[:,~common]=np.nan
        centers=[]
        for v in mean:
            if not np.isfinite(v).any(): centers.append(np.nan);continue
            peak=int(np.nanargmax(v)); lo=hi=peak; threshold=v[peak]*.7
            while lo>0 and v[lo-1]>=threshold: lo-=1
            while hi+1<len(v) and v[hi+1]>=threshold: hi+=1
            centers.append(float(np.average(grid[lo:hi+1],weights=v[lo:hi+1])))
        arrays.update({f'{mode}_envelope':mean.astype(np.float32),f'{mode}_counts':np.isfinite(values).sum(0).astype(np.int16),f'{mode}_centroid_ms':np.asarray(centers)})
        meta['modes'][mode]=dict(N=len(selected),participating_events_per_contact=np.sum([e['part'] for e in selected],axis=0).tolist())
    tmp=OUT/'mean_envelopes.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(OUT/'mean_envelopes.npz')
    meta['arrays_sha256']=sha(OUT/'mean_envelopes.npz')
    (OUT/'envelope_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(meta['modes'],ensure_ascii=False),flush=True)


def identity():
    if (OUT/'manifest.json').exists() and (OUT/'mean_spectra.npz').exists():
        return dict(kind='real_spectral_mean', manifest=str(OUT/'manifest.json'), sha256=sha(OUT/'mean_spectra.npz'))
    return dict(kind='cached_real_HFO_envelope_mean_not_spectrum',manifest=str(OUT/'envelope_manifest.json'),sha256=sha(OUT/'mean_envelopes.npz'))


def draw(ax, mode, xlim, display):
    """Fig2C spectrum cells + enhancement centroid; fixed grouped contact rows."""
    import matplotlib.pyplot as plt
    if not (OUT/'mean_spectra.npz').exists():
        with np.load(OUT/'mean_envelopes.npz') as z:
            order=display.contact_indices(z['contact_names'].astype(str).tolist())
            t=z['time_ms'];mean=z[f'{mode}_envelope'][order];ct=z[f'{mode}_centroid_ms'][order]
        meta=json.loads((OUT/'envelope_manifest.json').read_text());n=meta['modes'][mode]['N']
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#777777')
        edges=np.r_[t[0]-1,(t[:-1]+t[1:])/2,t[-1]+1]
        mesh=ax.pcolormesh(edges,np.arange(16)-.5,mean,cmap=cmap,vmin=0,vmax=1,shading='flat',rasterized=True)
        display.centroid_lines(ax,ct,color='#B2182B' if mode=='TA' else '#2166AC')
        for boundary in np.arange(.5,14.5,1): ax.axhline(boundary,color='white',ls='--',lw=.35,alpha=.4)
        display.contact_axis(ax)
        ax.set(xlim=xlim,xlabel='相对每事件最早谱质心 (ms)',title=f'患者{mode}：平均HFO包络，n={n}\n非频谱；仅显示共同可观测时段')
        return mesh
    with np.load(OUT/'mean_spectra.npz') as z:
        names = z['contact_names'].astype(str).tolist(); order = display.contact_indices(names)
        t = z['spec_t_ms']; spec = z[f'{mode}_spec'][order]; nf = len(z['spec_freq_hz'])
        centers = z[f'{mode}_centroid_ms'][order]; freqs = z[f'{mode}_centroid_freq_index'][order]
    meta = json.loads((OUT/'manifest.json').read_text()); n = meta['modes'][mode]['N']
    cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('#777777')
    edges = np.r_[t[0]-1, (t[:-1]+t[1:])/2, t[-1]+1]
    mesh = ax.pcolormesh(edges, np.arange(15*nf+1)/nf-.5, np.concatenate(spec,axis=0),
        cmap=cmap, vmin=0, vmax=1, shading='flat', rasterized=True)
    y = np.arange(15)-.5+(freqs+.5)/nf
    display.centroid_lines(ax, centers, y, color='#B2182B' if mode=='TA' else '#2166AC')
    for boundary in np.arange(.5,14.5,1): ax.axhline(boundary,color='white',ls='--',lw=.35,alpha=.4)
    display.contact_axis(ax)
    ax.set(xlim=xlim,xlabel='相对每事件最早谱质心 (ms)',title=f'患者{mode}：真实频谱平均，n={n}\nFig2C样式；点为平均谱主增强质心')
    return mesh


if __name__ == '__main__':
    build_envelope() if '--cached-envelope' in sys.argv else build()

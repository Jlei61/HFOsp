"""Read native bursts, separate input sources, and export transparent tables."""
from pathlib import Path
import sys,json,csv
import numpy as np
from scipy.signal import correlate,welch
ROOT=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(ROOT/'scripts/topic4_burst_regime'))
import metrics_v2 as metric
OUT=ROOT/'results/topic4_sef_hfo/core_burst_onset_brunel_v1_20260915'
OLD=ROOT/'results/topic4_sef_hfo/burst_regime_map_20260914'

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def window(z,group,lo,hi):
    names=z['group_names'].tolist();j=names.index(group);n=int(z['group_sizes'][j])
    m=metric.amend(metric.v1.summarize(z['active_counts_10ms'][round(lo*100):round(hi*100),j]/n,
        z['spike_counts_2ms'][round(lo*500):round(hi*500),j],n,burnin_s=0.))
    peaks=[]
    for e in m['events']:
        a=round((lo+e['start_s'])*500);b=round((lo+e['stop_s'])*500)
        if b>a:peaks.append(float(np.max(z['active_counts_2ms'][a:b,j]/n)))
    m['median_peak_active_fraction_2ms']=float(np.median(peaks)) if peaks else None
    m['threshold_burst_counts']={str(th):metric.v1.summarize(z['active_counts_10ms'][round(lo*100):round(hi*100),j]/n,
        z['spike_counts_2ms'][round(lo*500):round(hi*500),j],n,burnin_s=0.,onset_threshold=th)['n_bursts'] for th in (.075,.1,.125)}
    return m

def event_peaks(z,m,group,offset=2.):
    j=z['group_names'].tolist().index(group);n=int(z['group_sizes'][j]);rate=z['spike_counts_2ms'][:,j]/n/.002
    rows=[]
    for e in m['events']:
        lo=round((offset+e['start_s'])*500);hi=round((offset+e['stop_s'])*500)
        if hi<=lo:continue
        rows.append(dict(onset_s=offset+e['start_s'],peak_rate_hz=float(rate[lo:hi].max()),
            peak_active_fraction_10ms=e['peak_fraction'],duration_s=e['duration_s']))
    return rows

def acf(rate,maxlag=1.2):
    x=np.asarray(rate,float);x-=x.mean()
    c=correlate(x,x,mode='full',method='fft')[len(x)-1:len(x)+round(maxlag*500)]
    return c/c[0] if c[0]>0 else np.full(len(c),np.nan)

def main():
    base=[]
    for folder in sorted((OLD/'per_run').iterdir()):
        if not (folder/'result.json').exists() or not (folder/'metrics_v2.json').exists():continue
        r=read(folder/'result.json')
        if not (r['noise'] and r['topology']==2511):continue
        ms=read(folder/'metrics_v2.json')
        with np.load(folder/'trajectory.npz') as z:
            for g in ('coreAE','coreBE'):
                m=ms[g]
                j=z['group_names'].tolist().index(g)
                rate=z['spike_counts_2ms'][1000:,j]/int(z['group_sizes'][j])/.002
                observed_mean=float(rate.mean());observed_std=float(rate.std(ddof=1))
                assert np.isclose(observed_mean,m['mean_rate_hz'])
                base.append(dict(run=r['name'],ee=r['ee'],depth=r['depth'],seed=r['seed'],group=g,
                    metric={k:v for k,v in m.items() if k!='events'},events=event_peaks(z,m,g),
                    unthresholded_rate_mean_hz=observed_mean,unthresholded_rate_std_hz=observed_std,
                    intervals_s=np.diff([e['start_s'] for e in m['events'] if not e['left_censored']]).tolist()))
    assert len(base)==100
    write(OUT/'baseline_observables.json',base)
    comparisons=[];validation=[];cellstats=[];pulse=[];recovery=[]
    quiet=read(OUT/'quiescent_branch.json') if (OUT/'quiescent_branch.json').exists() else None
    for folder in sorted((OUT/'per_run').glob('ee*')):
        if not (folder/'result.json').exists():continue
        r=read(folder/'result.json');assert all(r['prefix_bitwise'].values())
        with np.load(folder/'trajectory.npz') as z,np.load(r['baseline']) as b:
            # Check effective drive and dense native records; no peak fitting.
            nframes=len(z['spike_counts_2ms']);post=z['external_counts_2ms'][3000:]
            assert np.all(np.isfinite(post))
            constant=np.ptp(z['core_rate_modulation_2ms'][3000:],axis=0).max() if len(post) else 0.
            err=None
            if r['arm']=='all_off_probe' and len(post):
                err=float(np.max(abs(post-z['group_sizes'][:2]*r['constant_core_input_per_ms']*2)))
                assert err<1e-8
                probe_count=int(z['spike_counts_2ms'][6000,0]) if nframes>6000 else None
                assert probe_count is None or probe_count>=int(z['group_sizes'][0])
                if nframes>=10000:
                    for lo,hi in [(8.,12.),(14.,20.)]:
                        counts=z['spike_counts_2ms'][round(lo*500):round(hi*500)]
                        pulse.append(dict(run=r['name'],ee=r['ee'],seed=r['seed'],window_s=[lo,hi],
                            group_spike_counts={g:int(counts[:,j].sum()) for j,g in enumerate(z['group_names'].tolist())}))
                    for core,g in enumerate(('coreAE','coreBE')):
                        segment=z['spike_counts_2ms'][6000:7000,core]
                        nz=np.flatnonzero(segment)
                        aux=z['core_current_voltage_2ms'][-500:,3*core:3*core+3]
                        target=quiet['region_summaries'][g]['voltage_min_mV'] if quiet else None
                        pulse.append(dict(run=r['name'],ee=r['ee'],seed=r['seed'],group=g,window_s=[12.,14.],
                            total_spikes=int(segment.sum()),forced_spikes=int(z['group_sizes'][0]) if core==0 else 0,
                            last_occupied_bin_end_after_pulse_ms=float((nz[-1]+1)*2) if len(nz) else None,
                            final_second_IE_mV=float(aux[:,0].mean()),final_second_II_mV=float(aux[:,1].mean()),
                            final_second_V_mV=float(aux[:,2].mean()),
                            final_second_max_voltage_eq_error_mV=float(abs(aux[:,2]-target).max()) if target is not None else None))
            assert constant<1e-12
            validation.append(dict(run=r['name'],prefix_bitwise=r['prefix_bitwise'],constant_modulation_range=float(constant),
                constant_external_count_error=err,duration_ms=r['duration_ms'],runaway_ms=r['runaway_ms']))
            windows=[('ou_removed',8.,20.)] if r['arm']=='ou_off' else [('all_removed',8.,12.),('probe_late',14.,20.)]
            for label,lo,hi in windows:
                if r['duration_ms']<hi*1000:continue
                for g in ('coreAE','coreBE'):
                    for context,array in [('baseline',b),('intervention',z)]:
                        m=window(array,g,lo,hi)
                        comparisons.append(dict(run=r['name'],ee=r['ee'],seed=r['seed'],group=g,arm=r['arm'],contrast=label,
                            context=context,lo_s=lo,hi_s=hi,burst_rate_hz=m['n_bursts']/(hi-lo),
                            metric={k:v for k,v in m.items() if k!='events'},events=event_peaks(array,m,g,lo)))
            # True 0.1-ms spike times; cell ISI and group burst IEI have different units.
            if r['arm']=='ou_off':
                times=z['exact_spike_time_ms']/1000;ids=z['exact_spike_cell']
                for label,lo,hi in [('before',2.,6.),('after',8.,20.)]:
                    for g,core in [('coreAE',0),('coreBE',1)]:
                        aux=z['core_current_voltage_2ms'][round(lo*500):round(hi*500),3*core:3*core+3]
                        inhibitory_rate=z['core_i_counts_2ms'][round(lo*500):round(hi*500),core]/z['core_i_sizes'][core]/.002
                        recovery.append(dict(run=r['name'],ee=r['ee'],seed=r['seed'],group=g,window=label,window_s=[lo,hi],
                            minimum_population_mean_V_mV=float(aux[:,2].min()),
                            peak_population_mean_IE_mV=float(aux[:,0].max()),peak_population_mean_II_mV=float(aux[:,1].max()),
                            peak_local_I_rate_hz=float(inhibitory_rate.max())))
                        selected=z['raster_sample_ids'];selected=selected[selected<len(z['core_index_E'])]
                        selected=selected[z['core_index_E'][selected]==core]
                        cvs=[];rates=[];nsp=[]
                        for cell in selected:
                            t=times[(ids==cell)&(times>=lo)&(times<hi)]
                            if len(t)>=21:
                                isi=np.diff(t);cvs.append(float(isi.std(ddof=1)/isi.mean()));rates.append(len(t)/(hi-lo));nsp.append(len(t))
                        group_metric=window(z,g,lo,hi)
                        cellstats.append(dict(run=r['name'],ee=r['ee'],seed=r['seed'],group=g,window=label,lo_s=lo,hi_s=hi,
                            min_spikes=21,sampled_cells=len(selected),eligible_cells=len(cvs),cell_isi_cv=cvs,cell_rate_hz=rates,
                            group_n_bursts=group_metric['n_bursts'],group_burst_iei_cv=group_metric['cv'] if group_metric['n_bursts']>=8 else None))
    write(OUT/'noise_comparisons.json',comparisons);write(OUT/'single_cell_irregularity.json',cellstats)
    write(OUT/'deterministic_return.json',pulse)
    write(OUT/'native_recovery_diagnostics.json',recovery)
    write(OUT/'native_validation.json',dict(completed=len(validation),expected=16,rows=validation))
    rows=[]
    for c in comparisons:
        row={k:v for k,v in c.items() if k not in ('metric','events')}
        row.update({k:c['metric'].get(k) for k in ('label','n_bursts','mean_rate_hz','cv','cv2','median_peak_active_fraction_2ms','iei_median_s')})
        rows.append(row)
    if rows:
        with (OUT/'noise_comparisons.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(baseline_core_runs=len(base),new_completed=len(validation),comparison_rows=len(rows))))

if __name__=='__main__':main()

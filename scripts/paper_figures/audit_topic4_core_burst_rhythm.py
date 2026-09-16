#!/usr/bin/env python3
"""Read-only population-rhythm audit on completed paired-noise trajectories.

Counts are unique E neurons active in a 2 ms frame. Strict interior bins avoid
mixing core and outside cells; individual cross-cycle cell identity is absent.
"""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.signal import welch,find_peaks,correlate
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scripts.paper_figures.audit_topic4_native_activity_shortcuts import regions

P=ROOT/'results/topic4_sef_hfo/contact_native_integrated_pilot'
OUT=ROOT/'results/topic4_sef_hfo/core_burst_rhythm_audit'
F=OUT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#df9137','#329ba6']
LABELS={'v2_1_pop1_de_b_002':'Original working point',
        'integrated_anchor1_B_minus':'New batch: placement 1',
        'native_reg_anchor2_A_minus':'Lower native penalty: placement 2',
        'tshape_anchor3_B_minus':'Wider bursts: placement 3'}


def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(x,prominence=.2):
    f,p=welch(x,fs=500,nperseg=2000,noverlap=1000)
    keep=(f>=.5)&(f<=100);freq=float(f[keep][np.argmax(p[keep])])
    ac=correlate(x-x.mean(),x-x.mean(),method='fft')[len(x)-1:]
    ac/=ac[0]
    ii,_=find_peaks(ac[10:501]);ii+=10
    lag=int(ii[np.argmax(ac[ii])]) if len(ii) else None
    sm=gaussian_filter1d(x,1)
    peaks,_=find_peaks(sm,distance=10,height=np.quantile(sm,.75),
                      prominence=prominence*(np.quantile(sm,.99)-np.quantile(sm,.1)))
    intervals=np.diff(peaks)*2.
    result=dict(n_peaks=len(peaks),frequency_peak_hz=freq,
                interval_median_ms=float(np.median(intervals)),
                interval_cv=float(intervals.std()/intervals.mean()),
                peak_active_fraction_median=float(np.median(x[peaks])),
                acf_peak_lag_ms=None if lag is None else lag*2.,
                acf_peak_value=None if lag is None else float(ac[lag]))
    return result,peaks,f,p,ac[:501]


def cycle_phase_pair(peaks):
    low=max(p[0] for p in peaks);high=min(p[-1] for p in peaks)
    frames=np.arange(low,high+1)
    ph=[np.interp(frames,p,2*np.pi*np.arange(len(p))) for p in peaks]
    delta=np.angle(np.exp(1j*(ph[1]-ph[0])))
    return float(abs(np.mean(np.exp(1j*delta)))),frames,delta


def main():
    F.mkdir(parents=True,exist_ok=True)
    confirmed=json.loads((P/'confirmation_scores.json').read_text())['candidates']
    training={r['candidate_id']:r for phase in ['history','inherited_A','adaptive_B']
              for r in json.loads((P/f'{phase}_scores.json').read_text())['candidates']}
    rows=[];sensitivity=[];pairs=[];examples={};sources=[]
    for r in confirmed:
        cid=r['candidate_id']
        for split,rr in [('training',training[cid]),('new_noise',r)]:
            for uid,u in rr['units'].items():
                path=Path(u['worker_path']);w=json.loads(path.read_text())
                with np.load(w['arrays']['path']) as z:
                    assert float(z['sheet_activity_frame_ms'])==2
                    grid,region,den,pure,nr=regions(z,np.asarray(r['candidate']['node_field']['centers_mm']),w['xy_geometry_audit']['distance_cutoff_mm'])
                    raw=z['sheet_activity_counts'].astype(float)
                    assert np.all(raw<=den[None]),'counts cannot exceed the local E-cell population'
                raw=raw[250:] # fixed 500 ms burn-in; no event selection or label filtering
                signals=[];peaksets=[];powers=[];acfs=[]
                for k in range(2):
                    mask=pure[k];n=int(den[mask].sum());total=int(np.sum(nr==k))
                    assert mask.any() and n>0
                    x=raw[:,mask].sum(1)/n
                    result,peaks,f,p,ac=metrics(x)
                    local=raw[:,mask]/den[mask]
                    corr=np.corrcoef(local.T);upper=corr[np.triu_indices(len(corr),1)]
                    rows.append(dict(candidate_id=cid,name=LABELS[cid],split=split,unit=uid,core=k+1,
                                     strict_interior_bins=int(mask.sum()),interior_E_neurons=n,
                                     full_core_E_neurons=total,interior_coverage=n/total,
                                     within_core_bin_correlation_median=float(np.median(upper)),**result))
                    signals.append(x);peaksets.append(peaks);powers.append(p);acfs.append(ac)
                    for prom in [.1,.2,.3]:
                        mm,*_=metrics(x,prom)
                        sensitivity.append(dict(candidate_id=cid,split=split,unit=uid,core=k+1,
                                                region='strict_interior',prominence=prom,**mm))
                    approx=raw[:,region==k].sum(1)/den[region==k].sum()
                    mm,*_=metrics(approx)
                    sensitivity.append(dict(candidate_id=cid,split=split,unit=uid,core=k+1,
                                            region='bin_center_approximation',prominence=.2,**mm))
                plv,frames,delta=cycle_phase_pair(peaksets)
                pairs.append(dict(candidate_id=cid,name=LABELS[cid],split=split,unit=uid,
                                  zero_lag_core_correlation=float(np.corrcoef(signals)[0,1]),
                                  cycle_phase_concentration=plv))
                if split=='new_noise' and uid.startswith('topo_6101_'):
                    examples[cid]=dict(signals=signals,peaks=peaksets,f=f,powers=powers,acfs=acfs,
                                       phase_frames=frames,phase_difference=delta)
                sources.append(dict(worker_json=str(path),arrays=w['arrays']['path'],
                                    candidate_id=cid,split=split,unit=uid))
    csvwrite(OUT/'core_metrics.csv',rows);csvwrite(OUT/'peak_rule_sensitivity.csv',sensitivity)
    csvwrite(OUT/'between_core_phase.csv',pairs)
    order=list(LABELS)
    fig,axs=plt.subplots(4,3,figsize=(15,10),gridspec_kw={'width_ratios':[2.1,1,1]})
    for i,cid in enumerate(order):
        e=examples[cid];times=(np.arange(len(e['signals'][0]))+.5)*.002+.5
        for k in range(2):
            sel=(times>=5)&(times<8)
            axs[i,0].plot(times[sel],e['signals'][k][sel],color=COLORS[k],lw=1,label=f'Core {k+1}')
            axs[i,1].plot(e['f'],e['powers'][k],color=COLORS[k],lw=1.2)
            axs[i,2].plot(np.arange(501)*2,e['acfs'][k],color=COLORS[k],lw=1.2)
        axs[i,0].set(ylabel=LABELS[cid]+'\nActive fraction / 2 ms',ylim=(0,1),xlabel='Time (s)')
        axs[i,1].set(xlim=(0,20),xlabel='Frequency (Hz)',ylabel='Power density')
        axs[i,2].set(xlim=(0,1000),xlabel='Lag (ms)',ylabel='Autocorrelation');axs[i,2].axhline(0,color='gray',lw=.6)
    axs[0,0].legend(loc='upper right',fontsize=8)
    for ax,title in zip(axs[0],['Native core activity: fixed 5–8 s excerpt','Full-record power spectrum','Full-record recurrence']):ax.set_title(title)
    fig.suptitle('Within each core: concentrated bursts recur at a preferred interval',fontsize=14)
    fig.tight_layout(rect=(0,.05,1,.96))
    fig.text(.5,.012,'Strictly interior bins only; 216–355 E cells per core in confirmation. Unsmooth native counts; no SEEG readout or event selection.\nOne topology is drawn; both topologies and both noise replays are in the CSV. A population rhythm is not proof of an autonomous limit cycle.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(F/f'core_activity_rhythm.{ext}',dpi=165,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(4,1,figsize=(13,8),sharex=True)
    for ax,cid in zip(axs,order):
        e=examples[cid];ax.plot(e['phase_frames']*.002+.5,e['phase_difference']/np.pi,'.',ms=1,color='#555a84')
        ax.set(ylabel='Phase / π',ylim=(-1.05,1.05),yticks=[-1,0,1]);ax.axhline(0,color='gray',lw=.5)
        ax.set_title(LABELS[cid],loc='left',fontsize=10)
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Core 1 versus core 2: cycle phase varies across the record')
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.text(.5,.01,'Phase interpolates between detected population peaks. Missing/small peaks affect phase; no phase-locking significance test is claimed.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(F/f'between_core_phase.{ext}',dpi=165,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'manifest.json',dict(status='COMPLETED_FROZEN_TRAJECTORY_DIAGNOSTIC',new_physical_runs=0,
          n_network_runs=len(sources),n_core_traces=len(rows),source_files=sources,
          observation='Unique excitatory cells firing at least once per nonoverlapping 2 ms frame; strict interior spatial bins',
          interval='Entire 0.5–24 s record; no primary-event or TA/TB selection',
          spectrum='Welch 4 s segments, 50% overlap, dominant peak searched 0.5–100 Hz',
          population_peak_rule='2 ms Gaussian only for locating peaks; gap >=20 ms; height >=75th percentile; prominence .2*(q99-q10), with .1/.3 sensitivity; active fraction read from unsmoothed counts',
          interval_CV='Standard deviation / mean of between-peak intervals; descriptive, no Poisson null significance claim',
          limitations=['strict interior covers only a subset of each core','single-cell identities and I-neuron traces not saved',
                       'no noise-removal or recurrence intervention; autonomous oscillation mechanism untested',
                       '3–4 Hz describes burst repetition, not an HFO carrier'],human_visual_acceptance='PENDING'))
    (F/'README.md').write_text('# Core 内周期爆发诊断\n\n### core_activity_rhythm.png / .pdf\n\n每行一个候选，左侧是固定 5–8 秒内两 core 的原生活跃细胞比例，中间为完整 0.5–24 秒功率谱，右侧为自相关。只用完全落在 core 内部的空间格，避免边缘神经元混入；未经过接触读出或事件筛选。**关注点**：单次群体同步爆发、近周期复发、HFO 载波是不同问题；图中的 3–4 Hz 是爆发重复率。\n\n### between_core_phase.png / .pdf\n\n根据两 core 相邻群体峰之间的线性相位，展示同一拓扑整段记录中的相位差。峰提取有明确阈值及敏感性 CSV，漏检小峰可能影响相位。**关注点**：各自有节律不自动代表两个 core 稳定锁相，也不证明无噪声时存在自主极限环。\n')
    for path in F.glob('*.png'):
        with Image.open(path) as im:im.verify()
    print('COMPLETE',len(sources),'networks',len(rows),'core traces')
    for cid in order:
        rr=[r for r in rows if r['candidate_id']==cid and r['split']=='new_noise']
        print(LABELS[cid],{k:[round(min(r[k] for r in rr),3),round(max(r[k] for r in rr),3)] for k in ['frequency_peak_hz','interval_median_ms','interval_cv','peak_active_fraction_median']})
    print('Phase concentration range',min(r['cycle_phase_concentration'] for r in pairs),max(r['cycle_phase_concentration'] for r in pairs))


if __name__=='__main__':main()

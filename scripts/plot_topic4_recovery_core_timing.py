"""Native population timing, independent of contact-event selection.

Peak interpolation is only a descriptive cycle coordinate, not an inferred
physiological state or evidence of a causal link between the two cores.
"""
import argparse,json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import analyze_topic4_propagation_recovery_night as review
an=review.an;rt=review.rt

def metrics(t,x,dt,prom=.2):
    sm=gaussian_filter1d(x,2/dt)
    peaks,_=find_peaks(sm,distance=max(1,round(20/dt)),height=np.quantile(sm,.75),
        prominence=prom*(np.quantile(sm,.99)-np.quantile(sm,.1)))
    tt=t[peaks];iv=np.diff(tt)
    f,p=welch(x,fs=1000/dt,nperseg=min(len(x),round(4000/dt)),noverlap=min(len(x)//2,round(2000/dt)))
    band=(f>=.5)&(f<=100)
    return dict(n_peaks=len(peaks),interval_median_ms=float(np.median(iv)) if len(iv) else None,
        interval_cv=float(iv.std()/iv.mean()) if len(iv) else None,
        dominant_population_frequency_hz=float(f[band][np.argmax(p[band])]) if band.any() else None),peaks,sm

def main(phase,candidate):
    old,plan,spec,cases=review.stage_cases(phase);out=review.night.OUT/('core_timing_'+phase);F=out/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    records=[];files=[]
    for c in cases:
      if c['base_id']!=candidate:continue
      seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==candidate and int(t)==c['topology']})
      for seed in seeds:
        path=an.run.result_path(c['output_stage'],candidate,c['topology'],seed);u=an.load_unit(path,1500)
        if u is None:continue
        r,a,primary=u;ids=an.all_detected_ids(r,1500);tt=a['trace_time_ms'];keep=tt>=1500;t=tt[keep];dt=float(np.median(np.diff(t)))
        fig,axes=plt.subplots(3,1,figsize=(13,8),layout='constrained');peaksets=[];row=dict(candidate=c['id'],seed=seed,source=str(path),groups={})
        for g,col,title in [('coreAE','#dc9228','左核'),('coreBE','#299995','右核')]:
          x=a['trace_'+g+'_spikes'][keep].astype(float)/len(a['group_'+g])*1000/dt
          m,peaks,sm=metrics(t,x,dt);peaksets.append(t[peaks]);row['groups'][g]=dict(m,neuron_count=len(a['group_'+g]),mean_rate_hz=float(x.mean()),
              prominence_sensitivity={str(p):metrics(t,x,dt,p)[0] for p in [.1,.3]})
          excerpt=t<4500;axes[0].plot(t[excerpt]/1000,sm[excerpt],c=col,lw=.9,label=title)
          axes[1].plot(t[peaks][1:]/1000,np.diff(t[peaks]),'.',c=col,ms=2,label=title)
        if all(len(p)>1 for p in peaksets):
          tt2=t[(t>=max(p[0] for p in peaksets))&(t<=min(p[-1] for p in peaksets))]
          ph=[np.interp(tt2,p,2*np.pi*np.arange(len(p))) for p in peaksets]
          d=np.angle(np.exp(1j*(ph[1]-ph[0])))/np.pi
          axes[2].plot(tt2[::10]/1000,d[::10],'.',c='.65',ms=1)
          for mode,col,label in [(1,'#c9473f','TA'),(0,'#377dab','TB')]:
            ii=ids[a['event_mode'][ids]==mode];et=a['event_time_ms'][ii]
            ok=(et>=tt2[0])&(et<=tt2[-1]);et=et[ok]
            # Interpolate the two unwrapped coordinates before taking the wrapped difference.
            eventphase=[np.interp(et,p,2*np.pi*np.arange(len(p))) for p in peaksets]
            dd=np.angle(np.exp(1j*(eventphase[1]-eventphase[0])))/np.pi
            axes[2].scatter(et/1000,dd,c=col,s=8,label=label,zorder=3)
          row['cycle_phase_concentration']=float(abs(np.mean(np.exp(1j*np.pi*d))))
        axes[0].set(ylabel='2 ms 平滑的群体平均率 (Hz)',xlabel='固定 1.5–4.5 s 片段')
        axes[1].set(ylabel='相邻群体峰间隔 (ms)',xlabel='完整记录时间 (s)')
        axes[2].set(ylabel='右核−左核的周期坐标差 / π',xlabel='完整记录时间 (s)',ylim=(-1.05,1.05))
        for ax in axes:ax.grid(alpha=.2);ax.legend(loc='upper right')
        fig.suptitle(review.display(c)+f'｜图 {c["topology"]}、噪声 {seed}\n群体峰来自全部核内发放；下图红蓝点仅标记全检测事件标签，不是预设的核心起源',fontsize=12)
        stem=f'{candidate}_{c["topology"]}_{seed}'
        for ext in ['png','pdf']:
          f=F/f'{stem}.{ext}';fig.savefig(f,dpi=170);files.append(f.name)
        plt.close(fig);records.append(row)
    rt.write(out/(candidate+'_manifest.json'),dict(records=records,producer=__file__,files=files,
        source_quantity='Exact 1ms spike counts for all E neurons assigned to each core, divided by neuron count and bin seconds; no electrode readout or selected-lineage filter.',
        peak_rule='2ms Gaussian smoothing; separation >=20ms; height >=75th percentile; prominence 0.2*(q99-q10), with 0.1/0.3 sensitivity.',
        frequency='Welch dominant population repetition frequency within 0.5-100Hz, 4s windows and 50 percent overlap; not an HFO carrier frequency.',
        cycle_coordinate='Linear phase between successive population peaks; missing or double peaks can distort this coordinate. No causality, bistability, autonomous cycle or significance claim.'))
    notes=[]
    for f in sorted(F.glob('*.png')):
      notes.append(f'### {f.name}\n\n上排为固定1.5–4.5秒内两核完整群体发放率，中排显示整段相邻群体峰间隔，下排显示群体峰之间插值得到的周期坐标差与全部检测事件的标签。橙和青区分左/右核，红和蓝仅区分TA/TB标签；找峰参数及敏感性结果保存在manifest中。**关注点**：事件标签的时间偏移是否伴随原生群体时间关系的变化；这不证明核心之间存在因果传播、自主振荡或双稳态。')
    (F/'README.md').write_text('\n\n'.join(notes)+'\n')
    print(json.dumps(dict(output=str(out),runs=len(records))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='long');p.add_argument('--candidate',required=True);a=p.parse_args();main(a.phase,a.candidate)

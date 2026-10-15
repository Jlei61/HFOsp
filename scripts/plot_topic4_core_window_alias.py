"""Compare an own-mode-mean TA and a patient-neighbor-selected TA, same replay.

No changed event boundaries, masks, classifier or training statistic.
"""
import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import analyze_topic4_propagation_recovery_night as review
from scripts import plot_topic4_recovery_core_timing as timing
from scripts.paper_figures import plot_topic4_recovery_review as mainfig
an=review.an;rt=review.rt


def main(phase='long',cid='refine_mid_EE075'):
    old,plan,spec,cases=review.stage_cases(phase)
    c=next(c for c in cases if c['base_id']==cid)
    out=review.night.OUT/('core_window_alias_'+phase)
    if phase!='long' or cid!='refine_mid_EE075':out=out/cid
    F=out/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42});rows=[];selections=[]
    for seed in [847101,847102]:
        path=an.run.result_path(c['output_stage'],cid,2511,seed);r,a,primary=an.load_unit(path,1500)
        ids=primary[a['event_mode'][primary]==1];typical=mainfig.representatives(a,primary)['TA']['event']
        compatible=int(ids[np.argmin(a['event_distance_modes'][ids,1])]);tt=a['trace_time_ms'];keep=tt>=1500;t=tt[keep]
        smooth={};peaks={}
        for group in ['coreAE','coreBE']:
            rate=a['trace_'+group+'_spikes'][keep].astype(float)*1000/len(a['group_'+group])
            _,ix,sm=timing.metrics(t,rate,1.);smooth[group]=sm;peaks[group]=t[ix]
        for i in ids:
            lo,hi=r['events'][int(i)]['window_ms'];cm=review.core_timing(r,a,int(i));et=an.event_timing(r,a,int(i),a['contact_names'])
            rows.append(dict(seed=seed,event=int(i),mode='TA',n_coreA_peaks=int(((peaks['coreAE']>=lo)&(peaks['coreAE']<hi)).sum()),
                n_coreB_peaks=int(((peaks['coreBE']>=lo)&(peaks['coreBE']<hi)).sum()),patient_neighbor_distance=float(a['event_distance_modes'][i,1]),
                support=int(a['event_support'][i]),**cm,**{k:v for k,v in et.items() if k!='event'}))
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained',gridspec_kw={'height_ratios':[1,2]})
        for col,(i,label) in enumerate([(typical,'自身TA均值附近'),(compatible,'患者特征距离最小的TA')]):
            lo,hi=r['events'][i]['window_ms'];sel=(t>=lo)&(t<hi)
            for group,color,title in [('coreAE','#d9533f','左核 E'),('coreBE','#2881ae','右核 E')]:
                axes[0,col].plot(t[sel]-lo,smooth[group][sel],c=color,lw=1,label=title)
                peak=peaks[group][(peaks[group]>=lo)&(peaks[group]<hi)]
                for pt in peak:axes[0,col].axvline(pt-lo,c=color,ls=':',lw=.5)
            axes[0,col].set(xlim=(0,250),ylabel='每细胞率 (Hz)',title=f'{label} · 事件 {i}');axes[0,col].legend(fontsize=8)
            dt=float(a['contact_envelope_dt_ms']);order=[list(a['contact_names']).index(n) for n in an.DISPLAY]
            env=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T
            env=env/np.maximum(env.max(1,keepdims=True),1e-20)
            axes[1,col].imshow(env,aspect='auto',extent=[0,250,14.5,-.5],cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            centroid=a['centroid_ms'][i,order]-lo;ok=np.isfinite(centroid)
            axes[1,col].scatter(centroid[ok],np.arange(15)[ok],s=12,c='#40cfe4')
            axes[1,col].set(yticks=range(15),yticklabels=an.DISPLAY,xlabel='距原250ms窗口起点 (ms)')
            q=r['events'][i]['qualifying_interval_ms'];axes[1,col].axvspan(q[0]-lo,q[1]-lo,facecolor='none',edgecolor='white',lw=.7,ls='--')
            selections.append(dict(seed=seed,event=i,rule=label,original_window_ms=[lo,hi],qualifying_interval_ms=q,arrays_sha256=r['arrays_sha256']))
        fig.suptitle(f'同一图、同一参数、噪声{seed}：相邻核心活动是否进入同一个观测窗？\n上：全部原生核内发放，2ms平滑；下：全部触点包络，青点为参与触点质心，白框为群体合格段\n窗口与评分均未改变；事件例不是新的TA亚型或因果来源分类',fontsize=11)
        for ext in ['png','pdf']:fig.savefig(F/f'core_window_{seed}.{ext}',dpi=180)
        plt.close(fig)
    an.writecsv(out/'all_primary_TA.csv',rows)
    rt.write(out/'manifest.json',dict(status='READ_ONLY_OBSERVATION_DIAGNOSTIC',candidate=cid,phase=phase,selections=selections,producer=__file__,producer_sha256=rt.sha(__file__),
        peak_rule='Inherited whole-record 2ms Gaussian, q75 height, .2*(q99-q10) prominence, >=20ms separation; not refitted per window.',
        selection='Own-mode-mean exemplar versus smallest frozen patient-mode distance in each replay; no route filter, no changed primary rule.',
        interpretation='Multiple core peaks in one window may expose concatenation of successive native activity. Peak count is descriptive, not causal lineage and not a new event-rejection gate.'))
    (F/'README.md').write_text('\n\n'.join(f'### {file.name}\n\n同一条件、同一噪声中，对比自身TA均值附近事件与事后患者邻域距离最小的TA；上排是完整核内群体率，下排是全部触点包络和原群体合格段。保留原250ms窗口、参与定义和质心，不删触点或移动边界。**关注点**：相邻核心活动是否在一个观测窗内被合并，以及它怎样改变SCL时序；群体峰不能替代因果谱系。' for file in sorted(F.iterdir()) if file.suffix in ['.png','.pdf'])+'\n')
    print(dict(output=str(out),TA_events=len(rows)),flush=True)


if __name__=='__main__':main()

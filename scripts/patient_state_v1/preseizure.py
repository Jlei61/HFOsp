"""Time-weighted preictal state levels versus non-overlapping changes.

Readout times only consume completed 250-ms event windows. All-data parameter
reconstructions are retrospective; forward-parameter readouts are separately named.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import filter_adf
from scripts.patient_state_v1.analyze_first import best_fits

def state_at(times,theta,data,events,filtered,inv):
    origin=float(data['origin_epoch']);nc=len(theta)-2;tau,sd=np.exp(theta[-2:]);idx=np.searchsorted(events.end_epoch.to_numpy(),times,side='right')-1
    m=np.zeros(len(times));v=np.full(len(times),sd*sd)
    offsets=np.array([r['offset'] for r in inv]);epochs=np.searchsorted(offsets,times,side='right')
    valid=(idx>=0)
    valid[valid]&=data['epoch'][idx[valid]]==epochs[valid]
    ii=idx[valid];dt=(times[valid]-origin)/3600-data['t'][ii];rho=np.exp(-dt/tau)
    m[valid]=rho*filtered['mean'][ii];v[valid]=rho*rho*filtered['variance'][ii]+sd*sd*(-np.expm1(-2*dt/tau))
    phase=2*np.pi*(times/3600)/24;x=np.column_stack([np.ones(len(times)),np.sin(phase),np.cos(phase)])
    eta=x[:,:nc]@theta[:nc];nodes,w=hermgauss(32);p=expit(eta[:,None]+m[:,None]+np.sqrt(2*v[:,None])*nodes)@(w/np.sqrt(np.pi))
    return m,v,p,eta

def covered(times,exposure):
    mask=np.zeros(len(times),bool)
    for r in exposure.itertuples():mask|=(times>=r.start_epoch)&(times<r.end_epoch)
    return mask

def window_summary(lo,hi,theta,data,events,filtered,inv,exposure):
    # Five-second midpoint integration; means are over observed physical time, not events.
    if hi<=lo:return dict(mean_s=np.nan,mean_tb_probability=np.nan,observed_seconds=0)
    edges=np.arange(lo,hi,5.);duration=np.minimum(edges+5,hi)-edges;t=edges+duration/2;mask=covered(t,exposure)
    if not mask.any():return dict(mean_s=np.nan,mean_tb_probability=np.nan,observed_seconds=0)
    m,v,p,_=state_at(t[mask],theta,data,events,filtered,inv)
    return dict(mean_s=np.average(m,weights=duration[mask]),mean_tb_probability=np.average(p,weights=duration[mask]),observed_seconds=duration[mask].sum())

def main():
    data=dict(np.load(RUN/'observations.npz'));events=pd.read_csv(RUN/'events.csv');exposure=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');best=best_fits();folds=json.loads((RUN/'splits.json').read_text())
    assert np.all(events.end_epoch.to_numpy()[:-1]<=events.start_epoch.to_numpy()[1:])
    rows=[];traces={};selected=wins[wins.window=='pre15'].sort_values('sz')
    for r in selected.itertuples():
        sz=inv[r.sz-1];onset=sz['onset'];previous=inv[r.sz-2]['offset'];latest=max(previous,onset-900);earlier=max(previous,onset-1800)
        for model in ('ou','ou_cycle'):
            for kind in ('retrospective_parameters','forward_parameters','preinterval_parameters'):
                scope='full'
                if kind=='forward_parameters':
                    possible=[f for f in folds if data['t'][f['test_start']]<=(onset-float(data['origin_epoch']))/3600]
                    if not possible:continue
                    scope=f"fold{possible[-1]['fold']}"
                if kind=='preinterval_parameters':
                    path=RUN/'preseizure_forward_fits'/f'sz{r.sz:02d}_{model}.json'
                    if not path.exists():continue
                    fit=json.loads(path.read_text());theta=np.array(fit['theta']);scope=f'before_sz{r.sz-1}_offset'
                    assert fit['job']['cutoff_epoch']<=latest
                else:theta=np.array(best[scope,model]['theta'])
                filt=filter_adf(theta,data,model=='ou_cycle')
                a=window_summary(latest,onset,theta,data,events,filt,inv,exposure);b=window_summary(earlier,latest,theta,data,events,filt,inv,exposure)
                endpoint=state_at(np.array([onset-1e-6]),theta,data,events,filt,inv)
                rows.append(dict(sz=r.sz,label=r.label,model=model,parameter_scope=kind,training_scope=scope,
                                 pre15_mean_s=a['mean_s'],prior15_mean_s=b['mean_s'],delta_nonoverlap_s=a['mean_s']-b['mean_s'],
                                 pre15_predicted_tb=a['mean_tb_probability'],prior15_predicted_tb=b['mean_tb_probability'],
                                 latest_observed_seconds=a['observed_seconds'],earlier_observed_seconds=b['observed_seconds'],
                                 endpoint_mean_s=endpoint[0][0],endpoint_sd=np.sqrt(endpoint[1][0]),
                                 nominal_latest_minutes=(onset-latest)/60,nominal_earlier_minutes=(latest-earlier)/60,
                                 pre15_observed_tb_rate=r.n_tb/r.observed_hours,pre15_observed_tb_share=r.n_tb/r.n_events if r.n_events else np.nan))
                if kind=='retrospective_parameters' and model=='ou':
                    ts=np.arange(max(previous,onset-3600),onset,5.);m,v,p,eta=state_at(ts,theta,data,events,filt,inv);mask=covered(ts,exposure)
                    traces[r.sz]=dict(t=ts,minutes=(ts-onset)/60,m=m,v=v,p=p,observed=mask,onset=onset,label=r.label)
    df=pd.DataFrame(rows);df.to_csv(RUN/'preseizure_state_readouts.csv',index=False)
    np.savez_compressed(RUN/'preseizure_traces.npz',**{f'sz{k}_{name}':value for k,v in traces.items() for name,value in v.items() if isinstance(value,np.ndarray)})
    fp=RUN/'figures';plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(3,4,figsize=(15,9),sharex=True,sharey=True)
    for ax,(sz,v) in zip(axs.ravel(),traces.items()):
        c='#2166ac' if v['label']=='TB' else '#b2182b';m=v['m'].copy();m[~v['observed']]=np.nan;sd=np.sqrt(v['v'])
        ax.plot(v['minutes'],m,c=c,lw=1);ax.fill_between(v['minutes'],m-1.96*sd,m+1.96*sd,color=c,alpha=.16)
        ax.axhline(0,c='gray',lw=.6);ax.axvline(-15,c='gray',ls=':',lw=.6);ax.set_title(f"SZ{sz} | {v['label']}-source",color=c)
        ax.set_xlim(-60,0);ax.set_ylim(-2.5,2.5)
    for ax in axs[-1]:ax.set_xlabel('Minutes before seizure')
    for ax in axs[:,0]:ax.set_ylabel('Filtered s (log-odds units)')
    fig.suptitle('All 12 qualified seizures: state level and uncertainty before onset\nAll-data OU parameters: retrospective reconstruction; filtering uses only completed past events',fontsize=12);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'preseizure_all_states.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(12,9),sharex='col')
    for col,sz in enumerate((19,22)):
        v=traces[sz];onset=v['onset'];lo=v['t'][0];edges=np.arange(lo,onset+1e-6,300.)
        if edges[-1]<onset:edges=np.r_[edges,onset]
        ta=[];tb=[];share=[];cx=[]
        for a,b in zip(edges[:-1],edges[1:]):
            mask=(events.start_epoch>=a)&(events.end_epoch<=b);n=int(mask.sum());nb=int(events.loc[mask,'label_tb'].sum());hours=0.
            for rr in exposure.itertuples():hours+=max(0,min(b,rr.end_epoch)-max(a,rr.start_epoch))/3600
            ta.append((n-nb)/hours if hours else np.nan);tb.append(nb/hours if hours else np.nan);share.append(nb/n if n else np.nan);cx.append(((a+b)/2-onset)/60)
        axs[0,col].plot(cx,ta,'o-',c='#b2182b',label='TA');axs[0,col].plot(cx,tb,'o-',c='#2166ac',label='TB');axs[0,col].set_title(f'SZ{sz}: TB-source seizure');axs[0,col].legend()
        pp=v['p'].copy();pp[~v['observed']]=np.nan;axs[1,col].plot(v['minutes'],pp,c='#6a3d9a',label='Filtered TB probability');axs[1,col].scatter(cx,share,c='#2166ac',s=25,label='Observed 5-min share');axs[1,col].set_ylim(0,1);axs[1,col].legend(fontsize=8)
        m=v['m'].copy();m[~v['observed']]=np.nan;sd=np.sqrt(v['v']);axs[2,col].plot(v['minutes'],m,c='#6a3d9a');axs[2,col].fill_between(v['minutes'],m-1.96*sd,m+1.96*sd,color='#6a3d9a',alpha=.15);axs[2,col].axhline(0,c='gray',lw=.6);axs[2,col].set_xlabel('Minutes before seizure')
        for ax in axs[:,col]:ax.axvline(-15,c='gray',ls=':',lw=.6);ax.set_xlim(-60,0)
    for ax,label in zip(axs[:,0],['Events / hour','TB probability / share','Filtered s']):ax.set_ylabel(label)
    fig.suptitle('Both TB-source seizures: rate changes and mode preference are distinct\nConditional-mark OU; all-data parameters; no seizure-type input and no Z/M coupling',fontsize=12);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'tb_seizures_rate_vs_state.{ext}',dpi=180)
    plt.close(fig)
    readme=fp/'README.md'
    if '### preseizure_all_states.png' in readme.read_text():
        readme.write_text(readme.read_text().split('### preseizure_all_states.png')[0].rstrip()+'\n')
    with readme.open('a') as f:
        f.write('\n### preseizure_all_states.png\n显示全部12次合格发作前最多60分钟的OU滤波状态及近似95%边缘不确定范围；PDF为同图。参数由全数据拟合，属于回顾性重建，任何时刻仅消费已经结束的事件窗口；没有跨发作强行连接状态。\n**关注点**：状态水平与上升趋势分别判断，TB组仅2次，不确定范围并非12次发作之间的置信区间。\n\n### tb_seizures_rate_vs_state.png\n并列展示两次TB型发作前的TA/TB率、TB比例和连续状态，5分钟分箱按实际覆盖计算。无事件窗口的比例缺失，事件率为零，缺覆盖不填零。\n**关注点**：总活动强弱与相对模式偏好可分离；全数据参数重建不是前瞻性发作预测，图待人工审阅。\n')
    write_json(RUN/'preseizure_audit.json',dict(status='COMPLETE',n_rows=len(df),n_qualified=12,n_tb_qualified=2,event_windows_nonoverlapping=True,
            all_event_window_durations_seconds=np.unique(np.round(events.end_epoch-events.start_epoch,6)),
            endpoint_uses_completed_events=True,window_average='5-second midpoint integration over observed time; not event weighted',
            uncertainty='Gaussian assumed-density marginal posterior approximation, conditional on point parameters',human_visual_review='pending'))
    print(df[(df.label=='TB')&(df.model=='ou')].to_string(index=False),flush=True)

if __name__=='__main__':main()

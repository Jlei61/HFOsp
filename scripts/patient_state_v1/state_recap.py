"""Readable patient state reconstruction, with explicit coordinates and gaps.

Hyperparameters use the full-record posterior median, so this is retrospective.
At each display query only completed labels enter the conditional state filter.
Displayed uncertainty is Gaussian filtering uncertainty at fixed parameters.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import filter_adf,slice_data
from scripts.patient_state_v1.advanced_controls import history_data

OUT=RUN/'state_recap_v1_33'

def main():
    tab=pd.read_csv(RUN/'independent_importance_posterior_v1_28/parameter_intervals.csv');p=tab[(tab.model=='ou_history')&(tab.target=='particle')].set_index('parameter');assert (p.diagnostic_status=='IMPORTANCE_DIAGNOSTICS_PASS').all()
    b=float(p.loc['baseline_log_odds','median']);g=float(p.loc['fast_history_coefficient','median']);tau=float(p.loc['tau_minutes','median'])/60;sd=float(p.loc['stationary_sd','median']);theta=np.array([b,g,0.,np.log(tau),np.log(sd)])
    data=dict(np.load(RUN/'observations.npz'));origin=float(data['origin_epoch']);d=history_data(data);f=filter_adf(theta,d,True,order=64)
    prefix=filter_adf(theta,slice_data(d,0,1000),True,order=64);assert np.array_equal(prefix['mean'],f['mean'][:1000])
    ev=pd.read_csv(RUN/'events.csv');ex=pd.read_csv(RUN/'exposure.csv');sz=json.loads((RUN/'seizures.json').read_text());extent=ex.end_epoch.max()-origin;sec=np.arange(0,extent+1,30.);qabs=sec+origin;epoch=np.searchsorted([s['offset'] for s in sz],qabs,side='right');idx=np.searchsorted(ev.end_epoch.to_numpy(),qabs,side='right')-1
    valid=np.zeros(len(sec),bool);segment=np.full(len(sec),-1,int)
    for k,row in enumerate(ex.itertuples()):
        inside=(qabs>=row.start_epoch)&(qabs<row.end_epoch);valid|=inside;segment[inside]=k
    m=np.zeros(len(sec));v=np.full(len(sec),sd*sd);ok=(idx>=0)&valid;ok[ok]&=d['epoch'][idx[ok]]==epoch[ok];lag=sec[ok]/3600-d['t'][idx[ok]];a=np.exp(-lag/tau);m[ok]=a*f['mean'][idx[ok]];v[ok]=a*a*f['variance'][idx[ok]]+sd*sd*(-np.expm1(-2*lag/tau));assert np.all(lag>=.25/3600-1e-7)
    nodes,w=hermgauss(64);w/=np.sqrt(np.pi);prob=expit(b+m[:,None]+np.sqrt(2*v[:,None])*nodes)@w
    lo=m-1.96*np.sqrt(v);hi=m+1.96*np.sqrt(v)
    curves=pd.DataFrame(dict(seconds=sec,hours=sec/3600,eligible=valid,exposure_segment=segment,epoch=epoch,residual_state_mean=m,residual_state_lower=lo,residual_state_upper=hi,slow_tb_propensity=prob,slow_propensity_lower=expit(b+lo),slow_propensity_upper=expit(b+hi)))
    OUT.mkdir(parents=True,exist_ok=True);curves.to_csv(OUT/'query_curve.csv',index=False)
    width=300.;edges=np.arange(0,np.ceil(extent/width)*width+width,width);n=np.histogram(ev.start_epoch-origin,edges)[0];tb=np.histogram(ev.start_epoch-origin,edges,weights=ev.label_tb.to_numpy(dtype=np.int64))[0];assert tb.sum()==ev.label_tb.sum();duration=np.zeros(len(n))
    for row in ex.itertuples():duration+=np.maximum(0,np.minimum(edges[1:],row.end_epoch-origin)-np.maximum(edges[:-1],row.start_epoch-origin))
    keep=duration>=.95*width
    for seizure in sz:keep&=~((edges[:-1]<seizure['offset']-origin)&(edges[1:]>seizure['onset']-origin))
    share=np.full(len(n),np.nan);good=keep&(n>=10);share[good]=tb[good]/n[good];rates=np.full((2,len(n)),np.nan);rates[0,keep]=(n[keep]-tb[keep])/duration[keep]*3600;rates[1,keep]=tb[keep]/duration[keep]*3600;mid=(edges[:-1]+edges[1:])/7200
    pd.DataFrame(dict(hour=mid,count=n,tb_count=tb,exposure_seconds=duration,eligible=keep,tb_share=share,ta_per_hour=rates[0],tb_per_hour=rates[1])).to_csv(OUT/'empirical_5min.csv',index=False)
    fig,axs=plt.subplots(3,1,figsize=(14,8),sharex=True,gridspec_kw={'height_ratios':[1,1.1,1.1]})
    axs[0].plot(mid,rates[0],color='#bb4050',lw=.8,label='TA');axs[0].plot(mid,rates[1],color='#296dac',lw=.8,label='TB');axs[0].set_ylabel('Events / observed hour\n5-min windows');axs[0].legend(loc='upper right',ncol=2,fontsize=9)
    axs[1].scatter(mid,share,s=7,color='#888888',alpha=.5,label='Observed 5-min TB share (n ≥ 10)')
    display=valid&np.r_[True,segment[1:]==segment[:-1]]
    mm=np.where(display,m,np.nan);ll=np.where(display,lo,np.nan);hh=np.where(display,hi,np.nan);pp=np.where(display,prob,np.nan)
    connected=np.isfinite(mm[:-1])&np.isfinite(mm[1:]);assert np.all(segment[:-1][connected]==segment[1:][connected])
    axs[1].plot(sec/3600,pp,color='#28668c',lw=1.,label='Slow TB propensity (short-memory term = 0)');axs[1].set(ylabel='TB fraction / probability',ylim=(0,1));axs[1].legend(loc='upper left',ncol=2,fontsize=8)
    axs[2].fill_between(sec/3600,ll,hh,color='#28668c',alpha=.15,label='Conditional state 95% interval');axs[2].plot(sec/3600,mm,color='#28668c',lw=.9,label='Filtered state deviation');axs[2].axhline(0,color='gray',ls=':',lw=.8);axs[2].set_ylabel('State deviation from baseline\nTB log-odds');axs[2].set_xlabel('Hours from first available artifact coverage');axs[2].legend(loc='lower right',ncol=2,fontsize=8)
    previous=0.
    for row in ex.itertuples():
        lo_h=(row.start_epoch-origin)/3600;hi_h=(row.end_epoch-origin)/3600
        if lo_h>previous:
            for ax in axs:ax.axvspan(previous,lo_h,color='#eeeeee',zorder=-5)
        previous=hi_h
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,extent/3600)
    fig.suptitle('E1146: continuous mode preference inferred from labelled interictal events\nOU + one-second mark memory; retrospective parameters; gaps/exclusions shaded, no event-triggered state reset',fontsize=12)
    fig.text(.5,.012,f'Fixed posterior-median parameters: time constant {tau*60:.1f} min; state SD {sd:.3f} log-odds. Bands exclude parameter uncertainty. Positive deviation increases TB relative to baseline.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.035,1,.93))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'patient_state_recap.{ext}',dpi=180)
    plt.close(fig)
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',model='OU-history',reference_parameters=dict(b=b,gamma=g,tau_hours=tau,stationary_sd=sd),parameter_source='Independent particle importance posterior medians, full data; not a new optimizer result',uncertainty='Gaussian assumed-density filter state variance at fixed reference parameters; not full parameter-integrated bands',coordinate='Residual x=s-b. x>0 raises TB propensity relative to b; it does not imply TB>50%. Slow probability averages sigmoid(b+x), excluding the one-second history term.',query='30-second display queries, only completed250ms marks, exact OU propagation from the last available event start; no lines drawn across absent/excluded support',causal_prefix_canary=True,limits='Filtering recursion uses past labels, but hyperparameters and frozen labels use the full record. This is retrospective reconstruction, not a future forecast or a measured physiological trajectory. Clinical boundaries use independent state priors, not an inferred physical seizure reset.'))
    readme=RUN/'figures/README.md'
    if '### patient_state_recap.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### patient_state_recap.png\n以通过独立粒子重要性诊断的OU-history参数中位数为参考，展示实际TA/TB率、5分钟TB份额和持续状态的条件过滤结果。状态明确使用相对基线的log-odds偏移，灰色缺失/临床排除区不连接曲线；95%带仅为固定参数下的高斯状态近似。\n**关注点**：正状态偏移提高的是相对TB倾向，不保证TB占多数；本图参数来自全记录，不能作为临床前瞻预测。作为当前状态回顾入口，早期full_state_timeline图保留作历史诊断，不用其周期残差状态代替本坐标。\n')
    print(json.dumps(dict(status='COMPLETE',n_queries=len(sec),n_eligible=int(valid.sum()),reference_tau_minutes=tau*60)))

if __name__=='__main__':main()

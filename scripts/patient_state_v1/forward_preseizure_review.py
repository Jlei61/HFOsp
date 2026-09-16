"""Plot preinterval-trained state reconstructions and distinguish level from change."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import filter_adf
from scripts.patient_state_v1.preseizure import state_at,covered
from scripts.patient_state_v1.analyze_first import best_fits

def permutation(x,label):
    good=np.isfinite(x);x=np.asarray(x)[good];label=np.asarray(label)[good];k=int(label.sum());observed=np.median(x[label])-np.median(x[~label]);stats=[]
    for ix in itertools.combinations(range(len(x)),k):
        a=np.zeros(len(x),bool);a[list(ix)]=True;stats.append(np.median(x[a])-np.median(x[~a]))
    return dict(n_ta=len(x)-k,n_tb=k,median_ta=np.median(x[~label]),median_tb=np.median(x[label]),difference=observed,exact_two_sided_p=np.mean(np.abs(stats)>=abs(observed)-1e-12),n_permutations=len(stats))

def main():
    data=dict(np.load(RUN/'observations.npz'));ev=pd.read_csv(RUN/'events.csv');ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');wins=wins[wins.window=='pre15'].sort_values('sz');table=pd.read_csv(RUN/'preseizure_state_readouts.csv');table=table[(table.model=='ou')&(table.parameter_scope=='preinterval_parameters')].sort_values('sz').copy();table['delta_tb_probability']=table.pre15_predicted_tb-table.prior15_predicted_tb;table['onset_hours']=[(inv[int(i)-1]['onset']-float(data['origin_epoch']))/3600 for i in table.sz]
    fig,axs=plt.subplots(3,4,figsize=(15,9),sharex=True,sharey=True);traces={};ret=np.array(best_fits()['full','ou']['theta']);retfilter=filter_adf(ret,data)
    for ax,r in zip(axs.flat,wins.itertuples()):
        onset=inv[r.sz-1]['onset'];previous=inv[r.sz-2]['offset'];fit=json.loads((RUN/'preseizure_forward_fits'/f'sz{r.sz:02d}_ou.json').read_text());theta=np.array(fit['theta']);filt=filter_adf(theta,data);ts=np.arange(max(previous,onset-3600),onset,5.);m,v,p,eta=state_at(ts,theta,data,ev,filt,inv);_,_,pr,_=state_at(ts,ret,data,ev,retfilter,inv);obs=covered(ts,ex);p[~obs]=np.nan;pr[~obs]=np.nan;lower=expit(eta+m-1.96*np.sqrt(v));upper=expit(eta+m+1.96*np.sqrt(v));lower[~obs]=np.nan;upper[~obs]=np.nan
        color='#2166ac' if r.label=='TB' else '#b2182b';x=(ts-onset)/60;ax.plot(x,p,c=color,lw=1,label='Preinterval-trained');ax.fill_between(x,lower,upper,color=color,alpha=.13);ax.plot(x,pr,c='gray',ls='--',lw=.7,label='All-data parameters');ax.axvline(-15,c='gray',lw=.6,ls=':');ax.set_title(f'SZ{r.sz} | {r.label}-source',color=color);ax.set_xlim(-60,0);ax.set_ylim(0,.85);traces[f'sz{r.sz}_minutes']=x;traces[f'sz{r.sz}_p_tb']=p
    axs[0,0].legend(fontsize=7)
    for ax in axs[-1]:ax.set_xlabel('Minutes before seizure')
    for ax in axs[:,0]:ax.set_ylabel('Filtered TB probability')
    fig.suptitle('Forward-parameter reconstruction before all 12 qualified seizures\nParameters use only data before the previous seizure offset; current seizure type never enters inference',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fp=RUN/'figures'
    for ext in ('png','pdf'):fig.savefig(fp/f'preseizure_forward_probabilities.{ext}',dpi=180)
    plt.close(fig);np.savez_compressed(RUN/'preseizure_forward_probability_traces.npz',**traces)
    measures=[('pre15_observed_tb_rate','Observed TB events / hour'),('pre15_observed_tb_share','Observed TB fraction'),('pre15_predicted_tb','Mean inferred TB probability'),('delta_tb_probability','Change vs preceding non-overlapping 15m')];results=[];fig,axs=plt.subplots(2,4,figsize=(16,8));labels=table.label.to_numpy()=='TB'
    for col,(key,title) in enumerate(measures):
        x=table[key].to_numpy();test=permutation(x,labels);results.append(dict(measure=key,**test));ax=axs[0,col]
        for i,r in enumerate(table.itertuples()):
            y=getattr(r,key);xx=(1 if r.label=='TB' else 0)+.035*((i%5)-2);ax.scatter(xx,y,c=r.onset_hours,cmap='viridis',vmin=0,vmax=115,s=45);ax.annotate(str(r.sz),(xx,y),xytext=(3,3+(i%3)*5 if y==0 else 0),textcoords='offset points',fontsize=7,va='center')
        ax.set_xticks([0,1],[f"TA-source ({test['n_ta']})",f"TB-source ({test['n_tb']})"]);ax.set_title(title,fontsize=10);ax.set_xlim(-.25,1.3)
        if key=='pre15_observed_tb_rate':ax.set_yscale('symlog',linthresh=5);ax.set_ylim(0,350)
        if key=='delta_tb_probability':ax.axhline(0,c='gray',lw=.7)
        bx=axs[1,col]
        for label,color in [('TA','#b2182b'),('TB','#2166ac')]:
            part=table[table.label==label];bx.scatter(part.onset_hours,part[key],c=color,label=label,s=40)
            for r in part.itertuples():bx.text(r.onset_hours+1,getattr(r,key),str(r.sz),fontsize=7)
        if key=='pre15_observed_tb_rate':bx.set_yscale('symlog',linthresh=5);bx.set_ylim(0,350)
        if key=='delta_tb_probability':bx.axhline(0,c='gray',lw=.7)
        bx.set_xlabel('Hours since recording origin');bx.set_xlim(0,120)
    axs[1,0].legend(fontsize=8);fig.suptitle('Higher preictal TB rate is not equivalent to an upward state drift\nEach point is one seizure; labels indicate seizure number; both TB-source cases occur in the early record',fontsize=12);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):fig.savefig(fp/f'preseizure_level_vs_change.{ext}',dpi=180)
    plt.close(fig);table.to_csv(RUN/'preseizure_forward_level_change.csv',index=False);write_json(RUN/'preseizure_level_change_audit.json',dict(status='COMPLETE',comparisons=results,tests='Exploratory, unadjusted exact label permutations; time-confounding remains and comparisons not independent',parameter_scope='before preceding seizure offset',n_seizures=12,n_tb=2,not_a_seizure_prediction_model=True))
    readme=fp/'README.md'
    if '### preseizure_forward_probabilities.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### preseizure_forward_probabilities.png\n显示12次合格发作前的连续TB概率，参数仅由上一发作结束前的数据拟合，当前间期段仅用于顺序更新状态。灰色虚线为全数据参数重建；带状范围是固定参数下隐状态发射概率的近似95%边缘范围。\n**关注点**：避免把全数据重建当成前推验证；概率是有事件被观测条件下的模式概率，不是发作概率。\n\n### preseizure_level_vs_change.png\n每点是一场发作，分别比较事件率、TB比例、推断状态的概率水平与相邻不重叠窗口的变化，下排同时显示记录时间。TB型只有两次，均在记录早期。\n**关注点**：较高的发作前TB事件率不自动说明状态单调积累；时间阶段与发作类型的混杂不能靠大量事件数消除。\n')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()

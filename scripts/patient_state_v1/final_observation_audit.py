"""Complete descriptive calibration and current-coordinate retrospective timeline.
No fitting or model selection. Calibration resamples six-hour blocks within fold.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import filter_adf,laplace
from scripts.patient_state_v1.advanced_controls import history_data
OUT=RUN/'final_observation_audit_v1_40'

def save(fig,name,caption):
    fig.savefig(RUN/'figures'/f'{name}.png',dpi=170);fig.savefig(RUN/'figures'/f'{name}.pdf');plt.close(fig)
    p=RUN/'figures/README.md'
    if '### '+name+'.png' not in p.read_text():
        with p.open('a') as f:f.write(f'\n### {name}.png\n{caption}\n')

def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(RUN/'forward_evidence_catalog/current_event_predictions.csv.gz');names=['constant','ewma','ou','ou_history'];rows=[];summary=[]
    fig,axs=plt.subplots(1,2,figsize=(11,4.6));colors=['#777777','#ad874a','#508b70','#296dac']
    for name,color in zip(names,colors):
        d=df[df.model==name].copy();assert len(d)==16157;d['bin']=np.minimum((d.p_tb*10).astype(int),9);d['block']=(d.hour//6).astype(int);d['brier']=(d.y-d.p_tb)**2;d['residual']=d.y-d.p_tb
        groups=[g for _,g in d.groupby(['fold','block'])];folds=np.array([int(g.fold.iloc[0]) for g in groups]);n=np.zeros((len(groups),10));y=n.copy();p=n.copy();br=np.zeros(len(groups));res=br.copy()
        for k,g in enumerate(groups):
            for b,h in g.groupby('bin'):n[k,b]=len(h);y[k,b]=h.y.sum();p[k,b]=h.p_tb.sum()
            br[k]=g.brier.sum();res[k]=g.residual.sum()
        rng=np.random.default_rng(9040);weights=np.zeros((5000,len(groups)),int)
        for fold in np.unique(folds):
            ix=np.where(folds==fold)[0];draw=rng.integers(0,len(ix),(5000,len(ix)))
            for j in range(len(ix)):weights[:,ix[j]]=(draw==j).sum(axis=1)
        bn=weights@n;by=weights@y;bc=np.divide(by,bn,out=np.full_like(by,np.nan),where=bn>0);total=bn.sum(axis=1)
        for b in range(10):
            count=int(n[:,b].sum())
            if count==0:continue
            support=int(np.count_nonzero(n[:,b]));pred=p[:,b].sum()/count;obs=y[:,b].sum()/count;estimable=support>=5 and np.isfinite(bc[:,b]).mean()>=.975
            lo,hi=np.nanquantile(bc[:,b],[.025,.975]) if estimable else [np.nan,np.nan];rows.append(dict(model=name,bin=b,n=count,n_blocks=support,predicted=pred,observed=obs,lower=lo,upper=hi,interval_estimable=estimable))
            if estimable:axs[0].plot([pred,pred],[lo,hi],color=color,alpha=.55,lw=1)
            axs[0].scatter(pred,obs,s=12+count/200,color=color,alpha=.8)
        cal=pd.DataFrame(rows);g=cal[cal.model==name];axs[0].plot(g.predicted,g.observed,color=color,lw=1,label=name)
        blo,bhi=np.quantile((weights@br)/total,[.025,.975]);rlo,rhi=np.quantile((weights@res)/total,[.025,.975]);summary.append(dict(model=name,n_events=len(d),n_blocks=len(groups),brier=d.brier.mean(),brier_lower=blo,brier_upper=bhi,mean_predicted=d.p_tb.mean(),mean_observed=d.y.mean(),observed_minus_predicted=d.residual.mean(),bias_lower=rlo,bias_upper=rhi,descriptive_ece=float(np.sum(g.n*abs(g.observed-g.predicted))/len(d))))
    s=pd.DataFrame(summary);pd.DataFrame(rows).to_csv(OUT/'reliability_bins.csv',index=False);s.to_csv(OUT/'calibration_summary.csv',index=False)
    axs[0].plot([0,1],[0,1],':',c='gray');axs[0].set(xlabel='Mean predicted TB probability',ylabel='Observed TB fraction',xlim=(0,1),ylim=(0,1),title='Fixed 0.1-wide bins; size = event count');axs[0].legend(fontsize=8)
    for k,r in enumerate(s.itertuples()):axs[1].errorbar(r.brier,k,xerr=[[r.brier-r.brier_lower],[r.brier_upper-r.brier]],fmt='o',color=colors[k],capsize=3)
    axs[1].set(yticks=range(4),yticklabels=names,xlabel='Brier score (lower is better)',title='Same 16,157 forward-test events')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Forward probability calibration: frozen development labels\n95% descriptive intervals: 5,000 six-hour block resamples within chronological fold',fontsize=11);fig.tight_layout(rect=(0,0,1,.9))
    save(fig,'forward_probability_calibration','固定16,157个前推测试事件，比较常量、局部记忆及两种OU模型的概率可靠性与Brier分数。误差范围按训练/测试折分层重采样6小时块；少于5个块支持的概率区间不画置信线。\n**关注点**：这是开发记录中的校准描述，不是独立患者验证；校准好坏与生成完整事件过程分开判断。')
    tab=pd.read_csv(RUN/'independent_importance_posterior_v1_28/parameter_intervals.csv');p=tab[(tab.model=='ou_history')&(tab.target=='particle')].set_index('parameter');b=p.loc['baseline_log_odds','median'];gamma=p.loc['fast_history_coefficient','median'];tau=p.loc['tau_minutes','median']/60;sd=p.loc['stationary_sd','median'];theta=np.array([b,gamma,0,np.log(tau),np.log(sd)])
    data=dict(np.load(RUN/'observations.npz'));d=history_data(data);smooth=laplace(theta,d,True,True);assert smooth['converged'];filtered=filter_adf(theta,d,True,order=64)
    ev=pd.read_csv(RUN/'events.csv');assert len(ev)==len(smooth['mode']);ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');labels=wins.set_index('sz').label.to_dict();origin=float(data['origin_epoch']);obs=pd.read_csv(RUN/'state_recap_v1_33/empirical_5min.csv');q=pd.read_csv(RUN/'state_recap_v1_33/query_curve.csv')
    pd.DataFrame(dict(event_index=np.arange(len(ev)),hour=d['t'],epoch=d['epoch'],coverage_segment=ev.coverage_segment,filtered_residual=filtered['mean'],filtered_variance=filtered['variance'],smoothed_residual=smooth['mode'],smoothed_variance=smooth['variance'])).to_csv(OUT/'filtered_and_smoothed_event_states.csv.gz',index=False)
    fig,axs=plt.subplots(4,1,figsize=(14,10),sharex=True)
    axs[0].plot(obs.hour,obs.ta_per_hour,c='#bb4050',lw=.8,label='TA');axs[0].plot(obs.hour,obs.tb_per_hour,c='#296dac',lw=.8,label='TB');axs[0].set_ylabel('Events / observed hour');axs[0].legend(ncol=2,fontsize=8)
    axs[1].scatter(obs.hour,obs.tb_share,s=6,c='#888888',alpha=.5,label='5-min observed TB share')
    for _,part in q[q.eligible].groupby('exposure_segment'):
        axs[1].plot(part.hours,part.slow_tb_propensity,c='#296dac',lw=.9)
        axs[2].plot(part.hours,part.residual_state_mean,c='#296dac',lw=.8);axs[2].fill_between(part.hours,part.residual_state_lower,part.residual_state_upper,color='#296dac',alpha=.13)
    axs[1].set(ylabel='TB share / slow propensity',ylim=(0,1));axs[1].legend(fontsize=8)
    for _,part in ev.assign(epoch=d['epoch']).groupby(['coverage_segment','epoch']):
        ix=part.index.to_numpy()[::10];axs[3].plot(d['t'][ix],smooth['mode'][ix],c='#c18339',lw=.8)
    axs[2].set_ylabel('Filtered x = s - b');axs[3].set_ylabel('Smoothed x = s - b');axs[3].set_xlabel('Hours from first available artifact coverage')
    for ax in axs[2:]:ax.axhline(0,c='gray',ls=':',lw=.7)
    previous=0
    for r in ex.itertuples():
        lo=(r.start_epoch-origin)/3600;hi=(r.end_epoch-origin)/3600
        if lo>previous:
            for ax in axs:ax.axvspan(previous,lo,color='#eeeeee',zorder=-5)
        previous=hi
    for seizure in inv:
        t=(seizure['onset']-origin)/3600;label=labels.get(seizure['sz'],'unknown');color={'TA':'#bb4050','TB':'#296dac'}.get(label,'#aaaaaa')
        for ax in axs:ax.axvline(t,color=color,alpha=.35,lw=.6)
        if label=='TB':axs[0].text(t,1.01,'SZ'+str(seizure['sz']),transform=axs[0].get_xaxis_transform(),ha='center',fontsize=7,color=color)
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,(ex.end_epoch.max()-origin)/3600)
    fig.suptitle('E1146 complete retrospective timeline: OU + one-second label memory\nSeizure onsets: TA red / TB blue / unqualified grey; missing or excluded support shaded',fontsize=12)
    fig.text(.5,.015,'Full-record posterior-median parameters; filtered bands are fixed-parameter Gaussian approximations. Smoothing uses future labels. Statistical clinical boundaries do not identify a biological reset.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.94));save(fig,'patient_state_filter_and_smoother','使用当前OU-history参数和统一的相对基线坐标，展示完整覆盖、TA/TB率、模式份额、过滤与Laplace平滑状态及冻结发作类型。滤波带仅含固定参数的高斯状态不确定性，平滑显式利用未来标签；缺失和临床排除段不连线。\n**关注点**：全数据回顾轨迹不能当作前向预测，发作边界上的统计先验也不是已识别的生物重置。')
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',n_events_per_calibration_model=16157,models=names,reliability_bins='Fixed edges0:0.1:1; 95%pointwise block-bootstrap intervals only with>=5occupied6h blocks; no simultaneous coverage claim',bootstrap='5000six-hour block draws stratified by chronological fold; identical seed and blocks for every model',no_new_fit=True,smoother_converged=smooth['converged'],smoother_iterations=smooth['iterations'],state_coordinate='Residualx=s-b; same OU-history fixed posterior-median parameters as state_recap_v1_33',limits='Development-label calibration, not independent clinical validation. Event-level labels remain fixed; no event-time generation here. Smoother uses future labels; all-data parameters also make filtered reconstruction retrospective.'))
    print(s.to_string(index=False));print('COMPLETE')
if __name__=='__main__':main()

"""Pair model predictions by held-out event and resample physical-time blocks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.round2 import beta_fit_predict,ewma_fit_predict
from scripts.patient_state_v1.analyze_first import metrics

def selected(folder,scope,model=None):
    rows=[]
    for p in folder.glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('status')=='COMPLETE' and r['job']['scope']==scope and (model is None or r['model']==model):rows.append((r,p))
    return max(rows,key=lambda q:q[0]['loglik'])

def block_bootstrap(table,width,replicates=5000):
    rng=np.random.default_rng(20260910+width);rows=[];models=table.model.unique()
    for baseline in ('constant','constant_within_coverage','ewma','ou'):
        for model in models:
            if model==baseline:continue
            m=table[table.model==model].copy();b=table[table.model==baseline];d=m.merge(b,on=['fold','index'],suffixes=('_m','_b'),validate='one_to_one')
            assert len(d)==len(m)==len(b),'Every comparison must retain the same held-out events'
            assert np.max(abs(d.hour_m-d.hour_b))<1e-8,'Event identity must agree with physical time'
            assert np.array_equal(d.y_m,d.y_b),'Event identity must agree with the observed label'
            d['hour']=d.hour_m;d['block']=np.floor(d.hour/width).astype(int);d['delta']=d.score_m-d.score_b
            grouped=d.groupby(['fold','block']).agg(delta=('delta','sum'),n=('index','size')).reset_index();num=np.zeros(replicates);den=np.zeros(replicates)
            for fold,g in grouped.groupby('fold'):
                ix=rng.integers(0,len(g),(replicates,len(g)));num+=g.delta.to_numpy()[ix].sum(axis=1);den+=g.n.to_numpy()[ix].sum(axis=1)
            boot=num/den;rows.append(dict(model=model,baseline=baseline,block_hours=width,n_blocks=len(grouped),n_events=len(d),mean_gain=d.delta.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975)))
    return rows

def main():
    data=dict(np.load(RUN/'observations.npz'));ev=pd.read_csv(RUN/'events.csv');folds=json.loads((RUN/'splits.json').read_text());saved=np.load(RUN/'forward_predictions.npz');records=[];scores=[]
    for f in folds:
        scope=f"fold{f['fold']}";lo,hi=f['test_start'],f['test_end'];pred={m:saved[scope+'_'+m] for m in ('constant','cycle','ou','ou_cycle','last_mark','last_mark_cycle')}
        groups=ev.coverage_segment.to_numpy()*1000+data['epoch'];pred['constant_within_coverage']=beta_fit_predict(data,groups,f['train_end'])[0][lo:hi];pred['ewma']=ewma_fit_predict(data,f['train_end'])[0][lo:hi]
        for model in ('cthmm2','ou_history'):
            r,p=selected(RUN/'advanced_controls_v1_2/fits',scope,model);pred[model]=np.load(p.with_suffix('.npz'))['predict_tb'][lo:hi]
        r,p=selected(RUN/'brownian_drift_v1_5/fits',scope);pred['brownian_drift']=np.load(p.with_suffix('.npz'))['predict_tb'][lo:hi]
        for model,pp in pred.items():
            y=data['y'][lo:hi];n=data['n'][lo:hi];pp=np.clip(pp,1e-12,1-1e-12);ll=y*np.log(pp)+(n-y)*np.log1p(-pp)
            records.append(pd.DataFrame(dict(fold=f['fold'],model=model,index=np.arange(lo,hi),hour=data['t'][lo:hi],score=ll,p_tb=pp,y=y)))
            scores.append(dict(fold=f['fold'],model=model,**metrics(pp,y,n)))
    table=pd.concat(records,ignore_index=True);table.to_csv(RUN/'all_forward_predictions.csv.gz',index=False);s=pd.DataFrame(scores);s.to_csv(RUN/'all_forward_model_scores.csv',index=False)
    intervals=pd.DataFrame(block_bootstrap(table,1)+block_bootstrap(table,6));intervals.to_csv(RUN/'forward_block_uncertainty.csv',index=False)
    fp=RUN/'figures';fig,axs=plt.subplots(1,2,figsize=(13,5));models=['last_mark','constant_within_coverage','ewma','cthmm2','brownian_drift','ou','ou_cycle','ou_history']
    for j,(base,title) in enumerate([('constant','Gain over fixed patient proportion'),('ewma','Gain over recent-proportion smoothing')]):
        for i,model in enumerate(models):
            if model==base:continue
            for width,offset,color in [(1,-.12,'#2166ac'),(6,.12,'#b2182b')]:
                r=intervals[(intervals.model==model)&(intervals.baseline==base)&(intervals.block_hours==width)].iloc[0]
                axs[j].errorbar(r.mean_gain,i+offset,xerr=[[r.mean_gain-r.lower],[r.upper-r.mean_gain]],fmt='o',color=color,ms=4,capsize=2,label=f'{width}h blocks' if i==0 else None)
        axs[j].axvline(0,c='gray',lw=.7);axs[j].set_yticks(range(len(models)),models);axs[j].set_xlabel('Held-out log-score gain / event');axs[j].set_title(title);axs[j].legend(fontsize=8)
    fig.suptitle('Paired time-block uncertainty, stratified by chronological fold\nExploratory model comparisons; events are not independent clinical replicates',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'forward_model_uncertainty.{ext}',dpi=180)
    plt.close(fig)
    h=pd.read_csv(RUN/'horizon_forecasts.csv');agg=h.groupby(['model','horizon_minutes'])[['loglik','n_events']].sum();fig,ax=plt.subplots(figsize=(7.5,4.5))
    for model in ('ou','ou_cycle','cycle'):
        mm=agg.loc[model];base=agg.loc['constant'];gain=(mm.loglik-base.loglik)/mm.n_events;ax.plot(mm.index,gain,'o-',label=model)
    ax.axhline(0,c='gray',lw=.7);ax.set_xlabel('Minutes without intervening event updates');ax.set_ylabel('Gain over constant / future event');ax.legend();ax.set_title('State information has a finite predictive horizon');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'forecast_horizon.{ext}',dpi=180)
    plt.close(fig)
    readme=fp/'README.md';txt=readme.read_text()
    if '### forward_model_uncertainty.png' not in txt:
        with readme.open('a') as f:f.write('\n### forward_model_uncertainty.png\n比较相同测试事件上的模型对数评分差异，分别以1小时和6小时时间块在各前推折内重采样。基线包括固定比例和近期比例平滑，误差条为探索性块重采样区间。\n**关注点**：动态模型是否超过强对照，以及结论对相关性时间块的选择是否敏感；不把事件数当临床重复。\n\n### forecast_horizon.png\n显示在1、5、15、60分钟内完全不消费中间事件标签时，对未来观测事件模式的预测增益。只评价没有跨发作边界的间期窗口，且参数在预测起点前已拟合。\n**关注点**：这是有事件发生条件下的模式预测，不是事件时间或发作预测。\n')
    write_json(RUN/'prediction_review_status.json',dict(status='COMPLETE',n_models=len(s.model.unique()),n_forward_events=table['index'].nunique(),bootstrap_replicates=5000,time_block_hours=[1,6],scope='development-data fixed templates; exploratory model comparison'))
    print(intervals[(intervals.model=='ou')&(intervals.baseline.isin(['constant','ewma','constant_within_coverage']))].to_string(index=False),flush=True)

if __name__=='__main__':main()

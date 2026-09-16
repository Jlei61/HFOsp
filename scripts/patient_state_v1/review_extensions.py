"""Audit prefix stationarity and compare successive observation/state versions."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected,block_bootstrap

def main():
    station=[];audit=[];base=best_fits();fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,model in zip(axs,['ou','ou_history']):
        actual={}
        for scope in ['full','fold0','fold1','fold2']:
            r=base[scope,'ou'] if model=='ou' else selected(RUN/'advanced_controls_v1_2/fits',scope,'ou_history')[0];actual[scope]=r['theta'][-2]
        rows={}
        for p in (RUN/'prefix_stationarity_v1_8/fits').glob(model+'_*.json'):
            r=json.loads(p.read_text());j=r['job']
            if j['model']==model and r['status']=='COMPLETE':rows.setdefault(j['rep'],{})[j['scope']]=r
        for rep,scopes in rows.items():
            if len(scopes)!=4:continue
            vals=np.array([r['theta'][-2] for r in scopes.values()]);station.append(dict(model=model,rep=rep,log_tau_range=np.ptp(vals),first_minus_full=scopes['fold0']['theta'][-2]-scopes['full']['theta'][-2],all_optimizer_success=all(r['success'] for r in scopes.values())))
        frame=pd.DataFrame(station);xx=frame[frame.model==model];actual_difference=actual['fold0']-actual['full'];actual_range=np.ptp(list(actual.values()));ax.hist(xx.first_minus_full,bins=22,color='#7799bb');ax.axvline(actual_difference,c='#b2182b',lw=2,label='Observed');ax.set_title(model);ax.set_xlabel('First-prefix minus full log correlation time');ax.set_ylabel('Stationary-model synthetic sequences');ax.legend()
        clean=xx[xx.all_optimizer_success]
        audit.append(dict(model=model,n_replicates=len(xx),observed_first_minus_full=actual_difference,null_interval=np.quantile(xx.first_minus_full,[.025,.5,.975]),one_sided_calibration_p=(1+(xx.first_minus_full>=actual_difference).sum())/(len(xx)+1),observed_log_tau_range=actual_range,range_calibration_p=(1+(xx.log_tau_range>=actual_range).sum())/(len(xx)+1),all_optimizer_success=bool(xx.all_optimizer_success.all()),n_all_success_replicates=len(clean),successful_only_one_sided_p=(1+(clean.first_minus_full>=actual_difference).sum())/(len(clean)+1)))
    pd.DataFrame(station).to_csv(RUN/'prefix_stationarity_v1_8/refit_summary.csv',index=False);write_json(RUN/'prefix_stationarity_v1_8/scientific_audit.json',dict(status='COMPLETE',results=audit,interpretation='The fitted stationary single-timescale model does not reproduce the observed prefix shift. Alternatives include multiple timescales, changing background, observation misspecification, and boundary assumptions. This is not a unique mechanism identification.'))
    fig.suptitle('Observed prefix change exceeds stationary-model calibration\n128 synthetic full sequences per model; nested prefixes refitted jointly',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(RUN/'figures'/f'prefix_stationarity_calibration.{ext}',dpi=180)
    plt.close(fig)
    df=pd.read_csv(RUN/'all_forward_predictions.csv.gz');data=dict(np.load(RUN/'observations.npz'));folds=json.loads((RUN/'splits.json').read_text());params=[];additional=[]
    for directory,kind in [('activity_adjusted_marks_v1_7','activity'),('two_timescale_marks_v1_9','two')]:
        best={}
        for p in (RUN/directory/'fits').glob('*.json'):
            r=json.loads(p.read_text());j=r['job']
            if r['status']!='COMPLETE':continue
            model=f"activity{j['minutes']}m_{'ou' if j['latent'] else 'history'}" if kind=='activity' else f"two_ou_{int(j['slow_tau'])}h{'_history' if j['history'] else ''}"
            key=(j['scope'],model)
            if key not in best or r['loglik']>best[key][0]['loglik']:best[key]=(r,p)
        for (scope,model),(r,path) in best.items():
            t=r['theta'];params.append(dict(scope=scope,model=model,loglik=r['loglik'],theta=json.dumps(t),success=r['success'],tau_fast_minutes=np.exp(t[-3])*60 if kind=='two' else np.nan,tau_at_lower_bound=bool(kind=='two' and np.exp(t[-3])<1/60*1.001)))
            if scope=='full':continue
            f=next(f for f in folds if scope==f"fold{f['fold']}");lo,hi=f['test_start'],f['test_end'];pp=np.load(path.with_suffix('.npz'))['predict_tb'][lo:hi];pp=np.clip(pp,1e-12,1-1e-12);y=data['y'][lo:hi];ll=y*np.log(pp)+(1-y)*np.log1p(-pp)
            additional.append(pd.DataFrame(dict(fold=f['fold'],model=model,index=np.arange(lo,hi),hour=data['t'][lo:hi],score=ll,p_tb=pp,y=y)))
    full=pd.concat([df,*additional],ignore_index=True);out=RUN/'model_extension_review_v1_9';out.mkdir(exist_ok=True);full.to_csv(out/'forward_predictions.csv.gz',index=False);pd.DataFrame(params).to_csv(out/'parameters.csv',index=False);score=full.groupby(['fold','model']).agg(loglik=('score','sum'),score_per_event=('score','mean'),n_events=('score','size')).reset_index();score.to_csv(out/'forward_scores.csv',index=False)
    intervals=pd.DataFrame(block_bootstrap(full,1)+block_bootstrap(full,6));intervals.to_csv(out/'forward_block_uncertainty.csv',index=False)
    models=['ou','ou_history','activity1m_ou','activity5m_ou','two_ou_6h','two_ou_6h_history','two_ou_24h','two_ou_24h_history'];fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,baseline in zip(axs,['ewma','ou']):
        for i,m in enumerate(models):
            if m==baseline:continue
            r=intervals[(intervals.model==m)&(intervals.baseline==baseline)&(intervals.block_hours==6)].iloc[0];ax.plot([r.lower,r.upper],[i,i],c='#2166ac');ax.plot(r.mean_gain,i,'o',c='#2166ac',ms=4)
        ax.axvline(0,c='gray',lw=.7);ax.set_yticks(range(len(models)),models);ax.set_xlabel('Paired forward log-score gain / event');ax.set_title(f'Against {baseline}; 6-hour block uncertainty')
    fig.suptitle('Additional temporal structure: prediction gain and model limitations\nExploratory comparisons on development data; boundary-hitting timescales are not identified optima',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(RUN/'figures'/f'model_extensions_forward.{ext}',dpi=180)
    plt.close(fig)
    readme=RUN/'figures/README.md'
    if '### prefix_stationarity_calibration.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### prefix_stationarity_calibration.png\n在实际事件时间和边界上，分别从固定参数OU与OU加快速历史模型生成128套标签序列，并重新拟合同样的嵌套训练前缀。红线是真实前缀与全数据的log时间常数差，柱形为该模型假设下的模拟分布。\n**关注点**：模拟保留了前缀估计之间的依赖，偏离只能说明当前固定参数模型不足，不能指定其唯一生理原因。\n\n### model_extensions_forward.png\n比较加入近期活动量或较慢背景后的前向模式预测，误差条使用按折分层的6小时时间块重采样。背景时间常数6或24小时是敏感性设置，没有使用发作标签挑模型。\n**关注点**：v1.9多个较快时间常数碰到1分钟下界，应继续检查下界，不能直接宣称发现1分钟生理过程。\n')
    write_json(out/'status.json',dict(status='COMPLETE',n_models=full.model.nunique(),n_forward_events=full['index'].nunique(),statistical_scope='exploratory, fixed all-data propagation templates',next_question='Resolve fast-time lower bounds and background behavior across excluded ictal intervals before accepting a two-timescale interpretation'))
    print(json.dumps(audit,indent=2,default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':main()

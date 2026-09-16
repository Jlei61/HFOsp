"""First-wave predictive audit and real-data figures; all-data fit remains retrospective."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import filter_adf,laplace,slice_data

def best_fits():
    best={}
    for f in (RUN/'cpu_fits').glob('*.json'):
        r=json.loads(f.read_text())
        if r.get('status')!='COMPLETE':continue
        key=(r['job']['scope'],r['model'])
        if key[0]=='synthetic':continue
        if key not in best or r['loglik']>best[key]['loglik']:best[key]=r
    return best

def history_design(data,tau,cyclic):
    previous=np.r_[0.,data['y'][:-1]/data['n'][:-1]-.5]
    previous[data['reset']]=0
    h=previous*np.exp(-data['dt']/tau)
    return np.column_stack([data['x'][:,:3 if cyclic else 1],h])

def fit_history(data,end,cyclic=False):
    best=None
    for tau in (1/3600,5/3600,1/60,.25,1.,6.):
        x=history_design(data,tau,cyclic);xx=x[:end];y=data['y'][:end];n=data['n'][:end]
        def fun(b):
            eta=xx@b
            return np.sum(n*np.logaddexp(0,eta)-y*eta),xx.T@(n*expit(eta)-y)
        fit=minimize(fun,np.zeros(xx.shape[1]),jac=True,method='L-BFGS-B',bounds=[(-8,8)]*xx.shape[1])
        if best is None or fit.fun<best['nll']:
            best=dict(nll=fit.fun,tau=tau,beta=fit.x,p=expit(x@fit.x))
    return best

def metrics(p,y,n):
    p=np.clip(p,1e-12,1-1e-12)
    return dict(n_events=int(n.sum()),loglik=float(np.sum(y*np.log(p)+(n-y)*np.log1p(-p))),
                log_score_per_event=float(np.sum(y*np.log(p)+(n-y)*np.log1p(-p))/n.sum()),
                brier=float(np.sum(y*(1-p)**2+(n-y)*p**2)/n.sum()),predicted_tb=float(np.dot(n,p)/n.sum()),observed_tb=float(y.sum()/n.sum()))

def main():
    data=dict(np.load(RUN/'observations.npz'));folds=json.loads((RUN/'splits.json').read_text());best=best_fits();rows=[];predictions={}
    for f in folds:
        start,end=f['test_start'],f['test_end'];scope=f"fold{f['fold']}"
        for model in ('constant','cycle','ou','ou_cycle','last_mark','last_mark_cycle'):
            extra={}
            if model.startswith('last_mark'):
                h=fit_history(data,f['train_end'],model.endswith('cycle'));p=h['p'];extra=dict(history_tau_hours=h['tau'])
            else:
                r=best[scope,model];theta=np.array(r['theta'])
                if model.startswith('ou'):p=filter_adf(theta,data,'cycle' in model)['predict_tb']
                else:p=expit(data['x'][:,:len(theta)]@theta)
                extra=dict(tau_hours=r.get('tau_hours'),stationary_sd=r.get('stationary_sd'))
            predictions[scope+'_'+model]=p[start:end]
            rows.append(dict(fold=f['fold'],model=model,**metrics(p[start:end],data['y'][start:end],data['n'][start:end]),**extra))
    scores=pd.DataFrame(rows);scores.to_csv(RUN/'forward_scores.csv',index=False)
    np.savez_compressed(RUN/'forward_predictions.npz',**predictions)
    state={}
    for model in ('ou','ou_cycle'):
        r=best['full',model];theta=np.array(r['theta']);a=filter_adf(theta,data,'cycle' in model);l=laplace(theta,data,'cycle' in model,True)
        state[model]=dict(**{k:v for k,v in a.items() if isinstance(v,np.ndarray)},smooth_mode=l['mode'],smooth_variance=l['variance'])
        np.savez_compressed(RUN/f'full_retrospective_{model}.npz',**state[model])
    syn=[]
    grouped={}
    for f in (RUN/'cpu_fits').glob('syn_*.json'):
        r=json.loads(f.read_text())
        if r.get('status')!='COMPLETE':continue
        j=r['job'];s=j['synthetic'];key=(s['seed'],r['model'])
        if key not in grouped or r['loglik']>grouped[key]['loglik']:grouped[key]=r
    for (seed,model),r in grouped.items():
        s=r['job']['synthetic'];syn.append(dict(seed=seed,model=model,true_tau=s['tau'],true_sd=s['sd'],cycle_amplitude=s['cyclic'][0],tau=r.get('tau_hours'),sd=r.get('stationary_sd'),loglik=r['loglik'],success=r['success']))
    pd.DataFrame(syn).to_csv(RUN/'synthetic_recovery.csv',index=False)
    fp=RUN/'figures';fp.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    origin=float(data['origin_epoch']);ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');labels=wins.set_index('sz').label.to_dict()
    edges=np.arange(0,np.ceil(data['t'].max()*12)/12+1/12,1/12);centers=(edges[:-1]+edges[1:])/2
    tb=np.histogram(data['t'],edges,weights=data['y'])[0];total=np.histogram(data['t'],edges,weights=data['n'])[0]
    exposure=np.zeros(len(centers))
    for row in ex.itertuples():exposure+=np.maximum(0,np.minimum(edges[1:],(row.end_epoch-origin)/3600)-np.maximum(edges[:-1],(row.start_epoch-origin)/3600))
    tr=np.divide(tb,exposure,out=np.full_like(exposure,np.nan),where=exposure>0)
    ar=np.divide(total-tb,exposure,out=np.full_like(exposure,np.nan),where=exposure>0)
    share=np.divide(tb,total,out=np.full_like(exposure,np.nan),where=total>0)
    pd.DataFrame(dict(t_hours=centers,exposure_hours=exposure,n_tb=tb,n_total=total,tb_rate=tr,ta_rate=ar,tb_share=share)).to_csv(RUN/'five_minute_observations.csv',index=False)
    fig,axs=plt.subplots(4,1,figsize=(15,10),sharex=True,gridspec_kw={'height_ratios':[1.2,1,1,1]})
    axs[0].plot(centers,ar,color='#b2182b',lw=.8,label='TA');axs[0].plot(centers,tr,color='#2166ac',lw=.8,label='TB');axs[0].set_ylabel('Events / hour\n5-min bins');axs[0].legend(ncol=2)
    axs[1].scatter(centers,share,s=6,c='gray',alpha=.5,label='Observed 5-min TB share')
    idx=np.arange(0,len(data['t']),10)
    for model,c in [('ou','#4c956c'),('ou_cycle','#6a3d9a')]:axs[1].plot(data['t'][idx],state[model]['predict_tb'][idx],c=c,lw=.8,label=model+' filter')
    axs[1].set_ylim(0,1);axs[1].set_ylabel('TB probability');axs[1].legend(ncol=3,fontsize=8)
    a=state['ou_cycle'];axs[2].plot(data['t'][idx],a['mean'][idx],c='#6a3d9a',lw=.8,label='Filtered mean, all-data parameters')
    axs[2].fill_between(data['t'][idx],a['mean'][idx]-1.96*np.sqrt(a['variance'][idx]),a['mean'][idx]+1.96*np.sqrt(a['variance'][idx]),color='#6a3d9a',alpha=.15)
    axs[2].axhline(0,c='gray',lw=.6);axs[2].set_ylabel('State s\nlog-odds units');axs[2].legend(fontsize=8)
    axs[3].plot(data['t'][idx],a['smooth_mode'][idx],c='#e69f00',lw=.8,label='Laplace smoothed mode (uses future events)');axs[3].set_ylabel('Smoothed s');axs[3].legend(fontsize=8)
    for ax in axs:
        for row in inv:
            t=(row['onset']-origin)/3600;lab=labels.get(row['sz'],'unknown');c={'TA':'#b2182b','TB':'#2166ac'}.get(lab,'#999999')
            ax.axvline(t,c=c,alpha=.3,lw=.7)
        for lo,hi in zip(edges[:-1][exposure==0],edges[1:][exposure==0]):ax.axvspan(lo,hi,color='#dddddd',alpha=.5,lw=0)
    axs[-1].set_xlabel('Hours from first available artifact coverage');fig.suptitle('E1146 first wave: retrospective state reconstruction; gaps shaded; seizure onsets colored\nFrozen interictal labels; 44,282 events; state inference independent of Z/M',fontsize=12);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'full_state_timeline.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));models=list(scores.model.unique());colors=plt.cm.tab10(np.arange(len(models)))
    for i,model in enumerate(models):
        g=scores[scores.model==model];base=scores[scores.model=='constant'].set_index('fold')
        gain=g.log_score_per_event.to_numpy()-base.loc[g.fold].log_score_per_event.to_numpy()
        axs[0].plot(g.fold,gain,'o-',label=model,color=colors[i]);axs[1].plot(g.fold,g.brier,'o-',color=colors[i])
    axs[0].axhline(0,c='gray',lw=.7);axs[0].set_ylabel('Held-out log-score gain / event');axs[1].set_ylabel('Held-out Brier score');axs[0].legend(fontsize=8)
    for ax in axs:ax.set_xlabel('Chronological forward fold');ax.set_xticks([0,1,2])
    fig.suptitle('Only past labels and training-period parameters used for each prediction');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'forward_prediction.{ext}',dpi=180)
    plt.close(fig)
    (fp/'README.md').write_text('### full_state_timeline.png\n显示完整可用序列的5分钟TA/TB率、模式占比和连续状态；PDF为同图。灰区无覆盖，竖线为发作起点，颜色沿用冻结的宽频合格类型，其他发作为灰色。全数据参数的滤波与使用未来事件的平滑均属回顾性描述。\n**关注点**：不能将全数据重建当作前向预测，状态单位为统计log-odds。\n\n### forward_prediction.png\n显示连续时间前推的三个测试段，比较常数、昼夜背景、OU状态以及上一事件记忆对照。测试预测仅用该时刻之前的标签和训练期参数，冻结模板仍来自开发数据。\n**关注点**：逐段增益是否稳定，以及状态是否超过短期标签记忆；图待人工审阅。\n')
    summary=dict(status='FIRST_WAVE_ANALYZED_PENDING_NUMERICAL_AND_SCIENTIFIC_REVIEW',best={str(k):v for k,v in best.items()},forward_scores=rows,
                 n_synthetic_best=len(syn),finished_unix=time.time(),human_visual_review='pending')
    write_json(RUN/'first_wave_summary.json',summary);print(scores.to_string(index=False),flush=True)

if __name__=='__main__':main()

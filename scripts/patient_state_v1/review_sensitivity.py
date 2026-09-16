"""Keep structural and boundary uncertainty separate from fitted-model error bars."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.sensitivity import dataset
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected

def main():
    folder=RUN/'boundary_sensitivity_v1_5';rows=[];sources={}
    for p in (folder/'fits').glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('status')!='COMPLETE':continue
        j=r['job'];key=(j['scenario'],j['scope'],r['model'])
        if key not in sources or r['loglik']>sources[key]['loglik']:sources[key]=r
    for (scenario,scope,model),r in sources.items():
        row=dict(scenario=scenario,scope=scope,model=model,tau_minutes=r['tau_hours']*60,stationary_sd=r['stationary_sd'],n_train=r['n_train'],n_retained=r['n_retained'],success=r['success'],n_reset=r['n_reset'])
        if scope!='full':
            d,keep=dataset(scenario);lo=r['n_train'];hi=np.sum(keep[:r['job']['original_test_end']]);p=d['y'][:lo].sum()/d['n'][:lo].sum();y=d['y'][lo:hi];n=d['n'][lo:hi];constant=np.sum(y*np.log(p)+(n-y)*np.log1p(-p));row.update(forward_loglik=r['forward']['loglik'],constant_loglik=constant,n_test=int(n.sum()),gain_vs_matched_constant=(r['forward']['loglik']-constant)/n.sum())
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(folder/'sensitivity_summary.csv',index=False)
    baseline=best_fits();fullbase=[]
    for m in ['ou','ou_history']:
        r=baseline['full','ou'] if m=='ou' else selected(RUN/'advanced_controls_v1_2/fits','full','ou_history')[0]
        fullbase.append(dict(scenario='original',scope='full',model=m,tau_minutes=np.exp(r['theta'][-2])*60,stationary_sd=np.exp(r['theta'][-1]),n_train=44282,n_retained=44282))
    full=pd.concat([pd.DataFrame(fullbase),frame[frame.scope=='full']]);full.to_csv(folder/'full_parameter_comparison.csv',index=False)
    order=['original','post15','post60','pre15','pre60','both60','carry_ictal','reset_coverage'];labels=['Original','Exclude post 15m','Exclude post 60m','Exclude pre 15m','Exclude pre 60m','Exclude both 60m','Carry across ictal gaps','Independent coverage starts']
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for m,color in [('ou','#2166ac'),('ou_history','#b2182b')]:
        pp=full[full.model==m].set_index('scenario').loc[order];axs[0].plot(pp.tau_minutes,np.arange(len(order)),'o-',c=color,label=m)
        ff=frame[(frame.model==m)&(frame.scope!='full')]
        for scenario,g in ff.groupby('scenario'):
            gain=(g.forward_loglik.sum()-g.constant_loglik.sum())/g.n_test.sum();axs[1].scatter(gain,order.index(scenario),c=color,s=30)
    axs[0].set_xlabel('Fitted correlation time (minutes)');axs[0].set_yticks(range(len(order)),labels);axs[0].legend();axs[0].set_title('Model structure and exclusion change tau')
    axs[1].set_yticks(range(len(order)),labels);axs[1].axvline(0,c='gray',lw=.7);axs[1].set_xlabel('Forward gain / event over same-data constant');axs[1].set_title('State information beyond seizure neighborhoods')
    fig.suptitle('Boundary sensitivity: exploratory robustness, not independent replication',fontsize=12);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(RUN/'figures'/f'state_boundary_sensitivity.{ext}',dpi=180)
    plt.close(fig)
    brown=[]
    for scope in ['full','fold0','fold1','fold2']:
        r,p=selected(RUN/'brownian_drift_v1_5/fits',scope);t=r['theta'];brown.append(dict(scope=scope,intercept=t[0],drift_per_hour=t[1],diffusion_per_sqrt_hour=np.exp(t[2]),initial_sd=np.exp(t[3]),loglik=r['loglik'],success=r['success']))
    pd.DataFrame(brown).to_csv(RUN/'brownian_drift_v1_5/parameter_summary.csv',index=False)
    readme=RUN/'figures/README.md'
    if '### state_boundary_sensitivity.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### state_boundary_sensitivity.png\n比较排除发作前后15或60分钟、跨发作延续以及覆盖段独立起点等设置下的相关时间与前向预测。每个排除条件的固定比例基线只用相同保留训练事件估计。\n**关注点**：状态信息并不只来自紧邻发作的事件，但相关时间随模型结构与分析范围变化，不能视为唯一生理时间常数。\n')
    write_json(folder/'review_status.json',dict(status='COMPLETE',n_best_fits=len(frame),all_selected_optimizer_success=bool(frame.success.all()),scope='same patient, frozen templates, no new biological evidence'))
    print(full[['scenario','model','tau_minutes','n_train']].to_string(index=False),flush=True)

if __name__=='__main__':main()

"""Display approximate profile support and independent likelihood checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json

def main():
    out=RUN/'free_tau_profile_v1_14';best={}
    for p in (out/'fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['carry'],j['method'],j['tau'])
        if r['status']=='COMPLETE' and (key not in best or r['loglik']>best[key]['loglik']):best[key]=r
    rows=[]
    for (carry,method,tau),r in best.items():rows.append(dict(carry=carry,method=method,tau_hours=tau,loglik=r['loglik'],success=r['success'],fast_tau_seconds=np.exp(r['theta'][2])*3600,fast_sd=np.exp(r['theta'][3]),background_sd=np.exp(r['theta'][4]),theta=json.dumps(r['theta'])))
    df=pd.DataFrame(rows);df['relative_loglik']=df.loglik-df.groupby(['carry','method']).loglik.transform('max');df.to_csv(out/'profile_summary.csv',index=False);fig,axs=plt.subplots(2,2,figsize=(12,8));support=[]
    for col,carry in enumerate([False,True]):
        for method,c in [('adf','#2166ac'),('laplace','#b35806')]:
            g=df[(df.carry==carry)&(df.method==method)].sort_values('tau_hours');axs[0,col].plot(g.tau_hours,g.relative_loglik,'o-',c=c,label=method.upper());axs[1,col].plot(g.tau_hours,g.fast_tau_seconds,'o-',c=c);q=g[g.relative_loglik>=-1.92];support.append(dict(carry=carry,method=method,grid_argmax_hours=float(g.loc[g.loglik.idxmax(),'tau_hours']),grid_support_hours=q.tau_hours.tolist(),all_optimizer_success=bool(g.success.all())))
        axs[0,col].axhline(-1.92,c='gray',ls=':',lw=.8);axs[0,col].set_title('Background carried through seizures' if carry else 'Independent background interval priors');axs[0,col].set_ylabel('Relative approximate profile log likelihood');axs[0,col].legend();axs[1,col].set_ylabel('Refitted fast correlation time (s)');axs[1,col].set_yscale('log')
        for ax in axs[:,col]:ax.set_xscale('log');ax.set_xlabel('Background correlation time (hours)');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('What is identified depends on the observation model and likelihood approximation\nFast label history retained; dotted line is a likelihood-ratio reference, not validated coverage',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'free_background_time_profile.{ext}',dpi=180)
    plt.close(fig)
    particle=[]
    for p in (RUN/'free_timescale_particle_audit_v1_14').glob('carry*.json'):
        r=json.loads(p.read_text());particle.append({k:r[k] for k in ['carry','history','method','particle_logmean','loglik_sd','likelihood_ess','adf_at_solution','laplace_at_solution','logmean_bootstrap_interval']})
    pd.DataFrame(particle).to_csv(out/'independent_likelihood_checks.csv',index=False)
    write_json(out/'scientific_audit.json',dict(status='COMPLETE',grid_profiles=support,particle_checks=particle,interpretation='Hours-scale background is supported within these mark models, while fast time and amplitude depend substantially on approximation. The profile is model-conditional and does not identify a physical SNN state.',monte_carlo_limit='Particle log-mean uncertainty is material for small differences; ADF/Laplace fit differences must not be read as exact likelihood differences'))
    readme=RUN/'figures/README.md'
    if '### free_background_time_profile.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### free_background_time_profile.png\n固定不同慢背景相关时间后重新拟合其余参数，比较Gaussian ADF与Laplace两种似然近似，以及发作排除区间间的背景延续方式。上排为相对profile似然，下排为同一次拟合得到的快速时间常数；均保留1秒事件历史项。\n**关注点**：小时尺度背景支持范围和快速时间尺度的近似敏感性；-1.92参考线未经过覆盖率校准，不能直接称为生理参数的95%置信区间。\n')
    print(json.dumps(support,indent=2))
if __name__=='__main__':main()

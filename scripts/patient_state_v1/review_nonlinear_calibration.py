"""Judge apparent potential structure against a known single-well generator."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.nonlinear_calibration import best_real
OUT=RUN/'nonlinear_calibration_v1_21'
def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='COMPLETE';files=list((OUT/'fits').glob('*.json'));assert len(files)==256;rows=[]
    for p in files:
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];rows.append(dict(seconds=j['seconds'],rep=j['rep'],kind=j['kind'],loglik=r['loglik'],success=r['success'],k=r['theta'][3] if j['kind']=='quartic' else np.nan))
    df=pd.DataFrame(rows);df.to_csv(OUT/'fit_summary.csv',index=False);best=best_real();summary=[];pairs=[];fig,axs=plt.subplots(2,2,figsize=(11,8))
    for col,sec in enumerate([15,60]):
        sub=df[df.seconds==sec];table=sub.pivot(index='rep',columns='kind',values='loglik');assert len(table)==64;gain=table.quartic-table.ou;q=sub[sub.kind=='quartic'].set_index('rep').loc[table.index];success=sub.pivot(index='rep',columns='kind',values='success').all(axis=1);real_gain=best[(sec,'full','quartic')]['loglik']-best[(sec,'full','ou')]['loglik'];real_k=best[(sec,'full','quartic')]['theta'][3];tail=int((gain>=real_gain-1e-9).sum());row=dict(seconds=sec,n=64,patient_gain=real_gain,patient_k=real_k,null_gain_quantiles=np.quantile(gain,[.025,.5,.975]),null_negative_k_fraction=float((q.k<0).mean()),null_as_negative_as_patient=int((q.k<=real_k).sum()),gain_as_large_count=tail,monte_carlo_p=(tail+1)/65,n_optimizer_success_pairs=int(success.sum()),success_only_p=(1+int((gain[success]>=real_gain).sum()))/(1+int(success.sum())));summary.append(row)
        for rep in table.index:pairs.append(dict(seconds=sec,rep=int(rep),gain=float(gain.loc[rep]),k=float(q.loc[rep,'k']),both_optimizer_success=bool(success.loc[rep])))
        axs[0,col].hist(gain,bins=18,color='#9db9ce');axs[0,col].axvline(real_gain,c='#b2182b',lw=1.5,label=f'Patient: {real_gain:.2f}');axs[0,col].set_title(f'{sec}-second observation bins');axs[0,col].set_xlabel('Quartic minus OU training log likelihood');axs[0,col].set_ylabel('Known single-OU sequences');axs[0,col].legend(fontsize=8)
        axs[1,col].scatter(q.k,gain,c=np.where(success,'#2166ac','#b2182b'),s=22,alpha=.65);axs[1,col].scatter([real_k],[real_gain],marker='*',c='#b2182b',s=150,label='Patient');axs[1,col].axvline(0,c='gray',lw=.8);axs[1,col].set_xlabel('Fitted quartic curvature k (negative permits two wells)');axs[1,col].set_ylabel('Quartic minus OU log likelihood');axs[1,col].legend(fontsize=8)
        for ax in axs[:,col]:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A negative fitted curvature must be judged against single-well data\n64 known-OU sequences per bin size; actual event counts and observation schedule fixed; Monte Carlo calibration',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'nonlinear_drift_ou_null_calibration.{ext}',dpi=180)
    plt.close(fig);pd.DataFrame(pairs).to_csv(OUT/'paired_null_results.csv',index=False);write_json(OUT/'scientific_review.json',dict(status='COMPLETE',results=summary,interpretation='Training likelihood and potential-shape evidence are judged conditionally against a stationary OU. Even a rejection of this null would not identify bistability: nonstationarity, informative observation and other drift shapes remain alternatives, and held-out improvement was not stable.',uncertainty='Finite Monte Carlo test with 64 sequences per setting; bin-size checks share one patient and are not independent confirmation; no seizure labels used'))
    readme=RUN/'figures/README.md'
    if '### nonlinear_drift_ou_null_calibration.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### nonlinear_drift_ou_null_calibration.png\n每种分箱下生成64条真实为单OU的标签序列，沿用患者事件数量、覆盖和排除边界，再拟合OU及非线性drift。上排比较非线性训练增益，下面显示同一次模拟的势曲率与增益，星号为患者。\n**关注点**：负曲率和训练增益在没有双稳态的生成真值下有多常见；该校准与前推预测应一起解释。\n')
    print(json.dumps(summary,indent=2,default=lambda x:x.tolist()))
if __name__=='__main__':main()

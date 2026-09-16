"""Separate evidence for nonlinearity, two wells, and numerical fit quality."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
OUT=RUN/'shape_and_approximation_review_v1_24'
def main():
    OUT.mkdir(exist_ok=True);root=RUN/'nonlinear_shape_profile_v1_23';assert json.loads((root/'status.json').read_text())['status']=='COMPLETE';best={}
    for p in (root/'fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';key=(r['job']['seconds'],r['job']['k'])
        if key not in best or r['loglik']>best[key]['loglik']:best[key]=r
    rows=[dict(seconds=sec,k=k,loglik=r['loglik'],success=r['success']) for (sec,k),r in best.items()];df=pd.DataFrame(rows);df['relative_loglik']=df.loglik-df.groupby('seconds').loglik.transform('max');df.to_csv(OUT/'shape_profile.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(11,4.5));profile=[]
    for ax,sec in zip(axs,[15,60]):
        part=df[df.seconds==sec].sort_values('k');ax.plot(part.k,part.relative_loglik,'o-',c='#2166ac');ax.axvline(0,c='gray',ls='--');ax.set_xscale('symlog',linthresh=2);ax.set_xlabel('Quartic curvature k');ax.set_ylabel('Relative profile log likelihood');ax.set_title(f'{sec}-second observation bins');ax.spines[['top','right']].set_visible(False);single=part[part.k>=0].loglik.max();overall=part.loglik.max();profile.append(dict(seconds=sec,best_grid_k=float(part.loc[part.loglik.idxmax(),'k']),single_well_loglik_loss=float(overall-single),all_success=bool(part.success.all()),near_peak_grid_k=part[part.relative_loglik>=-1.92].k.tolist()))
    fig.suptitle('Single-well and shallow two-well forces have similar likelihood\nOther parameters refitted at each k; profile reference ranges are not calibrated biological confidence intervals',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'nonlinear_single_well_profile.{ext}',dpi=180)
    plt.close(fig);root=RUN/'joint_adf_audit_v1_23';assert json.loads((root/'status.json').read_text())['status']=='COMPLETE';refs=[]
    for p in (root/'reference').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';refs.append(dict(source=r['job']['source'],points=r['job']['points'],reference=r['factorized_reference_loglik'],adf=r['adf_loglik']))
    ref=pd.DataFrame(refs);ref.to_csv(OUT/'independent_likelihood_reference.csv',index=False);g=ref[ref.points==2048].set_index('source').reindex(['laplace','adf','rate_grid_refit']);fig,axs=plt.subplots(1,3,figsize=(14,5));x=np.arange(len(g));axs[0].plot(x,g.reference-g.reference.max(),'o-',c='#b2182b',label='Factorized grid reference');axs[0].plot(x,g.adf-g.adf.max(),'o-',c='#2166ac',label='Gaussian ADF objective');axs[0].set_xticks(x,['Laplace fit','ADF fit','Grid rate fit'],rotation=20);axs[0].set_ylabel('Relative log likelihood');axs[0].set_title('Same independent-state model');axs[0].legend(fontsize=8,loc='center right')
    rows=[]
    for version,folder in [('Laplace','joint_two_state_generation_v1_20'),('ADF','joint_adf_audit_v1_23')]:
        for p in (RUN/folder/'runs').glob('*.json'):
            r=json.loads(p.read_text());assert r['status']=='COMPLETE';rows.append(dict(version=version,model=r['job']['model'],rep=r['job']['rep'],rate=r['rate_per_hour'],tb=r['tb_fraction']))
    gen=pd.DataFrame(rows);assert len(gen)==512;gen.to_csv(OUT/'joint_generation_versions.csv',index=False);real=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());intervals=[];order=[('Laplace','independent'),('ADF','independent'),('Laplace','coupled'),('ADF','coupled')]
    for ax,measure,target in zip(axs[1:],['rate','tb'],[real['rate_per_hour'],real['tb_fraction']]):
        ax.axvline(target,c='k',ls='--',label='Patient')
        for i,(version,model) in enumerate(order):
            q=gen[(gen.version==version)&(gen.model==model)][measure].quantile([.025,.5,.975]);ax.errorbar(q.iloc[1],i,xerr=[[q.iloc[1]-q.iloc[0]],[q.iloc[2]-q.iloc[1]]],fmt='o',c='#2166ac' if version=='ADF' else '#777777',capsize=2);intervals.append(dict(version=version,model=model,measure=measure,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],patient=target))
        ax.set_yticks(range(4),[f'{v} / {m}' for v,m in order],fontsize=8);ax.set_xlabel('Events / observed hour' if measure=='rate' else 'TB event fraction');ax.set_title('Generated total rate' if measure=='rate' else 'Generated mode balance');ax.legend(fontsize=8)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A better approximate objective does not guarantee better inference\nReference factorization applies only to independent states; 128 generated sequences per parameter set',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_likelihood_approximation_audit.{ext}',dpi=180)
    plt.close(fig);pd.DataFrame(intervals).to_csv(OUT/'generation_intervals.csv',index=False);write_json(OUT/'scientific_review.json',dict(status='COMPLETE',shape_profile=profile,approximation=dict(adf_parameter_improvement_in_adf_objective=float(g.loc['adf','adf']-g.loc['laplace','adf']),adf_parameter_change_in_reference=float(g.loc['adf','reference']-g.loc['laplace','reference']),interpretation='The approximate objective reverses the ranking of these parameter points. ADF refitting is not accepted as improved inference. Coupled fits require an independent particle likelihood check.'),scope='These are statistical observation/drift diagnostics, not a new SNN baseline or physiological parameter validation'))
    readme=RUN/'figures/README.md'
    if '### nonlinear_single_well_profile.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### nonlinear_single_well_profile.png\n固定势曲率后重新拟合基线、时间和幅度，比较负曲率双井侧与非负曲率单井侧的似然。两种分箱分别绘制，时间与幅度会随曲率重新适配。\n**关注点**：拒绝线性OU是否真的要求双井；当前单井侧的似然损失很小。\n\n### joint_likelihood_approximation_audit.png\n左侧在可精确分解的独立状态模型中，用细化状态网格核对参数点的似然排序；右侧比较相应参数生成的总事件率和TB比例。每个生成条件含128条匹配覆盖的新序列。\n**关注点**：Gaussian ADF评分更高是否真实提高似然，及生成单个指标改善是否足够支持模型。\n')
    print(json.dumps(profile,indent=2));print(pd.DataFrame(intervals).to_string(index=False))
if __name__=='__main__':main()

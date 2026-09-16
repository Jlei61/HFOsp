"""Keep label-definition sensitivity separate from predictive-model selection."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,ROOT,write_json
from scripts.patient_state_v1.analyze_first import best_fits
OUT=RUN/'label_bank_sensitivity_v1_19'
def main():
    source=ROOT/'results/interictal_propagation_masked/per_subject/epilepsiae_1146.json';original=json.loads(source.read_text());bank=np.load(ROOT/'results/topic5_preseizure_template_share/epilepsiae_1146/event_index.npz');assert np.array_equal(original['adaptive_cluster']['labels'],bank['template_label'])
    provenance=dict(source=str(source),field='adaptive_cluster.labels',producer='scripts/analyze_e1146_preseizure_template_share.py::event_data',name='Historical masked-rank adaptive-cluster labels',not_equated_to='Current Timing-only full-fit or current Timing+Space event labels',historical_file_name='observations_timing_only.npz retained as an output identifier only')
    write_json(OUT/'label_provenance.json',provenance)
    for filename in ['contract.json','label_crosswalk.json']:
        r=json.loads((OUT/filename).read_text());key='sensitivity_bank' if filename=='contract.json' else 'sensitivity';r[key]=provenance['name'];write_json(OUT/filename,r)
    best={}
    for p in (OUT/'fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];key=(j['scope'],j['model'])
        if key not in best or r['loglik']>best[key]['loglik']:best[key]=r
    rows=[]
    for (scope,model),r in best.items():
        t=np.array(r['theta']);i=1 if model=='ou' else 2;rows.append(dict(bank='Historical masked rank',scope=scope,model=model,baseline=t[0],tau_hours=np.exp(t[-1] if model=='two_ou_history' else t[i]),fast_seconds=np.exp(t[i])*3600 if model=='two_ou_history' else np.nan,success=r['success']))
    baseline=best_fits()
    for scope in ['full','fold0','fold1','fold2']:
        choices=[]
        for p in (RUN/'cpu_fits').glob('*.json'):
            r=json.loads(p.read_text());j=r.get('job',{})
            if j.get('scope')==scope and j.get('model')=='ou' and r.get('status')=='COMPLETE':choices.append(r)
        if choices:
            r=max(choices,key=lambda q:q['loglik']);t=r['theta'];rows.append(dict(bank='Current Timing+Space',scope=scope,model='ou',baseline=t[0],tau_hours=np.exp(t[1]),fast_seconds=np.nan,success=r['success']))
        for model,folder,pattern in [('ou_history','advanced_controls_v1_2',f'{scope}_ou_history_*.json'),('two_ou_history','two_timescale_free_tau_v1_13',f'{scope}_carry1_hist1_adf_*.json')]:
            choices=[json.loads(p.read_text()) for p in (RUN/folder/'fits').glob(pattern)];r=max(choices,key=lambda q:q['loglik']);t=r['theta'];rows.append(dict(bank='Current Timing+Space',scope=scope,model=model,baseline=t[0],tau_hours=np.exp(t[-1] if model=='two_ou_history' else t[2]),fast_seconds=np.exp(t[2])*3600 if model=='two_ou_history' else np.nan,success=r['success']))
    df=pd.DataFrame(rows);df.to_csv(OUT/'parameter_comparison.csv',index=False);fig,axs=plt.subplots(1,3,figsize=(12,4.8))
    for ax,model in zip(axs,['ou','ou_history','two_ou_history']):
        for bank,color in [('Current Timing+Space','#2166ac'),('Historical masked rank','#b35806')]:
            g=df[(df.bank==bank)&(df.model==model)].set_index('scope').reindex(['fold0','fold1','fold2','full']);ax.plot(range(4),g.tau_hours,'o-',c=color,label=bank)
        ax.set_xticks(range(4),['Prefix 1','Prefix 2','Prefix 3','Full'],rotation=20);ax.set_ylabel('Inferred OU time (hours)');ax.set_title(model.replace('_',' '));ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle('Same 44,282 event identities; different frozen propagation-label definitions\nSingle OU and history fits: Laplace; two OU history: ADF background time',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'label_definition_sensitivity.{ext}',dpi=180)
    plt.close(fig);write_json(OUT/'review.json',dict(status='COMPLETE',n_fits=24,all_success=bool(df.success.all()),rule='Do not compare absolute likelihoods across different response labels. This is a target-definition sensitivity, not independent replication or a reason to replace the primary bank.',provenance=provenance))
    readme=RUN/'figures/README.md'
    if '### label_definition_sensitivity.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### label_definition_sensitivity.png\n在相同44,282个间期事件上，比较当前Timing+Space与历史masked-rank adaptive-cluster标签推断出的状态时间。前3点分别只拟合对应训练前缀，末点为全记录拟合。\n**关注点**：时间尺度不稳定是否主要来自标签定义；不同标签的绝对似然不用于模型优劣比较，主标签未被替换。\n')
    print(df[df.scope=='full'].to_string(index=False))
if __name__=='__main__':main()

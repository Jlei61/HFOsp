"""Matched marginal-kernel control for the offline joint feature audit.

Same raw features, group scaling, and three Gaussian bandwidths. Compare the
mean of marginal kernels with their same-bandwidth product. Patient self-kernel
constant is omitted because it is identical for every candidate within a score.
No labels and no simulation dispatch here.
"""
from pathlib import Path
import csv,json,pickle,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import torch
from scripts import audit_topic4_joint_observation as audit
from scipy.stats import spearmanr


def main():
    torch.set_num_threads(2);device=audit.pilot.dense.gpu_device()
    with (audit.SOURCE/'analysis/training_objective.pkl').open('rb') as f:old=pickle.load(f)
    mapping=audit.JointMap(old)
    ev,names,_=audit.patient_source.patient()
    def groups(t):
        raw=old.raw_features(t)
        scaled={k:torch.as_tensor(raw[k]*np.sqrt(mapping.weights[k])/mapping.bandwidths[k],device=device) for k in audit.KEYS}
        return [scaled['rank'],torch.column_stack([scaled[k] for k in ['scl_pairs','icl_pairs','rod_lag']]),scaled['mask']]
    target=groups(ev.fit)
    def means(x,y):
        d=[]
        for a,b in zip(x,y):
            d.append((a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0))
        additive=sum(torch.exp(-v/(2*s*s)).sum() for v in d for s in [.5,1.,2.])/9
        total=sum(d)
        joint=sum(torch.exp(-total/(2*s*s)).sum() for s in [.5,1.,2.])/3
        return float(additive),float(joint)
    rows=[]
    for source in sorted((audit.SOURCE/'scores').glob('*.json')):
        score=audit.rt.read(source)
        if score['J'] is None:continue
        _,t,cn=audit.pilot.dense.load_times(Path(score['source']));assert cn==names
        x=groups(t);n=len(t);within=means(x,x);cross=means(x,target)
        values=[(w-n)/(n*(n-1))-2*c/(n*len(ev.fit)) for w,c in zip(within,cross)]
        rows.append(dict(candidate=score['candidate'],N=n,matched_additive_minus_constant=values[0],exact_joint_minus_constant=values[1]))
    ref=next(r for r in rows if r['candidate']=='dense_axis3_12')
    for row in rows:
        row['matched_additive_delta_from_incumbent']=row['matched_additive_minus_constant']-ref['matched_additive_minus_constant']
        row['exact_joint_delta_from_incumbent']=row['exact_joint_minus_constant']-ref['exact_joint_minus_constant']
    with (audit.OUT/'matched_kernel_control.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    original={r['candidate']:r for r in csv.DictReader((audit.OUT/'candidate_scores.csv').open())}
    add=min(rows,key=lambda r:r['matched_additive_minus_constant']);joint=min(rows,key=lambda r:r['exact_joint_minus_constant'])
    summary=dict(n_candidates=len(rows),control='identical features, bandwidths and group weights; only kernel combination changes',
        matched_additive_best=add,exact_joint_best=joint,
        approximate_exact_joint_rank_rho=float(spearmanr([float(original[r['candidate']]['joint_scaled']) for r in rows],
            [r['exact_joint_minus_constant'] for r in rows]).statistic),
        identical_winner=add['candidate']==joint['candidate'],
        selected=[r for r in rows if r['candidate'] in ['g2_b03_p01','dense_axis3_12','lf_bo02_01']],
        candidate_diagnostics={r['candidate']:{k:original[r['candidate']][k] for k in ['ALL_SCL_error','TA_SCL_error','TB_SCL_error']}
                               for r in [add,joint]},
        note='Differences from the incumbent are comparable within each statistic. Scores with the omitted patient constant are only ranking statistics, not absolute distances.')
    audit.rt.write(audit.OUT/'matched_kernel_control.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

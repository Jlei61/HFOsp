"""Assemble comparable forward evidence without mixing likelihood targets.

Current-event mark scores share exact event IDs. Fixed-horizon, binned mark,
joint time/mark, and altered-label/cohort diagnostics remain separate products.
No parameters are fitted and no best family is chosen by this consolidation.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_activity_clock import paired
OUT=RUN/'forward_evidence_catalog'

def main():
    OUT.mkdir(parents=True,exist_ok=True);data=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());expected=np.concatenate([np.arange(f['test_start'],f['test_end']) for f in folds]);parts={};sources={};duplicates=[]
    def add(part,source):
        name=part.model.iloc[0];part=part[['model','fold','index','hour','y','p_tb','score']].sort_values(['fold','index']).reset_index(drop=True)
        assert len(part)==16157 and np.array_equal(part['index'],expected);assert not part.duplicated(['fold','index']).any();ix=part['index'].to_numpy();assert np.array_equal(part.y,data['y'][ix]);assert np.max(abs(part.hour-data['t'][ix]))<1e-8;assert np.all((part.p_tb>0)&(part.p_tb<1));ll=part.y*np.log(part.p_tb)+(1-part.y)*np.log1p(-part.p_tb);assert np.max(abs(ll-part.score))<1e-8
        for f in folds:assert np.all(part.loc[(ix>=f['test_start'])&(ix<f['test_end']),'fold']==f['fold'])
        if name in parts:
            err=float(np.max(abs(parts[name].p_tb-part.p_tb)));assert err<1e-10,(name,err,source);duplicates.append(dict(model=name,source=source,max_probability_difference=err));sources[name].append(source)
        else:parts[name]=part;sources[name]=[source]
    for file in ['all_forward_predictions.csv.gz','model_extension_review_v1_9/forward_predictions.csv.gz','drift_version_review_v1_14/forward_predictions.csv.gz','activity_extension_review_v1_16/forward_predictions.csv.gz','activity_clock_v1_27/forward_predictions.csv.gz','learned_history_decay_v1_32/forward_predictions.csv.gz','learned_initial_offset_v1_36/forward_predictions.csv.gz']:
        df=pd.read_csv(RUN/file)
        for _,part in df.groupby('model',sort=False):add(part,file)
    extra={}
    for directory,kind in [('two_timescale_marks_v1_11','fixed_fast_seconds'),('two_timescale_free_tau_v1_13','free_without_history')]:
        for path in (RUN/directory/'fits').glob('*.json'):
            r=json.loads(path.read_text());j=r['job']
            if r['status']!='COMPLETE' or j['scope']=='full':continue
            if kind=='free_without_history':
                if j['history'] or not r['feasible']:continue
                name=f"OU2_nohist_{j['method']}_carry{int(j['carry'])}"
            else:name=f"OU2_fixed{int(j['slow_tau'])}h_hist{int(j['history'])}_carry{int(j['carry_slow'])}_lower1s"
            key=(name,j['scope'])
            if key not in extra or r['loglik']>extra[key][1]['loglik']:extra[key]=(path,r)
    rebuilt={}
    for (name,scope),(path,r) in extra.items():
        f=next(f for f in folds if scope==f"fold{f['fold']}");ix=np.arange(f['test_start'],f['test_end']);pp=np.clip(np.load(path.with_suffix('.npz'))['predict_tb'][ix],1e-12,1-1e-12);y=data['y'][ix];rebuilt.setdefault(name,[]).append(pd.DataFrame(dict(model=name,fold=f['fold'],index=ix,hour=data['t'][ix],y=y,p_tb=pp,score=y*np.log(pp)+(1-y)*np.log1p(-pp))))
    for name,records in rebuilt.items():add(pd.concat(records,ignore_index=True),'Training-likelihood-selected prefix fits: '+','.join(str(extra[name,f"fold{f['fold']}"][0].relative_to(RUN)) for f in folds))
    full=pd.concat(parts.values(),ignore_index=True);full.to_csv(OUT/'current_event_predictions.csv.gz',index=False);scores=full.groupby('model').agg(n_events=('index','size'),log_score=('score','sum'),score_per_event=('score','mean')).reset_index();scores.to_csv(OUT/'current_event_scores.csv',index=False);intervals=[]
    for name in parts:
        for baseline in ['constant','constant_within_coverage','ewma','ou','ou_history']:
            if name!=baseline:intervals.append(dict(model=name,baseline=baseline,**paired(parts[name],parts[baseline],6)))
    ints=pd.DataFrame(intervals);ints.to_csv(OUT/'current_event_block_comparisons.csv',index=False)
    for name,source in [('fixed_horizon_comparisons','tuned_horizon_review_v1_17/block_uncertainty.csv'),('joint_grid_comparisons','joint_resolution_review_v1_25/forward_summary.csv'),('binned_and_approximation_comparisons','joint_nonlinear_review_v1_21/forward_summary.csv')]:
        df=pd.read_csv(RUN/source);df.to_csv(OUT/(name+'.csv'),index=False)
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',n_current_event_models=len(parts),n_events_per_model=16157,current_event_sources=sources,duplicate_checks=duplicates,identity_validation='Exact frozen fold/event index and label match; physical time match and Bernoulli score recomputation for every row',current_event_uncertainty='5000paired6hour block resamples stratified by chronological fold, identical bootstrap scheme for all comparisons',separate_targets=dict(current_event='Mode of an occurring event given all completed previous labels and its occurrence time',fixed_horizon='Same13184eligible future events, with no intermediate label updates;0to120min horizons; compare horizon-tuned memory',binned_nonlinear='Within-bin mark counts;15/60s; quartic vs matched OU numerical control',joint='Joint event-time and mark density, including silent exposure;256/512grids; normalized per16157observed events but not a mark-only score'),not_pooled='Altered clinical exclusions and alternative label banks change evaluation eligibility/target; their within-contract sensitivity results remain in separate folders',limits='This catalog does not choose or accept a model by a maximum development score. Labels use the full development record, numerical approximation and generation adequacy remain separate. Full-data-only direct grid refit has no new prefix-fit validation.'))
    text='# 前推证据索引\n\n同一事件上的标签预测、固定时间跨度预测、分箱标签预测、事件时刻与标签联合密度分开报告，不能放进一个排名。所有结果仍使用开发记录中冻结的标签模板。\n\n'
    text+=f'- `current_event_scores.csv` 与 `current_event_block_comparisons.csv`：{len(parts)}个模型/数值版本，共同16,157个事件；逐行重新核对ID、标签、时间及Bernoulli评分。5个明确对照，6小时块重采样。原始模型身份及来源见 `scientific_audit.json`。\n'
    text+='- `fixed_horizon_comparisons.csv`：共同13,184个事件，0–120分钟不读中间标签；来自对应时距调优的记忆对照。\n- `binned_and_approximation_comparisons.csv`：非线性drift与匹配OU分箱对照，以及联合似然的高斯近似诊断。\n- `joint_grid_comparisons.csv`：完整联合事件时间/标签密度的256/512状态网格验证。包含无事件的观测时间，不等于仅模式评分。\n\n固定慢时间常数、不同边界处理和数值近似均保留独立名称。边界/标签敏感性改变了事件资格或标签，不与本表绝对分数合并；见 `../boundary_sensitivity_v1_5/` 和 `../label_bank_sensitivity_v1_19/`。这份索引不以最高探索分数宣布机制成立。\n'
    (OUT/'README.md').write_text(text);print(json.dumps(dict(status='COMPLETE',n_current_event_models=len(parts),n_events_per_model=16157,n_pairwise_comparisons=len(ints),duplicate_checks=len(duplicates))))

if __name__=='__main__':main()

"""Report which mark-sequence discrepancies remain after parameter integration."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_generation_matched import interval_table,figure
ROOT=RUN/'posterior_predictive_marks_v1_29'

def measures(r):
    w=r['windows']['5'];return dict(tb_fraction=r['tb_fraction'],adjacency_excess=r['adjacent_excess'],share_iqr=w['tb_share_quantiles'][3]-w['tb_share_quantiles'][1],share_lag1=w['tb_share_autocorrelation_lags124'][0],share_lag2=w['tb_share_autocorrelation_lags124'][1],share_lag4=w['tb_share_autocorrelation_lags124'][2])

def main():
    status=json.loads((ROOT/'status.json').read_text());assert status['status'] in ['COMPLETE','PARTIAL_PENDING_IMPORTANCE'];models=status['models'];rows=[]
    for p in (ROOT/'runs').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';rows.append(dict(model=r['job']['model'],rep=r['job']['rep'],**measures(r)))
    df=pd.DataFrame(rows);assert len(df)==512*len(models);raw=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());real=measures(raw);df.to_csv(ROOT/'summary.csv',index=False);tab=interval_table(df,real);tab['patient_in_central95_predictive_range']=(tab.patient>=tab.lower)&(tab.patient<=tab.upper);tails=[]
    for row in tab.itertuples():
        vals=df[df.model==row.model][row.measure].to_numpy();tails.append(float(np.mean(vals>=row.patient)))
    tab['predictive_fraction_at_least_patient']=tails;tab.to_csv(ROOT/'distribution_comparison.csv',index=False);write_json(ROOT/'scientific_audit.json',dict(status='COMPLETE' if len(models)==2 else 'PARTIAL_ONE_MODEL',models=models,results=tab.to_dict('records'),interpretation='Central95 predictive ranges and tail fractions are posterior predictive descriptions, not frequentist p-values or untouched validation',fixed_inputs='Event timing, event count, coverage and ictal-exclusion boundaries; only mode labels generated',limits='These selected summaries do not establish full joint distribution recovery. Prefix-dependent parameters, seizure-type confounding and physiological state mapping remain independent issues.'))
    figure(tab,[('tb_fraction','Overall TB fraction'),('adjacency_excess','Adjacent same-label excess'),('share_iqr','5-min TB fraction: interquartile width'),('share_lag1','5-min TB fraction: lag1'),('share_lag2','5-min TB fraction: lag2'),('share_lag4','5-min TB fraction: lag4')],models,['OU' if m=='ou' else 'OU + short memory' for m in models],'Conditional mark generation with parameter uncertainty\n512 new state/label sequences per model; full-data posterior check; event times are fixed','posterior_predictive_mode_sequences')
    readme=RUN/'figures/README.md'
    if '### posterior_predictive_mode_sequences.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### posterior_predictive_mode_sequences.png\n对每个通过重要性采样诊断的模型，从参数后验抽样后生成512条新状态和标签轨迹，固定患者事件时刻及覆盖边界。展示模式份额、相邻聚集和5分钟份额的分布宽度与滞后相关，误差条为包含参数不确定性的预测分布中央95%范围。\n**关注点**：点参数下的偏差是否仍在；这是全数据后验预测检查，事件率和事件间隔在此不能作为生成成功。\n')
    print(tab.to_string(index=False))

if __name__=='__main__':main()

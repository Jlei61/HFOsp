"""Exposure-matched generation from the terminal direct joint-grid refinement.

The baseline retains the existing Laplace point and first128 generated sequences.
The refined point is selected only by likelihood, never by the generated readouts.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.review_generation import extract
from scripts.patient_state_v1.review_generation_matched import interval_table,figure
import scripts.patient_state_v1.matched_generation as matched

OUT=RUN/'joint_grid_refit_generation_v1_26'

def main():
    path=RUN/'joint_grid_refit_v1_26/result.json';r=json.loads(path.read_text());assert r['status'] in ['OPTIMIZER_COMPLETE','OPTIMIZER_LIMIT_OR_FAILURE','BUDGET_LIMIT'];assert np.isfinite(r['refined_grid']['loglik']);b,a,c,ts,ss,tr,sr=unpack(r['best_grid']['theta'],True)
    matched.OUT=OUT;jobs=[];rows=[]
    for rep in range(128):
        old=json.loads((RUN/'joint_two_state_generation_v1_20/runs'/f'joint_two_state_coupled_{rep:03d}.json').read_text());assert old['status']=='COMPLETE';rows.append(dict(model='laplace_point',rep=rep,**extract(old)))
        cfg=dict(hours=24.,step=.5,deadtime=.25,a=a,b=b,tau_r=tr,sd_r=sr,tau_s=ts,sd_s=ss,c=c,kind=2,gamma=0.)
        jobs.append(dict(version='direct_grid',model='coupled',rep=rep,seed=old['job']['seed'],config=cfg))
    write_json(OUT/'contract.json',dict(question='Does directly improving the full joint likelihood repair patient distribution generation?',source_result=str(path),source_fit_status=r['status'],initial_grid256_loglik=r['initial_grid']['loglik'],best_grid256_loglik=r['best_grid']['loglik'],fine_grid512_loglik=r['refined_grid']['loglik'],theta=r['best_grid']['theta'],n_new_sequences=128,baseline='Existing128Laplace coupled-point sequences, same seed IDs; no selection or patient event replay',conditioned='Actual coverage and ictal exclusions; stationary initial distributions per inferred interval law; no biological seizure reset claimed',numerics='0.5s simulation-step discrepancy was separately checked against0.1s in v1.31; direct parameter point checked on512state grid',scope='Full-data development generation check; does not inherit earlier prefix-fit forward validation'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(matched.worker,j) for j in jobs])):
            z=f.result();assert z['status']=='COMPLETE';rows.append(dict(model='direct_grid_point',rep=z['job']['rep'],**extract(z)))
            if (i+1)%16==0:print(json.dumps(dict(done=i+1,total=128)),flush=True)
    df=pd.DataFrame(rows);assert len(df)==256;df.to_csv(OUT/'summary.csv',index=False);real=extract(json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text()));tab=interval_table(df,real);tab['patient_in_central95_range']=(tab.patient>=tab.lower)&(tab.patient<=tab.upper);tab.to_csv(OUT/'distribution_comparison.csv',index=False)
    figure(tab,[('rate','Events / observed hour'),('tb_fraction','Overall TB fraction'),('interval_median','Median interval (s)'),('fano','5-min count Fano factor'),('adjacency_excess','Adjacent same-label excess'),('share_lag1','5-min TB fraction: lag1')],['laplace_point','direct_grid_point'],['Laplace point','Direct joint-grid point'],'Does direct joint-likelihood refinement improve generated distributions?\n128 sequences per parameter point; actual gaps/exclusions fixed; likelihood-only parameter selection','joint_grid_refit_generation')
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',fit_status=r['status'],n_generated=128,parameter_source=str(path),comparison=tab.to_dict('records'),limits='Central95 ranges are simulation descriptions at fixed parameters, not parameter uncertainty or hypothesis-test p-values. A finite set of summaries cannot establish full distribution recovery. The terminal optimizer status is retained; a budget-limited point is not an optimum. Forward validation for this full-data-refined point was not run.'))
    write_json(OUT/'status.json',dict(status='COMPLETE',n_new_sequences=128,n_comparison_sequences=256,finished_unix=time.time()))
    readme=RUN/'figures/README.md'
    if '### joint_grid_refit_generation.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### joint_grid_refit_generation.png\n比较原Laplace参数与直接二维联合网格似然细化参数，各自在实际覆盖和临床排除边界上生成128条新事件时间及模式序列。误差条为固定参数模拟间中央95%范围，参数选择只依据训练似然。\n**关注点**：更可靠的联合似然是否解决模式份额、事件密度和间隔的同时失配；预算退出不得称为找到最优参数，也不继承旧前缀拟合的前推验收。\n')
    print(tab.to_string(index=False))

if __name__=='__main__':main()

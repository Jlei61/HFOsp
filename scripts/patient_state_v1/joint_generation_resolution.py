"""Check whether the joint generator discrepancy is a simulation-step artifact.

Existing Laplace parameters and patient exposure are fixed. This is a numerical
diagnostic, not a new fitted model or a new choice of parameters after generation.
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

OUT=RUN/'joint_generation_resolution_v1_31'

def main():
    matched.OUT=OUT;jobs=[];sources={};rows=[]
    for coupled in [False,True]:
        model='coupled' if coupled else 'independent'
        rs=[json.loads(p.read_text()) for p in (RUN/'joint_two_state_v1_20/fits').glob(f'b5_full_c{int(coupled)}_*.json')]
        r=max(rs,key=lambda r:r['loglik']);sources[model]=r
        b,a,c,ts,ss,tr,sr=unpack(r['theta'],coupled)
        for rep in range(64):
            oldpath=RUN/'joint_two_state_generation_v1_20/runs'/f'joint_two_state_{model}_{rep:03d}.json'
            old=json.loads(oldpath.read_text());assert old['status']=='COMPLETE';assert old['job']['config']['step']==.5
            config=dict(hours=24.,step=.1,deadtime=.25,a=a,b=b,tau_r=tr,sd_r=sr,tau_s=ts,sd_s=ss,c=c,kind=2,gamma=0.)
            for key,val in config.items():
                if key!='step':assert old['job']['config'][key]==val
            rows.append(dict(model=model+'_0.5s',rep=rep,**extract(old)))
            jobs.append(dict(version='simulation_step',model=model,rep=rep,seed=old['job']['seed'],config=config))
    write_json(OUT/'contract.json',dict(question='Does reducing the generator state-hold step from 0.5 to 0.1 seconds remove the rate/mode mismatch?',sources=sources,n_new_sequences=128,n_per_model=64,comparison='First64 preexisting0.5s replicates, and64 new0.1s replicates per model. Seed IDs match, but changing random-draw order does not create pathwise common random numbers.',fixed='Fitted parameters, initial law, actual coverage, clinical exclusions, and250ms effective retained support',selection='All scheduled replicates included; no parameter selection from generated summaries',scope='Numerical generation check, not physical refractoriness or a new model fit'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(matched.worker,j) for j in jobs])):
            r=f.result();assert r['status']=='COMPLETE';rows.append(dict(model=r['job']['model']+'_0.1s',rep=r['job']['rep'],**extract(r)))
            if (i+1)%16==0:print(json.dumps(dict(done=i+1,total=len(jobs))),flush=True)
    df=pd.DataFrame(rows);assert len(df)==256;df.to_csv(OUT/'summary.csv',index=False)
    real=extract(json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text()));tab=interval_table(df,real);tab.to_csv(OUT/'distribution_comparison.csv',index=False)
    models=['independent_0.5s','independent_0.1s','coupled_0.5s','coupled_0.1s']
    figure(tab,[('rate','Events / observed hour'),('tb_fraction','Overall TB fraction'),('interval_median','Median interval (s)'),('fano','5-min count Fano factor'),('adjacency_excess','Adjacent same-label excess'),('share_lag1','5-min TB fraction: lag1')],models,['Independent:0.5s','Independent:0.1s','Coupled:0.5s','Coupled:0.1s'],'Joint generation: simulation-step sensitivity\n64 sequences per condition; parameter points and observation exposure fixed','joint_generator_step_audit')
    readme=RUN/'figures/README.md'
    if '### joint_generator_step_audit.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### joint_generator_step_audit.png\n固定联合模型的两组既有参数点，分别以0.5秒与0.1秒状态保持步长生成事件，实际覆盖和发作排除边界相同，每条件64条轨迹。误差条为模拟间中央95%范围，种子编号相同但不是同一连续噪声路径。\n**关注点**：此前事件率和模式份额偏差是否来自仿真步长；这是数值检查，不是新的生理模型。\n')
    write_json(OUT/'status.json',dict(status='COMPLETE',n_new_sequences=128,n_comparison_sequences=256,finished_unix=time.time()))
    print(tab[tab.measure.isin(['rate','tb_fraction','interval_median'])].to_string(index=False))

if __name__=='__main__':main()

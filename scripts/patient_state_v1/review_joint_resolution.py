"""Audit observation-bin and state-grid resolution without changing the models."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.review_joint_nonlinear import bootstrap
OUT=RUN/'joint_resolution_review_v1_25'

def main():
    OUT.mkdir(exist_ok=True);best={};allfits=[]
    for p in (RUN/'joint_two_state_v1_20/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];allfits.append(r)
        assert r['status']=='COMPLETE' and r['success'] and r['state_converged']
        key=(j['seconds'],j['scope'],j['coupled'])
        if key not in best or r['loglik']>best[key]['loglik']:best[key]=r
    assert len(allfits)==72 and len(best)==24
    rows=[]
    for (sec,scope,c),r in sorted(best.items()):
        b,a,coupling,ts,ss,tr,sr=unpack(r['theta'],c)
        rows.append(dict(seconds=sec,scope=scope,coupled=c,baseline_logodds=b,baseline_lograte=a,c=coupling,mode_tau_minutes=ts*60,mode_sd=ss,rate_tau_seconds=tr*3600,rate_sd=sr,laplace_loglik=r['loglik'],source=r['job']['id']))
    params=pd.DataFrame(rows);params.to_csv(OUT/'bin_parameter_sensitivity.csv',index=False)
    contract=RUN/'joint_two_state_v1_20/contract.json';c=json.loads(contract.read_text());c['status']='COMPLETE: 72 fits at 1, 5, 15 seconds; all optimizers and latent mode solves successful';c['likelihood']='Effective shifted renewal event density plus binary marks; state constant inside 1/5/15-second integration bins';write_json(contract,c)
    root=RUN/'joint_grid_refinement_v1_24';grid=[]
    for p in (root/'evaluations').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];assert r['status']=='COMPLETE'
        grid.append(dict(scope=j['scope'],method=j['method'],coupled=j['coupled'],grid=j['grid'],loglik=r['loglik'],max_transition_mass_loss=r['max_transition_mass_loss'],max_sampled_edge_mass=r['max_sampled_posterior_edge_mass'],source=str(p)))
    tab=pd.DataFrame(grid);tab.to_csv(OUT/'grid_likelihoods.csv',index=False)
    complete=(root/'status.json').exists() and len(grid)==24
    summaries=[];d=prepare(5,.25)
    for size in [256,512]:
        frames=[]
        for fold in range(3):
            paths=[root/'evaluations'/f'fold{fold}_laplace_c{c}_g{size}.npz' for c in [0,1]]
            if not all(p.exists() for p in paths):break
            z=[np.load(p) for p in paths];lo,hi=int(z[0]['test_lo']),int(z[0]['test_hi']);assert all(int(a['test_lo'])==lo and int(a['test_hi'])==hi for a in z)
            frames.append(pd.DataFrame(dict(fold=fold,bin_index=np.arange(lo,hi),gain=z[1]['loglik_terms'][lo:hi]-z[0]['loglik_terms'][lo:hi],events=d['n'][lo:hi],hours=d['physical_exposure'][lo:hi],block=np.floor(d['t'][lo:hi]/6).astype(int))))
        if len(frames)==3:
            frame=pd.concat(frames,ignore_index=True);assert int(frame.events.sum())==16157
            frame.to_csv(OUT/f'grid{size}_forward_terms.csv',index=False);summaries.append(dict(grid=size,**bootstrap(frame)))
    pd.DataFrame(summaries).to_csv(OUT/'forward_summary.csv',index=False)
    write_json(OUT/'status.json',dict(status='COMPLETE' if complete else 'PARTIAL_GRID_PENDING',n_bin_fits=len(allfits),n_grid_evaluations=len(grid),forward=summaries,interpretation='Grid evaluates existing Laplace/ADF parameters, not a grid-optimized model; observation-bin and grid sensitivity must be reported separately. Held-out score is joint time/mark density including silent exposure, not seizure prediction.'))
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for c,color in [(False,'#557b99'),(True,'#bf6845')]:
        x=params[(params.scope=='full')&(params.coupled==c)].sort_values('seconds');axs[0].plot(x.seconds,x.mode_tau_minutes,'o-',color=color,label='Rate independent' if not c else 'Rate coupled')
    axs[0].set(xlabel='Observation bin (s)',ylabel='Mode-state time constant (min)',title='Laplace-fit bin sensitivity');axs[0].legend(fontsize=8)
    for method,ls in [('laplace','-'),('adf','--')]:
        for c,color in [(False,'#557b99'),(True,'#bf6845')]:
            x=tab[(tab.scope=='full')&(tab.method==method)&(tab.coupled==c)].sort_values('grid')
            if len(x):axs[1].plot(x.grid,x.loglik-x.loglik.iloc[-1],'o'+ls,color=color,label=f'{method.upper()}, c{int(c)}')
    axs[1].set(xlabel='Grid cells per state axis',ylabel='Log likelihood minus finest available',title='Full two-state integration');axs[1].legend(fontsize=7)
    for i,r in enumerate(summaries):axs[2].errorbar(r['gain_per_event'],i,xerr=[[r['gain_per_event']-r['lower']],[r['upper']-r['gain_per_event']]],fmt='o',color='#557b99',capsize=3)
    axs[2].axvline(0,color='.5',ls=':');axs[2].set(yticks=range(len(summaries)),yticklabels=[f"{r['grid']} x {r['grid']}" for r in summaries],xlabel='Coupled minus independent log score / event',title='Strict forward intervals; 6-h blocks')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Observation coupling: numerical resolution audit (patient E1146)',fontsize=12);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_resolution_audit.{ext}',dpi=180)
    plt.close(fig)
    readme=RUN/'figures/README.md';s=readme.read_text()
    if '### joint_resolution_audit.png' not in s:
        with readme.open('a') as f:f.write('\n### joint_resolution_audit.png\n左图比较1、5、15秒观测分箱下的状态时间常数，中图核对二维状态网格积分的分辨率，右图使用同一组严格前推事件区间比较联合观测评分。参数来自既有训练前缀，二维网格只重新积分，没有重新优化。\n**关注点**：数值精度、观测分箱及模型适用性是不同问题；联合评分改善不代表发作预测或物理机制确认。\n')
    print(params[params.scope=='full'].to_string(index=False));print(json.dumps(dict(complete=complete,grid_evaluations=len(grid),forward=summaries),indent=2))

if __name__=='__main__':main()

"""Inspect actual chains; a finished process is not a convergence result."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json

def longest_rejection(accepted):
    answer=[]
    for chain in accepted.T:
        longest=current=0
        for x in chain:
            current=0 if x else current+1;longest=max(longest,current)
        answer.append(longest)
    return answer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plots',action='store_true');args=ap.parse_args();out=RUN/'posterior_diagnostics';out.mkdir(exist_ok=True);results=[]
    for name,folder in [('ou','particle_posterior_v1_2/ou'),('ou_history','particle_posterior_v1_2/ou_history'),('two_ou_history','two_scale_particle_posterior_v1_15_16chains')]:
        root=RUN/folder
        if not (root/'checkpoint.npz').exists():continue
        meta=json.loads((root/'checkpoint.json').read_text());z=dict(np.load(root/'checkpoint.npz'));contract=json.loads((root/'contract.json').read_text());iteration=min(meta['iteration'],len(z['samples'])-1);warmup=contract['warmup'];base=dict(model=name,iteration=iteration,postwarmup_draws_per_chain=max(iteration-warmup,0),chains=z['samples'].shape[1],sampler_status=meta['status'],last_saved=meta['last_saved'])
        if iteration-warmup<200:base.update(status='INSUFFICIENT_POSTWARMUP',acceptance_gate_pass=False);results.append(base);continue
        a=z['samples'][warmup+1:iteration+1].transpose(1,0,2);names=contract['parameter_names'];idata=az.from_dict(posterior={n:a[:,:,i] for i,n in enumerate(names)});summary=az.summary(idata,hdi_prob=.95,round_to=8).reset_index(names='parameter');summary['mcse_relative_to_sd']=summary.mcse_mean/summary.sd;summary.to_csv(out/(name+'_summary.csv'),index=False);good=bool((summary.r_hat<1.01).all() and (summary.ess_bulk>=400).all() and (summary.ess_tail>=400).all() and (summary.mcse_relative_to_sd<.05).all());accepted=z['accepted'][warmup:iteration];base.update(status='DIAGNOSTICS_PASS' if good else 'NOT_CONVERGED',acceptance_gate_pass=good,summary=summary.to_dict('records'),acceptance_per_chain=accepted.mean(0),longest_rejection_per_chain=longest_rejection(accepted),gate='rank-normalized split Rhat <1.01, bulk/tail ESS >=400, mean MCSE/SD <.05 for every parameter',scope='Numerical posterior under the stated conditional mark model, not model adequacy or physical parameter identification');results.append(base)
        if args.plots:
            fig,axs=plt.subplots(len(names),2,figsize=(12,2.1*len(names)),squeeze=False)
            for i,n in enumerate(names):
                for chain in range(min(8,a.shape[0])):axs[i,0].plot(np.arange(warmup+1,iteration+1),a[chain,:,i],lw=.35,alpha=.55)
                axs[i,0].set_ylabel(n,fontsize=8);axs[i,1].hist(a[:,:,i].ravel(),bins=50,density=True,color='#688daa',alpha=.8);axs[i,1].set_xlabel(n,fontsize=8)
                for ax in axs[i]:ax.spines[['top','right']].set_visible(False)
            axs[-1,0].set_xlabel('Iteration');fig.suptitle(f'{name}: {base["status"]}\nFirst 8 chain IDs shown; all chains enter diagnostics; posterior is conditional on model',fontsize=11);fig.tight_layout(rect=(0,0,1,.95))
            for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'posterior_{name}_diagnostics.{ext}',dpi=160)
            plt.close(fig)
    report=dict(status='LIVE_DIAGNOSTIC_SNAPSHOT',created_unix=time.time(),results=results);write_json(out/'status.json',report);print(json.dumps([dict(model=r['model'],status=r['status'],iteration=r['iteration'],postwarmup=r['postwarmup_draws_per_chain'],max_rhat=max([s['r_hat'] for s in r.get('summary',[])],default=None)) for r in results],indent=2))
    if args.plots:
        readme=RUN/'figures/README.md';txt=readme.read_text()
        for r in results:
            name=f"posterior_{r['model']}_diagnostics.png"
            if (RUN/'figures'/name).exists() and '### '+name not in txt:
                with readme.open('a') as f:f.write(f'\n### {name}\n展示该模型warmup后的前8个固定链编号轨迹和全部链的参数直方图，图题标明当前数值诊断是否通过。Rhat、有效样本和Monte Carlo误差计算使用全部链。\n**关注点**：运行结束不等于后验收敛，通过数值诊断也不等于真实数据生成机制已验证。\n')
if __name__=='__main__':main()

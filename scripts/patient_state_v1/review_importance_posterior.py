"""Independent-weight diagnostics, parameter intervals, and approximation check."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,arviz as az
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
ROOT=RUN/'independent_importance_posterior_v1_28'

def quantile(x,w,q):
    order=np.argsort(x);xx=x[order];ww=w[order];cdf=np.cumsum(ww)-ww/2;return np.interp(q,cdf,xx)

def summarize(theta,logweight,names):
    valid=np.isfinite(logweight);x=theta[valid];lw=logweight[valid];lw-=logsumexp(lw);w=np.exp(lw);ess=1/np.sum(w*w);_,k=az.psislw(lw);mean=np.sum(w[:,None]*x,axis=0);sd=np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0));mcse=np.sqrt(np.sum(w[:,None]**2*(x-mean)**2,axis=0));rows=[]
    for i,name in enumerate(names):
        low,median,high=quantile(x[:,i],w,[.025,.5,.975]);rows.append(dict(parameter=name,mean=float(mean[i]),sd=float(sd[i]),mcse=float(mcse[i]),mcse_over_sd=float(mcse[i]/sd[i]),lower=float(low),median=float(median),upper=float(high)))
    return dict(importance_ess=float(ess),maximum_weight=float(w.max()),pareto_k=float(k),n_nonzero=len(w),summary=rows),valid,w

def main():
    results=[];allrows=[];plots=[]
    for model in ['ou','ou_history']:
        root=ROOT/model
        if not (root/'status.json').exists():continue
        status=json.loads((root/'status.json').read_text());assert status['status']=='COMPLETE';z=dict(np.load(root/'weighted_samples.npz'));contract=json.loads((root/'contract.json').read_text());assert len(z['theta'])==contract['n_proposals'];sources=[str(root/'weighted_samples.npz')]
        for ext in sorted(root.glob('extension_*')):
            if not (ext/'status.json').exists():continue
            es=json.loads((ext/'status.json').read_text());assert es['status']=='COMPLETE';ez=dict(np.load(ext/'weighted_samples.npz'));ec=json.loads((ext/'contract.json').read_text());assert ec['model']==model and ec['source']==contract['source'];assert not set(np.unique(ez['replicate']))&set(np.unique(z['replicate']));assert len(ez['theta'])==es['n_proposals'];z={key:np.concatenate([value,ez[key]]) for key,value in z.items()};sources.append(str(ext/'weighted_samples.npz'))
        np.savez_compressed(root/'posterior_samples.npz',**z);raw=z['theta'];names=contract['source']['parameter_names'];physical=raw.copy();physical[:,-2]=60*np.exp(raw[:,-2]);physical[:,-1]=np.exp(raw[:,-1]);physical_names=names[:-2]+['tau_minutes','stationary_sd']
        for target in ['particle','laplace']:
            report,valid,w=summarize(raw,z[target+'_logweight'],names);phys,_,_=summarize(physical,z[target+'_logweight'],physical_names);replicates=[];means=[];errors=[]
            for rep in np.unique(z['replicate']):
                ix=z['replicate']==rep;r,_,_=summarize(raw[ix],z[target+'_logweight'][ix],names);r['replicate']=int(rep);replicates.append(r);means.append([v['mean'] for v in r['summary']]);errors.append([v['mcse'] for v in r['summary']])
            means=np.array(means);errors=np.array(errors);zmax=0.
            for i in range(len(means)):
                for j in range(i):zmax=max(zmax,float(np.max(abs(means[i]-means[j])/np.maximum(np.sqrt(errors[i]**2+errors[j]**2),1e-12))))
            passed=bool(report['importance_ess']>=1000 and report['maximum_weight']<=.01 and report['pareto_k']<.7 and all(r['importance_ess']>=100 for r in replicates) and all(r['mcse_over_sd']<.05 for r in report['summary']) and zmax<4.)
            result=dict(model=model,target=target,status='IMPORTANCE_DIAGNOSTICS_PASS' if passed else 'IMPORTANCE_DIAGNOSTICS_NOT_PASSED',**report,n_total_proposals=len(raw),sample_sources=sources,posterior_samples=str(root/'posterior_samples.npz'),physical_summary=phys['summary'],independent_replicates=replicates,max_pairwise_mean_mcse_z=zmax,gate='Raw ESS>=1000, max normalized weight<=.01, Pareto k<.7, each independent replicate ESS>=100, every raw-parameter mean MCSE/SD<.05, max pairwise replicate mean difference<4 combined MCSE',interpretation='Conditional numerical posterior within one fixed model; raw weights used, Pareto smoothing is diagnostic only');results.append(result)
            for row in phys['summary']:allrows.append(dict(model=model,target=target,diagnostic_status=result['status'],**row))
            plots.append((model,target,physical[valid],w,passed))
    assert results,'No completed importance experiments yet';pd.DataFrame(allrows).to_csv(ROOT/'parameter_intervals.csv',index=False);write_json(ROOT/'scientific_audit.json',dict(status='COMPLETE' if len(results)==4 else 'PARTIAL_ONE_MODEL',results=results,limits='Importance diagnostics do not prove absence of missed posterior regions; proposal has full support and frozen broad component, and earlier parameter profiles supply additional coverage evidence. Model misspecification and physiological units remain separate. PMMH chains are unchanged.'))
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for row,model in enumerate(['ou','ou_history']):
        entries=[p for p in plots if p[0]==model]
        if not entries:continue
        for col,(index,label) in enumerate([(-2,'Time constant (min)'),(-1,'Stationary SD (log-odds)')]):
            limits=[quantile(x[:,index],w,[.005,.995]) for _,_,x,w,_ in entries];edges=np.linspace(min(a[0] for a in limits),max(a[1] for a in limits),45)
            for _,target,x,w,passed in entries:axs[row,col].hist(x[:,index],bins=edges,weights=w,histtype='stepfilled' if target=='particle' else 'step',alpha=.45 if target=='particle' else 1.,color='#2166ac' if target=='particle' else '#777777',lw=1.5,label='Particle weights' if target=='particle' else 'Laplace weights')
            ps=next(p[4] for p in entries if p[1]=='particle');axs[row,col].set(xlabel=label,ylabel='Posterior mass / bin',title=f'{model}: '+('importance diagnostics pass' if ps else 'diagnostics not passed'));axs[row,col].spines[['top','right']].set_visible(False)
    axs[0,0].legend(fontsize=8);fig.suptitle('Model-conditional uncertainty from independent parameter draws\nIndependent groups; identical proposal points for particle and Laplace comparisons; central99% display',fontsize=11);fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'independent_parameter_posterior.{ext}',dpi=180)
    plt.close(fig);readme=RUN/'figures/README.md'
    if '### independent_parameter_posterior.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### independent_parameter_posterior.png\n用冻结的显式参数提议分布生成多组独立样本，以无偏粒子似然计算重要性权重；相同参数点的Laplace权重另作近似对照。图题标明重要性诊断是否通过，展示中央99%范围，完整95%参数区间见配套CSV。\n**关注点**：参数抽样精度、似然近似和模型适用性必须分别判断；这些时间常数仍不是已识别的生理常数。\n')
    print(pd.DataFrame(allrows).to_string(index=False));print(json.dumps([dict(model=r['model'],target=r['target'],status=r['status'],ess=r['importance_ess'],max_weight=r['maximum_weight'],pareto_k=r['pareto_k'],max_rep_z=r['max_pairwise_mean_mcse_z']) for r in results],indent=2))

if __name__=='__main__':main()

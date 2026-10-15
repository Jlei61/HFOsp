"""Read-only input-law and native-core responses for the final correlation panel.

Correlations are descriptive within a replay, not causal direction or independent
network replicates. Fixed five-ms smoothing is for the native timing diagnostic.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scripts import analyze_topic4_propagation_recovery_night as s

COLORS={1.:'#444444',.5:'#cc7623',0.:'#397faf'}


def lag_curve(x,y,maxlag=100):
    x=np.asarray(x,float);y=np.asarray(y,float)
    x=x-x.mean();y=y-y.mean();den=x.std()*y.std()
    lags=np.arange(-maxlag,maxlag+1)
    out=[]
    for k in lags:
        a,b=(x[:k],y[-k:]) if k<0 else (x[k:],y[:-k]) if k>0 else (x,y)
        out.append(float(np.mean(a*b)/den) if den>0 else np.nan)
    # Positive lag means A is evaluated later than B; do not call it causality.
    return lags,np.asarray(out)


def main():
    spec=s.rt.read(s.night.OUT/'final_B_selection.json')
    assert all(s.an.run.complete(s.an.run.result_path(spec['stage'],c['id'],2511,seed)) for c in spec['candidates'] for seed in spec['seeds'])
    parents=list(dict.fromkeys(c['parent_id'] for c in spec['candidates']))
    cases=[s.rt.read(s.an.run.OUT/'candidates'/f'{p}.json') for p in parents]+spec['candidates']
    dest=s.night.OUT/'core_ou_effect_final_B';F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    rows=[];curves={};sources=[]
    for c in cases:
      parent=c['parent_id'] if c in spec['candidates'] else c['id'];rho=float(c.get('core_ou_correlation',1.))
      for seed in spec['seeds']:
        p=s.an.run.result_path(c['stage'],c['id'],2511,seed);r=s.rt.read(p);ap=s.rt.read(p.parent.parent/'applied_physics.json')
        with np.load(p.with_suffix('.npz')) as a:
            tt=a['trace_time_ms'];keep=(tt>=1500)&(tt<r['actual_duration_ms'])
            x=gaussian_filter1d(a['trace_coreAE_spikes'].astype(float),5.)[keep]
            y=gaussian_filter1d(a['trace_coreBE_spikes'].astype(float),5.)[keep]
            lag,cor=lag_curve(x,y);curves[(parent,seed,rho)]=(lag,cor)
            if rho==1:
                inp=np.repeat(a['trace_xi'][keep,None],2,axis=1)+ap['input']['core_mean_signal_per_ms']
            else:
                mixture=a['core_ou_mixture_values'];inp=mixture[mixture[:,0]>=1500,4:6]
            d=dict(candidate=c['id'],parent=parent,seed=seed,rho=rho,duration_ms=r['actual_duration_ms'],
                core_A_input_mean=float(inp[:,0].mean()),core_B_input_mean=float(inp[:,1].mean()),
                core_A_input_std=float(inp[:,0].std()),core_B_input_std=float(inp[:,1].std()),
                input_rate_correlation=float(np.corrcoef(inp.T)[0,1]),
                input_core_average_std_per_ms=float(inp.mean(1).std()),input_A_minus_B_std_per_ms=float((inp[:,0]-inp[:,1]).std()),
                theoretical_core_average_std_per_ms=.0033*np.sqrt(150/2)*np.sqrt((1+rho)/2),
                theoretical_A_minus_B_std_per_ms=.0033*np.sqrt(150/2)*np.sqrt(2*(1-rho)),
                native_smoothed_zero_lag_correlation=float(cor[lag==0][0]),
                theoretical_OU_std_per_ms=.0033*np.sqrt(150/2))
            if rho<1:
                audit=r['core_ou_mixture_audit']
                assert audit['intermediate_global_clip_steps']==0 and audit['final_negative_core_steps']==0
                assert max(audit['maximum_core_rate_error'].values())<1e-12 and audit['maximum_I_rate_error']==0
            rows.append(d);sources.append(dict(path=str(p),sha256=s.rt.sha(p),arrays_sha256=r['arrays_sha256']))
    s.an.writecsv(dest/'input_and_native_correlations.csv',rows)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    labels={parents[0]:'同核EE×0.75',parents[1]:'EE×0.85；输入均值×0.95'}
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),layout='constrained')
    for row,parent in enumerate(parents):
      for si,seed in enumerate(spec['seeds']):
        rr=sorted([r for r in rows if r['parent']==parent and r['seed']==seed],key=lambda z:z['rho'])
        axes[row,0].plot([r['rho'] for r in rr],[r['input_rate_correlation'] for r in rr],'-' if si==0 else '--',marker='os'[si],label=f'噪声{seed}')
        for k,col in [('A','#ac3b59'),('B','#397faf')]:
            axes[row,1].plot([r['rho'] for r in rr],[r[f'core_{k}_input_std'] for r in rr],'-' if si==0 else '--',color=col,marker='os'[si],label=f'核{k}；{seed}')
        for rho in [1.,.5,0.]:
            lag,cor=curves[(parent,seed,rho)]
            axes[row,2].plot(lag,cor,color=COLORS[rho],ls='-' if si==0 else '--',label=f'ρ={rho:g}；{seed}',lw=1)
      axes[row,0].plot([0,1],[0,1],':',c='.6',label='理论相关系数')
      axes[row,0].set(xlabel='设定两核慢输入相关系数 ρ',ylabel=labels[parent]+'\n实测条件到达率相关系数',xticks=[0,.5,1])
      axes[row,1].axhline(.0033*np.sqrt(150/2),c='.5',ls=':',label='OU稳态理论标准差')
      axes[row,1].set(xlabel='设定两核慢输入相关系数 ρ',ylabel='条件到达率标准差 (每 ms)',xticks=[0,.5,1])
      axes[row,2].axvline(0,c='.8',lw=.7)
      axes[row,2].set(xlabel='核心A相对B的评价时差 (ms)',ylabel='5 ms平滑原生核心发放的互相关')
      for ax in axes[row]:ax.legend(fontsize=6,loc='best');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('只改变两核慢输入的共同程度：输入是否正确改变，核心活动怎样响应？\n同一图2511；保留两条噪声；核外无随机输入。右图使用全部原生核心发放，不经过事件或接触读出筛选。',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(F/f'input_to_native_core_response.{ext}',dpi=200)
    plt.close(fig)
    s.rt.write(dest/'manifest.json',dict(status='COMPLETE_DESCRIPTIVE_DIAGNOSTIC',sources=sources,producer=__file__,producer_sha256=s.rt.sha(__file__),
        observable='Input conditional-rate mean/std/correlation after1500ms burn-in; native core spikes on1ms bins, Gaussian sigma5ms then normalized cross-covariance at lags-100..100ms.',
        unit='One60s parameter x topology x noise replay; serial samples are not independent networks.',
        no_claim='Cross-correlation is not causal drive, phase switching proof or patient propagation acceptance. Finite sample std/correlation need not exactly equal stationary theoretical values.',
        collective_variance='For per-core stationary OU variance v, Var((A+B)/2)=v*(1+rho)/2 and Var(A-B)=2*v*(1-rho). Per-core variance is fixed in law, not the variance of the core average or difference.',
        patient_labels='Not used in input or this native diagnostic.'))
    (F/'README.md').write_text('\n\n'.join(f'### input_to_native_core_response.{ext}\n\n两个既有背景分别改变两核OU共同程度，左图核对实际条件输入相关性，中图核对逐核边际散布，右图显示完整原生核心发放的时间关联；实/虚线分别为两条噪声。5ms平滑只用于右图诊断，不进入事件检测或训练；正时差表示评价A晚于B。**关注点**：输入改变是否真正影响两核相对活动；互相关不是因果、模式切换或患者传播恢复证明。' for ext in ['png','pdf'])+'\n')
    print(json.dumps(dict(output=str(dest),runs=len(rows))),flush=True)


if __name__=='__main__':main()

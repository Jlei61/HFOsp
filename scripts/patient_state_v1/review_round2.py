"""Summarize completed numerical/sequence evidence without upgrading mechanism claims."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json

def main():
    profiles=[];boot=[];shuffles=[];audits=[]
    for model in ('ou','ou_cycle'):
        for p in (RUN/'gpu_profiles'/model).glob('[0-9]*.json'):
            r=json.loads(p.read_text());profiles.append(dict(model=model,tau_hours=r['tau_hours'],particle_ll=r['particle_loglik'],laplace_ll=r['laplace_loglik'],mc_low=r['bootstrap_mc_interval'][0],mc_high=r['bootstrap_mc_interval'][1]))
    for p in (RUN/'round2').glob('bootstrap*.json'):
        r=json.loads(p.read_text())
        if r['status']=='COMPLETE':boot.append(dict(model=r['job']['model'],tau_hours=np.exp(r['theta'][-2]),sd=np.exp(r['theta'][-1]),success=r['success']))
    for p in (RUN/'sequence_controls').glob('*.json'):
        r=json.loads(p.read_text())
        if r['status']!='COMPLETE':continue
        gain=sum(x['ou_loglik']-x['coverage_loglik'] for x in r['rows'])/sum(x['n_events'] for x in r['rows'])
        shuffles.append(dict(scale=r['job']['scale'],gain=gain))
    pro=pd.DataFrame(profiles);boo=pd.DataFrame(boot);shu=pd.DataFrame(shuffles)
    pro.to_csv(RUN/'likelihood_profile_comparison.csv',index=False);boo.to_csv(RUN/'parameter_bootstrap.csv',index=False);shu.to_csv(RUN/'sequence_control_gains.csv',index=False)
    forward=pd.read_csv(RUN/'forward_scores.csv');control=pd.read_csv(RUN/'round2_controls.csv')
    ou=forward[forward.model=='ou'];coverage=control[control.model=='constant_within_coverage'];real=(ou.loglik.sum()-coverage.loglik.sum())/ou.n_events.sum()
    summary=dict(profile={},bootstrap={},shuffle={},real_ou_vs_coverage_gain=real)
    for model in ('ou','ou_cycle'):
        g=pro[pro.model==model];mx=g.particle_ll.max();support=g[g.particle_ll>=mx-1.92];b=boo[boo.model==model]
        summary['profile'][model]=dict(n_grid=len(g),particle_best_tau=g.loc[g.particle_ll.idxmax(),'tau_hours'],support_grid_min=support.tau_hours.min(),support_grid_max=support.tau_hours.max(),
                                      scope='Likelihood evaluated along Laplace-optimized nuisance profile; not fully reoptimized particle profile; 1.92 cutoff is approximate')
        summary['bootstrap'][model]=dict(n=len(b),tau_quantiles=b.tau_hours.quantile([.025,.5,.975]).to_numpy(),sd_quantiles=b.sd.quantile([.025,.5,.975]).to_numpy(),scope='Parametric bootstrap under fitted OU; does not capture model misspecification')
    for scale,g in shu.groupby('scale'):
        summary['shuffle'][scale]=dict(n=len(g),gain_quantiles=g.gain.quantile([.025,.5,.975]).to_numpy(),n_ge_real=int((g.gain>=real).sum()))
    fp=RUN/'figures';fig,axs=plt.subplots(1,3,figsize=(15,4.5));colors={'ou':'#1b9e77','ou_cycle':'#7570b3'}
    for model,c in colors.items():
        g=pro[pro.model==model].sort_values('tau_hours');mx=g.particle_ll.max();axs[0].plot(g.tau_hours*60,g.particle_ll-mx,c=c,label=model+' particle');axs[0].plot(g.tau_hours*60,g.laplace_ll-g.laplace_ll.max(),c=c,ls=':',alpha=.6,label=model+' Laplace')
        b=boo[boo.model==model];axs[1].scatter(b.tau_hours*60,b.sd,c=c,s=12,alpha=.45,label=model)
    axs[0].set_xscale('log');axs[0].set_ylim(-15,1);axs[0].axhline(-1.92,c='gray',lw=.7,ls='--');axs[0].set_xlabel('OU persistence (minutes)');axs[0].set_ylabel('Relative log likelihood');axs[0].legend(fontsize=7)
    axs[1].set_xlabel('Recovered persistence (minutes)');axs[1].set_ylabel('Recovered stationary SD');axs[1].legend(fontsize=8)
    scales=['epoch','coverage','15','1'];rng=np.random.default_rng(2309)
    for i,scale in enumerate(scales):
        a=shu[shu.scale==scale].gain.to_numpy();axs[2].scatter(i+rng.uniform(-.15,.15,len(a)),a,s=8,alpha=.3,c='#777777')
    axs[2].axhline(real,c='#b2182b',label='Real sequence');axs[2].set_xticks(range(4),['Interval','Coverage','15 min','1 min']);axs[2].set_ylabel('OU minus coverage-constant\nheld-out log score / event');axs[2].set_xlabel('Label counts preserved within');axs[2].legend(fontsize=8)
    fig.suptitle('Numerical and temporal controls: support for changing preference, not a unique mechanism');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(fp/f'parameter_and_sequence_audit.{ext}',dpi=180)
    plt.close(fig)
    marker='### parameter_and_sequence_audit.png'
    if marker not in (fp/'README.md').read_text():
        with (fp/'README.md').open('a') as f:f.write('\n'+marker+'\n左图比较GPU粒子似然与Laplace近似沿同一参数轮廓的形状；中图为每个模型128次参数自举；右图为保持不同时间尺度内标签数量、打乱顺序后重新拟合和前推评估的结果。虚线似然阈值是近似参考，粒子曲线没有重新优化所有干扰参数。\n**关注点**：能否区分持续时间和幅度，预测增益是否依赖块内时序；参数自举不包含模型失配或跨患者不确定性。\n')
    write_json(RUN/'round2_scientific_audit.json',summary)
    (RUN/'round2_review.md').write_text('''# 第二轮科学审阅

状态：本轮数值与序列检查已完成；9小时探索目标仍在进行。不能把模型拟合或所有队列完成当最终机制验收。

## 新证据

GPU粒子似然支持分钟到几十分钟尺度的有效偏好变化，并未把OU最佳时间推到数小时边界。数值近似会改变似然和最佳点，尤其很短tau时，故保留粒子校验；精确数值见 round2_scientific_audit.json。参数自举在当前模型内可恢复tau及幅度，但真实数据不同训练段的参数变化大于单次自举范围，不能称为患者固定生理常数。

保持整个覆盖段的标签数量并打乱顺序后，OU相对该覆盖段内恒定偏好的优势消失；保持15分钟内数量时保留部分优势，保持1分钟内数量时保留大部分优势。这支持块内分钟尺度模式比例变化是主要信息源，少量更快的标签历史仍可能提供额外预测。它不是OU生理机制独有证据。

逐次发作的参数现在只用前一发作结束前的数据学习，当前区间的标签仅用于在线滤波。两次TB型发作都早于最初三个测试折，已单独补足该前向边界，不能将原三折间期标签预测直接称为这两次发作的预测。SZ19最后15分钟的状态水平不高于此前15分钟；SZ22有上升。不同前缀模型的s原点随估计基线变化，因此跨次前向比较优先使用可比的TB概率，而非直接比较不同拟合的原始s数值。

## 总率接口的下一轮审阅

全部观测事件窗口为250毫秒且不重叠。对应legacy Epilepsiae packer把1146的packWinLen设为250毫秒，并删除相邻重叠候选的双方。最短间隔不能解读为神经元或网络的生理不应期。

v1.1加入分钟计数观测：分别拟合模式OU、总率OU与一个共享OU同时解释计数和标签，负二项计数只是所选事件产物的有效观测模型。15/60/300秒分箱使共同状态和总率参数明显变化，一些耦合符号也改变。需要检验前向联合预测与生成分布，不能直接接受“一个状态同时解释一切”。15秒的纯模式拟合与原逐事件tau接近，较粗分箱把tau推长，支持继续以逐事件模型作为模式基准。

后续保留两个方向：用有限的短期历史/离散切换对照检查连续OU是否充分；为总率建立尊重观测最小间隔的有效点过程，并明示它不等同于还原被packing删除的上游事件。SNN及Z/M始终冻结。
''',encoding='utf-8')
    print(summary,flush=True)

if __name__=='__main__':main()

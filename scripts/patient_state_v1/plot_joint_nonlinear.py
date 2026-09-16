"""Keep predictive gains and generated-distribution adequacy in the same view."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.nonlinear_calibration import best_real
from scripts.patient_state_v1.nonlinear_drift import generator
OUT=RUN/'joint_nonlinear_review_v1_21'
def main():
    scores=pd.read_csv(OUT/'forward_summary.csv');real=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());rows=[]
    for p in (RUN/'joint_two_state_generation_v1_20/runs').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';rows.append(dict(model=r['job']['model'],rep=r['job']['rep'],rate=r['rate_per_hour'],tb_fraction=r['tb_fraction'],adjacency_excess=r['adjacent_excess'],lag2=r['windows']['5']['tb_share_autocorrelation_lags124'][1]))
    df=pd.DataFrame(rows);assert len(df)==256;df.to_csv(OUT/'joint_generation.csv',index=False);intervals=[];fig,axs=plt.subplots(1,3,figsize=(13,4.8));selected=scores[(scores.family=='joint')&(scores.order==80)&scores.rate_first]
    for i,r in enumerate(selected.itertuples()):axs[0].errorbar(r.gain_per_event,i,xerr=[[r.gain_per_event-r.lower],[r.upper-r.gain_per_event]],fmt='o',c='#2166ac',capsize=3)
    axs[0].axvline(0,c='gray',lw=.8);axs[0].set_yticks(range(len(selected)),[f'{s}-second bins' for s in selected.seconds]);axs[0].set_xlabel('Held-out joint log-density gain / event');axs[0].set_title('State-to-rate coupling helps prediction')
    for ax,measure,target in zip(axs[1:],['rate','tb_fraction'],[real['rate_per_hour'],real['tb_fraction']]):
        ax.axvline(target,c='k',ls='--',label='Patient')
        for i,model in enumerate(['independent','coupled']):
            q=df[df.model==model][measure].quantile([.025,.5,.975]);ax.errorbar(q.iloc[1],i,xerr=[[q.iloc[1]-q.iloc[0]],[q.iloc[2]-q.iloc[1]]],fmt='o',c=['#777777','#2166ac'][i],capsize=3);intervals.append(dict(model=model,measure=measure,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],patient=target))
        ax.set_yticks([0,1],['Independent states','Coupled observation']);ax.set_xlabel('Events / observed hour' if measure=='rate' else 'TB event fraction');ax.set_title('Generated total rate' if measure=='rate' else 'Generated mode balance');ax.legend(fontsize=8)
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.set_ylim(-.6,1.6)
    fig.suptitle('Joint prediction improves, but the full generated distribution remains incompatible\nLeft: 16,157 held-out events, 6-hour block intervals; right: 128 generated sequences per condition, actual exposure',fontsize=11);fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_state_prediction_vs_generation.{ext}',dpi=180)
    plt.close(fig);pd.DataFrame(intervals).to_csv(OUT/'joint_generation_intervals.csv',index=False)
    best=best_real();fig,axs=plt.subplots(1,3,figsize=(13,4.8));barriers=[]
    for kind,c in [('ou','#777777'),('quartic','#2166ac')]:
        r=best[(15,'full',kind)];t=np.asarray(r['theta']);grid,pi,_,_=generator(t,kind,768);keep=(grid>=-3)&(grid<=1);mu=t[0];tau,A=np.exp(t[1:3]);u=grid-mu;force=-u/tau if kind=='ou' else -(t[3]*u+u**3/A**2)/tau;axs[0].plot(grid[keep],force[keep],c=c,label=kind.upper());axs[1].plot(grid[keep],pi[keep]/(grid[1]-grid[0]),c=c,label=kind.upper())
        if kind=='quartic':barriers.append(dict(bin_seconds=15,k=t[3],barrier_in_diffusion_potential_units=t[3]**2/4 if t[3]<0 else 0.,well_positions=(mu+np.array([-1,1])*A*np.sqrt(-t[3])).tolist() if t[3]<0 else [mu]))
    axs[0].axhline(0,c='gray',lw=.8);axs[0].set_xlabel('Mode preference s (TB log-odds)');axs[0].set_ylabel('Expected drift (log-odds / hour)');axs[0].set_ylim(-15,15);axs[0].set_title('Estimated force; no event reset');axs[1].set_xlabel('Mode preference s (TB log-odds)');axs[1].set_ylabel('Stationary density');axs[1].set_title('Fitted stationary density')
    selected=scores[scores.family=='nonlinear']
    for i,r in enumerate(selected.itertuples()):axs[2].errorbar(r.gain_per_event,i,xerr=[[r.gain_per_event-r.lower],[r.upper-r.gain_per_event]],fmt='o',c='#2166ac',capsize=3)
    axs[2].axvline(0,c='gray',lw=.8);axs[2].set_yticks(range(len(selected)),[f'{s}-second bins' for s in selected.seconds]);axs[2].set_xlabel('Held-out mark gain / event (×10⁻⁴)');axs[2].set_title('No stable predictive improvement');axs[2].set_ylim(-.6,1.6);axs[2].xaxis.set_major_locator(MaxNLocator(4));axs[2].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x*1e4:g}'))
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    axs[0].legend();fig.suptitle('Allowing nonlinear drift does not establish bistability\nFull-record fitted force shown at 15-second bins; predictive test uses prefix-only parameters; OU-null calibration reported separately',fontsize=11);fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'nonlinear_drift_force_and_prediction.{ext}',dpi=180)
    plt.close(fig);write_json(OUT/'potential_shape.json',barriers)
    readme=RUN/'figures/README.md'
    if '### joint_state_prediction_vs_generation.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### joint_state_prediction_vs_generation.png\n左侧比较是否允许模式状态影响总事件率的前推联合密度；右侧显示各128条新生成序列的总事件率和TB比例。生成过程只沿用患者覆盖与发作排除区间，间期事件时刻和标签均不回放。\n**关注点**：预测改善是否伴随分布恢复；当前耦合版本仍产生过高的TB比例。\n\n### nonlinear_drift_force_and_prediction.png\n展示同一状态观察框架下线性OU与三次非线性drift的拟合力、平稳密度和前推预测差异。左两图来自完整记录拟合，右图参数仅用各训练前缀拟合；单OU真值模拟校准仍在运行。\n**关注点**：浅双井的训练拟合不等于可重复的双稳态证据，不能用它宣称发作阈值或DDM吸收边界。\n')
    print(pd.DataFrame(intervals).to_string(index=False));print(barriers)
if __name__=='__main__':main()

"""Interpret observation-timing calibration against the frozen prefix statistic."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected
ROOT=RUN/'joint_prefix_calibration_v1_25'

def main():
    assert json.loads((ROOT/'status.json').read_text())['status']=='COMPLETE';paths=list((ROOT/'runs').glob('*.json'));assert len(paths)==128;rows=[];fail=[]
    for p in paths:
        r=json.loads(p.read_text())
        if r['status']!='COMPLETE':fail.append(str(p));continue
        for family in ['ou','ou_history']:
            fits={f['scope']:f['selected'] for f in r['fits'] if f['model']==family};assert len(fits)==4;logtaus={k:float(v['theta'][-2]) for k,v in fits.items()};rows.append(dict(coupled=r['job']['coupled'],rep=r['job']['rep'],family=family,first_minus_full=logtaus['fold0']-logtaus['full'],logtau_range=max(logtaus.values())-min(logtaus.values()),all_success=all(f['success'] for f in fits.values()),n_events=r['n_events'],tb_fraction=r['tb_fraction'],**{f'tau_{k}':float(np.exp(v)) for k,v in logtaus.items()}))
    assert not fail;df=pd.DataFrame(rows);df.to_csv(ROOT/'prefix_statistics.csv',index=False);bf=best_fits();observed={'ou':float(bf['fold0','ou']['theta'][-2]-bf['full','ou']['theta'][-2]),'ou_history':float(selected(RUN/'advanced_controls_v1_2/fits','fold0','ou_history')[0]['theta'][-2]-selected(RUN/'advanced_controls_v1_2/fits','full','ou_history')[0]['theta'][-2])};results=[]
    for (c,fam),g in df.groupby(['coupled','family']):
        for onlygood in [False,True]:
            q=g[g.all_success] if onlygood else g;vals=q.first_minus_full.to_numpy();n=len(vals);above=int(np.sum(vals>=observed[fam]));results.append(dict(coupled=bool(c),family=fam,successful_only=onlygood,n_sequences=n,observed=observed[fam],n_at_least_observed=above,monte_carlo_upper_tail_p=(above+1)/(n+1),null_quantiles=np.quantile(vals,[.025,.5,.975]).tolist() if n else [],n_optimizer_warning=int((~g.all_success).sum())))
    write_json(ROOT/'scientific_audit.json',dict(status='COMPLETE',results=results,statistic='First-prefix minus full log conditional-mark time constant, inherited from earlier calibration; positive means estimated timescale shrinks with added recording',interpretation='Calibrates a specific stationary informative-observation alternative at patient-fitted strength; not a physical nonstationarity test',limits='Joint generator already fails some patient marginals. Passing this one statistic cannot accept the complete model. Event packing and fixed ictal-exclusion boundaries remain approximations.'))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,fam in zip(axs,['ou','ou_history']):
        values=df[df.family==fam].first_minus_full.to_numpy();edges=np.linspace(values.min(),values.max(),19)
        for c,color in [(False,'#557b99'),(True,'#bf6845')]:
            g=df[(df.coupled==c)&(df.family==fam)];ax.hist(g.first_minus_full,bins=edges,alpha=.5,color=color,label='Independent event timing' if not c else 'State-dependent event timing')
        ax.axvline(observed[fam],color='black',lw=1.5,ls='--',label='Patient');ax.set(title=fam,xlabel='First-prefix minus full log time constant',ylabel='Generated sequences');ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle('Can a stationary observation coupling imitate prefix dependence?\nNew event times and labels on fixed patient exposure; 64 sequences per generator',fontsize=11);fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_observation_prefix_calibration.{ext}',dpi=180)
    plt.close(fig)
    readme=RUN/'figures/README.md'
    if '### joint_observation_prefix_calibration.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### joint_observation_prefix_calibration.png\n在患者拟合的独立或耦合观察模型下，各生成64条新事件时间和标签序列，再按原有记录前缀重拟合单OU及OU加短时记忆。虚线为患者首个前缀与全记录的log时间常数差，比较是否需要时变参数才能产生该现象。\n**关注点**：这是针对观察过程解释的校准，生成模型本身的模式比例等缺口仍须单独验收。\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()

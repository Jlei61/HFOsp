"""Compare activity-driven state speed with unchanged mark baselines."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
ROOT=RUN/'activity_clock_v1_27'

def paired(a,b,width):
    d=a.merge(b,on=['fold','index'],suffixes=('_a','_b'),validate='one_to_one');assert len(d)==len(a)==len(b)==16157;assert np.array_equal(d.y_a,d.y_b);assert np.max(abs(d.hour_a-d.hour_b))<1e-8;d['block']=np.floor(d.hour_a/width).astype(int);d['gain']=d.score_a-d.score_b;g=d.groupby(['fold','block']).agg(gain=('gain','sum'),n=('index','size')).reset_index();rng=np.random.default_rng(90127+width);num=np.zeros(5000);den=np.zeros(5000)
    for _,part in g.groupby('fold'):
        ix=rng.integers(0,len(part),(5000,len(part)));num+=part.gain.to_numpy()[ix].sum(axis=1);den+=part.n.to_numpy()[ix].sum(axis=1)
    lo,hi=np.quantile(num/den,[.025,.975]);return dict(mean_gain=float(d.gain.mean()),lower=float(lo),upper=float(hi),n_events=len(d),n_blocks=len(g),block_hours=width)

def main():
    assert json.loads((ROOT/'status.json').read_text())['status']=='COMPLETE';best={}
    for p in (ROOT/'fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];key=(j['scope'],j['history'])
        if key not in best or r['penalized_loglik']>best[key][1]['penalized_loglik']:best[key]=(p,r)
    assert len(best)==8;params=[];records=[];d=np.load(RUN/'observations.npz')
    for (scope,h),(p,r) in sorted(best.items()):
        t=r['theta'];beta=t[-1];base=r['covariate_info']['training_rate_per_hour'];speed500=np.exp(beta*np.clip(np.log(500/base),-4,4));params.append(dict(scope=scope,history=h,b=t[0],gamma=t[1] if h else 0,tau_minutes=np.exp(t[-3])*60,tau_at500perhour_minutes=np.exp(t[-3])*60/speed500,sd=np.exp(t[-2]),beta=beta,training_rate_per_hour=base,laplace_loglik=r['laplace_loglik'],penalized_loglik=r['penalized_loglik'],success=r['success'],source=str(p)))
        if scope=='full':continue
        lo,hi=r['job']['end'],r['job']['test_end'];z=np.load(p.with_suffix('.npz'));pp=np.clip(z['predict_tb'][lo:hi],1e-12,1-1e-12);y=d['y'][lo:hi];records.append(pd.DataFrame(dict(model='activity_clock_history' if h else 'activity_clock',fold=int(scope[4:]),index=np.arange(lo,hi),hour=d['t'][lo:hi],y=y,p_tb=pp,score=y*np.log(pp)+(1-y)*np.log1p(-pp))))
    pf=pd.DataFrame(params);pf.to_csv(ROOT/'selected_parameters.csv',index=False);new=pd.concat(records,ignore_index=True);new.to_csv(ROOT/'forward_predictions.csv.gz',index=False);old=pd.read_csv(RUN/'all_forward_predictions.csv.gz');summaries=[]
    for model in ['activity_clock','activity_clock_history']:
        for baseline in ['constant','ewma','ou','ou_history']:
            for width in [1,6]:summaries.append(dict(model=model,baseline=baseline,**paired(new[new.model==model],old[old.model==baseline],width)))
    scores=pd.DataFrame(summaries);scores.to_csv(ROOT/'forward_summary.csv',index=False);comparison=scores[(scores.block_hours==6)&(((scores.model=='activity_clock')&(scores.baseline=='ou'))|((scores.model=='activity_clock_history')&(scores.baseline=='ou_history')))];gates={r.model:bool(r.lower>0) for r in comparison.itertuples()};write_json(ROOT/'scientific_audit.json',dict(status='COMPLETE',strong_forward_increment_gate=gates,uncertainty='5000 paired resamples of physical-time blocks, stratified by frozen chronological fold; same16157events',scope='Current-event mark prediction given completed past events and occurrence time; not a fixed-horizon forecast or neural feedback identification',reference_time_constant='Tau at a fixed500/h activity reference avoids comparing prefix-specific rate normalizations as though they were identical physical conditions',regularization='Extra speed coefficient beta has Normal(0,1) penalty; amplitude and restoring speed scale together under the clock',limits='Training gain alone is insufficient; changing the activity clock may also worsen prefix stability. No seizure type or future label enters fits.'))
    fig,axs=plt.subplots(1,3,figsize=(14,4));scopes=['fold0','fold1','fold2','full']
    for h,color in [(False,'#557b99'),(True,'#bf6845')]:
        g=pf[pf.history==h].set_index('scope').loc[scopes];label='Clock OU + short memory' if h else 'Clock OU';axs[0].plot(range(4),g.beta,'o-',color=color,label=label);axs[1].plot(range(4),g.tau_at500perhour_minutes,'o-',color=color,label=label)
    axs[0].axhline(0,color='gray',ls=':');axs[0].set(ylabel='Activity-to-state-speed coefficient',title='One extra coefficient');axs[1].set(ylabel='Time constant at 500 events/h (min)',title='Common activity reference')
    for ax in axs[:2]:ax.set_xticks(range(4),['Prefix1','Prefix2','Prefix3','Full'],rotation=15)
    for i,r in enumerate(comparison.itertuples()):axs[2].errorbar(r.mean_gain*1e4,i,xerr=[[(r.mean_gain-r.lower)*1e4],[(r.upper-r.mean_gain)*1e4]],fmt='o',color='#557b99' if i==0 else '#bf6845',capsize=3)
    axs[2].axvline(0,color='gray',ls=':');axs[2].set(yticks=[0,1],yticklabels=['OU','OU + short memory'],xlabel='Gain / event (x 10^-4)',title='Forward gain over fixed speed');axs[0].legend(fontsize=7)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Does recent total activity explain mode-state speed?\nCompleted-event covariate; training-only normalization; 6-h block uncertainty',fontsize=11);fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'activity_dependent_state_speed.{ext}',dpi=180)
    plt.close(fig);readme=RUN/'figures/README.md'
    if '### activity_dependent_state_speed.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### activity_dependent_state_speed.png\n让模式状态的回复和波动速度随已完成总事件的一分钟活动历史共同改变，比较四个训练前缀的速度系数和同一500次/小时参考活动下的时间常数。右图为相对匹配固定速度模型的严格前推增益，区间按6小时块计算。\n**关注点**：额外的速度系数是否带来可重复预测增益、是否真正减少时间常数不稳定；不把条件统计关联解释为网络因果反馈。\n')
    print(pf.to_string(index=False));print(scores[scores.block_hours==6].to_string(index=False));print(json.dumps(gates))

if __name__=='__main__':main()

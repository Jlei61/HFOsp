"""A compact paired input-correlation response figure, using existing observables."""
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import analyze_topic4_propagation_recovery_night as s


def order_stat(x,names):
    i,j=list(names).index('ICL11'),list(names).index('ICL9')
    ok=np.isfinite(x[:,i])&np.isfinite(x[:,j]);d=x[ok,j]-x[ok,i]
    return dict(n=int(ok.sum()),probability=float(np.mean((d>0)+.5*(d==0))) if len(d) else None,ties=int((d==0).sum()))


def main():
    src=s.night.OUT/'analysis_final_B'/'run_observations.csv'
    if not src.exists():raise RuntimeError('Complete numerical analysis required')
    with src.open() as f:obs=list(csv.DictReader(f))
    spec=s.rt.read(s.night.OUT/'final_B_selection.json');parents=list(dict.fromkeys(c['parent_id'] for c in spec['candidates']))
    cases=[s.rt.read(s.an.run.OUT/'candidates'/f'{p}.json') for p in parents]+spec['candidates']
    ev=s.rt.load_evaluator(s.rt.read(s.an.run.PARENT));patient=np.asarray(ev.fit);labels=np.asarray(ev.fit_labels)
    reference={};rows=[];sources=[]
    for c in cases:
      parent=c['id'] if c['id'] in parents else c['parent_id']
      for seed in spec['seeds']:
        p=s.an.run.result_path(c['stage'],c['id'],2511,seed);r=s.rt.read(p)
        with np.load(p.with_suffix('.npz')) as z:
            ids=s.an.analysis_ids(r,{'primary_event_indices':z['primary_event_indices']},1500);x=z['centroid_ms'][ids];mode=z['event_mode'][ids];names=z['contact_names']
        match={q['mode']:q for q in obs if q['base_id']==c['id'] and int(q['seed'])==seed and q['layer']=='primary'}
        assert len(match)==3
        a=match['TA'];tb=order_stat(x[mode==0],names)
        rows.append(dict(candidate=c['id'],parent=parent,seed=seed,rho=c.get('core_ou_correlation',1.),
            n=len(ids),TA_n=int((mode==1).sum()),TA_fraction=float((mode==1).mean()),
            TA_rod_lag_median_ms=float(a['SCL_minus_ICL_lag_median_ms']),
            TA_rod_lag_q05_ms=float(a['SCL_minus_ICL_lag_q05_ms']),TA_rod_lag_q95_ms=float(a['SCL_minus_ICL_lag_q95_ms']),
            TB_joint_pair_n=tb['n'],TB_left_tip_before_next_probability=tb['probability'],TB_pair_ties=tb['ties']))
        sources.append(dict(path=str(p),json_sha256=s.rt.sha(p),arrays_sha256=r['arrays_sha256']))
    pref=s.an.measures(patient[labels==1],patient[labels==1],names)
    reference=dict(TA_fraction=float((labels==1).mean()),TA_n=int((labels==1).sum()),TB_n=int((labels==0).sum()),
        TA_rod_lag_median_ms=pref['SCL_minus_ICL_lag_median_ms'],TA_rod_lag_q05_ms=pref['SCL_minus_ICL_lag_q05_ms'],TA_rod_lag_q95_ms=pref['SCL_minus_ICL_lag_q95_ms'],
        TB_pair=order_stat(patient[labels==0],names))
    dest=s.night.OUT/'core_ou_propagation_response';F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    s.an.writecsv(dest/'values.csv',rows)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    rowlabels=['核内EE×0.75','核内EE×0.85；输入均值×0.95']
    for ri,parent in enumerate(parents):
      for si,seed in enumerate(spec['seeds']):
        rr=sorted([r for r in rows if r['parent']==parent and r['seed']==seed],key=lambda q:q['rho'],reverse=True)
        xx=np.asarray([r['rho'] for r in rr]);style=dict(marker='os'[si],ls='-' if si==0 else '--',lw=1.1,ms=5,label=f'噪声{seed}')
        axes[ri,0].plot(xx,[r['TA_fraction'] for r in rr],c='#ab418f',**style)
        med=np.array([r['TA_rod_lag_median_ms'] for r in rr]);low=np.array([r['TA_rod_lag_q05_ms'] for r in rr]);high=np.array([r['TA_rod_lag_q95_ms'] for r in rr])
        axes[ri,1].errorbar(xx+(.015 if si==0 else -.015),med,yerr=[med-low,high-med],c='#bd3934',capsize=2,**style)
        axes[ri,2].plot(xx,[r['TB_left_tip_before_next_probability'] for r in rr],c='#2679b0',**style)
        for x,r in zip(xx,rr):
            axes[ri,0].annotate(f"{r['TA_n']}/{r['n']}",(x,r['TA_fraction']),xytext=(0,8 if si==0 else -16),textcoords='offset points',ha='center',fontsize=7)
            axes[ri,2].annotate(f"n={r['TB_joint_pair_n']}",(x,r['TB_left_tip_before_next_probability']),xytext=(0,8 if si==0 else -16),textcoords='offset points',ha='center',fontsize=7)
      axes[ri,0].axhline(reference['TA_fraction'],c='.5',ls=':',label='患者FIT比例')
      axes[ri,1].axhspan(reference['TA_rod_lag_q05_ms'],reference['TA_rod_lag_q95_ms'],color='.85',zorder=-2)
      axes[ri,1].axhline(reference['TA_rod_lag_median_ms'],c='.4',ls=':',label='患者TA中位数及5–95%')
      axes[ri,2].axhline(reference['TB_pair']['probability'],c='.5',ls=':',label='患者TB顺序概率')
      axes[ri,0].set(ylabel=rowlabels[ri]+'\n合格事件中的TA比例',ylim=(-.025,.78))
      axes[ri,1].set(ylabel='TA：参与SCL − 参与ICL质心中位差 (ms)',ylim=(-75,150))
      axes[ri,2].set(ylabel='TB：ICL11早于ICL9的概率',ylim=(-.08,.88))
      for ax in axes[ri]:
        ax.set(xlim=(1.13,-.13),xticks=[1,.5,0],xlabel='两核慢输入相关系数 ρ（1共享 → 0独立）')
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=6.5,loc='upper left')
    fig.suptitle('增加某类事件的出现，不等于恢复该类传播\n同一图2511、同一几何；每点一条60秒重演。圆实线/方虚线保留两条噪声；只改变两核OU相关性。',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(F/f'core_input_correlation_to_propagation.{ext}',dpi=200)
    plt.close(fig)
    s.rt.write(dest/'manifest.json',dict(status='COMPLETE_DESCRIPTIVE_PARAMETER_RESPONSE',reference=reference,sources=sources,
        observations_source=str(src),observations_sha256=s.rt.sha(src),producer=__file__,producer_sha256=s.rt.sha(__file__),
        interpretation='Primary events only. Frequency and pair probabilities are replay summaries; error bars are within-replay TA-event5-95% ranges, not CIs. Gray band is patient FIT event variability, not a significance test. Rod timing depends on actual participant sets.',
        pair_definition='ICL11 is leftmost ICL; ICL9 is the next adopted left endpoint. Only jointly participating contacts; ties contribute0.5. Denominators shown, missing is not zero.',
        random_input='Same seed identities across rho do not imply identical realized Poisson inputs. Per-core marginal OU law preserved, common-average and difference variances change.',
        new_metric=False,independent_validation=False,accepted_patient_propagation=False))
    (F/'README.md').write_text('\n\n'.join(f'### core_input_correlation_to_propagation.{ext}\n\n固定两个既有背景，依次降低两核OU输入相关性；三列分别显示TA占比、TA杆间质心时差及TB左端顺序概率，每个点是一条60秒重演。中间误差棒是运行内事件5–95%范围，灰带是患者FIT事件范围，均非置信区间；左右列同时标出事件数或联合参与数。**关注点**：同一种干预可能提高TA支持量，却保留条件路径残差；两条噪声并非两个新拓扑，也不构成独立确认。' for ext in ['png','pdf'])+'\n')
    print({'output':str(dest),'runs':len(rows),'reference':reference},flush=True)


if __name__=='__main__':main()

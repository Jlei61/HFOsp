"""All-condition propagation diagnostics and paired time-block precision audit."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import long_topic4_core_extent as long
from scripts.paper_figures import plot_topic4_core_extent_pilot as plot
rt=long.rt;OUT=long.OUT


def score(c,seeds,design,end):
    obj=rt.load_objective(design);units=[]
    for seed in seeds:
        path=long.unit_path(c,seed,end)
        if not path.exists():units.append(dict(seed=seed,status='MISSING',loss_off=None));continue
        r=rt.read(path)
        with np.load(path.with_suffix('.npz')) as z:
            ix=[i for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=min(end,r['actual_duration_ms'])]
            table=z['centroid_ms'][ix]
        s=obj.score_network(table)
        if r['actual_duration_ms']<end:s={**s,'status':'PHYSICAL_RUNAWAY_OR_INCOMPLETE','loss_off':None}
        units.append(dict(seed=seed,path=str(path),**s))
    loss=float(np.mean([u['loss_off'] for u in units])) if all(u['loss_off'] is not None for u in units) else None
    return dict(candidate=c,units=units,loss_off=loss)


def precision(plan,stage):
    candidates=plan['candidates'];seeds=stage['seeds'];end=stage['duration_ms'];settings=plan['precision']
    ev=rt.load_evaluator(plan['parent_design']);names=rt.load_observation_contract(plan['parent_design'])['contact_names']
    scl=np.array([n.startswith('SCL') for n in names]);icl=np.array([n.startswith('ICL') for n in names])
    nc,nu,nb,nm,nch=len(candidates),len(seeds),int(np.ceil(end/settings['block_ms'])),2,len(names)
    counts=np.zeros((nc,nu,nb,nm));active=np.zeros((nc,nu,nb,nm,nch));no_scl=np.zeros_like(counts);no_icl=np.zeros_like(counts)
    physical={};segments=[]
    for ci,c in enumerate(candidates):
        physical[ci]=[]
        for ui,seed in enumerate(seeds):
            path=long.unit_path(c,seed,end);r=rt.read(path);physical[ci].append(r['physical_status'])
            with np.load(path.with_suffix('.npz')) as z:
                for i in z['primary_event_indices']:
                    e=r['events'][i]
                    if e['window_ms'][0]<1500 or e['window_ms'][1]>min(end,r['actual_duration_ms']):continue
                    m=int(z['event_mode'][i]);b=min(nb-1,int(e['event_time_ms']/settings['block_ms']));mask=np.isfinite(z['centroid_ms'][i])
                    counts[ci,ui,b,m]+=1;active[ci,ui,b,m]+=mask;no_scl[ci,ui,b,m]+=not mask[scl].any();no_icl[ci,ui,b,m]+=not mask[icl].any()
            for b in range(nb):
                segments.append(dict(candidate=c['id'],seed=seed,start_ms=b*settings['block_ms'],end_ms=min(end,(b+1)*settings['block_ms']),
                    mode0=int(counts[ci,ui,b,0]),mode1=int(counts[ci,ui,b,1]),
                    M0_no_SCL=int(no_scl[ci,ui,b,0]),M1_no_SCL=int(no_scl[ci,ui,b,1])))
    ref=np.stack([np.isfinite(np.asarray(ev.fit)[ev.fit_labels==m]).mean(0) for m in range(nm)])
    def estimate(n,a,s):
        # Equal observed network/noise units; absent modes remain unestimated.
        with np.errstate(divide='ignore',invalid='ignore'):
            part=np.mean(abs(a/n[...,None]-ref),axis=-1)
            missing=s/n
        return part,missing
    n=counts.sum(2);a=active.sum(2);s=no_scl.sum(2);part,missing=estimate(n,a,s)
    rng=np.random.default_rng(9093901);reps=settings['bootstrap_replicates'];dp=np.full((reps,nc,nm),np.nan);ds=dp.copy()
    for k in range(reps):
        u=rng.integers(0,nu,nu);b=rng.integers(0,nb,(nu,nb))
        sampled_n=np.stack([counts[:,uu,bb].sum(1) for uu,bb in zip(u,b)],axis=1)
        sampled_a=np.stack([active[:,uu,bb].sum(1) for uu,bb in zip(u,b)],axis=1)
        sampled_s=np.stack([no_scl[:,uu,bb].sum(1) for uu,bb in zip(u,b)],axis=1)
        bp,bs=estimate(sampled_n,sampled_a,sampled_s)
        dp[k]=(bp-bp[0:1]).mean(1);ds[k]=(bs-bs[0:1]).mean(1)
    def interval(values):
        ok=np.isfinite(values)
        if ok.mean()<.9:return None
        return np.percentile(values[ok],[2.5,97.5]).tolist()
    rows=[];passed=True
    for ci,c in enumerate(candidates):
        all_runaway=all(x=='RUNAWAY' for x in physical[ci])
        for m in range(nm):
            pc=interval(dp[:,ci,m]);sc=interval(ds[:,ci,m])
            support=bool(n[ci,:,m].sum()>=settings['min_pooled_mode_events'] and np.sum(n[ci,:,m]>=5)>=settings['min_runs_with_five_mode_events'])
            precise=bool(support and pc is not None and sc is not None and pc[1]-pc[0]<=settings['participation_delta_CI_width'] and sc[1]-sc[0]<=settings['no_SCL_delta_CI_width'])
            if all_runaway:precise=True
            if not precise:passed=False
            rows.append(dict(candidate=c['id'],mode=m,events_by_noise=n[ci,:,m].astype(int).tolist(),
                no_SCL_by_noise=s[ci,:,m].astype(int).tolist(),no_ICL_by_noise=no_icl.sum(2)[ci,:,m].astype(int).tolist(),
                participation_MAE_by_noise=part[ci,:,m].tolist(),
                participation_delta_CI=pc,no_SCL_delta_CI=sc,precision_met=precise,
                interpretation='consistently runaway; not an interictal candidate' if all_runaway else 'propagation review required regardless of score',
                physical_status=physical[ci]))
    return dict(precision_targets_met=passed,stage=stage,definition='Paired resampling of noise units and 15-s blocks; FIT reference fixed; no independent-event significance claim; development precision, not mechanism acceptance.',rows=rows),segments


def main():
    plan=rt.read(OUT/'plan.json');stage=rt.read(OUT/'active_analysis_stage.json');end=stage['duration_ms'];seeds=stage['seeds']
    view=OUT/f'analysis_{round(end)}_{len(seeds)}noise';view.mkdir(exist_ok=True)
    local=copy.deepcopy(plan);local['training_seeds']=seeds;local['observation_seeds']=seeds;local['duration_ms']=end
    rt.write(view/'plan.json',local);rt.write(view/'all_candidates.json',plan['candidates'])
    audit,segments=precision(plan,stage);rt.write(view/'precision_audit.json',audit);rt.write(OUT/'precision_audit.json',audit)
    plot.csv_write(view/'time_block_observations.csv',segments)
    plot.OUT=view;plot.F=view/'figures';plot.ANALYSIS_END_MS=end;plot.SHOW_ALL_GEOMETRIES=True;plot.RUN_ROLE='paired_comparison'
    plot.MODE_NAMES={0:'TB',1:'TA'};plot.MODE_ORDER=[1,0]
    plot.CONDITION_LABELS={'baseline':'Original radii','expand_A_2.5':'Left 2.5 mm','expand_B_2.5':'Right 2.5 mm','expand_AB_2.5':'Both 2.5 mm','expand_A_4':'Left 4 mm','expand_B_4':'Right 4 mm','expand_AB_4':'Both 4 mm'}
    plot.p.path_for=lambda c,s:long.unit_path(c,s,end)
    plot.p.score_candidate=lambda c,s:score(c,s,plan['parent_design'],end)
    sys.argv=['plot_long'];plot.main()
    txt=['# 长程传播比较：阶段结果','',f'本阶段：7 个固定几何条件，{len(seeds)} 条配对噪声，每条 {end/1000:g} 秒；同一拓扑 2511。',
         '全部条件均进入比较，没有按 N>=16 或损失淘汰。损失是辅助诊断；“精度达标”只表示下列观测比较的支持量，不代表患者传播机制已恢复。',
         '病例参考为冻结患者 FIT，bootstrap 使用噪声及 15 秒时间块，不把所有事件当作独立网络样本。',
         '', '|条件|模式|各噪声事件数|各噪声缺 SCL 数|参与误差差值 95% 区间|精度支持|','|---|---|---|---|---|---|']
    for row in audit['rows']:
        txt.append(f'|{plot.CONDITION_LABELS.get(row["candidate"],row["candidate"])}|{plot.MODE_NAMES[row["mode"]]}|{row["events_by_noise"]}|{row["no_SCL_by_noise"]}|{row["participation_delta_CI"]}|{row["precision_met"]}|')
    txt+=['','差值为当前条件减基线，负值表示参与误差下降。两类时序、ICL/SCL 完整招募及原生空间过程应联合判读；不能用某一参与比例改善代替整种传播恢复。',
          '所有条件的图件和多事件 GIF 均在 figures/；time_block_observations.csv 用于比较前后时间段。黑底读出为发放密度包络，不是 HFO 频谱。',
          '当前判定：'+('观测精度目标满足，等待科学目视审阅。' if audit['precision_targets_met'] else '部分比较仍缺少精度支持，按预定计划增加配对噪声或时长。'),
          '稳定改善、稳定恶化或两模式间取舍均可形成结果；不要求得到阳性后才停止。']
    (view/'scientific_report.md').write_text('\n'.join(txt)+'\n')
    (OUT/'scientific_report.md').write_text(f'# 当前长程比较\n\n最新阶段：[科学报告]({view.name}/scientific_report.md)；[全部图件]({view.name}/figures/README.md)。\n\n'+txt[-2]+'\n')
    readme=view/'figures/README.md';body=readme.read_text().replace('两条配对噪声',f'{len(seeds)} 条配对噪声').replace('棕色与蓝色分别为噪声 847101 与 847102','颜色对应图例列出的噪声种子').replace('两条训练轨迹','本阶段全部配对轨迹')
    readme.write_text(body)
    rt.write(OUT/'latest_analysis.json',dict(path=str(view),precision_targets_met=audit['precision_targets_met']))


if __name__=='__main__':main()

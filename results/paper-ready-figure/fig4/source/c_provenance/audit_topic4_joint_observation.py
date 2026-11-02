"""Frozen-output audit of label-free same-event coupling, never a live search loss.

The maps, relative group weights, bandwidths, and positive CAL scale are fixed
from the inherited patient feature contract before opening candidate scores.
"""
from pathlib import Path
import csv, json, pickle, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_xy_block_pilot as pilot
from scripts import analyze_topic4_three_observable_bo as patient_source
from src.topic4_three_observable_objective import signed_statistic
rt=pilot.rt
SOURCE=pilot.SOURCE
OUT=pilot.OUT/'joint_feature_audit'
KEYS=['rank','scl_pairs','icl_pairs','rod_lag','mask']


class JointMap:
    def __init__(self, old, seed=2026091601):
        self.old=old; self.seed=seed
        # Three observable groups have equal total weight. The timing group
        # gives equal weight to SCL, ICL, and between-shaft timing.
        self.weights=dict(rank=1/3,scl_pairs=1/9,icl_pairs=1/9,rod_lag=1/9,mask=1/3)
        self.bandwidths={k:old.maps[k]['bandwidth'] for k in KEYS}
        dim=sum(old.maps[k]['weights'].shape[0] for k in KEYS)
        rng=np.random.default_rng(seed)
        self.W=np.column_stack([rng.normal(size=(dim,2048))/s for s in [.5,1.,2.]])
        self.phase=rng.uniform(0,2*np.pi,len(self.W[0]))

    def encode(self,raw):
        return np.column_stack([raw[k]*np.sqrt(self.weights[k])/self.bandwidths[k] for k in KEYS])

    def features(self,raw,device):
        x=self.encode(raw)
        if device=='cpu': return np.cos(x@self.W+self.phase)*np.sqrt(2/self.W.shape[1])
        import torch
        with torch.no_grad():
            w=torch.as_tensor(self.W,device=device); phase=torch.as_tensor(self.phase,device=device)
            return np.concatenate([(torch.cos(torch.as_tensor(x[i:i+256],device=device)@w+phase)*np.sqrt(2/self.W.shape[1])).cpu().numpy()
                                   for i in range(0,len(x),256)])


def statistic(features,target):
    return signed_statistic(features.sum(0),float(np.square(features).sum()),len(features),target)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    device=pilot.dense.gpu_device()
    import torch
    torch.set_num_threads(2)
    with (SOURCE/'analysis/training_objective.pkl').open('rb') as f: old=pickle.load(f)
    assert not any(hasattr(old,k) for k in ['km','proportions','mode_means'])
    ev,names,_=patient_source.patient(); assert names==old.names
    cal_contract=rt.read(patient_source.A/'positive_scale_calibration.json')
    calids=np.asarray(cal_contract['parent_indices'],int)
    fit,cal=ev.fit,ev.patient[calids]
    mapping=JointMap(old)
    contract=dict(schema='same_event_joint_observation_offline_audit.v1',seed=mapping.seed,patient_FIT_N=len(fit),patient_CAL_N=len(cal),
        old_objective=str(SOURCE/'analysis/training_objective.pkl'),old_objective_sha256=rt.sha(SOURCE/'analysis/training_objective.pkl'),
        bandwidths=mapping.bandwidths,group_weights=mapping.weights,n_features=mapping.W.shape[1],
        label_usage=False,feature='Gaussian multiscale kernel on concatenated rank, timing and participation features of one event',
        target='all patient FIT events; no TA/TB conditional target',statistics='signed empirical off-self-pair statistic; dependent events not asserted iid',
        scale='median positive A over the original 256 block-CAL draws of 16 events; never negative Doff as denominator',
        counterfactual='feature-layer reassociation within identical participation masks; tests objective sensitivity, not a physically realizable trajectory',
        use='OFFLINE_DIAGNOSTIC_ONLY; does not modify or select the running XY pilot',created_unix=time.time())
    dest=OUT/'frozen_contract.json'
    if dest.exists():
        previous=rt.read(dest)
        assert {k:v for k,v in previous.items() if k!='created_unix'}=={k:v for k,v in contract.items() if k!='created_unix'}
    else: rt.write(dest,contract)
    np.savez_compressed(OUT/'feature_map.npz',weights=mapping.W,phases=mapping.phase)
    print('Patient feature contract frozen',flush=True)
    raw_fit=old.raw_features(fit); raw_cal=old.raw_features(cal)
    ffit=mapping.features(raw_fit,device); target=ffit.mean(0); del ffit
    fcal=mapping.features(raw_cal,device)
    draws=cal_contract['draws']; scale=float(np.median([np.sum((fcal[d['cal_indices']].mean(0)-target)**2) for d in draws]))
    assert scale>0
    np.savez_compressed(OUT/'target.npz',target=target,scale=np.asarray(scale))
    np.testing.assert_allclose(mapping.features({k:v[:5] for k,v in raw_cal.items()},'cpu'),fcal[:5],rtol=1e-12,atol=1e-12)
    rt.write(OUT/'calibration.json',dict(scale=scale,draws=draws,source=str(patient_source.A/'positive_scale_calibration.json'),
        source_sha256=rt.sha(patient_source.A/'positive_scale_calibration.json'),cpu_gpu_parity=True))
    # Reassociate all timing blocks jointly within participation-pattern strata.
    # Every individual feature-block marginal, and hence the old additive loss,
    # remains exactly fixed. There is no event label in this construction.
    rng=np.random.default_rng(mapping.seed+1)
    pattern=np.packbits(np.isfinite(cal),axis=1)
    _,group=np.unique(pattern,axis=0,return_inverse=True)
    group_ids=[np.flatnonzero(group==g) for g in np.unique(group)]
    base=statistic(fcal,target); coupling=[]
    for rep in range(12):
        perm=np.arange(len(cal))
        for ids in group_ids: perm[ids]=rng.permutation(ids)
        shuffled={k:(raw_cal[k][perm] if k in ['scl_pairs','icl_pairs','rod_lag'] else raw_cal[k]) for k in KEYS}
        assert all(np.array_equal(shuffled[k].sum(0),raw_cal[k].sum(0)) or
                   np.allclose(shuffled[k].sum(0),raw_cal[k].sum(0),rtol=1e-12,atol=1e-12) for k in KEYS)
        q=statistic(mapping.features(shuffled,device),target)
        coupling.append(dict(repetition=rep,original_Doff=base['D_off'],shuffled_Doff=q['D_off'],
            increase=q['D_off']-base['D_off'],moved_fraction=float(np.mean(perm!=np.arange(len(cal))))))
    # Numerically verify old score invariance on one explicit reassociation.
    original=old.mapped(raw_cal,device); scrambled=old.mapped(shuffled,device)
    original_score={k:statistic(v,old.targets[k]) for k,v in original.items()}
    shuffled_score={k:statistic(v,old.targets[k]) for k,v in scrambled.items()}
    maximum=max(abs(original_score[k]['D_off']-shuffled_score[k]['D_off']) for k in original)
    assert maximum<1e-10
    del original,scrambled
    rt.write(OUT/'coupling_sanity.json',dict(original=base,permutations=coupling,old_max_abs_Doff_change=maximum,
        detects_reassociation=all(q['increase']>0 for q in coupling),
        qualification='sensitivity to feature-level dependence only; not proof of biological mode recovery'))
    print('Coupling audit',coupling[0],flush=True)
    # Model pool opened only after map, patient target, and calibration freeze.
    rows=[]
    for sp in sorted((SOURCE/'scores').glob('*.json')):
        score=rt.read(sp)
        if score['J'] is None: continue
        r,t,cn=pilot.dense.load_times(Path(score['source'])); assert cn==names
        phi=mapping.features(old.raw_features(t),device); q=statistic(phi,target)
        rows.append(dict(candidate=score['candidate'],N=len(t),old_J=score['J'],joint_scaled=q['D_off']/scale,
            joint_A=q['A'],joint_B=q['B'],joint_Doff=q['D_off'],source=score['source']))
    # Matched-size patient block reference: consecutive CAL events within block.
    # Descriptive sampling distribution, not an iid significance test.
    blocks=ev.blocks[calids]; refrows=[]
    for n in sorted({r['N'] for r in rows}):
        eligible=[np.flatnonzero(blocks==b) for b in np.unique(blocks) if np.sum(blocks==b)>=n]
        if not eligible: continue
        values=[]
        for rep in range(128):
            ids=eligible[int(rng.integers(len(eligible)))]; start=int(rng.integers(len(ids)-n+1)); chunk=ids[start:start+n]
            values.append(statistic(fcal[chunk],target)['D_off']/scale)
        refrows.append(dict(N=n,median=float(np.median(values)),q05=float(np.quantile(values,.05)),q95=float(np.quantile(values,.95)),draws=128))
    rt.write(OUT/'patient_block_reference.json',refrows)
    # Labels only enter this separate post-scoring diagnostic association.
    diagnostic=rt.read(SOURCE/'analysis/diagnostic_observables.json')
    for row in rows:
        for mode in ['ALL','TA','TB']:
            a=diagnostic['runs'][row['candidate']][mode]; b=diagnostic['patient'][mode]
            vals=[abs(v['order_probability']-b['pairs'][k]['order_probability']) for k,v in a['pairs'].items()
                  if v['shaft']=='SCL' and v['order_probability'] is not None and b['pairs'][k]['order_probability'] is not None]
            row[mode+'_SCL_error']=float(np.mean(vals)) if vals else None
            row[mode+'_N']=a['N']
    with (OUT/'candidate_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    old_best=min(rows,key=lambda r:r['old_J']); joint_best=min(rows,key=lambda r:r['joint_scaled'])
    order_old=[r['candidate'] for r in sorted(rows,key=lambda r:r['old_J'])]
    order_joint=[r['candidate'] for r in sorted(rows,key=lambda r:r['joint_scaled'])]
    from scipy.stats import spearmanr
    summary=dict(status='OFFLINE_AUDIT_COMPLETE_NOT_ADOPTED',candidate_count=len(rows),old_best=old_best,joint_best=joint_best,
        top10_overlap=len(set(order_old[:10])&set(order_joint[:10])),
        score_rank_correlation=float(spearmanr([r['old_J'] for r in rows],[r['joint_scaled'] for r in rows]).statistic),
        selected=[r for r in rows if r['candidate'] in ['g2_b03_p01','dense_axis3_12','lf_bo02_01']],
        coupling_all_positive=all(q['increase']>0 for q in coupling),time=time.time())
    rt.write(OUT/'summary.json',summary)
    plot(rows,summary,refrows)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def plot(rows,summary,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
        'font.size':8,'axes.labelsize':9,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none'})
    F=OUT/'figures';F.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    fig.subplots_adjust(left=.095,right=.96,bottom=.23,top=.80,wspace=.35)
    a,b=axes
    a.scatter([r['old_J'] for r in rows],[r['joint_scaled'] for r in rows],s=12,c='#B8BDC2',edgecolors='none')
    selected=[('g2_b03_p01','上一展示点','#8B6A9B','^'),('dense_axis3_12','原目标最佳','#0072B2','o'),
              ('lf_bo02_01','新增联合点最佳','#D55E00','s')]
    selected=[(cid,'联合表征最低点' if cid==summary['joint_best']['candidate'] else label,col,marker)
              for cid,label,col,marker in selected]
    byid={r['candidate']:r for r in rows}
    for cid,label,col,marker in selected:
        r=byid[cid];a.scatter(r['old_J'],r['joint_scaled'],c=col,s=30,marker=marker,label=label,zorder=4)
        b.plot([0,1,2],[r[m+'_SCL_error'] for m in ['ALL','TA','TB']],color=col,marker=marker,markersize=4,lw=1.1)
    j=summary['joint_best']
    if j['candidate'] not in [c[0] for c in selected]:
        a.scatter(j['old_J'],j['joint_scaled'],s=90,marker='*',c='#111111',label='联合表征最低点',zorder=5)
        b.plot([0,1,2],[j[m+'_SCL_error'] for m in ['ALL','TA','TB']],color='#111111',marker='*',ms=7,lw=1.1)
    a.set(xlabel='原三项无标签损失 ↓',ylabel='同事件联合表征距离 ↓',title='同一批冻结轨迹重评分')
    b.set(xticks=[0,1,2],xticklabels=['整体','TA 诊断','TB 诊断'],ylabel='SCL 先后概率误差 ↓',ylim=(0,.6),title='SCL 条件模式是否同步改善')
    b.axhline(0,color='#888888',ls=':',lw=.6)
    fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.985),ncol=3,fontsize=8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            box=t.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,t.get_text()
    fig.savefig(F/'joint_observation_audit.png',dpi=400);fig.savefig(F/'joint_observation_audit.svg');plt.close(fig)
    (F/'README.md').write_text('### joint_observation_audit.png\n左图为同一批冻结模型轨迹的新旧无标签分数比较，两轴分别按各自患者尺度缩放，数值大小不能直接比较。右图仅作事后SCL诊断，TA/TB标签不进入联合表征的构建、标定或候选选择。\n**关注点**：联合距离改善不等于TB恢复；每点仅为同一网络/噪声的一条轨迹，尚无独立确认。\n')
    rt.write(OUT/'figure_metadata.json',dict(source=str(OUT/'candidate_scores.csv'),producer=str(Path(__file__).resolve()),
        labels='post-score diagnostic only',human_visual_review='pending',formats=['png','editable_svg'],not_a_live_objective=True))


if __name__=='__main__': main()

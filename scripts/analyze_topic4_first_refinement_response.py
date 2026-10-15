"""Fixed first completed refinement contrast; not a candidate nomination."""
import json
import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import spearmanr
from scripts import analyze_topic4_geometry_threshold_refinement as analysis

p, rt, run = analysis.p, analysis.rt, analysis.run
OUT = run.OUT / 'first_refinement_response'
SEED = 847101
CONDITIONS = [('baseline', '原半径'), ('expand_A_4', '左核 4 mm'),
              ('A4_dose_preserved', '左核 4 mm＋保持阈值总量')]


def paired_summary():
    source=run.OUT/'running_observations/per_run_observations.csv'
    rows=list(csv.DictReader(source.open()))
    seeds=[847101,847102]
    selected=[r for r in rows if r['candidate'] in {c for c,_ in CONDITIONS} and int(r['seed']) in seeds]
    assert len(selected)==12 and all(r['physical_status']=='COMPLETE_NO_RUNAWAY' for r in selected)
    out=OUT/'paired_replays';figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    rt.write(out/'source_observations.json',dict(source=str(source),source_sha256=rt.sha(source),rows=selected))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    metrics=[('participation_MAE','参与概率误差 ↓'),('rank_rho','平均质心顺序相关 ↑'),
             ('no_SCL','整条 SCL 缺失比例 ↓'),('fraction','模式标签占比'),('OOD_fraction','患者特征邻域外比例 ↓')]
    fig,axes=plt.subplots(2,5,figsize=(17,8),layout='constrained')
    for row,(mode,lab) in enumerate([(1,'TA'),(0,'TB')]):
        for ax,(key,title) in zip(axes[row],metrics):
            for j,(seed,color) in enumerate(zip(seeds,['#277da8','#e47832'])):
                values=[float(next(r for r in selected if r['candidate']==cid and int(r['seed'])==seed and int(r['mode'])==mode)[key]) for cid,_ in CONDITIONS]
                ax.plot(range(3),values,'o-',color=color,label=f'噪声重演 {j+1}（{seed}）')
            ax.set(title=f'{lab} · {title}',xticks=range(3),xticklabels=[label for _,label in CONDITIONS])
            ax.tick_params(axis='x',rotation=35,labelsize=8);ax.grid(alpha=.15)
            ax.set_ylim(0,1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('同一张网络、两条配对噪声：扩大左核后，减弱阈值调制的影响\n颜色只区分噪声重演；每点是一条完整 90 秒运行；两个差值同向不等于统计显著或完整传播恢复')
    for extension in ['png','pdf']:fig.savefig(figures/f'paired_observations.{extension}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    (figures/'README.md').write_text('### paired_observations.png\n\n固定比较原半径、左核 4 mm、左核 4 mm 且保持双核阈值偏移总量；全部条件共享一张网络，蓝色和橙色仅区分两条动力学噪声重演。每点来自完整 90 秒运行中排除启动期后的全部合格事件；没有按患者相似度选择事件。PDF 为同图。**关注点**：TB 参与和平均顺序是否改善，以及 TA 顺序、SCL 缺失和邻域外事件是否同步改善；模式占比没有单调越大越好的方向。\n')
    print(json.dumps(dict(output=str(out),n_runs=6)))


def main():
    global SEED, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--paired-summary',action='store_true')
    parser.add_argument('--seed',type=int,choices=[847101,847102],default=847101)
    args=parser.parse_args()
    if args.paired_summary:
        paired_summary()
        return
    SEED=args.seed
    if SEED!=847101:
        OUT=OUT/f'noise_{SEED}'
    OUT.mkdir(exist_ok=True)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    plan = rt.read(run.OUT / 'plan.json')
    evaluator = rt.load_evaluator(plan['parent_design'])
    patient, patient_labels = np.asarray(evaluator.fit), np.asarray(evaluator.fit_labels)
    p.ANALYSIS_END_MS = 90000
    p.p.path_for = analysis.path_for
    units, rows, thresholds, records = {}, [], [], []
    for cid, name in CONDITIONS:
        c = {'id': cid, 'reference': cid != 'A4_dose_preserved'}
        unit = p.load_unit(c, SEED)
        if unit is None or unit[0]['actual_duration_ms'] != 90000:
            raise RuntimeError(f'Incomplete fixed comparison: {cid}')
        r, ar, ids = unit
        units[cid] = unit
        records.append(r)
        names = ar['contact_names']
        scl, icl = np.char.startswith(names, 'SCL'), np.char.startswith(names, 'ICL')
        for mode, lab in [(1, 'TA'), (0, 'TB')]:
            selected = ids[ar['event_mode'][ids] == mode]
            table, ref = ar['centroid_ms'][selected], patient[patient_labels == mode]
            mean_rank, ref_rank = p.avg(p.ranks(table)), p.avg(p.ranks(ref))
            valid = np.isfinite(mean_rank) & np.isfinite(ref_rank)
            widths = []
            for i in selected:
                lo, hi = r['events'][int(i)]['window_ms']
                env = ar['contact_envelope'][int(lo / 2):int(hi / 2)]
                cumulative = np.cumsum(env, axis=0)
                mass = cumulative[-1]
                mask = np.isfinite(ar['centroid_ms'][i]) & (mass > 0)
                q10, q90 = [np.argmax(cumulative >= q * mass, axis=0) * 2 for q in (.1, .9)]
                widths.append(float(np.median((q90 - q10)[mask])))
            rows.append(dict(candidate=cid, label=name, seed=SEED, mode=lab,
                n=len(selected), fraction=len(selected) / len(ids),
                participation_MAE=float(np.abs(np.isfinite(table).mean(0) - np.isfinite(ref).mean(0)).mean()),
                rank_rho=float(spearmanr(mean_rank[valid], ref_rank[valid]).statistic),
                no_SCL=float((~np.isfinite(table)[:, scl].any(1)).mean()),
                no_ICL=float((~np.isfinite(table)[:, icl].any(1)).mean()),
                OOD_fraction=float(np.mean([r['events'][int(i)]['support'] == -1 for i in selected])),
                local_width_ms_q05_median_q95=np.percentile(widths, [5, 50, 95]).tolist()))
        with np.load(analysis.path_for(c, SEED).with_suffix('.npz')) as z:
            for core in ['A', 'B']:
                indices = z[f'group_core{core}E']
                delta = z['vtheta'][indices].astype(float) - 18
                thresholds.append(dict(candidate=cid, core=core, n=len(indices),
                    mean_threshold_mV=float(z['vtheta'][indices].mean()),
                    lowering_sum_mV=float(-delta[delta < 0].sum()),
                    raising_sum_mV=float(delta[delta > 0].sum())))
    rt.write(OUT / 'observations.json', rows)
    rt.write(OUT / 'thresholds_by_core.json', thresholds)
    paired = dict(same_original_external_draws=len({r['state_audit']['legacy_external_counts_sha256'] for r in records}) == 1,
                  same_graph_weights=all(all(r['static_array_identity'][key] == records[0]['static_array_identity'][key]
                    for key in ['ampa_topology_sha256', 'ampa_values_sha256', 'gaba_topology_sha256', 'gaba_values_sha256']) for r in records))
    rt.write(OUT / 'paired_checks.json', paired)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'pdf.fonttype': 42})
    def save(fig, name):
        for extension in ['png', 'pdf']:
            fig.savefig(figures / f'{name}.{extension}', dpi=150, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    limit=max(float(np.abs(ar['vtheta'][:len(ar['h'])]-18).max()) for _,ar,_ in units.values())
    for ax, (cid, name) in zip(axes, CONDITIONS):
        _, ar, _ = units[cid]
        condition=next(c for c in plan['references']+plan['candidates'] if c['id']==cid)
        sample=np.arange(0,len(ar['h']),3)
        im=ax.scatter(*ar['positions_E'][sample].T,c=ar['vtheta'][sample]-18,s=2,
                      cmap='coolwarm',vmin=-limit,vmax=limit,linewidths=0,rasterized=True)
        for center,radius in zip(condition['centers_mm'],condition['radii_mm']):
            ax.add_patch(Circle(center,radius,fill=False,color='black',lw=.8))
        for rod in ['SCL','ICL']:
            indices=sorted([i for i,n in enumerate(ar['contact_names']) if n.startswith(rod)],
                           key=lambda i:int(ar['contact_names'][i][len(rod):]),reverse=True)
            xy=ar['contact_xy_mm'][indices]
            ax.plot(*xy.T,color='black',lw=.7)
            ax.scatter(*xy.T,s=15,facecolors='white',edgecolors='black',linewidths=.5)
            for i in [indices[0],indices[-1]]:
                ax.annotate(ar['contact_names'][i],ar['contact_xy_mm'][i],xytext=(3,3),textcoords='offset points',fontsize=7)
        means=[t['mean_threshold_mV'] for t in thresholds if t['candidate']==cid]
        ax.text(.03,.02,f'核 A / B 均值：{means[0]:.3f} / {means[1]:.3f} mV',transform=ax.transAxes,fontsize=9)
        ax.set(title=name,xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=list(axes),shrink=.75,label='兴奋性细胞阈值相对背景 18 mV 的变化 (mV)')
    fig.suptitle('实际阈值场与固定 SEEG 布局；三个条件共用色标\n颜色表示细胞参数，不是活动；蓝色阈值较低，红色阈值较高；圆圈为 core 范围')
    save(fig,'actual_threshold_fields')

    metrics = [('participation_MAE', '参与概率误差 ↓'), ('rank_rho', '平均质心顺序相关 ↑'),
               ('no_SCL', '整条 SCL 缺失比例 ↓'), ('fraction', '模式标签占比'),
               ('OOD_fraction', '患者特征邻域外比例 ↓')]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), layout='constrained')
    for row, lab in enumerate(['TA', 'TB']):
        for ax, (metric, title) in zip(axes[row], metrics):
            values = [next(r[metric] for r in rows if r['candidate'] == cid and r['mode'] == lab) for cid, _ in CONDITIONS]
            ax.plot(range(3), values, 'o-', color='#336f91')
            ax.set(title=f'{lab} · {title}', xticks=range(3), xticklabels=[name for _, name in CONDITIONS])
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.margins(y=.20)
            bottom, top = ax.get_ylim()
            ax.set_ylim(max(-1 if metric == 'rank_rho' else 0, bottom), min(1, top))
            ax.grid(alpha=.15)
            for x, y in enumerate(values):
                ax.annotate(f'{y:.3f}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    fig.suptitle(f'固定参数对照：同一网络、噪声 {SEED}；每点是一条完整 90 秒重演\n单噪声开发诊断，不能据此提名或接受患者传播恢复')
    save(fig, 'paired_observations')

    patients = analysis.patient_examples()
    display_names = [f'SCL{i}' for i in range(9, 5, -1)] + [f'ICL{i}' for i in range(11, 0, -1)]
    chosen = CONDITIONS[1:]
    data, selections = {}, []
    for lab, mode in [('TA', 1), ('TB', 0)]:
        d = patients[lab]
        order = [list(d['names']).index(name) for name in display_names]
        data[(lab, 0)] = dict(mass=d['mass'][order], mask=d['mask'][order], id=d['event_id'])
        for col, (cid, _) in enumerate(chosen, 1):
            r, ar, ids = units[cid]
            indices = ids[ar['event_mode'][ids] == mode]
            phi = ar['event_phi'][indices]
            i = int(indices[np.argmin(((phi - phi.mean(0)) ** 2).sum(1))])
            lo, hi = r['events'][i]['window_ms']
            order = [list(ar['contact_names']).index(name) for name in display_names]
            data[(lab, col)] = dict(mass=ar['contact_envelope'][int(lo / 2):int(hi / 2)][:, order].T,
                                   mask=np.isfinite(ar['centroid_ms'][i])[order], id=i)
            selections.append(dict(candidate=cid, seed=SEED, mode=lab, event_index=i, window_ms=[lo, hi]))
    rt.write(OUT / 'representative_events.json', dict(rule='Nearest event to its own run and mode feature mean; not nearest to patient.', events=selections))
    for scale in ['event', 'contact']:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
        for row, lab in enumerate(['TA', 'TB']):
            for col in range(3):
                d = data[(lab, col)]
                mass = d['mass']
                denominator = max(float(mass.max()), 1e-20) if scale == 'event' else np.maximum(mass.max(1, keepdims=True), 1e-20)
                ax = axes[row, col]
                im = ax.imshow(mass / denominator, aspect='auto', extent=[0, 250, 14.5, -.5], cmap='magma', vmin=0, vmax=1, interpolation='nearest')
                ax.axhline(3.5, color='#66bbbb', lw=.7)
                ax.set(title=('患者 Fig2C' if col == 0 else chosen[col - 1][1]) + f' · {lab} · 事件 {d["id"]}',
                       yticks=range(15), yticklabels=[name + (' *' if not valid else '') for name, valid in zip(display_names, d['mask'])], xlabel='原始事件窗口时间 (ms)')
                ax.tick_params(axis='y', labelsize=8)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65, label='包络 / ' + ('整事件峰值' if scale == 'event' else '各接触点峰值'))
        fig.suptitle('固定的单噪声前后对照：患者／扩大左核／扩大左核并保持阈值总量\n相同触点顺序及 250 ms 时间轴；患者为 HFO 包络，模型为完整发放密度包络；* 为未参与触点')
        save(fig, f'patient_parent_candidate_{scale}_scale')
    if args.native:
        nativeout=OUT/'native_review'
        nativeout.mkdir(exist_ok=True)
        nativefig=nativeout/'figures'
        nativefig.mkdir(exist_ok=True)
        selected=[dict(next(c for c in plan['references'] if c['id']=='expand_A_4'),reference=True),
                  next(c for c in plan['candidates'] if c['id']=='A4_dose_preserved')]
        local=copy.deepcopy(plan)
        local.update(candidates=selected,training_seeds=[SEED],observation_seeds=[SEED])
        rt.write(nativeout/'plan.json',local)
        rt.write(nativeout/'all_candidates.json',selected)
        objective=rt.load_objective(plan['parent_design'])
        def scoring(c,seeds):
            if any(s != SEED for s in seeds):
                raise ValueError('This fixed comparison contains only noise 847101')
            scores=[dict(seed=s,**objective.score_network(units[c['id']][1]['centroid_ms'][units[c['id']][2]])) for s in seeds]
            return dict(candidate=c,units=scores,loss_off=float(np.mean([s['loss_off'] for s in scores])) if all(s['loss_off'] is not None for s in scores) else None)
        p.OUT=nativeout;p.F=nativefig;p.SHOW_ALL_GEOMETRIES=True
        p.RUN_ROLE='single_noise_diagnostic';p.MODE_NAMES={0:'TB (M0)',1:'TA (M1)'};p.MODE_ORDER=[1,0]
        p.p.score_candidate=scoring
        sys.argv=['plot']
        p.main()
    print(json.dumps({'output': str(OUT), 'observations': rows, 'paired_checks': paired}, ensure_ascii=False))


if __name__ == '__main__':
    main()

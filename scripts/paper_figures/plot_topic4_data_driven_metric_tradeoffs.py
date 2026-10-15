"""Compare observed trade-offs among recorded, label-free Bayesian proposals.

Only actual simulations are plotted. Lines join three categorical observables
of ONE candidate, not interpolated parameter responses or optimizer trajectories.
"""
from pathlib import Path
from itertools import combinations
import csv
import hashlib
import json
import math

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE = Path('/data/hfosp/topic4_sef_hfo')
ROUND = BASE / 'geometry_axis_joint_distribution_20260917'
PRIOR = BASE / 'label_free_dense_parameter_search_20260915'
OUT = ROUND / 'key_results/data_driven_metric_tradeoffs'
FIG = OUT / 'figures'
METRICS = ['rank_error', 'within_rod_order_error', 'participation_error']
LABELS = ['平均 rank 误差', '杆内先后概率误差', '触点参与概率误差']
COLORS = ['#377DA3', '#7863A8', '#C38B25', '#C35367', '#268777']
MARKERS = ['o', 's', '^', 'D', 'P']
ROLES = ['平均顺序最好', '杆内顺序最好', '参与概率最好', '原分布损失最低', '联合分布损失最低']
PARAMS = ['left_x_mm', 'left_y_mm', 'right_x_mm', 'right_y_mm', 'EE_core_to_out_scale', 'EE_angle_offset_deg']

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'legend.frameon': False, 'savefig.facecolor': 'white',
})


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_contract():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    contract = dict(
        question='无标签优化发现的不同参数组合，在哪些观测上更接近患者、在哪些观测上更远？',
        conclusion='实测候选存在三项观测的取舍；最低分布损失不等于三项观测同时最好。',
        archetype='quantitative grid', backend='python',
        evidence=dict(main='五个预定选点规则对应的实际候选及其参数',
                      companion='所有可追溯无标签贝叶斯提案的三项观测两两比较'),
        pool_rule='Sep15 and Sep17 recorded Bayesian proposals only; exclude warm starts, grids, factorial cells, and old label-aware BO',
        selection='argmin of each of three ALL observables and two frozen losses; union, no additional filtering',
        labels_used=False, new_simulations=0,
        patient='frozen FIT, 19770 events; no 32+32 morphology packet used for numerical comparison',
        metrics=dict(
            rank_error='Within-event participating contact ranks normalized to [0,1]; contact conditional means compared by absolute error; contacts equal within shaft, shafts equal.',
            within_rod_order_error='For each jointly participating within-shaft contact pair, P(t_j>t_i)+0.5*P(t_j=t_i); absolute model-patient difference; pairs equal within shaft and shafts equal.',
            participation_error='Per-contact participation probability absolute model-patient difference; contacts equal within shaft and shafts equal.'),
        statistics=dict(unit='one 60 s trajectory per parameter combination; topology 2511, noise 847401; events nested in trajectory',
                        intervals='none; single fixed topology/noise, no empirical between-seed uncertainty; no iid event error bars',
                        tests='none; observed differences, not significance or unique mechanism identification'),
        comparable_parameters='Same frozen physics except two core XY positions, core-to-outside EE weight and global EE axis offset.',
        limits=['Joint parameter changes are not isolated causal effects.',
                'Same seed does not guarantee unchanged adjacency or identical timewise noise innovations when geometry changes.',
                'Two search rounds have different search ranges and objectives; do not attribute differences to optimizer superiority.',
                'Post hoc observable winners are not independently trained single-objective models.',
                'Observed Pareto membership does not prove a structural trade-off or a global optimum.',
                'No scalar Overall score is constructed from the three observables.'],
        export=dict(main_mm=[183,142], companion_mm=[183,77], formats=['png','svg'], dpi=400,
                    min_font_pt=6, editable_svg_text=True, pdf='not requested; omitted'),
    )
    write(OUT/'figure_contract.json', contract)
    return contract


def recompute(times, names, patient):
    mask = np.isfinite(times)
    ranks = np.full_like(times, np.nan, dtype=float)
    for i, row in enumerate(times):
        n = int(mask[i].sum())
        assert n >= 2
        ranks[i, mask[i]] = (rankdata(row[mask[i]], method='average') - 1) / (n - 1)
    errors = []
    n_pairs = []
    means = np.nanmean(ranks, axis=0)
    for shaft in ['SCL', 'ICL']:
        ix = [i for i, n in enumerate(names) if n.startswith(shaft)]
        rank_error = np.mean([abs(means[i]-patient['contacts'][names[i]]['mean']) for i in ix])
        part_error = np.mean([abs(mask[:,i].mean()-patient['contacts'][names[i]]['participation']) for i in ix])
        pair_errors = []
        for i, j in combinations(ix, 2):
            valid = mask[:,i] & mask[:,j]
            assert valid.any(), 'do not silently drop a missing pair'
            dt = times[valid,j] - times[valid,i]
            prob = np.mean((dt > 0) + .5*(dt == 0))
            pair_errors.append(abs(prob-patient['pairs'][f'{names[i]}→{names[j]}']['order_probability']))
        n_pairs.append(len(pair_errors))
        errors.append([rank_error, np.mean(pair_errors), part_error])
    assert n_pairs == [6,55]
    return np.mean(errors, axis=0)


def gather():
    table = {r['candidate']: r for r in csv.DictReader(open(ROUND/'analysis/observables.csv')) if r['mode']=='ALL'}
    patient = read(ROUND/'analysis/diagnostic_observables.json')['patient']['ALL']
    assert patient['N'] == 19770
    provenance = {}
    for root, n in [(PRIOR,4),(ROUND,3)]:
        for b in range(1,n+1):
            path = root/'proposals'/f'batch_{b:02d}.json'
            prop = read(path)
            assert len(prop['ids']) == 8
            for cid, v, kind in zip(prop['ids'],prop['vectors'],prop['kinds']):
                assert cid not in provenance
                assert kind in ['CUDA_autograd_qLogEI','global_posterior_uncertainty','qLogNEHVI']
                provenance[cid] = dict(proposal=str(path), proposal_sha256=sha(path),
                                       batch=b, proposal_kind=kind, proposed_vector=v,
                                       search_round=root.name)
    assert len(provenance) == 56
    rows = []
    fixed_params = None
    for cid, prov in sorted(provenance.items()):
        score_path = ROUND/'scores'/f'{cid}.json'
        score = read(score_path)
        assert score['status']=='SCORABLE' and score['N']>=16
        np.testing.assert_allclose(score['vector'],prov['proposed_vector'],atol=1e-10,rtol=0)
        src = Path(score['source'])
        assert sha(src)==score['source_sha256']
        trajectory = read(src)
        assert trajectory['physical_status']=='COMPLETE_NO_RUNAWAY'
        assert trajectory['actual_duration_ms']==60000
        app_path = src.parents[1]/'applied_physics.json'
        app = read(app_path)
        assert app['topology_seed']==2511 and app['dynamics_seed']==847401
        assert app['threshold']['n_raised']==0
        candidate = app['candidate']
        pars = candidate['parameters']
        actual = [*np.asarray(candidate['centers_mm']).ravel(),
                  app['graph']['stage_audits']['weights']['factors']['EE_core_to_out_scale'],
                  app['graph']['kernel']['angle_offset_deg']]
        np.testing.assert_allclose(actual,score['vector'],atol=1e-10,rtol=0)
        frozen = {k:v for k,v in pars.items() if k not in ['EE_core_to_out_scale','EE_angle_offset_deg']}
        frozen.update(radii=candidate['radii_mm'],core_mean_rate_scale=candidate['core_mean_rate_scale'],
                      core_ou_correlation=candidate['core_ou_correlation'])
        if fixed_params is None: fixed_params=frozen
        assert frozen==fixed_params, 'incompatible background parameter condition'
        with np.load(src.with_suffix('.npz')) as z:
            names=z['contact_names'].astype(str).tolist()
            ids=np.array([int(i) for i in z['primary_event_indices']
                          if trajectory['events'][i]['window_ms'][0]>=1500
                          and trajectory['events'][i]['window_ms'][1]<=60000],int)
            t=z['centroid_ms'][ids]
        assert len(ids)==score['N']==int(table[cid]['N'])
        observed = recompute(t,names,patient)
        np.testing.assert_allclose(observed,[float(table[cid][k]) for k in METRICS],atol=1e-12,rtol=0)
        row=dict(candidate=cid,N=len(ids),**dict(zip(METRICS,map(float,observed))),
                 J=score['J'],J_joint=score['J_joint'],**dict(zip(PARAMS,map(float,actual))),
                 source=str(src),source_sha256=sha(src),applied_physics=str(app_path),
                 applied_physics_sha256=sha(app_path),**{k:v for k,v in prov.items() if k!='proposed_vector'})
        rows.append(row)
    x=np.array([[r[k] for k in METRICS] for r in rows])
    for i,r in enumerate(rows):
        r['observed_pareto']=not np.any(np.all(x<=x[i],axis=1)&np.any(x<x[i],axis=1))
    selected=[]
    for i,key in enumerate(METRICS+['J','J_joint']):
        winner=min(rows,key=lambda r:(r[key],r['candidate']))
        assert winner['observed_pareto']
        assert winner['candidate'] not in [r['candidate'] for r in selected]
        selected.append(dict(winner,display_id=f'P{i+1}',role=ROLES[i],color=COLORS[i],marker=MARKERS[i],selection_metric=key))
    for name, data in [('all_bayesian_candidates.csv',rows),('selected_candidates.csv',selected)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(OUT/'selection.json',dict(pool_N=len(rows),all_compatible_pool_N=len(table),
        excluded_N=len(table)-len(rows),exclusion='not from the two recorded label-free Bayesian proposal streams',
        observed_pareto_N=sum(r['observed_pareto'] for r in rows),selected=selected,
        no_labels_consumed=True,raw_trajectories_recomputed=len(rows),frozen_parameters=fixed_params,
        source_hashes={str(p):sha(p) for p in [ROUND/'analysis/observables.csv',ROUND/'analysis/diagnostic_observables.json',Path(__file__)]}))
    return rows, selected


def save(fig, stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    errors=[]
    fonts=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bbox=text.get_window_extent(renderer)
            fonts.append(text.get_fontsize())
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1+1 or bbox.y1 > fig.bbox.y1+1:
                errors.append(text.get_text())
    assert not errors, errors
    assert min(fonts)>=6
    fig.savefig(FIG/f'{stem}.png',dpi=400)
    fig.savefig(FIG/f'{stem}.svg')
    plt.close(fig)
    return dict(stem=stem,min_font_pt=min(fonts),text_outside_canvas=errors)


def profiles(selected):
    fig=plt.figure(figsize=(183/25.4,142/25.4))
    ax=fig.add_axes([.10,.46,.57,.44])
    legend=fig.add_axes([.71,.46,.28,.44]);legend.axis('off')
    for i,r in enumerate(selected):
        y=[r[k] for k in METRICS]
        ax.plot(range(3),y,color=r['color'],marker=r['marker'],lw=1.4,ms=4,
                label=f"{r['display_id']} {r['role']}",zorder=3)
        yy=.90-i*.17
        legend.plot([.00,.12],[yy,yy],transform=legend.transAxes,color=r['color'],
                    marker=r['marker'],markevery=[1],lw=1.4,ms=4)
        legend.text(.17,yy,f"{r['display_id']}  {r['role']}",transform=legend.transAxes,va='center',fontsize=7)
        legend.text(.17,yy-.07,' / '.join(f'{v:.3f}' for v in y),transform=legend.transAxes,
                    va='center',fontsize=6.5,color='#555555')
    ax.set(xlim=(-.12,2.12),ylim=(0,.46),xticks=range(3),
           xticklabels=['平均 rank\n误差','杆内先后概率\n误差','触点参与概率\n误差'],ylabel='患者–模型绝对误差 ↓')
    ax.set_yticks([0,.1,.2,.3,.4]);ax.grid(axis='y',color='#E9E9E9',lw=.55)
    ax.set_axisbelow(True)
    ax.axhline(0,color='#888888',lw=.7,ls='--')
    fig.text(.05,.951,'a',fontsize=9,weight='bold')
    fig.text(.10,.950,'同一个参数组合，在三个观测上有不同的取舍',fontsize=9)
    fig.text(.10,.917,'56 个无标签贝叶斯提案；各观测内比较高低，不跨观测求和',fontsize=6.5,color='#555555')
    fig.text(.10,.364,'b   每条彩线对应的实际参数组合',fontsize=8,weight='bold')
    tabax=fig.add_axes([.075,.106,.92,.223]);tabax.axis('off')
    cell=[]
    for r in selected:
        cell.append([f"{r['display_id']}  {r['role']}",
                     f"({r['left_x_mm']:.2f}, {r['left_y_mm']:.2f})",
                     f"({r['right_x_mm']:.2f}, {r['right_y_mm']:.2f})",
                     f"{r['EE_core_to_out_scale']:.3f}",
                     f"{r['EE_angle_offset_deg']:+.2f}",str(r['N'])])
    table=tabax.table(cellText=cell,colLabels=['方案','左核 (x, y) mm','右核 (x, y) mm','向外 EE 倍率','EE 轴偏移 (°)','事件数'],
        colWidths=[.28,.21,.21,.115,.12,.065],cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(6.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD');cell.set_linewidth(.45)
        if r==0:
            cell.set_facecolor('#F2F3F4');cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white')
            if c==0:cell.set_text_props(color=selected[r-1]['color'],weight='bold',ha='left')
    fig.text(.075,.061,'患者 FIT：19,770 个事件；每组合 60 s，固定拓扑 2511 / 噪声 847401；SCL 与 ICL 等权。',fontsize=6.3)
    fig.text(.075,.035,'每条线连接同一组合的三项实测误差；不是单参数响应曲线。“最好”仅指本次候选池的实测最小值。',fontsize=6.3)
    return save(fig,'three_observable_profiles')


def pairwise(rows,selected):
    fig,axs=plt.subplots(1,3,figsize=(183/25.4,77/25.4))
    fig.subplots_adjust(left=.09,right=.984,bottom=.29,top=.76,wspace=.43)
    limits={k:math.ceil(max(r[k] for r in rows)*1.08/.05)*.05 for k in METRICS}
    for ax,(a,b),letter in zip(axs,[(0,1),(0,2),(1,2)],'abc'):
        ka,kb=METRICS[a],METRICS[b]
        ax.scatter([r[ka] for r in rows],[r[kb] for r in rows],s=10,c='#C8CDD2',edgecolors='none',zorder=1)
        pareto=[r for r in rows if r['observed_pareto']]
        ax.scatter([r[ka] for r in pareto],[r[kb] for r in pareto],s=19,facecolors='none',edgecolors='#7F858B',lw=.6,zorder=2)
        for r in selected:
            ax.scatter(r[ka],r[kb],s=26,marker=r['marker'],c=r['color'],edgecolors='white',lw=.35,zorder=4)
            dx,dy={'P1':(-9,5),'P2':(4,-10),'P3':(4,2),'P4':(5,4),'P5':(-13,-11)}[r['display_id']]
            ax.annotate(r['display_id'],(r[ka],r[kb]),xytext=(dx,dy),textcoords='offset points',
                        fontsize=6,color=r['color'],weight='bold',zorder=5)
        ax.set(xlim=(0,limits[ka]),ylim=(0,limits[kb]),xlabel=LABELS[a]+' ↓',ylabel=LABELS[b]+' ↓')
        ax.grid(color='#EFEFEF',lw=.5);ax.set_axisbelow(True)
        ax.text(-.2,1.08,letter,transform=ax.transAxes,fontsize=9,weight='bold')
    fig.text(.09,.931,'所有 56 个优化器提案：三项观测之间的实测取舍',fontsize=9)
    fig.text(.09,.845,'灰点：全部候选；灰圈：三指标均不被其他候选同时改善的 13 个组合；彩色与上图对应。',fontsize=6.5)
    fig.text(.09,.137,'每点为一个参数组合；均已完成真实仿真。左下更好；不连拟合前沿、不插值。',fontsize=6.5)
    fig.text(.09,.069,'这是单网络 / 单噪声下的开发结果，不能据此认定不可消除的机制取舍或患者双模式恢复。',fontsize=6.3)
    return save(fig,'three_observable_pairwise')


def main():
    source_contract()
    rows,selected=gather()
    rendered=[profiles(selected),pairwise(rows,selected)]
    import xml.etree.ElementTree as ET
    from PIL import Image
    for f in FIG.glob('*.svg'):ET.parse(f)
    for f in FIG.glob('*.png'):
        with Image.open(f) as im:im.load()
    write(OUT/'qa.json',dict(raw_metric_parity='56/56, absolute tolerance 1e-12',
          all_proposals_match_applied_parameters=True,all_background_parameters_equal=True,
          decoded_all_outputs=True,figures=rendered,human_review='PENDING',agent_visual_review='PENDING',
          pdf_audit='NOT_APPLICABLE: user requested PNG/SVG, no PDF produced'))
    (FIG/'README.md').write_text(
        '### three_observable_profiles.png\n'
        '五条彩线来自两轮无标签贝叶斯提案的 56 个实际候选，分别按三项总体观测及两个冻结训练损失取实测最小者。下表给出同色组合的双核 XY、核内→核外 EE 倍率、全局 EE 轴偏移与事件数；所有候选的固定背景一致。\n'
        '**关注点**：不同指标的最优组合不同，连线仅连接同一候选的三个类别轴，不是控制其他参数后的剂量反应；三个误差不组成新的 Overall 指标。\n\n'
        '### three_observable_pairwise.png\n'
        '三格展示全部 56 个实际候选在三项观测两两构成的平面上，五个着色候选与主图一致。空心灰圈标出三维经验非支配集合，不是该二维投影独立计算的前沿，也不代表连续参数空间的真实边界。\n'
        '**关注点**：是否存在同时改善多个观测的候选，以及参与结构改善是否伴随时序误差增加；单种子、不做显著性判断。\n')
    (OUT/'scientific_note.md').write_text(
        '# 三项观测的 data-driven 候选取舍\n\n'
        '本图只使用两轮无标签贝叶斯算法派发的 56 个真实参数组合，24 个来自当前双目标轮，32 个来自上一轮；其他 170 个相容历史/网格/交互条件不进入这次比较池，避免把人工扫参曲线称为优化器发现。32 个旧提案中有 8 个由已拟合 GP 的后验不确定性选择，仍是数据驱动探索，不是随机手选。P3 正是这种探索提案。\n\n'
        '数值目标为患者完整冻结 FIT 的 19,770 个事件，不使用形态小包或 TA/TB 标签。三个指标均先在每杆内平均，再令 SCL/ICL 各占一半；平均 rank 仅对参与事件计算，杆内顺序仅对共同参与的触点对计算，参与误差则覆盖全部事件。不同组合事件量不同，实际 N 随图报告。\n\n'
        '五条线依次取平均 rank、杆内顺序、参与误差、原训练损失、联合训练损失的实测最小值，是事后观测比较，不能写成五种观测权重分别重新优化。五者都处在本候选池三指标经验非支配集合上。不能将联合参数变化解释成某一个参数的独立作用；此前位置和方向改变亦会改变实际连接及神经元成员，相同种子并非所有物理细节完全相同。\n\n'
        'P1 顺序模板平均最接近但参与误差高；P2 杆内顺序最好；P3 参与误差最低，却有最大的杆内顺序误差；P4/P5 是当前两个冻结训练目标的最好组合，三项解释性观测并非同时最好。仅是现有搜索结果中的取舍，不证明这种冲突不可消除；仍不接受患者双模式完整恢复。\n\n'
        '图为 PNG 与可编辑 SVG，不生成 PDF 或图册。本次只重算、核验与绘图，不改变损失或运行新仿真。\n')
    print(json.dumps(dict(out=str(OUT),pool=len(rows),selected=[{k:r[k] for k in ['display_id','candidate','role',*METRICS]} for r in selected]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

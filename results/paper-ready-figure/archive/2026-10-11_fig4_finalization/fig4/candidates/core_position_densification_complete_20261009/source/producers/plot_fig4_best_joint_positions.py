#!/usr/bin/env python3
"""Low-J_joint core-position density with explicit pool, cutoff and sensitivity."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures import plot_fig4_joint_position_progress as progress
read,write,sha=progress.read,progress.write,progress.sha
RUN=progress.RUN
OLD=Path('/data/hfosp/topic4_sef_hfo/geometry_axis_joint_distribution_20260917')
FRACTIONS=(.05,.10,.20)
MAIN=.20

def freeze(out, run_dir=None):
    run=Path(run_dir) if run_dir is not None else RUN
    dest=out/'source/snapshot.json'
    if dest.exists():
        snapshot=read(dest)
        if run_dir is not None:assert Path(snapshot['source_run'])==run
        return snapshot
    paths=sorted((run/'scores').glob('*.json'));rows=[]
    objective=read(run/'plan.json')['objective']
    for path in paths:
        r=read(path);cid=r['candidate'];src=Path(r['source']);trajectory=read(src)
        assert sha(src)==r['source_sha256']
        assert trajectory['status']=='COMPLETE' and trajectory['actual_duration_ms']==60000.
        assert trajectory['job']['topology_seed']==2511 and trajectory['job']['dynamics_seed']==847401
        assert r['objective_sha256']==objective['sha256']
        assert r.get('J_joint') is None or np.isfinite(r['J_joint'])
        applied=read(src.parents[1]/'applied_physics.json')
        xy=np.array(applied['threshold']['centers_mm']);assert applied['threshold']['n_raised']==0
        np.testing.assert_allclose(xy.ravel(),r['vector'][:4],rtol=0,atol=1e-12)
        assert r['J_joint']==r['joint']['scaled']
        r.update(centers_mm=xy.tolist(),score_path=str(path),score_sha256=sha(path))
        rows.append(r);shutil.copy2(path,out/'source/scores'/path.name)
    # A trajectory is one observation even if several paths happen to refer to it.
    identities=[(r['source_sha256'],tuple(r['vector'])) for r in rows]
    assert len(set(identities))==len(rows)
    finite=sorted([r for r in rows if r['J_joint'] is not None],key=lambda r:(r['J_joint'],r['candidate']))
    selections={}
    for fraction in FRACTIONS:
        count=math.ceil(fraction*len(finite));cutoff=finite[count-1]['J_joint']
        selected=[r for r in finite if r['J_joint']<=cutoff]
        selections[str(fraction)]=dict(fraction=fraction,nominal_count=count,count=len(selected),cutoff_J_joint=cutoff,
            candidate_ids=[r['candidate'] for r in selected],historical=sum(r.get('reused',False) for r in selected),
            new=sum(not r.get('reused',False) for r in selected))
    state=read(run/'status.json');geometry=progress.style.source.plate.EXPANDED/'source/position_map_source.json'
    if not (out/'source/electrode_geometry.json').exists():shutil.copy2(geometry,out/'source/electrode_geometry.json')
    shutil.copy2(run/'plan.json',out/'source/plan.json')
    shutil.copy2(OLD/'plan.json',out/'source/old_joint_plan.json');write(out/'source/run_status.json',state)
    # Freeze raw geometry audits for every candidate entering the main/sensitivity displays.
    for r in finite[:selections[str(max(FRACTIONS))]['count']]:
        cid=r['candidate'];src=Path(r['source'])
        shutil.copy2(src,out/'source/selected_audits'/f'{cid}_trajectory.json')
        shutil.copy2(src.parents[1]/'applied_physics.json',out/'source/selected_audits'/f'{cid}_applied.json')
    snapshot=dict(asof=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),records=rows,
        pool_completed=len(rows),pool_scorable=len(finite),historical_completed=sum(r.get('reused',False) for r in rows),
        new_completed=sum(not r.get('reused',False) for r in rows),new_scorable=sum(not r.get('reused',False) for r in finite),
        selection_method='Rank every compatible finite frozen J_joint; lowest ceil(fraction*N), include exact cutoff ties; no position-based filtering.',
        main_fraction=MAIN,selections=selections,sigma_mm=.35,selection_design='Exploratory visualization selected after search, not a preregistered stability test.',
        objective='Same frozen J_joint for the entire pool, regardless of historical proposal objective.',
        legacy_search_boundary_source=str(OLD/'plan.json'),best=finite[0],source_run=str(run))
    write(dest,snapshot)
    with (out/'source/ranked_candidates.csv').open('w') as f:
        fields=['rank','candidate','J_joint','J','N','historical','A_x_mm','A_y_mm','B_x_mm','B_y_mm','selected_main']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();selected=set(selections[str(MAIN)]['candidate_ids'])
        for rank,r in enumerate(finite,1):
            w.writerow(dict(rank=rank,candidate=r['candidate'],J_joint=r['J_joint'],J=r['J'],N=r['N'],historical=r.get('reused',False),
                **dict(zip(fields[6:10],np.array(r['centers_mm']).ravel())),selected_main=r['candidate'] in selected))
    return snapshot

def sensitivity(out,data):
    old=read(out/'source/old_joint_plan.json')['bounds'];lo=np.array(old['lower'][:4]).reshape(2,2);hi=np.array(old['upper'][:4]).reshape(2,2)
    stats=[]
    for key,s in data['selections'].items():
        ids=set(s['candidate_ids']);rows=[r for r in data['records'] if r['candidate'] in ids];xy=np.array([r['centers_mm'] for r in rows])
        for core in range(2):
            x=xy[:,core];mean=x.mean(0);d=np.linalg.norm(x-mean,axis=1)
            diameter=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2).max()
            stats.append(dict(fraction=float(key),core='AB'[core],n=len(x),mean_xy_mm=mean.tolist(),
                std_xy_mm=x.std(0,ddof=1).tolist(),coordinate_range_mm=np.ptp(x,axis=0).tolist(),
                radial_median_mm=float(np.median(d)),radial_90pct_mm=float(np.quantile(d,.9)),diameter_mm=float(diameter),
                on_old_joint_box_boundary=int(np.any(np.isclose(x,lo[core],rtol=0,atol=1e-8)|np.isclose(x,hi[core],rtol=0,atol=1e-8),axis=1).sum()),
                boundary_interpretation='Geometric coincidence with the old joint search box; not every historical candidate was generated by this box.'))
    write(out/'selection_sensitivity.json',dict(statistics=stats,selections=data['selections'],sigma_mm=.35,
        interpretation='Sensitivity to selection fraction. Concentration is descriptive and affected by historical search coverage and adaptive sampling.'))
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path)
    p.add_argument('--snapshot-dir',type=Path,help='Reuse a prior frozen source pool for a directly comparable presentation.')
    p.add_argument('--only-main',action='store_true');a=p.parse_args()
    out=a.out or ROOT/'results/paper-ready-figure/fig4/candidates'/('best_joint_positions_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    if a.snapshot_dir is not None and not (out/'source/snapshot.json').exists():
        shutil.copytree(a.snapshot_dir/'source',out/'source')
        snapshot=read(out/'source/snapshot.json');snapshot['main_fraction']=MAIN
        snapshot['parent_snapshot']=dict(path=str((a.snapshot_dir/'source/snapshot.json').resolve()),sha256=sha(a.snapshot_dir/'source/snapshot.json'))
        snapshot['display_revision']='2026-09-29: top 20 percent; full-height colorbar; no header; current paper-ready Figure 4 typography.'
        write(out/'source/snapshot.json',snapshot)
        table=out/'source/ranked_candidates.csv'
        with table.open() as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;ranked=list(reader)
        chosen=set(snapshot['selections'][str(MAIN)]['candidate_ids'])
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in ranked:row['selected_main']=row['candidate'] in chosen;writer.writerow(row)
    for directory in ('source/scores','source/selected_audits','figures','qa'):(out/directory).mkdir(parents=True,exist_ok=True)
    data=freeze(out);stats=sensitivity(out,data)
    current=read(ROOT/'results/paper-ready-figure/fig4/current_version.json')
    font_registry=ROOT/current['registry'];reference=read(font_registry);fonts=reference['font_pt']
    display=dict(show_header=False,colorbar_match_xy_height=True,font_family='DejaVu Sans',
        font_pt={**fonts,'xy_label':reference['checks']['C_axis_label_font_pt']},
        typography_reference=str(font_registry),typography_reference_sha256=sha(font_registry))
    progress.plt.rcParams.update({'font.family':'DejaVu Sans','font.size':fonts['tick'],'axes.labelsize':fonts['label'],'xtick.labelsize':fonts['tick'],'ytick.labelsize':fonts['tick'],
        'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
    panels=[];display_fractions=(MAIN,) if a.only_main else FRACTIONS
    for fraction in display_fractions:
        selected=data['selections'][str(fraction)];ids=set(selected['candidate_ids']);rows=[r for r in data['records'] if r['candidate'] in ids]
        percent=round(fraction*100);stem='core-position-best-joint' if fraction==MAIN else f'core-position-top{percent:02d}-sensitivity'
        selection=dict(records=rows,stem=stem,title=rf'Best {percent}% by $J_{{\mathrm{{joint}}}}$',phase='Low-loss candidate positions',
            summary=f"n = {len(rows)} / {data['pool_scorable']} scored configurations",xy_tick_pad={'x':-4,'y':0},xy_label_pad={'x':2,'y':2},display=display)
        panels.append(progress.position_figure(out,data,selection=selection))
    main_selection=data['selections'][str(MAIN)]
    write(out/'figure_registry.json',dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',asof=data['asof'],main_fraction=MAIN,
        main_selection=main_selection,pool_completed=data['pool_completed'],pool_scorable=data['pool_scorable'],historical_completed=data['historical_completed'],
        new_completed=data['new_completed'],new_scorable=data['new_scorable'],density_panels=panels,
        scientific_status='DESCRIPTIVE_LOW_LOSS_REGIONS; CORE_B_NOT_UNIQUELY_LOCALIZED; EXPANDED_SEARCH_NOT_COMPLETE',
        sources={str(p):sha(p) for p in (Path(__file__),Path(progress.__file__),Path(progress.style.__file__),out/'source/snapshot.json')},
        outputs={p.name:sha(p) for p in sorted((out/'figures').glob('*')) if p.suffix in ('.png','.pdf','.svg')}))
    entries=[]
    for panel,fraction in zip(panels,display_fractions):
        s=data['selections'][str(fraction)]
        entries.append(f"### {panel['stem']}.png\n按同一个冻结J_joint从{data['pool_scorable']}个可评分候选中取最低{fraction:.0%}，共{s['count']}个配置（含并列阈值），阈值J_joint≤{s['cutoff_J_joint']:.6f}。原XY坐标、电极几何、0.35 mm高斯核和近XY正交视角保持一致，单色紫色高度／竖向色条表示条件位置密度；本子集含{s['historical']}个历史、{s['new']}个新增配置。色条与投影XY平面等高，图内不加标题或样本量说明，字体按当前paper-ready Fig4配置统一。\n**关注点**：直接展示低loss候选的位置；Core B的分散和多峰如实保留，历史边界及不均匀搜索会影响集中程度；不是独立重复拟合、参数后验或唯一固定位置证据。\n")
    (out/'figures/README.md').write_text('\n'.join(entries))
    main_stats=[s for s in stats if s['fraction']==MAIN]
    text=(f"# 低J_joint候选的位置分布\n\n快照：{data['asof']}。统一评分池为{data['pool_completed']}个完整配置：{data['historical_completed']}个兼容历史轨迹＋{data['new_completed']}个本轮新增轨迹；其中{data['pool_scorable']}个可评分。历史提案曾使用不同目标，但此次选择全部按相同的冻结J_joint排序，不按XY筛选。\n\n"
        f"用户指定主图取前{MAIN:.0%}，即{main_selection['count']}个（J_joint≤{main_selection['cutoff_J_joint']:.6f}）；5%、10%和20%的数值比较保留为筛选数量敏感性检查。主图中{main_selection['historical']}个来自历史、{main_selection['new']}个来自新扩围；不把旧候选冒称新扩围优化结果。X/Y刻度标签间距分别调至−4/0 pt，轴标题间距均2 pt，视角、物理坐标及核宽保持不变。\n\n")
    for s in main_stats:text+=f"Core {s['core']}：平均位置{np.round(s['mean_xy_mm'],3).tolist()} mm，XY跨度{np.round(s['coordinate_range_mm'],3).tolist()} mm，距子集均值的90%半径{s['radial_90pct_mm']:.3f} mm。这些均值／分散量是所选样本的描述，不是置信区间。\n\n"
    text+=f"排版沿用当前Fig4注册字号：XY轴标题{display['font_pt']['xy_label']:g} pt，刻度{fonts['tick']:g} pt，Core A/B图例{fonts['legend']:g} pt，色条标题{fonts['label']:g} pt。竖向色条上下端精确对齐投影XY平面；图内标题及样本量说明移除，筛选数量保留于本说明和注册表。\n\n"
    text+='目前支持低loss候选偏向较小的优势区域；Core A比Core B集中，Core B不宜描述为唯一固定点。最佳三条历史联合搜索的B位置都在旧搜索盒边界，同一拓扑／噪声、自适应候选依赖、历史采样密集和旧位置约束都可能贡献集中。扩大范围的自适应优化尚未完成，不能据此确认跨优化重复或真实解剖位置稳定。用户此次更正改变展示的条件选择规则，不修改后台256个配置的物理、目标或运行预算。\n'
    (out/'scientific_note.md').write_text(text)
    print(out,flush=True)

if __name__=='__main__':main()

"""Audit finite joint search, show tested XY geometry and SCL order residuals.

Joint-candidate colours describe the full parameter condition, not a conditional
XY response surface. Existing trajectories and frozen diagnostic labels only.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path('/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915')
DEST = ROOT / 'key_results/geometry_scl_review_20260916'
CASES = [('g2_b03_p01', '上一展示工作点', '#8B6A9B', '^'),
         ('dense_axis3_12', '本轮整体最佳', '#0072B2', 'o'),
         ('lf_bo02_01', '新增联合点最佳', '#D55E00', 's')]


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1, text.get_text()
    fig.savefig(DEST/'figures'/f'{name}.png', dpi=400)
    fig.savefig(DEST/'figures'/f'{name}.svg')
    plt.close(fig)


def main():
    (DEST/'figures').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
        'font.size':9, 'axes.labelsize':9.5, 'axes.titlesize':10, 'xtick.labelsize':8.5,
        'ytick.labelsize':8.5, 'legend.fontsize':8.5, 'legend.frameon':False,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.facecolor':'white'})
    scores = {p.stem:read(p) for p in (ROOT/'scores').glob('*.json')}
    candidates = {p.stem:read(p) for p in (ROOT/'candidates').glob('*.json')}
    diagnostic = read(ROOT/'analysis/diagnostic_observables.json')
    proposals = [read(p) for p in sorted((ROOT/'proposals').glob('batch_*.json'))]
    history=[]
    for batch in proposals:
        seen = min((r for r in batch['training_snapshot'] if r['J'] is not None),key=lambda r:r['J'])
        new = min((scores[c] for c in batch['ids']),key=lambda r:r['J'])
        history.append(dict(batch=batch['batch'],observed_count=len(batch['training_snapshot']),
            incumbent_before=seen['candidate'],J_before=seen['J'],new_best=new['candidate'],
            J_new_best=new['J'],retained_J=min(seen['J'],new['J']),halfwidth=batch['halfwidth'],
            saw_final_best=any(r['candidate']=='dense_axis3_12' for r in batch['training_snapshot'])))
    write(DEST/'optimizer_history.json',history)
    stat=[];pairs=[]
    for cid, label, _, _ in CASES + [('g2_b01_p01','上一版有标签目标首名','#666666','D')]:
        for mode in ['ALL','TA','TB']:
            q=diagnostic['runs'][cid][mode];p=diagnostic['patient'][mode]
            for shaft in ['SCL','ICL']:
                keys=[key for key,v in q['pairs'].items() if v['shaft']==shaft
                      and v['order_probability'] is not None and p['pairs'][key]['order_probability'] is not None]
                errors=[abs(q['pairs'][k]['order_probability']-p['pairs'][k]['order_probability']) for k in keys]
                stat.append(dict(candidate=cid,label=label,mode=mode,shaft=shaft,N=q['N'],
                    available_pairs=len(keys),order_probability_MAE=float(np.mean(errors)),
                    participation_MAE=float(np.mean([abs(v['participation']-p['contacts'][n]['participation'])
                        for n,v in q['contacts'].items() if n.startswith(shaft)]))))
                for key in keys:
                    if shaft=='SCL':
                        v=q['pairs'][key];pv=p['pairs'][key]
                        pairs.append(dict(candidate=cid,mode=mode,pair=key,n_model=v['n'],n_patient=pv['n'],
                            patient_probability=pv['order_probability'],model_probability=v['order_probability'],
                            patient_median_ms=pv['median'],model_median_ms=v['median']))
    for name, rows in [('shaft_metrics',stat),('scl_pairs',pairs)]:
        with (DEST/f'{name}.csv').open('w') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    # XY maps retain the true 20-mm electrode geometry. The same candidate IDs
    # and full-condition scores appear in both panels; no interpolated field.
    ids=[cid for b in proposals for cid in b['ids']]
    assert len(ids)==32
    js=np.array([scores[c]['J'] for c in ids])
    source=Path(scores['dense_axis3_12']['source']).with_suffix('.npz')
    with np.load(source) as z:
        contact_xy=z['contact_xy_mm'];names=z['contact_names'].tolist()
    anchor=candidates[read(ROOT/'plan.json')['reference_id']]
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,96/25.4))
    fig.subplots_adjust(left=.08,right=.86,bottom=.16,top=.79,wspace=.20)
    geometry=[]
    for core,ax in enumerate(axes):
        for shaft,col in [('SCL','#4C9CAB'),('ICL','#C7873C')]:
            ix=[i for i,n in enumerate(names) if n.startswith(shaft)]
            xy=contact_xy[ix]
            ax.plot(xy[:,0],xy[:,1],'-o',color=col,markersize=3,linewidth=.7,alpha=.85,zorder=1)
            for i in [ix[0],ix[-1]]:
                x,y=contact_xy[i];ax.text(x,y+(-1.05 if names[i]=='ICL1' else .55),names[i],fontsize=7.5,color=col,ha='center')
        point=np.array([candidates[c]['centers_mm'][core] for c in ids])
        sc=ax.scatter(point[:,0],point[:,1],c=js,cmap='viridis',norm=LogNorm(vmin=1.38,vmax=6.31),
                      s=25,edgecolor='white',linewidth=.35,zorder=3)
        xy0=np.array(anchor['centers_mm'][core]);ax.add_patch(Circle(xy0,anchor['radii_mm'][core],
                     fill=False,color='#BBBBBB',linestyle=':',linewidth=.8,zorder=1))
        ax.scatter(*xy0,s=40,facecolor='none',edgecolor='#555555',marker='o',linewidth=.9,zorder=5)
        for cid,marker,size,color in [('dense_axis3_12','*',105,'#111111'),('lf_bo02_01','D',40,'#D55E00')]:
            xy=np.array(candidates[cid]['centers_mm'][core])
            ax.scatter(*xy,s=size,marker=marker,color=color,edgecolor='white',linewidth=.65,zorder=6)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xticks=[0,10,20],yticks=[0,10,20],
               xlabel='x (mm)',title=['左核 XY','右核 XY'][core])
        if core==0:ax.set_ylabel('y (mm)')
    cb=fig.colorbar(sc,cax=fig.add_axes([.895,.23,.018,.48]),ticks=[1.4,2,3,6])
    cb.ax.set_yticklabels(['1.4','2','3','6']);cb.ax.minorticks_off();cb.set_label('完整参数组合的损失 ↓',labelpad=7)
    fig.legend(handles=[Line2D([],[],marker='o',markerfacecolor='none',color='#555555',linestyle='none',label='参考核'),
                        Line2D([],[],marker='*',color='#111111',linestyle='none',markersize=8,label='整体最佳'),
                        Line2D([],[],marker='D',color='#D55E00',linestyle='none',label='新增联合点最佳')],
               loc='upper center',bbox_to_anchor=(.47,.985),ncol=3)
    save(fig,'joint_core_xy_positions')
    for cid in ids:
        c=candidates[cid];xy=np.array(c['centers_mm']);delta=xy[1]-xy[0]
        geometry.append(dict(candidate=cid,left_x=xy[0,0],left_y=xy[0,1],right_x=xy[1,0],right_y=xy[1,1],
            separation_mm=float(np.linalg.norm(delta)),core_axis_deg=float(np.rad2deg(np.arctan2(delta[1],delta[0]))),
            EE_out=c['parameters']['EE_core_to_out_scale'],EE_angle=c['parameters']['EE_angle_offset_deg'],J=scores[cid]['J']))
    with (DEST/'joint_geometry.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(geometry[0]));w.writeheader();w.writerows(geometry)

    # Conditional labels are read only for post-nomination diagnostic display.
    keys=[k for k,v in diagnostic['patient']['ALL']['pairs'].items() if v['shaft']=='SCL']
    assert len(keys)==6
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,88/25.4),sharey=True)
    fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.78,wspace=.15)
    for ax,mode in zip(axes,['TA','TB']):
        ref=diagnostic['patient'][mode]
        ax.plot(range(6),[ref['pairs'][k]['order_probability'] for k in keys],'-o',
                color='#111111',linewidth=1.6,markersize=4,label='患者')
        for cid,label,color,marker in CASES:
            q=diagnostic['runs'][cid][mode]
            ax.plot(range(6),[q['pairs'][k]['order_probability'] for k in keys],'-',marker=marker,
                    color=color,markersize=3.5,linewidth=1.05,label=label)
        ax.axhline(.5,color='#AAAAAA',linestyle=':',linewidth=.7)
        ax.set(ylim=(0,1.035),yticks=[0,.25,.5,.75,1],xticks=range(6),
               xticklabels=[k.replace('SCL','') for k in keys],title=mode,
               xlabel='SCL 触点对（前者 → 后者）')
        ax.tick_params(axis='x',pad=4)
    axes[0].set_ylabel('后者质心晚于前者的概率',labelpad=7)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.99),ncol=2,
               columnspacing=1.5,handlelength=1.8)
    save(fig,'scl_conditional_order')
    write(DEST/'figure_metadata.json',dict(
        claim='Joint search retained the incumbent, while pooled fit can mask conditional SCL error.',
        geometry='32 real joint candidates, shown on the unchanged SEEG geometry. Colours are full-condition loss, not isolated XY effects. No interpolation.',
        SCL='Six within-SCL pair probabilities, conditional on both contacts participating. Ties count 0.5.',
        topology=2511,noise=847401,patient_fit_N=19770,labels_used_for_optimization=False,
        case_names={cid:label for cid,label,_,_ in CASES},
        extra_comparator='g2_b01_p01 is the previous label-conditioned objective winner; included in CSV, not confused with the previous displayed point.',
        uncertainty='Descriptive single-trajectory summaries; events are not independent network replicates.',
        separate_timing_loss='training_timing_decomposition.json; pooled training feature distance differs from conditional probability MAE.',
        scientific_verdict='PARTIAL_RECOVERY; TB SCL order not recovered.',agent_visual_review='PENDING',human_visual_review='PENDING'))
    (DEST/'figures/README.md').write_text(
        '### joint_core_xy_positions.png\n两图分别显示32个真实联合候选的左右核XY，使用相同20×20 mm SEEG布局。颜色为该完整六参数组合的无标签联合损失，圆圈/星形/菱形分别标记参考核、全池最佳和新增联合点最佳。其他核、EE权重和方向也会随候选变化，因此不将该图称为固定其他参数的二维响应面；未测区域没有插值。**关注点**：实际已搜索的核位置范围及对应组合分数，不从点颜色单独归因位置效应。\n\n'
        '### scl_conditional_order.png\n比较患者、上一展示工作点、本轮整体最佳和新增联合点最佳的SCL六个触点对先后概率，分别显示TA/TB。概率仅使用同一事件中共同参与的两触点，实际每对事件数保存在scl_pairs.csv；0.5表示两种顺序等概率，不代表零时差。各曲线为固定拓扑2511/噪声847401，TA/TB仅用于诊断，未反馈本轮优化。**关注点**：总体分数更优的候选，是否反而使TB内SCL顺序偏向与患者不同的一侧。\n',encoding='utf-8')
    print(json.dumps(dict(history=history,figures=[str(p) for p in (DEST/'figures').glob('*.png')]),ensure_ascii=False))


if __name__=='__main__':
    main()

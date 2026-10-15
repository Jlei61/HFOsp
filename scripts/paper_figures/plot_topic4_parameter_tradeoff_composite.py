"""Compact parameter-response composite and 3-D observed candidate trade-offs.

Existing response statistics and five Bayesian nominees are reused unchanged.
No re-ranking, smoothing, additional simulation or inferred response surface.
"""
from pathlib import Path
import csv
import json
import hashlib
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE=Path('/data/hfosp/topic4_sef_hfo')
OLD=BASE/'label_free_dense_parameter_search_20260915'
ROUND=BASE/'geometry_axis_joint_distribution_20260917'
SOURCE=ROUND/'key_results/data_driven_metric_tradeoffs'
OUT=SOURCE/'compact_composite'
FIG=OUT/'figures'
KEYS=['rank_error','within_rod_order_error','participation_error']
ROLES=['平均顺序最佳','杆内顺序最佳','参与概率最佳','原损失最低','联合损失最低']
BLUE,ORANGE='#0072B2','#D55E00'
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
    'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,
    'ytick.labelsize':6,'legend.fontsize':6.5,'legend.frameon':False,
    'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    responses=OLD/'key_results/overall_acceptance_20260916/source_data.csv'
    selected=read(SOURCE/'selection.json')['selected']
    assert len(selected)==5
    rows=list(csv.DictReader(open(SOURCE/'all_bayesian_candidates.csv')))
    assert len(rows)==56
    for r in rows:
        for key in KEYS:r[key]=float(r[key])
        r['observed_pareto']=r['observed_pareto']=='True'
    byid={r['candidate']:r for r in rows}
    for r in selected:
        np.testing.assert_array_equal([r[k] for k in KEYS],[byid[r['candidate']][k] for k in KEYS])
    curves=[r for r in csv.DictReader(open(responses)) if r['observable'] in ['rank','local_order']]
    assert len(curves)==156 and all(r['mode']=='ALL' for r in curves)
    current={r['candidate']:r for r in csv.DictReader(open(ROUND/'analysis/observables.csv')) if r['mode']=='ALL'}
    for r in curves:
        key={'rank':'rank_error','local_order':'within_rod_order_error'}[r['observable']]
        assert abs(float(r['value'])-float(current[r['candidate']][key]))<1e-12
    write(OUT/'figure_contract.json',dict(
        question='How do measured parameter responses relate to the observed trade-offs among Bayesian candidates?',
        claim='The earlier one-parameter responses and the later joint parameter combinations answer complementary questions.',
        archetype='quantitative grid',backend='Python matplotlib',
        main_panels='Left: unchanged six one-parameter curves, two observables; right: five Bayesian workpoints across three observables.',
        scatter='All 56 real Bayesian proposals on three raw observable axes; same five workpoint colours and markers.',
        reuse='Exact statistics reuse and layout adaptation; no raster panel compositing.',
        reference='g1_axis3_plus; centers (3.44921432,12.12890135),(16.47920304,4.71551153); EEout=1.25; axis=0 deg',
        lines='Left: guide to measured one-parameter points. Right: connects categorical observables of one combination, not a parameter path.',
        units='Raw absolute errors; no cross-metric aggregation. Fixed topology2511/noise847401; 60 s per condition.',
        uncertainty='Single fixed seed, no between-seed error bars or inferential claims.',
        output=dict(composite_mm=[183,108],scatter_mm=[183,104],formats=['png','svg'],dpi=400,min_font_pt=6),
        annotations='Only panel labels, short titles, axes and legends; definitions and full parameter table stay in source files.',
        exclusions='No previously plotted data removed: 156 response values and 56 Bayesian candidates retained; old static two-dimensional pairwise panels superseded by three-dimensional scatter.',
        source_hashes={str(p):sha(p) for p in [responses,SOURCE/'selection.json',SOURCE/'all_bayesian_candidates.csv',Path(__file__)]}))
    return curves,rows,selected


def workpoint_handles(selected):
    return [Line2D([],[],color=r['color'],marker=r['marker'],lw=1.15,ms=4,
                   label=f"{r['display_id']}  {ROLES[i]}") for i,r in enumerate(selected)]


def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];sizes=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer);sizes.append(t.get_fontsize())
            if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:outside.append(t.get_text())
    assert not outside,outside
    assert min(sizes)>=6
    fig.savefig(FIG/f'{name}.png',dpi=400);fig.savefig(FIG/f'{name}.svg');plt.close(fig)
    return dict(figure=name,min_font_pt=min(sizes),text_outside_canvas=outside)


def composite(curves,selected):
    fig=plt.figure(figsize=(183/25.4,108/25.4))
    grid=fig.add_gridspec(2,3,left=.078,right=.625,bottom=.135,top=.81,wspace=.35,hspace=.60)
    settings=[(0,'左核 x','位移 (mm)',[-1.5,0,1.5]),
              (1,'左核 y','位移 (mm)',[-1.5,0,1.5]),
              (4,'核内→核外 EE','权重倍率',[1,1.25,1.5]),
              (2,'右核 x','位移 (mm)',[-1.5,0,1.5]),
              (3,'右核 y','位移 (mm)',[-2,-1,0]),
              (5,'全局 EE 方向','轴偏移 (°)',[-20,0,20])]
    for n,(j,title,xlabel,ticks) in enumerate(settings):
        ax=fig.add_subplot(grid[n//3,n%3]);twin=ax.twinx()
        for target,key,color,marker in [(ax,'rank',BLUE,'o'),(twin,'local_order',ORANGE,'s')]:
            line=sorted([r for r in curves if int(r['axis'])==j and r['observable']==key],key=lambda r:float(r['parameter_value']))
            x=np.array([float(r['parameter_value']) for r in line]);y=np.array([float(r['value']) for r in line])
            assert len(x)==13 and np.all(np.diff(x)>0) and np.all((y>=0)&(y<.42))
            target.plot(x,y,color=color,marker=marker,ms=2.4,lw=.95,mew=.3)
            target.set(ylim=(0,.42),yticks=[0,.2,.4]);target.tick_params(axis='y',colors=color,length=2.5,pad=2)
        ax.tick_params(axis='y',labelleft=n%3==0)
        twin.tick_params(axis='y',labelright=n%3==2)
        ax.spines['left'].set_color(BLUE);ax.spines['right'].set_visible(False)
        twin.spines['left'].set_visible(False);twin.spines['right'].set_visible(True)
        twin.spines['right'].set_color(ORANGE);twin.spines['bottom'].set_visible(False)
        ref=1.25 if j==4 else 0
        ax.axvline(ref,color='#AAAAAA',ls=':',lw=.65,zorder=0)
        ax.set(title=title,xlabel=xlabel,xticks=ticks);ax.margins(x=.06)
        ax.tick_params(axis='x',length=2.5,pad=2)
    fig.text(.021,.925,'a',fontsize=9,weight='bold')
    fig.text(.077,.925,'参考工作点周围的单参数响应',fontsize=8)
    fig.text(.025,.482,'误差 ↓',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=7)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ms=3,lw=1,label='平均 rank'),
                        Line2D([],[],color=ORANGE,marker='s',ms=3,lw=1,label='杆内顺序'),
                        Line2D([],[],color='#AAAAAA',ls=':',lw=.65,label='参考值')],
               loc='upper left',bbox_to_anchor=(.071,.896),ncol=3,columnspacing=.85,
               handlelength=1.5,handletextpad=.35,borderaxespad=0)
    right=fig.add_axes([.747,.370,.235,.44])
    for r in selected:
        right.plot(range(3),[r[k] for k in KEYS],color=r['color'],marker=r['marker'],lw=1.15,ms=3.5)
    right.set(xlim=(-.17,2.17),ylim=(0,.46),yticks=[0,.1,.2,.3,.4],
              xticks=range(3),xticklabels=['平均\nrank','杆内\n顺序','触点\n参与'],ylabel='误差 ↓')
    right.grid(axis='y',color='#EAEAEA',lw=.5);right.set_axisbelow(True)
    fig.text(.69,.925,'b',fontsize=9,weight='bold')
    fig.text(.746,.925,'优化工作点的取舍',fontsize=8)
    fig.legend(handles=workpoint_handles(selected),loc='upper left',bbox_to_anchor=(.739,.239),
               ncol=1,handlelength=1.6,handletextpad=.5,labelspacing=.5,borderaxespad=0,fontsize=6.5)
    return save(fig,'parameter_response_with_workpoints')


def scatter3d(rows,selected):
    fig=plt.figure(figsize=(183/25.4,104/25.4))
    ax=fig.add_axes([.015,.105,.715,.835],projection='3d',computed_zorder=False)
    xyz=np.array([[r[k] for k in KEYS] for r in rows])
    limits=np.ceil(xyz.max(axis=0)*1.06/.05)*.05
    ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1))
    ax.view_init(elev=24,azim=-53)
    ax.scatter(*xyz.T,s=13,c='#BFC5CD',alpha=.72,edgecolors='none',depthshade=False,zorder=1)
    p=np.array([[r[k] for k in KEYS] for r in rows if r['observed_pareto']])
    ax.scatter(*p.T,s=28,facecolors='none',edgecolors='#8B9199',lw=.65,depthshade=False,zorder=2)
    for r in selected:
        ax.scatter(*[r[k] for k in KEYS],s=42,c=r['color'],marker=r['marker'],
                   edgecolors='white',lw=.5,depthshade=False,zorder=5)
    for axis,lim,label in zip([ax.xaxis,ax.yaxis,ax.zaxis],limits,
             ['平均 rank 误差 ↓','杆内顺序误差 ↓','触点参与误差 ↓']):
        axis.set_major_formatter(FormatStrFormatter('%.1f'))
        axis.set_ticks(np.arange(0,lim+.01,.1))
        axis.set_pane_color((1,1,1,0));axis._axinfo['grid'].update(color=(.88,.89,.91,1),linewidth=.5)
        axis.line.set_color('#999999');axis.line.set_linewidth(.6)
        axis.set_label_text(label,fontsize=7)
        axis.labelpad=4
    ax.set(xlim=(0,limits[0]),ylim=(0,limits[1]),zlim=(0,limits[2]))
    ax.tick_params(pad=0,labelsize=6)
    handles=workpoint_handles(selected)+[
        Line2D([],[],color='#BFC5CD',marker='o',ms=3.5,linestyle='none',label='全部候选（56）'),
        Line2D([],[],color='#8B9199',marker='o',mfc='none',ms=4,linestyle='none',label='非支配组合（13）')]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.747,.56),
               borderaxespad=0,handlelength=1.5,handletextpad=.5,labelspacing=.85,fontsize=7)
    return save(fig,'three_observable_tradeoff_3d')


def main():
    curves,rows,selected=load()
    plots=[composite(curves,selected),scatter3d(rows,selected)]
    import xml.etree.ElementTree as ET
    from PIL import Image
    for f in FIG.glob('*.png'):
        with Image.open(f) as im:im.load()
    for f in FIG.glob('*.svg'):
        tree=ET.parse(f)
        assert any(t.tag.endswith('text') for t in tree.getroot().iter())
    write(OUT/'qa.json',dict(figures=plots,all_outputs_decoded=True,response_values_retained=156,
        bayesian_candidates_retained=56,selected_workpoints_unchanged=5,new_simulations=0,
        response_parity_with_current_scores=True,agent_visual_review='PENDING',human_visual_review='PENDING'))
    (FIG/'README.md').write_text(
        '### parameter_response_with_workpoints.png\n'
        '左侧复用此前参考工作点附近六个标量各13点的单参数响应，蓝色为平均rank误差、橙色为杆内先后概率误差，两个Y轴均保持0–0.42。右侧放五个无标签贝叶斯提案工作点的三项观测，五种颜色含义与3D图一致；完整参数保留于上级 selected_candidates.csv，不再占用图内空间。\n'
        '**关注点**：左侧为固定其他参数的历史扫描，右侧为不同参数组合，不能将二者混作同一最优工作点的局部导数；均为固定拓扑2511/噪声847401的60秒结果。\n\n'
        '### three_observable_tradeoff_3d.png\n'
        '三个坐标分别为平均rank、杆内顺序、触点参与概率误差，展示全部56个实际无标签贝叶斯提案及相同五个工作点。灰圈为三项观测的经验非支配组合；使用正交投影、固定原始数值尺度，不插值拟合表面。\n'
        '**关注点**：不同工作点在三个观测上的取舍，不意味着连续参数空间不存在更好的共同解；图内只保留轴名及简短图例。\n')
    print(json.dumps(dict(out=str(OUT),figures=plots),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

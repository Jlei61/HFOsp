"""Compact XY/connection response, all Bayesian profiles and model-only examples."""
from pathlib import Path
import csv, json, hashlib, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scripts import report_topic4_geometry_axis_joint_search as report
from scripts.analyze_topic4_core_connectivity_search import load_unit
from scripts.render_topic4_shape_output_gifs import field_tile

BASE=Path('/data/hfosp/topic4_sef_hfo')
ROUND=BASE/'geometry_axis_joint_distribution_20260917'
OLD=BASE/'label_free_dense_parameter_search_20260915'
XY=BASE/'xy_block_joint_observation_pilot_20260916'
SOURCE=ROUND/'key_results/data_driven_metric_tradeoffs'
OUT=SOURCE/'compact_abcd';FIG=OUT/'figures'
KEYS=['rank_error','within_rod_order_error','participation_error']
TIME_COLORS=plt.get_cmap('Blues')(np.linspace(.22,.90,7))
TIME_CMAP=ListedColormap(TIME_COLORS)
TIME_NORM=BoundaryNorm(np.arange(.5,8),7)
BLUE,ORANGE='#0072B2','#D55E00'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
 'font.size':6,'axes.labelsize':6,'axes.titlesize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,
 'legend.fontsize':6,'legend.frameon':False,'axes.linewidth':.55,
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
 'savefig.facecolor':'white'})


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(SOURCE/'all_bayesian_candidates.csv')))
    history=read(SOURCE/'waveforms_trajectory/batch_history.json')
    assert len(rows)==56
    for r in rows:
        for k in KEYS:r[k]=float(r[k])
        r['batch']=history['point_to_batch'][r['candidate']]
    assert [sum(r['batch']==i for r in rows) for i in range(1,8)]==[8]*7
    choice=read(SOURCE/'selection.json')['selected']
    # Preserve the exact two requested workpoints and previously shown events.
    cases=[dict(next(r for r in choice if r['display_id']==old),panel=panel,old_display_id=old)
           for old,panel in [('P1','C'),('P3','D')]]
    plan=read(XY/'plan.json');ref=read(XY/'candidates'/f"{plan['reference_id']}.json")
    table={r['candidate']:r for r in csv.DictReader(open(XY/'analysis/observables.csv')) if r['mode']=='ALL'}
    grids=[];all_ids=set()
    for grid in plan['grids']:
        core=grid['core'];values=[]
        for iy,row in enumerate(grid['ids']):
            for ix,cid in enumerate(row):
                c=read(XY/'candidates'/f'{cid}.json');xy=np.array(c['centers_mm'][core])
                expected=np.array(ref['centers_mm'][core])+[grid['x_offsets_mm'][ix],grid['y_offsets_mm'][iy]]
                np.testing.assert_allclose(xy,expected,atol=1e-12,rtol=0)
                assert c['centers_mm'][1-core]==ref['centers_mm'][1-core]
                assert c['parameters']==ref['parameters']
                values.append(dict(candidate=cid,core=core,x=float(xy[0]),y=float(xy[1]),
                    **{k:float(table[cid][k]) for k in KEYS}))
                all_ids.add(cid)
        assert len(values)==25;grids.append(values)
    assert len(all_ids)==49
    response=list(csv.DictReader(open(OLD/'key_results/overall_acceptance_20260916/source_data.csv')))
    response=[r for r in response if int(r['axis']) in [4,5] and r['observable'] in ['rank','local_order']]
    assert len(response)==52 and all(r['mode']=='ALL' for r in response)
    cur={r['candidate']:r for r in csv.DictReader(open(ROUND/'analysis/observables.csv')) if r['mode']=='ALL'}
    for r in response:
        key={'rank':'rank_error','local_order':'within_rod_order_error'}[r['observable']]
        assert abs(float(r['value'])-float(cur[r['candidate']][key]))<1e-12
    write(OUT/'figure_contract.json',dict(
        question='Where do XY and connectivity changes affect observed propagation, and what do optimization trade-offs look like natively?',
        archetype='image plate + quant',backend='Python matplotlib',
        panels=dict(A='Two genuine conditional XY grids, each split-cell encodes rank and within-rod errors; retain outgoing EE and direction curves.',
                    B='All 56 real label-free Bayesian proposals; seven shared Blues batch colours, darker means later.',
                    C='Former P1, model-only TA/TB first primary events and their native fields.',
                    D='Former P3, same event identities and display contract as prior waveforms.'),
        geometry='Two 5x5 measured grids; move one core jointly in XY, hold other core and parameters fixed. Split cells are not interpolation.',
        split_cells=dict(lower_right='mean normalized rank error, Blues',upper_left='within-rod order probability error, Oranges',scale=[0,.30]),
        background_difference='XY grid reference dense_axis3_12 has right y=5.46551153 mm; EE/direction curves retain g1_axis3_plus reference with right y=4.71551153 mm. Both use EEout1.25, angle0; not one common local response surface.',
        statistics='Patient FIT target19770; one topology2511/noise847401, 60 s each. ALL observables, no TA/TB optimization labels.',
        excluded='A omits the four old separate-coordinate curves, replaced at user request by full XY grids. B omits none of the 56 proposals. C/D remove patient and model-mean panels, retain previously selected single events and native snapshots.',
        native='All E activity, 2 ms frames, +0/+30/+60 ms relative earliest participating centroid; shared vmax75.',
        display='Fixed SCL9..6 then ICL11..1, fixed -125..200 ms axis, no per-contact realignment. Grey means missing or outside event window.',
        output=dict(width_mm=183,height_mm=195,dpi=400,formats=['png','svg'],min_font_pt=6),
        source_files={str(p):sha(p) for p in [SOURCE/'all_bayesian_candidates.csv',SOURCE/'waveforms_trajectory/batch_history.json',XY/'plan.json',XY/'analysis/observables.csv',OLD/'key_results/overall_acceptance_20260916/source_data.csv',Path(__file__)]}))
    for name,data in [('xy_grid_source.csv',[r for g in grids for r in g]),('all_workpoints_source.csv',rows),('connection_response_source.csv',response)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return rows,cases,grids,ref,response


def xy_panels(fig,grids,ref):
    norm=Normalize(0,.30)
    for k,values in enumerate(grids):
        left=[.082,.345][k];width=.18
        ax=fig.add_axes([left,.765,width,width*183/195])
        lower=[];upper=[]
        for r in values:
            x,y=r['x'],r['y'];lo=(x-.125,y-.125);br=(x+.125,y-.125);tr=(x+.125,y+.125);tl=(x-.125,y+.125)
            lower.append([lo,br,tr]);upper.append([lo,tr,tl])
        for polygons,key,cmap in [(lower,'rank_error','Blues'),(upper,'within_rod_order_error','Oranges')]:
            artist=PolyCollection(polygons,array=np.array([r[key] for r in values]),cmap=cmap,norm=norm,
                                  edgecolors='white',linewidths=.22)
            ax.add_collection(artist)
        center=np.array(ref['centers_mm'][k]);ax.scatter(*center,marker='*',s=30,c='white',edgecolor='#222222',lw=.65,zorder=4)
        ax.set(xlim=(center[0]-.625,center[0]+.625),ylim=(center[1]-.625,center[1]+.625),aspect='equal',
               xticks=center[0]+np.array([-.5,0,.5]),yticks=center[1]+np.array([-.5,0,.5]),
               xlabel='x (mm)',ylabel='y (mm)',title=['Core A','Core B'][k])
        ax.tick_params(length=2,pad=1.5);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for x,cmap,label in [(.082,'Blues','平均 rank 误差 ↓'),(.345,'Oranges','杆内顺序误差 ↓')]:
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([x,.704,.18,.009]),
                        orientation='horizontal',ticks=[0,.15,.30])
        bar.ax.tick_params(length=1.5,pad=1,labelsize=6);bar.outline.set_visible(False)
        bar.ax.set_title(label,fontsize=6,pad=2)


def connections(fig,response):
    for j,x,title,xlabel,ticks in [(4,.082,'核内→核外 EE','权重倍率',[1,1.25,1.5]),
                                   (5,.345,'全局 EE 方向','轴偏移 (°)',[-20,0,20])]:
        ax=fig.add_axes([x,.553,.18,.100]);twin=ax.twinx()
        for target,key,color,marker in [(ax,'rank',BLUE,'o'),(twin,'local_order',ORANGE,'s')]:
            rr=sorted([r for r in response if int(r['axis'])==j and r['observable']==key],key=lambda r:float(r['parameter_value']))
            xx=[float(r['parameter_value']) for r in rr];yy=[float(r['value']) for r in rr]
            assert len(xx)==13 and np.all(np.diff(xx)>0)
            target.plot(xx,yy,color=color,marker=marker,ms=1.8,lw=.9,mew=.3)
            target.set(ylim=(0,.42),yticks=[0,.2,.4]);target.tick_params(axis='y',colors=color,pad=2,length=2)
        ax.spines['left'].set_color(BLUE);twin.spines['right'].set_visible(True);twin.spines['right'].set_color(ORANGE)
        twin.spines['left'].set_visible(False);twin.spines['bottom'].set_visible(False)
        ax.axvline(1.25 if j==4 else 0,color='#999999',ls=':',lw=.6)
        ax.set(title=title,xlabel=xlabel,xticks=ticks);ax.tick_params(axis='x',length=2,pad=2);ax.margins(x=.06)
        if j==4:ax.set_ylabel('误差 ↓')


def profiles(fig,rows,cases):
    ax=fig.add_axes([.684,.625,.282,.31])
    for row in sorted(rows,key=lambda r:(r['batch'],r['candidate'])):
        ax.plot(range(3),[row[k] for k in KEYS],color=TIME_COLORS[row['batch']-1],lw=.65,
                marker='o',ms=1.8,mew=0,alpha=1,zorder=1+row['batch']/10)
    lookup={r['candidate']:r for r in rows}
    for case in cases:
        r=lookup[case['candidate']];col=TIME_COLORS[r['batch']-1]
        ax.plot(range(3),[r[k] for k in KEYS],color=col,lw=1.5,marker='o',ms=3,mec='#555555',mew=.35,zorder=4)
        ax.annotate(case['panel'],(2,r['participation_error']),xytext=(9,0),textcoords='offset points',
                    fontsize=7,weight='bold',va='center',ha='left')
    ax.set(xlim=(-.12,2.28),ylim=(0,.5),yticks=[0,.1,.2,.3,.4,.5],xticks=range(3),
           xticklabels=['平均\nrank','杆内\n顺序','触点\n参与'],ylabel='误差 ↓')
    ax.grid(axis='y',color='#EBEBEB',lw=.5);ax.set_axisbelow(True)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=TIME_NORM,cmap=TIME_CMAP),
        cax=fig.add_axes([.684,.560,.282,.012]),orientation='horizontal',ticks=np.arange(1,8))
    bar.outline.set_visible(False);bar.ax.tick_params(length=0,pad=2,labelsize=6)
    bar.set_label('优化批次：早 → 晚',fontsize=6,labelpad=2)


def examples(fig,cases):
    manifests=[]
    for case,left in zip(cases,[.069,.565]):
        old=SOURCE/'waveforms_trajectory'/case['old_display_id']
        previous=read(old/'comparison_manifest.json')
        source=Path(case['source']);run,a,ids=load_unit(source,1500.)
        assert sha(source)==case['source_sha256']
        candidate=read(ROUND/'candidates'/f"{case['candidate']}.json")
        physics=read(source.parents[1]/'applied_physics.json')
        names=a['contact_names'].astype(str).tolist();order=report.DISPLAY.contact_indices(names)
        dt=float(a['contact_envelope_dt_ms']);fd=float(a['sheet_activity_frame_ms']);assert fd==2
        panel_events=[]
        for entry,bottom in zip(previous['events'],[.298,.077]):
            event=entry['representative_event'];mode=entry['mode']
            assert int(a['event_mode'][event])=={'TA':1,'TB':0}[mode]
            lo,hi=run['events'][event]['window_ms'];t=a['centroid_ms'][event];zero=float(np.nanmin(t))
            mass=a['contact_envelope'][round(lo/dt):round(hi/dt)].T.astype(float).copy()
            mass/=np.maximum(mass.max(1,keepdims=True),1e-20);mass[~np.isfinite(t)]=np.nan
            ax=fig.add_axes([left,bottom,.110,.17]);cm=plt.get_cmap('magma').copy();cm.set_bad('#777777')
            ax.imshow(mass[order],extent=[lo-zero,hi-zero,14.5,-.5],aspect='auto',cmap=cm,vmin=0,vmax=1,interpolation='nearest')
            report.DISPLAY.centroid_lines(ax,t[order]-zero,color='#00CFFF')
            report.DISPLAY.contact_axis(ax);ax.tick_params(labelsize=6,length=2,pad=1.5)
            ax.set(xlim=(-125,200),xticks=[0,100],xlabel='时间 (ms)' if mode=='TB' else '',facecolor='#AAAAAA',title=f'{mode} · 事件 {event}')
            for text in ax.get_yticklabels():text.set_fontsize(6)
            frames=[]
            for col,offset in enumerate([0,30,60]):
                fa=fig.add_axes([left+.143+col*.088,bottom,.077,.17])
                frame=round((zero+offset)/fd)
                assert lo<=zero+offset<hi
                fa.imshow(field_tile(a['sheet_activity_counts'][frame],candidate,a,physics,75,size=220),extent=[0,20,0,20])
                fa.set(title=f'+{offset} ms',xticks=[0,20],yticks=[0,20],aspect='equal')
                fa.tick_params(labelsize=6,length=1.8,pad=1.5,labelleft=col==0,labelbottom=col==1)
                if col==0:fa.set_ylabel('y (mm)',labelpad=1)
                if col==1:fa.set_xlabel('x (mm)',labelpad=1)
                frames.append(dict(relative_ms=offset,frame=frame))
            panel_events.append(dict(mode=mode,event=event,field_frames=frames,mode_N=entry['N_model']))
        manifests.append(dict(panel=case['panel'],candidate=case['candidate'],source=str(source),source_sha256=sha(source),
             chronological_batch={'C':1,'D':3}[case['panel']],events=panel_events,
             selection='exact same first-primary-per-mode events as previously displayed; not claimed to be medoids',
             contact_order=list(report.DISPLAY.CONTACT_ORDER),field_scale=[0,75],envelope_scale=[0,1]))
        del run,a
    write(OUT/'model_example_manifest.json',manifests)
    return manifests


def main():
    rows,cases,grids,ref,response=load()
    fig=plt.figure(figsize=(183/25.4,195/25.4))
    xy_panels(fig,grids,ref);connections(fig,response);profiles(fig,rows,cases)
    labels=[('A',.018,.974,'位置与连接参数响应',.080),('B',.615,.974,'全部优化工作点（56）',.684),
            ('C',.018,.497,'平均顺序最佳 · 第1批',.069),('D',.514,.497,'参与概率最佳 · 第3批',.565)]
    for label,x,y,title,tx in labels:
        fig.text(x,y,label,fontsize=9,weight='bold');fig.text(tx,y,title,fontsize=7.5)
    manifests=examples(fig,cases)
    # Compact physical readout scales, replacing explanatory paragraphs.
    for x,cmap,norm,title,ticks in [(.145,'magma',Normalize(0,1),'发放包络',[0,1]),
                                   (.657,'inferno',Normalize(0,75),'原生活动 / 2 ms / 网格',[0,75])]:
        bar=fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=norm),cax=fig.add_axes([x,.024,.19,.009]),orientation='horizontal',ticks=ticks)
        bar.outline.set_visible(False);bar.ax.tick_params(length=0,pad=1,labelsize=6);bar.ax.set_title(title,fontsize=6,pad=2)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];fonts=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer);fonts.append(t.get_fontsize())
            if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:outside.append(t.get_text())
            assert not any(s in t.get_text() for s in ['P1','P2','P3','P4','P5','患者'])
    assert not outside,outside
    assert min(fonts)>=6
    name='optimization_response_propagation_abcd'
    fig.savefig(FIG/f'{name}.png',dpi=400);fig.savefig(FIG/f'{name}.svg');plt.close(fig)
    from PIL import Image
    import xml.etree.ElementTree as ET
    with Image.open(FIG/f'{name}.png') as im:im.load()
    tree=ET.parse(FIG/f'{name}.svg');assert any(t.tag.endswith('text') for t in tree.getroot().iter())
    write(OUT/'qa.json',dict(xy_conditions=49,xy_display_cells=50,connection_response_values=52,
        bayesian_profiles=56,batch_counts=[8]*7,all_profiles_original_values=True,
        exact_previous_events_preserved=True,source_npz_hashes_verified=True,all_outputs_decoded=True,
        text_outside_canvas=outside,min_font_pt=min(fonts),patient_panels=0,p_number_labels=0,
        examples=manifests,agent_visual_review='PENDING',human_visual_review='PENDING'))
    (FIG/'README.md').write_text(
        '### optimization_response_propagation_abcd.png\n'
        'A将核位置改为Core A/B各一张真实5×5二维网格，每格蓝色右下半为平均rank误差、橙色左上半为杆内顺序误差；星号为局部参考点。A下方保留此前EE权重及方向响应，B显示全部56个无标签贝叶斯提案，颜色与3D图一致，批次越晚越深；C/D标出对应两个例子。\n'
        'C/D只展示原两个工作点的TA/TB单事件发放包络与同事件+0/30/60ms原生场，不含患者对照，事件身份、通道顺序、时间轴及色标均延用前图。核XY网格与EE/方向曲线采用各自原始固定背景，具体差别见figure_contract.json；本次未重跑、重新选例或改动优化目标。\n'
        '**关注点**：二维位置与连接响应、优化批次中的三指标取舍，以及两类实际传播能否支持总体指标的解释；单种子开发结果仍不等于完整患者模式恢复。\n')
    print(json.dumps(dict(figure=str(FIG/f'{name}.png'),profiles=56,xy_unique_conditions=49,events=manifests),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

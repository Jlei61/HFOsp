"""English A-D plate: parameter responses, measured search, low/high examples."""
from pathlib import Path
import argparse, csv, json, hashlib, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
from scripts.paper_figures.plot_topic4_workpoint_waveforms_trajectory import Arrow3D
from scripts import report_topic4_geometry_axis_joint_search as report
from scripts.analyze_topic4_core_connectivity_search import load_unit
from scripts.render_topic4_shape_output_gifs import field_tile

ROUND=Path('/data/hfosp/topic4_sef_hfo/geometry_axis_joint_distribution_20260917')
SOURCE=ROUND/'key_results/data_driven_metric_tradeoffs/full_search_history'
OUT=ROUND/'key_results/data_driven_metric_tradeoffs/compact_english_acceptance'
FIG=OUT/'figures'
KEYS=['rank_error','within_rod_order_error','participation_error']
COLORS=plt.get_cmap('Blues')(np.linspace(.22,.90,16))
CMAP=ListedColormap(COLORS);NORM=BoundaryNorm(np.arange(.5,17),16)
BLUE,ORANGE='#0072B2','#D55E00'
WIDTH,HEIGHT=238.,235.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'legend.frameon':False,'axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
    'savefig.facecolor':'white'})


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p):return list(csv.DictReader(open(p)))
def axes(fig,x,y,w,h,**kw):return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT],**kw)
def label(fig,x,y,text,**kw):return fig.text(x/WIDTH,y/HEIGHT,text,fontsize=kw.pop('fontsize',8),**kw)


def load(selection):
    rows=table(SOURCE/'all_workpoints_source.csv')
    for r in rows:
        for k in KEYS+['J','J_joint']:r[k]=float(r[k])
        r['stage']=int(r['stage']);r['N']=int(r['N'])
        r['display_mean_error']=float(np.mean([r[k] for k in KEYS]))
    assert len(rows)==226 and len({r['candidate'] for r in rows})==226
    assert read(SOURCE/'qa.json')['centroid_metrics_recomputed']==226
    key={'mean':'display_mean_error','loss':'J','rank':'rank_error'}[selection]
    ordered=sorted(rows,key=lambda r:(r[key],r['candidate']))
    cases=[dict(ordered[0],panel='C'),dict(ordered[-1],panel='D')]
    curves=table(SOURCE/'connection_response_source.csv');xy=table(SOURCE/'xy_grid_source.csv')
    for r in xy:
        r['core']=int(r['core'])
        for k in ['x','y']+KEYS:r[k]=float(r[k])
    assert len(curves)==52 and len(xy)==50
    contract=dict(question='Show measured parameter responses and search chronology, then inspect actual propagation from low/high observable-error workpoints.',
        archetype='image plate + quant',backend='Python matplotlib',
        panels=dict(A='Two connection curves above four measured XY heatmaps; rows are mean-rank and within-rod errors, columns Core A/B.',
                    B='All 226 profiles on the left and the same 226 points in 3D on the right; shared vertical Iteration order scale; C/D stars.',
                    C='Lower-error workpoint: same-row TA/TB contact envelopes and their two stacked rows of native snapshots.',
                    D='Higher-error workpoint with identical axes, scales and event-selection rule.'),
        workpoint_selection=dict(key=key,rule='Minimum and maximum among all 226 conditions; stable candidate-ID tie break.',
            display_mean_definition='Arithmetic mean of the three displayed errors, equal weights, no rescaling. Only a descriptive illustration selector, not a revised training loss or an overall recovery metric.' if selection=='mean' else None,
            selection_status='User option if replied; otherwise explicitly communicated reversible plotting assumption.'),
        event_selection='Within each frozen model mode, choose an actual event medoid under squared Euclidean distance of participation plus masked normalized-rank features; SCL/ICL equally weighted, participation/rank blocks equally weighted. No patient distance used.',
        iteration='Same 16 admission/proposal stages as validated history; not individual worker order or equal-size BO generations. Objective switches after stages5 and12; no trajectory edge crosses these switches.',
        observations='Same frozen patient FIT19770 target; topology2511/noise847401;60s;burn-in1500ms. TA/TB used only to organize C/D; early historical optimization did use labels.',
        background=read(SOURCE/'figure_contract.json')['background_difference'],
        display='Fixed SCL9..6 then ICL11..1; fixed -125..200ms. Envelope per-contact max normalized; native all-E2ms frames0/30/60ms, shared0..75 scale. No patient panel.',
        omissions='No data omitted from A/B. Two workpoints with one actual representative event per mode are explicitly requested summaries. Legacy anchor and narrative annotations removed from canvas; source history retains provenance.',
        export=dict(width_mm=WIDTH,height_mm=HEIGHT,minimum_font_pt=8,formats=['png','svg'],dpi=400),
        new_simulations=0,source_files={str(p):sha(p) for p in [SOURCE/'all_workpoints_source.csv',SOURCE/'xy_grid_source.csv',SOURCE/'connection_response_source.csv',SOURCE/'history.json',SOURCE/'trajectory_representatives.json',Path(__file__)]})
    write(OUT/'figure_contract.json',contract);write(OUT/'selected_workpoints.json',dict(selection=key,weights=[1/3]*3 if selection=='mean' else None,cases=cases))
    return rows,cases,curves,xy


def parameters(fig,curves,xy):
    # Curve panels retain the old fixed background and exact 13-point samples.
    for j,x,xlabel,xticks in [(4,14,'Core-to-surround EE\n(multiplier)',[1,1.25,1.5]),
                              (5,55,'EE axis offset (°)',[-20,0,20])]:
        ax=axes(fig,x,208,27,19);twin=ax.twinx()
        for target,key,color,marker in [(ax,'rank',BLUE,'o'),(twin,'local_order',ORANGE,'s')]:
            r=sorted([r for r in curves if int(r['axis'])==j and r['observable']==key],key=lambda r:float(r['parameter_value']))
            assert len(r)==13
            target.plot([float(q['parameter_value']) for q in r],[float(q['value']) for q in r],color=color,marker=marker,ms=2.2,lw=1)
            target.set(ylim=(0,.42),yticks=[0,.2,.4]);target.tick_params(axis='y',colors=color,pad=1.5,length=2)
        ax.spines['left'].set_color(BLUE);twin.spines['right'].set_visible(True);twin.spines['right'].set_color(ORANGE)
        twin.spines['left'].set_visible(False);twin.spines['bottom'].set_visible(False)
        ax.axvline(1.25 if j==4 else 0,color='#999999',ls=':',lw=.7)
        ax.set(xlabel=xlabel,xticks=xticks);ax.tick_params(axis='x',pad=2,length=2)
        if j==4:ax.set_ylabel('Error ↓',labelpad=2)
    label(fig,25,192,'Core A',ha='center');label(fig,66,192,'Core B',ha='center')
    for row,(key,cmap,y,title) in enumerate([('rank_error','Blues',168,'Mean rank error'),('within_rod_order_error','Oranges',137,'Within-rod error')]):
        for core,x in [(0,14),(1,55)]:
            r=[r for r in xy if r['core']==core];xs=np.unique([q['x'] for q in r]);ys=np.unique([q['y'] for q in r])
            assert len(xs)==len(ys)==5
            z=np.array([[next(q[key] for q in r if q['x']==xx and q['y']==yy) for xx in xs] for yy in ys])
            ax=axes(fig,x,y,23,23)
            ax.imshow(z,origin='lower',extent=[xs[0]-.125,xs[-1]+.125,ys[0]-.125,ys[-1]+.125],cmap=cmap,vmin=0,vmax=.3,interpolation='nearest',aspect='equal')
            ax.scatter(xs[2],ys[2],marker='*',s=38,c='white',edgecolor='#333333',lw=.7)
            ax.set(xticks=[xs[0],xs[-1]],yticks=[ys[0],ys[-1]],ylabel='y (mm)')
            ax.set_xlabel('x (mm)' if row==1 else '')
            ax.tick_params(pad=1.5,length=2,labelbottom=row==1)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,.3),cmap=cmap),cax=axes(fig,83,y,2,23),ticks=[0,.15,.3],orientation='vertical')
        cbar.outline.set_visible(False);cbar.ax.tick_params(length=2,pad=2);cbar.set_label(title,labelpad=3)


def profile(fig,rows,cases):
    ax=axes(fig,113,148,37,79)
    for r in sorted(rows,key=lambda r:(r['stage'],r['candidate'])):
        ax.plot(range(3),[r[k] for k in KEYS],color=COLORS[r['stage']-1],lw=.5,alpha=1,zorder=1+r['stage']/20)
    for r in cases:
        ax.plot(range(3),[r[k] for k in KEYS],color=COLORS[r['stage']-1],lw=1.8,marker='*',ms=8,mec='#222222',mew=.55,zorder=5)
        ax.annotate(r['panel'],(2,r['participation_error']),xytext=(5,0),textcoords='offset points',fontsize=9,weight='bold',va='center')
    ax.set(xlim=(-.12,2.28),ylim=(0,.5),yticks=[0,.1,.2,.3,.4,.5],xticks=range(3),
           xticklabels=['Mean\nrank','Within-rod\norder','Contact\nparticipation'],ylabel='Error ↓')
    ax.grid(axis='y',color='#EAEAEA',lw=.55);ax.set_axisbelow(True)
    return ax


def trajectory(fig,rows,cases):
    ax=axes(fig,153,140,61,91,projection='3d',computed_zorder=False)
    xyz=np.array([[r[k] for k in KEYS] for r in rows])
    ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-55)
    ax.scatter(*xyz.T,c=[r['stage'] for r in rows],cmap=CMAP,norm=NORM,s=9,alpha=.9,depthshade=False,edgecolors='none',zorder=1)
    records=read(SOURCE/'trajectory_representatives.json')['records']
    for objective,ls in [('mode_conditioned',':'),('label_free','--'),('joint_label_free','-')]:
        r=[r for r in records if r['objective']==objective]
        for a,b in zip(r,r[1:]):
            p=np.array(a['xyz']);q=np.array(b['xyz'])
            ax.plot(*np.array([p,q]).T,color='#343434',lw=.8,ls=ls,zorder=3)
            ax.add_artist(Arrow3D(p+.28*(q-p),p+.78*(q-p),mutation_scale=6,arrowstyle='-|>',color='#343434',lw=.7,zorder=4))
    for r in cases:
        ax.scatter(*[r[k] for k in KEYS],color=COLORS[r['stage']-1],marker='*',s=100,edgecolors='#222222',lw=.7,depthshade=False,zorder=5)
    for axis,label in [(ax.xaxis,'Rank error'),(ax.yaxis,'Within-rod error'),(ax.zaxis,'Participation error')]:
        axis.set_ticks([0,.2,.4]);axis.set_pane_color((1,1,1,0));axis._axinfo['grid'].update(color=(.88,.90,.92,1),linewidth=.5)
        axis.set_label_text(label,fontsize=8);axis.labelpad=1
    ax.set(xlim=(0,.4),ylim=(0,.5),zlim=(0,.4));ax.tick_params(pad=0,labelsize=8)
    # The y-axis zero meets x=0.4 in this view; do not superimpose the two labels.
    ax.set_yticks([.2,.4])
    fig.canvas.draw()
    for r in cases:
        x,y,_=proj3d.proj_transform(*[r[k] for k in KEYS],ax.get_proj())
        ax.annotate(r['panel'],(x,y),xytext=(4,7),textcoords='offset points',fontsize=9,weight='bold',zorder=6)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=axes(fig,226,152,2,73),orientation='vertical',ticks=[1,5,10,16])
    bar.outline.set_visible(False);bar.ax.tick_params(length=2,pad=2);bar.set_label('Iteration order',labelpad=3)
    for boundary in [5.5,12.5]:bar.ax.axhline(boundary,color='#444444',lw=.65)
    return ax


def mode_medoid(times,names):
    mask=np.isfinite(times);ranks=np.zeros_like(times)
    for k,t in enumerate(times):
        n=int(mask[k].sum());assert n>=2
        ranks[k,mask[k]]=(rankdata(t[mask[k]],method='average')-1)/(n-1)
    weights=np.array([1/np.sqrt(2*sum(q.startswith('SCL' if n.startswith('SCL') else 'ICL') for q in names)) for n in names])
    features=np.concatenate([mask*weights,ranks*weights],axis=1)/np.sqrt(2)
    distances=np.sum((features-features.mean(0))**2,axis=1)
    return int(np.argmin(distances)),distances


def propagation(fig,cases):
    manifests=[];field_clip=[]
    for case,bottom in zip(cases,[74,12]):
        source=Path(case['source']);assert sha(source)==case['source_sha256']
        run,a,ids=load_unit(source,1500)
        ids=np.array([int(i) for i in ids if run['events'][i]['window_ms'][1]<=60000],int)
        assert len(ids)==case['N']
        physics=read(source.parents[1]/'applied_physics.json');candidate=physics['candidate']
        names=a['contact_names'].astype(str).tolist();order=report.DISPLAY.contact_indices(names)
        dt=float(a['contact_envelope_dt_ms']);fd=float(a['sheet_activity_frame_ms']);assert fd==2
        entries=[]
        for mode,m,x,row in [('TA',1,14,1),('TB',0,56,0)]:
            mid=ids[a['event_mode'][ids]==m];assert len(mid)>0
            pick,distance=mode_medoid(a['centroid_ms'][mid],names);i=int(mid[pick])
            lo,hi=run['events'][i]['window_ms'];times=a['centroid_ms'][i];zero=float(np.nanmin(times));part=np.isfinite(times)
            env=a['contact_envelope'][round(lo/dt):round(hi/dt)].T.astype(float).copy()
            env/=np.maximum(env.max(1,keepdims=True),1e-20);env[~part]=np.nan
            ax=axes(fig,x,bottom,27,49);cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777')
            ax.imshow(env[order],extent=[lo-zero,hi-zero,14.5,-.5],aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            report.DISPLAY.centroid_lines(ax,times[order]-zero,color='#00CFFF')
            ax.set(xlim=(-125,200),ylim=(14.5,-.5),yticks=range(15),yticklabels=report.DISPLAY.CONTACT_ORDER,
                   xticks=[0,100],xlabel='Time (ms)' if case['panel']=='D' else '',facecolor='#AAAAAA')
            ax.axhline(3.5,color='white',lw=.7)
            ax.tick_params(labelsize=8,length=2,pad=1.5,labelbottom=case['panel']=='D')
            for text in ax.get_yticklabels():text.set_color('#239BAF' if text.get_text().startswith('SCL') else '#D17C1F')
            label(fig,x+13.5,bottom+51,mode,ha='center')
            field_y=bottom+25 if row else bottom
            label(fig,103,field_y+12,mode,ha='right',va='center')
            frames=[]
            for col,offset in enumerate([0,30,60]):
                frame=round((zero+offset)/fd);assert lo<=zero+offset<hi
                fa=axes(fig,107+col*26,field_y,24,24)
                fa.imshow(field_tile(a['sheet_activity_counts'][frame],candidate,a,physics,75,size=300),extent=[0,20,0,20])
                fa.set(xticks=[0,20],yticks=[0,20],aspect='equal')
                fa.tick_params(labelsize=8,length=2,pad=1,labelleft=(col==0 and row==0),labelbottom=(col==0 and row==0 and case['panel']=='D'))
                if row==1:label(fig,119+col*26,bottom+51,f'{offset} ms',ha='center')
                if col==0 and row==0 and case['panel']=='D':fa.set_xlabel('x (mm)',labelpad=1)
                frames.append(dict(offset_ms=offset,frame_index=frame))
                field_clip.append(dict(panel=case['panel'],mode=mode,frame=frame,clipped_fraction=float(np.mean(a['sheet_activity_counts'][frame]>75))))
            entries.append(dict(mode=mode,event=i,N_mode=len(mid),medoid_distance=float(distance[pick]),window_ms=[lo,hi],zero_centroid_ms=zero,frames=frames))
        label(fig,97,bottom+24.5,'y (mm)',rotation=90,rotation_mode='anchor',ha='center',va='center')
        manifests.append(dict(panel=case['panel'],candidate=case['candidate'],source=str(source),source_sha256=sha(source),events=entries,
            contact_order=list(report.DISPLAY.CONTACT_ORDER),time_limits_ms=[-125,200],field_scale=[0,75],envelope_scale=[0,1]))
        del run,a
    for x,cmap,norm,ticks,name in [(87,'magma',Normalize(0,1),[0,1],'Normalized envelope'),(191,'inferno',Normalize(0,75),[0,75],'Native activity / 2 ms / bin')]:
        cb=fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=norm),cax=axes(fig,x,52,2,35),orientation='vertical',ticks=ticks)
        cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=2);cb.set_label(name,labelpad=3)
    write(OUT/'propagation_manifest.json',dict(cases=manifests,selection='Within-model mode medoids; not patient-nearest events',native_clipping=field_clip))
    return manifests


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection',choices=['mean','loss','rank'],default='mean');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    rows,cases,curves,xy=load(args.selection)
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4))
    parameters(fig,curves,xy);profile(fig,rows,cases);trajectory(fig,rows,cases);examples=propagation(fig,cases)
    for s,x,y in [('A',2,230),('B',103,230),('C',2,126),('D',2,64)]:label(fig,x,y,s,fontsize=11,weight='bold')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];fonts=[];strings=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            fonts.append(t.get_fontsize());strings.append(t.get_text());bb=t.get_window_extent(renderer)
            if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.x1+.5 or bb.y1>fig.bbox.y1+.5:outside.append(t.get_text())
            assert not any('\u4e00'<=c<='\u9fff' for c in t.get_text()),t.get_text()
    assert min(fonts)>=8
    assert not outside,outside
    stem='parameter_search_propagation_compact'
    fig.savefig(FIG/f'{stem}.png',dpi=400);fig.savefig(FIG/f'{stem}.svg');plt.close(fig)
    from PIL import Image
    import xml.etree.ElementTree as ET
    with Image.open(FIG/f'{stem}.png') as im:im.load()
    svg=ET.parse(FIG/f'{stem}.svg');assert any(e.tag.endswith('text') for e in svg.getroot().iter())
    write(OUT/'qa.json',dict(profiles=226,scatter_points=226,heatmaps=4,heatmap_cells=100,connection_curves=4,connection_values=52,
        all_text_english=True,minimum_font_pt=min(fonts),text_outside_canvas=outside,global_title=False,narrative_caption_on_canvas=False,
        selection=args.selection,selected=[r['candidate'] for r in cases],events=examples,outputs_decoded=True,
        all_colorbars_vertical=True,iteration_colormap_shared=True,agent_visual_review='PENDING',human_visual_review='PENDING'))
    (FIG/'README.md').write_text('### parameter_search_propagation_compact.png\nA上排保留EE强度/方向两组13点响应，下排为Core A/B的平均rank与杆内顺序四张真实XY网格；B把全部226条误差曲线和同一226点的3D轨迹并排，竖向Iteration order沿用16个历史提案/入池阶段，目标改变处不连接轨迹。C/D按selected_workpoints.json中的明确规则选择低、高误差工作点；各模式取模型自身参与—rank分布的实际medoid事件，左侧TA/TB同排，右侧同步原生场上下两排。\n所有坐标、标签为英文，最小8pt，SCL/ICL通道槽位及时间轴固定；包络与原生场色标跨条件共享。三个展示误差的均值若用于选图，仅为描述性选例，不是新训练loss或传播恢复的总体结论；不通过患者相似度挑单事件。\n**关注点**：同一三个观测的低、高误差是否对应不同的实际传播，兼顾未参与触点；A的XY与EE/方向仍使用各自历史固定背景，详见figure_contract.json。\n')
    print(json.dumps(dict(figure=str(FIG/f'{stem}.png'),selected=[dict(panel=c['panel'],candidate=c['candidate'],errors=[c[k] for k in KEYS],stage=c['stage']) for c in cases]),ensure_ascii=False))


if __name__=='__main__':main()

"""P1/P3 contact envelopes plus chronological 3-D Bayesian batch diagnostics."""
from pathlib import Path
import csv,json,sys,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap,BoundaryNorm
from mpl_toolkits.mplot3d import proj3d
from scripts.paper_figures import plot_topic4_parameter_tradeoff_composite as previous

ROUND=previous.ROUND;SOURCE=previous.SOURCE
OUT=SOURCE/'waveforms_trajectory';FIG=OUT/'figures'
KEYS=previous.KEYS


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Arrow3D(FancyArrowPatch):
    def __init__(self,start,end,**kwargs):
        super().__init__((0,0),(0,0),**kwargs)
        self.vertices=np.array([start,end]).T
    def do_3d_projection(self,renderer=None):
        x,y,z=proj3d.proj_transform(*self.vertices,self.axes.get_proj())
        self.set_positions((x[0],y[0]),(x[1],y[1]))
        return float(np.min(z))


def history():
    rows=list(csv.DictReader(open(SOURCE/'all_bayesian_candidates.csv')))
    for r in rows:
        for k in KEYS+['J','J_joint']:r[k]=float(r[k])
    byid={r['candidate']:r for r in rows};selected=read(SOURCE/'selection.json')['selected']
    batches=[];identity={}
    for rid,root in enumerate([previous.OLD,ROUND],1):
        earlier=set()
        for path in sorted((root/'proposals').glob('batch_*.json')):
            d=read(path);number=len(batches)+1
            snapshot=d['training_snapshot']
            training=snapshot if isinstance(snapshot,list) else read(snapshot)['rows']
            seen={r['candidate'] for r in training}
            assert earlier<=seen and not set(d['ids'])&seen
            earlier.update(d['ids'])
            new=[byid[c] for c in d['ids']]
            winner=min(new,key=lambda r:r['J'])
            record=dict(number=number,round=rid,round_batch=d['batch'],ids=d['ids'],
                        representative=winner['candidate'],representative_J=winner['J'],
                        representative_xyz=[winner[k] for k in KEYS],
                        timestamp=d['time'],proposal=str(path),proposal_sha256=sha(path),
                        time_local=datetime.datetime.fromtimestamp(d['time'],datetime.timezone(datetime.timedelta(hours=8))).isoformat())
            batches.append(record)
            for cid in d['ids']:identity[cid]=number
    assert len(rows)==56 and len(batches)==7 and len(identity)==56
    assert all(b['timestamp']>a['timestamp'] for a,b in zip(batches,batches[1:]))
    for r in selected:r['chronological_batch']=identity[r['candidate']]
    write(OUT/'batch_history.json',dict(batches=batches,point_to_batch=identity,selected=selected,
        selection='Minimum original frozen J among eight NEW proposals in each batch; not incumbent, ancestry, or worker finish order.',
        arrows='Within each round, chronological connection of batch representatives; no edge across changed objective/search range.',
        old_round='single label-free J, four batches',new_round='J and J_joint, three batches',
        labels_used=False,new_simulations=0))
    return rows,selected,batches,identity


def trajectory(rows,selected,batches,identity):
    fig=plt.figure(figsize=(183/25.4,112/25.4))
    ax=fig.add_axes([.01,.14,.70,.82],projection='3d',computed_zorder=False)
    xyz=np.array([[r[k] for k in KEYS] for r in rows]);limits=np.ceil(xyz.max(0)*1.06/.05)*.05
    ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-53)
    colors=plt.get_cmap('Blues')(np.linspace(.22,.90,7))
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(.5,8),7)
    time_index=np.array([identity[r['candidate']] for r in rows])
    dots=ax.scatter(*xyz.T,c=time_index,cmap=cmap,norm=norm,s=17,alpha=.78,depthshade=False,edgecolors='none',zorder=1)
    for rid,style in [(1,'--'),(2,'-')]:
        bs=[b for b in batches if b['round']==rid]
        for a,b in zip(bs,bs[1:]):
            start=np.array(a['representative_xyz']);end=np.array(b['representative_xyz'])
            ax.plot(*np.array([start,end]).T,color='#333333',lw=1,ls=style,zorder=3)
            # The arrow head sits on the measured line; neither endpoint is a new simulated point.
            ax.add_artist(Arrow3D(start+.30*(end-start),start+.78*(end-start),
                mutation_scale=8,arrowstyle='-|>',color='#222222',lw=.85,zorder=4))
    for b in batches:
        ax.scatter(*b['representative_xyz'],s=18,c='#222222',marker='o',depthshade=False,zorder=4)
    for r in selected:
        ax.scatter(*[r[k] for k in KEYS],c=r['color'],marker=r['marker'],s=43,
                   edgecolors='white',lw=.5,depthshade=False,zorder=5)
    for axis,lim,label in zip([ax.xaxis,ax.yaxis,ax.zaxis],limits,
              ['平均 rank 误差 ↓','杆内顺序误差 ↓','触点参与误差 ↓']):
        axis.set_ticks(np.arange(0,lim+.01,.1));axis.set_pane_color((1,1,1,0))
        axis._axinfo['grid'].update(color=(.89,.90,.92,1),linewidth=.5)
        axis.line.set_color('#999999');axis.line.set_linewidth(.6)
        axis.set_label_text(label,fontsize=7);axis.labelpad=4
    ax.set(xlim=(0,limits[0]),ylim=(0,limits[1]),zlim=(0,limits[2]));ax.tick_params(pad=0,labelsize=6)
    fig.canvas.draw()
    # Leader-labelled chronological batch representatives. Batch members are unordered.
    offsets={1:(-30,40),2:(-28,20),3:(25,27),4:(-36,-25),5:(-12,-42),6:(27,-7),7:(28,-24)}
    for b in batches:
        x,y,_=proj3d.proj_transform(*b['representative_xyz'],ax.get_proj())
        ax.annotate(str(b['number']),(x,y),xytext=offsets[b['number']],textcoords='offset points',
                    fontsize=7,weight='bold',color='#222222',ha='center',va='center',zorder=8,
                    arrowprops=dict(arrowstyle='-',color='#666666',lw=.45))
    handles=[Line2D([],[],color=r['color'],marker=r['marker'],ms=4,lw=1,
        label=f"{r['display_id']} {previous.ROLES[i]} · 第{r['chronological_batch']}批") for i,r in enumerate(selected)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.726,.81),fontsize=6.5,
               handlelength=1.4,handletextpad=.4,labelspacing=.8,borderaxespad=0)
    fig.legend(handles=[Line2D([],[],color='#333333',ls='--',lw=1,label='前轮：1 → 4'),
                        Line2D([],[],color='#333333',ls='-',lw=1,label='本轮：5 → 7')],
               loc='upper left',bbox_to_anchor=(.726,.47),fontsize=7,handlelength=2,labelspacing=.7,borderaxespad=0)
    fig.text(.726,.32,'箭头连接各批原损失最低的新候选',fontsize=6)
    fig.text(.726,.275,'每批 8 个并行提案，批内无先后',fontsize=6)
    bar=fig.colorbar(dots,cax=fig.add_axes([.14,.055,.43,.024]),orientation='horizontal',ticks=np.arange(1,8))
    bar.outline.set_visible(False);bar.ax.tick_params(length=0,labelsize=6,pad=2)
    bar.set_label('提案批次：早 → 晚',fontsize=6.5,labelpad=3)
    previous.FIG=FIG
    return previous.save(fig,'three_observable_optimization_batches')


def waveform_panels(selected):
    from scripts import report_topic4_geometry_axis_joint_search as report
    from scripts.analyze_topic4_core_connectivity_search import load_unit
    report.patient_mean.OUT=ROUND/'analysis/patient_display'
    for path,digest in read(ROUND/'analysis/figure_contract.json')['files'].items():
        assert sha(path)==digest,'patient display source changed'
    picks=[r for r in selected if r['display_id'] in ['P1','P3']]
    maximum=1.
    for pick in picks:
        with np.load(Path(pick['source']).with_suffix('.npz')) as z:
            dt=float(z['sheet_activity_frame_ms'])
            maximum=max(maximum,float(np.quantile(z['sheet_activity_counts'][round(1500/dt):],.999)))
    entries=[]
    for pick in picks:
        cid=pick['candidate'];score=read(ROUND/'scores'/f'{cid}.json');source=Path(score['source'])
        assert sha(source)==score['source_sha256']
        run,arrays,ids=load_unit(source,1500.)
        ids=np.asarray([int(i) for i in ids if run['events'][i]['window_ms'][1]<=run['actual_duration_ms']],int)
        names=arrays['contact_names'].astype(str).tolist()
        candidate=read(ROUND/'candidates'/f'{cid}.json');candidate['topology']=2511
        physics=read(source.parents[1]/'applied_physics.json')
        folder=OUT/pick['display_id'];folder.mkdir(exist_ok=True)
        path=report.comparison(score,candidate,run,arrays,ids,physics,folder,names,
                display_name=f"{pick['display_id']} · {pick['role']}",field_vmax=maximum)
        entry=dict(display_id=pick['display_id'],candidate=cid,snapshot=str(path),N=len(ids),
                   TA=int((arrays['event_mode'][ids]==1).sum()),TB=int((arrays['event_mode'][ids]==0).sum()),
                   source=str(source),source_sha256=sha(source),field_vmax_shared_P1_P3=maximum,
                   selection='first primary event per frozen mode; not selected for patient similarity',
                   waveform='patient HFO envelope, model firing-density envelope, not raw voltage or spectrum')
        entries.append(entry)
        (folder/'README.md').write_text(f'### patient_model_mean_event_field.png\n{pick["display_id"]}（{pick["role"]}）的患者平均包络、模型全部同类事件平均包络、首个合格单事件与同步原生场。通道固定按SCL/ICL分组；P1/P3共享原生场色标。\n**关注点**：总体摘要最优是否掩盖模式内传播偏差；患者为冻结32+32形态包，包络不冒充原始电压波形或频谱。\n')
        del run,arrays
    write(OUT/'waveform_manifest.json',dict(entries=entries,report_source_sha256=sha(report.__file__),new_simulations=0))
    return entries


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    write(OUT/'figure_contract.json',dict(
        question='What do P1 and P3 actually generate, and in what chronological order were candidates proposed?',
        backend='Python matplotlib',archetype='image plate + quant',
        waveform='Reuse accepted patient mean, all-event model mean, first primary event and synchronized native fields; fixed contacts and axes.',
        path='Two chronological paths through minimum-original-J NEW candidate per batch; no within-batch ordering and no edge between changed optimization rounds.',
        exports=['400 dpi PNG','editable SVG'],no_pdf=True,minimum_font_pt=6,
        scientific_limits='Descriptive batch representatives, not parent-child ancestry or best-so-far incumbent; sample mode labels used for display only.'))
    rows,selected,batches,identity=history()
    plot=trajectory(rows,selected,batches,identity)
    entries=waveform_panels(selected)
    from PIL import Image
    import xml.etree.ElementTree as ET
    for f in OUT.rglob('*.png'):
        with Image.open(f) as im:im.load()
    for f in OUT.rglob('*.svg'):ET.parse(f)
    write(OUT/'qa.json',dict(trajectory=plot,all_56_proposals_retained=True,batch_order_matches_timestamps=True,
         previous_batch_feedback_present=True,all_outputs_decoded=True,waveforms=entries,
         agent_visual_review='PENDING',human_visual_review='PENDING'))
    (FIG/'README.md').write_text('### three_observable_optimization_batches.png\n颜色深浅按真实提案记录给56个候选标出七批顺序，1–4是前轮原目标优化，5–7是本轮双目标优化。两段箭头分别连接各批8个新增候选中原损失J最低的实际点，数字为批次；不是累计最优、并行worker完成顺序或父子关系。\n**关注点**：P1早于P3；P4/P5同批，无先后。目标及搜索范围在两轮之间改变，因此两段路径不跨轮连接。\n')
    print(json.dumps(dict(out=str(OUT),waveforms=entries),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Native SNN spatial figures; no aggregate-rate interpolation or state relabel."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from PIL import Image,ImageDraw,ImageFont
import figures as fig
OUT,FIG=fig.OUT,fig.FIG
read=fig.read
SUMMARY=read(OUT/'native_summary.json');DETAIL=read(OUT/'native_details.json')
PROV=read(OUT/'metric_provenance.json');NAMES=PROV['display_order']
RUNS={key:dict(z=np.load(OUT/'native'/key/'trajectory.npz'),
    ob=read(OUT/'native'/key/'observation.json'),physics=read(OUT/'native'/key/'applied_physics.json')) for key in ('a','b','cd')}
VMAX=max(int(d['z']['sheet_activity_counts'][1000:].max()) for d in RUNS.values())
ENVMAX=max(float(d['z']['contact_envelope'][1000:].max()/.002) for d in RUNS.values())
SELECTION=[]
def order(d):return [d['z']['contact_names'].tolist().index(n) for n in NAMES]
def title(row):return f'{fig.J} = {row["J"]:.9f}  (parameter of {row["reduced_rows"]})'

def contacts(ax,labels=True):
    ax.set(yticks=range(15),yticklabels=NAMES if labels else [],ylim=(14.5,-.5))
    ax.tick_params(axis='y',labelsize=8,length=2)
    for i,t in enumerate(ax.get_yticklabels()):t.set_color('#35a6b7' if i<4 else '#d48024')
    ax.axhline(3.5,color='white',lw=.8)

def geometry(ax,d):
    c=d['physics']['candidate']
    for center,radius in zip(c['centers_mm'],c['radii_mm']):ax.add_patch(Circle(center,radius,fill=False,ec='white',lw=1.1))
    xy=d['z']['contact_xy_mm'];ax.scatter(xy[:,0],xy[:,1],s=12,facecolors='none',edgecolors='cyan',lw=.6)
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
    ax.tick_params(labelsize=8,length=2)

def overview():
    f=plt.figure(figsize=(16.4,9.6));grid=f.add_gridspec(3,3,width_ratios=[1.05,1.45,.65],hspace=.52,wspace=.28)
    f.subplots_adjust(left=.055,right=.905,bottom=.09,top=.86)
    for i,row in enumerate(SUMMARY):
        d=RUNS[row['native_key']];z=d['z'];tt=(np.arange(len(z['six_group_counts_2ms']))+.5)*.002
        size=np.bincount(z['region'],minlength=6);rates=gaussian_filter1d(z['six_group_counts_2ms']/size/.002,1.5,axis=0)
        net=(rates[:,:3]*size[:3]).sum(1)/size[:3].sum();win=(tt>=4)&(tt<=6)
        ax=f.add_subplot(grid[i,0])
        for yy,c,lw in [(rates[:,0],fig.AB[0],1),(rates[:,1],fig.AB[1],1),(net,'black',1.35)]:ax.plot(tt[win],yy[win],color=c,lw=lw)
        assert rates[win,:2].max()<=500
        ax.set(xlim=(4,6),ylim=(0,500),yticks=[0,250,500],xlabel='Time (s)' if i==2 else '',ylabel='E rate (Hz / cell)')
        ax.set_title(title(row),loc='left',fontsize=10.5,pad=9)
        ax=f.add_subplot(grid[i,1]);env=z['contact_envelope'][win][:,order(d)].T/.002
        im=ax.imshow(env,aspect='auto',extent=[4,6,14.5,-.5],cmap='magma',vmin=0,vmax=ENVMAX,interpolation='nearest')
        contacts(ax);ax.set(xlabel='Time (s)' if i==2 else '')
        if i==0:ax.set_title('All fixed contacts; common rate scale',fontsize=11)
        ax=f.add_subplot(grid[i,2]);ax.axis('off')
        if i==0:ax.set_title('Pooled propagation errors ↓',fontsize=11,pad=9)
        items=[('Mean rank',row['rank_error']),('Within-shaft order',row['within_shaft_order_error']),('Participation',row['participation_error'])]
        for y,(label,v) in zip([.82,.56,.30],items):
            ax.text(0,y,label,fontsize=10);ax.text(1,y,f'{v:.3f}',fontsize=13,ha='right',weight='bold')
        ax.text(0,.02,f'Valid / detected: {row["N_valid"]} / {row["N_detected_in_window"]}',fontsize=10)
    cb=f.add_axes([.93,.31,.010,.30]);f.colorbar(im,cax=cb,label='Contact E rate (Hz / cell)')
    f.suptitle('Native SNN: matched core coupling, spatial and contact readouts',fontsize=16,y=.978)
    f.text(.50,.925,'All three runs show two-core bursts; b–d closure states were not reproduced',ha='center',fontsize=12)
    f.legend(handles=[Line2D([],[],color=c,label=n) for c,n in zip(fig.AB+['black'],['Core A E','Core B E','All E'])],
        loc='lower left',bbox_to_anchor=(.04,.014),ncol=3,frameon=False)
    f.text(.52,.029,'Metrics: all valid events in 2–12 s; waveforms: identical 4–6 s windows',fontsize=10)
    fig.save(f,'03_native_network_contacts_metrics',
        '三个参数取值的原生SNN波形与固定触点包络，右列沿用既有三项无标签摘要误差，全部有效事件先合并再统计。波形显示相同4–6秒窗口，统计窗口为2–12秒；c/d只对应一次J=1.38原生对照。',
        '三次原生运行都表现为双核regular bursts，未复现六群体模型b–d的高背景；此图不能称作四条分支各自的空间传播。')

def event_figure(key,ids,number):
    d=RUNS[key];z=d['z'];physics=d['physics'];idx=order(d);row=next(x for x in SUMMARY if x['native_key']==key)
    f=plt.figure(figsize=(17.6,4.25*len(ids)+.9));grid=f.add_gridspec(len(ids),5,width_ratios=[1.95,1,1,1,1],hspace=.42,wspace=.28)
    f.subplots_adjust(left=.055,right=.915,bottom=.08,top=.86 if len(ids)==2 else .80)
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#888888')
    for line,event in enumerate(ids):
        e=d['ob']['events'][event];lo,hi=e['window_ms'];cent=z['centroid_ms'][event];zero=float(np.nanmin(cent));part=np.isfinite(cent)
        mass=z['contact_envelope'][round(lo/2):round(hi/2)].T.copy();mass/=np.maximum(mass.max(1,keepdims=True),1e-20);mass[~part]=np.nan
        ax=f.add_subplot(grid[line,0]);ax.imshow(mass[idx],extent=[lo-zero,hi-zero,14.5,-.5],aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        for group in [np.arange(4),np.arange(4,15)]:ax.plot(cent[idx][group]-zero,group,'o-',color='cyan',ms=3,lw=.8)
        contacts(ax);ax.set(xlabel='Time from earliest centroid (ms)',xlim=(-90,180),title=f'Event {event}: contact envelope')
        # Missing event support remains grey outside the measured window.
        ax.set_facecolor('#bbb')
        chosen=[]
        for col,offset in enumerate([-20,0,40,80],1):
            ax=f.add_subplot(grid[line,col]);frame=int(round((zero+offset)/2-.5));actual=(frame+.5)*2
            image=ax.imshow(z['sheet_activity_counts'][frame],origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=VMAX,interpolation='nearest')
            geometry(ax,d);ax.set_title(f'{offset:+d} ms',fontsize=12)
            if col==1:ax.set_ylabel('y (mm)')
            chosen.append(dict(requested_offset_ms=offset,actual_offset_ms=actual-zero,frame_index=frame))
        SELECTION.append(dict(native_key=key,event=int(event),window_ms=[lo,hi],zero_ms=zero,snapshots=chosen,
            selection='chronological eligible events; no TA/TB or similarity filtering',page=number))
    cb=f.add_axes([.94,.29,.010,.40]);f.colorbar(image,cax=cb,label='E spikes / 2 ms / 1 mm²')
    f.suptitle('Native SNN — '+title(row),fontsize=15,y=.98)
    f.text(.5,.932 if len(ids)==2 else .89,'Consecutive eligible events, pooled across TA/TB; white cores, cyan contacts',ha='center',fontsize=11)
    name=f'04_native_{key}_snapshots_{number:02d}'
    fig.save(f,name,'同一原生SNN中按时间选取的相邻有效事件，左为逐触点归一化发放密度包络，右为四个真实2 ms全体E发放场；所有页面共用绝对发放色标。触点保持固定杆序，未参与行灰色，质心线不跨杆。',
        '事件未按TA/TB筛选，显示选择不参与统计；传播快照只属于此原生SNN参数对照，不属于未复现的降阶高背景态。')

def metrics_figure():
    f,axes=plt.subplots(1,3,figsize=(12.2,4.3));f.subplots_adjust(left=.07,right=.99,bottom=.20,top=.78,wspace=.32)
    for ax,key,title_ in zip(axes,['rank_error','within_shaft_order_error','participation_error'],['Mean normalized rank','Within-shaft order probability','Contact participation']):
        vals=[r[key] for r in SUMMARY]
        ax.bar(range(3),vals,color=['#597ea6','#8c6d9f','#569c80'],width=.58)
        for i,v in enumerate(vals):ax.text(i,v+.014,f'{v:.3f}',ha='center',fontsize=12)
        ax.set(xticks=range(3),xticklabels=['a: same J','b: same J','c/d: same J'],ylim=(0,.36),ylabel='Absolute error vs patient FIT ↓',title=title_)
        ax.tick_params(axis='x',labelsize=9)
    f.suptitle('Parameter-matched native controls: three pooled propagation summaries',fontsize=13,y=.97)
    f.text(.5,.055,'N = 28 / 30 / 30 valid events; c and d share one native control, not two observed attractors',ha='center',fontsize=10)
    fig.save(f,'05_native_three_metrics','三个参数对照的三项误差：平均rank、同杆共同参与条件下的先后概率、每个触点的参与概率。先在每杆内平均误差，再对SCL/ICL等权；所有有效事件合并，未分TA/TB作主图。','这些是同参数原生运行的值，不是四条降阶分支各自的条件传播指标；28/30/30个事件不等于独立网络重复。')

def all_events():
    f,axes=plt.subplots(3,1,figsize=(15,10));f.subplots_adjust(left=.10,right=.91,bottom=.07,top=.89,hspace=.35)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#ddd')
    for ax,row in zip(axes,SUMMARY):
        key=row['native_key'];d=RUNS[key];ids=DETAIL[key]['all_event_indices_in_window'];z=d['z']
        mat=z['centroid_ms'][ids][:,order(d)]
        rel=mat-np.nanmin(mat,axis=1,keepdims=True)
        im=ax.imshow(rel.T,aspect='auto',cmap=cmap,vmin=0,vmax=120,interpolation='nearest')
        contacts(ax);ax.set_title(title(row)+f'    valid {row["N_valid"]} / detected {len(ids)}',loc='left',fontsize=11)
        xs=np.arange(len(ids));ax.set(xticks=xs[::3],xticklabels=[ids[i] for i in xs[::3]],xlabel='Detected event index (chronological)')
        invalid=[j for j,e in enumerate(ids) if e not in DETAIL[key]['valid_event_indices']]
        ax.scatter(invalid,[-.85]*len(invalid),marker='x',color='#bd3937',s=30,clip_on=False)
    cb=f.add_axes([.934,.30,.013,.36]);f.colorbar(im,cax=cb,label='Contact centroid delay (ms)')
    f.suptitle('Every detected event in 2–12 s; all propagation modes pooled',fontsize=15)
    f.text(.5,.927,'Red ×: excluded by frozen event rules; grey: contact not participating',ha='center',fontsize=11)
    fig.save(f,'06_all_events_contact_sequence','2–12秒内全部检测事件按真实时间排列，不按传播模式重新分组。每列减去该事件最早有效质心；红叉标出冻结规则排除的重叠/持续事件，灰格表示未参与。','全量图与事件清单保留不利检测结果；三项指标只用符合原合同的有效事件，不能从几个截图推断整体恢复。')

def raw_metrics():
    target=PROV['patient_target'];f,axes=plt.subplots(2,3,figsize=(16,9));f.subplots_adjust(left=.07,right=.98,bottom=.14,top=.86,wspace=.32,hspace=.45)
    cols=['black','#597ea6','#8c6d9f','#569c80'];labels=['Patient FIT','Native J(a)','Native J(b)','Native J(c/d)']
    summaries=[target]+[DETAIL[k]['summary'] for k in ('a','b','cd')]
    native_names=PROV['contact_names'];ix=[native_names.index(n) for n in NAMES]
    for col,(key,title_) in enumerate([('mean_rank','Mean normalized rank'),('participation','Participation probability')]):
        ax=axes[0,col]
        for q,color,label in zip(summaries,cols,labels):
            y=np.array(q['mean_rank'])[ix] if key=='mean_rank' else np.array([q['contacts'][n]['participation'] for n in NAMES])
            for inds in [np.arange(4),np.arange(4,15)]:ax.plot(inds,y[inds],'o-',ms=3,lw=1,color=color,label=label if inds[0]==0 else None)
        ax.axvline(3.5,color='#999',ls=':',lw=.7);ax.set(ylim=(-.04,1.04),xticks=range(15),xticklabels=NAMES,title=title_,ylabel='Value')
        ax.tick_params(axis='x',rotation=65,labelsize=8)
    # Show all 6 SCL pairs and 55 ICL pairs separately; no hidden support fill.
    for row,shaft in [(0,'SCL'),(1,'ICL')]:
        ax=axes[row,2] if row==0 else axes[1,0]
        pairs=[k for k,v in target['pairs'].items() if v['shaft']==shaft]
        for q,color,label in zip(summaries,cols,labels):ax.plot(range(len(pairs)),[q['pairs'][k]['order_probability'] for k in pairs],'.-',ms=3,lw=.8,color=color,label=label)
        ax.set(ylim=(-.04,1.04),xticks=list(range(6)) if shaft=='SCL' else list(range(0,55,10)),title=f'{shaft}: all {len(pairs)} within-shaft pairs',ylabel='P(second contact later)',xlabel='Fixed contact-pair index')
    for ax in axes[1,1:]:ax.axis('off')
    axes[1,1].legend(handles=[Line2D([],[],color=c,label=l) for c,l in zip(cols,labels)],loc='upper left',frameon=False,fontsize=12)
    axes[1,2].text(0,.9,'Same frozen definitions\n\nAbsent contact → missing rank\nPair order → joint participation\nSCL and ICL → equal weight',va='top',fontsize=11,linespacing=1.6)
    f.suptitle('Underlying contact and pair summaries behind the three errors',fontsize=15)
    fig.save(f,'07_raw_contact_metrics','提供三项标量背后的15触点平均rank、参与概率以及SCL全部6对和ICL全部55对的先后概率。黑线为同一冻结患者FIT参照，三个模型参数共用顺序与定义。','合并摘要误差降低并不自动代表两类传播分布都恢复；共同参与的事件数在metric_provenance中逐对导出。')

def movies():
    # Full 4–6 s windows, chronological, neither event extraction nor TA/TB selection.
    palette=(plt.get_cmap('magma')(np.linspace(0,1,254))[:,:3]*255).astype(np.uint8)
    palette=np.vstack([palette,np.array([[255,255,255],[0,255,255]],np.uint8)])
    for key,d in RUNS.items():
        frames=[];z=d['z'];field=z['sheet_activity_counts'];c=d['physics']['candidate'];xy=z['contact_xy_mm']
        for frame in range(2000,3000,2):
            ar=np.rint(np.clip(field[frame]/VMAX,0,1)*253).astype(np.uint8)[::-1]
            im=Image.fromarray(ar,'P').resize((360,360),resample=Image.Resampling.NEAREST);im.putpalette(palette.ravel().tolist())
            draw=ImageDraw.Draw(im)
            for center,radius in zip(c['centers_mm'],c['radii_mm']):
                x,y=center[0]*18,(20-center[1])*18;r=radius*18;draw.ellipse([x-r,y-r,x+r,y+r],outline=254,width=2)
            for x,y in xy:
                x,y=x*18,(20-y)*18;draw.ellipse([x-2,y-2,x+2,y+2],outline=255,width=1)
            draw.text((6,5),f'Native {key}: {(frame+.5)*2/1000:.3f} s',fill=254)
            frames.append(im)
        name=f'08_native_{key}_4to6s.gif';frames[0].save(FIG/name,save_all=True,append_images=frames[1:],duration=40,loop=0,optimize=False,disposal=2)
        fig.MANIFEST.append(dict(name=name[:-4],pixels=[360,360],caption='同参数原生SNN在4–6秒的完整二维活动动画，按真实时间顺序每4 ms采样一张2 ms计数场，以10倍慢速播放；白线为core，青点为全部虚拟触点。',focus='全体E活动，不按TA/TB筛选；没有把六群体率在空间插值。c/d只共用一个同参数对照。',format='gif'))
    (OUT/'spatial_movie_metadata.json').write_text(json.dumps(dict(window_ms=[4000,6000],source_bin_ms=2,frame_stride_ms=4,playback_frame_ms=40,color_limits=[0,VMAX],units='E spikes / 2 ms / 1 mm2',no_mode_filter=True),indent=2)+'\n')

def main():
    overview();metrics_figure();all_events();raw_metrics()
    for key in RUNS:
        # Every eligible event whose earliest centroid is in the common 4–6 s
        # display window. This choice is independent of labels and matches GIFs.
        cent=RUNS[key]['z']['centroid_ms']
        ids=[i for i in DETAIL[key]['valid_event_indices'] if 4000<=np.nanmin(cent[i])<6000]
        for page in range((len(ids)+1)//2):event_figure(key,ids[2*page:2*page+2],page+1)
    movies()
    (OUT/'snapshot_selection.json').write_text(json.dumps(dict(events=SELECTION,field_vmax=VMAX,all_events_unfiltered_by_mode=True),indent=2)+'\n')
    fig.manifest()
    # GIF entries use their true extension in the generated README.
    p=FIG/'README.md';txt=p.read_text()
    for key in RUNS:txt=txt.replace(f'08_native_{key}_4to6s.png',f'08_native_{key}_4to6s.gif')
    p.write_text(txt)
    print('SPATIAL_FIGURES_COMPLETE',len(fig.MANIFEST),flush=True)

if __name__=='__main__':main()

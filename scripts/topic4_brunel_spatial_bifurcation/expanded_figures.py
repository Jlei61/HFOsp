"""Reference-style bifurcation composite with matched native spatial/readout columns."""
from common import *
from model import SpatialBrunel
from expanded_readouts import DEST,OLD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from src.snn_contact_display import CONTACT_ORDER,SHAFT_COLORS,contact_indices

F=DEST/'figures'
COL=['#2267ac','#b04da7','#222222']

def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=3)

def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(F/f'{name}.{ext}',dpi=200,bbox_inches='tight')

def trajectories():
    summary=read(DEST/'readouts/result.json')
    records=[]
    for row in summary['rows']:
        rec=read(row['source']);rec['data']=dict(np.load(rec['trajectory']));rec['envelopes']=dict(np.load(Path(row['source']).parent/'envelopes.npz'))
        rec['smooth_rate']=gaussian_filter1d(rec['data']['regional_rates_hz'],5,axis=0)
        records.append(rec)
    return summary,records

def choose_window(rec):
    ob=rec['firing'];ids=ob['primary_event_indices'];events=ob['observation']['events']
    eligible=[i for i in ids if events[i]['window_ms'][0]>=1500 and events[i]['window_ms'][1]<=8500]
    if eligible:
        # Fixed display rule: the eligible event nearest the middle of the run.
        i=min(eligible,key=lambda i:abs(np.mean(events[i]['window_ms'])-5000));lo,hi=events[i]['window_ms']
        aa=int(lo);bb=int(hi);r=rec['smooth_rate'];peaks=[aa+int(np.argmax(r[aa:bb,k])) for k in range(2)]
        active=[p for k,p in enumerate(peaks) if r[p,k]>10]
        anchor=min(active)+1 if active else int((lo+hi)/2)
        return dict(anchor_ms=anchor,event_index=int(i),window_ms=[anchor-50,anchor+150],kind='isolated contact event',core_peaks_ms=peaks)
    # Rest/continuous activity are shown as fixed windows; never synthesize an event.
    return dict(anchor_ms=5000,event_index=None,window_ms=[4950,5150],kind='fixed observation window',core_peaks_ms=None)

def branch_data():
    pieces=[]
    q=read(DEST/'stationary/result.json')['rows'];q=[x for x in q if x['high_rate_cores']=='low'];q=sorted(q,key=lambda x:x['J_EE_core'])
    pieces.append(q)
    for path in [OUT/'g20/arclength_v2/result.json',DEST/'low_arclength/result.json',DEST/'low_arclength_v2/result.json',DEST/'high_arclength_v2/result.json']:
        q=read(path)['rows']
        if path.parent.name=='low_arclength':q=q[:782]
        pieces.append(q)
    folds=[read(OUT/'g20/fold/result.json')]+[r for r in read(DEST/'folds/result.json')['rows'] if 'error' not in r]
    return pieces,folds

def draw_branch(ax,k,records,selected,zoom=False):
    pieces,folds=branch_data();h=[read(OUT/f'g20/hopf_{c}_calibrated_full/result.json') for c in 'AB']
    for piece in pieces:
        xx=np.array([p['J_EE_core'] for p in piece]);yy=np.array([p['rates_hz'][k] for p in piece])
        yy[xx>1.3]=np.nan
        ax.plot(xx,yy,color=COL[k],lw=1.4,ls=':',alpha=.85)
    track=read(DEST/'persistent_mode/result.json')
    if track['status']=='COMPLETE' and track['positive_at_all_sampled_points']:
        ax.plot([q['J_EE_core'] for q in track['rows']],[q['rates_hz'][k] for q in track['rows']],color=COL[k],ls='--',lw=1.6)
    low=read(OUT/'g20/figure_branch.json')['rows']
    low=[q for q in low if q['J_EE_core']<=h[0]['J_EE_core']]
    earlier=[p for p in pieces[0] if p['J_EE_core']<.78];low=earlier+low
    ax.plot([q['J_EE_core'] for q in low],[q['rates_hz'][k] for q in low],color=COL[k],lw=1.8)
    original=read(OUT/'g20/figure_branch.json')['rows'];original=[q for q in original if q['J_EE_core']>=h[0]['J_EE_core']]
    ax.plot([q['J_EE_core'] for q in original],[q['rates_hz'][k] for q in original],color=COL[k],lw=1.6,ls='--')
    for i,q in enumerate(h):
        ax.scatter(q['J_EE_core'],q['rates_hz'][k],s=24,c=['#1b6bb5','#b34a9d'][i],zorder=5)
        if zoom:ax.annotate(f'H{i+1}',(q['J_EE_core'],q['rates_hz'][k]),xytext=(-18,18+16*i),textcoords='offset points',fontsize=9)
    for q in folds:
        ax.plot(q['J_EE_core'],q['rates_hz'][k],'o',ms=3,mfc='white',mec='#222222',mew=.8)
    if not zoom:
        for q,label,off in [(folds[0],'Fold',(-38,-18)),(max(folds,key=lambda q:q['J_EE_core']),'Fold 1.201',(18,-24)),
                (read(DEST/'folds/low_arclength_v2_1/result.json'),'Fold 1.094',(18,9))]:
            ax.annotate(label,(q['J_EE_core'],q['rates_hz'][k]),xytext=off,textcoords='offset points',fontsize=9,
                arrowprops=dict(arrowstyle='-',lw=.6,color='#333333'))
        for _,rec in selected:
            if '_high' in rec['tag']:continue
            y=rec['smooth_rate'][500:,k];ax.errorbar(rec['J_EE_core'],y.mean(),yerr=[[y.mean()-y.min()],[y.max()-y.mean()]],
                fmt='o',color='#d98200',ms=3.5,lw=.7,capsize=2,alpha=.75,zorder=4)
        for letter,rec in selected:
            y=rec['smooth_rate'][500:,k].mean();x=rec['J_EE_core']
            if '_high' in rec['tag']:
                values=rec['smooth_rate'][500:,k]
                ax.errorbar(x,y,yerr=[[y-values.min()],[values.max()-y]],fmt='s',color='#d98200',ms=4,lw=.7,capsize=2)
            ax.annotate(letter,(x,y),xytext=(5,4),textcoords='offset points',color='#ad2c22',fontweight='bold',fontsize=12)
        for J in [1.3,1.6,2.]:
            q=read(DEST/f'modes/upper_J{J:g}/result.json')
            state=np.load(DEST/f'modes/upper_J{J:g}/modes.npz')
            if q['roots']:
                y=_spectral_model().regional_rates(state['rates'])[k]
                ax.plot(J,y,marker='x',color='#333333',ms=5,mew=1.2,ls='',zorder=5)
        ax.set(xlim=(.38,2.035),ylim=(0,550),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} E rate (Hz / cell)',title=f'Core {"AB"[k]}')
        ax.set_yscale('symlog',linthresh=1,linscale=.6);ax.set_yticks([0,1,10,100,400]);ax.set_yticklabels(['0','1','10','100','400'])
        ax.set_xticks([.4,.8,1.2,1.6,2.]);style(ax)
    else:
        ax.set(xlim=(.94,1.065),ylim=(.3,16));ax.set_yscale('log');ax.set_yticks([1,10]);ax.set_yticklabels(['1','10']);ax.set_xticks([.95,1,1.05]);ax.tick_params(labelsize=8)
        ax.set_title('Onset and stationary folds',fontsize=9,pad=3)

def waveform(ax,rec,selection,show_xlabel=False):
    x=rec['data']['time_ms']/1000;r=rec['smooth_rate']
    for k in range(2):ax.plot(x,r[:,k],color=COL[k],lw=.9)
    y=(754*r[:,0]+786*r[:,1]+30460*r[:,2])/32000;ax.plot(x,y,color='black',lw=.65)
    top=2 if max(r[500:,:2].max(),1)<5 else 510
    ax.set(xlim=(.5,10),ylim=(0,top),yticks=[0,top] if top==2 else [0,250,500]);style(ax)
    a,b=np.array(selection['window_ms'])/1000;ax.axvspan(a,b,color='#cdb581',alpha=.25,lw=0)
    if show_xlabel:ax.set_xlabel('Time (s)')
    else:ax.tick_params(labelbottom=False)

def field_counts(rec,counts):
    a=rec['data']['field_E_hz']*counts[None,:]/1000
    return a.reshape(-1,20,2,20,2).sum((2,4))

def spatial(ax,field,anchor,offset,geo,labels=False):
    t=int(anchor+offset);arr=field[max(t-1,0):t+1].sum(0)
    im=ax.imshow(arr,origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=70,interpolation='nearest')
    for center in geo['centers_mm']:ax.add_patch(plt.Circle(center,float(geo['core_radius_mm']),fill=False,color='white',lw=.8))
    xy=geo['contact_xy'];ax.scatter(xy[:,0],xy[:,1],s=6,facecolors='none',edgecolors='cyan',linewidths=.5)
    ax.set(xticks=[0,20],yticks=[0,20],title=f'{offset:+g} ms')
    ax.tick_params(labelsize=8,length=2)
    if not labels:ax.tick_params(labelleft=False)
    return im

def seeg(ax,rec,selection,kind,contract,order,last=False):
    env=rec['envelopes'][kind];base=np.array(contract['baseline']);high=np.array(contract['reference_high_q995'])
    normalized=np.maximum((env-base)/(high-base),0)
    anchor=selection['anchor_ms'];times=np.arange(len(env))*2+1-anchor;take=(times>=-50)&(times<=150)
    im=ax.imshow(normalized[take][:,order].T,aspect='auto',origin='upper',extent=(-50,150,14.5,-.5),cmap='magma',vmin=0,vmax=1)
    q=rec[kind];candidates=q['primary_event_indices'];chosen=None
    if candidates:
        closest=min(candidates,key=lambda i:abs(np.mean(q['observation']['events'][i]['window_ms'])-anchor))
        if abs(np.mean(q['observation']['events'][closest]['window_ms'])-anchor)<100:chosen=closest
    if chosen is not None:
        cc=np.asarray(q['centroids_ms'][chosen],float)[order]-anchor
        for sl in [slice(0,4),slice(4,15)]:ax.plot(cc[sl],np.arange(15)[sl],'-o',color='#68e7df',ms=2,lw=.7)
    ax.set(yticks=range(15),yticklabels=CONTACT_ORDER,xticks=[0,100],ylim=(14.5,-.5));ax.tick_params(axis='y',labelsize=7,length=2)
    for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    ax.axhline(3.5,color='white',lw=.7)
    if last:ax.set_xlabel('Relative time (ms)')
    else:ax.tick_params(labelbottom=False)
    return im

_MODEL=None
def _spectral_model():
    global _MODEL
    if _MODEL is None:_MODEL=SpatialBrunel(response='calibrated_full')
    return _MODEL

def main():
    F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    summary,records=trajectories();lookup={r['J_EE_core']:r for r in records if '_high' not in r['tag']}
    target=[.6,.88,1.04,1.3,1.6]
    selected=[(chr(97+i),lookup[J]) for i,J in enumerate(target) if J in lookup]
    high=next((r for r in records if '_high' in r['tag']),None)
    if high is not None:selected.append(('f',high))
    elif 1.8 in lookup:selected.append(('f',lookup[1.8]))
    if 2. in lookup:selected.append(('g',lookup[2.]))
    geo=dict(np.load(OUT/'operators/g40/geometry.npz'));source=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    geo['contact_xy']=source['contact_xy'];geo['core_radius_mm']=source['core_radius_mm'];sizes=geo['group_size'];E=geo['population']==0
    counts=np.bincount(geo['group_cell'][E],weights=sizes[E],minlength=1600)
    order=contact_indices(summary['contact_names']);contracts={k:read(OLD/f'observer_{k}.json') for k in ['firing','current_hfo']}
    selections={r['tag']:choose_window(r) for _,r in selected};meta=[]
    with PdfPages(F/'core_bifurcation_spatial_seeg_comparison.pdf') as pdf:
        for kind in ['firing','current_hfo']:
            n=len(selected);fig=plt.figure(figsize=(26,14));outer=fig.add_gridspec(n,4,width_ratios=[1.45,1.02,1.32,.9],wspace=.32,hspace=.62,left=.045,right=.987,bottom=.075,top=.94)
            left=outer[:,0].subgridspec(2,1,hspace=.3)
            for k in range(2):
                ax=fig.add_subplot(left[k]);draw_branch(ax,k,records,selected)
                inset=ax.inset_axes([.065,.59,.35,.35]);draw_branch(inset,k,records,selected,zoom=True)
            for i,(letter,rec) in enumerate(selected):
                selection=selections[rec['tag']];ax=fig.add_subplot(outer[i,1]);waveform(ax,rec,selection,i==n-1)
                label=fr'{letter}   $J_{{\mathrm{{EE,core}}}}={rec["J_EE_core"]:g}$'+('  · high initial state' if '_high' in rec['tag'] else '')
                ax.set_title(label,loc='left',fontsize=12,pad=6)
                sub=outer[i,2].subgridspec(1,3,wspace=.12);field=field_counts(rec,counts)
                for j,offset in enumerate([-10,30,70]):
                    axis=fig.add_subplot(sub[j]);im=spatial(axis,field,selection['anchor_ms'],offset,geo,j==0)
                    if i==n-1:axis.set_xlabel('x (mm)',fontsize=9)
                axis=fig.add_subplot(outer[i,3]);im2=seeg(axis,rec,selection,kind,contracts[kind],order,i==n-1)
                if kind=='firing':meta.append(dict(label=letter,J_EE_core=rec['J_EE_core'],tag=rec['tag'],selection=selection,
                    trajectory=rec['trajectory'],exact_readout=rec['exact_readout']))
            fig.text(.217,.982,'Spatial population equilibria',ha='center',fontsize=15)
            fig.text(.487,.982,'Native network activity',ha='center',fontsize=15)
            fig.text(.716,.982,'Same-window 2D activity',ha='center',fontsize=15)
            fig.text(.925,.982,'SEEG-site firing envelope' if kind=='firing' else 'SEEG current HFO envelope',ha='center',fontsize=14)
            fig.legend(handles=[Line2D([0],[0],color=COL[k],label=name,lw=1) for k,name in enumerate(['Core A','Core B','All E'])],
                loc='upper center',bbox_to_anchor=(.487,.969),ncol=3,frameon=False,fontsize=9,handlelength=1.3,columnspacing=1)
            handles=[Line2D([0],[0],color='#333333',lw=1.5,label='Stable equilibrium'),Line2D([0],[0],color='#333333',ls='--',label='Unstable equilibrium'),
                Line2D([0],[0],color='#333333',ls=':',label='Unclassified segment'),Line2D([0],[0],color='#333333',marker='o',mfc='white',ls='',ms=4,label='Stationary fold'),
                Line2D([0],[0],color='#333333',marker='x',ls='',ms=5,label='Positive-growth mode'),
                Line2D([0],[0],color='#d98200',marker='o',ls='',label='Native mean / min-max')]
            fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.04,.005),ncol=3,frameon=False,fontsize=9)
            cax=fig.add_axes([.65,.025,.15,.009]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Spikes / 2 ms / 1 mm cell',fontsize=9);cb.ax.tick_params(labelsize=8)
            cax=fig.add_axes([.875,.025,.105,.009]);cb=fig.colorbar(im2,cax=cax,orientation='horizontal');cb.set_label('Reference-normalized envelope',fontsize=9);cb.ax.tick_params(labelsize=8)
            name='core_bifurcation_spatial_seeg_'+kind;save(fig,name);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    # Each state has a separate, legible waveform/raster and event/readout sheet.
    for letter,rec in selected:
        selection=selections[rec['tag']];fig=plt.figure(figsize=(13,8));g=fig.add_gridspec(3,4,height_ratios=[1,.7,1.6],hspace=.55,wspace=.55)
        ax=fig.add_subplot(g[0,:]);waveform(ax,rec,selection,True);ax.set_ylabel('Rate (Hz)');ax.set_title(fr'{letter}: $J_{{\mathrm{{EE,core}}}}={rec["J_EE_core"]:g}$'+(' — high initial state' if '_high' in rec['tag'] else ' — reset initial state'))
        ax=fig.add_subplot(g[1,:]);z=rec['data'];ax.scatter(z['raster_time_ms']/1000,z['raster_id'],s=2,color='black',lw=0,rasterized=True)
        ax.set(xlim=(.5,10),ylim=(0,750),yticks=[125,375,625],yticklabels=['Core A','Core B','Surround E'],xlabel='Time (s)',ylabel='Neuron sample');style(ax)
        field=field_counts(rec,counts)
        for j,offset in enumerate([-10,30,70]):
            ax=fig.add_subplot(g[2,j]);spatial(ax,field,selection['anchor_ms'],offset,geo,True);ax.set_xlabel('x (mm)')
        ax=fig.add_subplot(g[2,3]);seeg(ax,rec,selection,'firing',contracts['firing'],order,True)
        save(fig,f'state_{letter}_J{rec["J_EE_core"]:g}');plt.close(fig)
    # Parameter-wide event statistics and the original three contact observables.
    valid=[r for r in summary['rows'] if '_high' not in r['tag']];valid.sort(key=lambda r:r['J_EE_core']);js=[r['J_EE_core'] for r in valid]
    fig,axes=plt.subplots(2,3,figsize=(16,10),gridspec_kw={'height_ratios':[.85,1.4]})
    for k in range(2):
        axes[0,0].plot(js,[r['dynamics'][k]['IEI_CV'] for r in valid],'o-',color=COL[k],label=f'Core {"AB"[k]}')
        axes[0,1].plot(js,[r['dynamics'][k]['quiet_fraction'] for r in valid],'o-',color=COL[k])
    axes[0,0].set_ylabel('Event-interval CV');axes[0,0].legend(frameon=False);axes[0,1].set_ylabel('Time below 5 Hz')
    axes[0,2].plot(js,[r['firing']['N'] for r in valid],'o-',label='Firing observer',color='#555555')
    axes[0,2].plot(js,[r['current_hfo']['N'] for r in valid],'s-',label='Current HFO observer',color='#e07b20');axes[0,2].set_ylabel('Eligible contact events');axes[0,2].legend(frameon=False)
    for ax in axes[0]:ax.set_xlabel(r'$J_{\mathrm{EE,core}}$');style(ax)
    a=np.asarray([r['firing']['mean_rank'] for r in valid],float)[:,order].T
    b=np.asarray([r['firing']['participation'] for r in valid],float)[:,order].T
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    c=np.asarray([[np.asarray(r['firing']['within_shaft_order_probability'],float)[order[i],order[j]] for i,j in pairs] for r in valid]).T
    for ax,arr,title in [(axes[1,0],a,'Mean propagation rank'),(axes[1,1],c,'Within-SCL order probability'),(axes[1,2],b,'Contact participation')]:
        cmap=matplotlib.colormaps['viridis'].copy();cmap.set_bad('#cccccc');im=ax.imshow(arr,aspect='auto',vmin=0,vmax=1,cmap=cmap)
        ax.set(xticks=range(len(js)),xticklabels=[f'{x:g}' for x in js],title=title,xlabel=r'$J_{\mathrm{EE,core}}$');ax.tick_params(axis='x',rotation=60,labelsize=9)
        if arr.shape[0]==15:ax.set(yticks=range(15),yticklabels=CONTACT_ORDER);ax.axhline(3.5,color='white',lw=.8)
        else:ax.set(yticks=range(6),yticklabels=[f'{CONTACT_ORDER[i]} → {CONTACT_ORDER[j]}' for i,j in pairs])
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    fig.tight_layout();save(fig,'parameter_rhythm_and_contact_observables');plt.close(fig)
    write(DEST/'figure_metadata.json',dict(reference_pdf='/home/honglab/leijiaxin/.codex/attachments/8bd0114b-16b3-4896-9bcd-9af1f70fdedc/core_bifurcation_composite_comparison.pdf',
        layout='Core A/B bifurcation column, same-run native waveform, three raw 2D snapshots, exact contact readout',states=meta,
        left='Stationary spatial mean-field branches plus separately labeled finite-time native means/extrema; no periodic-orbit continuation implied',
        right='All three columns share the same original-neuron native trajectory and selected time window',
        readout='Page1 firing envelope; page2 actual model current proxy after 80-250Hz/Hilbert. Neither page is a patient spectrogram.',
        spectral_status='Dashed branches have a numerically confirmed positive-growth mode of the response approximation; dotted segments unclassified. Upper-branch dynamic response calibration remains extrapolated.',human_visual_acceptance=False))
    (F/'README.md').write_text('### core_bifurcation_spatial_seeg_comparison.pdf\n按用户参考图组织：左侧Core A/B空间群体固定点分支，右侧为同参数同一原始SNN轨迹的波形、二维快照与精确触点读出。两页分别显示发放包络与原电流型SEEG proxy的80–250 Hz包络；虚线表示已检出正增长模态，点线表示尚未分类的分支段；高率动态响应仍属于外推。**关注点**：固定点折叠与原生burst不是同一种观测，橙色均值/极值不是周期轨道。\n\n### core_bifurcation_spatial_seeg_firing.png\n主图第一页面，与PDF使用同一状态和绘图代码。空间列共享0–70发放/2 ms/1 mm网格色标，触点包络使用冻结参考的逐触点尺度。**关注点**：安静或持续活动使用固定时间窗，不伪造事件质心。\n\n### core_bifurcation_spatial_seeg_current_hfo.png\n主图第二页面，右侧改为实际模型电流生成的HFO包络；它不是频谱，也不以微伏解释。其余列和时间锚与第一页相同。**关注点**：发放与电流观测的可检出事件数和时序可能不同。\n\n### parameter_rhythm_and_contact_observables.png\n完整参数范围内的core间隔CV、低活动时间和有效触点事件数，以及平均rank、SCL六对先后概率和触点参与概率。统计使用冻结检测器，灰格为无可估计事件；完整ICL成对概率保存在数值结果中。**关注点**：有效事件减少不等于网络停止活动。\n\n'+''.join(f'### state_{letter}_J{r["J_EE_core"]:g}.png\n状态{letter}的独立大图，包含相同轨迹的两核/全E波形、E神经元样本raster、三个二维时间片和触点发放包络。对应合图同一时间窗。**关注点**：逐事件空间演化与两核先后，raster各区仅抽样250个神经元。\n\n' for letter,r in selected))
    print('FIGURES COMPLETE',len(selected),flush=True)

if __name__=='__main__':main()

"""All-rate composite: own equilibria, own time series, own space/contact fields."""
from rate_field import *
from run_rate_field import dynamics
from expanded_readouts import observer,smooth2,describe,OLD
from src.snn_contact_display import CONTACT_ORDER,SHAFT_COLORS,contact_indices
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm

FIGURE_BASE=RATE_OUT
F=FIGURE_BASE/'figures';COL=['#2267ac','#b04da7','#222222']
SCL_REVISION=False


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=3)


def load_cases(prefer_scl=False):
    contract=read(OLD/'observer_firing.json');names=contract['contact_names'];records=[]
    scl=np.array([n.startswith('SCL') for n in names]);model=RateField()
    settings=[('a',.934,'Before H1: low activity'),('b',.942,'H1 < J < H2: alternating intervals'),
              ('c',.95,'After H2: repeated bursts'),('d',1.3,'Regular bursts'),('e',2.,'Continuous high core activity')]
    for letter,J,title in settings:
        folder=RATE_OUT/'runs/main'/f'J{J:.7f}';meta=read(folder/'result.json');z=dict(np.load(folder/'trajectory.npz'))
        assert z['contact_names'].tolist()==names
        env=smooth2(z['contact_rate_hz'].reshape(-1,2,15).sum(1)/1000)
        ob=observer.observe(env.T,2.,contract);mu=np.asarray(ob['centroid_ms'],float).reshape(-1,15)
        ids=[i for i in ob['primary_event_indices'] if ob['events'][i]['window_ms'][0]>=2000 and ob['events'][i]['window_ms'][1]<=10000]
        eligible=[i for i in ids if 2200<=np.mean(ob['events'][i]['window_ms'])<=9700]
        scl_ids=[i for i in ids if np.isfinite(mu[i,scl]).any()]
        if eligible:
            pool=[i for i in eligible if i in scl_ids] if prefer_scl else eligible
            if not pool:pool=eligible
            selected=min(pool,key=lambda i:abs(np.mean(ob['events'][i]['window_ms'])-6000))
            finite=mu[selected][np.isfinite(mu[selected])];anchor=float(finite.min())
            selection=('SCL-participating eligible event nearest 6 s, earliest participating contact centroid'
                if prefer_scl and selected in scl_ids else 'Eligible contact event nearest 6 s, earliest participating contact centroid')
        else:
            # If there is no qualified contact event, show an actual core burst
            # if present; otherwise a fixed window. Never manufacture a centroid.
            p=meta['dynamics'][0]['peak_times_ms']+meta['dynamics'][1]['peak_times_ms']
            p=[v for v in p if 2200<=v<=9700];anchor=float(min(p,key=lambda v:abs(v-6000))) if p else 6000.
            selected=None;selection='Actual core peak nearest 6 s' if p else 'Fixed 6 s window, no eligible burst'
        row=dict(letter=letter,J_EE_core=J,title=title,data=z,envelope=env,observation=ob,centroids=mu,selected=selected,
            anchor_ms=anchor,selection_rule=selection,metrics=describe(mu[ids],names),eligible_event_indices=ids,
            dynamics=dynamics(gaussian_filter1d(z['regional_rates_hz'],5,axis=0),2000),source=str(folder/'trajectory.npz'))
        row['smooth']=gaussian_filter1d(z['regional_rates_hz'],5,axis=0)
        row['SCL_event_count']=len(scl_ids);row['selected_SCL_count']=int(np.isfinite(mu[selected,scl]).sum()) if selected is not None else 0
        if J<1.:
            eq,ok,_=model.solve(J);assert ok
            row['exact_low_equilibrium_rates_hz']=model.regional_rates(eq)
            row['exact_low_equilibrium_residual']=float(abs(model.residual(eq,J)).max())
        records.append(row)
    return records,contract


def draw_branch(ax,k,records,inset=False):
    b=np.load(RATE_OUT/'equilibrium_branch.npz');J=b['J'];y=b['regional'][:,k];hopf=read(RATE_OUT/'hopfs.json')['rows']
    # The static branch is reused only because RHS(y*)=0 has been independently
    # checked. All old temporal stability labels and old Hopf values are discarded.
    ax.plot(J,y,':',color='#777777',lw=1.1)
    h=hopf[0];first=np.flatnonzero(J>=h['J_EE_core'])[0]
    ax.plot(np.r_[J[:first],h['J_EE_core']],np.r_[y[:first],h['rates_hz'][k]],color=COL[k],lw=1.8)
    spec=read(RATE_OUT/'branch_spectrum.json')['rows'];lookup={q['index']:q for q in spec};ix=sorted(lookup)
    runs=[]
    for i,j in zip(ix[:-1],ix[1:]):
        if lookup[i]['positive'] and lookup[j]['positive']:
            if runs and runs[-1][1]==i:runs[-1][1]=j
            else:runs.append([i,j])
    # One plotting call per continuous segment preserves the dash pattern.
    # Restarting dashes at every short numerical interval looks falsely solid.
    for i,j in runs:ax.plot(J[i:j+1],y[i:j+1],'--',color=COL[k],lw=1.3)
    earliest=next((i for i in ix if lookup[i]['positive']),None)
    if earliest is not None and earliest>=first:
        ax.plot(np.r_[h['J_EE_core'],J[first:earliest+1]],np.r_[h['rates_hz'][k],y[first:earliest+1]],'--',color=COL[k],lw=1.3)
    for n,q in enumerate(hopf):
        ax.plot(q['J_EE_core'],q['rates_hz'][k],'o',color=COL[n],ms=5)
        if inset:ax.annotate(f'H{n+1}',(q['J_EE_core'],q['rates_hz'][k]),xytext=(2,12+12*n),textcoords='offset points',color=COL[n],fontsize=10)
    if inset:
        for q in records[:3]:
            exact=q['exact_low_equilibrium_rates_hz'][k];ax.plot(q['J_EE_core'],exact,'s',color='#d18100',ms=4)
            ax.annotate(q['letter'],(q['J_EE_core'],exact),xytext=(4,-13),textcoords='offset points',fontsize=10)
        ax.set(xlim=(.931,.954),ylim=(.68,.86),xticks=[.934,.942,.95]);ax.tick_params(labelsize=8)
        ax.set_title('Same rate DDE: H1 / H2',fontsize=10);style(ax);return
    folds=read(OUT/'critical_revision/fold_audit.json')['rows']
    for q in folds:ax.plot(q['J_EE_core'],q['rates_hz'][k],'o',mfc='white',mec='#333333',ms=3.5)
    for q in records:
        a=q['smooth'][2000:,k];mean=a.mean()
        if SCL_REVISION:
            ax.plot(q['J_EE_core'],mean,'s',ms=4,color='#d18100',zorder=5)
            for value,marker in [(a.min(),'v'),(a.max(),'^')]:ax.plot(q['J_EE_core'],value,marker,ms=5,mfc='white',mec='#d18100',mew=1.,zorder=4)
        else:ax.errorbar(q['J_EE_core'],mean,yerr=[[mean-a.min()],[a.max()-mean]],fmt='s',ms=4,color='#d18100',lw=.9,capsize=3,zorder=5)
        ax.annotate(q['letter'],(q['J_EE_core'],mean),xytext=(6,3),textcoords='offset points',fontsize=11,color='#9a5200',fontweight='bold')
    ax.set(xlim=(.4,2.03),ylim=(0,550),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} rate (Hz / cell)',title=f'Core {"AB"[k]}')
    ax.set_yscale('symlog',linthresh=1,linscale=.65);ax.set_yticks([0,1,10,100,400]);ax.set_yticklabels(['0','1','10','100','400']);style(ax)


def wave(ax,q,xlim=(2,10),label=False):
    t=q['data']['time_ms']/1000
    for k in range(2):ax.plot(t,q['smooth'][:,k],color=COL[k],lw=.85)
    ax.plot(t,gaussian_filter1d(q['data']['all_E_rate_hz'],5),color=COL[2],lw=.7)
    take=(t>=xlim[0])&(t<=xlim[1]);mx=q['smooth'][take,:2].max();top=1 if mx<1 else 300 if mx<300 else 500
    ax.set(xlim=xlim,ylim=(0,top),yticks=[0,.5,1] if top==1 else [0,top/2,top]);style(ax)
    ax.axvspan((q['anchor_ms']-50)/1000,(q['anchor_ms']+150)/1000,color='#dfc593',alpha=.5,lw=0)
    if label:ax.set_xlabel('Time (s)')
    else:ax.tick_params(labelbottom=False)


def spatial(ax,q,offset,geo,labels=False):
    t=int(round(q['anchor_ms']+offset))-1;a=q['data']['field_E_hz'][max(0,t-1):t+2].mean(0).reshape(20,20)
    im=ax.imshow(a,origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,vmin=0,vmax=500),interpolation='nearest')
    for center in geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
    xy=geo['contact_xy'];ax.scatter(xy[:,0],xy[:,1],s=7,facecolors='none',edgecolors='cyan',linewidths=.6)
    ax.set(xticks=[0,20],yticks=[0,20],title=f'{offset:+g} ms');ax.tick_params(labelsize=8,length=2)
    if not labels:ax.tick_params(labelleft=False)
    return im


def contact(ax,q,contract,last=False):
    order=contact_indices(contract['contact_names']);base=np.asarray(contract['baseline']);hi=np.asarray(contract['reference_high_q995'])
    env=np.maximum((q['envelope']-base)/(hi-base),0);t=np.arange(len(env))*2+1-q['anchor_ms'];take=(t>=-50)&(t<=150)
    im=ax.imshow(env[take][:,order].T,aspect='auto',origin='upper',extent=(-50,150,14.5,-.5),cmap='magma',vmin=0,vmax=1)
    if q['selected'] is not None:
        mu=q['centroids'][q['selected']][order]-q['anchor_ms']
        for sl in [slice(0,4),slice(4,15)]:ax.plot(mu[sl],np.arange(15)[sl],'-o',color='#64e6e1',ms=2,lw=.7)
    ax.axhline(3.5,color='white',lw=.6);ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xticks=[0,100]);ax.tick_params(axis='y',length=2,labelsize=7)
    for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    if last:ax.set_xlabel('Relative time (ms)')
    else:ax.tick_params(labelbottom=False)
    return im


def main():
    F.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    records,contract=load_cases(SCL_REVISION);s=RateField();geo=dict(s.geo)
    geo['contact_xy']=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')['contact_xy']
    fig=plt.figure(figsize=(25,12.5));g=fig.add_gridspec(5,4,width_ratios=[1.3,1.02,1.25,1.0],left=.05,right=.985,top=.91,bottom=.11,wspace=.33,hspace=.64)
    left=g[:,0].subgridspec(2,1,hspace=.27)
    for k in [0,1]:
        ax=fig.add_subplot(left[k]);draw_branch(ax,k,records);ins=ax.inset_axes([.55,.12,.43,.33]);draw_branch(ins,k,records,True)
    for i,q in enumerate(records):
        ax=fig.add_subplot(g[i,1]);wave(ax,q,label=i==4)
        ax.set_title(f'{q["letter"]}   {q["title"]}\n'+fr'$J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:g}$',loc='left',fontsize=11,pad=5)
        sg=g[i,2].subgridspec(1,3,wspace=.10)
        for j,offset in enumerate([-20,40,120] if SCL_REVISION else [-20,20,70]):
            ax=fig.add_subplot(sg[j]);im=spatial(ax,q,offset,geo,j==0)
            if i==4:ax.set_xlabel('x (mm)',fontsize=9)
        ax=fig.add_subplot(g[i,3]);im2=contact(ax,q,contract,i==4)
        if SCL_REVISION:ax.set_title(f'Single event: SCL {q["selected_SCL_count"]}/4' if q['selected'] is not None else 'Fixed low-activity window',fontsize=10)
    for x,title in zip([.19,.451,.700,.900],['Rate-model equilibrium branches','Rate-model activity','Rate-model 2D field','SEEG-site rate envelope']):fig.text(x,.982,title,ha='center',fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color=COL[k],label=n) for k,n in enumerate(['Core A','Core B','All E'])],loc='upper center',bbox_to_anchor=(.455,.966),ncol=3,frameon=False,fontsize=9)
    handles=[Line2D([0],[0],color='black',label='Stable low equilibrium'),Line2D([0],[0],color='black',ls='--',label='Unstable equilibrium'),
        Line2D([0],[0],marker='o',mfc='white',mec='black',ls='',label='Stationary fold'),
        Line2D([0],[0],marker='s',color='#d18100',ls='',label='Simulated mean' if SCL_REVISION else 'Rate simulation: mean / min–max')]
    if SCL_REVISION:handles.extend([Line2D([0],[0],marker=m,mfc='white',mec='#d18100',ls='',label=l) for m,l in [('v','Simulated minimum'),('^','Simulated maximum')]])
    if any(not q['positive'] for q in read(RATE_OUT/'branch_spectrum.json')['rows']):handles.append(Line2D([0],[0],color='#777777',ls=':',label='Stability unresolved'))
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.04,.002),ncol=2,frameon=False,fontsize=9)
    cb=fig.colorbar(im,cax=fig.add_axes([.635,.048,.14,.012]),orientation='horizontal',ticks=[0,10,50,100,250,500]);cb.set_label('E rate (Hz / cell)',fontsize=10)
    cb=fig.colorbar(im2,cax=fig.add_axes([.835,.048,.135,.012]),orientation='horizontal');cb.set_label('Fixed-reference normalized envelope',fontsize=9)
    save(fig,'spatial_rate_only_bifurcation_composite')
    # Larger single-case sheets: a spatial rate heatmap replaces spike raster.
    for q in records:
        fig=plt.figure(figsize=(15,9));g=fig.add_gridspec(3,4,height_ratios=[1,1.3,1.1],hspace=.45,wspace=.43,left=.07,right=.97,bottom=.08,top=.91)
        ax=fig.add_subplot(g[0,:]);wave(ax,q,label=True);ax.set_ylabel('Rate (Hz / cell)')
        ax=fig.add_subplot(g[1,:]);im=ax.imshow(q['data']['field_E_hz'].T,aspect='auto',origin='lower',extent=(0,10,0,400),cmap='inferno',norm=PowerNorm(.55,0,500))
        ax.set(xlim=(2,10),xlabel='Time (s)',ylabel='Spatial cell index');fig.colorbar(im,ax=ax,pad=.01,label='E rate (Hz / cell)')
        for j,offset in enumerate([-20,40,120] if SCL_REVISION else [-20,20,70]):spatial(fig.add_subplot(g[2,j]),q,offset,geo,True)
        contact(fig.add_subplot(g[2,3]),q,contract,True)
        fig.suptitle(f'{q["letter"]}   {q["title"]}, J={q["J_EE_core"]:g} — autonomous spatial rate model',fontsize=15)
        save(fig,f'rate_case_{q["letter"]}')
    # Three original observables retain their contact/pair structure, rather
    # than being replaced by one uninformative averaged scalar per observable.
    fig,axes=plt.subplots(5,3,figsize=(17,14),gridspec_kw={'width_ratios':[1,1.2,1]},layout='constrained');order=contact_indices(contract['contact_names'])
    for i,q in enumerate(records):
        a,b,c=axes[i];m=q['metrics'];a.set_title(f'{q["letter"]}: J={q["J_EE_core"]:g}, n={m["N"]} events',loc='left',fontsize=11)
        if m['N']:
            for sl in [slice(0,4),slice(4,15)]:a.plot(np.arange(15)[sl],np.array(m['mean_rank'])[order][sl],'-o',ms=3)
            v=np.array(m['within_shaft_order_probability'])[np.ix_(order,order)];im=b.imshow(v,vmin=0,vmax=1,cmap='coolwarm',interpolation='nearest')
            c.bar(np.arange(15),np.array(m['participation'])[order],color=[SHAFT_COLORS[n[:3]] for n in CONTACT_ORDER])
        else:
            for ax in [a,b,c]:ax.text(.5,.5,'No eligible isolated events',ha='center',va='center',transform=ax.transAxes,fontsize=10)
        for ax in [a,c]:ax.set(ylim=(0,1),xticks=np.arange(15),xticklabels=CONTACT_ORDER);ax.tick_params(axis='x',rotation=65,labelsize=7);style(ax)
        b.set(xlim=(-.5,14.5),ylim=(14.5,-.5),aspect='equal',xticks=[0,4,14],xticklabels=['SCL9','ICL11','ICL1'],
              yticks=[0,3,4,14],yticklabels=['SCL9','SCL6','ICL11','ICL1']);b.tick_params(labelsize=8)
        b.tick_params(axis='x',rotation=45)
        a.set_ylabel('Normalized mean rank');c.set_ylabel('Participation probability')
    axes[0,1].set_title('Within-shaft P(column later than row)');fig.colorbar(im,ax=axes[:,1],fraction=.025,pad=.025)
    save(fig,'rate_only_propagation_metrics')
    # Show both observed lead orders from the same middle-case rate trajectory.
    # These labels are core peak order, not imported patient TA/TB classes.
    q=records[1];a,c=[np.array(v['peak_times_ms']) for v in q['dynamics'][:2]]
    delta=a[:,None]-c[None,:];pairs=[]
    for i in range(len(a)):
        j=int(np.argmin(abs(delta[i])))
        if abs(delta[i,j])<=150 and np.argmin(abs(delta[:,j]))==i:pairs.append((int(a[i]),int(c[j])))
    direction_examples=[];fig=plt.figure(figsize=(19,7.5));gg=fig.add_gridspec(2,3,width_ratios=[1,1.7,1],left=.055,right=.98,bottom=.09,top=.87,wspace=.3,hspace=.48)
    for i,sign in enumerate([1,-1]):
        pair=min([p for p in pairs if (p[1]-p[0])*sign>0],key=lambda p:abs(min(p)-6000));ex=q.copy();ex['anchor_ms']=float(min(pair));ex['selected']=None
        candidates=q['eligible_event_indices']
        if candidates:
            event=min(candidates,key=lambda n:abs(np.nanmean(q['centroids'][n])-min(pair)))
            if abs(np.nanmean(q['centroids'][event])-min(pair))<150:ex['selected']=event
        ax=fig.add_subplot(gg[i,0]);wave(ax,ex,xlim=((min(pair)-100)/1000,(min(pair)+220)/1000),label=True)
        ax.set_title('Core A first' if sign==1 else 'Core B first',loc='left');ax.set_ylabel('Rate (Hz / cell)')
        sub=gg[i,1].subgridspec(1,4,wspace=.1)
        for j,off in enumerate([-40,0,40,80]):spatial(fig.add_subplot(sub[j]),ex,off,geo,j==0)
        contact(fig.add_subplot(gg[i,2]),ex,contract,True)
        direction_examples.append(dict(lead_core='A' if sign==1 else 'B',core_peak_times_ms=pair,anchor_ms=ex['anchor_ms'],contact_event=ex['selected']))
    fig.suptitle('Two propagation examples from the same autonomous rate trajectory, J=0.942',fontsize=15)
    fig.legend(handles=[Line2D([0],[0],color=COL[k],label=n) for k,n in enumerate(['Core A','Core B','All E'])],
        loc='upper left',bbox_to_anchor=(.055,.935),ncol=3,frameon=False,fontsize=10)
    save(fig,'rate_two_direction_examples')
    metadata=[]
    for q in records:
        metadata.append({k:q[k] for k in ['letter','J_EE_core','title','source','anchor_ms','selection_rule','metrics','dynamics','eligible_event_indices',
            'SCL_event_count','selected_SCL_count','exact_low_equilibrium_rates_hz','exact_low_equilibrium_residual'] if k in q})
    write(FIGURE_BASE/'figure_metadata.json',dict(states=metadata,model='All panels from the same explicit spatial rate DDE',native_trajectory_panels=0,
        display_selection='Prefer SCL-participating eligible events where available; descriptive conditional examples, no change in event statistics' if SCL_REVISION else 'Eligible event nearest 6 s',
        simulation_markers='Separate mean/minimum/maximum markers at exact J; no vertical range bars' if SCL_REVISION else 'Mean and minimum-to-maximum vertical range',
        two_direction_examples=direction_examples,
        contact_observer_source=str(OLD/'observer_firing.json'),
        rate_units='Hz per neuron represented by each population; continuous expected rate, not a spike train',
        snapshots='Same trajectory and same event window as the contact readout. PowerNorm gamma=.55, shared 0–500 Hz scale.',
        simulation_initialization='All-zero rates and recurrent states; constant external moments; deterministic; discard first 2 s for summaries',
        spectrum='Recomputed for rate_field.py; 183 sampled post-H1 equilibria have positive-root certificates. This does not enumerate all secondary bifurcations. Orange ranges are simulation extrema, not Floquet-certified cycles.',
        native_equivalence='NOT_VALIDATED',human_visual_acceptance=False))
    text='### spatial_rate_only_bifurcation_composite.png\n全部动态列来自同一个二维rate方程；左侧固定点沿用同一静态方程，Hopf与稳定性按新动态方程重新计算。右侧依次为自主rate波形、同窗二维活动、相同触点几何上的发放率包络。**关注点**：虚线表示固定点不稳定；橙色为有限时长仿真的均值与极值，不能将其当作完成了周期轨道延拓或SNN等价验证。\n\n'
    if SCL_REVISION:text=text.replace('右侧依次为自主rate波形、同窗二维活动、相同触点几何上的发放率包络。','右侧优先选取有SCL参与的单次合格事件，时间序列和总体统计保持原始数据；空心上下三角与方块分别表示仿真极值和均值。')
    for q in records:text+=f'### rate_case_{q["letter"]}.png\n展示J={q["J_EE_core"]:g}的自主rate波形、400空间格的活动时空图、二维快照和触点包络。时空图中的每行是一个空间格，并非神经元spike raster。**关注点**：核对同一时间窗的活动位置和触点响应；没有合格事件时不伪造参与质心。\n\n'
    text+='### rate_only_propagation_metrics.png\n对五个rate状态展示平均传播rank、杆内共同参与触点的先后概率、触点参与概率，保留原15触点及两杆结构。使用固定参考检测合同；无合格孤立事件的状态显示不可估计。**关注点**：这是rate模型自身的事件统计，不是患者拟合得分，也不是与SNN等价的误差。\n'
    text+='\n### rate_two_direction_examples.png\n从同一J=0.942的rate轨迹中各取一个Core A先、Core B先的事件，展示相同时间窗的波形、四幅二维快照和触点包络。选择依据为150 ms内相互最近的两核burst峰值，并各取最接近6秒的事件。**关注点**：核心先后标签不是患者TA/TB分类，是否恢复原SNN的方向内时序还需独立比较。\n'
    if (F/'rate_alternating_intervals.png').exists():text+='\n### rate_alternating_intervals.png\n把J=0.942的运行延长到30秒，展示后段波形、事件间隔序列及相邻间隔返回图。两个核心均趋近长短间隔交替，完整群体活动约785 ms重复。**关注点**：CV非零不能据此认定irregular或混沌；此图也未证明这个模式由倍周期分岔产生。\n'
    (F/'README.md').write_text(text)
    print('FIGURES',F,[(q['J_EE_core'],q['metrics']['N']) for q in records],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scl-events',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();SCL_REVISION=args.scl_events
    if args.output:FIGURE_BASE=args.output
    elif SCL_REVISION:FIGURE_BASE=RATE_OUT/'scl_event_revision'
    F=FIGURE_BASE/'figures';main()

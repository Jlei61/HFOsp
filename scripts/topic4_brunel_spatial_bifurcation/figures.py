"""Current spatial equilibrium bifurcation and matched native-network examples."""
from common import *
from model import SpatialBrunel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d

COL=['#2267ac','#d66c26','#333333']

def save(fig,name):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def decorate(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=4)

def main():
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':15,'legend.fontsize':12,
        'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':1.1})
    s=SpatialBrunel(response='calibrated_full');dest=OUT/'figures';dest.mkdir(exist_ok=True)
    h=[read(OUT/f'g20/hopf_{c}_calibrated_full/result.json') for c in ['A','B']];fold=read(OUT/'g20/fold/result.json')
    branch=OUT/'g20/figure_branch.json'
    if not branch.exists():
        rr=None;points=[]
        for J in np.unique(np.r_[np.linspace(.78,1.,45),[x['J_EE_core'] for x in h]]):
            rr,ok,_=s.solve(float(J),rr);assert ok;points.append(dict(J_EE_core=J,rates_hz=s.regional_rates(rr)))
        points.extend(read(OUT/'g20/arclength_v2/result.json')['rows'][1:]);write(branch,dict(rows=points))
    points=read(branch)['rows'];js=np.array([x['J_EE_core'] for x in points]);rates=np.array([x['rates_hz'] for x in points])
    fig=plt.figure(figsize=(12,11));grid=fig.add_gridspec(3,6,height_ratios=[1.15,1.,1.1],hspace=.5,wspace=.75)
    axrate=[fig.add_subplot(grid[0,:3]),fig.add_subplot(grid[0,3:])]
    for k,ax in enumerate(axrate):
        stable=js<h[0]['J_EE_core'];first=np.flatnonzero(~stable)[0]
        ax.plot(js[:first+1],rates[:first+1,k],color=COL[k],lw=2.4)
        ax.plot(js[first:],rates[first:,k],color=COL[k],lw=2.4,ls='--')
        ax.set(xlim=(.78,1.025),ylim=(.4,2.1),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Equilibrium rate (Hz)',title=f'Core {"AB"[k]}')
        ax.set_yscale('log');ax.set_yticks([.4,.6,1.,1.5,2.]);ax.set_yticklabels(['0.4','0.6','1','1.5','2'])
        ax.minorticks_off();decorate(ax)
        for i,hh in enumerate(h):
            x=hh['J_EE_core'];y=hh['rates_hz'][k];ax.scatter(x,y,s=50,c=COL[i],zorder=5)
            ax.annotate(f'H{i+1}',(x,y),xytext=(-35,35 if i==0 else 65),textcoords='offset points',color=COL[i],
                arrowprops=dict(arrowstyle='-',color=COL[i],lw=1),fontsize=13)
        ax.scatter(fold['J_EE_core'],fold['rates_hz'][k],marker='s',s=48,c='black',zorder=5)
        ax.annotate('Fold',(fold['J_EE_core'],fold['rates_hz'][k]),xytext=(-35,-35 if k==0 else 15),textcoords='offset points',fontsize=13)
        ax.text(-.15,1.16,'AB'[k],transform=ax.transAxes,fontsize=21,fontweight='bold')
    fig.legend(handles=[Line2D([0],[0],color='black',lw=2,label='Stable equilibrium'),Line2D([0],[0],color='black',lw=2,ls='--',label='Unstable equilibrium')],
        loc='upper center',bbox_to_anchor=(.5,.98),ncol=2,frameon=False)
    growth=fig.add_subplot(grid[1,:3]);freq=fig.add_subplot(grid[1,3:])
    trace=read(OUT/'g20/mode_trace_calibrated_full/result.json')['rows']
    for k,core in enumerate(['A','B']):
        data=[]
        for p in trace:
            candidates=[q for q in p['roots'] if q['core']==core and q['lambda_per_ms'][1]>.001]
            if candidates:
                q=max(candidates,key=lambda x:x['lambda_per_ms'][0]);data.append([p['J_EE_core'],*q['lambda_per_ms']])
        data.append([h[k]['J_EE_core'],0,2*np.pi*h[k]['frequency_hz']/1000]);data=np.array(sorted(data))
        growth.plot(data[:,0],data[:,1]*1000,color=COL[k],lw=2.3,label=f'Core {core} mode')
        freq.plot(data[:,0],data[:,2]*1000/(2*np.pi),color=COL[k],lw=2.3)
        growth.scatter(h[k]['J_EE_core'],0,color=COL[k],s=42);growth.annotate(f'H{k+1}',(h[k]['J_EE_core'],0),xytext=(-15,15+k*15),textcoords='offset points',color=COL[k])
    growth.axhline(0,color='black',lw=1);growth.set(xlim=(.88,1),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=r'Growth $\mathrm{Re}\,\lambda$ (s$^{-1}$)')
    freq.set(xlim=(.88,1),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Mode frequency (Hz)')
    growth.legend(frameon=False,loc='upper left');decorate(growth);decorate(freq)
    for letter,ax in zip('CD',[growth,freq]):ax.text(-.15,1.14,letter,transform=ax.transAxes,fontsize=21,fontweight='bold')
    contact=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')['contact_xy']
    for k,(name,source) in enumerate([('H1',OUT/'g20/hopf_A_calibrated_full/critical.npz'),('H2',OUT/'g20/hopf_B_calibrated_full/critical.npz'),('Fold',OUT/'g20/fold/fold.npz')]):
        ax=fig.add_subplot(grid[2,2*k:2*k+2]);data=np.load(source);v=data['vector'] if 'vector' in data else data['right']
        size=s.geo['group_size'];cells=s.geo['group_cell'];amp=np.bincount(cells[s.E],weights=size[s.E]*abs(v[s.E]),minlength=400)/np.maximum(np.bincount(cells[s.E],weights=size[s.E],minlength=400),1)
        amp/=amp.max();im=ax.imshow(amp.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
        for idx,center in enumerate(s.geo['centers_mm']):
            ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1.2));ax.text(center[0],center[1]+2.2,'AB'[idx],ha='center',color='white',fontsize=12)
        ax.scatter(contact[:,0],contact[:,1],s=9,facecolors='none',edgecolors='cyan',linewidths=.5)
        ax.set(xlabel='x (mm)',ylabel='y (mm)' if k==0 else '',xticks=[0,10,20],yticks=[0,10,20]);ax.set_title(name)
        ax.text(-.2,1.08,'EFG'[k],transform=ax.transAxes,fontsize=21,fontweight='bold')
    cb=fig.colorbar(im,ax=fig.axes[-3:],fraction=.025,pad=.025);cb.set_label('Relative mode amplitude');cb.set_ticks([0,.5,1])
    save(fig,'spatial_bifurcation_onset')

    paths=sorted((OUT/'native_private_only').glob('J*/trajectory.npz'))
    composite=plt.figure(figsize=(13,12));outer=composite.add_gridspec(4,1,hspace=.32)
    for k,path in enumerate(paths):
        data=np.load(path);contract=read(path.parent/'contract.json');J=contract['J_EE_core'];x=data['time_ms']/1000
        axes=[];sub=outer[k].subgridspec(2,1,height_ratios=[1,.8],hspace=.08)
        a=composite.add_subplot(sub[0]);b=composite.add_subplot(sub[1],sharex=a)
        for i in (0,1):a.plot(x,gaussian_filter1d(data['regional_rates_hz'][:,i],5),color=COL[i],lw=1.1,label=f'Core {"AB"[i]}')
        allE=(754*data['regional_rates_hz'][:,0]+786*data['regional_rates_hz'][:,1]+30460*data['regional_rates_hz'][:,2])/32000
        a.plot(x,gaussian_filter1d(allE,5),color='black',lw=.9,label='All E')
        a.set(xlim=(.5,5),ylim=(-5,390),ylabel='Rate (Hz)');a.set_yticks([0,200]);a.tick_params(labelbottom=False);decorate(a)
        a.text(.01,1.01,fr'$J_{{\mathrm{{EE,core}}}}={J:g}$',transform=a.transAxes,fontsize=15,va='bottom')
        if k==0:a.legend(loc='upper right',ncol=3,frameon=False)
        b.scatter(data['raster_time_ms']/1000,data['raster_id'],s=2.3,color='black',linewidths=0,rasterized=True)
        b.set(ylim=(-5,750),yticks=[125,375,625],yticklabels=['A','B','Surround']);b.axhline(250,color='#999999',lw=.6);b.axhline(500,color='#999999',lw=.6)
        b.set_ylabel('E raster');decorate(b)
        if k==3:b.set_xlabel('Time (s)')
        else:b.tick_params(labelbottom=False)
        a.text(-.1,1.03,'abcd'[k],transform=a.transAxes,fontsize=20,fontweight='bold')
        single,aa=plt.subplots(2,1,figsize=(11,5),sharex=True,gridspec_kw={'height_ratios':[1,.9],'hspace':.1})
        for i in (0,1):aa[0].plot(x,gaussian_filter1d(data['regional_rates_hz'][:,i],5),color=COL[i],lw=1.25,label=f'Core {"AB"[i]}')
        aa[0].plot(x,gaussian_filter1d(allE,5),color='black',lw=1,label='All E');aa[0].legend(ncol=3,frameon=False,loc='upper right')
        aa[0].set(ylabel='Rate (Hz)',ylim=(-5,390),title=fr'$J_{{\mathrm{{EE,core}}}}={J:g}$')
        aa[1].scatter(data['raster_time_ms']/1000,data['raster_id'],s=4,color='black',linewidths=0,rasterized=True)
        aa[1].set(xlim=(.5,5),ylim=(-5,750),yticks=[125,375,625],yticklabels=['Core A','Core B','Surround E'],xlabel='Time (s)',ylabel='Neuron sample')
        for axis in aa:decorate(axis)
        save(single,f'native_J{J:g}_rate_raster')
    save(composite,'native_bifurcation_neighborhood')
    (dest/'README.md').write_text('### spatial_bifurcation_onset.png\n展示当前拓扑6101在400空间单元、935率群体上的稳态分支、校正后的动态特征根和空间临界模。稳定性属于整个空间固定点，因此A、B两条投影从第一个Hopf起同时用虚线；未计算的周期轨道不画。**关注点**：H1、H2是两个不同的局部振荡失稳模；Fold是已不稳定分支的消失，不能当作首次burst阈值。\n\n### native_bifurcation_neighborhood.png\n四个相邻核内EE倍率下原始40000神经元SNN的Core A/B、全E放电率与E神经元样本raster。固定Z=1、保留原始M与private Poisson、共享OU波动为零，与线性分析工作点一致；每条件只有一条5秒轨迹。**关注点**：理论边界两侧都可能出现有限噪声触发事件，跨过Hopf不等于立即形成严格规则burst。\n\n'+''.join(f'### native_J{J:g}_rate_raster.png\n对应核内EE倍率{J:g}的独立大图，与合图使用同一原始轨迹。raster固定抽样Core A、Core B及surround E各250个神经元。**关注点**：观察两核先后、事件之间的安静区间和间隔变化；样本点不代表全部神经元。\n\n' for J in [.88,.94,.96,1.]))
    write(OUT/'figure_metadata.json',dict(primary_grid=20,space_cells=400,rate_groups=935,response='calibrated_full',
        topology=6101,parameter='J_EE_core scales both within-A and within-B EE edges',Z='fixed 1',M='original dynamic',
        native_parameter_conditions=[.88,.94,.96,1.],layout='equilibrium branches, temporal spectrum, spatial eigenmodes; separate native traces',
        scientific_status='Linear spatial mean-field analysis; full nonlinear rate/SNN propagation equivalence and periodic continuation remain unvalidated',human_visual_acceptance=False))

if __name__=='__main__':main()

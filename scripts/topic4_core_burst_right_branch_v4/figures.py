"""Extended, single-axis bifurcation figures from verified numerical orbits.

Separate families are never joined across an unresolved branch connection.
Native state markers retain their original parameter and mean-rate coordinates.
"""
from common import *
import sys,csv
sys.path.insert(0,str(ROOT/'scripts/topic4_core_bifurcation_states_v3'))
import plot as p
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import resample,find_peaks

FIG=OUT/'figures'
MEAN=r'$\langle r_{\mathrm{E,core\ A}}\rangle$'

def catalog():
    old=p.load_cycles();chosen={}
    for c in old:
        c.update(family='low_burst',classification='stable');chosen[(c['family'],c['g'])]=c
    excluded=[]
    for path in (OUT/'periodic').rglob('g*_N2048.npz'):
        if path.parent.name=='refined':continue
        z=np.load(path);g=float(z['g']);r=z['r'];T=float(z['T'])
        relative=path.parent.relative_to(OUT/'periodic')
        flpath=OUT/relative/'floquet'/f'g{g:.8f}_dt0.1.json'
        if not flpath.exists():
            excluded.append(dict(source=str(path),reason='Floquet not yet evaluated'));continue
        fl=read(flpath)
        if fl['classification']!='stable':
            excluded.append(dict(source=str(path),reason=fl['classification'],floquet=fl));continue
        family=path.parent.name if path.parent.name in ('tonic','mixed') else ('low_burst' if g<1.1765 else 'recruited_burst')
        rr=resample(r,8192,axis=0)
        c=dict(g=g,r=r,T=T,N=len(r),path=str(path),mean=r.mean(0)*1000,lo=rr.min(0)*1000,hi=rr.max(0)*1000,
               max_transverse=fl['maximum_transverse_modulus'],phase_error=fl['phase_error'],family=family,classification='stable')
        chosen[(family,g)]=c
    cycles=sorted(chosen.values(),key=lambda c:(c['family'],c['g']))
    write('excluded_from_stable_curves.json',excluded)
    return cycles

def families(cycles):
    return [[c for c in cycles if c['family']==name] for name in ('low_burst','recruited_burst','mixed','tonic')]

def periodic_lines(ax,cycles,shade=True):
    for seq in families(cycles):
        if not seq:continue
        xx=[c['g'] for c in seq];lo=np.maximum(0,[c['lo'][0] for c in seq]);hi=[c['hi'][0] for c in seq]
        if shade:ax.fill_between(xx,lo,hi,color=p.GREEN,alpha=.065)
        ax.plot(xx,hi,'o-',color=p.GREEN,ms=3.8,lw=1.8)
        ax.plot(xx,lo,'o-',color=p.GREEN,ms=3.8,lw=1.8)
        ax.plot(xx,[c['mean'][0] for c in seq],'o-',color=p.ORANGE,ms=4,lw=2.8,zorder=5)
        for c in (seq[0],seq[-1]):
            ax.plot(c['g'],c['mean'][0],'o',mfc='white',mec=p.ORANGE,ms=5.5,mew=1.4,zorder=6)

def main_axis(ax,cycles,native,fold,arc):
    # Reuse the original equilibrium producer and its stability semantics.
    p.branches(ax,[],fold,arc)
    periodic_lines(ax,cycles)
    for row in native:p.number_marker(ax,row['g'],row['mean'],row['number'],row['color'],size=270)
    ax.set(xlim=(.46,1.65),ylim=(-.05,510))
    ax.set_xticks([.5,.7,.85,1.,1.15,1.3,1.45,1.6]);ax.set_xticklabels(['0.5','0.7','0.85','1.0','1.15','1.3','1.45','1.6'])
    ax.set_yticks([0,.5,1,3,10,30,100,300,500]);ax.set_yticklabels(['0','0.5','1','3','10','30','100','300','500'])
    ax.legend(handles=[Line2D([],[],color=p.BLUE,label='Stable equilibrium'),
        Line2D([],[],color=p.RED,ls='--',label='Unstable equilibrium'),
        Line2D([],[],color=p.GREEN,label='Periodic maximum / minimum'),
        Line2D([],[],color=p.ORANGE,lw=2.8,label='Period mean  '+MEAN),
        Line2D([],[],color='black',marker='o',ls='none',label='Native SNN mean (1–4)')],
        frameon=False,loc='upper left',fontsize=11.5,labelspacing=.6)
    ax.set_title('Core burst branch continuation',loc='left',fontsize=19,weight='bold',pad=15)
    assert not ax.child_axes

def main_figures(cycles,native,fold,arc,book):
    fig,ax=plt.subplots(figsize=(9.5,8.4));fig.subplots_adjust(left=.12,right=.965,bottom=.12,top=.91)
    main_axis(ax,cycles,native,fold,arc);assert len(fig.axes)==1
    p.save(fig,'00_extended_single_axis_bifurcation',
        '在一个坐标轴中把核内EE倍率延伸至1.6；橙线为已求解周期轨道的完整周期均值，绿色为峰谷包络。沿用原平衡分支和真实位置的四个原生SNN编号点；纵轴在1 Hz以下为线性、以上为对数。',
        '仅画Floquet已核验稳定的周期解；不同轨道族之间断线，空心端点表示本次已核验范围，不能当成新的分岔点。',book)
    fig=plt.figure(figsize=(12.8,12.3));ax=fig.add_axes([.085,.525,.89,.39])
    main_axis(ax,cycles,native,fold,arc);p.native_panel(fig,native,box=(.085,.09,.89,.285),title_y=.439)
    p.save(fig,'01_extended_bifurcation_four_native_states',
        '上方使用与独立图完全相同的单坐标轴延拓结果；下方保留四个原生SNN状态的波形和30细胞raster。编号1–4在主图和例图保持一致。',
        '右侧新曲线属于联合降阶系统；下方原生状态并未随新曲线重新归类，图中没有内嵌窗或灰色说明小字。',book)

def right_linear(cycles,book):
    fig,ax=plt.subplots(figsize=(9.2,8));fig.subplots_adjust(left=.12,right=.96,bottom=.12,top=.90)
    periodic_lines(ax,cycles)
    ax.set(xlim=(1.12,1.63),ylim=(-7,450),xlabel=p.JLABEL,ylabel='Core A E rate (Hz / neuron)')
    ax.set_xticks([1.15,1.2,1.3,1.4,1.5,1.6]);ax.set_yticks([0,100,200,300,400])
    ax.set_title('Period mean and oscillation envelope',loc='left',fontsize=19,weight='bold',pad=15)
    ax.legend(handles=[Line2D([],[],color=p.ORANGE,lw=2.8,label='Period mean  '+MEAN),Line2D([],[],color=p.GREEN,label='Periodic maximum / minimum')],
              loc='upper left',frameon=False,fontsize=12)
    p.save(fig,'02_period_mean_linear_rate',
        '单独以线性率轴查看右侧完整周期均值和峰谷包络，使高率分支的均值上升与振幅缩小能够直接比较。横轴仍为实际核内EE倍率，均值不是任意有限时间窗的平均。',
        '1.4以后核心谷值保持在很高水平，应描述为高率背景上的周期振荡；本图没有把分支间缺口补成已知连接。',book)

def diagnostics(cycles,book):
    fig,axs=plt.subplots(2,1,figsize=(9.3,8.7),sharex=True);fig.subplots_adjust(left=.13,right=.96,bottom=.11,top=.89,hspace=.3)
    for seq in families(cycles):
        if not seq:continue
        xx=[c['g'] for c in seq]
        axs[0].plot(xx,[c['T'] for c in seq],'o-',ms=4,color=p.BLUE)
        mm=np.array([c['max_transverse'] for c in seq]);below=mm<1e-10
        axs[1].plot(xx,np.where(below,np.nan,mm),'o-',ms=4,color=p.BLUE)
        if below.any():axs[1].scatter(np.array(xx)[below],np.full(below.sum(),1.6e-10),marker='v',s=32,color=p.BLUE)
    axs[0].set(ylabel='Full-network period (ms)',ylim=(0,2450))
    axs[1].axhline(1,color='black',ls='--',lw=1.2);axs[1].set(yscale='log',ylim=(1e-10,3),xlim=(1.12,1.63),ylabel=r'Max transverse $|\mu|$',xlabel=p.JLABEL)
    axs[1].legend(handles=[Line2D([],[],marker='v',color=p.BLUE,ls='none',label=r'$|\mu| < 10^{-10}$')],loc='lower right',frameon=False,fontsize=12)
    axs[1].set_xticks([1.15,1.2,1.3,1.4,1.5,1.6]);axs[1].text(1.125,1.15,'Stability boundary',fontsize=11,va='bottom')
    fig.text(.13,.947,'Period and Floquet stability',fontsize=19,weight='bold')
    p.save(fig,'03_period_and_floquet_stability',
        '上方给出联合六群体周期解的完整周期，下方给出排除自治相位乘子后的最大横向Floquet乘子模。所有展示点均通过相位乘子接近1的数值检查，横向模小于1支持周期轨道稳定。',
        '低于1e-10的值以下三角表示，不解释浮点噪声底附近的精确数值；单个A核周期内多峰不等于倍周期分岔。',book)

def waveforms(cycles,book):
    selected=[next(c for c in cycles if c['g']==g) for g in (1.15,1.18,1.3,1.37,1.4,1.6)]
    fig,axs=plt.subplots(6,1,figsize=(10.5,12.3));fig.subplots_adjust(left=.12,right=.96,bottom=.065,top=.89,hspace=.78)
    for ax,c in zip(axs,selected):
        r=resample(c['r'],4096,axis=0)*1000;T=c['T']
        # One common phase shift keeps complete core bursts inside the window.
        # Never shift A, B and surround independently.
        r=np.roll(r,round(.2*len(r))-np.argmax(r[:,0]),axis=0)
        rr=np.r_[r,r,r[:1]];tt=np.arange(len(rr))*T/len(r)
        for i,color,label in ((0,p.BLUE,'Core A E'),(1,'#8059a3','Core B E'),(2,p.GREEN,'Surround E')):
            ax.plot(tt,rr[:,i],color=color,label=label,lw=1.65)
        ax.axhline(c['mean'][0],color=p.ORANGE,ls='--',lw=1.5,label=MEAN)
        ax.set(xlim=(0,2*T),ylim=(-8,450),yticks=[0,200,400],ylabel='Rate (Hz)',xlabel='Time (ms)')
        ax.set_xticks([0,T,2*T]);ax.set_xticklabels(['0',f'{T:.1f}',f'{2*T:.1f}'])
        ax.set_title(f'{p.JLABEL} = {c["g"]:g}',loc='left',fontsize=13,pad=5)
    fig.text(.12,.959,'Two periods of the jointly solved network',fontsize=19,weight='bold')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.112,.938),ncol=4,frameon=False,fontsize=12)
    p.save(fig,'04_joint_core_periodic_waveforms',
        '每行展示同一联合六群体周期解的两个完整周期，A核、B核及周边E群体使用同一时间坐标。橙色虚线为A核完整周期均值；每行只做共同循环移位，没有把不同群体各自移相。',
        '1.3时A核出现周期内双峰；1.37时B核处于高率背景而A核仍回落；1.4和1.6时两核谷值均升高。所有曲线是确定性降阶率，不是新SNN spike raster。',book)

def core_means(cycles,book):
    fig,ax=plt.subplots(figsize=(9.2,8));fig.subplots_adjust(left=.12,right=.96,bottom=.12,top=.90)
    for seq in families(cycles):
        if not seq:continue
        for i,color in ((0,p.BLUE),(1,'#8059a3'),(2,p.GREEN)):
            ax.plot([c['g'] for c in seq],[c['mean'][i] for c in seq],'o-',color=color,lw=2,ms=4)
    ax.set(xlim=(1.12,1.63),ylim=(-7,450),xlabel=p.JLABEL,ylabel='Period mean rate (Hz / neuron)')
    ax.set_xticks([1.15,1.2,1.3,1.4,1.5,1.6]);ax.set_yticks([0,100,200,300,400])
    ax.set_title('Core A, core B and surround',loc='left',fontsize=19,weight='bold',pad=15)
    ax.legend(handles=[Line2D([],[],color=p.BLUE,label=r'$\langle r_{\mathrm{A,E}}\rangle$'),Line2D([],[],color='#8059a3',label=r'$\langle r_{\mathrm{B,E}}\rangle$'),Line2D([],[],color=p.GREEN,label=r'$\langle r_{\mathrm{surround,E}}\rangle$')],loc='upper left',frameon=False,fontsize=15)
    p.save(fig,'05_joint_core_period_means',
        '比较同一联合系统周期解的A核E、B核E及周边E完整周期均值。不同轨道族分别连线，因此同一参数可以有多个已求解状态。',
        '两核同时接受相同EE倍率缩放，但其具体连接和阈值分布并非镜像；混合状态中不能用A核均率代表B核。',book)

def main():
    cycles=catalog();native=p.load_native();fold=read(V2/'fold.json');arc=read(V2/'equilibrium_spectrum.json')
    p.OUT=OUT;p.FIG=FIG;FIG.mkdir(exist_ok=True);p.MANIFEST.clear();p.DESCRIPTIONS.clear()
    with PdfPages(FIG/'extended_core_bifurcation.pdf') as book:
        main_figures(cycles,native,fold,arc,book);right_linear(cycles,book);diagnostics(cycles,book);waveforms(cycles,book);core_means(cycles,book)
    meta=[]
    for c in cycles:
        row={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items() if k!='r'}
        # Count only the middle copy so periodic wrap-around peaks are included.
        pk=find_peaks(np.tile(c['r'][:,0],3),prominence=np.ptp(c['r'][:,0])*.12,distance=max(1,len(c['r'])//100))[0]
        row['A_peaks_per_period']=int(np.sum((pk>=len(c['r']))&(pk<2*len(c['r']))));meta.append(row)
    write('displayed_periodic_orbits.json',meta);write('figure_manifest.json',p.MANIFEST)
    flat=[dict(J_EE_core=c['g'],family=c['family'],period_ms=c['T'],A_mean_hz=c['mean'][0],A_min_hz=c['lo'][0],A_max_hz=c['hi'][0],B_mean_hz=c['mean'][1],surround_mean_hz=c['mean'][2],max_transverse=c['max_transverse'],source=c['path']) for c in cycles]
    with (OUT/'periodic_observables.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    (FIG/'README.md').write_text('# 核内EE右侧周期分支延拓\n\n'+'\n'.join(p.DESCRIPTIONS)+'\n### extended_core_bifurcation.pdf\n六页图册，包含独立分岔图、下配原生四状态的合图、线性均值图、周期与Floquet图、联合周期波形及两核均值对照。原始周期解与数值验证位于上级目录。\n**关注点**：当前为待用户目视检查的候选版本，尚未宣称通过人工验收。\n')
    print('FIGURES_COMPLETE',len(cycles),json.dumps(p.MANIFEST),flush=True)

if __name__=='__main__':main()

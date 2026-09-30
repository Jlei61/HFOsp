"""Spatial eigenvectors and complete, masked contact-order diagnostics."""
from common import *
from expanded_figures import F,save,style,COL
from model import SpatialBrunel
from src.snn_contact_display import CONTACT_ORDER,SHAFT_COLORS,contact_indices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=OUT/'expanded'

def mode_map(ax,s,v,title,contacts,radius):
    size=s.geo['group_size'];cell=s.geo['group_cell'];mask=s.E
    numerator=np.bincount(cell[mask],weights=size[mask]*abs(v[mask])**2,minlength=400)
    denominator=np.bincount(cell[mask],weights=size[mask],minlength=400)
    amp=np.sqrt(numerator/np.maximum(denominator,1));amp/=amp.max()
    im=ax.imshow(amp.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=1,cmap='viridis')
    for k,xy in enumerate(s.geo['centers_mm']):
        ax.add_patch(plt.Circle(xy,radius,fill=False,color='white',lw=.9))
        ax.text(xy[0],xy[1]+2.2,'AB'[k],color='white',ha='center',fontsize=10)
    ax.scatter(contacts[:,0],contacts[:,1],s=7,facecolors='none',edgecolors='cyan',linewidths=.5)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20],title=title)
    return im

def main():
    F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    s=SpatialBrunel(response='calibrated_full')
    native_geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    contacts=native_geo['contact_xy'];radius=float(native_geo['core_radius_mm'])
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained');sources=[]
    settings=[('H1',OUT/'g20/hopf_A_calibrated_full','critical.npz'),('H2',OUT/'g20/hopf_B_calibrated_full','critical.npz'),
        ('Fold',BASE/'folds/low_arclength_v2_0','fold.npz'),('Fold',BASE/'folds/low_arclength_v2_1','fold.npz')]
    for ax,(name,folder,file) in zip(axes[0],settings):
        data=np.load(folder/file);result=read(folder/'result.json');v=data['vector'] if 'vector' in data else data['right']
        title=f'{name}: J={float(data["J"]):.4f}'
        if 'frequency_hz' in result:title+=f'\n{result["frequency_hz"]:.2f} Hz at zero growth'
        else:title+='\nStationary zero mode'
        im=mode_map(ax,s,v,title,contacts,radius);sources.append(dict(name=name,source=str(folder/file)))
    for ax,J in zip(axes[1],[1.3,1.6,2.]):
        folder=BASE/f'modes/upper_J{J:g}';data=np.load(folder/'modes.npz');i=int(np.argmax(data['roots'].real))
        lam=data['roots'][i];mode_map(ax,s,data['vectors'][i],f'Upper branch: J={J:g}\nGrowth {lam.real*1000:.1f} /s; {lam.imag*1000/(2*np.pi):.1f} Hz',contacts,radius)
        sources.append(dict(name=f'upper_J{J:g}',source=str(folder/'modes.npz'),vector_index=i))
    ax=axes[1,3]
    for color,J in zip(['#e69f00','#009e73','#8040a3'],[1.3,1.6,2.]):
        q=np.load(BASE/f'modes/upper_J{J:g}/modes.npz');rr=q['roots']
        ax.scatter(rr.real*1000,rr.imag*1000/(2*np.pi),s=25,color=color,label=f'J={J:g}')
    ax.axvline(0,color='black',lw=.8);ax.set(xlabel=r'Growth Re $\lambda$ (s$^{-1}$)',ylabel='Frequency (Hz)',title='Refined growing modes')
    ax.legend(frameon=False);style(ax)
    fig.colorbar(im,ax=list(axes.flat[:7]),shrink=.5,pad=.01,label='Relative E-component RMS amplitude')
    save(fig,'spatial_critical_and_upper_eigenmodes');plt.close(fig)
    write(BASE/'eigenmode_figure_metadata.json',dict(sources=sources,spatial_observable='Cellwise neuron-weighted RMS of the E part of the eigenvector, normalized to its own maximum',
        meaning='Infinitesimal modes of the population response approximation, not finite-amplitude propagation frames. High-rate local response calibration is extrapolated.'))

    trace=read(BASE/'persistent_mode/result.json')
    fig,aa=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,ax in enumerate(aa):
        vals=[q['lambda_per_ms'][0]*1000 if k==0 else q['frequency_hz'] for q in trace['rows']]
        ax.plot([q['J_EE_core'] for q in trace['rows']],vals,'o-',ms=3,color='#666666',label='Same tracked mode')
        for J in [1.3,1.6,2.]:
            q=np.load(BASE/f'modes/upper_J{J:g}/modes.npz');v=q['roots'][np.argmax(q['roots'].real)]
            ax.plot(J,v.real*1000 if k==0 else v.imag*1000/(2*np.pi),'s',color='#bd5534',ms=5,label='Largest growth among extracted modes' if J==1.3 else None)
        ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=r'Growth Re $\lambda$ (s$^{-1}$)' if k==0 else 'Frequency (Hz)');style(ax)
    aa[0].axhline(0,color='black',lw=.7);aa[0].set_ylim(-5,145);aa[1].legend(frameon=False,fontsize=9)
    save(fig,'upper_branch_mode_continuation');plt.close(fig)

    summary=read(BASE/'readouts/result.json');records=summary['rows'];order=contact_indices(summary['contact_names'])
    records=sorted(records,key=lambda q:('_high' in q['tag'],q['J_EE_core']))
    pairs=[(i,j) for lo,hi in [(0,4),(4,15)] for i in range(lo,hi) for j in range(i+1,hi)]
    fig,axes=plt.subplots(1,2,figsize=(15,16),layout='constrained')
    cmap=matplotlib.colormaps['viridis'].copy();cmap.set_bad('#cccccc')
    for ax,kind,title in zip(axes,['firing','current_hfo'],['Firing observer','Current HFO observer']):
        matrix=np.array([[np.array(q[kind]['within_shaft_order_probability'],float)[order[i],order[j]] for i,j in pairs] for q in records]).T
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=1)
        labels=[f'{q["J_EE_core"]:g}'+(' high' if '_high' in q['tag'] else '')+f'\nn={q[kind]["N"]}' for q in records]
        ax.set(xticks=range(len(records)),xticklabels=labels,yticks=range(len(pairs)),yticklabels=[f'{CONTACT_ORDER[i]} → {CONTACT_ORDER[j]}' for i,j in pairs],
            title=title,xlabel=r'$J_{\mathrm{EE,core}}$ and eligible event count')
        ax.tick_params(axis='x',labelsize=9,rotation=55);ax.tick_params(axis='y',labelsize=8)
        ax.axhline(5.5,color='white',lw=2)
        for label,(i,j) in zip(ax.get_yticklabels(),pairs):label.set_color(SHAFT_COLORS[CONTACT_ORDER[i][:3]])
    fig.colorbar(im,ax=axes,shrink=.35,pad=.01,label='P(first contact earlier | both participated)')
    save(fig,'within_rod_order_all_pairs');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    fig.suptitle(r'Changes from the $J_{\mathrm{EE,core}}=0.88$ reference trajectory',fontsize=14)
    keys=['mean_rank_difference','within_shaft_order_difference','participation_difference']
    titles=['Mean rank change','Within-rod order change','Participation change']
    reset=[q for q in records if '_high' not in q['tag']]
    for row,kind in enumerate(['firing','current_hfo']):
        for k,key in enumerate(keys):
            ax=axes[row,k]
            def metric(q):
                value=q['difference_from_J0p88'][kind]
                return value[key] if value is not None else np.nan
            ax.plot([q['J_EE_core'] for q in reset],[metric(q) for q in reset],'o-',color=COL[row],ms=4,label='Reset initial state')
            for q in records:
                if '_high' in q['tag']:ax.plot(q['J_EE_core'],metric(q),'s',color='#d98200',ms=6,label='High initial state')
            ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',title=titles[k] if row==0 else '',ylabel=('Firing' if row==0 else 'Current HFO')+'\nShaft-balanced mean absolute change')
            ax.set_xlim(.4,2.);ax.set_xticks([.4,.8,1.2,1.6,2.])
            ax.axhline(0,color='#888888',lw=.6);style(ax)
            if k==0:ax.legend(frameon=False,fontsize=9)
    save(fig,'three_contact_statistics_change_from_J0p88');plt.close(fig)
    prefix=(F/'README.md').read_text().split('\n### spatial_critical_and_upper_eigenmodes.png')[0]
    with (F/'README.md').open('w') as f:
        f.write(prefix)
        f.write('\n### spatial_critical_and_upper_eigenmodes.png\n显示H1/H2及两个较高率折点的空间零模/临界模，并显示J=1.3、1.6、2.0高率固定点的正增长模态。地图是E分量的逐格加权RMS相对振幅；特征值平面另列提取并校正的全部正增长复根。**关注点**：这些是无穷小扰动模态，不是有限burst的传播快照；高率响应滤波的外推尚待工作点验证。\n\n### upper_branch_mode_continuation.png\n沿J=1.3至2.0跟踪同一空间模态的增长率和频率；另以方块显示三个工作点中已提取模态的最大增长率。**关注点**：最大增长模态可以更换，不能把不同模态误连为同一分支。\n\n### within_rod_order_all_pairs.png\n完整显示SCL六对及ICL五十五对触点在共同参与时的先后概率，分别使用发放与电流HFO观测器。每列标出有效事件数，灰色为无可估计共同参与事件。**关注点**：固定参数下的高初态列单独展示；概率条件于事件筛选与共同参与。\n\n### three_contact_statistics_change_from_J0p88.png\n三个指标分别概括平均传播rank、杆内相对顺序与触点参与度相对于J=0.88参考轨迹的变化；两杆先各自平均，再等权汇总。两行分别使用发放和电流HFO观测器，无有效事件的条件不赋零误差。**关注点**：这是参数效应，不是患者拟合分数，也不是rate/SNN一致性验证。\n')
    print('DIAGNOSTIC FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()

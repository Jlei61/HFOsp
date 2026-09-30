"""Spatial diagnostic of finite-population versus density correspondence."""
from compare_density_spatial import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    base=OUT/'particle_controls/selected_g40'
    sources=[base/f'D0.225000_Nscale{n}_seed1901_4000ms_microscopic' for n in (1,4,16)]
    sources.append(OUT/'qualification/selected_g40/D0.225000_degree6_dv0.125_4000ms')
    data=[extract(p,.225) for p in sources]
    geo=np.load(OUT/'operators/selected_g40_theta0.25/geometry.npz');centers=geo['centers_mm']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.linewidth':1.,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(10.8,3.1),layout='constrained',sharex=True,sharey=True)
    identities=['40,000 cells','160,000 cells','640,000 cells','Density limit']
    for i,(ax,row,label) in enumerate(zip(axes,data,identities)):
        im=ax.imshow(row['image'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=500,interpolation='nearest')
        for j,(x,y) in enumerate(centers):
            ax.add_patch(Circle((x,y),1.45,fill=False,color='#20c7d2',lw=1.2))
            ax.text(x,y+1.7,'AB'[j],color='#20c7d2',ha='center',va='bottom',fontsize=10)
        ax.text(0,1.07,f'{chr(65+i)}  {label}',transform=ax.transAxes,ha='left',fontsize=12)
        ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        ax.tick_params(direction='out',length=3)
    axes[0].set_ylabel('y (mm)')
    bar=fig.colorbar(im,ax=axes,shrink=.79,pad=.025,ticks=[0,250,500]);bar.set_label('E rate (Hz)')
    stem='fig_population_limit_spatial'
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'{stem}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    rows=[]
    for row,label in zip(data,identities):
        rows.append(dict(identity=label,source=row['source'],complete_event_count=row['event_count'],
            core_B_minus_A_crossing_ms=row['core_B_minus_A_crossing_ms'],spatial_vs_density=compare(data[-1],row)))
    meta=dict(D=.225,window_ms=[1000,4000],spatial_window_ms=50,definition=data[-1]['definition'],
        individual_parameters='Particle panels retain original individual threshold, frozen Z and dynamic M',rows=rows,
        scope='Single-seed finite-window correspondence diagnostic, not a bifurcation diagram or complete equivalence acceptance',
        human_visual_acceptance='PENDING')
    (OUT/'population_limit_spatial.json').write_text(json.dumps(safe(meta),indent=2)+'\n')
    entry='\n### fig_population_limit_spatial.png / .pdf / .svg\n\n在相同 D=0.225 下，对照 4 万、16 万、64 万个保留逐细胞阈值、Z、M 的粒子网络与确定性密度模型。每格为 1–4 s 窗内完整事件在全局峰值附近 50 ms 的空间率图，再跨事件取中位数；空间格不是独立重复。该图只是单种子、有限时窗的群体极限诊断，不能代替目标分岔图或完整动力学验收。**关注点**：增加群体后空间形态接近密度极限，但事件持续过程及两核时间差仍须单独核对。\n'
    readme=folder/'README.md';old=readme.read_text() if readme.exists() else ''
    if f'### {stem}.' not in old:readme.write_text(old+entry)


if __name__=='__main__':main()

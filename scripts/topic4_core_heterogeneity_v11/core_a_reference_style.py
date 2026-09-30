"""Reference-style parameter-plane line figure; no invented phase trajectories."""
from core_a_focus_figures import pairs,DEST,OUT,ROOT,STD,MEAN,read,write
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG=DEST/'figures';NAME='core_A_reference_style'
BLUE='#3979c2';GREEN='#72a844';RED='#b93836';BLACK='#303030'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':13,
 'axes.linewidth':.85,'axes.spines.top':True,'axes.spines.right':True,
 'pdf.fonttype':42,'ps.fonttype':42,'xtick.labelsize':10,'ytick.labelsize':10})
J=r'$J_{\mathrm{EE,core}}$'

def label(ax,text,xy,xytext,color=BLACK):
    ax.annotate(text,xy,xytext=xytext,ha='center',va='center',color=color,fontsize=11,
        bbox=dict(fc='white',ec='none',pad=1.2),
        arrowprops=dict(arrowstyle='-',lw=.7,color=color,shrinkA=3,shrinkB=3),zorder=8)

def draw():
    data=read(OUT/'bifurcation_curves.json');checks=pairs()
    fig,axes=plt.subplots(1,2,figsize=(12.4,7.1))
    fig.subplots_adjust(left=.08,right=.975,bottom=.255,top=.85,wspace=.27)
    for ax,kind,letter in zip(axes,['spread','mean'],['A','B']):
        ax.text(.01,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=20)
        title='Threshold heterogeneity' if kind=='spread' else 'Mean threshold'
        ax.text(.50,1.10,title,transform=ax.transAxes,ha='center',fontsize=14)
        ax.set_xlabel(J,labelpad=8)
        curves={}
        for key,c in [('PD3',GREEN),('PD2',BLUE)]:
            if kind=='spread':
                rr=sorted([r for r in data if r['label']==key],key=lambda r:r['h'])
                yy=np.array([r['sigma_A_mV'] for r in rr])
            else:
                rr=sorted(read(OUT/'threshold_nodes'/f'{key}.json')['points'],key=lambda r:r['mean_A_mV'])
                yy=np.array([r['mean_A_mV'] for r in rr])
            xx=np.array([r['g'] for r in rr]);curves[key]=(xx,yy)
            ln,=ax.plot(xx,yy,c=c,lw=1.35,zorder=3)
            assert np.array_equal(ln.get_xdata(),xx) and np.array_equal(ln.get_ydata(),yy)
            ax.plot(xx[-1],yy[-1],marker='*',c=RED,ms=10,zorder=9)
            if kind=='mean' and key=='PD3':
                ax.plot(xx[0],yy[0],'o',mfc='white',mec=c,ms=5,zorder=5)
        if kind=='spread':
            ax.set(xlim=(1.373,1.394),ylim=(-.045,.785),
                   ylabel=r'$\sigma_A$ (mV)',xticks=[1.375,1.380,1.385,1.390],yticks=[0,.2,.4,.6])
            ax.text(.015,1.025,rf'$\bar\theta_A={MEAN:.2f}$ mV',transform=ax.transAxes,fontsize=10,va='bottom')
            # Inset lies between the periodic curves and does not obscure them.
            ins=ax.inset_axes([.34,.635,.27,.27])
            rr=sorted([r for r in data if r['label']=='SN' and r['critical_core']=='A'],key=lambda r:r['h'])
            sw=read(OUT/'threshold_nodes/AB_onset_switch.json')
            xx=[sw['g']]+[r['g'] for r in rr];yy=[sw['sigma_A_mV']]+[r['sigma_A_mV'] for r in rr]
            ins.plot(xx,yy,c=BLACK,lw=1.);ins.plot(xx[-1],yy[-1],'*',c=RED,ms=7)
            ins.plot(xx[0],yy[0],'s',mfc='white',mec=BLACK,ms=3.5)
            ins.set(xlim=(1.119,1.181),ylim=(.709,.744),xticks=[1.13,1.17],yticks=[.72,.74])
            ins.text(1.150,.735,'SN',fontsize=9)
            ins.text(.68,.10,'A/B',transform=ins.transAxes,fontsize=7,ha='right')
            label(ax,'PD3',(np.interp(.28,curves['PD3'][1],curves['PD3'][0]),.28),(1.375,.40),GREEN)
            label(ax,'PD2',(np.interp(.29,curves['PD2'][1],curves['PD2'][0]),.29),(1.3915,.38),BLUE)
            for h in [0,1]:ax.plot(1.38,h*STD,'o',color=BLACK,ms=4,zorder=6)
            pair=next(q for q in checks if q['axis']=='spread')
            ax.plot(pair['g'],pair['value']*STD,'o',color=BLACK,ms=4,zorder=6)
            label(ax,'A burst +\nhigh background\ncoexistence*',
                  (pair['g'],pair['value']*STD),(1.3825,.20),BLACK)
            yscan=.055;ylab=.025;x1,x2=1.375,1.3915
        else:
            ax.set(xlim=(1.359,1.395),ylim=(16.635,17.315),
                   ylabel=r'$\bar\theta_A$ (mV)',xticks=[1.36,1.37,1.38,1.39],yticks=[16.7,16.9,17.1,17.3])
            ax.text(.015,1.025,rf'$\sigma_A={STD:.2f}$ mV',transform=ax.transAxes,fontsize=10,va='bottom')
            ins=ax.inset_axes([.085,.685,.29,.245])
            rr=sorted(read(OUT/'threshold_nodes/mean_fold.json'),key=lambda r:r['mean_A_mV'])
            xx=[r['g'] for r in rr];yy=[r['mean_A_mV'] for r in rr]
            ins.plot(xx,yy,c=BLACK,lw=1);ins.plot(xx[-1],yy[-1],'*',c=RED,ms=7)
            ins.set(xlim=(.40,1.17),ylim=(16.47,17.31),xticks=[.5,1.0],yticks=[16.5,17.2])
            ins.text(.79,16.84,'SN',fontsize=9)
            label(ax,'PD3',(np.interp(16.975,curves['PD3'][1],curves['PD3'][0]),16.975),(1.363,16.98),GREEN)
            label(ax,'PD2',(np.interp(16.98,curves['PD2'][1],curves['PD2'][0]),16.98),(1.3922,17.08),BLUE)
            ax.plot(1.38,MEAN,'o',color=BLACK,ms=4,zorder=6)
            pair=next(q for q in checks if q['axis']=='mean')
            ax.plot(pair['g'],pair['value'],'o',color=BLACK,ms=4,zorder=6)
            label(ax,'A burst +\nhigh background\ncoexistence*',
                  (pair['g'],pair['value']),(1.380,16.81),BLACK)
            yscan=16.680;ylab=16.652;x1,x2=1.362,1.392
        ins.set_title('A resting fold',fontsize=9,pad=4)
        ins.set_xlabel(J,fontsize=8,labelpad=0);ins.tick_params(labelsize=7.5,length=2,pad=2)
        ins.set_facecolor('white')
        ax.annotate('',xy=(x2,yscan),xytext=(x1,yscan),
                    arrowprops=dict(arrowstyle='->',lw=.8,color='#777777'))
        ax.text((x1+x2)/2,ylab,'Increasing EE',ha='center',fontsize=9,color='#666666')
        ax.tick_params(direction='out',length=3.5,pad=4)
    legend=[Line2D([],[],color=GREEN,lw=1.35,label='PD3: A high-background branch'),
            Line2D([],[],color=BLUE,lw=1.35,label='PD2: A burst branch'),
            Line2D([],[],color=RED,marker='*',ls='',ms=9,label='Reference critical point')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.53,.105),ncol=3,frameon=False,fontsize=9.5)
    fig.text(.08,.063,'* Coexistence verified at black points. Arrows: parameter-scan direction. Open circle: continuation limit.',fontsize=9)
    fig.text(.08,.025,'Full A/B/surround feedback retained. The EE multiplier still acts on both cores; only A-dominated bifurcations are shown.',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{NAME}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def record():
    sources=[OUT/'bifurcation_curves.json',OUT/'threshold_nodes/mean_fold.json',
             OUT/'threshold_nodes/PD2.json',OUT/'threshold_nodes/PD3.json',DEST/'coexistence_checks.json']
    write(DEST/f'{NAME}.json',dict(change='Reference-inspired thin line, marked critical points, inset onset panels and annotated coexistence examples',
        same_equations=True,same_parameter_axes=True,new_simulations=False,
        arrow_meaning='Increasing EE parameter, not time evolution or a state-space vector field',
        region_label_scope='Coexistence at marked points, not exhaustive phase classification',
        smoothing='None; actual adjacent critical coordinates connected, no extrapolation',
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        human_acceptance='PENDING'))
    p=FIG/'README.md';text=p.read_text();heading=f'### {NAME}.pdf'
    if heading not in text:
        p.write_text(text+f'''\n{heading}

按用户参考图的白底细曲线、节点与局部窗形式重排，保留真正的参数平面：左为阈值标准差，右为平均阈值，横轴仍是原共同 EE 倍率。小窗展示 A 主导的静息 fold，主窗标出 A 的 PD2/PD3 与已验证共存点；箭头仅表示参数扫描方向。**关注点**：曲线间可以有多个稳定分支；没有将参考图的状态轨迹或 nullcline 搬到参数平面。
''')

if __name__=='__main__':draw();record()

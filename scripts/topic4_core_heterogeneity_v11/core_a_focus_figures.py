"""Keep A-dominated critical loci, with explicit meaning of overlapping branches."""
from core_a_focus import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG=DEST/'figures';FIG.mkdir(exist_ok=True)
BLUE='#296ba5';GREEN='#238a70';BLACK='#333333'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
 'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
J=r'$J_{\mathrm{EE,core}}$'

def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def pairs():
    out=[]
    for f in DEST.glob('*_burst_N2048.json'):
        other=f.with_name(f.name.replace('_burst_','_high_'))
        if not other.exists():continue
        a,b=read(f),read(other)
        assert a['stable'] and b['stable'] and abs(a['g']-b['g'])<1e-9
        assert a['rate_min_hz'][0]<5 and b['rate_min_hz'][0]>5
        out.append(dict(axis=a['axis'],value=a['value'],g=a['g'],
                        burst_source=a['source'],high_source=b['source']))
    return out

def parameter():
    data=read(OUT/'bifurcation_curves.json');switch=read(OUT/'threshold_nodes/AB_onset_switch.json')
    fig,aa=plt.subplots(2,2,figsize=(11.8,8.5));fig.subplots_adjust(left=.085,right=.965,top=.92,bottom=.17,wspace=.30,hspace=.44)
    ax=aa[0,0];r=sorted([r for r in data if r['label']=='SN' and r['critical_core']=='A'],key=lambda r:r['h'])
    x=[switch['g']]+[v['g'] for v in r];y=[switch['sigma_A_mV']]+[v['sigma_A_mV'] for v in r]
    ax.plot(x,y,c=BLACK,lw=1.3);ax.plot(*[x[0],y[0]],'D',c=BLACK,ms=4.5)
    ax.set(xlim=(1.12,1.18),ylim=(.710,.742),ylabel=r'A threshold spread $\sigma_A$ (mV)')
    ax.set_title('a   A-led rest instability',loc='left',fontweight='bold')
    ax.text(1.151,.733,'SN of A',fontsize=11)
    ax.annotate('A / B switch',xy=(x[0],y[0]),xytext=(1.15,.714),ha='center',fontsize=9,
                arrowprops=dict(arrowstyle='-',lw=.7))
    ax.text(.04,.93,rf'Mean fixed: {MEAN:.2f} mV',transform=ax.transAxes,fontsize=9)
    ax=aa[0,1];r=sorted(read(OUT/'threshold_nodes/mean_fold.json'),key=lambda r:r['mean_A_mV'])
    assert all(q['critical_core']=='A' for q in r)
    ax.plot([q['g'] for q in r],[q['mean_A_mV'] for q in r],c=BLACK,lw=1.3)
    ax.set(xlim=(.4,1.18),ylim=(16.48,17.30),ylabel=r'A mean threshold $\bar\theta_A$ (mV)')
    ax.set_title('b   A-led rest instability',loc='left',fontweight='bold')
    ax.text(.81,16.94,'SN of A',fontsize=11)
    ax.text(.04,.93,rf'Spread fixed: {STD:.2f} mV',transform=ax.transAxes,fontsize=9)
    for ax,axis in [(aa[1,0],'spread'),(aa[1,1],'mean')]:
        for key,c in [('PD3',GREEN),('PD2',BLUE)]:
            if axis=='spread':
                rr=sorted([q for q in data if q['label']==key],key=lambda r:r['h']);yy=[q['sigma_A_mV'] for q in rr]
            else:
                rr=sorted(read(OUT/'threshold_nodes'/f'{key}.json')['points'],key=lambda r:r['mean_A_mV']);yy=[q['mean_A_mV'] for q in rr]
            ax.plot([q['g'] for q in rr],yy,c=c,lw=1.4)
            if axis=='mean' and key=='PD3':ax.plot(rr[0]['g'],yy[0],'o',mfc='white',mec=c,ms=4.5)
        if axis=='spread':
            ax.set(xlim=(1.374,1.392),ylim=(-.02,.765),ylabel=r'A threshold spread $\sigma_A$ (mV)')
            ax.set_title('c   A cycle stability boundaries',loc='left',fontweight='bold')
            for h in [0,1]:ax.plot(1.38,h*STD,'D',c='#ad8129',ms=4.5,zorder=7)
            ax.text(1.375,.15,'PD3',color=GREEN,rotation=90)
            ax.text(1.389,.40,'PD2',color=BLUE,rotation=90)
        else:
            ax.set(xlim=(1.360,1.393),ylim=(16.65,17.30),ylabel=r'A mean threshold $\bar\theta_A$ (mV)')
            ax.set_title('d   A cycle stability boundaries',loc='left',fontweight='bold')
            ax.plot(1.38,MEAN,'D',c='#ad8129',ms=4.5,zorder=7)
            ax.text(1.365,17.03,'PD3',c=GREEN)
            ax.text(1.3885,16.90,'PD2',c=BLUE,rotation=90)
        for q in pairs():
            if q['axis']==axis:ax.plot(q['g'],q['value']*STD if axis=='spread' else q['value'],'D',c='#ad8129',ms=4.5,zorder=7)
    for ax in aa.flat:ax.set_xlabel(J);ax.tick_params(length=3)
    handles=[Line2D([],[],c=BLACK,label='A-led equilibrium fold'),
             Line2D([],[],c=BLUE,label='PD2: A burst parent'),Line2D([],[],c=GREEN,label='PD3: A high-background parent'),
             Line2D([],[],c='#ad8129',marker='D',ls='',ms=4,label='Two stable A states verified')]
    fig.legend(handles=handles,ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(.54,.045),fontsize=9.5)
    fig.text(.085,.015,'Full six-population feedback retained. Original shared A/B EE axis; only A-dominated bifurcations are displayed.',fontsize=9)
    save(fig,'core_A_bifurcation_parameters')

def intervals():
    cp=read(ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916/critical_points.json')
    pd2=next(r['JEE_core'] for r in cp if r['label']=='PD2');pd3=next(r['JEE_core'] for r in cp if r['label']=='PD3')
    fig,ax=plt.subplots(figsize=(10.6,3.8));fig.subplots_adjust(left=.26,right=.97,bottom=.30,top=.80)
    ax.axvspan(pd3,pd2,color='#efdfae',alpha=.60,zorder=0)
    for y,xs,c in [(1,[1.36,pd2],BLUE),(0,[pd3,1.401],GREEN)]:ax.plot(xs,[y,y],c=c,lw=5,solid_capstyle='butt')
    ax.plot([pd2,pd2],[-.4,1.25],ls=':',c=BLUE,lw=1)
    ax.plot([pd3,pd3],[-.4,1.25],ls=':',c=GREEN,lw=1)
    ax.plot(pd2,1,'^',mfc='white',mec=BLUE,ms=8);ax.plot(pd3,0,'^',mfc='white',mec=GREEN,ms=8)
    ax.text(pd3,1.37,'PD3',ha='center',c=GREEN);ax.text(pd2,1.37,'PD2',ha='center',c=BLUE)
    ax.text((pd2+pd3)/2,.48,'Both stable',ha='center',fontsize=12)
    ax.set(xlim=(1.36,1.401),ylim=(-.4,1.6),xlabel=J,yticks=[0,1],yticklabels=['A high-background\nperiodic parent','A burst\nperiodic parent'])
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=12)
    ax.set_title('What lies between PD3 and PD2?',loc='left',fontweight='bold',pad=16)
    fig.text(.05,.12,'Reference threshold setting. Bars show stability of these two periodic parents; the overlap permits both A states.',fontsize=9.5)
    fig.text(.05,.035,'This does not enumerate all attractors: stable doubled-period branches also occur near PD3 on its lower-EE side.',fontsize=9)
    save(fig,'meaning_between_A_boundaries')

def documentation():
    write(DEST/'coexistence_checks.json',pairs())
    (DEST/'report.md').write_text('''# 只展示 Core A 主导的分岔

## 口径

保留六群体、突触滤波及全部延迟，B 和 surround 动态反馈均参与求解。主图只展示临界率模态集中在 A 的解失稳，不能称为与外部无关的孤立 A 系统。此次显示筛选没有更改横轴：J_EE,core 仍同时缩放 AA、BB 的 EE；用户关于“是否只扫描 A 的 EE”的选择尚未收到，不能把现有结果改标成 J_EE,A。

显示筛选采用 A_E+A_I 占临界率模态平方范数至少 80%，坐标均为每神经元放电率，不按细胞数加权。这里 retained 的 PD2、PD3 在实际曲线各点均超过 99%，因此并非靠贴近阈值选择来保留；它们分别主要涉及 A_E 和 A_I。该占比用于定位模态，不能解释为细胞或解剖因果贡献比例。

LP1 主要在 B_E，PD1 主要在 B_I，故不作为 A 主图边界。原 onset cycle fold 是 A/B 共同模态，不能仅因为画在 A 的放电率图上就称为 A 内部分岔；LP0/PD0 的主要周边模态也不纳入。共享分岔仍可能影响 A 的可达状态，本图不因此宣称穷尽 A 的所有状态变化。

SN 只保留 A 主导段。原来小异质性下的长竖线由 B 控制，已从 A 图去掉；A/B 切换点仍标在已知区段末端，没有向 B 先失稳后的背景状态外推一个 A 的静态 fold。

## 为什么保留的周期边界仍接近竖直

这表示临界 EE 随纵轴参数变化较小，不是时间轨迹，也不表示 A 的放电率恒定。PD2、PD3 分别属于 A burst 母周期支和 A 高背景母周期支。它们在参数平面里挨着，不能因此说每个条带都是唯一状态。

原参考条件下，PD3<J<PD2 是两个已延拓 T 周期母支的稳定范围重叠：A 可以保持 burst，也可以保持高背景振荡。两个初始历史到达不同稳定解，并不矛盾。PD3 左侧还有窄的稳定 2T 子支，因此重叠区外不应一概标成只有一种吸引子。主图菱形只标已有完整周期解及谱检验的共存实例；新增点在 coexistence_checks.json 中回到两个轨道及两种步长的谱结果。

## 数值与交付

输入沿用已接受的实际分岔点；mode_audit.json/csv 逐项记录保留或移除依据。额外的两处内部参数条件从对应母支逐步延拓，分别求精确周期轨道，进行 1024/2048 周期网格与 RK4 两个步长的横向 Floquet 检验；求解失败的直接大步尝试没有作为结果。

core_A_bifurcation_parameters 是参数平面主图；meaning_between_A_boundaries 单独解释参考条件下的重叠区。未完成局部线保留空心端点。仍属六群体降阶模型，未经原生空间 SNN 或临床组织对应验证；图待用户人工检查。
''')
    (FIG/'README.md').write_text('''# Core A 主导分岔

### core_A_bifurcation_parameters.pdf

保留完整六群体反馈，只展示 A 主导的 SN、PD2、PD3；两种纵轴分别是 A 阈值标准差与均值。菱形为两个 A 状态同时稳定的已验证实例，原共用 A/B 核内 EE 横轴未改名。**关注点**：A 的临界模式，以及参数平面不等于唯一状态分区。

### meaning_between_A_boundaries.pdf

在原参考阈值设置下，分别画 A burst 母周期支与 A 高背景母周期支的稳定区间。着色只表示这两个母支的稳定范围重叠，不穷尽全部吸引子。**关注点**：两条 PD 线之间可以存在两种稳定 A 状态。

各图同时有 PNG/SVG，待人工目视确认。
''')

if __name__=='__main__':parameter();intervals();documentation()

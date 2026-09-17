"""Figure candidates: saved bifurcation branches and separately identified SNN."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse, csv, json
import numpy as np
from scipy.signal import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/topic4_sef_hfo'
OUT=BASE/'core_spatial_readout_v10_20260916';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
SEQ=read(OUT/'periodic_branches.json');EQ=read(OUT/'equilibrium_branches.json')
STATES=read(OUT/'reduced_states.json')
CP=list(csv.DictReader((BASE/'core_network_bifurcation_v7_20260916/critical_points.csv').open()))
COUNT=np.array(read(OUT/'parameter_mapping.json')['counts'])
J=r'$J_{\mathrm{EE,core}}$'
COL=dict(eq='#286aa4',mean='#c8831d',hi='#258253',lo='#329cac')
AB=['#2769a5','#9a4c8f'];NET=['#161616','#35936b','#d57635']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
    'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
MANIFEST=[]

def save(fig,name,caption,focus):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();outside=[]
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box=item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                outside.append(item.get_text())
    for ext in ('png','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=190)
    with Image.open(FIG/f'{name}.png') as im:im.load();shape=im.size
    MANIFEST.append(dict(name=name,pixels=shape,caption=caption,focus=focus,text_outside_canvas=outside))
    plt.close(fig)

def cp(name):return next(p for p in CP if p['label']==name)

def sparse(rows,group,limit=23):
    """Thin symbols in displayed (J, symlog rate) distance, not sorted J."""
    if len(rows)<=limit:return np.arange(len(rows))
    xy=np.array([[r['J_exact'],np.log10(1+max(r['hi'][group],0)),np.log10(1+max(r['lo'][group],0))] for r in rows])
    span=np.ptp(xy,axis=0);span=np.maximum(span,1e-9)
    steps=np.sqrt(np.sum((np.diff(xy,axis=0)/span)**2,axis=1));s=np.r_[0,np.cumsum(steps)]
    return np.unique(np.r_[0,np.searchsorted(s,np.linspace(0,s[-1],limit)),len(rows)-1])

def curves(ax,group,extrema=True,xoffset=0.,xfactor=1.):
    tx=lambda x:(np.asarray(x)-xoffset)*xfactor
    for branch in EQ:
        stable=branch[0]['stable']
        ax.plot(tx([r['J_exact'] for r in branch]),[r['mean'][group] for r in branch],
            color=COL['eq'],lw=1.6,ls='-' if stable else '--',zorder=4 if stable else 5)
    for rows in SEQ:
        stable=rows[0]['stable'];xx=tx([r['J_exact'] for r in rows])
        ax.plot(xx,[r['mean'][group] for r in rows],color=COL['mean'],lw=1.6,
            ls='-' if stable else '--',zorder=3 if stable else 4)
        if extrema:
            take=sparse(rows,group)
            for key in ('hi','lo'):
                ax.plot(xx[take],[max(0,rows[i][key][group]) for i in take],ls='none',marker='s',ms=2.8,
                    mfc=COL[key] if stable else 'white',mec=COL[key],mew=.6,zorder=2)

def yscale(ax,ylim=(-.03,480)):
    ax.set_yscale('symlog',linthresh=1,linscale=.65,base=10)
    ax.set(ylim=ylim,yticks=[0,1,10,100,400],yticklabels=['0','1','10','100','400'])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

def critical(ax,name,group,text,offset):
    p=cp(name);x=float(p['JEE_core']);y=float(p['AB'[group]+'_mean_hz'])
    ax.plot(x,y,marker='^' if name.startswith('PD') else 'o',ms=5.3,mfc='white',mec='#111',zorder=8)
    ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',fontsize=9,
        ha='right' if offset[0]<0 else 'left',va='center',
        arrowprops=dict(arrowstyle='-',lw=.6,color='#222'),
        bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=9)

def core_axis(ax,group):
    curves(ax,group);yscale(ax)
    ax.set(xlim=(.45,1.62),ylabel=f'Core {"AB"[group]} E (Hz / cell)',xticks=[.5,.75,1,1.25,1.5])
    critical(ax,'Low-rate equilibrium fold',group,'Fold',(-30,-10))
    critical(ax,'Cycle fold',group,'Cycle fold',(-30,11))
    critical(ax,'LP0c',group,'LP0 / PD0',(-25,22))
    offsets=([(-7,27),(-80,14),(39,-17),(-36,18)] if group==0 else
             [(-15,15),(-40,-7),(37,-31),(-40,31)])
    for label,off in zip(['LP1','PD1','PD2','PD3'],offsets):critical(ax,label,group,label,off)
    for r in STATES:
        if r['label']=='b' and group==1:offset=(-62,-29)
        else:offset={'a':(-40,-18),'b':(-53,-37),'c':(23,-33),'d':(35,12)}[r['label']]
        ax.plot(r['J_exact'],r['mean'][group],'o',ms=4,color='#a63429',zorder=10)
        ax.annotate(r['label'],(r['J_exact'],r['mean'][group]),xytext=offset,textcoords='offset points',
            color='#a63429',weight='bold',fontsize=12,
            arrowprops=dict(arrowstyle='-',color='#a63429',lw=.6),
            bbox=dict(facecolor='white',edgecolor='none',pad=.3),zorder=11)

def b_zoom(parent,bounds):
    ax=parent.inset_axes(bounds)
    for s in ax.spines.values():s.set_visible(True);s.set_linewidth(.6)
    curves(ax,1,extrema=False)
    branch=EQ[1];pts=[r for r in branch if 1.06<r['J_exact']<1.13 and r['mean'][0]<5][::2]
    ax.plot([r['J_exact'] for r in pts],[r['mean'][1] for r in pts],'o',ms=3.8,mfc='white',mec=COL['eq'],mew=.8,zorder=8)
    p=cp('Low-rate equilibrium fold');ax.plot(float(p['JEE_core']),float(p['B_mean_hz']),'o',mfc='white',mec='black',ms=4,zorder=9)
    ax.set(xlim=(1.04,1.14),ylim=(.255,.325),xticks=[1.05,1.1],yticks=[.26,.30,.32])
    ax.set_title('B: overlapping low-rate projections',fontsize=8.5,pad=4)
    ax.set_xlabel(J,fontsize=8,labelpad=1);ax.set_ylabel('Hz / cell',fontsize=8,labelpad=1)
    ax.tick_params(labelsize=8,length=2,pad=2)

def period_panel(ax):
    for rows in SEQ:
        ax.plot([r['J_exact'] for r in rows],[r['T_full_ms'] for r in rows],
            color='#745295' if rows[0]['period_multiple']==2 else COL['mean'],
            ls='-' if rows[0]['stable'] else '--',lw=1.4)
    ax.set(xlim=(.45,1.62),ylim=(100,2500),yscale='log',yticks=[150,500,2000],yticklabels=['150','500','2000'],
        xticks=[.5,.75,1,1.25,1.5],xlabel=J,ylabel=r'$T_{\mathrm{full}}$ (ms)')
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # PD daughter windows are narrower than the main plot can resolve.
    ins=ax.inset_axes([.035,.34,.39,.57]);p=cp('PD1');jc=float(p['JEE_core'])
    branch=next(s for s in SEQ if s[0]['branch_id']=='PD1_2T')
    ins.plot([(r['J_exact']-jc)*1e7 for r in branch],[r['T_full_ms'] for r in branch],'o-',ms=3,lw=1,color='#745295')
    control=read(OUT/'pd1_comparison.json')['mother']
    ins.plot([0,(control['J_exact']-jc)*1e7],[float(p['T_ms']),control['T_full_ms']],color=COL['mean'],lw=1.1)
    ins.plot(0,float(p['T_ms']),'^',mfc='white',mec='black',ms=4)
    ins.set(xlim=(-.9,.9),ylim=(165,395),yticks=[187.5,375],yticklabels=['T','2T'],xticks=[-.75,0,.75])
    ins.tick_params(labelsize=7,pad=1,length=2)
    ins.set_xlabel(r'$(J-J_{\mathrm{PD1}})\times10^7$',fontsize=8,labelpad=1)
    for s in ins.spines.values():s.set_visible(True);s.set_linewidth(.5)

def waveform(row):
    r=resample(np.load(row['path'])['r'],16384,axis=0)*1000
    # Same physical time, no stretching. Fourier orbit periodically repeated.
    t=np.arange(0,2000,.25);phase=(t%row['T_full_ms'])/row['T_full_ms']*len(r)
    a=np.floor(phase).astype(int);f=phase-a
    out=r[a]*(1-f[:,None])+r[(a+1)%len(r)]*f[:,None]
    return t/1000,out

def main_figure():
    fig=plt.figure(figsize=(15.5,12.2))
    a=fig.add_axes([.065,.66,.45,.27]);b=fig.add_axes([.065,.33,.45,.27]);t=fig.add_axes([.065,.09,.45,.165])
    for k,ax in enumerate([a,b]):core_axis(ax,k);ax.set_xlabel(J)
    b_zoom(b,[.075,.49,.415,.38]);period_panel(t)
    a.set_title('Same network solutions: Core A / B projections',loc='left',fontsize=12,pad=9)
    for i,row in enumerate(STATES):
        top=.91-i*.213
        ax=fig.add_axes([.635,top-.08,.345,.079]);net=fig.add_axes([.635,top-.155,.345,.055])
        time,r=waveform(row)
        for j in (0,1):ax.plot(time,r[:,j],color=AB[j],lw=.85)
        allE=(r[:,:3]*COUNT[:3]).sum(1)/COUNT[:3].sum()
        for y,col,ls in [(allE,NET[0],'-'),(r[:,2],NET[1],'-'),(r[:,5],NET[2],'--')]:net.plot(time,y,color=col,lw=.9,ls=ls)
        ax.set(xlim=(0,2),ylim=(-5,420),yticks=[0,200,400],ylabel='Core E (Hz)')
        net.set(xlim=(0,2),ylim=(-2,525),yticks=[0,250,500],ylabel='Network (Hz)')
        ax.tick_params(axis='x',labelbottom=False)
        net.set_xticks([0,.5,1,1.5,2]);net.set_xlabel('Time (s)' if i==3 else '')
        ax.set_title(f'{row["label"]}   {row["title"]}',loc='left',fontsize=11,pad=20)
        ax.text(0,1.08,f'{J} = {row["J_exact"]:.9f}     '+r'$T_{\mathrm{full}}$'+f' = {row["T_full_ms"]:.2f} ms',
            transform=ax.transAxes,fontsize=9,va='bottom')
    handles=[Line2D([],[],color=COL['eq'],label='Equilibrium'),Line2D([],[],color=COL['mean'],label='Periodic mean'),
        Line2D([],[],marker='s',ls='none',color=COL['hi'],label='Time maximum'),Line2D([],[],marker='s',ls='none',color=COL['lo'],label='Time minimum'),
        Line2D([],[],color='black',marker='s',ms=4,label='Stable'),Line2D([],[],color='black',ls='--',marker='s',ms=4,mfc='white',label='Unstable')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.006),ncol=3,frameon=False,fontsize=9,columnspacing=1.)
    fig.legend(handles=[Line2D([],[],color=c,label=s,ls='--' if s=='Surround I' else '-') for c,s in zip(AB+NET,['Core A E','Core B E','All E','Surround E','Surround I'])],
        loc='lower right',bbox_to_anchor=(.99,.025),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Bifurcation branches and whole-network rate rhythms',fontsize=16,y=.985)
    fig.text(.53,.951,'Deterministic six-population model',ha='center',fontsize=11)
    save(fig,'00_corrected_bifurcation_network',
        '左侧A/B投影共享逐解编号与稳定性，极值改为稀疏实心或空心方块，底部显示完整周期和PD1的T/2T局部支。右侧a–d为讨论指定四条稳定周期轨道，全部采用相同2秒时间窗；全网E按细胞数加权，加入surround E/I。',
        '这是确定性六群体模型；每个极值属于同一完整周期，不是置信区间。c/d同为J=1.38；空间SNN能否实现这些状态需另核验。')

def detail_figures():
    fig,axes=plt.subplots(1,2,figsize=(13,5.8));fig.subplots_adjust(left=.075,right=.98,bottom=.16,top=.86,wspace=.24)
    for group,ax in enumerate(axes):
        curves(ax,group);yscale(ax);ax.set(xlim=(1.32,1.415),xticks=[1.32,1.34,1.36,1.38,1.40],ylim=(15,440),yticks=[20,50,100,200,400],yticklabels=['20','50','100','200','400'],xlabel=J,ylabel=f'Core {"AB"[group]} E (Hz / cell)')
        for name,off in zip(['LP1','PD1','PD2','PD3'],[(-8,25),(-35,-25),(20,-20),(-25,26)]):critical(ax,name,group,name,off)
    fig.suptitle('Right-hand branches: means and sparse orbit extrema',fontsize=15)
    save(fig,'01_right_branch_detail','右端母周期支以及已验证短2T子支的A/B投影；均值连线保留延拓顺序，包含折返。稳定及不稳定轨道均显示时间极值。','均值线与另一轨道极值的交叉不代表轨道相连；连通性来自逐解延拓。')
    comp=read(OUT/'pd1_comparison.json');fig,axes=plt.subplots(3,2,figsize=(12,9));fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.90,wspace=.27,hspace=.43)
    for col,row in enumerate([comp['mother'],comp['daughter']]):
        r=resample(np.load(row['path'])['r'],16384,axis=0)*1000
        # Each panel spans the daughter full period, with the mother repeated.
        ncycles=2 if col==0 else 1;rr=np.tile(r,(ncycles,1));tt=np.arange(len(rr))*row['T_full_ms']/len(r)
        for j,label,color in [(0,'Core A E',AB[0]),(1,'Core B E',AB[1])]:axes[0,col].plot(tt,rr[:,j],color=color,lw=1.1,label=label)
        axes[1,col].plot(tt,rr[:,2],color=NET[1],label='Surround E');axes[1,col].plot(tt,rr[:,5],color=NET[2],ls='--',label='Surround I')
        axes[0,col].set_title(('Same mother: T' if col==0 else 'Stable daughter: 2T')+f'\n{J} = {row["J_exact"]:.12f}',fontsize=11)
        for k in (0,1):axes[k,col].set(xlim=(0,375.1),xticks=[0,100,200,300],ylabel='Rate (Hz / cell)',xlabel='Time (ms)')
        half=rr[len(rr)//2:]-rr[:len(rr)//2]
        for j,color,label in [(2,NET[1],'Surround E'),(5,NET[2],'Surround I')]:axes[2,col].plot(tt[:len(half)],half[:,j],color=color,label=label)
        axes[2,col].set(xlim=(0,187.6),xticks=[0,50,100,150],ylim=(-.85,.85),xlabel='Time in first half (ms)',ylabel=r'$r(t+T_{full}/2)-r(t)$ (Hz)')
    axes[0,0].legend(frameon=False,ncol=2);axes[1,0].legend(frameon=False,ncol=2)
    fig.suptitle('PD1: does the alternation reach the surrounding populations?',fontsize=15)
    save(fig,'02_PD1_same_parent_comparison','PD1两侧等距取样：新求得同一母支的T周期作为对照，和已验证稳定2T子轨道使用同一约375 ms观察窗。底行为实际轨道半周期差，不是临界向量的任意幅值。','周边E/I确有交替，峰值差约0.27/0.74 Hz；这仍不能证明二维传播方向交替。')

def manifest():
    path=OUT/'figure_manifest.json';old=read(path) if path.exists() else []
    keep=[r for r in old if r['name'] not in {r['name'] for r in MANIFEST}]
    rows=keep+MANIFEST;path.write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    (FIG/'README.md').write_text('\n\n'.join(f'### {r["name"]}.{r.get("format","png")}\n{r["caption"]} '+('同名SVG保留可编辑文字。' if r.get('format','png')=='png' else '')+f'**关注点**：{r["focus"]}' for r in rows)+'\n')

if __name__=='__main__':main_figure();detail_figures();manifest()

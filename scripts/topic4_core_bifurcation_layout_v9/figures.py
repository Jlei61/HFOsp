"""Reorganize saved bifurcation solutions and matching waveforms into one figure."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.signal import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/topic4_core_bifurcation_v2'))
from rate_paths import saved_path
BASE = ROOT/'results/topic4_sef_hfo'
V2 = BASE/'core_burst_bifurcation_v2_20260915'
V7 = BASE/'core_network_bifurcation_v7_20260916'
OUT = BASE/'core_bifurcation_composite_v9_20260916'
FIG = OUT/'figures'
FIG.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads(Path(p).read_text())
SEQ = read(V7/'displayed_curve_sequences.json')
EQ = read(V2/'equilibrium_spectrum.json')
FOLD = read(V2/'fold.json')
CP = list(csv.DictReader((V7/'critical_points.csv').open()))
CASES = read(V7/'reduced_condition_coordinates.json')
J = r'$J_{\mathrm{EE,core}}$'
COL = {'eq':'#286aa4', 'mean':'#c8831d', 'hi':'#258253', 'lo':'#329cac'}
AB = ['#2769a5', '#9a4c8f']
POINT = '#a5342a'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
    'axes.labelsize':13, 'axes.titlesize':14, 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False,
    'xtick.labelsize':11, 'ytick.labelsize':11, 'legend.fontsize':10.5})

def critical(label):
    return next(r for r in CP if r['label'] == label)

def crit_xy(label, group):
    p = critical(label)
    return float(p['JEE_core']), float(p['AB'[group]+'_mean_hz'])

def selected_cases():
    ans = []
    for letter, number, title in [('a','20a','Low-rate equilibrium'),
        ('b','20b','Coexisting periodic burst'), ('c','12a','Two-core bursts; before LP1'),
        ('d','15a','A bursts / B high background'),
        ('e','15b','Both cores: high background')]:
        row = dict(next(r for r in CASES if r['number'] == number))
        row['path'] = str(saved_path(row['path']))
        row.update(letter=letter, title=title, source_id=number)
        if row['kind'] == 'Stable periodic orbit':
            z = np.load(row['path'])
            row['rate'] = resample(z['r'], 8192, axis=0)*1000
            assert abs(float(z['g'])-row['g']) < 1e-12
            assert np.max(abs(z['r'].mean(0)*1000-np.array(row['mean']))) < 1e-9
        else:
            row['rate'] = np.tile(row['mean'], (1024,1))
        ans.append(row)
    return ans

SELECTED = selected_cases()

def plot_branches(ax, group, xoffset=0, xfactor=1, lw=1.6):
    xx = lambda v: (np.asarray(v)-xoffset)*xfactor
    for direction, ls in [(-1,'-'),(1,'--')]:
        rows = [r for r in EQ if r['direction'] == direction]
        ax.plot(xx([FOLD['g']]+[r['g'] for r in rows]),
            [FOLD['r_hz'][group]]+[r['r_hz'][group] for r in rows],
            color=COL['eq'], ls=ls, lw=lw)
    for rows in SEQ:
        assert len({r['stable'] for r in rows}) == 1
        stable = rows[0]['stable']
        gx = xx([r['g'] for r in rows])
        if stable:
            for key in ('hi','lo'):
                ax.plot(gx, [r[key][group] for r in rows], color=COL[key], lw=lw*.7)
        ax.plot(gx,[r['mean'][group] for r in rows],color=COL['mean'],
            ls='-' if stable else '--',lw=lw)

def symlog(ax, ylim=(-.03,480), ticks=None):
    ax.set_yscale('symlog', linthresh=1, linscale=.65, base=10)
    ts = ticks or [0,1,10,100,400]
    ax.set(ylim=ylim, yticks=ts, yticklabels=[f'{v:g}' for v in ts])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

def cp_label(ax, label, group, name, offset, size=10.5, xoffset=0, xfactor=1):
    g,y = crit_xy(label, group)
    g = (g-xoffset)*xfactor
    marker = '^' if label.startswith('PD') else ('s' if 'equilibrium' in label else 'o')
    ax.plot(g,y,marker=marker,mfc='white',mec='#202020',mew=1,ms=5,zorder=8)
    ax.annotate(name,(g,y),xytext=offset,textcoords='offset points',fontsize=size,
        ha='right' if offset[0]<0 else 'left', va='center',
        arrowprops=dict(arrowstyle='-',lw=.7,color='#333333'),
        bbox=dict(facecolor='white',edgecolor='none',pad=.7),zorder=9)

def representative(ax, row, group, offset=None, xoffset=0, xfactor=1):
    x,y = (row['g']-xoffset)*xfactor, row['mean'][group]
    ax.plot(x,y,marker='o',color=POINT,ms=4.5,zorder=10)
    if offset is not None:
        ax.annotate(row['letter'],(x,y),xytext=offset,textcoords='offset points',
            color=POINT,fontsize=12,fontweight='bold',ha='center',va='center',
            arrowprops=dict(arrowstyle='-',lw=.65,color=POINT),
            bbox=dict(facecolor='white',edgecolor='none',pad=.4),zorder=11)

def main_axis(ax, group, narrow=False):
    plot_branches(ax,group)
    symlog(ax)
    ax.set(xlim=(.45,1.62),xlabel=J,ylabel=f'Core {"AB"[group]} E rate (Hz / cell)')
    ax.set_xticks([.5,.75,1,1.25,1.5])
    ax.set_title(f'Core {"AB"[group]}',loc='left',fontweight='bold',pad=8)
    cp_label(ax,'Low-rate equilibrium fold',group,'Fold',(-25,14))
    cp_label(ax,'Cycle fold',group,'Cycle fold',(-28,-13))
    cp_label(ax,'LP0c',group,'LP0 / PD0',(-32,18))
    positions = ([(-12,27),(-51,1),(27,-4),(-26,16)] if group == 0 else
                 [(-15,18),(-32,-17),(70,-20),(12,-30)])
    for name,offset in zip(['LP1','PD1','PD2','PD3'],positions):
        cp_label(ax,name,group,name,offset)
    for row in SELECTED:
        if row['letter'] in ('a','b'):
            representative(ax,row,group)
        elif group == 0:
            representative(ax,row,group,{'c':(-38,-22),'d':(30,-30),'e':(40,14)}[row['letter']])
        else:
            representative(ax,row,group,{'c':(-45,-12),'d':(55,-43),'e':(55,-4)}[row['letter']])

def onset_inset(parent, bounds):
    ax = parent.inset_axes(bounds)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.7)
    plot_branches(ax,0,xoffset=1.1218,xfactor=1e6,lw=1.1)
    ax.set(xlim=(13,41),ylim=(-.25,12),xticks=[15,25,35],yticks=[0,5,10])
    ax.tick_params(labelsize=8.5,length=2,pad=2)
    ax.set_title('Onset coexistence',fontsize=10.5,pad=5)
    ax.set_xlabel(r'$(J_{\mathrm{EE,core}}-1.1218)\times10^6$',fontsize=8.5,labelpad=2)
    ax.set_ylabel('A E rate (Hz)',fontsize=8.5,labelpad=2)
    cp_label(ax,'Cycle fold',0,'Cycle fold',(7,10),8.5,1.1218,1e6)
    hc = (float(critical('HC limit estimate')['JEE_core'])-1.1218)*1e6
    ax.axvline(hc,ls=':',color='#333333',lw=.7)
    ax.text(hc-1,4.8,'HC*',ha='right',fontsize=8.5)
    for row in SELECTED[:2]:
        representative(ax,row,0,(-10,7) if row['letter']=='a' else (9,4),1.1218,1e6)
    return ax

def right_inset(parent,bounds,group):
    ax = parent.inset_axes(bounds)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.7)
    plot_branches(ax,group,lw=1.05)
    symlog(ax,(30,420),[30,100,300])
    ax.set(xlim=(1.32,1.41),xticks=[1.33,1.37,1.41])
    ax.tick_params(labelsize=8.5,length=2,pad=2)
    ax.set_title('Cycle fold and period doubling',fontsize=10.5,pad=5)
    ax.set_xlabel(J,fontsize=9,labelpad=2)
    ax.set_ylabel(f'{"AB"[group]} E rate (Hz)',fontsize=8.5,labelpad=2)
    pos = [(-20,-7),(-23,6),(-4,-23),(9,-40)] if group == 1 else [(-22,8),(-18,-13),(9,1),(-20,12)]
    for label,offset in zip(['LP1','PD1','PD2','PD3'],pos):
        cp_label(ax,label,group,label,offset,size=8.5)
    return ax

def waveform(ax,row,last=False,compact=False):
    rate = row['rate']
    if row['kind']=='Stable equilibrium':
        time = np.linspace(0,1,len(rate))
        plotted = rate
        horizon = 1
        ax.set(ylim=(0,.55),yticks=[0,.25,.5])
    else:
        time = np.arange(2*len(rate)+1)*row['T']/len(rate)/1000
        plotted = np.vstack([rate,rate,rate[:1]])
        horizon = 2*row['T']/1000
        ax.set(ylim=(-5,420),yticks=[0,200,400])
    for group in (0,1):
        ax.plot(time,plotted[:,group],color=AB[group],lw=1.25,label=f'Core {"AB"[group]}')
    jtext = f'{row["g"]:.5f}' if row['letter'] in ('a','b') else f'{row["g"]:.3f}'
    ax.set_title(f'{row["letter"]}   {row["title"]}',fontsize=11.5,loc='left',pad=5)
    ax.text(1,1.08,f'{J} = {jtext}',transform=ax.transAxes,ha='right',va='bottom',fontsize=10)
    if row['kind']!='Stable equilibrium':
        tx,ty,ha = (.03,.86,'left') if row['letter'] in ('b','c') else ((.03,.18,'left') if row['letter']=='d' else (.98,.08,'right'))
        ax.text(tx,ty,f'T = {row["T"]:.1f} ms',transform=ax.transAxes,
            ha=ha,fontsize=9.5,bbox=dict(facecolor='white',edgecolor='none',pad=.6))
    ax.set_xlim(0,horizon)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.tick_params(labelsize=10)
    if last:
        ax.set_xlabel('Time (s)',fontsize=12)

def legends(fig,y=.055):
    handles=[Line2D([],[],color=COL['eq'],label='Equilibrium'),
        Line2D([],[],color=COL['mean'],label='Periodic mean'),
        Line2D([],[],color=COL['hi'],label='Stable maximum'),
        Line2D([],[],color=COL['lo'],label='Stable minimum'),
        Line2D([],[],color='#222222',ls='-',label='Stable'),
        Line2D([],[],color='#222222',ls='--',label='Unstable')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,y),
        ncol=3,frameon=False,columnspacing=1.5,handlelength=2,fontsize=10)
    fig.legend(handles=[Line2D([],[],color=AB[i],label=f'Core {"AB"[i]} E') for i in (0,1)],
        loc='upper right',bbox_to_anchor=(.975,.962),ncol=2,frameon=False,fontsize=11)

def assemble_joint():
    fig = plt.figure(figsize=(15,12.2))
    axa = fig.add_axes([.072,.555,.47,.36])
    axb = fig.add_axes([.072,.145,.47,.36])
    for group,ax in enumerate([axa,axb]):
        main_axis(ax,group)
    onset_inset(axa,[.075,.56,.405,.35])
    right_inset(axb,[.075,.47,.405,.42],1)
    for i,row in enumerate(SELECTED):
        ax = fig.add_axes([.66,.787-i*.161,.318,.110])
        waveform(ax,row,last=i==4)
    fig.text(.610,.51,'Population E rate (Hz / cell)',rotation=90,ha='center',va='center',fontsize=13)
    fig.suptitle('Core bifurcations and corresponding network waveforms',fontsize=18,y=.99)
    legends(fig)
    return fig

def assemble_reference():
    fig = plt.figure(figsize=(15,9.8))
    axa = fig.add_axes([.072,.155,.50,.765])
    main_axis(axa,0)
    onset_inset(axa,[.065,.655,.38,.24])
    for i,row in enumerate(SELECTED):
        ax = fig.add_axes([.69,.786-i*.16,.29,.108])
        waveform(ax,row,last=i==4,compact=True)
    fig.text(.645,.51,'Population E rate (Hz / cell)',rotation=90,ha='center',va='center',fontsize=13)
    fig.suptitle('Core A bifurcation structure with matched A/B waveforms',fontsize=18,y=.99)
    legends(fig,y=.04)
    return fig

def save(fig,name,book):
    fig.canvas.draw()
    for ext in ('png','pdf'):
        fig.savefig(FIG/f'{name}.{ext}',dpi=210)
    book.savefig(fig)
    with Image.open(FIG/f'{name}.png') as im:
        im.load()
        pixels=list(im.size)
    result=dict(name=name,pixels=pixels,main_axes=len(fig.axes),
        inset_axes=sum(len(a.child_axes) for a in fig.axes))
    plt.close(fig)
    return result

def main():
    manifest=[]
    with PdfPages(FIG/'core_bifurcation_composite_comparison.pdf') as book:
        manifest.append(save(assemble_joint(),'00_joint_core_bifurcation_composite',book))
        manifest.append(save(assemble_reference(),'01_core_A_reference_layout',book))
    source_rows=[]
    for row in SELECTED:
        source_rows.append({k:v for k,v in row.items() if k!='rate'})
    metadata=dict(model='Frozen deterministic six-population delayed rate closure, v2-v7',
        changed='Figure composition only',native_snn_points_or_rasters_added=False,
        curve_sequences=len(SEQ),periodic_curve_points=sum(map(len,SEQ)),
        source_curve_path=str(V7/'displayed_curve_sequences.json'),
        source_curve_sha256=hashlib.sha256((V7/'displayed_curve_sequences.json').read_bytes()).hexdigest(),
        yscale=dict(type='symlog',linear_below_hz=1,linscale=.65),
        waveform_time='Seconds; two full periods for each periodic solution, one second for the equilibrium',
        phase='Original saved joint phase; no independent alignment between A and B',
        source_conditions=source_rows,figures=manifest,
        coexistence_pairs=[['a','b'],['d','e']],
        HC_star='Finite-period numerical homoclinic-limit evidence, not an infinite-time connecting-orbit proof',
        branch_types='Fold: equilibrium saddle-node; LP/cycle fold: periodic saddle-node; PD: period doubling',
        human_acceptance='PENDING')
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    (FIG/'README.md').write_text('''# 分岔骨架与对应波形：复合排版 v9

### 00_joint_core_bifurcation_composite.png / .pdf
左侧上下为同一联合网络的 Core A/B 分岔投影，分别嵌入起始共存微区与右侧周期分岔放大；右侧五行对应 a–e 的真实已保存率模型解，每行同时画 A/B。红色小圆点及字母仅对应这五个临界附近的解，主轴仍为零附近线性、以上对数；没有恢复原生SNN编号菱形。
**关注点**：a/b 同 J 的低率与 burst 共存，以及 d/e 同 J 的两种周期态；右侧行顺序不表示一次扫参轨迹。

### 01_core_A_reference_layout.png / .pdf
采用更接近用户参考图的单一 Core A 大分岔轴，左上放大起始共存区，右侧仍是相同 a–e 的 A/B 波形；LP1/PD1–PD3 保留在主轴。两个版本使用完全相同数值，不随版式更换状态或参数。
**关注点**：比较单轴主图与 A/B 上下对齐方案的整体可读性；Core B 的完整分支见上一张。

### core_bifurcation_composite_comparison.pdf
按顺序收录上述两种完整复合排版，供本轮目视比较。Fold 是平衡点鞍结，LP/cycle fold 是周期轨道鞍结，PD 是倍周期；HC* 为有限周期延拓支持的同宿极限估计。
**关注点**：这是确定性率模型解的重新组织，未新增原生 SNN 仿真或 raster，也未扩张既有分岔证明的范围。
''')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    main()

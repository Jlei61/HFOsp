"""Square bifurcation + numbered native-state examples; preserve model layers."""
from extend import ROOT, OUT, SOURCE
from model import System
from periodic import Orbit
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.signal import resample
from PIL import Image

NATIVE = ROOT / 'results/topic4_sef_hfo/burst_regime_map_20260914'
V1 = ROOT / 'results/topic4_sef_hfo/core_burst_onset_brunel_v1_20260915'
FIG = OUT / 'figures'
JLABEL = r'$J_{\mathrm{EE,core}}$'
BLUE, RED, GREEN, ORANGE = '#2366a2', '#bb4437', '#22856d', '#cf8d31'
STATES = [(1,.5,'Resting / low activity','#657b8c'),
          (2,.7,'Irregular bursts','#268b91'),
          (3,.85,'Intermediate','#d2942b'),
          (4,1.,'Regular bursts','#8855a3')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,
 'axes.titlesize':14,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.,
 'lines.linewidth':2.,'pdf.fonttype':42,'savefig.facecolor':'white'})
MANIFEST=[]
DESCRIPTIONS=[]

def read(path):
    return json.loads(path.read_text())

def save(fig, stem, desc, focus, book):
    for ext in ('png','pdf'):
        fig.savefig(FIG/f'{stem}.{ext}', dpi=220, facecolor='white')
    book.savefig(fig)
    with Image.open(FIG/f'{stem}.png') as im:
        im.load(); size=list(im.size)
    MANIFEST.append(dict(stem=stem,pixels=size))
    DESCRIPTIONS.append(f'### {stem}.png / .pdf\n{desc}\n**关注点**：{focus}\n')
    plt.close(fig)

def load_native():
    meta=read(NATIVE/'figures/native_burst_four_states_waveform_raster.json')
    sample=np.array(meta['raster_neuron_ids'],int)
    sample=sample[np.linspace(0,len(sample)-1,30,dtype=int)]
    base=read(V1/'baseline_observables.json')
    rows=[]
    for number,g,name,color in STATES:
        old=next(x for x in meta['rows'] if x['parameters']['ee']==g)
        path=NATIVE/'per_run'/old['name']/'trajectory.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==old['trajectory_sha256']
        with np.load(path) as z:
            col=z['group_names'].tolist().index('coreAE')
            n=int(z['group_sizes'][col]); assert n==720
            rate=z['spike_counts_2ms'][:,col]/n/.002
            t=(np.arange(len(rate))+.5)*.002
            keep=np.isin(z['raster_cell'],sample)
            rt=z['raster_time_ms'][keep]/1000-.001
            ri=np.searchsorted(sample,z['raster_cell'][keep])+1
        mean=float(rate[(t>=2)&(t<20)].mean())
        assert abs(mean-old['mean_rate_hz'])<1e-10
        obs=next(x for x in base if x['ee']==g and x['depth']==1 and x['seed']==848101 and x['group']=='coreAE')
        rows.append(dict(number=number,g=g,name=name,color=color,t=t,rate=rate,rt=rt,ri=ri,
            cv=old['interval_cv'],n_bursts=old['n_bursts'],mean=mean,source=str(path),
            source_sha256=old['trajectory_sha256'],sample=sample,observables=obs))
    return rows

def load_cycles():
    chosen={}
    for root in (SOURCE,OUT):
        for p in (root/'periodic').glob('g*_N*.npz'):
            z=np.load(p); g=float(z['g']); N=len(z['r'])
            if float(z['residual'])>1e-8: continue
            if g not in chosen or N>chosen[g]['N']:
                f=root/'floquet'/f'g{g:.8f}_dt0.1.json'
                if not f.exists(): continue
                fl=read(f); mm=np.array([complex(*v) for v in fl['multipliers']])
                k=np.argmin(abs(mm-1)); assert abs(mm[k]-1)<.005 and max(abs(np.delete(mm,k)))<1
                r=z['r']; rr=resample(r,max(8192,len(r)),axis=0)
                chosen[g]=dict(g=g,r=r,T=float(z['T']),N=N,path=str(p),
                    mean=r.mean(0)*1000,lo=rr.min(0)*1000,hi=rr.max(0)*1000,
                    max_transverse=float(max(abs(np.delete(mm,k)))))
    return [chosen[k] for k in sorted(chosen)]

def number_marker(ax,x,y,num,color,size=290):
    ax.scatter([x],[y],s=size,c=color,edgecolors='white',linewidths=1.4,zorder=12)
    ax.text(x,y,str(num),ha='center',va='center',color='white',fontweight='bold',fontsize=12,zorder=13)

def branches(ax,cyc,fold,arc,native=True):
    low=sorted([x for x in arc if x['direction']==-1],key=lambda x:x['g'])
    up=[x for x in arc if x['direction']==1]
    j=fold['g']; rf=fold['r_hz'][0]
    ax.plot([x['g'] for x in low]+[j],[x['r_hz'][0] for x in low]+[rf],color=BLUE)
    ax.plot([j]+[x['g'] for x in up],[rf]+[x['r_hz'][0] for x in up],ls='--',color=RED)
    # This branch stops before the second-core equilibrium turn; mark its
    # numerical endpoint as open, not as a second identified bifurcation.
    ax.plot(up[-1]['g'],up[-1]['r_hz'][0],marker='o',mfc='white',mec=RED,ms=4)
    xx=np.array([c['g'] for c in cyc]); hi=np.array([c['hi'][0] for c in cyc]); lo=np.array([c['lo'][0] for c in cyc]); mean=np.array([c['mean'][0] for c in cyc])
    ax.fill_between(xx,lo,hi,color=GREEN,alpha=.07)
    ax.plot(xx,hi,'o-',color=GREEN,ms=3.5); ax.plot(xx,lo,'o-',color=GREEN,ms=3.5)
    ax.plot(xx,mean,color=ORANGE,lw=2.1)
    ax.scatter([j],[rf],s=55,c='black',zorder=7)
    ax.axvline(j,color='#444444',ls=':',lw=1.1)
    ax.set(xlim=(.46,1.19),ylim=(-.05,430),xlabel=JLABEL,ylabel='Core A E rate (Hz / neuron)')
    ax.set_yscale('symlog',linthresh=1)
    ax.set_yticks([0,.5,1,3,10,30,100,300]); ax.set_yticklabels(['0','0.5','1','3','10','30','100','300'])
    ax.set_xticks([.5,.7,.85,1.,1.1,1.175]); ax.set_xticklabels(['0.5','0.7','0.85','1.0','1.1','1.175'])
    ax.annotate('Fold  1.1254',xy=(j,rf),xytext=(.87,.30),fontsize=11,
                arrowprops=dict(arrowstyle='-',lw=1,color='black'))

def native_panel(fig,rows,box=(.085,.12,.89,.245),window=(4.,7.),title_y=.425):
    left,bottom,width,height=box
    gap=.036; w=(width-3*gap)/4
    for k,row in enumerate(rows):
        x=left+k*(w+gap)
        a=fig.add_axes([x,bottom+height*.64,w,height*.36])
        b=fig.add_axes([x,bottom,w,height*.50])
        sel=(row['t']>=window[0])&(row['t']<window[1]); rs=(row['rt']>=window[0])&(row['rt']<window[1])
        a.plot(row['t'][sel],row['rate'][sel],color=row['color'],lw=1.05)
        # Linear, explicitly ticked per-panel scales keep background spikes
        # visible without compressing burst recruitment into a few pixels.
        a.set_ylim(0,max(2,row['rate'][sel].max()*1.12)); a.yaxis.set_major_locator(MaxNLocator(3))
        a.tick_params(axis='x',labelbottom=False); a.set_xlim(window)
        b.vlines(row['rt'][rs],row['ri'][rs]-.37,row['ri'][rs]+.37,color='#15191c',lw=1.15)
        b.set(xlim=window,ylim=(.4,30.6),yticks=[1,15,30],xlabel='Time (s)')
        if window[1]-window[0]>1:
            a.set_xticks([4,5,6,7]); b.set_xticks([4,5,6,7])
        else:
            a.set_xticks([5,5.2,5.4,5.6]); b.set_xticks([5,5.2,5.4,5.6])
        if k==0:
            a.set_ylabel('Rate (Hz)'); b.set_ylabel('E neuron')
        fig.text(x,title_y,str(row['number']),color='white',weight='bold',fontsize=12,
                 bbox=dict(boxstyle='circle,pad=.25',fc=row['color'],ec='none'))
        short=['Resting','Irregular bursts','Intermediate','Regular bursts'][k]
        fig.text(x+.025,title_y,short,color=row['color'],weight='bold',fontsize=12)
        cv='no bursts' if row['cv'] is None else f'IEI CV {row["cv"]:.2f}'
        fig.text(x,title_y-.025,f'{JLABEL} = {row["g"]:g}  |  {cv}',fontsize=10.3)

def composite(rows,cyc,fold,arc,book):
    fig=plt.figure(figsize=(12.8,12.3))
    fig.text(.085,.963,'Core burst onset',fontsize=20,weight='bold')
    ax=fig.add_axes([.085,.525,.89,.39])
    branches(ax,cyc,fold,arc)
    for row in rows:
        number_marker(ax,row['g'],row['mean'],row['number'],row['color'])
    handles=[Line2D([],[],color=BLUE,label='Stable equilibrium'),Line2D([],[],color=RED,ls='--',label='Unstable equilibrium'),
        Line2D([],[],color=GREEN,label='Burst maximum / minimum'),Line2D([],[],color=ORANGE,label='Burst time mean'),
        Line2D([],[],color='black',marker='o',ls='none',ms=7,label='Native SNN mean (1–4)')]
    ax.legend(handles=handles,loc='upper left',fontsize=10.5,frameon=False,borderaxespad=.7,labelspacing=.45)
    xx=np.array([c['g'] for c in cyc])
    # Both linear-rate enlargements are inside the main bifurcation axes.
    # Their lower edge stays above the native state markers and red branch.
    a=ax.inset_axes([.36,.735,.225,.225]); b=ax.inset_axes([.655,.735,.225,.225])
    a.plot(xx,[c['hi'][0] for c in cyc],'o-',ms=3.7,color=GREEN)
    b.plot(xx,[c['mean'][0] for c in cyc],'o-',ms=3.7,color=ORANGE,label='Mean')
    b.plot(xx,[c['lo'][0] for c in cyc],'o-',ms=3.2,color=GREEN,label='Minimum')
    for q in (a,b):
        q.axvline(fold['g'],color='#444444',ls=':',lw=1.)
        q.set_xlim(1.10,1.181); q.set_xticks([1.10,1.125,1.15,1.175]); q.ticklabel_format(axis='x',style='plain',useOffset=False)
        q.tick_params(axis='both',labelsize=9)
        q.set_ylabel('Rate (Hz)',fontsize=10)
    a.set_ylim(220,280); a.set_yticks([220,250,280]); a.set_title('Burst peak',fontsize=11,pad=5)
    b.set_ylim(-.7,20); b.set_yticks([0,10,20]); b.set_title('Mean / minimum',fontsize=11,pad=5)
    b.legend(frameon=False,fontsize=8.5,loc='upper left',borderaxespad=.2,handlelength=1.1,labelspacing=.2)
    native_panel(fig,rows,box=(.085,.09,.89,.285),title_y=.439)
    save(fig,'01_square_bifurcation_four_states_inset',
        '按用户修订去掉主图的灰色说明小字，将1.1之后的周期峰率、平均率和最小率放大图内嵌于同一主坐标轴。下方四个状态共用固定30个细胞及4–7秒窗口，各波形使用明确标注的线性率轴；IEI CV仍取2–20秒。',
        '编号圆点的Y值是原SNN的2–20秒时间平均率，不能当作降阶平衡点；四个原生例子均位于降阶折点左侧。',book)

def eigendiagnostic(fold,book):
    fine=read(SOURCE/'fine_fold_branch.json')
    fig,axs=plt.subplots(2,2,figsize=(9.6,9.1))
    fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.13,hspace=.56,wspace=.38)
    a,b,c,d=axs.ravel(); j=fold['g']
    for direction,color,ls in [(-1,BLUE,'-'),(1,RED,'--')]:
        rows=sorted([r for r in fine if r['direction']==direction],key=lambda r:r['g'])
        xx=[r['g'] for r in rows]
        a.plot(xx,[r['r_hz'][0] for r in rows],color=color,ls=ls)
        b.plot(xx,[r['lambda_per_s'][0] for r in rows],color=color,ls=ls)
    a.scatter([j],[fold['r_hz'][0]],s=60,c='black')
    a.set(ylim=(.39,.53),ylabel='Core A E rate (Hz / neuron)',title='Stable and saddle branches meet')
    b.axhline(0,color='#444444',lw=.8);b.scatter([j],[0],s=60,c='black')
    b.set(ylim=(-9,9),ylabel=r'Re $\lambda$ (s$^{-1}$)',title='One real eigenvalue reaches zero')
    b.text(.05,.61,r'Im $\lambda = 0$',transform=b.transAxes,va='top',fontsize=12)
    for q in (a,b):
        q.axvline(j,color='#555555',ls=':',lw=1);q.set(xlim=(j-.006,j+.0005),xlabel=JLABEL)
        q.ticklabel_format(axis='x',useOffset=False);q.set_xticks([1.120,1.122,1.124,1.126])
    labels=['A E','B E','S E','A I','B I','S I']
    for q,vals,title in [(c,fold['v'],'Right mode: rate displacement'),(d,fold['w'],'Left characteristic vector')]:
        vals=np.array(vals);q.bar(range(6),vals,color=[BLUE if x>=0 else RED for x in vals],width=.63)
        q.axhline(0,color='#444444',lw=.8);q.set(xticks=range(6),xticklabels=labels,ylim=(-1.17,1.30),title=title)
        q.tick_params(axis='x',rotation=30)
        for i,val in enumerate(vals):
            q.text(i,val+(.06 if val>=0 else -.06),f'{val:.3f}',fontsize=10,ha='center',va='bottom' if val>=0 else 'top')
    c.set_ylabel(r'$v$  ($\|v\|_2=1$)');d.set_ylabel(r'$w$  ($w^{\mathsf{T}}v=1$)')
    fig.text(.10,.954,'The low-rate fold and its critical mode',weight='bold',fontsize=19)
    fig.text(.10,.916,f'{JLABEL} = {j:.8f}  |  Core A E = {fold["r_hz"][0]:.6f} Hz',fontsize=13)
    fig.text(.10,.054,'Six-population characteristic vectors; S = surround. Full 402-coordinate vectors are retained in v2.',fontsize=10.2)
    save(fig,'02_square_fold_eigenvalues_vectors',
        '以线性坐标放大低率折点，并复用v2经过时延谱与配点收敛验证的实特征值及左右特征向量。右模态集中于A核E群体；左列位移向量与右列特征矩阵敏感度向量含义不同。',
        '局部非退化saddle-node的证据不变；六群体向量不是原始40000个神经元的逐细胞向量。',book)

def core_relationship(cyc,fold,book):
    graph=np.load(SOURCE/'projected_graph.npz'); W=graph['W'].sum(0)
    for aa,bb in (([0,3],[1,4]),([1,4],[0,3])):
        assert np.all(W[np.ix_(aa,bb)]==0)
    z=min(cyc,key=lambda c:abs(c['g']-1.15));r=z['r'];T=z['T']
    phase_lag=((np.argmax(r[:,1])-np.argmax(r[:,0]))/len(r)*T+T/2)%T-T/2
    # Verify each population repeats after the same solved full-state period.
    rel=dict(core_A_E_count=int(graph['count'][0]),core_B_E_count=int(graph['count'][1]),
             core_to_core_direct_weight_sum=float(W[np.ix_([0,3],[1,4])].sum()+W[np.ix_([1,4],[0,3])].sum()),
             coupling='indirect through surround populations',J_EE_core=z['g'],period_ms=T,
             B_minus_A_peak_lag_ms=float(phase_lag),lag_interpretation='phase difference on one solved periodic orbit; not a causal intervention',
             fold_rate_right_vector=fold['v'],fold_rates_hz=fold['r_hz'])
    (OUT/'core_relationship.json').write_text(json.dumps(rel,indent=2)+'\n')
    fig=plt.figure(figsize=(9.6,9.1))
    fig.text(.09,.95,'Core A and B: two spatial populations',fontsize=19,weight='bold')
    ax=fig.add_axes([.06,.57,.9,.30]);ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    centers=[(1.6,1.65),(5.,1.65),(8.4,1.65)]
    for (x,y),lab,col in zip(centers,['Core A\n720 E cells','Surround','Core B\n742 E cells'],[BLUE,'#737d83','#8059a3']):
        ax.add_patch(Circle((x,y),.88,facecolor=col,edgecolor='none',alpha=.12));ax.text(x,y,lab,ha='center',va='center',fontsize=14,color=col,weight='bold')
    for x1,x2 in [(2.5,4.1),(5.9,7.5)]:
        ax.add_patch(FancyArrowPatch((x1,1.65),(x2,1.65),arrowstyle='<->',mutation_scale=17,lw=2,color='#51595e'))
    for x in [1.6,8.4]:ax.text(x,.37,JLABEL+' scales local E→E',ha='center',fontsize=11.5)
    ax.text(5,2.8,'No direct A↔B edges in this realized graph',ha='center',fontsize=12)
    a=fig.add_axes([.11,.20,.84,.31])
    rr=np.roll(r,len(r)//5-np.argmax(r[:,0]),axis=0);rt=np.tile(rr,(2,1));tt=np.arange(len(rt))*T/len(r)/1000
    a.plot(tt,rt[:,0]*1000,color=BLUE,label='Core A E');a.plot(tt,rt[:,1]*1000,color='#8059a3',ls='--',label='Core B E')
    a.set(xlim=(0,2*T/1000),ylim=(-4,285),xlabel='Time (s)',ylabel='Rate (Hz / neuron)',title=f'Solved burst orbit at {JLABEL} = 1.15')
    a.legend(frameon=False,loc='upper right')
    fig.text(.11,.115,f'Full-network period: {T:.3f} ms.  B peak follows A peak by {phase_lag:.2f} ms on this orbit.',fontsize=11.5)
    fig.text(.11,.075,'The fold mode begins in A; the later burst recruits both cores. A/B are spatial identities, not state labels.',fontsize=10.5)
    save(fig,'03_core_a_b_relationship',
        '上方按实际投影连接画出两核经周边群体间接相连的示意，下方显示已求解周期轨道中的A、B核E率。图中时差是同一周期内峰值相位差，未进行因果消融。',
        'A/B是不同空间群体，局部临界模态与非线性burst招募范围是两个读出。',book)

def state_details(rows,book):
    for row in rows:
        fig,axs=plt.subplots(2,2,figsize=(11,8.7),gridspec_kw={'height_ratios':[1,2.4]})
        fig.subplots_adjust(left=.09,right=.97,bottom=.15,top=.83,hspace=.20,wspace=.30)
        for col,window in enumerate(((4.,7.),(5.,5.6))):
            a,b=axs[:,col]; sel=(row['t']>=window[0])&(row['t']<window[1]); rs=(row['rt']>=window[0])&(row['rt']<window[1])
            a.plot(row['t'][sel],row['rate'][sel],color=row['color'],lw=1.3)
            a.set(xlim=window,ylim=(0,max(2,row['rate'][sel].max()*1.12)),ylabel='Rate (Hz / E cell)')
            a.tick_params(axis='x',labelbottom=False);a.yaxis.set_major_locator(MaxNLocator(3))
            b.vlines(row['rt'][rs],row['ri'][rs]-.34,row['ri'][rs]+.34,color='#111111',lw=1.5)
            b.set(xlim=window,ylim=(.4,30.6),yticks=[1,10,20,30],xlabel='Time (s)',ylabel='Core A E neuron')
            a.set_title('Same 3-second window' if col==0 else 'Same 600-ms close-up',loc='left',fontsize=13)
        cv='No detected bursts' if row['cv'] is None else f'IEI CV = {row["cv"]:.3f}; {row["n_bursts"]} bursts'
        fig.text(.09,.955,f'{row["number"]}  {row["name"]}',fontsize=21,weight='bold',color=row['color'])
        fig.text(.09,.906,f'{JLABEL} = {row["g"]:g}  |  {cv}  |  mean = {row["mean"]:.3f} Hz',fontsize=13)
        fig.text(.09,.075,'Native SNN. Rate: all 720 Core A E cells. Raster: occupied 2 ms bins of the same 30 recorded cells.',fontsize=11)
        save(fig,f'04_state_{row["number"]}_large_raster',
            f'编号{row["number"]}（核内EE倍率{row["g"]:g}）的大幅独立图，左侧4–7秒，右侧5.0–5.6秒。两侧使用同一批固定30个细胞，放电率使用全部720个A核E细胞；点以较粗竖线显示。',
            '这些是保存的原生2ms占用记录；波形与raster时间对齐，无响应排序或新增SNN仿真。',book)

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    rows=load_native();cyc=load_cycles();fold=read(SOURCE/'fold.json');arc=read(SOURCE/'equilibrium_spectrum.json')
    if max(c['g'] for c in cyc)<1.175: raise RuntimeError('Right-side continuation and stability checks not complete')
    with PdfPages(FIG/'core_bifurcation_four_states_booklet.pdf') as book:
        composite(rows,cyc,fold,arc,book)
        eigendiagnostic(fold,book)
        core_relationship(cyc,fold,book)
        state_details(rows,book)
    mapping=[]
    for row in rows:
        mapping.append({k:row[k] for k in ('number','g','name','cv','n_bursts','mean','source','source_sha256')})
        mapping[-1].update(J_EE_core=row['g'],raster_neuron_ids=row['sample'].tolist(),
            before_closure_fold=bool(row['g']<fold['g']),native_is_equilibrium=False)
    (OUT/'numbered_native_states.json').write_text(json.dumps(mapping,indent=2)+'\n')
    cycle_rows=[dict(J_EE_core=c['g'],period_ms=c['T'],core_A_E_mean_hz=float(c['mean'][0]),
        core_A_E_max_hz=float(c['hi'][0]),core_A_E_min_hz=float(c['lo'][0]),N=c['N'],
        maximum_transverse_modulus=c['max_transverse'],source=c['path']) for c in cyc]
    (OUT/'displayed_periodic_branch.json').write_text(json.dumps(cycle_rows,indent=2)+'\n')
    (OUT/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
    (FIG/'README.md').write_text('# Core分岔与四状态编号对照 v3\n\n'+'\n'.join(DESCRIPTIONS)+
        '\n### core_bifurcation_four_states_booklet.pdf\n上述7张图的矢量图册，主图采用内嵌放大窗，已去掉灰色说明小字。原生例子只作相同参数的观测对照，未当作降阶吸引子分支。\n**关注点**：待用户目视检查；数值结果保持原版。\n'
        '\n### 01_square_bifurcation_four_states.png / .pdf\n保留的前一版主图，右侧另列两个放大窗，带灰色说明小字。当前展示请使用文件名带inset的修订图。\n**关注点**：这份旧图的布局已被用户要求修改。\n'
        '\n### core_bifurcation_four_states_booklet_before_insets.pdf\n内嵌布局修订前的7页图册，保留用于比较视觉修改。其数值与当前图册一致。\n**关注点**：当前图册为不带before_insets后缀的文件。\n')
    print('PLOTS_COMPLETE',json.dumps(MANIFEST),flush=True)

if __name__=='__main__':
    main()

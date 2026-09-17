"""Scientific figures from continued orbits, critical modes and Floquet spectra.

Main figure uses one coordinate axis. No extrapolated links across unsolved gaps.
Native SNN numbered examples retain their true parameter/rate coordinates.
"""
from common import *
sys.path.insert(0,str(ROOT/'scripts/topic4_core_bifurcation_states_v3'))
import plot as p
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import resample,find_peaks

FIG=OUT/'figures';MEAN=r'$\langle r_{\mathrm{E,core\ A}}\rangle$'
COLORS=[p.BLUE,'#8059a3',p.GREEN]
CRITICAL=[('LP1','folds/burst_end_fold_N2048.json'),('PD1','flips/mixed_lower_flip_N2048.json'),('PD2','flips/mixed_flip_N2048.json'),('PD3','flips/tonic_lower_flip_N2048.json')]

def orbit(path,stable=True,**extra):
    z=np.load(path);r=z['r'];rr=resample(r,max(4096,len(r)),axis=0)
    return dict(g=float(z['g']),T=float(z['T']),r=r,mean=r.mean(0)*1000,lo=rr.min(0)*1000,hi=rr.max(0)*1000,path=str(path),stable=stable,**extra)

def collect():
    critical={name:read(OUT/file) for name,file in CRITICAL};seq=[]
    old=read(V4/'displayed_periodic_orbits.json')
    for family in ['low_burst','recruited_burst','mixed','tonic']:
        a=[orbit(x['path'],family=family) for x in old if x['family']==family]
        if family=='recruited_burst':
            for folder in ['dense_up','dense_down','join_doublet_down']:
                a.extend(orbit(x,family=family) for x in (OUT/'periodic'/folder).glob('*.npz'))
            for folder in ['doublet_up','doublet_down']:
                a.extend(orbit(x['source'],family=family) for x in read(OUT/'arcs'/folder/'progress.json'))
        seq.append(sorted(a,key=lambda x:x['g']))
    names=['burst_end','burst_end_continued','mixed_low','mixed_low_continued','mixed_high','mixed_high_continued','mixed_high_continued_N1024','mixed_high_fine','tonic_low','tonic_low_continued','tonic_low_continued_N1024','surround_recruited_back']
    for name in names:
        path=OUT/'arcs'/name/'progress.json'
        if not path.exists():continue
        a=[]
        for x in read(path):
            stable=(name=='burst_end' and x['index']<15) or (name=='mixed_low' and x['g']>=critical['PD1']['g']) or (name in ['mixed_high','mixed_high_fine'] and x['g']<=critical['PD2']['g']) or (name=='tonic_low' and x['g']>=critical['PD3']['g']) or (name=='surround_recruited_back' and x['index']<41)
            a.append(orbit(x['source'],stable=stable,family=name))
        seq.append(a)
    for name in ['burst_to_B200','mixed_to_B200','meet130_mixed_high_continued_N1024','meet130_tonic_low_continued_N1024']:
        path=OUT/'means'/name/'progress.json'
        if path.exists():seq.append([orbit(x['source'],stable=False,family=name) for x in read(path)])
    # Insert each exactly refined critical orbit into the appropriate arc segment.
    mapping={'LP1':'burst_end','PD1':'mixed_low','PD2':'mixed_high','PD3':'tonic_low'}
    for name,c in critical.items():
        a=next(a for a in seq if a and a[0]['family']==mapping[name]);b=orbit(c['source'],family=mapping[name]);k=min(range(len(a)),key=lambda i:abs(a[i]['g']-b['g']));
        if k and abs(a[k-1]['g']-b['g'])<abs(a[min(k+1,len(a)-1)]['g']-b['g']):a.insert(k,b)
        else:a.insert(k+1,b)
    # Assemble one traversal per connection. Overplotting the independently
    # traced reverse path would fill the gaps of dashed instability lines.
    stable=seq[:4];family_index={'burst_end':1,'mixed_low':2,'mixed_high':2,'mixed_high_fine':2,'tonic_low':3,'surround_recruited_back':1}
    for a in seq[4:]:
        if a and a[0]['family'] in family_index:stable[family_index[a[0]['family']]].extend(c for c in a if c['stable'])
    for a in stable:a.sort(key=lambda c:c['g'])
    def chain(names):
        a=[]
        for k,name in enumerate(names):
            rows=read(OUT/'arcs'/name/'progress.json')
            if k+1<len(names):
                init=read(OUT/'arcs'/names[k+1]/'initialization.json');matches=[i for i,x in enumerate(rows) if x['source']==init['current']]
                if matches:rows=rows[:matches[0]+1]
            a.extend(orbit(x['source'],stable=False,family='connector') for x in rows)
        return a
    a=chain(['burst_end','burst_end_continued']);a=[c for c in a if c['mean'][1]<=200 and not (Path(c['path']).parent.name=='burst_end' and int(Path(c['path']).name[5:8])<15)]
    b=chain(['mixed_low','mixed_low_continued']);b=[c for c in b if c['mean'][1]>=200 and not (Path(c['path']).parent.name=='mixed_low' and c['g']>=critical['PD1']['g'])]
    a.append(orbit(OUT/'means/burst_to_B200/mean200.000000.npz',stable=False,family='connector'))
    first=[orbit(critical['LP1']['source'],stable=False,family='connector')]+a+list(reversed(b))+[orbit(critical['PD1']['source'],stable=False,family='connector')]
    a=chain(['mixed_high','mixed_high_continued','mixed_high_continued_N1024']);a=[c for c in a if not (Path(c['path']).parent.name=='mixed_high' and c['g']<=critical['PD2']['g']) and not (Path(c['path']).parent.name.endswith('N1024') and int(Path(c['path']).name[5:8])>=25)]
    b=chain(['tonic_low','tonic_low_continued','tonic_low_continued_N1024']);b=[c for c in b if not (Path(c['path']).parent.name=='tonic_low' and c['g']>=critical['PD3']['g']) and not (Path(c['path']).parent.name.endswith('N1024') and int(Path(c['path']).name[5:8])>=40)]
    meet=OUT/'means/connection_mixed_high_continued_N1024/mean210.000000.npz'
    if meet.exists():a.append(orbit(meet,stable=False,family='connector'))
    second=[orbit(critical['PD2']['source'],stable=False,family='connector')]+a+list(reversed(b))+[orbit(critical['PD3']['source'],stable=False,family='connector')]
    return stable+[first,second],critical

def lines(ax,seq,group=0,envelope=False):
    for a in seq:
        if len(a)<2:continue
        xx=np.array([x['g'] for x in a]);mm=np.array([x['mean'][group] for x in a]);stable=np.array([x['stable'] for x in a])
        color=p.ORANGE if group==0 else COLORS[group]
        ax.plot(xx,mm,color=color,lw=2.5 if stable.all() else 1.65,ls='-' if stable.all() else '--',zorder=4)
        if envelope:
            for key in ['hi','lo']:
                yy=np.maximum(0,[x[key][group] for x in a]);coords=np.column_stack([xx,yy]);segments=np.stack([coords[:-1],coords[1:]],axis=1)
                # The mean carries the densely folded unstable connector; its
                # envelope is in the separate detailed rate figure.
                keep=stable[:-1]&stable[1:]
                ax.add_collection(LineCollection(segments[keep],colors=p.GREEN,linewidths=1.25,zorder=2))

def critical_markers(ax,critical,annotate=True):
    offsets={'LP1':(1.287,115),'PD1':(1.285,43),'PD2':(1.475,80),'PD3':(1.495,205)}
    for name,c in critical.items():
        y=c['mean_hz'][0];ax.scatter(c['g'],y,c='black',marker='s' if name.startswith('LP') else 'D',s=43,zorder=10)
        if annotate:ax.annotate(name,xy=(c['g'],y),xytext=offsets[name],fontsize=12,weight='bold',arrowprops=dict(arrowstyle='-',color='black',lw=.9))

def main_axis(ax,seq,native,critical):
    p.branches(ax,[],read(V2/'fold.json'),read(V2/'equilibrium_spectrum.json'))
    lines(ax,seq,envelope=True);critical_markers(ax,critical)
    for row in native:p.number_marker(ax,row['g'],row['mean'],row['number'],row['color'],size=260)
    ax.set(xlim=(.46,1.64),ylim=(-.05,510));ax.set_xticks([.5,.7,.85,1,1.15,1.3,1.45,1.6]);ax.set_xticklabels(['0.5','0.7','0.85','1.0','1.15','1.3','1.45','1.6'])
    ax.set_title('Core burst bifurcation diagram',loc='left',fontsize=19,weight='bold',pad=15)
    ax.legend(handles=[Line2D([],[],color=p.BLUE,label='Stable equilibrium'),Line2D([],[],color=p.RED,ls='--',label='Unstable equilibrium'),Line2D([],[],color=p.ORANGE,lw=2.5,label='Stable period mean  '+MEAN),Line2D([],[],color=p.ORANGE,ls='--',label='Unstable period mean'),Line2D([],[],color=p.GREEN,label='Stable periodic maximum / minimum'),Line2D([],[],color='black',marker='o',ls='none',label='Native SNN mean (1–4)')],loc='upper left',frameon=False,fontsize=11)
    assert not ax.child_axes

def main_plots(seq,native,critical,book):
    fig,ax=plt.subplots(figsize=(9.6,8.6));fig.subplots_adjust(left=.12,right=.965,bottom=.115,top=.91);main_axis(ax,seq,native,critical)
    assert len(fig.axes)==1
    p.save(fig,'00_connected_core_bifurcation','单坐标轴显示原平衡分支、稳定周期均值与峰谷，以及延拓得到的不稳定周期均值。LP1为周期轨道鞍结，PD1–PD3为倍周期点；均值线的陡升本身不作分岔证据。','实线与虚线表示稳定性；编号1–4仍是原生SNN的实际均值坐标。1.176附近的低周边活动周期族与招募周边活动周期族不强行补线。',book)
    fig=plt.figure(figsize=(12.8,12.3));ax=fig.add_axes([.085,.525,.89,.39]);main_axis(ax,seq,native,critical);p.native_panel(fig,native,box=(.085,.09,.89,.285),title_y=.439)
    p.save(fig,'01_connected_bifurcation_four_states','上方为同一单轴分岔图；下方是低活动、不规则burst、中间态、规则burst的原生SNN波形与raster。四个编号与主图逐一对应。','下排不是降阶周期轨道生成的伪spike图，不能将不稳定周期支等同于原生SNN的irregular状态。',book)
    fig,axs=plt.subplots(2,1,figsize=(10.5,10.3),sharex=True);fig.subplots_adjust(left=.12,right=.96,bottom=.09,top=.92,hspace=.28)
    for i,ax in enumerate(axs):
        lines(ax,seq,group=i);ax.set(xlim=(1.23,1.43),ylim=(-3,405),ylabel=f'Core {"AB"[i]} period mean (Hz)');ax.set_title(f'Core {"AB"[i]}',loc='left',weight='bold')
        for name,c in critical.items():ax.scatter(c['g'],c['mean_hz'][i],s=45,marker='s' if name.startswith('LP') else 'D',color='black',zorder=9)
    axs[1].set_xlabel(p.JLABEL);axs[0].legend(handles=[Line2D([],[],color=p.ORANGE,label='Stable periodic orbit'),Line2D([],[],color=p.ORANGE,ls='--',label='Unstable periodic orbit')],frameon=False,loc='upper left')
    p.save(fig,'02_connected_period_means_linear','线性纵轴分别展示同一联合系统的A核和B核完整周期均值。两侧延拓的连接检验使用全部六群体波形和周期，而非只凭均值相交。','多次折返属于已延拓的不稳定周期支；同一参数下多个稳定轨道可共存，不能画成唯一状态随参数的单值函数。',book)

def smooth_change(seq,book):
    allc=[c for a in seq for c in a];gg=[1.244,1.2462,1.248,1.254];chosen=[min(allc,key=lambda c:abs(c['g']-g)) for g in gg]
    fig,axs=plt.subplots(4,2,figsize=(11.8,10.8),gridspec_kw={'width_ratios':[1.5,1]});fig.subplots_adjust(left=.085,right=.97,bottom=.075,top=.89,wspace=.28,hspace=.6)
    obs=[]
    for k,c in enumerate(chosen):
        r=resample(c['r'],8192,axis=0)*1000;T=c['T'];r=np.roll(r,round(.35*len(r))-np.argmax(r[:,0]),axis=0);t=np.arange(len(r))*T/len(r)
        for j in range(3):axs[k,0].plot(t,r[:,j],color=COLORS[j],lw=1.6,label=['Core A E','Core B E','Surround E'][j])
        axs[k,0].axhline(c['mean'][0],color=p.ORANGE,ls='--',lw=1.5);axs[k,0].set(xlim=(0,T),ylim=(-5,370),ylabel='Rate (Hz)',xlabel='Time (ms)');axs[k,0].set_title(f'{p.JLABEL} = {c["g"]:g}',loc='left',fontsize=12)
        axs[k,1].plot(t,r[:,0],color=p.BLUE);axs[k,1].fill_between(t,0,r[:,0],color=p.BLUE,alpha=.18);axs[k,1].set(xlim=(.28*T,.65*T),ylim=(-5,350),xlabel='Time (ms)')
        dur=float(T*(r[:,0]>100).mean());obs.append(dict(g=c['g'],T_ms=T,A_mean_hz=float(c['mean'][0]),A_area_spikes_per_neuron=float(c['mean'][0]*T/1000),A_above_100_hz_ms=dur,source=c['path']))
        axs[k,1].set_title(f'Mean {c["mean"][0]:.2f} Hz   |   T {T:.2f} ms',loc='left',fontsize=11)
    fig.text(.085,.955,'A steep mean-rate change on one stable periodic branch',fontsize=18,weight='bold');fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.08,.937),ncol=3,frameon=False)
    p.save(fig,'03_continuous_waveform_doublet','对1.244–1.254加密后，A核波形先延长，再出现第二峰，完整周期变化很小。两端在J=1.2462求得相同的六群体周期解，且局部Floquet抽样均稳定。','第二峰增加每周期放电面积，使均值迅速上升；一个周期内双峰不是周期翻倍，未发现这里有局部轨道分岔。',book);write('doublet_waveform_observables.json',obs)

def floquet_points():
    rows=[]
    for path in (OUT/'poincare').rglob('*.json'):
        a=read(path)
        if a.get('method')=='rk4':a['file']=str(path);rows.append(a)
    return rows

def eigen_figures(critical,book):
    murows=floquet_points();fig,axs=plt.subplots(2,2,figsize=(11.5,8.6));fig.subplots_adjust(left=.09,right=.96,bottom=.1,top=.89,wspace=.3,hspace=.55)
    folders={'LP1':['burst_end'],'PD1':['mixed_low'],'PD2':['mixed_high_fine'],'PD3':['tonic_low']};ranges={'LP1':(-3e-7,1e-8),'PD1':(-1e-4,1e-4),'PD2':(-4e-7,2e-7),'PD3':(-1.5e-4,1e-4)};scales={'LP1':1e7,'PD1':1e5,'PD2':1e7,'PD3':1e5}
    for ax,(name,c) in zip(axs.flat,critical.items()):
        a=[x for x in murows if Path(x['source']).parent.name in folders[name] and ranges[name][0]<x['g']-c['g']<ranges[name][1]]
        a.extend(x for x in murows if Path(x['source']).name==Path(c['source']).name)
        a.sort(key=lambda x:x['g']);ax.scatter([(x['g']-c['g'])*scales[name] for x in a],[x['multipliers'][0][0] for x in a],c=['black' if abs(x['g']-c['g'])<1e-12 else p.BLUE if x['max_transverse']<1 else p.RED for x in a],s=45,zorder=4)
        ax.axhline(1 if name=='LP1' else -1,color='black',ls='--',lw=1);ax.axvline(0,color='black',ls=':',lw=1)
        ax.set(xlabel=rf'$J-J_c$ ($10^{{-{int(np.log10(scales[name]))}}}$)',ylabel='Transverse Floquet multiplier',ylim=((0,2) if name=='LP1' else (-2,.1)));ax.set_title(f'{name}   $J_c$ = {c["g"]:.10f}',loc='left',fontsize=12)
    fig.text(.09,.952,'Critical eigenvalues of the return map',fontsize=18,weight='bold')
    p.save(fig,'04_critical_floquet_multipliers','LP1的非平凡Floquet乘子到达+1；三个PD点的实乘子穿过−1。自治相位方向在Poincaré截面上投影去除，临界条件另由周期/反周期线性化零空间独立确认。','只把临界附近已计算乘子画成点；极强不稳定区的单周期传播可能病态，不能据其浮点误差判定新的临界点。',book)
    fig,axs=plt.subplots(4,2,figsize=(11.5,11));fig.subplots_adjust(left=.095,right=.96,bottom=.07,top=.84,wspace=.28,hspace=.66);mode_table=[]
    for k,(name,c) in enumerate(critical.items()):
        z=np.load(c['source']);N=len(z['r']);v=z['right_null'][:-1].reshape(N,6) if name=='LP1' else z['mode'];lv=z['left_null'][:-1].reshape(N,6) if name=='LP1' else z['left_mode']
        # Keep the collocation phase: an antiperiodic eigenfunction cannot be
        # circularly shifted as if its endpoints had the same sign.
        v=v.copy();v/=np.max(abs(v));energy=np.sum(lv*lv,axis=0);energy/=energy.sum()
        phase=np.arange(N)/N
        for j in range(3):axs[k,0].plot(phase,v[:,j],color=COLORS[j],lw=1.5,label=['Core A E','Core B E','Surround E'][j])
        for j in range(3):axs[k,0].plot(phase,v[:,j+3],color=COLORS[j],ls='--',lw=1.5,label=['Core A I','Core B I','Surround I'][j])
        axs[k,0].set(xlim=(0,1),ylim=(-1.08,1.08),xlabel='Phase / period',ylabel='Right rate mode');axs[k,0].set_title(name+('   cycle fold' if name=='LP1' else '   period doubling'),loc='left',fontsize=12)
        axs[k,1].bar(np.arange(6),energy,color=COLORS+COLORS,alpha=.85);axs[k,1].set_xticks(range(6));axs[k,1].set_xticklabels(['A E','B E','S E','A I','B I','S I']);axs[k,1].set(ylabel='Left rate-mode fraction',ylim=(0,1));mode_table.append(dict(name=name,right_rate_fraction=(np.sum(v*v,axis=0)/np.sum(v*v)).tolist(),left_rate_fraction=energy.tolist()))
    fig.text(.095,.95,'Right critical modes and adjoint rate-mode components',fontsize=17,weight='bold');fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.09,.932),ncol=3,frameon=False)
    p.save(fig,'05_critical_right_and_left_modes','左列为周期边值问题的临界右率模态；PD使用反周期边界条件，LP使用相位约束后的固定参数零模态。右列给出对应伴随率模态六群体的平方范数比例。','这是六群体降阶模型的模态分量，非原生SNN逐细胞eigenvector；LP还含周期变化分量，右列只比较率分量。',book);write('critical_mode_components.json',mode_table)

def period_doubling(critical,book):
    fig,axs=plt.subplots(1,3,figsize=(13,4.7));fig.subplots_adjust(left=.07,right=.98,bottom=.19,top=.8,wspace=.34)
    data=[]
    for ax,name,folder in zip(axs,['PD1','PD2','PD3'],['mixed_lower_period2','mixed_period2','tonic_lower_period2']):
        c=critical[name];rows=sorted([read(x) for x in (OUT/'periodic'/folder).glob('*.json')],key=lambda x:x['amplitude']);stable=name!='PD2'
        xx=[(x['g']-c['g'])*1e8 for x in rows];yy=[x['half_period_difference'] for x in rows];ax.plot([0]+xx,[0]+yy,'o-' if stable else 'o--',color=p.BLUE if stable else p.RED,ms=6)
        ax.axvline(0,color='black',ls=':',lw=1);ax.set(xlabel=r'$J-J_c$ ($10^{-8}$)',ylabel='Half-period difference',title=name+('  supercritical' if stable else '  subcritical'));data.extend(dict(name=name,stable=stable,**x) for x in rows)
    fig.text(.07,.92,'The computed branches with twice the parent period',fontsize=18,weight='bold')
    p.save(fig,'06_period_doubled_branches','从反周期临界模态切换到2T周期边值问题，并用非零半周期差排除重复两遍的T周期解。PD2子支位于母支稳定侧且不稳定；PD1和PD3在降低J时产生稳定2T子支。','超/亚临界判断依赖实际求得的子支方向与Floquet稳定性；参数窗极窄，须使用局部坐标查看。',book);write('period_doubled_branch_table.json',data)

def switching(book):
    names=['after_burst_fold','after_mixed_flip'];fig,axs=plt.subplots(2,1,figsize=(10.8,7.2));fig.subplots_adjust(left=.11,right=.96,bottom=.11,top=.88,hspace=.42)
    for ax,name in zip(axs,names):
        path=OUT/'transitions'/f'{name}_dt0.025.npz'
        if not path.exists():path=OUT/'transitions'/f'{name}_dt0.05.npz'
        z=np.load(path);r=z['r']*1000;t=np.arange(len(r))*float(z['dt'])/1000
        for i in range(2):ax.plot(t,r[:,i],color=COLORS[i],lw=1,label=f'Core {"AB"[i]} E')
        ax.set(xlim=(0,2),ylim=(-8,420),xlabel='Time after parameter step (s)',ylabel='Rate (Hz)');ax.set_title(('Beyond LP1' if name==names[0] else 'Beyond PD2')+f'   {p.JLABEL} = {float(z["g"]):g}',loc='left',fontsize=13)
    fig.text(.11,.947,'Observed destinations after loss of stability',fontsize=18,weight='bold');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.96,.928),ncol=2,frameon=False)
    p.save(fig,'07_switching_after_bifurcations','从临界周期轨道的完整延迟历史出发，小幅提高J并积分8秒。越过LP1后到达A核burst/B核高率状态；越过PD2后到达两核高率振荡。','这是给定初始历史和参数阶跃的去向，不能把局部分岔定理解释为对所有初始条件都只有一个去向。',book)

def extrema(seq,book):
    fig,axs=plt.subplots(2,1,figsize=(10.5,8.7),sharex=True);fig.subplots_adjust(left=.12,right=.96,bottom=.1,top=.91,hspace=.3)
    for ax,key,label in zip(axs,['hi','lo'],['Periodic maximum','Periodic minimum']):
        for a in seq:
            stable=all(c['stable'] for c in a);ax.plot([c['g'] for c in a],np.maximum(0,[c[key][0] for c in a]),color=p.GREEN,lw=2 if stable else 1.35,ls='-' if stable else '--')
        ax.set(xlim=(1.23,1.43),ylabel='Core A E rate (Hz)',title=label)
    axs[1].set_xlabel(p.JLABEL);axs[0].legend(handles=[Line2D([],[],color=p.GREEN,label='Stable periodic orbit'),Line2D([],[],color=p.GREEN,ls='--',label='Unstable periodic orbit')],frameon=False)
    p.save(fig,'08_stable_unstable_periodic_extrema','补充展示同一批稳定与不稳定周期轨道的A核最大值、最小值，作为完整周期均值图的经典极值分岔读出。所有极值来自周期波形的傅里叶重采样。','均值、峰值、谷值是同一条周期支的不同投影，局部投影折返不应自动当成新的动力学分岔。',book)

def main():
    seq,critical=collect();native=p.load_native();FIG.mkdir(exist_ok=True);p.FIG=FIG;p.OUT=OUT;p.MANIFEST.clear();p.DESCRIPTIONS.clear()
    with PdfPages(FIG/'core_branch_connections.pdf') as book:
        main_plots(seq,native,critical,book);smooth_change(seq,book);eigen_figures(critical,book);period_doubling(critical,book);switching(book);extrema(seq,book)
    write('figure_manifest.json',p.MANIFEST);write('displayed_curve_sequences.json',[[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items() if k!='r'} for c in a] for a in seq])
    (FIG/'README.md').write_text('# 核内周期分支连接与分岔\n\n'+'\n'.join(p.DESCRIPTIONS)+'\n### core_branch_connections.pdf\n九页图册汇总主分岔图、四状态合图、两核均值、波形变化、临界乘子与模态、倍周期子支、切换轨迹和周期极值。\n**关注点**：候选图已供本地自查，仍待用户目视检查。\n')
    print('FIGURES_COMPLETE',json.dumps(p.MANIFEST),flush=True)

if __name__=='__main__':main()

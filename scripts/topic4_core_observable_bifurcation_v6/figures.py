"""Pair core projections without confusing orbit identity and observables."""
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import resample
from PIL import Image
from peak_exchange import peaks

sys.path.insert(0,str(ROOT/'scripts/topic4_core_bifurcation_states_v3'))
import plot as native_plot

FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
J=r'$J_{\mathrm{EE,core}}$'
BLUE='#2366a2';RED='#bb4437';ORANGE='#cf8d31';GREEN='#22856d';CYAN='#2799b0';PURPLE='#8059a3'
FAMILY=[BLUE,ORANGE,PURPLE,RED];GROUP=[BLUE,PURPLE,GREEN]
MANIFEST=[];CAPTIONS=[]
plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.titlesize':14,'legend.fontsize':11,
 'xtick.labelsize':11,'ytick.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
 'font.family':'DejaVu Sans','pdf.fonttype':42,'savefig.facecolor':'white'})

def save(fig,name,description,focus):
    assert all(not ax.child_axes for ax in fig.axes)
    for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',dpi=190)
    with Image.open(FIG/f'{name}.png') as im:im.load();size=list(im.size)
    MANIFEST.append(dict(name=name,pixels=size,axes=len(fig.axes),insets=0))
    CAPTIONS.append(f'### {name}.png / .pdf\n{description}\n**关注点**：{focus}\n')
    plt.close(fig)

def orbit(path,stable=True):
    z=np.load(path);r=z['r'];rr=resample(r,max(8192,len(r)),axis=0)
    return dict(g=float(z['g']),T=float(z['T']),mean=(r.mean(0)*1000).tolist(),
        lo=(rr.min(0)*1000).tolist(),hi=(rr.max(0)*1000).tolist(),path=str(path),stable=stable)

def critical():
    names={'LP1':'folds/burst_end_fold_N4096.json','PD1':'flips/mixed_lower_flip_N2048.json',
           'PD2':'flips/mixed_flip_N2048.json','PD3':'flips/tonic_lower_flip_N2048.json',
           'LP0a':'folds/surround_first_fold_N2048.json','LP0b':'folds/surround_second_fold_N2048.json',
           'LP0c':'folds/surround_recruited_fold_N4096.json'}
    return {key:read(V5/path) for key,path in names.items()}

def collect(c):
    seq=read(V5/'displayed_curve_sequences.json')
    low=read(V4/'arcs/recruitment_turn/progress.json')
    seq[0]+= [orbit(row['source']) for row in low[:5]]
    seq[0]+=[orbit(c['LP0a']['source'])]
    seq[0].sort(key=lambda x:x['g'])
    seq[1]+=[orbit(c['LP0c']['source'])];seq[1].sort(key=lambda x:x['g'])
    return seq

def curve(ax,rows,key,group,color,lw=2,style=None):
    if len(rows)<2:return
    ax.plot([x['g'] for x in rows],np.maximum(0,[x[key][group] for x in rows]),
            color=color,lw=lw,ls=style or ('-' if rows[0]['stable'] else '--'))

def main_axis(ax,seq,c,group,native):
    fold=read(V2/'fold.json');eq=read(V2/'equilibrium_spectrum.json')
    low=sorted([x for x in eq if x['direction']==-1],key=lambda x:x['g'])
    up=[x for x in eq if x['direction']==1]
    for a,col,ls in [(low,BLUE,'-'),(up,RED,'--')]:
        xx=[x['g'] for x in a];yy=[x['r_hz'][group] for x in a]
        if ls=='-':xx.append(fold['g']);yy.append(fold['r_hz'][group])
        else:xx.insert(0,fold['g']);yy.insert(0,fold['r_hz'][group])
        ax.plot(xx,yy,color=col,ls=ls,lw=2)
    ax.plot(up[-1]['g'],up[-1]['r_hz'][group],marker='o',mfc='white',mec=RED,ms=4)
    for k,a in enumerate(seq):
        curve(ax,a,'mean',group,ORANGE,2.4 if k<4 else 1.5)
        if k<4:
            curve(ax,a,'hi',group,GREEN,1.5);curve(ax,a,'lo',group,CYAN,1.5)
    # Period-doubled branches are computed at tiny parameter offsets; retain
    # them here, and show their separate local figure from v5 for legibility.
    for name,folder in [('PD1','mixed_lower_period2'),('PD2','mixed_period2'),('PD3','tonic_lower_period2')]:
        a=[orbit(c[name]['source'],name!='PD2')]+[orbit(p,name!='PD2') for p in sorted((V5/'periodic'/folder).glob('*.npz'))]
        curve(ax,a,'mean',group,PURPLE,2)
    for row in native:
        native_plot.number_marker(ax,row['g'],row['means'][group],row['number'],row['color'],size=185)
    off=[{'LP1':(1.285,100),'PD1':(1.27,43),'PD2':(1.49,75),'PD3':(1.49,200)},
         {'LP1':(1.29,43),'PD1':(1.23,155),'PD2':(1.47,130),'PD3':(1.50,260)}][group]
    for name in ['LP1','PD1','PD2','PD3']:
        a=c[name];xy=(a['g'],a['mean_hz'][group]);ax.scatter(*xy,c='black',marker='s' if name.startswith('LP') else 'D',s=36,zorder=10)
        ax.annotate(name,xy,xytext=off[name],fontsize=11,weight='bold',arrowprops=dict(arrowstyle='-',lw=.9,color='black'))
    ax.scatter(fold['g'],fold['r_hz'][group],c='black',s=38,zorder=8)
    ax.annotate('Fold 1.1254',(fold['g'],fold['r_hz'][group]),xytext=(.83,.10 if group else .30),arrowprops=dict(arrowstyle='-',color='black'),fontsize=11)
    ax.annotate('R',(1.176275,23.8 if group==0 else 14.7),xytext=(1.11,55 if group==0 else 38),fontsize=13,weight='bold',arrowprops=dict(arrowstyle='-',color='black'))
    ax.set(xlim=(.46,1.64),ylim=(-.05,510),xlabel=J,ylabel=f'Core {"AB"[group]} E rate (Hz / neuron)')
    ax.set_yscale('symlog',linthresh=1);ax.set_yticks([0,.5,1,3,10,30,100,300]);ax.set_yticklabels(['0','0.5','1','3','10','30','100','300'])
    ax.set_xticks([.5,.7,.85,1,1.15,1.3,1.45,1.6]);ax.set_xticklabels(['0.5','0.7','0.85','1.0','1.15','1.3','1.45','1.6'])
    ax.set_title('Core '+ 'AB'[group],loc='left',weight='bold',fontsize=20,pad=14)

def legend():
    return [Line2D([],[],color=BLUE,label='Stable equilibrium'),Line2D([],[],color=RED,ls='--',label='Unstable equilibrium'),
      Line2D([],[],color=ORANGE,lw=2.4,label=r'Stable periodic mean  $\langle r_E\rangle$'),Line2D([],[],color=ORANGE,ls='--',label='Unstable periodic mean'),
      Line2D([],[],color=GREEN,label='Stable periodic maximum'),Line2D([],[],color=CYAN,label='Stable periodic minimum'),
      Line2D([],[],color='black',marker='o',ls='none',label='Native SNN mean (1–4)')]

def main_plots(seq,c):
    native=native_plot.load_native()
    for row in native:
        z=np.load(row['source']);sel=(row['t']>=2)&(row['t']<20);means=[]
        for name in ['coreAE','coreBE']:
            i=z['group_names'].tolist().index(name);means.append(float((z['spike_counts_2ms'][sel,i]/int(z['group_sizes'][i])/.002).mean()))
        row['means']=means
    write('native_AB_coordinates.json',[{k:x[k] for k in ['number','g','means','source','source_sha256']} for x in native])
    fig,axs=plt.subplots(1,2,figsize=(17.8,8.4));fig.subplots_adjust(left=.06,right=.98,bottom=.115,top=.91,wspace=.21)
    for i,ax in enumerate(axs):main_axis(ax,seq,c,i,native);ax.legend(handles=legend(),loc='upper left',frameon=False,fontsize=10.5)
    save(fig,'00_core_AB_bifurcation','相同联合网络的A核和B核投影，横纵坐标与稳定性规则完全一致。均值、最大值和最小值分别用橙、绿、青色，R标记1.176附近需放大查看的招募区域。','A/B不是分别独立计算的两个分岔系统；同一分岔参数和全网络谱在两张图中共享，纵坐标读出不同。')
    for i in range(2):
        fig,ax=plt.subplots(figsize=(9.4,8.5));fig.subplots_adjust(left=.12,right=.96,bottom=.115,top=.91)
        main_axis(ax,seq,c,i,native);ax.legend(handles=legend(),loc='upper left',frameon=False)
        save(fig,f'0{i+1}_core_{"AB"[i]}_bifurcation',f'Core {"AB"[i]}的独立近方形分岔图，不含内嵌坐标轴。编号1–4来自同一批原生SNN的对应核均值，周期支来自确定性降阶模型。','R区的尖点与右侧共存轨道需结合局部展开图阅读；2T子支参数窗极小，另见V5倍周期子支图。')
    # Preserve the requested four native-SNN states below the updated A diagram.
    fig=plt.figure(figsize=(12.8,12.3));ax=fig.add_axes([.085,.525,.89,.39]);main_axis(ax,seq,c,0,native)
    ax.legend(handles=legend(),loc='upper left',frameon=False,fontsize=10)
    native_plot.native_panel(fig,native,box=(.085,.09,.89,.285),title_y=.439)
    save(fig,'03_core_A_four_native_states','更新后的A核分岔图下保留低活动、不规则burst、中间态和规则burst的原生SNN波形与raster。所有编号都放在对应实际参数和原生均值处。','未将降阶周期波形伪造为spike raster；这些SNN状态与确定性分岔阈值不能直接等同。')

def left_branches(c):
    rows=read(V4/'arcs/recruitment_turn/progress.json')
    low=[orbit(V4/'periodic/low_burst/g1.17625000_N2048.npz')] if (V4/'periodic/low_burst/g1.17625000_N2048.npz').exists() else []
    old=read(V5/'displayed_curve_sequences.json')[0];low=[x for x in old if x['g']>1.1762]
    low+=[orbit(row['source']) for row in rows[:5]]+[orbit(c['LP0a']['source'])]
    low.append(orbit(OUT/'periodic/low_coexist_exact/g1.17627535_N4096.npz'));low.sort(key=lambda x:x['g'])
    highrows=read(V5/'arcs/surround_recruited_back/progress.json')
    high=[orbit(x['source']) for x in highrows[:41]]+[orbit(c['LP0c']['source'])];high.sort(key=lambda x:x['g'])
    highunst=[orbit(c['LP0c']['source'],False)]+[orbit(x['source'],False) for x in highrows[41:49]]
    lowunst=[orbit(c['LP0a']['source'],False),orbit(rows[5]['source'],False),orbit(c['LP0b']['source'],False)]
    return low,high,lowunst,highunst,highrows

def recruitment(c):
    low,high,lu,hu,raw=left_branches(c);gx=read(OUT/'peak_exchange.json')
    fig,axs=plt.subplots(2,2,figsize=(12.5,9.7));fig.subplots_adjust(left=.09,right=.975,bottom=.095,top=.88,wspace=.28,hspace=.42)
    for i,ax in enumerate(axs.flat[:3]):
        for a,col in [(low,BLUE),(high,ORANGE),(lu,BLUE),(hu,ORANGE)]:curve(ax,a,'mean',i,col)
        for name,col in [('LP0a',BLUE),('LP0c',ORANGE)]:
            q=c[name];ax.scatter(q['g'],q['mean_hz'][i],c=col,marker='s',s=47,zorder=8)
        ax.axvline(gx['g'],color='black',lw=.8,ls=':')
        ax.set(xlim=(1.17623,1.176335),xlabel=J,ylabel='Period mean (Hz)',title=['Core A','Core B','Surround E'][i]);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.set_xticks([1.17624,1.17628,1.17632]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    axs[0,0].set_ylim(15.5,27.1)
    axs[0,0].annotate('LP0c',(c['LP0c']['g'],c['LP0c']['mean_hz'][0]),xytext=(1.17628,25.8),arrowprops=dict(arrowstyle='-',color='black'),fontsize=12)
    axs[0,0].annotate('LP0a / LP0b',(c['LP0a']['g'],c['LP0a']['mean_hz'][0]),xytext=(1.176267,18.4),arrowprops=dict(arrowstyle='-',color='black'),fontsize=11)
    ax=axs[1,1];pp=[]
    for row in raw[20:41]:
        z=np.load(row['source']);pp.append(dict(g=float(z['g']),**peaks(z['r'],float(z['T']))))
    pp.sort(key=lambda a:a['g']);x=[a['g'] for a in pp]
    ax.plot(x,[a['primary_hz'] for a in pp],color=BLUE,lw=1.6,label='Earlier A peak')
    ax.plot(x,[a['secondary_hz'] for a in pp],color=PURPLE,lw=1.6,label='Later A peak')
    ax.plot(x,[max(a['primary_hz'],a['secondary_hz']) for a in pp],color=GREEN,lw=3,alpha=.7,label='Global maximum')
    ax.scatter(gx['g'],gx['primary_hz'],c='black',s=38,zorder=7)
    ax.set(xlim=(1.176248,1.176315),ylim=(220,310),xlabel=J,ylabel='A peak rate (Hz)',title='Which peak is the maximum?')
    ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.set_xticks([1.17625,1.17628,1.17631]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax.legend(frameon=False,loc='lower left',fontsize=10)
    fig.suptitle('R: narrow cycle folds and a change of the tallest peak',x=.09,ha='left',fontsize=18,weight='bold')
    fig.legend(handles=[Line2D([],[],color=BLUE,label='Low-surround cycle'),Line2D([],[],color=ORANGE,label='Recruited-surround cycle'),Line2D([],[],color='black',ls='--',label='Unstable limb')],loc='upper left',bbox_to_anchor=(.082,.95),ncol=3,frameon=False)
    save(fig,'04_recruitment_region','放大约1.17625–1.17630的周期支，分别显示A、B与周边E的均值及A两个局部峰。LP0a/b是间隔约1.82e−8的低周边支折返，LP0c是招募周边支的折点；本尺度下a/b无法分开。','最大值在J=1.1762741938由两个不同时间的峰交换身份，因此可以出现尖角而周期轨道本身仍稳定光滑；两族未证实的全局连接不补线。')
    write('recruitment_peak_tracks.json',pp)

def waveform(path,n=8192):
    z=np.load(path);r=resample(z['r'],n,axis=0)*1000;T=float(z['T'])
    r=np.roll(r,n//2-int(np.argmax(r[:,2])),axis=0);return (np.arange(n)-n//2)*T/n,r,T

def coexist_left():
    cases=[('Low-surround cycle',OUT/'periodic/low_coexist_exact/g1.17627535_N4096.npz'),('Recruited-surround cycle',OUT/'periodic/recruited_coexist_exact/g1.17627535_N4096.npz')]
    fig,axs=plt.subplots(3,2,figsize=(12.6,9.5),sharex=True,sharey='row');fig.subplots_adjust(left=.09,right=.97,bottom=.1,top=.84,wspace=.18,hspace=.28)
    table=[]
    for k,(name,path) in enumerate(cases):
        t,r,T=waveform(path);a=orbit(path);table.append(dict(name=name,**a))
        for i in range(3):
            ax=axs[i,k];ax.plot(t,r[:,i],color=GROUP[i],lw=1.6);ax.axhline(a['mean'][i],color=ORANGE,ls='--',lw=1.3)
            ax.set(xlim=(-220,150),ylim=(-5,310 if i<2 else 215))
            ax.text(.04,.86,f'Mean {a["mean"][i]:.2f} Hz',transform=ax.transAxes,fontsize=11)
            if k==0:ax.set_ylabel(['Core A E (Hz)','Core B E (Hz)','Surround E (Hz)'][i])
        axs[0,k].set_title(f'{name}\nT = {T:.3f} ms',loc='left',pad=12)
        axs[-1,k].set_xlabel('Time relative to surround maximum (ms)')
    fig.suptitle(f'Two stable cycles at the same {J} = {table[0]["g"]:.10f}',x=.09,ha='left',fontsize=18,weight='bold')
    save(fig,'05_same_J_recruitment_waveforms','在完全相同J=1.1762753544求解两条完整周期轨道，两者的Floquet横向乘子模都小于1。招募支的周边活动显著增强，A出现第二次放电而B的完整周期均值反而降低。','每列的A、B、周边使用同一时间轴与同一条六群体轨道；虚横线表示该群体完整周期均值。')
    write('same_J_recruitment.json',table)

def right_projections(seq,c):
    fig,axs=plt.subplots(2,3,figsize=(15.8,8.8),sharex=True,sharey='col');fig.subplots_adjust(left=.07,right=.98,bottom=.1,top=.84,wspace=.21,hspace=.36)
    for group in range(2):
        for k,key in enumerate(['mean','hi','lo']):
            ax=axs[group,k]
            for j,a in enumerate(seq):curve(ax,a,key,group,FAMILY[j] if j<4 else '#292929',2.3 if j<4 else 1.3)
            for name in ['LP1','PD1','PD2','PD3']:
                q=c[name];v=orbit(q['source']);ax.scatter(q['g'],v[key][group],c='black',s=22,marker='s' if name=='LP1' else 'D',zorder=8)
            ax.set(xlim=(1.23,1.43),ylabel=f'Core {"AB"[group]} (Hz)',title=['Period mean','Periodic maximum','Periodic minimum'][k],ylim=((-8,405) if k!=1 else (305,398)))
            if group==1:ax.set_xlabel(J)
    # Different branches are named on the mean projection where they separate.
    for g,name,text in [(0,'PD1',(1.245,130)),(0,'PD2',(1.405,140)),(0,'PD3',(1.395,265)),(1,'LP1',(1.285,105))]:
        q=c[name];axs[g,0].annotate(name,(q['g'],q['mean_hz'][g]),xytext=text,fontsize=10,arrowprops=dict(arrowstyle='-',color='black'))
    fig.suptitle('One joint branch, six observable projections',x=.07,ha='left',fontsize=19,weight='bold')
    fig.legend(handles=[Line2D([],[],color=ORANGE,label='Recruited bursts'),Line2D([],[],color=PURPLE,label='A bursting / B high'),Line2D([],[],color=RED,label='A high / B high'),Line2D([],[],color='#292929',ls='--',label='Unstable T-period branches')],loc='upper left',bbox_to_anchor=(.065,.94),ncol=4,frameon=False)
    save(fig,'06_AB_mean_maximum_minimum','按轨道状态着色，把A、B的均值、最大值、最小值拆为六张等价投影。黑虚线保留已连接的不稳定T周期支，稳定支不跨共存轨道端点强行相连。','右侧A谷值从接近0到超过100Hz的空隙是不同稳定周期族的差别；虚线继续经过中间读出值，并不是一个稳定谷值函数发生无穷陡跳变。')

def coexist_right():
    paths=[V4/'periodic/mixed/g1.38000000_N2048.npz',V4/'periodic/tonic/g1.38000000_N2048.npz']
    # Use the actual stored paths in the accepted branch inventory.
    seq=read(V5/'displayed_curve_sequences.json');paths=[Path(min(seq[i],key=lambda a:abs(a['g']-1.38))['path']) for i in (2,3)]
    fig,axs=plt.subplots(2,2,figsize=(12.5,8.0),sharex=True,sharey=True);fig.subplots_adjust(left=.09,right=.97,bottom=.11,top=.80,wspace=.2,hspace=.43)
    table=[]
    for row,path in enumerate(paths):
        t,r,T=waveform(path);a=orbit(path);assert abs(a['g']-1.38)<1e-12;table.append(a)
        for i in range(2):
            ax=axs[row,i];ax.plot(t,r[:,i],color=GROUP[i],lw=2);ax.axhline(a['mean'][i],color=ORANGE,ls='--',lw=1.3)
            ax.set(xlim=(-70,70),ylim=(-8,420),ylabel='Rate (Hz)',title=f'Core {"AB"[i]}   mean {a["mean"][i]:.1f} / min {a["lo"][i]:.1f} / max {a["hi"][i]:.1f} Hz')
            if row==1:ax.set_xlabel('Time relative to surround maximum (ms)')
        axs[row,0].text(.02,.82 if row==0 else .08,('A bursting / B high' if row==0 else 'A high / B high')+f'\nT = {T:.2f} ms',transform=axs[row,0].transAxes,fontsize=11)
    fig.suptitle(f'Coexisting stable states at {J} = 1.38',x=.09,ha='left',fontsize=19,weight='bold')
    save(fig,'07_same_J_right_waveforms','固定J=1.38，比较A burst/B高活动与两核高活动的两条稳定周期解。A峰值几乎一致，谷值和周期均值却显著不同；B在两条轨道上均保持较高谷值。','周期振荡是在高背景上调制还是伴随近零静默，必须联合均值和谷值判断；不能仅凭最大值称为同一种burst。')
    write('same_J_right.json',table)

def projection_intersection(seq):
    q=read(OUT/'projection_intersection.json');fig,axs=plt.subplots(1,3,figsize=(15,4.9));fig.subplots_adjust(left=.06,right=.98,bottom=.18,top=.76,wspace=.27)
    ax=axs[0]
    for name,key,col,ls in [('stable','lo',CYAN,'-'),('unstable','mean',ORANGE,'--')]:
        a=[orbit(path,name=='stable') for path in (OUT/'periodic'/('projection_'+name)).glob('*.npz')]
        if name=='stable':a+=seq[3]
        a.sort(key=lambda x:x['g']);curve(ax,a,key,0,col,style=ls)
    ax.scatter(q['g'],q['stable']['refined_A_min_hz'],c='black',s=48,zorder=8);ax.set(xlim=(1.3775,1.382),ylim=(140,270),xlabel=J,ylabel='Core A E rate (Hz)',title='Equal coordinates, different orbits')
    ax.legend(handles=[Line2D([],[],color=CYAN,label='Stable orbit: minimum'),Line2D([],[],color=ORANGE,ls='--',label='Unstable orbit: mean')],frameon=False,fontsize=10,loc='upper left')
    for i in range(2):
        ax=axs[i+1]
        for key,col,ls in [('stable',CYAN,'-'),('unstable',ORANGE,'--')]:
            t,r,T=waveform(q[key]['source']);ax.plot(t,r[:,i],color=col,ls=ls,lw=1.8,label=f'{key.capitalize()}  T={T:.3f} ms')
        ax.set(xlim=(-65,65),ylim=(-8,415),xlabel='Time relative to surround peak (ms)',ylabel='Rate (Hz)',title='Core '+ 'AB'[i])
    axs[1].legend(frameon=False,fontsize=10,loc='lower right')
    fig.suptitle(f'Projection crossing at {J} = {q["g"]:.10f}',x=.06,ha='left',fontsize=18,weight='bold')
    save(fig,'08_projection_crossing','数值求根定位不稳定轨道A均值等于稳定轨道A谷值的交点，误差小于2e−9Hz。两条完整周期轨道的周期、A波形和稳定性都不同，B波形则较相似。','交叉的是不同轨道的不同统计量，不是轨道在状态空间连接；同一轨道始终满足最小值≤均值≤最大值。')

def transitions():
    fig,axs=plt.subplots(3,2,figsize=(12.6,9.1),sharex=True,sharey='row');fig.subplots_adjust(left=.09,right=.97,bottom=.1,top=.84,wspace=.18,hspace=.28)
    for k,name in enumerate(['recruitment_up','recruitment_down']):
        z=np.load(OUT/'transitions'/f'{name}_dt0.025.npz');t=np.arange(len(z['r']))*float(z['dt'])/1000;r=z['r']*1000
        for i in range(3):
            axs[i,k].plot(t,r[:,i],color=GROUP[i],lw=1)
            axs[i,k].set(xlim=(0,3),ylim=(-5,310 if i<2 else 215))
            if k==0:axs[i,k].set_ylabel(['Core A E (Hz)','Core B E (Hz)','Surround E (Hz)'][i])
        axs[0,k].set_title(('From low-surround fold' if k==0 else 'From recruited-surround fold')+f'\n{J} = {float(z["g"]):.5f}',loc='left')
        axs[-1,k].set_xlabel('Time after parameter step (s)')
    fig.suptitle('Observed switching from specified periodic histories',x=.09,ha='left',fontsize=18,weight='bold')
    save(fig,'09_recruitment_switching','分别从低周边和招募周边折点的完整延迟历史出发，向上或向下改变J并积分8秒，图示前3秒。上调到1.17632招募周边，下调到1.17623返回低周边周期活动。','这是明确初始历史下的去向验证，不把两个有限参数阶跃当作精确准静态切换阈值。')

def micro_and_modes():
    from validate_report import mu
    v=read(OUT/'numerical_validation.json');pd=v['PD0'];f={x['label']:x for x in v['fold_grid_checks']};ref=f['LP0b']['g']
    near=read(OUT/'arcs/between_fold_and_flip/progress.json');approach=read(OUT/'arcs/approach_first_fold/progress.json');between=read(OUT/'arcs/between_low_folds/progress.json')
    assert len(near)==25 and len(approach)==21 and len(between)==21
    raw=read(V4/'arcs/recruitment_turn/progress.json')
    seqs=[([orbit(x['source']) for x in approach],BLUE,'-'),
        ([orbit(x['source']) for x in between],RED,'--'),
        ([orbit(x['source']) for x in near if x['fraction']<=1],BLUE,'-'),
        ([orbit(x['source']) for x in near if x['fraction']>=1]+[orbit(raw[6]['source'])],RED,'--')]
    fig,axs=plt.subplots(2,2,figsize=(12.5,9.2));fig.subplots_adjust(left=.095,right=.97,bottom=.10,top=.85,wspace=.3,hspace=.42)
    for k,(scale,xlim) in enumerate([(1e8,(-.25,2.35)),(1e12,(-1,24))]):
        ax=axs[0,k]
        for seq,col,ls in (seqs if k==0 else seqs[2:]):
            ax.plot([(x['g']-ref)*scale for x in seq],[x['mean'][2] for x in seq],color=col,ls=ls,lw=2)
        for name in (['LP0a','LP0b'] if k==0 else ['LP0b','PD0']):
            a=pd if name=='PD0' else f[name];x=(a['g']-ref)*scale;y=a['mean_hz'][2]
            ax.scatter(x,y,color='black',s=45,marker='D' if name=='PD0' else 's',zorder=8)
            offset=(10,12) if name in ['LP0a','PD0'] else (14,-17)
            ax.annotate(name,(x,y),xytext=offset,textcoords='offset points',fontsize=11)
        ax.set(xlim=xlim,ylim=((.335,.427) if k==0 else (.39975,.4010)),xlabel=rf'$J-J_{{\mathrm{{LP0b}}}}$ ($10^{{-{int(np.log10(scale))}}}$)',ylabel='Surround E period mean (Hz)',title=['Two very close cycle folds','Stable interval ending at PD0'][k])
    ax=axs[1,0];points=[(f['LP0b']['mean_hz'][2],mu(V5/'folds/surround_second_fold_N2048.npz')['multipliers'][0][0])]
    points += [(a['mean_hz'][2],a['floquet']['multipliers'][0][0]) for a in v['micro_neighborhood']]
    points.append((pd['mean_hz'][2],pd['floquet']['multipliers'][0][0]));points.sort()
    ax.axhline(1,color='black',lw=1,ls=':');ax.axhline(-1,color='black',lw=1,ls=':')
    for x,y in points:ax.scatter(x,y,c='black' if abs(abs(y)-1)<.005 else BLUE if abs(y)<1 else RED,s=52,zorder=5)
    ax.set(xlabel='Surround E period mean (Hz)',ylabel='Leading transverse Floquet multiplier',title='Independent return-map eigenvalues',ylim=(-1.85,1.25));ax.ticklabel_format(axis='x',useOffset=False)
    ax=axs[1,1];children=sorted([a for a in v['PD0_children'] if a['amplitude']<=.0004],key=lambda a:a['amplitude'])
    ax.plot([0]+[(a['g']-pd['g'])*1e12 for a in children],[0]+[a['half_period_difference']*1e5 for a in children],color=PURPLE,lw=1.3)
    for a in children:ax.scatter((a['g']-pd['g'])*1e12,a['half_period_difference']*1e5,color=BLUE if a['floquet']['max_transverse']<1 else RED,s=48)
    ax.set(xlabel=r'$J-J_{\mathrm{PD0}}$ ($10^{-12}$)',ylabel=r'Half-period difference ($10^{-5}$)',title='Local branch with twice the period')
    fig.suptitle('The hidden fold–flip sequence inside region R',x=.095,ha='left',fontsize=18,weight='bold')
    fig.legend(handles=[Line2D([],[],color=BLUE,label='Stable T-period segment'),Line2D([],[],color=RED,ls='--',label='Unstable T-period segment'),Line2D([],[],marker='o',color=BLUE,ls='none',label='Stable 2T'),Line2D([],[],marker='o',color=RED,ls='none',label='Unstable 2T')],loc='upper left',bbox_to_anchor=(.085,.948),ncol=4,frameon=False,fontsize=10.5)
    save(fig,'10_hidden_fold_flip_sequence','极窄横轴展开LP0a→LP0b折返以及LP0b之后的稳定T支和PD0失稳。下排给出独立Floquet乘子穿越−1的数值点，以及实际2T子支的非零半周期差。','两个放大层级分别是1e−8和1e−12参数尺度；这些精确数值只适用于当前冻结模型，不能解读为生物参数精度。')
    rows=read(OUT/'critical_mode_components.json');fig,axs=plt.subplots(1,2,figsize=(12.2,7.2));fig.subplots_adjust(left=.10,right=.94,bottom=.15,top=.82,wspace=.30)
    for ax,key,title in zip(axs,['right_rate_fraction','left_rate_fraction'],['Right critical rate mode','Adjoint rate mode']):
        arr=np.array([x[key] for x in rows])*100;im=ax.imshow(arr,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_xticks(range(6));ax.set_xticklabels(['A E','B E','S E','A I','B I','S I']);ax.set_yticks(range(len(rows)));ax.set_yticklabels([x['name'] for x in rows]);ax.set_title(title,pad=12)
        for i in range(len(rows)):
            for j in range(6):ax.text(j,i,'<0.1' if arr[i,j]<.05 else f'{arr[i,j]:.1f}',ha='center',va='center',fontsize=10,color='white' if arr[i,j]>55 else 'black')
    fig.suptitle('Where the joint-network critical modes appear',x=.10,ha='left',fontsize=18,weight='bold')
    fig.text(.1,.07,'Squared norm of rate components (%)',fontsize=12)
    save(fig,'11_joint_critical_mode_components','比较左侧折点/PD0与右侧LP1/PD1–PD3的六群体临界右率模和伴随率模。相同临界模式在A、B的占比不同，解释两张读出图为何可表现得很不一样。','比例未按群体细胞数加权，不是因果贡献；周期折点还含周期变化分量，此处仅展示率分量。')

def main():
    c=critical();seq=collect(c);main_plots(seq,c);recruitment(c);coexist_left();right_projections(seq,c);coexist_right();projection_intersection(seq);transitions()
    if (OUT/'numerical_validation.json').exists():micro_and_modes()
    write('figure_manifest.json',MANIFEST);write('displayed_curve_sequences.json',seq)
    (FIG/'README.md').write_text('# Core A/B分岔投影与尖点核查\n\n'+'\n'.join(CAPTIONS)+'\n候选版本：已执行数值与图像自查，仍待用户目视检查。\n')
    print('FIGURES_COMPLETE',json.dumps(MANIFEST),flush=True)

if __name__=='__main__':main()

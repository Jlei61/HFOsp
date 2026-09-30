"""Critical-interval cases, dense equilibrium branches, and fold-family audit."""
from common import *
from expanded_figures import waveform,spatial,seeg,field_counts,choose_window,COL,style
from expanded_readouts import OLD,compare
from model import SpatialBrunel
from src.snn_contact_display import contact_indices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
BASE=OUT/'critical_revision';F=BASE/'figures'

def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=200)

def cases():
    sources=read(OUT/'expanded/readouts/result.json')['rows']+read(BASE/'readouts/result.json')['rows']
    result=[]
    settings=[('L',.6,'Low-activity reference'),('H1−',.948,'Before H1'),('H1+ / H2−',.955,'Between H1 and H2'),('H2+',.963,'After H2'),('R',1.3,'More regular bursting')]
    for label,J,title in settings:
        q=next(q for q in sources if q['J_EE_core']==J and '_high' not in q['tag'])
        rec=read(ROOT/q['source']);rec['data']=dict(np.load(ROOT/rec['trajectory']))
        rec['envelopes']=dict(np.load((ROOT/q['source']).parent/'envelopes.npz'));rec['smooth_rate']=gaussian_filter1d(rec['data']['regional_rates_hz'],5,axis=0)
        rec.update(case_label=label,case_title=title,summary=q);result.append(rec)
    return result

def branch(ax,k,records,inset=False):
    b=np.load(BASE/'branch.npz');J=b['J'];y=b['regional'][:,k];h=[read(OUT/f'g20/hopf_{c}_calibrated_full/result.json') for c in 'AB']
    sp=read(BASE/'stability/result.json');cert={q['index']:q['positive_mode_found'] for q in sp['rows']}
    ax.plot(J,y,ls=':',color='#999999',lw=1.)
    first=np.flatnonzero(J>=h[0]['J_EE_core'])[0]
    xx=np.r_[J[:first],h[0]['J_EE_core']];yy=np.r_[y[:first],h[0]['rates_hz'][k]]
    ax.plot(xx,yy,color=COL[k],lw=1.7)
    # Classified segments require positive eigenpairs at their sampled ends.
    points=sorted(i for i in cert if i>=first)
    runs=[]
    for i,j in zip(points[:-1],points[1:]):
        if cert[i] and cert[j]:
            if runs and runs[-1][1]==i:runs[-1][1]=j
            else:runs.append([i,j])
    for i,j in runs:ax.plot(J[i:j+1],y[i:j+1],color=COL[k],lw=1.4,ls='--')
    if points and cert[points[0]]:
        ax.plot(np.r_[h[0]['J_EE_core'],J[first:points[0]+1]],np.r_[h[0]['rates_hz'][k],y[first:points[0]+1]],color=COL[k],lw=1.4,ls='--')
    for n,q in enumerate(h):
        ax.plot(q['J_EE_core'],q['rates_hz'][k],'o',color=COL[n],ms=5,zorder=5)
        if inset:ax.annotate(f'H{n+1}',(q['J_EE_core'],q['rates_hz'][k]),xytext=(1,15+13*n),textcoords='offset points',fontsize=10,color=COL[n])
    folds=read(BASE/'fold_audit.json')['rows']
    if not inset:
        for q in folds:ax.plot(q['J_EE_core'],q['rates_hz'][k],'o',mfc='white',mec='#222222',ms=3,mew=.8,zorder=4)
        for q,text in [(max(folds,key=lambda x:x['J_EE_core']),'Fold 1.201'),(min(folds,key=lambda x:abs(x['J_EE_core']-1.0942)),'Fold 1.094')]:
            ax.annotate(text,(q['J_EE_core'],q['rates_hz'][k]),xytext=(18,-12),textcoords='offset points',fontsize=9,
                arrowprops=dict(arrowstyle='-',lw=.7))
        for rec in [records[0],records[-1]]:
            idx=int(np.argmin(abs(J-rec['J_EE_core'])));ax.plot(J[idx],y[idx],'s',color='#d17b00',ms=4)
            ax.annotate(rec['case_label'],(J[idx],y[idx]),xytext=(6,3),textcoords='offset points',fontsize=11,color='#a35400')
        ax.set(xlim=(.4,2.01),ylim=(0,550),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} equilibrium rate (Hz)',title=f'Core {"AB"[k]}')
        ax.set_yscale('symlog',linthresh=1,linscale=.6);ax.set_yticks([0,1,10,100,400]);ax.set_yticklabels(['0','1','10','100','400']);ax.set_xticks([.4,.8,1.2,1.6,2.]);style(ax)
    else:
        for n,c in enumerate('AB'):
            other=read(OUT/f'g40/hopf_{c}_calibrated_full/result.json')['J_EE_core']
            ax.axvspan(other,h[n]['J_EE_core'],color=COL[n],alpha=.10,lw=0)
        vals=[]
        for rec in records[1:4]:
            idx=int(np.argmin(abs(J-rec['J_EE_core'])));ax.plot(J[idx],y[idx],'s',color='#d17b00',ms=4,zorder=6);vals.append(y[idx])
        ax.set(xlim=(.946,.965),ylim=(min(vals)*.96,max(vals)*1.06),xticks=[.948,.955,.963],title='Hopf neighborhood')
        ax.tick_params(labelsize=8);ax.set_title('Hopf neighborhood',fontsize=10)

def main():
    F.mkdir(exist_ok=True);plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    records=cases();old=read(OUT/'expanded/readouts/result.json');names=old['contact_names'];order=contact_indices(names)
    geo=dict(np.load(OUT/'operators/g40/geometry.npz'));native=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    for key in ['contact_xy','core_radius_mm']:geo[key]=native[key]
    E=geo['population']==0;counts=np.bincount(geo['group_cell'][E],weights=geo['group_size'][E],minlength=1600)
    contracts={kind:read(OLD/f'observer_{kind}.json') for kind in ['firing','current_hfo']}
    selections={q['tag']:choose_window(q) for q in records}
    with PdfPages(F/'core_bifurcation_critical_cases.pdf') as pdf:
        for kind in ['firing','current_hfo']:
            fig=plt.figure(figsize=(25,12));g=fig.add_gridspec(5,4,width_ratios=[1.3,1.03,1.26,1.0],left=.05,right=.985,top=.905,bottom=.085,wspace=.32,hspace=.56)
            left=g[:,0].subgridspec(2,1,hspace=.28)
            for k in range(2):
                ax=fig.add_subplot(left[k]);branch(ax,k,records)
                ins=ax.inset_axes([.55,.10,.43,.40]);branch(ins,k,records,True)
            for i,q in enumerate(records):
                selection=selections[q['tag']];ax=fig.add_subplot(g[i,1]);waveform(ax,q,selection,i==4)
                ax.set_title(f'{q["case_label"]}   {q["case_title"]}\n'+fr'$J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:g}$',loc='left',fontsize=11,pad=5)
                cells=field_counts(q,counts);sub=g[i,2].subgridspec(1,3,wspace=.12)
                for j,offset in enumerate([-10,30,70]):
                    a=fig.add_subplot(sub[j]);im=spatial(a,cells,selection['anchor_ms'],offset,geo,j==0)
                    if i==4:a.set_xlabel('x (mm)',fontsize=9)
                ax=fig.add_subplot(g[i,3]);im2=seeg(ax,q,selection,kind,contracts[kind],order,i==4)
                ax.tick_params(axis='y',labelsize=8)
            for x,title in zip([.194,.453,.700,.902],['Spatial equilibrium branches (1 mm)','Native network activity','Same-window 2D activity','SEEG-site firing envelope' if kind=='firing' else 'SEEG current HFO envelope']):
                fig.text(x,.982,title,ha='center',fontsize=14)
            fig.legend(handles=[Line2D([0],[0],color=COL[k],label=n) for k,n in enumerate(['Core A','Core B','All E'])],loc='upper center',bbox_to_anchor=(.453,.967),ncol=3,frameon=False,fontsize=9)
            fig.legend(handles=[Line2D([0],[0],color='black',label='Stable equilibrium'),Line2D([0],[0],color='black',ls='--',label='Unstable equilibrium'),
                Line2D([0],[0],color='black',marker='o',mfc='white',ls='',ms=4,label='Stationary fold'),Line2D([0],[0],color='#d17b00',marker='s',ls='',ms=4,label='Selected J for native case')],
                loc='lower left',bbox_to_anchor=(.045,.004),ncol=2,frameon=False,fontsize=9)
            cb=fig.colorbar(im,cax=fig.add_axes([.635,.033,.14,.01]),orientation='horizontal');cb.set_label('Spikes / 2 ms / 1 mm cell',fontsize=9)
            cb=fig.colorbar(im2,cax=fig.add_axes([.864,.033,.11,.01]),orientation='horizontal');cb.set_label('Reference-normalized envelope',fontsize=9)
            save(fig,f'core_bifurcation_critical_cases_{kind}');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)

    # Both Hopf comparisons share the middle case. Full spectral signs, not
    # waveform appearance, define the three selected parameter intervals.
    sp=read(BASE/'stability/result.json')['rows'];fig,aa=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for core in range(2):
        data=[]
        for q in sp:
            if not (.943<=q['J_EE_core']<=.968 and q['index']<113):continue
            roots=[r for r in q['roots'] if np.argmax(r['regional_energy'])==core]
            if roots:
                r=max(roots,key=lambda x:x['regional_energy'][core]);data.append([q['J_EE_core'],r['lambda_per_ms'][0]*1000,r['frequency_hz']])
        a=np.array(data);aa[0].plot(a[:,0],a[:,1],color=COL[core],lw=2,label=f'Core {"AB"[core]} mode');aa[1].plot(a[:,0],a[:,2],color=COL[core],lw=2)
        h=read(OUT/f'g20/hopf_{"AB"[core]}_calibrated_full/result.json');aa[0].plot(h['J_EE_core'],0,'o',color=COL[core]);aa[0].annotate(f'H{core+1}',(h['J_EE_core'],0),xytext=(3,10+12*core),textcoords='offset points',color=COL[core])
    for a in aa:
        for q in records[1:4]:a.axvline(q['J_EE_core'],color='#999999',lw=.8,ls=':')
        a.set(xlim=(.946,.965),xticks=[.948,.955,.963],xlabel=r'$J_{\mathrm{EE,core}}$');style(a)
    aa[0].axhline(0,color='black',lw=.8);aa[0].set_ylabel(r'Growth Re $\lambda$ (s$^{-1}$)');aa[1].set_ylabel('Mode frequency (Hz)');aa[0].legend(frameon=False)
    save(fig,'two_hopf_eigenvalue_crossings');plt.close(fig)

    # The repeated folds are clearer in a joint A/B state projection.
    b=np.load(BASE/'branch.npz');audit=read(BASE/'fold_audit.json');folds=audit['rows'];palette=['#2474ab','#e07b23']
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    subset=(b['J']>.99)&(b['J']<1.067)&(b['regional'][:,0]<17)&(b['regional'][:,1]<13)
    data=b['regional'][subset];js=b['J'][subset]
    for k,a in enumerate(axes[0]):
        a.plot(js,data[:,k],color=COL[k],lw=1.2,ls='--');a.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} equilibrium rate (Hz)',xlim=(.997,1.065),ylim=(0,16))
        for i,q in enumerate(folds):
            if q['J_EE_core']>1.067:continue
            color=palette[q['mode_family']-1] if q['mode_family']<=2 else '#777777'
            a.plot(q['J_EE_core'],q['rates_hz'][k],'o',color=color,ms=5)
        style(a)
    a=axes[1,0];a.plot(data[:,0],data[:,1],color='#777777',lw=1)
    for q in folds:
        if q['J_EE_core']>1.067:continue
        color=palette[q['mode_family']-1] if q['mode_family']<=2 else '#777777';a.plot(q['rates_hz'][0],q['rates_hz'][1],'o',color=color,ms=6)
    fixed=read(BASE/'fixed_J/result.json')
    for q in fixed['rows']:a.plot(*q['rates_hz'][:2],'s',mfc='white',mec='black',ms=5)
    a.set(xlabel='Core A equilibrium rate (Hz)',ylabel='Core B equilibrium rate (Hz)',title=f'{fixed["distinct_roots"]} equilibria at the same J={fixed["J_EE_core"]:g}',xlim=(.9,5.6),ylim=(1,5.4));style(a)
    a=axes[1,1];groups=audit['mode_families'];permutation=[i for g in groups for i in g];mat=np.array(audit['mode_overlap'])[np.ix_(permutation,permutation)]
    im=a.imshow(mat,vmin=0,vmax=1,cmap='viridis');labels=[f'{folds[i]["J_EE_core"]:.5f}' for i in permutation]
    a.set(xticks=range(12),xticklabels=labels,yticks=range(12),yticklabels=labels,title='Null-vector overlap');a.tick_params(axis='x',rotation=70,labelsize=8);a.tick_params(axis='y',labelsize=8)
    fig.colorbar(im,ax=a,shrink=.75,label='Neuron-weighted |cosine|')
    fig.legend(handles=[Line2D([0],[0],marker='o',color=palette[0],ls='',label='Repeated Core B fold family 1'),Line2D([0],[0],marker='o',color=palette[1],ls='',label='Repeated Core B fold family 2')],loc='outside upper center',ncol=2,frameon=False)
    save(fig,'fold_branch_connections_and_families');plt.close(fig)

    # Legible per-case raster sheets preserve the exact main-figure window.
    for n,q in enumerate(records):
        selection=selections[q['tag']];fig=plt.figure(figsize=(13,8));g=fig.add_gridspec(3,4,height_ratios=[1,.8,1.6],hspace=.5,wspace=.5)
        ax=fig.add_subplot(g[0,:]);waveform(ax,q,selection,True);ax.set_title(f'{q["case_label"]}: {q["case_title"]}, J={q["J_EE_core"]:g}');ax.set_ylabel('Rate (Hz)')
        z=q['data'];ax=fig.add_subplot(g[1,:]);ax.scatter(z['raster_time_ms']/1000,z['raster_id'],s=2,color='black',lw=0,rasterized=True)
        ax.set(xlim=(.5,10),ylim=(0,750),yticks=[125,375,625],yticklabels=['Core A','Core B','Surround E'],xlabel='Time (s)');style(ax)
        cell=field_counts(q,counts)
        for j,offset in enumerate([-10,30,70]):
            ax=fig.add_subplot(g[2,j]);spatial(ax,cell,selection['anchor_ms'],offset,geo,True);ax.set_xlabel('x (mm)')
        ax=fig.add_subplot(g[2,3]);seeg(ax,q,selection,'firing',contracts['firing'],order,True)
        save(fig,f'case_{n+1}_J{q["J_EE_core"]:g}');plt.close(fig)
    baseline=records[1]['summary'];stats=[]
    for q in records:
        row=q['summary'];stats.append(dict(label=q['case_label'],role=q['case_title'],J_EE_core=q['J_EE_core'],dynamics=row['dynamics'],
            eligible_events={k:row[k]['N'] for k in ['firing','current_hfo']},
            change_from_before_H1={k:compare(row[k],baseline[k],names) for k in ['firing','current_hfo']},selection=selections[q['tag']],trajectory=str(ROOT/q['trajectory'])))
    write(BASE/'figure_metadata.json',dict(states=stats,selection_rule='Low-activity reference; before both grid estimates of H1; between both grids H1/H2; after both grid estimates of H2; one more-regular bursting reference.',
        shared_middle_case='J=.955 is simultaneously after H1 and before H2. It is not claimed to be a distinct native-SNN attractor.',
        branch_points=len(b['J']),spectral_points=read(BASE/'stability/result.json')['spectral_points'],native_runs='One paired private-input seed, 10 seconds per case; Z=1, original M; common OU fluctuation zero.',
        limits='Left: spatial population equilibria and linear response approximation. Right: original-neuron SNN at the same J; no nonlinear rate/SNN propagation equivalence or periodic/Floquet continuation asserted.',human_visual_acceptance=False))
    text='### core_bifurcation_critical_cases.pdf\n两页主图按临界区间选案例：低活动、H1前、两Hopf之间、H2后、较规则burst；中间案例由两组前后比较共用。左侧重新拼接完整固定点分支，细节窗聚焦Hopf，右侧三列来自同一原始SNN轨迹及时间窗。**关注点**：浅色带表示两种空间网格的临界值范围，不是统计置信区间；右侧不是固定点或降阶周期轨道。\n\n'
    text+='### two_hopf_eigenvalue_crossings.png\n跟踪A、B两个空间模态在虚轴两侧的增长率及频率，三个竖线对应主图的三个临界区间案例。**关注点**：H2发生在已被H1扰动模态破坏稳定性的固定点分支上，不是第二次从稳定静息态出发。\n\n'
    text+='### fold_branch_connections_and_families.png\n用两个率投影、A/B联合固定点平面及零模重合度显示密集折点的连接。两个重复Core B折点家族分别出现在不同Core A背景上，同一J可有多个不同固定点。**关注点**：沿弧长的折返不等于随J递增发生多次稳定burst切换；精细网格对应另见数值检查。\n\n'
    for n,q in enumerate(records):text+=f'### case_{n+1}_J{q["J_EE_core"]:g}.png\n主图“{q["case_title"]}”的独立大图，含相同轨迹的波形、样本raster、空间快照和触点包络。**关注点**：原生SNN在理论Hopf前后均可能出现有限噪声下的burst，不能仅凭肉眼变化标分岔。\n\n'
    (F/'README.md').write_text(text);print('CRITICAL FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()

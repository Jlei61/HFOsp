"""Scientific figures from saved states and actual refined critical points."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import csv

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':12,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLORS=['#e7e8ea','#e7b75d','#e98155','#be5d76','#4d9b86','#496984','#c7c0d8']
LABELS=['Rest','A burst, B quiet','Both burst','A burst, B high','B burst, A high','Both high, oscillating','Unresolved']


def category(row):
    if row['kind']=='equilibrium_candidate':return 0
    if row['kind']=='unresolved':return 6
    lo=np.array(row['min_hz']);hi=np.array(row['max_hz'])
    if hi[1]<5 and hi[0]>5:return 1
    if hi[0]<5 and hi[1]>5:return 6  # This unobserved class is not named in this figure.
    if max(lo[:2])<5:return 2
    if lo[0]<5:return 3
    if lo[1]<5:return 4
    return 5


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def boundaries():
    styles={'SN':('#282828','-',1.9),'LPC_onset':('#b48718','-',1.8),'LP1':('#a62635','-',1.8),
            'PD1':('#1454b8','--',1.5),'PD2':('#852a99','--',1.5),'PD3':('#168688','--',1.5)}
    rows=[dict(r,source=str((OUT/'equilibrium_fold.json').relative_to(ROOT))) for r in read(OUT/'equilibrium_fold.json')]
    out=[('SN',rows,styles['SN'],'DOMAIN_COVERED')]
    for label,folders in {'LPC_onset':['LPC_onset','LPC_onset_strict','LPC_onset_tight','LPC_onset_independent'],'LP1':['LP1','LP1_joint','LP1_endpoint'],'PD1':['PD1'],'PD2':['PD2_joint','PD2_joint_resume'],'PD3':['PD3']}.items():
        for folder in folders:
            choices=list((OUT/'boundaries'/folder).glob('curve_N*.json'))
            if not choices:continue
            p=max(choices,key=lambda f:int(f.stem.split('N')[-1]));d=read(p)
            out.append((label,d['points'],styles[label],d['status']))
    merged=[]
    for label in dict.fromkeys(q[0] for q in out):
        selected={};parts=[q for q in out if q[0]==label]
        for _,points,style,status in parts:
            for row in points:
                key=round(row['h'],8)
                if key not in selected or row.get('N',0)>=selected[key].get('N',0):selected[key]=row
        points=sorted(selected.values(),key=lambda r:r['h'])
        status='DOMAIN_COVERED' if points[0]['h']<1e-9 and points[-1]['h']>1-1e-9 else 'PARTIAL_UNRESOLVED_ENDPOINT'
        merged.append((label,points,parts[0][2],status))
    return merged


def draw_curves(ax,zoom=False):
    for label,rows,(color,ls,lw),status in boundaries():
        p=sorted(rows,key=lambda r:r['h'])
        ax.plot([r['g'] for r in p],[r['h'] for r in p],color=color,ls=ls,lw=lw,zorder=5)
        if status.startswith('PARTIAL'):
            row=min(p,key=lambda r:r['h']);ax.plot(row['g'],row['h'],'o',mfc='white',mec=color,ms=5,zorder=6)
    ax.set_xlabel(r'Core recurrent EE multiplier $g$')
    ax.set_ylabel(r'Core A threshold spread $h=\sigma_A/\sigma_{A,0}$')
    ax.set_ylim(0,1);ax.set_xlim(1.08,1.44);ax.set_xticks([1.1,1.2,1.3,1.4]);ax.set_yticks([0,.25,.5,.75,1.])


def map_figure(full=False):
    states=[]
    for f in sorted((OUT/'state_scan_v2').glob('*/*/g*.json')):
        row=read(f);row['state_index']=category(row);states.append(row)
    for f in (OUT/'long_transients').glob('*.json'):
        row=read(f);row['state_index']=category(row)
        states=[r for r in states if (r['h'],round(r['g'],6),r['direction'])!=(row['h'],round(row['g'],6),row['direction'])]
        states.append(row)
    hs=sorted(set(r['h'] for r in states));gs=sorted(set(round(r['g'],6) for r in states))
    def edges(a,limits):return np.r_[limits[0],(np.array(a[1:])+a[:-1])/2,limits[1]]
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.1),gridspec_kw={'width_ratios':[1,1,1]})
    draw_curves(axes[0]);axes[0].set_title('a   Continued bifurcations',loc='left',fontweight='bold')
    handles=[Line2D([],[],color=c,ls=ls,lw=lw,label=label.replace('LPC_onset','Cycle fold')) for label,(c,ls,lw) in {k:s for k,r,s,st in boundaries()}.items()]
    axes[0].legend(handles=handles,fontsize=9,loc='lower center',bbox_to_anchor=(.60,.02),frameon=False,ncol=1)
    for ax,direction,title in zip(axes[1:],['up','down'],['b   Increasing EE','c   Decreasing EE']):
        data=np.full((len(hs),len(gs)),np.nan)
        for row in states:
            if row['direction']==direction:data[hs.index(row['h']),gs.index(round(row['g'],6))]=row['state_index']
        ax.pcolormesh(edges(gs,(.5,1.61)),edges(hs,(0,1)),np.ma.masked_invalid(data),cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(-.5,7.5),7),shading='flat',rasterized=True)
        draw_curves(ax);ax.set_ylabel('');ax.set_title(title,loc='left',fontweight='bold')
    observed={r['state_index'] for r in states}
    fig.legend(handles=[Patch(color=c,label=t) for i,(c,t) in enumerate(zip(COLORS,LABELS)) if i in observed],loc='lower center',bbox_to_anchor=(.52,-.075),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(bottom=.18,wspace=.22)
    if full:
        for ax in axes:ax.set_xlim(.5,1.6);ax.set_xticks([.5,.75,1.,1.25,1.5])
        axes[0].get_legend().set_bbox_to_anchor((.30,.02))
    fig.text(.07,.025,'Colors: finite-time attraction candidates. Lines: solved critical conditions. Open endpoint: continuation unresolved.',fontsize=10)
    save(fig,'heterogeneity_two_parameter_full' if full else 'heterogeneity_two_parameter')
    write('state_map_summary.json',states)
    with (OUT/'state_map_summary.csv').open('w') as f:
        keys=['h','g','direction','kind','state_index','period_ms','recurrence_error_hz','duration_ms','source'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(states)
    curves=[]
    for label,rows,style,status in boundaries():
        for row in rows:curves.append(dict(row,label=label,curve_status=status))
    write('bifurcation_curves.json',curves)
    with (OUT/'bifurcation_curves.csv').open('w') as f:
        keys=['label','h','g','sigma_A_mV','T_ms','N','orbit_residual','critical_residual','null_residual','critical_core','curve_status','source'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(curves)


def onset():
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.7))
    for ax in axes:
        for label,rows,(c,ls,lw),status in boundaries():
            if label not in ('SN','LPC_onset'):continue
            pp=sorted(rows,key=lambda r:r['h']);ax.plot([r['g'] for r in pp],[r['h'] for r in pp],c=c,ls=ls,lw=lw,label=label.replace('SN','Low-state fold').replace('LPC_onset','Cycle fold'))
            if status.startswith('PARTIAL'):ax.plot(pp[0]['g'],pp[0]['h'],'o',mfc='white',mec=c,ms=6)
        ax.set_xlabel(r'Core recurrent EE multiplier $g$');ax.set_ylabel(r'Threshold spread $h$')
    axes[0].set(xlim=(1.11,1.185),ylim=(0,1));axes[0].set_title('a   First low-state instability',loc='left',fontweight='bold')
    axes[0].text(1.176,.46,'B leads',rotation=90,va='center',fontsize=10)
    axes[0].legend(frameon=False,loc='lower left',fontsize=10)
    axes[1].set(xlim=(1.12,1.177),ylim=(.96,1.001));axes[1].set_title('b   Near the reference distribution',loc='left',fontweight='bold')
    axes[1].text(1.127,.967,'Cycle existence and low-state stability\nrequire separate boundaries.',fontsize=10)
    fig.tight_layout();save(fig,'heterogeneity_onset_detail')


def response():
    from validate import response as calculate
    if not (OUT/'population_response.json').exists():calculate()
    rows=read(OUT/'population_response.json');fig,axes=plt.subplots(1,3,figsize=(12.5,3.9));colors=plt.cm.viridis(np.linspace(.1,.85,len(rows)))
    for row,c in zip(rows,colors):
        h=row['h'];s=System(h);values=s.actual_thresholds_A
        if h==0:axes[0].axvline(s.mean_A,c=c,lw=1.7,label=f'h = {h:g}')
        else:
            counts,edges=np.histogram(values,bins=np.linspace(14,18.05,52),density=True)
            axes[0].stairs(counts,edges,color=c,lw=1.4,label=f'h = {h:g}')
        axes[1].plot(row['mu_mV'],row['rate_hz'],c=c,lw=1.5)
        axes[2].plot(row['mu_mV'],row['rate_hz'],c=c,lw=1.5)
    axes[0].set(xlabel='Core A threshold (mV)',ylabel='Density',xlim=(14,18.1),ylim=(0,5));axes[0].legend(frameon=False,fontsize=9)
    axes[1].set(xlabel='Mean input (mV)',ylabel='Core A response (Hz / cell)',xlim=(0,30),ylim=(0,105))
    axes[2].set(xlabel='Mean input (mV)',ylabel='Core A response (Hz / cell)',xlim=(8,20),yscale='log',ylim=(1e-5,10))
    for ax,t in zip(axes,['a   Mean held fixed','b   Population transfer','c   Low-rate response']):ax.set_title(t,loc='left',fontweight='bold')
    fig.tight_layout();save(fig,'heterogeneity_transfer')


def coexistence():
    fig,axes=plt.subplots(2,1,figsize=(9,5.4),sharex=True,sharey=True)
    for ax,direction,title in zip(axes,['up','down'],['a   Burst state after increasing EE','b   High-background state after decreasing EE']):
        p=OUT/'state_scan_v2/h0.00000'/direction/'g1p38000.npz';z=np.load(p);r=z['r'][-3000:]*1000;t=np.arange(len(r))*.0005
        for i,c,l in zip([0,1,3,4],['#cc3c42','#ec9a37','#286cac','#6ca69e'],['A E','B E','A I','B I']):ax.plot(t,r[:,i],c=c,lw=1.05,label=l)
        ax.set_ylabel('Rate (Hz / cell)');ax.set_title(title,loc='left',fontweight='bold');ax.set_ylim(-10,700);ax.set_yticks([0,200,400,600])
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.55,-.035),fontsize=10)
    axes[1].set_xlabel('Time (s)');fig.suptitle('Same parameters: h = 0, g = 1.38',fontsize=12)
    fig.tight_layout();save(fig,'heterogeneity_coexisting_states')


if __name__=='__main__':
    map_figure();map_figure(full=True);onset();response();coexistence()

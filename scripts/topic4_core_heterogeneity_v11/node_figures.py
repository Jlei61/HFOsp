"""Parameter planes centered on actual critical nodes, plus matched branches."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hashlib

DEST=OUT/'threshold_nodes';FIG=DEST/'figures';FIG.mkdir(exist_ok=True)
S=System();STD=S.original_std_A;MEAN=S.mean_A
C={'SN':'#292929','LPC_onset':'#bd8220','LP1':'#bc4447',
   'PD1':'#2868ae','PD2':'#8d4da0','PD3':'#218d7d'}
NAMES={'SN':'SN','LPC_onset':'Cycle fold','LP1':'LP1','PD1':'PD1','PD2':'PD2','PD3':'PD3'}
V7=ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916'
CP=read(V7/'critical_points.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
 'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
J=r'$J_{\mathrm{EE,core}}$'

def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def critical(label):
    return next(r for r in CP if r['label']==label)

def curves(ax,data,keys,ykey='sigma_A_mV',ref=None):
    for key in keys:
        rows=sorted(data[key],key=lambda r:r[ykey]);x=[r['g'] for r in rows];y=[r[ykey] for r in rows]
        ax.plot(x,y,c=C[key],lw=1.25)
        ax.plot(x,y,'.',c=C[key],ms=2.5)
        if key in ['LPC_onset','LP1'] and ykey=='sigma_A_mV':
            ax.plot(x[0],y[0],'o',mfc='white',mec=C[key],ms=4.5,zorder=6)
        ax.plot(x[-1],y[-1],'^' if key.startswith('PD') else 's',
                mfc='white',mec=C[key],ms=5,clip_on=False,zorder=7)

def note(ax,txt,xy,pos,color='black',size=10):
    ax.annotate(txt,xy,xytext=pos,ha='center',va='center',fontsize=size,color=color,
      arrowprops=dict(arrowstyle='-',lw=.65,color=color),
      bbox=dict(fc='white',ec='none',pad=1,alpha=.96),zorder=12)

def parameter():
    rows=read(OUT/'bifurcation_curves.json');hcur={k:[r for r in rows if r['label']==k] for k in C}
    mean={'SN':read(DEST/'mean_fold.json')}
    for k in ['LP1','PD1','PD2','PD3']:
        p=DEST/f'{k}.json'
        if p.exists():mean[k]=read(p)['points']
    fig,aa=plt.subplots(2,2,figsize=(11.8,8.8));fig.subplots_adjust(left=.09,right=.96,bottom=.19,top=.92,hspace=.42,wspace=.30)
    ax=aa[0,0];curves(ax,hcur,['SN','LPC_onset']);ax.set(xlim=(1.115,1.184),ylim=(.702,.744),ylabel=r'Threshold spread $\sigma_A$ (mV)')
    ax.set_title('a   Heterogeneity: burst onset',loc='left',fontweight='bold')
    ax.set_xticks([1.12,1.14,1.16,1.18]);ax.set_yticks([.71,.72,.73,.74])
    q=read(DEST/'AB_onset_switch.json');ax.plot(q['g'],q['sigma_A_mV'],'D',ms=5,c='#594475')
    note(ax,'A / B onset switch',(q['g'],q['sigma_A_mV']),(1.153,.707),'#594475',9.5)
    note(ax,'SN: A leads',(1.15,.726),(1.160,.734),C['SN'])
    note(ax,'Cycle fold',(1.12168,.730),(1.132,.727),C['LPC_onset'])
    ax.text(.03,.94,rf'Mean fixed: {MEAN:.2f} mV',transform=ax.transAxes,fontsize=9)
    ax=aa[0,1];curves(ax,hcur,['LP1','PD1','PD2','PD3']);ax.set(xlim=(1.315,1.401),ylim=(.702,.744),ylabel=r'Threshold spread $\sigma_A$ (mV)')
    ax.set_title('b   Heterogeneity: cycle branches',loc='left',fontweight='bold');ax.set_yticks([.71,.72,.73,.74])
    for k,x,y in [('LP1',1.335,.725),('PD1',1.3505,.710),('PD3',1.3748,.715),('PD2',1.391,.727)]:
        ax.text(x,y,k,c=C[k],rotation=0 if k=='LP1' else 90,fontsize=10)
    ax=aa[1,0];curves(ax,mean,['SN'],ykey='mean_A_mV')
    ax.set(xlim=(.4,1.18),ylim=(16.46,17.31),ylabel=r'Mean threshold $\bar\theta_A$ (mV)')
    ax.set_title('c   Mean threshold: burst onset',loc='left',fontweight='bold')
    ax.set_xticks([.4,.6,.8,1.,1.2]);ax.text(.04,.92,rf'Spread fixed: {STD:.2f} mV',transform=ax.transAxes,fontsize=9)
    ax.text(.87,16.96,'SN',fontsize=11)
    ax.text(.53,17.05,'Rest stable',fontsize=10,color='#555555')
    note(ax,'Reference',(mean['SN'][0]['g'],MEAN),(1.01,17.12),size=9)
    ax=aa[1,1];keys=[k for k in ['LP1','PD1','PD2','PD3'] if k in mean and len(mean[k])>1]
    curves(ax,mean,keys,ykey='mean_A_mV');ax.set(xlim=(1.332,1.40),ylim=(16.61,17.31),ylabel=r'Mean threshold $\bar\theta_A$ (mV)')
    ax.set_title('d   Mean threshold: cycle branches',loc='left',fontweight='bold')
    if 'LP1' in mean and len(mean['LP1'])==1:
        rr=mean['LP1'][0];ax.plot(rr['g'],rr['mean_A_mV'],'s',mfc='white',mec=C['LP1'],ms=5,zorder=7)
        note(ax,'LP1*',(rr['g'],rr['mean_A_mV']),(1.361,17.19),C['LP1'],9)
    for k in keys:
        rr=sorted(mean[k],key=lambda r:r['mean_A_mV']);r=rr[len(rr)//2]
        dx={'LP1':-.002,'PD1':-.003,'PD2':.002,'PD3':.003}[k]
        ax.text(r['g']+dx,r['mean_A_mV'],k,c=C[k],rotation=90,fontsize=10)
        raw=read(DEST/f'{k}.json')
        if raw['status']!='DONE' or k=='LP1':ax.plot(rr[0]['g'],rr[0]['mean_A_mV'],'o',mfc='white',mec=C[k],ms=5,zorder=8)
    for ax in aa.flat:ax.set_xlabel(J);ax.tick_params(direction='out',length=3)
    handles=[Line2D([],[],c=C[k],lw=1.3,label=NAMES[k]) for k in C]
    fig.legend(handles=handles,ncol=6,frameon=False,loc='lower center',bbox_to_anchor=(.54,.09),fontsize=10)
    fig.text(.09,.055,'Top: zoom of the sensitive spread range; full range is shown separately. Bottom: newly solved mean-threshold scan.',fontsize=9)
    fig.text(.09,.027,'Squares / triangles: reference nodes. Open circles: unfinished endpoints. * Mean-threshold LP1: reference point only.',fontsize=9)
    save(fig,'critical_parameter_planes')
    # Keep the full spread range available, with thin, correctly scaled curves.
    fig,ax=plt.subplots(figsize=(6.7,4.7));curves(ax,hcur,list(C))
    ax.set(xlim=(1.105,1.41),ylim=(0,.75),xlabel=J,ylabel=r'Threshold spread $\sigma_A$ (mV)')
    ax.set_title('Full spread range',loc='left');ax.legend(handles=handles,ncol=2,frameon=False,loc='upper left',fontsize=9)
    ax.text(1.20,.32,'B sets the first low-rate fold\nwhen A spread is small.',fontsize=10)
    fig.tight_layout();save(fig,'full_spread_range')

def branch(ax):
    eq=read(V2/'equilibrium_spectrum.json');f=read(V2/'fold.json')
    for dr,ls in [(-1,'-'),(1,'--')]:
        rr=[r for r in eq if r['direction']==dr]
        ax.plot([f['g']]+[r['g'] for r in rr],[f['r_hz'][0]]+[r['r_hz'][0] for r in rr],c='#296c9d',ls=ls,lw=1.3)
    for rr in read(V7/'displayed_curve_sequences.json'):
        assert len(set(r['stable'] for r in rr))==1
        ax.plot([r['g'] for r in rr],[r['mean'][0] for r in rr],c='#bf8324',
                ls='-' if rr[0]['stable'] else '--',lw=1.25)

def branches():
    fig,aa=plt.subplots(1,2,figsize=(11.8,4.9));fig.subplots_adjust(left=.085,right=.96,bottom=.28,top=.86,wspace=.28)
    for ax in aa:branch(ax);ax.set_xlabel(J);ax.set_ylabel('Core A E mean rate (Hz / cell)')
    aa[0].set(xlim=(1.1208,1.127),ylim=(.20,13));aa[0].set_title('a   Reference slice: onset nodes',loc='left',fontweight='bold')
    aa[1].set(xlim=(1.33,1.40),ylim=(30,380));aa[1].set_title('b   Reference slice: cycle nodes',loc='left',fontweight='bold')
    mapping=[(aa[0],'Low-rate equilibrium fold','SN',(1.1258,3.3)),
             (aa[0],'Cycle fold','Cycle fold',(1.124,10.5)),
             (aa[1],'LP1','LP1',(1.353,110)),(aa[1],'PD1','PD1',(1.338,80)),
             (aa[1],'PD2','PD2',(1.390,130)),(aa[1],'PD3','PD3',(1.386,340))]
    for ax,key,label,pos in mapping:
        r=critical(key);xy=(r['JEE_core'],r['A_mean_hz']);cc=C[{'Cycle fold':'LPC_onset'}.get(label,label)]
        ax.plot(*xy,marker='^' if label.startswith('PD') else 's',mfc='white',mec=cc,ms=6,zorder=8)
        note(ax,label,xy,pos,cc)
    # The extremely short LP0/PD0 family belongs to its own unresolved 2-D locus.
    handles=[Line2D([],[],c='#296c9d',label='Equilibrium'),Line2D([],[],c='#bf8324',label='Periodic mean'),
             Line2D([],[],c='#333333',ls='-',label='Stable'),Line2D([],[],c='#333333',ls='--',label='Unstable')]
    fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.52,.12),frameon=False)
    fig.text(.085,.065,rf'Same reference model: mean threshold {MEAN:.2f} mV, spread {STD:.2f} mV. Markers match the parameter-plane curves.',fontsize=9)
    fig.text(.085,.025,'PD1, PD2 and PD3 belong to different periodic-branch boundaries; they are not three successive doublings of one orbit.',fontsize=9)
    save(fig,'matched_reference_branches')

if __name__=='__main__':parameter();branches()

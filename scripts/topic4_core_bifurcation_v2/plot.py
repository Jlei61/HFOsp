"""Publication-size diagnostic figures from continued equilibria and orbits."""
from model import System,OUT
from spectral import leading
from periodic import Orbit
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import resample

BLUE='#2366a2';RED='#bb4437';GREEN='#22856d';GRAY='#727a82';GOLD='#bc7d27'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':15,'axes.titlesize':16,
 'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':1.1,'lines.linewidth':2.5,'pdf.fonttype':42,'savefig.facecolor':'white'})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
FOOT='Six-population delay-rate closure • current SNN working-point and dynamic-response correspondence not validated'

def read(name):return json.loads((OUT/name).read_text())
def save(fig,name,book):
    for i,ax in enumerate(fig.axes):
        pos=ax.get_position();fig.text(pos.x0-.06,pos.y1+.025,chr(65+i),fontweight='bold',fontsize=18,va='bottom')
    fig.savefig(FIG/f'{name}.png',dpi=240,bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');book.savefig(fig,bbox_inches='tight');plt.close(fig)
def footer(fig,text=FOOT):fig.text(.5,.015,text,ha='center',va='bottom',fontsize=10.5,color='#515b64')
def orbits():
    files={}
    for p in (OUT/'periodic').glob('g*_N*.npz'):
        z=np.load(p);g=float(z['g']);N=len(z['r'])
        if float(z['residual'])>1e-8:continue
        if g not in files or N>files[g][0]:files[g]=(N,p)
    return [(g,np.load(p)) for g,(N,p) in sorted(files.items())]

def main():
    s=System();f=read('fold.json');gc=f['g'];rf=f['r_hz'][0];fine=read('fine_fold_branch.json');arc=read('equilibrium_spectrum.json');cyc=orbits()
    cg=np.array([g for g,z in cyc]);period=np.array([float(z['T']) for g,z in cyc]);maxr=np.array([z['r'][:,0].max()*1000 for g,z in cyc]);minr=np.array([z['r'][:,0].min()*1000 for g,z in cyc])
    # Require independent periodic stability before drawing a stable-cycle branch.
    fl=[]
    for g,z in cyc:
        q=read(f'floquet/g{g:.8f}_dt0.1.json');mm=np.array([complex(*v) for v in q['multipliers']]);neutral=np.argmin(abs(mm-1));trans=np.delete(mm,neutral)
        assert abs(mm[neutral]-1)<.005 and max(abs(trans))<1,(g,mm)
        fl.append((g,abs(mm[neutral]-1),max(abs(trans))))
    Ctheory=np.pi*f['dynamic_normalization_ms']/np.sqrt(f['transversality']*f['quadratic']/2)
    X=1/np.sqrt(cg-gc);Cfit,offset=np.polyfit(X,period,1);pred=Cfit*X+offset;r2=1-np.sum((period-pred)**2)/np.sum((period-period.mean())**2)
    theoryoffset=np.mean(period-Ctheory*X)
    fit=dict(C_theory_ms=float(Ctheory),C_fitted_ms=float(Cfit),offset_ms=float(offset),r2=float(r2),max_abs_error_ms=float(abs(period-pred).max()),relative_prefactor_difference=float(abs(Cfit/Ctheory-1)),g_range=[float(cg.min()),float(cg.max())],n_orbits=len(cyc))
    (OUT/'period_scaling.json').write_text(json.dumps(fit,indent=2)+'\n')
    with PdfPages(FIG/'core_burst_classical_bifurcation_booklet.pdf') as book:
        fig=plt.figure(figsize=(12,9));gs=fig.add_gridspec(2,2,height_ratios=[1.45,1],left=.09,right=.97,bottom=.12,top=.92,hspace=.42,wspace=.32)
        ax=fig.add_subplot(gs[0,:]);low=[x for x in arc if x['direction']==-1];low=sorted(low,key=lambda x:x['g']);up=[x for x in arc if x['direction']==1]
        ax.plot([x['g'] for x in low]+[gc],[x['r_hz'][0] for x in low]+[rf],color=BLUE,label='Stable low-rate equilibrium')
        ax.plot([gc]+[x['g'] for x in up],[rf]+[x['r_hz'][0] for x in up],color=RED,ls='--',label='Unstable equilibrium')
        ax.fill_between(cg,minr,maxr,color=GREEN,alpha=.075)
        ax.plot(cg,maxr,'o-',color=GREEN,ms=6,label='Stable periodic burst: maximum / minimum')
        ax.plot(cg,minr,'o-',color=GREEN,ms=6)
        ax.scatter([gc],[rf],s=90,c='black',zorder=6);ax.axvline(gc,color='black',ls=':',lw=1.2)
        ax.annotate('Saddle-node\n$g_c = %.8f$'%gc,(gc,rf),xytext=(.88,.57),textcoords='data',arrowprops=dict(arrowstyle='-',lw=1.1),fontsize=13)
        ax.set(yscale='symlog',ylim=(-.03,330),xlim=(.5,1.155),xlabel='Within-core E→E weight multiplier, $g$',ylabel='Core A E rate (Hz / neuron)')
        ax.set_yscale('symlog',linthresh=1);ax.set_yticks([0,.2,.5,1,3,10,30,100,300]);ax.set_yticklabels(['0','.2','.5','1','3','10','30','100','300'])
        ax.legend(loc='upper left',frameon=False,fontsize=11.5);ax.set_title('Equilibrium and periodic-orbit continuation',loc='left',pad=12)
        ax.text(.57,35,'Elevated equilibrium remains unstable',color=RED,fontsize=12)
        ax.annotate('Stable burst\nenvelope',xy=(1.14,240),xytext=(1.025,120),color=GREEN,ha='left',fontsize=12,arrowprops=dict(arrowstyle='-',color=GREEN,lw=1.2))
        a=fig.add_subplot(gs[1,0]);b=fig.add_subplot(gs[1,1])
        for direction,col,ls in [(-1,BLUE,'-'),(1,RED,'--')]:
            rows=sorted([x for x in fine if x['direction']==direction],key=lambda x:x['g']);xx=np.array([x['g'] for x in rows]);yy=np.array([x['r_hz'][0] for x in rows]);la=np.array([x['lambda_per_s'][0] for x in rows])
            a.plot(xx,yy,color=col,ls=ls);b.plot(xx,la,color=col,ls=ls)
        a.scatter([gc],[rf],s=65,c='black');a.set(xlim=(gc-.006,gc+.0005),ylim=(.39,.53),xlabel='$g$',ylabel='Core A E rate (Hz / neuron)',title='Fold: stable and saddle branches meet')
        b.axhline(0,color='black',lw=1);b.axvline(gc,color='black',ls=':',lw=1);b.scatter([gc],[0],s=65,c='black')
        b.set(xlim=(gc-.006,gc+.0005),ylim=(-9,9),xlabel='$g$',ylabel=r'Critical $\mathrm{Re}\,\lambda$ (s$^{-1}$)',title='A real eigenvalue reaches zero')
        b.text(.05,.55,r'$\mathrm{Im}\,\lambda = 0$ on both local branches',transform=b.transAxes,va='bottom',fontsize=12)
        for aa in [a,b]:aa.ticklabel_format(axis='x',style='plain',useOffset=False);aa.locator_params(axis='x',nbins=4)
        footer(fig);save(fig,'01_classical_bifurcation',book)

        fig=plt.figure(figsize=(12,9));gs=fig.add_gridspec(2,2,left=.09,right=.97,bottom=.16,top=.92,hspace=.55,wspace=.35)
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1]);ev=np.array([complex(*z) for z in read('eigen_validation.json')['fold_spectra']['64']])
        keep=(ev.real>-150)&(abs(ev.imag)<200);a.scatter(ev[keep].real,ev[keep].imag,s=90,c=BLUE,edgecolors='white',zorder=4);a.scatter([0],[0],s=160,c='black',marker='*',zorder=5)
        a.axvline(0,color=GRAY,ls=':',lw=1);a.axhline(0,color=GRAY,lw=.8);a.set(xlim=(-115,12),ylim=(-35,35),xlabel=r'$\mathrm{Re}\,\lambda$ (s$^{-1}$)',ylabel=r'$\mathrm{Im}\,\lambda$ (s$^{-1}$)',title='Spectrum at the low-rate fold')
        a.annotate('Critical mode: 0',xy=(0,0),xytext=(-8,17),ha='right',arrowprops=dict(arrowstyle='-',color='black'),fontsize=12)
        a.annotate('Other core: −22.36',xy=(-22.356,0),xytext=(-53,-22),ha='center',arrowprops=dict(arrowstyle='-'),fontsize=12)
        a.text(-55.556,7,'Synaptic poles\n−55.56 (×3)',ha='center',fontsize=11)
        checks=read('eigen_validation.json')['selected'][-2:]
        for row,col in zip(checks,[BLUE,RED]):
            lam=row['spectra']['64'][0];b.scatter([lam[0]],[lam[1]],s=100,c=col,label='Stable branch' if row['direction']<0 else 'Saddle branch')
        b.scatter([0],[0],s=160,c='black',marker='*');b.axvline(0,color=GRAY,ls=':',lw=1);b.axhline(0,color=GRAY,lw=.8)
        b.set(xlim=(-.14,.14),ylim=(-.07,.07),xlabel=r'$\mathrm{Re}\,\lambda$ (s$^{-1}$)',ylabel=r'$\mathrm{Im}\,\lambda$ (s$^{-1}$)',title=r'Spectral zoom at $g_c-10^{-6}$');b.legend(loc='upper center',frameon=False,fontsize=11)
        labels=['A E','B E','Surround E','A I','B I','Surround I'];v=np.array(f['v']);w=np.array(f['w']);x=np.arange(6)
        c.bar(x,v,color=[BLUE if q>=0 else RED for q in v],width=.65);d.bar(x,w,color=[BLUE if q>=0 else RED for q in w],width=.65)
        for aa,vals in [(c,v),(d,w)]:
            aa.axhline(0,color='black',lw=.8);aa.set_xticks(x,labels,rotation=30,ha='right');aa.set_ylim(-1.2,1.35)
            for j,val in enumerate(vals):aa.text(j,val+(.08 if val>=0 else -.08),f'{val:.3f}',ha='center',va='bottom' if val>=0 else 'top',fontsize=11)
        c.set(title='Right eigenvector: activity displacement',ylabel=r'$v$  ($\|v\|_2=1$)');d.set(title='Left characteristic vector: sensitivity',ylabel=r'$w$  ($w^\mathsf{T}v=1$)')
        footer(fig,'Six population coordinates; the projected mode does not resolve individual-neuron eigenvectors.');save(fig,'02_eigenvalues_and_eigenvectors',book)

        fig,(a,b)=plt.subplots(1,2,figsize=(12,5.5));fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.88,wspace=.32)
        a.plot(cg,period/1000,'o-',color=GREEN,ms=7);a.axvline(gc,color='black',ls=':',lw=1)
        a.set(xlabel='$g$',ylabel='Burst period (s)',title='Finite-amplitude bursts slow near the fold',xlim=(gc-.0006,cg.max()+.001));a.ticklabel_format(axis='x',useOffset=False)
        xx=np.linspace(0,X.max()*1.03,200);b.plot(xx,(Ctheory*xx+theoryoffset)/1000,color='black',ls='--',label='Slope predicted by fold normal form');b.scatter(X,period/1000,s=65,c=GREEN,zorder=3,label='Solved periodic orbits')
        b.set(xlabel=r'$(g-g_c)^{-1/2}$',ylabel='Burst period (s)',title='Inverse-square-root bottleneck scaling',xlim=(0,xx.max()),ylim=(0,period.max()/1000*1.08))
        b.text(.05,.94,f'Predicted slope: {Ctheory:.3f} ms\nFitted slope: {Cfit:.3f} ms\n$R^2$ = {r2:.8f}',transform=b.transAxes,va='top',fontsize=12)
        b.legend(loc='lower right',frameon=False,fontsize=10.5);footer(fig,'Local saddle-node confirmed; finite-period continuation and the global return support a SNIC-type onset.');save(fig,'03_period_divergence',book)

        fig,(a,b)=plt.subplots(1,2,figsize=(12,5.7));fig.subplots_adjust(left=.09,right=.97,bottom=.2,top=.88,wspace=.32)
        z=np.load(OUT/'fold_global_return.npz');t=np.arange(len(z['r']))*float(z['dt'])/1000;rate=z['r'][:,0]*1000
        a.plot(t,rate,color=BLUE);a.axhline(rf,color='black',ls=':',lw=1);a.set_yscale('symlog',linthresh=1);a.set(xlabel='Time (s)',ylabel='Core A E rate (Hz / neuron)',title=r'One excursion at $g=g_c$, then return',xlim=(0,16),ylim=(0,300))
        a.text(.97,.92,'No periodic forcing\nInitial displacement: +0.02 Hz',transform=a.transAxes,ha='right',va='top',fontsize=11)
        keep=t>3;b.plot(t[keep],(rf-rate[keep]),color=BLUE);b.set_yscale('log');b.set(xlabel='Time after initialization (s)',ylabel='Distance below fold rate (Hz)',title='The trajectory approaches the same fold',xlim=(3,16));b.text(.95,.95,f'At 16 s: {rf-rate[-1]:.6f} Hz below $r_c$',ha='right',va='top',transform=b.transAxes,fontsize=11)
        footer(fig,'The finite-time return is a numerical global-connection diagnostic, not a rigorous infinite-time proof.');save(fig,'04_global_return',book)

        fig,(a,b)=plt.subplots(1,2,figsize=(12,5.5));fig.subplots_adjust(left=.09,right=.97,bottom=.2,top=.88,wspace=.35)
        ff=np.array(fl);resolved=ff[:,2]>1e-10
        a.semilogy(ff[resolved,0],ff[resolved,2],'o-',color=GREEN,ms=7)
        a.scatter(ff[~resolved,0],np.full((~resolved).sum(),1e-10),marker='v',s=65,c=GREEN)
        a.text(.06,.08,r'$\triangledown$: below $10^{-10}$',transform=a.transAxes,fontsize=11)
        a.axhline(1,color='black',ls='--',lw=1);a.set(xlabel='$g$',ylabel='Largest transverse Floquet modulus',title='Periodic orbits are attracting',ylim=(3e-11,2));a.ticklabel_format(axis='x',useOffset=False)
        checks=[read(f'floquet/g1.15000000_dt{dt}.json') for dt in ['0.1','0.05','0.025']];dd=np.array([q['dt_ms'] for q in checks]);ee=[]
        for q in checks:
            mm=np.array([complex(*v) for v in q['multipliers']]);ee.append(min(abs(mm-1)))
        b.loglog(dd,ee,'o-',color=BLUE,ms=7,label='Neutral multiplier error');b.loglog(dd,ee[-1]*(dd/dd[-1])**2,'--',color=GRAY,label='Second-order reference')
        b.set(xlabel='Variational integration step (ms)',ylabel=r'$|\mu_{\mathrm{phase}}-1|$',title='Autonomous phase mode converges to 1');b.legend(frameon=False,fontsize=11)
        footer(fig,'Full delay-history monodromy; five leading multipliers; neutral phase direction excluded from transverse stability.');save(fig,'05_floquet_stability',book)

        targets=[cyc[0],min(cyc,key=lambda x:abs(x[0]-1.13)),cyc[-1]]
        fig,axes=plt.subplots(3,1,figsize=(12,9));fig.subplots_adjust(left=.1,right=.97,bottom=.12,top=.93,hspace=.44)
        for ax,(g,z) in zip(axes,targets):
            rr=z['r'];T=float(z['T']);rr=np.roll(rr,len(rr)//4-np.argmax(rr[:,0]),axis=0)
            tt=np.arange(len(rr)*2)*T/len(rr)/1000;repeated=np.tile(rr,(2,1))
            ax.plot(tt,repeated[:,0]*1000,color=BLUE,label='Core A E');ax.plot(tt,repeated[:,3]*1000,color=RED,alpha=.8,label='Core A I');ax.set(xlim=(0,2*T/1000),ylim=(0,430),xlabel='Time (s)',ylabel='Rate (Hz / neuron)');ax.set_title(f'$g={g:g}$     period = {T/1000:.3f} s',loc='left')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.975,.98),frameon=False,ncol=2)
        footer(fig,'Waveforms reconstructed from solved periodic orbits; both firing rates refer to cells inside core A.');save(fig,'06_continued_burst_waveforms',book)
    print('FIGURES COMPLETE',json.dumps(fit),flush=True)

if __name__=='__main__':main()

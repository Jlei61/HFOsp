"""Main bifurcation figure (D = 1 - <Z_E> on the prescribed path; global E rate Hz/neuron) + spatial panels.

Left: equilibria (stable solid / unstable dashed / unclassified grey dots), periodic-orbit branch mean
(other colour, stable solid / unstable dashed), periodic extrema (filled = stable, open = unstable),
critical points with distinct markers (SN, LPC, H, PD, torus), insets for narrow structures.
Right: 2-D E-rate fields of the same rate model before the key transition, near the critical point and
just after entering broad activity, common window and colour scale. No native trajectory overlay, no
titles or grey annotations.
"""
from common_v3 import *
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse,glob
geo=dict(np.load(OPERATORS/'g20/geometry.npz'));centers=geo['centers_mm']
def branch_rows(label):
    p=DEST/'equilibria'/label/'result.json'
    return read(p)['rows'] if p.exists() else []
def stability_map(label):
    p=DEST/'equilibrium_stability'/f'{label}_sampled.json'
    if not p.exists():return {}
    return {r['index']:r for r in read(p)['rows']}
def segments(rows,stab):
    """split branch into runs of equal classification using the nearest sampled stability."""
    idx=np.array([r['index'] for r in rows]);D=np.array([r['D'] for r in rows]);g=np.array([r['global_E_hz'] for r in rows])
    sidx=np.array(sorted(stab));cls=[]
    for i in idx:
        if len(sidx)==0:cls.append('unclassified');continue
        j=sidx[np.argmin(abs(sidx-i))];q=stab[j]
        if abs(j-i)>60 or q['unstable_roots'] is None:cls.append('unclassified')
        else:cls.append('stable' if q['unstable_roots']==0 else 'unstable')
    return D,g,np.array(cls)
def main(a):
    fig=plt.figure(figsize=(15,7.2));gs=fig.add_gridspec(3,3,width_ratios=[2.4,1,1])
    ax=fig.add_subplot(gs[:,0]);ax.set_xlim(0,1);ax.set_yscale('log');ax.set_xlabel(r'$D=1-\langle Z_E\rangle$');ax.set_ylabel('Global E rate (Hz / neuron)')
    style=dict(stable=dict(ls='-',color='k'),unstable=dict(ls='--',color='k'))
    for label in a.branches:
        rows=branch_rows(label)
        if not rows:continue
        D,g,cls=segments(rows,stability_map(label))
        start=0
        for k in range(1,len(D)+1):
            if k==len(D) or cls[k]!=cls[start]:
                if cls[start]=='unclassified':ax.plot(D[start:k],g[start:k],'.',color='0.6',ms=2)
                else:ax.plot(D[start:k],g[start:k],lw=1.2,**style[cls[start]])
                start=k
        info=read(DEST/'equilibria'/label/'result.json')
        for f in info.get('folds',[]):
            ax.plot(f['D'],f['global_E_hz'],marker='D',mfc='white',mec='k',ms=6,ls='none')
    # periodic branches: classification only between Floquet-assessed orbits with the same verdict
    def cycle_plot(ax,label,every=25):
        p=DEST/'periodic'/label/'continuation.json'
        if not p.exists():return
        rows=read(p)['rows'];D=np.array([r['D'] for r in rows]);m=np.array([r['global_mean_hz'] for r in rows]);lo=np.array([r['global_min_hz'] for r in rows]);hi=np.array([r['global_max_hz'] for r in rows])
        fl={}
        for f in glob.glob(str(DEST/'periodic/floquet'/f'{label}_*.json')):
            q=read(f);fl[q['orbit']]=q['stability']
        assessed=[(i,fl[r['orbit']]) for i,r in enumerate(rows) if r['orbit'] in fl]
        def cls(k):
            left=[(i,c) for i,c in assessed if i<=k];right=[(i,c) for i,c in assessed if i>=k]
            if not left or not right:return 'unknown'
            cl=left[-1][1];cr=right[0][1]
            if cl==cr and cl in('STABLE_SAMPLED','UNSTABLE'):return cl
            return 'unknown'
        for k in range(len(D)-1):
            c=cls(k);ls='-' if c=='STABLE_SAMPLED' else ('--' if c=='UNSTABLE' else ':')
            ax.plot(D[k:k+2],m[k:k+2],color='tab:orange',lw=1.4,ls=ls)
        for k in range(0,len(D),max(1,len(D)//every)):
            c=cls(k);mf='tab:green' if c=='STABLE_SAMPLED' else 'white';mec='tab:green' if c!='unknown' else '0.5'
            ax.plot([D[k],D[k]],[max(lo[k],.05),hi[k]],marker='s',ms=4,mfc=mf,mec=mec,ls='none')
    for label in a.cycles:cycle_plot(ax,label)
    for f in glob.glob(str(DEST/'periodic/LPC*.json')):
        q=read(f);ax.plot(q['D'],q.get('global_mean_hz',np.nan),marker='*',ms=12,color='tab:red',ls='none')
    for f in glob.glob(str(DEST/'periodic/hopf/*.json')):
        q=read(f)
        if 'D' in q:ax.plot(q['D'],q['global_E_hz'],marker='^',ms=9,color='tab:purple',ls='none')
    # insets for narrow structures
    def inset(bounds,xr,yr,log=True):
        ins=ax.inset_axes(bounds);ins.set_xlim(*xr);ins.set_ylim(*yr)
        if log:ins.set_yscale('log')
        for label in a.branches:
            rows=branch_rows(label)
            if not rows:continue
            D,g,cls=segments(rows,stability_map(label));start=0
            for k in range(1,len(D)+1):
                if k==len(D) or cls[k]!=cls[start]:
                    if cls[start]=='unclassified':ins.plot(D[start:k],g[start:k],'.',color='0.6',ms=2)
                    else:ins.plot(D[start:k],g[start:k],lw=1.,**style[cls[start]])
                    start=k
            for f in read(DEST/'equilibria'/label/'result.json').get('folds',[]):ins.plot(f['D'],f['global_E_hz'],marker='D',mfc='white',mec='k',ms=5,ls='none')
        for label in a.cycles:cycle_plot(ins,label,every=40)
        for f in glob.glob(str(DEST/'periodic/LPC*.json')):
            q=read(f);ins.plot(q['D'],q.get('global_mean_hz',np.nan),marker='*',ms=10,color='tab:red',ls='none')
        ins.tick_params(labelsize=7);ax.indicate_inset_zoom(ins,edgecolor='0.4');return ins
    inset([.36,.30,.24,.26],(0,.035),(.1,1.5))
    inset([.66,.55,.30,.22],(.33,.42),(300,500),log=False)
    handles=[Line2D([],[],color='k',ls='-',label='Equilibrium: stable'),Line2D([],[],color='k',ls='--',label='Equilibrium: unstable'),Line2D([],[],marker='.',color='0.6',ls='none',label='Equilibrium: unclassified'),
             Line2D([],[],color='tab:orange',ls='-',label='Cycle mean: stable'),Line2D([],[],color='tab:orange',ls='--',label='Cycle mean: unstable'),Line2D([],[],color='tab:orange',ls=':',label='Cycle mean: not classified'),
             Line2D([],[],marker='s',mfc='tab:green',mec='tab:green',ls='none',label='Cycle extrema: stable'),Line2D([],[],marker='s',mfc='white',mec='tab:green',ls='none',label='Cycle extrema: unstable'),
             Line2D([],[],marker='D',mfc='white',mec='k',ls='none',label='Fold of equilibria (SN)'),Line2D([],[],marker='*',color='tab:red',ls='none',ms=11,label='Fold of cycles (LPC)'),Line2D([],[],marker='^',color='tab:purple',ls='none',label='Hopf (H)')]
    ax.legend(handles=handles,loc='lower right',bbox_to_anchor=(1.0,.02),fontsize=7,frameon=False)
    # right: spatial panels from a frozen-Z run family (stage B / C snapshots)
    panels=a.panels or []
    for i,spec in enumerate(panels[:6]):
        path,t0,title=spec.split(':');z=np.load(path);t=z['time_ms'];fld=z['field_E_hz'];k=np.searchsorted(t,float(t0));img=fld[max(0,k-a.window//2):k+a.window//2].mean(0).reshape(20,20)
        axp=fig.add_subplot(gs[i%3,1+i//3]);im=axp.imshow(img,origin='lower',cmap='magma',vmin=0,vmax=a.vmax,extent=[0,20,0,20]);axp.set_title(title,fontsize=9)
        for c,lab in zip(centers,'AB'):axp.add_patch(plt.Circle(c,1.5,fill=False,ec='cyan',lw=1));axp.text(c[0],c[1]+1.9,lab,color='cyan',ha='center',fontsize=8)
        axp.set_xlabel('x (mm)');axp.set_ylabel('y (mm)')
    if panels:fig.colorbar(im,ax=fig.axes[1:],shrink=.7,label='E rate (Hz / neuron)')
    fig.tight_layout();out=DEST/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'fig_zm_bifurcation_v3.{ext}',dpi=160)
    log('wrote',out/'fig_zm_bifurcation_v3.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branches',nargs='+',default=['lower','lower_cont','upper','upper_cont']);p.add_argument('--cycles',nargs='+',default=['cycleUp','cycleDown'])
    p.add_argument('--panels',nargs='*');p.add_argument('--window',type=int,default=100);p.add_argument('--vmax',type=float,default=300.);main(p.parse_args())

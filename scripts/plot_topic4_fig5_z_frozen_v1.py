#!/usr/bin/env python3
"""Draw the actual frozen-filtered-v1 equilibrium and periodic branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.signal import resample
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT as BASE
OUT=BASE/"frozen_filtered_v1"


def read(name,default=None):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else default


def main():
    eq=Equilibrium();m=eq.m;folder=OUT/'figures';folder.mkdir(exist_ok=True)
    h1=read('core_b_crossing.json');h2=read('oscillatory_crossing.json');fold=json.loads((BASE/'low_fold_check.json').read_text())
    status={}
    for name in ['equilibrium_stability_certificates.json']:
        for r in read(name,[]):
            key=(r['branch'],r['index'])
            if r['status'].startswith('UNSTABLE'):status[key]='unstable'
    orbits={};qa=[]
    for core,dirname in [('A','periodic_from_crossing'),('B','periodic_from_crossing_b')]:
        for f in sorted((OUT/dirname).glob('amp*_N*.npz')):
            a=np.load(f);s=float(a['s']);N=len(a['r']);error=float(a['residual_hz'])
            if error>1e-5:continue
            r=a['r'];full=resample(r,8*N,axis=0)
            re=(full[:,:3200]*m.w_u).reshape(8*N,400,8).sum(2)
            g=re@m.count_e/m.count_e.sum()*1000
            amp=float(a['control_amplitude_hz'])
            row=dict(core=core,control_amplitude_hz=amp,s=s,mean_z=1-s,N=N,period_ms=float(a['T']),minimum_hz=float(g.min()),maximum_hz=float(g.max()),mean_hz=float(g.mean()),
                     residual_hz=error,unit_minimum_hz=float(full.min()*1000),unit_maximum_hz=float(full.max()*1000),file=str(f))
            qa.append(row);key=(core,amp)
            if row['minimum_hz']>=0 and row['unit_minimum_hz']>=-.01 and (key not in orbits or N>orbits[key]['N']):orbits[key]=row
    chosen=sorted(orbits.values(),key=lambda x:(x['core'],x['control_amplitude_hz']))
    (OUT/'periodic_readouts.json').write_text(json.dumps(dict(all_resolutions=qa,displayed=chosen),indent=2)+'\n')
    if chosen:
        with (OUT/'periodic_readouts.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(chosen[0]));w.writeheader();w.writerows(chosen)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    blue,red,green,orange,gray='#2565a7','#c44034','#21866f','#d28c29','#9aa3ab'
    for view in ['full','detail']:
        fig,ax=plt.subplots(figsize=(10.1,8.3));fig.subplots_adjust(left=.13,right=.97,bottom=.19,top=.79)
        fig.text(.13,.96,'Spatial-Z branch continuation',fontweight='bold',fontsize=22,va='top')
        fig.text(.13,.905,'Fig.5 frozen-Z reduction v1 · filtered input variance · deterministic mean input',fontsize=11,color='#5e6266')
        fig.text(.13,.877,'Exploratory model result: native irregular-burst boundary not reproduced',fontsize=10,color='#5e6266')
        if True:
            ax.axvspan(-.32,0,color='#ededed',zorder=-10)
            if view=='full':ax.text(-.285,410,'Mathematical extension: '+r'$Z>1$',fontsize=10,color='#656565',va='top')
        # The stable base and the first oscillatory crossing are outside resource bounds.
        a=np.load(BASE/'extended_resting_equilibria.npz');x=a['s'];y=np.average(a['r_hz'][:,:400],weights=m.count_e,axis=1)
        # Insert refined Hopf points to avoid shifting the visible stability boundary to a grid node.
        x=np.r_[x,h1['s'],h2['s']];y=np.r_[y,h1['mean_e_hz'],h2['mean_e_hz']];order=np.argsort(x);x=x[order];y=y[order]
        sel=x<=h1['s'];ax.plot(x[sel],y[sel],color=blue,lw=2.5)
        sel=x>=h1['s'];ax.plot(x[sel],y[sel],color=red,lw=2.2,ls='--')
        for name in ['extended_low_equilibria','low_to_recruited_equilibria','recruited_equilibria']:
            a=np.load(BASE/f'{name}.npz');x=a['s'];y=np.average(a['r_hz'][:,:400],weights=m.count_e,axis=1)
            pts=np.c_[x,y];segments=np.stack([pts[:-1],pts[1:]],axis=1)
            for kind,color,style in [('unknown',gray,':'),('unstable',red,'--')]:
                inds=[i for i in range(len(x)-1) if ('unstable' if status.get((name,i))=='unstable' and status.get((name,i+1))=='unstable' else 'unknown')==kind]
                ax.add_collection(LineCollection(segments[inds],colors=color,linestyles=style,linewidths=1.9,zorder=2))
        for core,mark in [('B','o'),('A','s')]:
            rows=[r for r in chosen if r['core']==core]
            if not rows:continue
            sx=np.array([r['s'] for r in rows]);lo=np.array([r['minimum_hz'] for r in rows]);hi=np.array([r['maximum_hz'] for r in rows]);mean=np.array([r['mean_hz'] for r in rows])
            ax.fill_between(sx,lo,hi,color=green,alpha=.065,lw=0,zorder=0)
            ax.plot(sx,hi,mark+'-',color=green,lw=2.3,ms=4,zorder=4);ax.plot(sx,lo,mark+'-',color=green,lw=2.3,ms=4,zorder=4)
            ax.plot(sx,mean,mark+'-',color=orange,lw=2.3,ms=4,zorder=5)
        for info,label,dy in [(h1,'H1 (core B)',38),(h2,'H2 (core A)',76),(fold,'Low-state fold',37)]:
            ax.plot(info['s'],info['mean_e_hz'],'o',mfc='white',mec='#151515',ms=6,zorder=9)
            dx=(-65 if view=='full' else 0) if label.startswith('H1') else 3
            ax.annotate(label,(info['s'],info['mean_e_hz']),xytext=(dx,dy),textcoords='offset points',fontsize=11,arrowprops=dict(arrowstyle='-',color='#555',lw=.8))
        s3=float(eq.ss[list(eq.times).index(9.42)])
        spans=[(0,.27),(.69,1)] if view=='full' and chosen else [(0,1)]
        for ymin,ymax in spans:ax.axvline(s3,ymin=ymin,ymax=ymax,color='#656565',ls=(0,(2,3)),lw=1.2,zorder=1)
        if view=='full':ax.text(s3-.004,190,'Fig.5 ③\n'+r'$\langle Z_E\rangle=0.7712$',ha='right',va='center',fontsize=11,color='#4e4e4e')
        ax.set_yscale('symlog',linthresh=.1,linscale=.7)
        ticks=[0,.05,.1,.3,1,3,10,30,100,300,500];ax.set_yticks(ticks);ax.set_yticklabels([f'{t:g}' for t in ticks]);ax.minorticks_off()
        ax.set_ylim(-.004,520);ax.set_xlim((-.31,.325) if view=='full' else (-.115,.06))
        if view=='detail':
            ax.set_ylim(.012,10)
            ax.set_yticks([.02,.05,.1,.3,1,3,10]);ax.set_yticklabels(['0.02','0.05','0.1','0.3','1','3','10'])
            ax.text(.02,.025,'Gray area: '+r'$Z>1$'+' (outside resource bounds)',transform=ax.transAxes,fontsize=10,color='#656565')
        ax.set_xlabel(r'Z depletion along the spatial path,  $s=1-\langle Z_E\rangle$',fontsize=13,labelpad=10)
        ax.set_ylabel('Global E firing rate (Hz / neuron)',fontsize=14)
        top=ax.secondary_xaxis('top',functions=(lambda x:1-x,lambda z:1-z));top.set_xlabel(r'Mean inhibitory resource,  $\langle Z_E\rangle$',fontsize=12,labelpad=8);top.spines['top'].set_visible(True)
        handles=[Line2D([],[],color=blue,lw=2.5,label='Stable equilibrium'),Line2D([],[],color=red,lw=2.2,ls='--',label='Unstable equilibrium'),Line2D([],[],color=gray,lw=1.8,ls=':',label='Equilibrium: stability unresolved'),Line2D([],[],color=green,lw=2.3,label='Periodic maximum / minimum'),Line2D([],[],color=orange,lw=2.6,label='Period mean')]
        ax.legend(handles=handles,loc='upper left',fontsize=10.5,frameon=False,bbox_to_anchor=(.02,.88 if view=='full' else .99))
        if view=='full' and chosen:
            small=ax.inset_axes([.65,.34,.32,.30])
            small.set_facecolor('#fafafa')
            rest=np.load(BASE/'extended_resting_equilibria.npz')
            xx=np.r_[rest['s'],h1['s'],h2['s']]
            yy=np.r_[np.average(rest['r_hz'][:,:400],weights=m.count_e,axis=1),h1['mean_e_hz'],h2['mean_e_hz']]
            order=np.argsort(xx);xx,yy=xx[order],yy[order]
            small.plot(xx[xx<=h1['s']],yy[xx<=h1['s']],color=blue,lw=1.7)
            small.plot(xx[xx>=h1['s']],yy[xx>=h1['s']],color=red,lw=1.5,ls='--')
            for core,mark in [('B','o'),('A','s')]:
                rows=[r for r in chosen if r['core']==core]
                if not rows:continue
                sx=[r['s'] for r in rows]
                for key,color in [('minimum_hz',green),('maximum_hz',green),('mean_hz',orange)]:
                    small.plot(sx,[r[key] for r in rows],mark+'-',color=color,lw=1.5,ms=2.7)
            xmin=min(r['s'] for r in chosen)-.003;xmax=max(r['s'] for r in chosen)+.003
            small.set_xlim(xmin,xmax)
            small.set_ylim(min(r['minimum_hz'] for r in chosen)*.92,max(r['maximum_hz'] for r in chosen)*1.08)
            small.tick_params(labelsize=8);small.set_xlabel('Z depletion, s',fontsize=9,labelpad=2)
            small.set_ylabel('Global E rate (Hz)',fontsize=9,labelpad=3)
            small.set_title('Oscillatory branches: Z > 1',fontsize=9,pad=6)
        fig.text(.13,.09,'Green / orange: solved oscillatory invariant waveforms; cycle stability is unclassified.',fontsize=10,color='#62666a')
        fig.text(.13,.061,'H1 / H2: oscillatory crossings. Circles / squares: core B / A families. Fig.5 ③ remains unresolved.',fontsize=10,color='#62666a')
        name='fig5_z_branch_continuation'+('_detail' if view=='detail' else '')
        for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
        plt.close(fig)
    print('PLOTTED',len(chosen),'periodic points',flush=True)


if __name__=='__main__':main()

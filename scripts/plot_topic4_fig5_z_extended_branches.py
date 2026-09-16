"""Draw only computed branches; preserve continuation order through folds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.signal import resample
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT as BASE
from continue_topic4_fig5_z_periodic_arc import OUT,PRIOR


def read(path,default):return json.loads(path.read_text()) if path.exists() else default


def main():
    eq=Equilibrium();m=eq.m;dest=OUT/'figures';dest.mkdir(exist_ok=True)
    h1=read(PRIOR/'core_b_crossing.json',{});h2=read(PRIOR/'oscillatory_crossing.json',{});fold=read(BASE/'low_fold_check.json',{})
    certified={}
    for path in [PRIOR/'equilibrium_stability_certificates.json',OUT/'equilibrium_stability_certificates.json',*OUT.glob('*complex_certificates*.json')]:
        for r in read(path,[]):
            if r['status'].startswith('UNSTABLE'):certified[r['branch'],r['index']]=True
    branches=[]
    for folder,names in [(BASE,['extended_low_equilibria']),(OUT,['equilibrium_trusted_forward','equilibrium_low_trusted_continued','equilibrium_high_trusted','equilibrium_high_trusted_continued'])]:
        for name in names:
            path=folder/f'{name}.npz'
            if not path.exists():continue
            z=np.load(path);branches.append(dict(name=name,s=z['s'],rate=np.average(z['r_hz'][:,:400],weights=m.count_e,axis=1)))
    cycles=[]
    for row in read(PRIOR/'periodic_readouts.json',{}).get('displayed',[]):
        cycles.append(dict(row,family=row['core'],order=row['control_amplitude_hz']))
    for row in read(OUT/'extended_periodic_resolution_qa.json',[]):
        path=Path(row['file']);family='A' if '_a_' in str(path) else 'B'
        if row['off_grid_residual_hz']>1e-4 or row['unit_minimum_hz']<-.01:continue
        if path.stem.startswith('amp'):order=float(path.stem.split('_')[0][3:])
        else:order=10+int(path.stem[5:])
        cycles.append(dict(row,family=family,order=order))
    for row in read(OUT/'refined_periodic_resolution_qa.json',[]):
        if row['off_grid_residual_hz']<1e-4 and row['unit_minimum_hz']>=-.01:
            index=int(Path(row['file']).stem[5:]);cycles=[r for r in cycles if not(r['family']=='A' and r['order']==10+index)];cycles.append(dict(row,family='A',order=10+index))
    physical={};candidates=[]
    for path in list((OUT/'periodic_physical').glob('s*_N*.npz'))+list((OUT/'arc_physical_N512').glob('point[0-9]*.npz')):
        z=np.load(path);r=z['r'];N=len(r);s=float(z['s'])
        if float(z['residual_hz'])>1e-5 or N<512:continue
        dense=resample(r,4*N,axis=0);macro=(dense[:,:3200]*m.w_u).reshape(4*N,400,8).sum(2);g=macro@m.count_e/m.count_e.sum()*1000
        row=dict(file=str(path),s=s,mean_z=1-s,N=N,period_ms=float(z['T']),minimum_hz=float(g.min()),maximum_hz=float(g.max()),mean_hz=float(g.mean()),unit_minimum_hz=float(dense.min()*1000),residual_hz=float(z['residual_hz']),family='physical',order=s)
        if s not in physical or physical[s]['N']<N:physical[s]=row
    for row in physical.values():
        if row['N']>=1024 and row['unit_minimum_hz']>=-.01:cycles.append(row)
        else:candidates.append(row)
    colors=dict(stable='#2565a7',unstable='#c44034',extrema='#21866f',mean='#d28c29',unknown='#8c969f')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    for detail in [False,True]:
        fig,ax=plt.subplots(figsize=(10.5,9.0));fig.subplots_adjust(left=.13,right=.965,bottom=.18,top=.72)
        fig.text(.13,.968,'Expanded spatial-Z branch continuation',fontsize=22,fontweight='bold',va='top')
        fig.text(.13,.920,'Current Fig.5 rate reduction · filtered variance · exact delayed map',fontsize=11,color='#60656b')
        fig.text(.13,.891,'Exploratory reduction: the native irregular-burst boundary is not established',fontsize=10.5,color='#60656b')
        ax.axvspan(-.4,0,color='#efefef',zorder=-5)
        if not detail:ax.text(-.136,450,'Z > 1\nmathematical extension',fontsize=10,color='#70757a',va='top')
        z=np.load(BASE/'extended_resting_equilibria.npz');x=np.r_[z['s'],h1['s'],h2['s']];y=np.r_[np.average(z['r_hz'][:,:400],weights=m.count_e,axis=1),h1['mean_e_hz'],h2['mean_e_hz']];order=np.argsort(x);x,y=x[order],y[order]
        for sel,color,style in [(x<=h1['s'],colors['stable'],'-'),(x>=h1['s'],colors['unstable'],'--')]:ax.plot(x[sel],y[sel],color=color,lw=2.3,ls=style)
        for b in branches:
            points=np.c_[b['s'],b['rate']];segments=np.stack([points[:-1],points[1:]],axis=1)
            known=np.array([certified.get((b['name'],k),False) and certified.get((b['name'],k+1),False) for k in range(len(points)-1)])
            for mask,color,ls,lw in [(~known,colors['unknown'],':',1.2),(known,colors['unstable'],'--',1.8)]:
                if mask.any():ax.add_collection(LineCollection(segments[mask],colors=color,linestyles=ls,linewidths=lw,zorder=2))
            if b['name'].endswith('_continued'):ax.plot(b['s'][-1],b['rate'][-1],'o',mfc='white',mec=colors['unknown'],ms=4,zorder=6)
        for family,marker in [('A','s'),('B','o'),('physical','o')]:
            rows=sorted([r for r in cycles if r['family']==family],key=lambda r:r['order'])
            if not rows:continue
            xx=[r['s'] for r in rows];lo=[r['minimum_hz'] for r in rows];hi=[r['maximum_hz'] for r in rows]
            ax.fill_between(xx,lo,hi,color=colors['extrema'],alpha=.06,zorder=1)
            for key,color in [('minimum_hz',colors['extrema']),('maximum_hz',colors['extrema']),('mean_hz',colors['mean'])]:ax.plot(xx,[r[key] for r in rows],marker+'-',ms=3.7,lw=2.0,color=color,zorder=4)
        if not detail:
            for row in read(OUT/'deterministic_comparison.json',[]):
                d=row['readouts']['global'];mean=d['mean_last_1s']
                ax.errorbar([row['s']],[mean],yerr=[[mean-d['minimum_last_1s']],[d['maximum_last_1s']-mean]],fmt='D',ms=5,mfc='white',mec='#20262d',ecolor='#20262d',capsize=3,elinewidth=1,zorder=7)
        for h,label,offset in [(h1,'H1',(-30,27)),(h2,'H2',(4,53)),(fold,'Low-state fold',(18,35))]:
            if detail and label=='H1':offset=(-30,60)
            if detail and label=='Low-state fold':offset=(-115,62)
            ax.plot(h['s'],h['mean_e_hz'],'o',ms=5,mfc='white',mec='#222',zorder=8)
            ax.annotate(label,(h['s'],h['mean_e_hz']),xytext=offset,textcoords='offset points',fontsize=10,arrowprops=dict(arrowstyle='-',lw=.7,color='#555'))
        s3=float(eq.ss[list(eq.times).index(9.42)])
        ax.axvline(s3,color='#72777c',lw=1.1,ls=(0,(2,3)),zorder=0)
        if not detail:ax.text(s3-.006,80,'Fig.5 ③\n'+r'$s=0.22884$'+'\nDirect run: sustained high rate',ha='right',fontsize=10,color='#5b6268')
        secondary=read(OUT/'physical_secondary_fold_check.json',{})
        if not detail and secondary:
            ax.plot(secondary['s'],secondary['mean_e_hz'],'o',mfc='white',mec='#333',ms=5,zorder=7)
            ax.annotate('Stationary turn\n'+r'$s=0.14313$',(secondary['s'],secondary['mean_e_hz']),xytext=(22,-38),textcoords='offset points',fontsize=10,arrowprops=dict(arrowstyle='-',lw=.7,color='#555'))
        ax.set_yscale('symlog',linthresh=.1,linscale=.7);ax.minorticks_off()
        ticks=[0,.05,.1,.3,1,3,10,30,100,300,500] if not detail else [.03,.05,.1,.2,.3,.5]
        ax.set_yticks(ticks);ax.set_yticklabels([f'{x:g}' for x in ticks]);ax.set_ylim((-.002,520) if not detail else (.025,.5));ax.set_xlim((-.145,.325) if not detail else (-.113,-.019))
        ax.set_xlabel(r'Z depletion along the measured spatial path,  $s=1-\langle Z_E\rangle$',fontsize=12.5,labelpad=12)
        ax.set_ylabel('Global E firing rate (Hz / neuron)',fontsize=14)
        top=ax.secondary_xaxis('top',functions=(lambda x:1-x,lambda z:1-z));top.set_xlabel(r'Mean inhibitory resource,  $\langle Z_E\rangle$',fontsize=12,labelpad=10);top.spines['top'].set_visible(True)
        legend=[Line2D([],[],color=colors['stable'],lw=2,label='Stable equilibrium'),Line2D([],[],color=colors['unstable'],lw=2,ls='--',label='Unstable equilibrium'),Line2D([],[],color=colors['unknown'],lw=1.5,ls=':',label='Equilibrium: stability unresolved'),Line2D([],[],color=colors['extrema'],lw=2,label='Periodic maximum / minimum'),Line2D([],[],color=colors['mean'],lw=2,label='Period mean')]
        if not detail:legend.append(Line2D([],[],marker='D',ms=5,mfc='white',mec='#20262d',ls='',label='Direct run: finite-time mean / range'))
        fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.125,.866),ncol=2,fontsize=10,frameon=False,columnspacing=2.0)
        fig.text(.13,.095,'Curves retain continuation order through folds; open ends are computed limits.',fontsize=10,color='#656b70')
        fig.text(.13,.065,'Cycle stability is unclassified. H1 / H2 are oscillatory crossings in the Z > 1 extension.',fontsize=10,color='#656b70')
        name='fig5_z_branch_extension'+('_detail' if detail else '')
        for extension in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{extension}',dpi=190)
        plt.close(fig)
    report=dict(equilibrium_branches=[dict(name=b['name'],points=len(b['s']),minimum_s=float(b['s'].min()),maximum_s=float(b['s'].max())) for b in branches],displayed_cycles=cycles,candidates=candidates)
    (OUT/'figure_data.json').write_text(json.dumps(report,indent=2)+'\n');print('PLOTTED',report['equilibrium_branches'],flush=True)


if __name__=='__main__':main()

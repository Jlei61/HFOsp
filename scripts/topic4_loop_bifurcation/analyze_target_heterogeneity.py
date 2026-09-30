#!/usr/bin/env python3
"""Assess all local groups and display the diagnostic edge failure/repair."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_target_heterogeneity import OUT,SEEDS


def main():
    result=read(OUT/'result.json');assert result['status']=='COMPLETE'
    z=dict(np.load(OUT/'reference.npz'));ids=z['cell_group_index'];sizes=z['group_sizes'];G=len(sizes)
    native=z['native_counts'].reshape(100,200,G).sum(1)/sizes/.02
    names=z['moment_names'].tolist();mom=z['native_moments'][99::100]
    mean=mom[:,[names.index('IE'),names.index('II')]].transpose(0,2,1)
    std=np.sqrt(np.maximum(mom[:,[names.index('IE2'),names.index('II2')]].transpose(0,2,1)-mean**2,0))
    labels=read(ROOT/'native_exit_branch_inputs/contract.json')['selected_groups'];cache={};rows=[]
    for job in result['completed']:
        d=dict(np.load(OUT/f'{job}.npz'));cellrate=d['rate_Hz'].reshape(100,20,len(ids)).mean(1);m=d['moments']
        rate=np.stack([cellrate[:,ids==g].mean(1) for g in range(G)],axis=1)
        groupmean=np.stack([m[:,ids==g,:2].mean(1) for g in range(G)],axis=1)
        groupsecond=np.stack([(m[:,ids==g,2:]+m[:,ids==g,:2]**2).mean(1) for g in range(G)],axis=1)
        groupstd=np.sqrt(np.maximum(groupsecond-groupmean**2,0));groups=[]
        for g in range(G):
            a=float(native[25:,g].mean());p=float(rate[25:,g].mean())
            groups.append(dict(group=int(z['selected_groups'][g]),role=labels[g]['role'],native_rate_Hz=a,density_rate_Hz=p,
                rate_delta_Hz=p-a,native_tail_current_std_mv=std[50:,g].mean(0).tolist(),
                density_tail_current_std_mv=groupstd[50:,g].mean(0).tolist(),
                current_mean_error_mv=(groupmean[50:,g]-mean[50:,g]).mean(0).tolist()))
        rows.append(dict(job=job,groups=groups));cache[job]=dict(rate=rate,mean=groupmean,std=groupstd)
    out=dict(status='COMPLETE',rows=rows,unit='One native conditional continuation; six numerical local conditions, not six native outcomes.',
        meaning='Target input heterogeneity is separated from input variance heterogeneity. Source counts and R/G are prescribed; no network or stability certification.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'analysis.json',out)
    for row in rows:print(row['job'],row['groups'][0],flush=True)
    plot(z,native,std,cache)


def plot(z,native,std,cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    edge=int(np.flatnonzero(z['selected_groups']==531)[0]);tr=42+(np.arange(100)+.5)*.02;tm=42+np.arange(1,201)*.01
    arms={'homogeneous':('#287c8e','Group inputs'),'individual_mean':('#cc7722','Individual means'),'individual_full':('#7a3b94','Individual means + variances')}
    axes[0,0].plot(tr,native[:,edge],'k',lw=1.3,label='Native')
    for j in range(2):axes[1,j].plot(tm,std[:,edge,j],'k',lw=1.3)
    for arm,(color,label) in arms.items():
        rr=np.array([cache[f'{arm}_num{s}']['rate'] for s in SEEDS]);ss=np.array([cache[f'{arm}_num{s}']['std'] for s in SEEDS])
        axes[0,0].plot(tr,rr[:,:,edge].mean(0),color=color,lw=1.1,label=label)
        axes[0,0].fill_between(tr,rr[:,:,edge].min(0),rr[:,:,edge].max(0),color=color,alpha=.2)
        axes[0,1].scatter(native[25:].mean(0),rr[:,25:].mean((0,1)),s=26,facecolors='none',edgecolors=color,label=label)
        for j in range(2):
            yy=ss[:,:,edge,j];axes[1,j].plot(tm,yy.mean(0),color=color,lw=1.1)
            axes[1,j].fill_between(tm,yy.min(0),yy.max(0),color=color,alpha=.2)
    axes[0,1].plot([0,650],[0,650],':',color='.6',lw=.8)
    axes[0,1].set(xlim=(-15,650),ylim=(-15,650),xlabel='Native group rate (Hz)',ylabel='Local density rate (Hz)',title='All 16 groups, 42.5–44 s')
    axes[0,0].set(ylabel='20-ms E rate (Hz)',title='Edge group 531')
    axes[0,0].legend(frameon=False,fontsize=8,loc='lower right')
    for j,title in enumerate(['E input spread','I input spread']):axes[1,j].set(title=title,ylabel='Within-group SD (mV equiv.)')
    for ax in [axes[0,0],axes[1,0],axes[1,1]]:ax.set(xlim=(42,44),xlabel='Native time (s)');ax.axvline(42.5,color='.75',ls=':',lw=.8)
    fig.suptitle('Target-cell input heterogeneity restores the missed edge activity',weight='bold')
    fig.text(.5,-.02,'Local diagnostic with native source counts and R/G prescribed; Z/K held. Fixed original graph, no fitted parameters; autonomous coupling not certified.',ha='center',fontsize=8)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'target_input_heterogeneity.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':main()

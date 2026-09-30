#!/usr/bin/env python3
"""Native versus local density at fixed actual-field Z=.21/K=9."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_exit_branch_density import OUT,SOURCE,SEEDS,T,physical


def main():
    result=read(OUT/'result.json');assert result['status']=='COMPLETE'
    assert read(SOURCE/'observer_audit.json')['status']=='PASS'
    assert sha(physical.__file__)==read(OUT/'contract.json')['engine_sha256']
    z=dict(np.load(OUT/'input_summary.npz'));sizes=z['pars'][:,5];G=len(sizes)
    labels=read(SOURCE/'contract.json')['selected_groups']
    assert np.array_equal(z['selected_groups'],[v['group'] for v in labels])
    masks={key:np.array([v['population']==pop and (v['region']==2 if key=='surround_E' else v['region']!=2 if key=='core_E' else True) for v in labels])
           for key,pop in [('surround_E',0),('core_E',0),('I',1)]}
    names=z['moment_names'].tolist();native=z['native_counts'].reshape(T//200,200,G).sum(1)
    native_curr=z['native_moments'][99::100][:,[names.index('IE'),names.index('II')]].transpose(0,2,1)
    native_std=np.sqrt(np.maximum(z['native_moments'][99::100][:,[names.index('IE2'),names.index('II2')]].transpose(0,2,1)-native_curr**2,0))
    rows=[];cache={}
    for job in result['completed']:
        d=dict(np.load(OUT/f'{job}.npz'));rates=d['rate_Hz'];M=d['moments']
        assert rates.shape==(2000,G) and M.shape==(200,G,7)
        assert np.isfinite(rates).all() and np.isfinite(M).all()
        pred=rates.reshape(100,20,G).sum(1)*sizes/1000
        regions=[]
        for region,mask in masks.items():
            w=sizes[mask]/sizes[mask].sum();windows=[]
            for label,lo,hi in [('initial',0,25),('tail',25,100)]:
                a=native[lo:hi,mask].sum(1);p=pred[lo:hi,mask].sum(1)
                windows.append(dict(window=label,time_s=[42+lo*.02,42+hi*.02],native_count=int(a.sum()),
                   density_count=float(p.sum()),count_ratio=float(p.sum()/a.sum()) if a.sum() else None,
                   native_rate_Hz=float(a.sum()/sizes[mask].sum()/((hi-lo)*.02)),
                   density_rate_Hz=float(p.sum()/sizes[mask].sum()/((hi-lo)*.02)),
                   relative_L2=float(np.linalg.norm(p-a)/np.linalg.norm(a)) if np.linalg.norm(a) else None))
            currents=[]
            for j,key in enumerate(['IE','II']):
                a=native_curr[50:,mask,j]@w;p=M[50:,mask,3+j]@w
                currents.append(dict(variable=key,native_mean=float(a.mean()),density_mean=float(p.mean()),
                    signed_mean_error=float((p-a).mean()),RMS_error=float(np.sqrt(np.mean((p-a)**2))),
                    native_mean_within_group_std=float((native_std[50:,mask,j]@w).mean()),
                    density_mean_within_group_std=float((M[50:,mask,5+j]@w).mean())))
            regions.append(dict(region=region,windows=windows,currents_tail=currents))
        groups=[]
        for g in range(G):
            a=native[25:,g].sum();p=pred[25:,g].sum()
            groups.append(dict(group=int(z['selected_groups'][g]),role=labels[g]['role'],N=int(sizes[g]),
                  native_count=int(a),density_count=float(p),count_ratio=float(p/a) if a else None,
                  native_rate_Hz=float(a/sizes[g]/1.5),density_rate_Hz=float(p/sizes[g]/1.5),
                  current_mean_error_mv=(M[50:,g,3:5]-native_curr[50:,g]).mean(0).tolist()))
        rows.append(dict(job=job,regions=regions,groups=groups));cache[job]=d
    out=dict(status='COMPLETE',rows=rows,statistical_unit='One original native conditional continuation; 16 geometric groups, two numerical streams per arm. Groups are not independent network replications.',
        source_interval_s=[42.,44.],current_time_alignment='Updated current at native tick99 equals local post100step current; no extra timestep shift.',
        scope='Input-forced local check with held native Z/K; it does not validate the freely coupled network or stability.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'analysis.json',out)
    plot(z,cache,sizes,masks,native,native_curr)
    for row in rows:
        print(row['job'],[(r['region'],r['windows'][1]['native_rate_Hz'],r['windows'][1]['density_rate_Hz'],r['windows'][1]['count_ratio']) for r in row['regions']])


def plot(z,cache,sizes,masks,native,native_curr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,4,figsize=(14,8),sharex=True,layout='constrained')
    t_rate=42+(np.arange(100)+.5)*.02;t_current=42+np.arange(1,201)*.01
    colors={'projected_full':'#287c8e','measured_full':'#cc7722'}
    panels=dict(masks);panels['edge_E']=z['selected_groups']==531
    for col,(region,mask) in enumerate(panels.items()):
        w=sizes[mask]/sizes[mask].sum()
        axes[0,col].plot(t_rate,native[:,mask].sum(1)/sizes[mask].sum()/.02,'k',lw=1.4,label='Native')
        for j in range(2):axes[j+1,col].plot(t_current,native_curr[:,mask,j]@w,'k',lw=1.3)
        for arm,color in colors.items():
            for row in range(3):
                yy=[]
                for seed in SEEDS:
                    d=cache[f'{arm}_num{seed}']
                    yy.append(d['rate_Hz'].reshape(100,20,len(sizes)).mean(1)[:,mask]@w if row==0 else d['moments'][:,mask,2+row]@w)
                yy=np.array(yy);tx=t_rate if row==0 else t_current
                axes[row,col].plot(tx,yy.mean(0),color=color,lw=1.0,label='Projected means' if arm=='projected_full' else 'Measured means')
                axes[row,col].fill_between(tx,yy.min(0),yy.max(0),color=color,alpha=.2)
        axes[0,col].set_title({'surround_E':'Selected surround E','core_E':'Selected core E','I':'Selected I','edge_E':'Edge E: group 531'}[region],weight='bold')
        axes[2,col].set_xlabel('Native time (s)')
        for ax in axes[:,col]:ax.axvline(42.5,color='.75',ls=':',lw=.8);ax.set_xlim(42,44)
    for row,label in enumerate(['20-ms rate (Hz)','E input\n(mV equiv.)','I input\n(mV equiv.)']):axes[row,0].set_ylabel(label)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Actual exit fields, K=9: local response under native inputs',weight='bold')
    fig.text(.5,-.017,'Z/K held; native incoming activity and R/G prescribed. Bands show two numerical streams, not uncertainty across native networks.',ha='center',fontsize=8)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'conditional_exit_branch_density.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':main()

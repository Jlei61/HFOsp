#!/usr/bin/env python3
"""Fixed native-time assessment of a conditional local density diagnostic."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OUT,SOURCE,physical


def describe(pred,truth):
    return dict(native_count=float(truth.sum()),predicted_count=float(pred.sum()),
                count_ratio=float(pred.sum()/truth.sum()) if truth.sum() else None,
                fixed50ms_relative_L2=float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1.)))


def main():
    result=read(OUT/'result.json');assert result['status']=='COMPLETE'
    contract=read(OUT/'contract.json');assert sha(physical.__file__)==contract['engine_sha256']
    z=dict(np.load(OUT/'native_inputs.npz'));selected=z['selected_groups'];G=len(selected)
    sizes=z['pars'][:,5];names=z['moment_names'].tolist();mom=z['native_moments']
    native=z['native_counts'][5000:].reshape(50,500,G).sum(1)
    end_mom=mom[99::100];native_z=mom[100:30000:100,names.index('z')]
    native_var=np.stack([end_mom[:,names.index('ampa2')]-end_mom[:,names.index('ampa')]**2,
                         end_mom[:,names.index('gaba2')]-end_mom[:,names.index('gaba')]**2],axis=-1)
    masks=dict(surround_E=np.arange(G)<9,core_E=(np.arange(G)>=9)&(np.arange(G)<11),I=np.arange(G)>=11)
    rows=[];cache={}
    for job in result['completed']:
        d=dict(np.load(OUT/f'{job}.npz'));rates=d['rate_Hz'];M=d['moments']
        assert rates.shape==(3000,G) and M.shape==(300,G,7)
        assert np.isfinite(rates).all() and np.isfinite(M).all()
        bins=rates[500:].reshape(50,50,G).sum(1)/1000*sizes
        # Counts re-read from physical rates; Z end-step versus next native pre-step.
        zerror=M[:-1,:,0]-native_z
        group=[]
        for g in range(G):
            group.append(dict(group=int(selected[g]),**describe(bins[:,g],native[:,g]),
                              Z_RMS_error=float(np.sqrt(np.mean(zerror[49:,g]**2))),
                              Z_error_at2990ms=float(zerror[-1,g]),
                              native_Z_2990ms=float(native_z[-1,g]),density_Z_2990ms=float(M[-2,g,0]),
                              native_current_variance_mv2=native_var[50:,g].mean(0).tolist(),
                              density_current_variance_mv2=M[50:,g,4:6].mean(0).tolist()))
        regional=[]
        for region,mask in masks.items():
            w=sizes[mask]/sizes[mask].sum()
            regional.append(dict(region=region,**describe(bins[:,mask].sum(1),native[:,mask].sum(1)),
                Z_error_at2990ms=float(zerror[-1,mask]@w),
                native_Z_2990ms=float(native_z[-1,mask]@w),density_Z_2990ms=float(M[-2,mask,0]@w)))
        rows.append(dict(job=job,groups=group,regions=regional));cache[job]=dict(bins=bins,moments=M)
    contrast=[]
    for variance in ['full','private_sensitivity']:
        for region,mask in masks.items():
            comparisons=[]
            for seed in [927641,927642]:
                values=[]
                for mean in ['projected','measured']:
                    job=f'{mean}_{variance}_num{seed}'
                    values.append(next(r for r in rows if r['job']==job)['regions'][list(masks).index(region)])
                comparisons.append(dict(numerical_seed=seed,
                    count_ratio_change=values[1]['count_ratio']-values[0]['count_ratio'],
                    relative_L2_change=values[1]['fixed50ms_relative_L2']-values[0]['fixed50ms_relative_L2'],
                    Z_error_change=values[1]['Z_error_at2990ms']-values[0]['Z_error_at2990ms']))
            contrast.append(dict(variance=variance,region=region,paired_numerical_streams=comparisons))
    write(OUT/'analysis.json',dict(status='COMPLETE',rows=rows,paired_mean_contrasts=contrast,
        statistical_unit='One original native trajectory.16selectedgroups andtimebins dependent; two MonteCarlo streams measure numerical variability only.',
        interpretation='Input-conditioned local diagnostic. Private sensitivity is not exact conditional covariance or a replacement networkmodel. No fitting,newnative trajectories,autonomouscorrespondence or stabilityclaim.',
        source_sha256=sha(__file__),engine_sha256=sha(physical.__file__),formal_bifurcation_allowed=False))
    for row in rows:print(row['job'],[(r['region'],round(r['count_ratio'],5),round(r['fixed50ms_relative_L2'],5),round(r['Z_error_at2990ms'],6)) for r in row['regions']],flush=True)
    plot(z,cache,native,masks,sizes)


def plot(native_inputs,cache,native,masks,sizes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(11.7,8.3),gridspec_kw={'width_ratios':[2.1,1.]},layout='constrained')
    colors={'projected_full':'#287c8e','measured_full':'#cc7722'};titles=['Surround E (9 selected groups)','Core E (2 selected groups)','I (5 selected groups)']
    starts=np.arange(500,3000,50)/1000+.025
    for i,(region,mask) in enumerate(masks.items()):
        n=sizes[mask].sum();ref=native[:,mask].sum(1)/n/.05
        axes[i,0].plot(starts,ref,color='black',lw=1.6,label='Native')
        for name,color in colors.items():
            values=np.array([cache[f'{name}_num{s}']['bins'][:,mask].sum(1)/n/.05 for s in [927641,927642]])
            label='Projected means' if name.startswith('projected') else 'Measured means'
            axes[i,0].plot(starts,values.mean(0),color=color,lw=1.1,label=label)
            axes[i,0].fill_between(starts,values.min(0),values.max(0),color=color,alpha=.2)
            bins=np.array([cache[f'{name}_num{s}']['bins'][:,mask].sum(0)/native[:,mask].sum(0) for s in [927641,927642]])
            x=np.arange(mask.sum())+(0 if name.startswith('projected') else .15)
            axes[i,1].plot(x,bins.mean(0),'o',color=color,ms=4)
            axes[i,1].vlines(x,bins.min(0),bins.max(0),color=color,lw=2)
        axes[i,0].set_title(titles[i],loc='left',weight='bold');axes[i,0].set_ylabel('Population rate (Hz)')
        axes[i,0].set_xlim(.5,3);axes[i,1].axhline(1,color='black',lw=.8,ls='--')
        axes[i,1].set_xticks(np.arange(mask.sum())+.075,native_inputs['selected_groups'][mask],rotation=45)
        axes[i,1].set_ylabel('Count / native count');axes[i,1].set_ylim(.75,1.25)
        axes[i,1].set_xlim(-.5,mask.sum()-.3)
    axes[0,0].legend(ncol=3,frameon=False,loc='upper left',bbox_to_anchor=(0,1.33),fontsize=9)
    axes[-1,0].set_xlabel('Native time (s)');axes[-1,1].set_xlabel('Fixed geometric group')
    fig.suptitle('Conditional local density response — native incoming activity prescribed',weight='bold')
    fig.text(.5,-.013,'Full physical variance shown; bands = two numerical streams, not native uncertainty. G/K off; autonomous closure and bifurcation unverified.',ha='center',fontsize=9)
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(dest/f'conditional_density_response.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'figure_metadata.json',dict(producer_sha256=sha(__file__),agent_visual='PENDING',human_review='PENDING',
        primary_arms='full variance; all private sensitivity results retained inanalysis',display_count_ratio_limits=[.75,1.25]))


if __name__=='__main__':main()

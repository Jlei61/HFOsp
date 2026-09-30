#!/usr/bin/env python3
"""Fixed-window G/K local closure assessment; no fitted acceptance rule."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,time
import numpy as np
from campaign import read,write,sha,ROOT
from conditional_exit_density import OUT,physical


def main(wait=False):
    while not (OUT/'result.json').exists():
        if not wait:raise RuntimeError('Local exit runs are incomplete')
        write(OUT/'analysis_progress.json',dict(status='WAITING_LOCAL_CONDITIONS',pid=os.getpid(),updated_epoch=time.time()))
        time.sleep(20)
    result=read(OUT/'result.json');assert result['status']=='COMPLETE'
    z=dict(np.load(OUT/'input_summary.npz'));G=len(z['selected_groups']);sizes=z['pars'][:,5]
    assert sha(physical.__file__)==read(OUT/'contract.json')['engine_sha256']
    names=z['moment_names'].tolist();mom=z['native_moments'];native=z['native_counts'].reshape(165,200,G).sum(1)
    masks=dict(surround_E=np.arange(G)<9,core_E=(np.arange(G)>=9)&(np.arange(G)<11),I=np.arange(G)>=11)
    source_mom=mom[100::100];rows=[];cache={}
    for job in result['completed']:
        d=dict(np.load(OUT/f'{job}.npz'));M=d['moments'];rates=d['rate_Hz']
        assert rates.shape==(3300,G) and M.shape==(330,G,7) and np.isfinite(M).all()
        pred=rates.reshape(165,20,G).sum(1)*sizes/1000
        regions=[]
        for region,mask in masks.items():
            w=sizes[mask]/sizes[mask].sum();windows=[]
            for label,lo,hi in [('suppression',0,10),('G_tail',10,45),('recovery',45,165)]:
                actual=native[lo:hi,mask].sum(1);p=pred[lo:hi,mask].sum(1)
                windows.append(dict(window=label,time_s=[16.7+lo*.02,16.7+hi*.02],native_count=int(actual.sum()),
                    density_expected_count=float(p.sum()),count_ratio=float(p.sum()/actual.sum()) if actual.sum() else None,
                    count_RMS_per20ms=float(np.sqrt(np.mean((p-actual)**2))),
                    relative_L2=float(np.linalg.norm(p-actual)/np.linalg.norm(actual)) if np.linalg.norm(actual) else None))
            state=[]
            for key,ch in [('Z',0),('K',1),('M',2)]:
                truth=source_mom[:,names.index(key),:][:,mask]@w;value=M[:-1,:,ch][:,mask]@w
                state.append(dict(variable=key,RMS_error=float(np.sqrt(np.mean((value-truth)**2))),
                    max_absolute_error=float(abs(value-truth).max()),native_final19990ms=float(truth[-1]),
                    density_final19990ms=float(value[-1])))
            regions.append(dict(region=region,windows=windows,state=state))
        group=[]
        for g in range(G):
            group.append(dict(group=int(z['selected_groups'][g]),native_count=int(native[:,g].sum()),
                density_count=float(pred[:,g].sum()),
                Z_RMS_error=float(np.sqrt(np.mean((M[:-1,g,0]-source_mom[:,names.index('Z'),g])**2))),
                K_RMS_error=float(np.sqrt(np.mean((M[:-1,g,1]-source_mom[:,names.index('K'),g])**2))),
                Z_std_RMS_error=float(np.sqrt(np.mean((M[:-1,g,5]-np.sqrt(np.maximum(source_mom[:,names.index('Z2'),g]-source_mom[:,names.index('Z'),g]**2,0)))**2))),
                K_std_RMS_error=float(np.sqrt(np.mean((M[:-1,g,6]-np.sqrt(np.maximum(source_mom[:,names.index('K2'),g]-source_mom[:,names.index('K'),g]**2,0)))**2)))))
        rows.append(dict(job=job,regions=regions,groups=group));cache[job]=d
    raw=mom[:,[names.index('IE'),names.index('II')]]
    mean_errors=np.sqrt(np.mean((z['projected_current_means']-raw)**2,axis=0))
    out=dict(status='COMPLETE',rows=rows,projected_raw_current_RMS_error_mv=mean_errors.tolist(),
        statistical_unit='Oneoriginalnativeexit,16geometricgroups; two numericalstreams areMonteCarloerror,not independentnative outcomes.',
        scope='Actualtime-varyingG/K local-response diagnostic, withnativeincomingcountsandglobalR/Gprescribed. Noautonomousfeedbackorlinearstabilitycertification.',
        formal_bifurcation_allowed=False,source_sha256=sha(__file__))
    write(OUT/'analysis.json',out);write(OUT/'analysis_progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    for row in rows:
        print(row['job'],[(r['region'],r['windows'][0]['count_ratio'],[(a['variable'],a['max_absolute_error']) for a in r['state'][:2]]) for r in row['regions']],flush=True)
    plot(z,cache,sizes,masks)


def plot(z,cache,sizes,masks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(10.5,8.2),sharex='col',layout='constrained')
    names=z['moment_names'].tolist();colors={'projected_full':'#287c8e','measured_full':'#cc7722'}
    for col,region in enumerate(['surround_E','core_E']):
        mask=masks[region];w=sizes[mask]/sizes[mask].sum()
        counts=z['native_counts'].reshape(3300,10,len(sizes)).sum(1)[:,mask].sum(1)/sizes[mask].sum()*1000
        axes[0,col].plot(16.7+(np.arange(3300)+.5)/1000,counts,color='black',lw=.8,label='Native')
        native_mom=z['native_moments'][100::100];tx=z['time_ms'][100::100]/1000
        for row,key in [(1,'Z'),(2,'K')]:axes[row,col].plot(tx,native_mom[:,names.index(key),:][:,mask]@w,color='black',lw=1.6)
        for arm,color in colors.items():
            label='Projected means' if arm.startswith('projected') else 'Measured means'
            for row,ch in [(0,None),(1,0),(2,1)]:
                curves=[]
                for seed in [927651,927652]:
                    d=cache[f'{arm}_num{seed}']
                    curves.append(d['rate_Hz'][:,mask]@w if row==0 else d['moments'][:-1,:,ch][:,mask]@w)
                yy=np.array(curves);t=16.7+(np.arange(3300)+.5)/1000 if row==0 else tx
                axes[row,col].plot(t,yy.mean(0),color=color,lw=1.1,label=label if row==0 else None)
                axes[row,col].fill_between(t,yy.min(0),yy.max(0),color=color,alpha=.2)
        axes[0,col].set_title('Selected surround E' if col==0 else 'Selected core E',loc='left',weight='bold')
        axes[2,col].set_xlabel('Native time (s)')
        for ax in axes[:,col]:
            ax.axvline(16.9,color='.7',lw=.7,ls=':');ax.axvline(17.6,color='.7',lw=.7,ls=':');ax.set_xlim(16.7,20.)
    for row,label in enumerate(['1-ms population rate (Hz)','Mean Z','Mean K / gL']):axes[row,0].set_ylabel(label)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Conditional local response through the natural exit',weight='bold')
    fig.text(.5,-.015,'Actual incoming activity and global R/G prescribed; own Z/M/K evolve. Bands: numerical streams only. Autonomous closure and stability not certified.',ha='center',fontsize=8.5)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'conditional_exit_density.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)

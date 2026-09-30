#!/usr/bin/env python3
"""Retain spatial discrepancies after the paired-count exit-time check."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_coupled_density_exit import native_data,candidate,ADAPTED
from analyze_native_count_noise import load_native

OUT=ROOT/'native_count_noise'
WINDOWS=[(13.7,13.8),(17,17.1),(18,20)]


def main():
    assert read(OUT/'comparison.json')['status']=='COMPLETE'
    for seed in [928701,928702]:assert read(OUT/'runs'/f'count{seed}'/'paired_input_audit.json')['status']=='PASS'
    geo=dict(np.load(ADAPTED/'geometry.npz'))
    data=[native_data(geo)]+[load_native(seed,geo) for seed in [928701,928702]]+[candidate(seed,geo) for seed in [927671,927672]]
    names=['Native original','Native count 928701','Native count 928702','Density 927671','Density 927672']
    rows=[];all_fields=[]
    for lo,hi in WINDOWS:
        fields=[];rates=[]
        for d in data:
            keep=(d['field_time_ms']>=lo*1000)&(d['field_time_ms']<hi*1000)
            mask=(d['rate_time_ms']>=lo*1000)&(d['rate_time_ms']<hi*1000)
            fields.append(d['fields'][keep].mean(0));rates.append(d['rates'][mask].mean(0))
        fields=np.array(fields);rms=np.zeros((5,5))
        for i in range(5):
            for j in range(5):rms[i,j]=np.sqrt(np.average((fields[i]-fields[j])**2,weights=data[0]['cell_counts']))
        rows.append(dict(interval_s=[lo,hi],mean_rates_allE_A_B_other=np.array(rates).tolist(),mean_fields_Hz=fields.tolist(),pairwise_weighted_mean_field_RMS_Hz=rms.tolist()))
        all_fields.append(fields)
    report=dict(status='COMPLETE',names=names,rows=rows,
        selection='Posthoc spatialdiscrepancylocalization atfirstdip/firstreactivation/latewindow; not independentprospectivevalidation.',
        interpretation='Original andtwo new countstreams formthree conditionalnative realizations. Thetwo earlyexitnative trajectories are displayedseparately, not pooledas anindependentcohortorusedtoestimateconfidencebands. Reactivationhere is sustainedactivity, notinterictaleventreturn.',
        native_correspondence_certified=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'spatial_comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,5,figsize=(16,10),layout='constrained')
    for row,((lo,hi),fields) in enumerate(zip(WINDOWS,all_fields)):
        for col in range(5):
            ax=axes[row,col];im=ax.imshow(fields[col].reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            for center in geo['centers_mm']:ax.add_patch(Circle(center,1.5,fill=False,color='#00bec7',lw=.8))
            ax.set_title(f'{names[col]}\n{lo:g}–{hi:g} s',fontsize=9);ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if col==0:ax.set_ylabel('y (mm)')
            if row==2:ax.set_xlabel('x (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,pad=.01,label='E population rate (Hz)')
    fig.suptitle('Exit timing can vary while spatial correspondence remains a separate question',weight='bold')
    fig.text(.5,-.012,'Same nativeinitialstate andOU path. Actualcountrealizations versus numericaldensitystreams; sharedcolor scale, no phasealignment.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'native_count_noise_space.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);write(OUT/'spatial_figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))
    print('PAIRWISE FIELD RMS18-20',rows[-1]['pairwise_weighted_mean_field_RMS_Hz'],flush=True)


if __name__=='__main__':main()

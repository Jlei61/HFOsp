#!/usr/bin/env python3
"""Compare actual finite events before entry and during autonomous return."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as obs


def main(root,name):
    folder=root/'runs'/name
    finite=obs.read(folder/'finite_event_resource_audit.json')
    extent=obs.read(folder/'event_extent_audit.json')
    result=obs.read(folder/'result.json')
    first,second=result['tracker']['entries'][:2]
    data=obs.load(folder,['regions_1ms','field_5ms'])
    geo=np.load(root/'geometry.npz')
    rates=gaussian_filter1d(data['regions_1ms'][:,:2].astype(float)*1000/
                           geo['region_counts'][:2],2,axis=0)
    groups={
        'Before entry':[e.copy() for e in extent['episodes'] if e['finite_event'] and
                        e['start_s']>=.2 and e['end_s']<first['onset_s']],
        'During return':[e.copy() for e in extent['episodes'] if e['finite_event'] and
                         e['start_s']>=first['confirmation_s'] and e['end_s']<second['onset_s']]}
    assert len(groups['Before entry'])==finite['preentry_finite_event_count']
    expected=[e for e in finite['post_first_entry_finite_events'] if e['end_s']<second['onset_s']]
    assert [(e['start_s'],e['end_s']) for e in groups['During return']]==[(e['start_s'],e['end_s']) for e in expected]
    summary={}
    for label,events in groups.items():
        for event in events:
            lo,hi=round(event['start_s']*1000),round(event['end_s']*1000)
            piece=rates[lo:hi]
            onset=[]
            for j in range(2):
                peak=piece[:,j].max()
                hits=np.flatnonzero(piece[:,j]>=max(5.,.2*peak)) if peak>=5 else []
                onset.append(float(hits[0]) if len(hits) else None)
            event['core_B_minus_A_onset_ms']=onset[1]-onset[0] if all(v is not None for v in onset) else None
        summary[label]=dict(n=len(events),metrics={})
        for key in ['duration_s','peak_all_E_Hz','native_recruited_area_union',
                    'native_peak_simultaneous_area','preceding_quiet_s','core_B_minus_A_onset_ms']:
            values=np.array([e[key] for e in events if e[key] is not None],float)
            summary[label]['metrics'][key]=dict(n=len(values),min=float(values.min()),
                median=float(np.median(values)),max=float(values.max())) if len(values) else dict(n=0)
    plt.rcParams.update({'font.size':15,'axes.labelsize':18,'xtick.labelsize':14,'ytick.labelsize':15})
    fig,axes=plt.subplots(1,4,figsize=(19,5.3))
    features=[('duration_s',1000,'Event duration (ms)'),('peak_all_E_Hz',1,'Peak all-E rate (Hz)'),
              ('native_recruited_area_union',1,'Recruited area (fraction)'),
              ('core_B_minus_A_onset_ms',1,'Core B − A onset (ms)')]
    for k,(key,mult,label) in enumerate(features):
        valid_counts=[]
        for i,(group,color) in enumerate(zip(groups,['#397fa6','#32957c'])):
            values=np.array([e[key]*mult for e in groups[group] if e[key] is not None])
            valid_counts.append(len(values))
            jitter=np.linspace(-.16,.16,len(values)) if len(values)>1 else np.zeros(len(values))
            axes[k].scatter(i+jitter,values,s=22,c=color,alpha=.65,edgecolors='none')
            if len(values):axes[k].plot([i-.23,i+.23],[np.median(values)]*2,c=color,lw=2.6)
        axes[k].set(xticks=[0,1],xticklabels=[f'Before\nn={valid_counts[0]}',
                        f'Return\nn={valid_counts[1]}'],ylabel=label,xlim=(-.5,1.5))
        axes[k].spines[['top','right']].set_visible(False)
        axes[k].text(-.17,1.04,'ABCD'[k],transform=axes[k].transAxes,fontsize=23,weight='bold')
    axes[3].axhline(0,c='.65',ls=':',lw=.8)
    fig.subplots_adjust(left=.065,right=.985,top=.90,bottom=.21,wspace=.52)
    out=folder/'returned_repertoire';figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figures/f'returned_event_comparison.{ext}',dpi=170)
    plt.close(fig)
    # Same preentry example as the main figure, and first finite event AFTER
    # the operational low-state confirmation. Frame windows remain native10ms.
    prior=obs.read(folder/'figure_metadata.json')['state_times_s'][1]
    before=min(groups['Before entry'],key=lambda e:abs(e['peak_time_s']-prior))
    after=[e for e in groups['During return'] if e['start_s']>=result['tracker']['recoveries'][0]['confirmation_s']]
    examples=[before,after[0]] if after else []
    frame_records=[]
    if examples:
        fig,axes=plt.subplots(2,6,figsize=(19,7.6))
        for row,event in enumerate(examples):
            requested=np.linspace(event['start_s'],event['end_s'],6)
            for col,tm in enumerate(requested):
                first_bin=round((tm-.005)/.005)
                field=data['field_5ms'][first_bin:first_bin+2].sum(0)/geo['cell_e_counts']/.01
                actual=first_bin*.005+.005
                ax=axes[row,col]
                ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],
                          norm=PowerNorm(.6,0,500),cmap='magma',interpolation='nearest')
                for xy in geo['centers_mm']:ax.add_patch(plt.Circle(xy,float(geo['core_radius_mm']),
                    fill=False,ec='#41c6c9',lw=1.2))
                ax.set(xticks=[0,20],yticks=[0,20],xlabel='x (mm)')
                ax.set_title(f'{actual:.3f} s',fontsize=16)
                if col==0:ax.set_ylabel(('Before entry' if row==0 else 'Return event')+'\ny (mm)')
                else:ax.set_yticklabels([])
                frame_records.append(dict(row=row,column=col,time_s=actual,
                    window_s=[first_bin*.005,(first_bin+2)*.005],rate_Hz=field.tolist()))
        fig.subplots_adjust(left=.07,right=.99,top=.92,bottom=.10,wspace=.22,hspace=.40)
        for ext in ['png','pdf']:fig.savefig(figures/f'event_field_sequences.{ext}',dpi=170)
        plt.close(fig)
    output=dict(source=str(folder),groups=groups,summary=summary,frame_records=frame_records,
        event_definition='Same finite event observer:10ms whole-E peak>=20Hz, active duration<=300ms, bounded by>=30ms<5Hz. Return group includes fully ended events after first confirmation and before second onset.',
        onset_lag='Actual1ms core-neighborhood rates, Gaussian sigma2ms display/measurement filter; first crossing max(5Hz,20%within-event peak). B minusA, positive meansA earlier. May be affected by event-window boundaries; not patient labels.',
        example_selection='Preentry event nearest main-figure state2; first finite event after low-state confirmation. Six uniformly selected actual times per event, native10ms fields. No event-time warping or spatial interpolation.',
        statistical_unit='One network/noise trajectory; event dots are descriptive repeated observations, not independent simulations. No inferential p values.',
        scientific_boundary='Finite activity has returned but its strength, duration, recruitment and core timing may differ from the initial interictal repertoire. Passing the200ms high gate again does not prove a new sustained seizure state.',
        human_review='PENDING')
    (out/'repertoire_review.json').write_text(json.dumps(output,indent=2)+'\n')
    (figures/'README.md').write_text('### returned_event_comparison.png / .pdf\n同一连续轨迹中，首次进入之前与第一次高活动结束、再次进入之前的有限事件分别展示时长、峰率、原生招募面积及双核起始时差。每个点是一件事件，横线为中位数，不作跨事件独立重复的统计检验。**关注点**：返回短事件不代表已经返回原来的间期传播分布。\n\n### event_field_sequences.png / .pdf\n比较主图的一个进入前事件与低态确认后的第一个有限事件；各行六帧为该实际事件时段内均匀取时的10ms原生细胞场，共用0–500Hz与gamma=.6显示。**关注点**：时间轴未压缩或扭曲，观察初始局部传播和返回后广泛同步招募是否属于相同机制。\n')
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    args=p.parse_args();main(args.root,args.name)

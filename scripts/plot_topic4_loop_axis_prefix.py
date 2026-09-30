#!/usr/bin/env python3
"""Common 0–8s native graph comparison, without selecting events by morphology."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_topic4_loop_axis_native as run
import analyze_topic4_interictal_recurrence as audit
import run_topic4_rhythm_preserving_feedback as rhythm


def prefix(folder,subdir,keys,end=8.):
    parts={k:[] for k in keys}
    for path in sorted((folder/subdir).glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as z:
            for key in keys:parts[key].append(z[key])
            if 'end_step' in z and int(z['end_step'])>=end*10000:break
            if 'time_ms' in z and len(z['time_ms']) and z['time_ms'][-1]>=end*1000:break
    if not parts[keys[0]]:return None
    return {k:np.concatenate(v) for k,v in parts.items()}


def main():
    sources=[('current',run.native.SOURCE,run.native.NAME)]+[
        (c,run.OUT/c,f'{c}_s9108405') for c in ['rotated','isotropic']]
    rows=[];traces=[]
    for condition,root,name in sources:
        folder=root/'runs'/name
        d=prefix(folder,'chunks',['spikes_1ms','regions_1ms','raster','slow_time_ms','Z','inputs'])
        if d is None or len(d['spikes_1ms'])<8000:continue
        fb=prefix(folder,'feedback_chunks',['time_ms','G_applied_mean','K_mean','R_global_Hz'])
        with np.load(root/'geometry.npz') as g:counts=np.r_[32000,g['region_counts'][:3]]
        raw=np.c_[d['spikes_1ms'][:8000,0],d['regions_1ms'][:8000,:3]]
        rates=raw.reshape(800,10,4).sum(1)/counts/.01
        events=rhythm.strict_events(rates,8.)
        activation=json.loads((folder/'feedback_activation.json').read_text())['first_feedback_s']
        windows={}
        for label,lo,hi in [('initial',.5,8.),('fixed_0p5_6s',.5,6.)]:
            r=rates[round(lo*100):round(hi*100)]
            sample=(d['slow_time_ms']>=lo*1000)&(d['slow_time_ms']<hi*1000)
            zz=d['Z'][sample];tt=d['slow_time_ms'][sample]/1000
            before_feedback=activation is None or activation>=hi
            budget=None
            if before_feedback:
                # Column8 is the fraction whose native delivered inhibitory
                # input reaches the Z-load threshold. Before added feedback,
                # this is exactly the local-load eligibility in the Z law.
                budget=dict(mean_recovery_eligible_fraction=float(np.mean(1-zz[:,8])),
                    mean_Z=float(np.mean(zz[:,0])),
                    mean_sampled_natural_dZ_per_s=float(np.mean((1-zz[:,8]-zz[:,0])/5.)),
                    observed_mean_Z_slope_per_s=float((zz[-1,0]-zz[0,0])/(tt[-1]-tt[0])),
                    sampled_times_s=[float(tt[0]),float(tt[-1])],
                    definition='Before added feedback: mean dZ/dt=(fraction with local inhibitory load below threshold - mean Z)/5s. Eligibility and Z sampled every20ms; finite endpoint slope reported separately.')
            windows[label]=dict(window_s=[lo,hi],events=audit.interval_events(events,lo,hi),
                joint_quiet_fraction=float(np.mean(np.all(r[:,:3]<5,axis=1))),
                peak_Hz=r.max(0).tolist(),mean_Hz=r.mean(0).tolist(),
                before_added_feedback=before_feedback,native_Z_budget=budget)
        temporal=audit.temporal_audit(rates)
        rows.append(dict(condition=condition,observed_window_s=[0,8],first_feedback_s=activation,
            windows=windows,all_activity_bouts=events,entries_within8s=temporal['entries'],
            paired_exogenous_input_records_exact=None))
        traces.append((d,fb,rates))
    assert rows and rows[0]['condition']=='current'
    expected=traces[0][0]['inputs'][:80]
    for row,(d,fb,rates) in zip(rows,traces):
        assert np.array_equal(expected,d['inputs'][:80])
        row['paired_exogenous_input_records_exact']=True
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':11,'xtick.labelsize':8,'ytick.labelsize':8})
    fig,axes=plt.subplots(5,len(rows),figsize=(5.1*len(rows),8.4),squeeze=False,
        gridspec_kw={'height_ratios':[2.5,1.8,.20,1,1]},sharex=True)
    colors=['#8952ab','#cf3e87','#159cbe','#37698b']
    feedback_limit=max(1.,np.ceil(max(float(np.max(fb[k][fb['time_ms']<8000]))
        for _,fb,_ in traces for k in ['G_applied_mean','K_mean'])))
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for c,(row,(d,fb,rates)) in enumerate(zip(rows,traces)):
        ax=axes[0,c];it,ix=np.where(d['raster'][:80000])
        for lo,hi,color in [(0,20,colors[1]),(20,40,colors[2]),(40,60,colors[3]),(60,80,'#c17730')]:
            sel=(ix>=lo)&(ix<hi)
            ax.scatter(it[sel]*.0001,mapping[ix[sel]],s=.7,lw=0,c=color,rasterized=True)
        ax.set(ylim=(-2,101),yticks=[16.5,52.5,78,93],yticklabels=['Core A E','Core B E','Other E','I'],
            title={'current':'Current axis','rotated':'Rotated structure','isotropic':'Isotropic structure'}[row['condition']])
        ax=axes[1,c];t=(np.arange(800)+.5)*.01
        for j,label in enumerate(['All E','Core A','Core B','Other E']):ax.plot(t,rates[:,j],lw=.55,c=colors[j],label=label)
        ax.set(ylim=(0,510),ylabel='E rate (Hz)')
        if c==0:ax.legend(frameon=False,ncol=2,fontsize=7,loc='upper right')
        ax=axes[2,c];quiet=np.all(rates[:,:3]<5,axis=1)
        ax.imshow(quiet[None,:],origin='lower',extent=(0,8,0,1),aspect='auto',interpolation='nearest',cmap='Greys',vmin=0,vmax=1)
        ax.set(yticks=[],ylabel='Quiet',xticks=[0,2,4,6,8]);ax.spines[:].set_visible(False)
        ax=axes[3,c];sel=d['slow_time_ms']<8000
        for j,k in [(0,0),(1,5),(2,6)]:ax.plot(d['slow_time_ms'][sel]/1000,d['Z'][sel,k],lw=.85,c=colors[j])
        ax.set(ylim=(0,1.03),ylabel='Resource Z')
        ax=axes[4,c];sel=fb['time_ms']<8000
        ax.plot(fb['time_ms'][sel]/1000,fb['G_applied_mean'][sel],c='#385a34',label='Applied G')
        ax.plot(fb['time_ms'][sel]/1000,fb['K_mean'][sel],c='#bc7837',label='K')
        ax.set(ylabel='Mean g / gL',xlabel='Time (s)',ylim=(0,feedback_limit*1.04))
        if c==0:ax.legend(frameon=False,ncol=2,fontsize=7,loc='upper left')
        for r in [0,1,3,4]:
            axes[r,c].spines[['top','right']].set_visible(False)
            when=row['first_feedback_s']
            if when is not None and when<8:axes[r,c].axvline(when,c='#555555',ls=':',lw=.8)
        for ax in axes[:,c]:ax.set_xlim(0,8)
    fig.subplots_adjust(left=.085,right=.98,top=.94,bottom=.085,hspace=.23,wspace=.25)
    fig.text(.5,.024,'Black strip: All E and both cores < 5 Hz. Dotted line: first feedback activation. Same external input.',ha='center',fontsize=9)
    dest=run.OUT/'prefix_comparison';dest.mkdir(exist_ok=True)
    files=[]
    for ext in ['png','pdf']:
        path=dest/f'axis_native_first8s.{ext}';fig.savefig(path,dpi=180,facecolor='white')
        files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    run.native.write(dest/'analysis.json',dict(rows=rows,files=files,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='PREFIX_COMPLETE' if len(rows)==3 else 'WAITING_REMAINING_PREFIX',
        interpretation='A lack of jointquiet-bounded short events does not establish loss of core bursting. The fixed0.5–6s window is a descriptive diagnostic, not an additional prespecified primary outcome; before-feedback eligibility is checked per graph, and local-load Z budgets are reported only when the whole window precedes activation.',
        structural_limits='Source partner identity, outdegree and lowered-source output strength change; rotation also attenuates anisotropy. Not a pure orientation control.',human_review='PENDING'))
    (dest/'README.md').write_text('### axis_native_first8s.png\n\n三种结构在同一未来输入下的0–8秒原生记录，尚未完成8秒的结构暂不显示。每列依次为固定80细胞raster、两核及全E和核外率、共同低活动条带、Z、实际施加G与K；黑条表示全E及两核同时低于5Hz，虚线为新增反馈首次激活。\n\n**关注点**：核仍能爆发但活动尾部未分离时，短事件检测可能将它们连成一个长活动段；这不等于核爆发消失。结构对照的输出度和低阈值来源输出强度并未匹配，因此不能单独归因于轴方向。图待人工审阅。\n')
    print(json.dumps([dict(condition=r['condition'],brief=r['windows']['initial']['events']['brief_count'],
        fixed_window_brief=r['windows']['fixed_0p5_6s']['events']['brief_count'],feedback_s=r['first_feedback_s']) for r in rows]),flush=True)


if __name__=='__main__':main()

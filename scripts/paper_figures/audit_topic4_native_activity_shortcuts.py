#!/usr/bin/env python3
"""Audit frozen G3 full-sheet activity versus the actual training observer.

No physics, objective, event membership, or patient REVIEW packet is changed.
Native partition is exact at the saved 1 mm / 2 ms voxel level. Counts are
E cells with at least one spike in each frame, not unrestricted spike counts.
It is not an exact per-neuron core partition or causal source identification.
"""
from pathlib import Path
import csv, json, pickle, sys, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.topic4_interictal_repaired_evaluation import rank_features
from src.topic4_envelope_joint_pilot import envelope_descriptor

R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
P=ROOT/'results/topic4_sef_hfo/contact_timing_shape_pilot'
OUT=R/'native_activity_shortcut_audit';F=OUT/'figures'
IDS=['v2_1_pop1_de_b_002','v2_anchor_old_joint__baseline',
     'v2_anchor_support_rank__vth_low','v2_anchor_support_rank__baseline',
     'v2_anchor_old_joint__tau_d_GABA_ms_high','v2_1_pop0_de_a_001','v2_anchor_historical__baseline']
NAMES=['Best feature fit','Reference placement A','Placement B / lower threshold shift',
       'Placement B','Placement A / longer inhibition','Alternative placement','Historical reference']
COLORS=['#9f3d63','#347ba4','#bc8643','#758b4f','#8064a1','#5e9893','#747474']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def csvwrite(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dist(a):
    x=np.asarray(a,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0)
    q=np.quantile(x,[.05,.25,.5,.75,.95])
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std()),median=float(q[2]),
                q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4]))

def regions(z,centers,radius):
    yy,xx=np.mgrid[:20,:20];xy=np.stack([xx+.5,yy+.5],-1)
    dd=np.linalg.norm(xy[:,:,None,:]-centers[None,None,:,:],axis=-1)
    reg=np.where(dd.min(-1)<=radius,dd.argmin(-1),2)
    pos=z['positions_E'].astype(float);ij=np.minimum(np.floor(pos).astype(int),19)
    nb=ij[:,1]*20+ij[:,0];den=np.bincount(nb,minlength=400).reshape(20,20)
    h=z['h']>0;nearest=np.linalg.norm(pos[:,None]-centers,axis=-1).argmin(1)
    nr=np.where(h,nearest,2)
    # Pure bins provide a second comparison without assigning boundary neurons.
    cn=np.array([np.bincount(nb[nr==k],minlength=400).reshape(20,20) for k in range(3)])
    pure=np.array([(cn[k]==den)&(den>0) for k in range(3)])
    return xy,reg,den,pure,nr

def plot_geometry(ax,xy,centers,radius):
    for k,color in enumerate(['#ef9d38','#3ebac0']):
        ax.add_patch(Circle(centers[k],radius,fill=False,ec=color,lw=1.3))
    ax.scatter(xy[:,0],xy[:,1],s=10,facecolors='none',edgecolors='white',linewidths=.7)
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',ylabel='y (mm)')

def main():
    F.mkdir(parents=True,exist_ok=True)
    gs=json.loads((R/'g3_scores.json').read_text())['candidates']
    byid={c['candidate_id']:c for c in gs}
    manifest=json.loads((R/'execution/confirmation_24s/candidate_manifest.json').read_text())
    candidates={c['candidate_id']:c for c in manifest['candidates']}
    # Use known identities where available, retaining every actual G3 candidate.
    ids=[i for i in IDS if i in byid]+[i for i in byid if i not in IDS]
    names={i:NAMES[IDS.index(i)] if i in IDS else i for i in ids}
    with (R/'training_objective_v2_1.pkl').open('rb') as f:objective=pickle.load(f)
    events=[];units=[];splits=[];examples={};sources={};max_mu_error=0.
    for ci,cid in enumerate(ids):
        c=candidates[cid];centers=np.asarray(c['node_field']['centers_mm'])
        for unit,u in sorted(byid[cid]['units'].items()):
            p=Path(u['worker_path']);w=json.loads(p.read_text())
            op=p.parent.parent/'repaired_observation'/p.name
            obs=json.loads(op.read_text())
            assert obs['worker_lineage_onsets_used_for_training'] is False
            with np.load(op.with_suffix('.npz')) as z:
                indices=z['primary_event_indices'];mu=z['centroid_ms'][indices].astype(float)
            modes=objective.km.predict(rank_features(mu))
            rescored=objective.score_network(mu)
            assert abs(rescored['loss_off']-u['score']['loss_off'])<1e-10
            with np.load(p.with_suffix('.npz')) as z:
                raw=z['sheet_activity_counts'].astype(float);fam=z['directed_lineage_labels']
                local=z['local_directed_lineage_labels'];env=z['contact_envelope'].astype(float)
                dt=float(z['sheet_activity_frame_ms']);assert dt==float(z['contact_envelope_dt_ms'])==2.
                contacts=z['contact_xy_mm'];cnames=z['contact_names'].astype(str).tolist()
                radius=w['xy_geometry_audit']['distance_cutoff_mm']
                grid,reg,den,pure,nr=regions(z,centers,radius)
                v=z['vtheta'][:len(nr)];h=z['h']>0
                # All neurons in these bins are at least 1 mm from all contacts.
                all_far=np.linalg.norm(grid[:,:,None]-contacts,axis=-1).min(-1)>1+np.sqrt(.5)
                core=(reg<2);outside=(reg==2)
                row=dict(candidate_id=cid,condition=names[cid],unit=unit,n_events=len(indices),
                    core_mean_threshold_mV=float(v[h].mean()),outside_mean_threshold_mV=float(v[~h].mean()),
                    raised_core_fraction=float(np.mean(v[h]>18)),core_E_count=int(h.sum()),
                    core1_E_count=int(np.sum(nr==0)),core2_E_count=int(np.sum(nr==1)),
                    radius_mm=float(radius),boundary_clearance_mm=float(w['xy_geometry_audit']['minimum_clearance_mm']),
                    old_loss=float(rescored['loss_off']),full_envelope_training=True,
                    raw_legacy_onset_rows=int(len(z['onsets'])),pure_core_bins=int(pure[:2].any(0).sum()),
                    whole_run_mass_far_from_contacts=float(raw[:,all_far].sum()/raw.sum()))
            unit_events=[]
            for k,idx in enumerate(indices):
                e=obs['events'][int(idx)];lo,hi=np.rint(np.array(e['window_ms'])/dt).astype(int)
                a=raw[lo:hi];labels=fam[lo:hi];ll=local[lo:hi]
                total=float(a.sum());fl=np.bincount(np.maximum(labels,0).ravel(),weights=a.ravel())
                dominant=int(np.argmax(fl[1:])+1) if len(fl)>1 else 0
                parts=np.array([a*((labels==dominant)&(labels>0)),a*((labels>0)&(labels!=dominant)),a*(labels<=0)])
                assert np.array_equal(parts.sum(0),a)
                assert abs(parts.sum()-total)<1e-8
                fraction=float(fl[dominant]/total) if dominant else 0.
                lmass=np.bincount(np.maximum(ll,0).ravel(),weights=a.ravel())
                frame_family=np.array([np.bincount(np.maximum(lab,0).ravel(),weights=aa.ravel(),minlength=len(fl)) for aa,lab in zip(a,labels)])
                frame_dominant=frame_family[:,1:].max(1) if len(fl)>1 else np.zeros(len(a))
                # Sequential families alone do not increase this simultaneous-mass statistic.
                concurrent_other=float((frame_family[:,1:].sum(1)-frame_dominant).sum()/total)
                topids=np.argsort(fl[1:])[::-1][:2]+1
                top_overlap=None
                if len(topids)==2 and np.all(fl[topids]>0):
                    ft=frame_family[:,topids]/fl[topids][None]
                    top_overlap=float(np.minimum(ft[:,0],ft[:,1]).sum())
                regional=np.array([a[:,reg==j].sum(1) for j in range(3)])
                assert np.array_equal(regional.sum(0),a.sum((1,2)))
                peak=int(a.sum((1,2)).argmax());peaks=np.argmax(regional[:2],axis=1)
                peak_lag=float((peaks[1]-peaks[0])*dt)
                core_density=float(a[peak,core].sum()/den[core].sum())
                outside_density=float(a[peak,outside].sum()/den[outside].sum())
                density_ratio=outside_density/core_density if core_density>0 else None
                pc=pure[:2].any(0);po=pure[2]
                pure_ratio=float((a[peak,po].sum()/den[po].sum())/(a[peak,pc].sum()/den[pc].sum())) if a[peak,pc].sum()>0 else None
                mass=np.maximum(env[:,lo:hi]-np.array(e['local_baseline'])[:,None],0)
                desc=envelope_descriptor(mass,(np.arange(lo,hi)+.5)*dt,np.isfinite(mu[k]))
                err=float(np.nanmax(abs(desc['centroid']-mu[k])));max_mu_error=max(max_mu_error,err);assert err<.002
                ev=dict(candidate_id=cid,condition=names[cid],unit=unit,event_id=int(idx),mode=int(modes[k]),
                    window_start_ms=float(lo*dt),raw_active_cell_frames=total,dominant_family_id=dominant,
                    dominant_family_fraction=fraction,other_family_fraction=float(parts[1].sum()/total),
                    unassigned_or_collision_fraction=float(parts[2].sum()/total),
                    collision_fraction=float(a[labels<0].sum()/total),
                    unassigned_fraction=float(a[labels==0].sum()/total),n_local_labels_with_spikes=int(np.sum(lmass[1:]>0)),
                    n_family_labels_with_spikes=int(np.sum(fl[1:]>0)),
                    effective_assigned_family_count=float(fl[1:].sum()**2/np.sum(fl[1:]**2)) if fl[1:].sum()>0 else None,
                    simultaneous_secondary_family_mass_fraction=concurrent_other,
                    top_two_families_temporal_overlap=top_overlap,
                    core2_minus_core1_peak_ms=peak_lag,absolute_core_peak_lag_ms=abs(peak_lag),
                    outside_to_core_peak_per_neuron_ratio=density_ratio,
                    core_silent_at_global_peak=bool(core_density==0),
                    outside_peak_density_share=outside_density/(outside_density+core_density),
                    pure_bins_outside_to_core_peak_per_neuron_ratio=pure_ratio,
                    mass_far_from_contacts_fraction=float(a[:,all_far].sum()/total),
                    **desc['statistics'])
                events.append(ev);unit_events.append(ev)
                for si,sname in enumerate(['largest_family','other_families','unassigned_or_collision']):
                    for ri,rname in enumerate(['core1_bins','core2_bins','outside_bins']):
                        splits.append(dict(candidate_id=cid,unit=unit,event_id=int(idx),family_group=sname,
                            region=rname,active_cell_frames=float(parts[si,:,reg==ri].sum()),fraction_of_window=float(parts[si,:,reg==ri].sum()/total)))
                key=(cid,int(modes[k]))
                if ci<2 and key not in examples:
                    examples[key]=dict(event=ev,raw=a,parts=parts,reg=reg,den=den,regional=regional,
                        time=(np.arange(hi-lo)+.5)*dt,contacts=contacts,names=cnames,
                        centers=centers,radius=radius,mass=mass,peak=peak)
            for met in ['dominant_family_fraction','absolute_core_peak_lag_ms','n_local_labels_with_spikes',
                        'outside_to_core_peak_per_neuron_ratio','pure_bins_outside_to_core_peak_per_neuron_ratio',
                        'simultaneous_secondary_family_mass_fraction','outside_peak_density_share','mass_far_from_contacts_fraction','overlap']:
                row[met+'_median']=dist([e[met] for e in unit_events])['median']
            for lab in [0,1]:
                ee=[e for e in unit_events if e['mode']==lab]
                row[f'mode{lab}_n']=len(ee)
                row[f'mode{lab}_core1_leads_fraction']=float(np.mean([e['core2_minus_core1_peak_ms']>0 for e in ee])) if ee else None
            units.append(row)
            sources[str(op)]=hashlib.sha256(op.read_bytes()).hexdigest()
    csvwrite(OUT/'events.csv',events);csvwrite(OUT/'units.csv',units);csvwrite(OUT/'family_by_region.csv',splits)
    summaries=[]
    for cid in ids:
        ee=[e for e in events if e['candidate_id']==cid]
        for lab in [-1,0,1]:
            ev=ee if lab==-1 else [e for e in ee if e['mode']==lab]
            for metric in ['dominant_family_fraction','absolute_core_peak_lag_ms','n_local_labels_with_spikes',
                           'outside_to_core_peak_per_neuron_ratio','pure_bins_outside_to_core_peak_per_neuron_ratio',
                           'simultaneous_secondary_family_mass_fraction','outside_peak_density_share','mass_far_from_contacts_fraction','overlap']:
                summaries.append(dict(candidate_id=cid,condition=names[cid],mode=lab,metric=metric,**dist([e[metric] for e in ev])))
    csvwrite(OUT/'event_distribution_summary.csv',summaries)
    # Patient comparison uses TRAIN blocks only; this does not consume pilot REVIEW.
    patient=list(csv.DictReader((P/'patient_training_event_observables.csv').open()))
    pd={str(k):dist([float(e['overlap']) for e in patient if int(e['mode'])==k]) for k in [0,1]}
    write(OUT/'audit.json',dict(status='FROZEN_OUTPUT_AUDIT_COMPLETE',units=len(units),events=len(events),
        exact_reproduction_of_frozen_training_scores=True,maximum_envelope_centroid_rounding_error_ms=max_mu_error,
        training_uses_worker_lineage_onsets=False,full_envelope_has_no_lineage_mask=True,
        native_partition='exact 1mm/2ms saved voxel counts; core bin centers approximate true membership',
        native_count_unit='distinct E neurons with >=1 spike per 2ms frame; summed over frames gives active-cell-frames, not unrestricted spike count',
        largest_family='most raw mass in each repaired window; NOT an event selected for training',
        exact_neuron_lineage_contact_decomposition='NOT_RECONSTRUCTIBLE: per-neuron spikes were deleted before saving',
        patient_TRAIN_only_interval_overlap=pd,patient_review_opened=False,
        root_count_interpretation='segmentation-dependent IDs, not measured independent physical initiators',sources=sources))
    fig,axs=plt.subplots(1,4,figsize=(15,4.8));fig.subplots_adjust(left=.055,right=.99,bottom=.36,top=.84,wspace=.34)
    mets=['dominant_family_fraction','absolute_core_peak_lag_ms','outside_peak_density_share','overlap']
    titles=['Largest family: fraction of activity','Core peak separation (ms)','Outside peak density share','Contact interval overlap']
    for ax,met,title in zip(axs,mets,titles):
        for ci,cid in enumerate(ids):
            vv=[u[met+'_median'] for u in units if u['candidate_id']==cid]
            ax.scatter(np.arange(len(vv))*.07+ci-.10,vv,s=28,c=COLORS[ci]);ax.plot([ci-.17,ci+.17],[np.median(vv)]*2,c=COLORS[ci],lw=2)
        ax.set(title=title,xticks=range(len(ids)),xticklabels=[f'C{i+1}' for i in range(len(ids))],ylim=(0,None));ax.grid(axis='y',alpha=.15)
    axs[2].axhline(.5,c='gray',lw=1,ls='--');axs[2].set_ylim(0,1)
    axs[3].axhspan(min(pd['0']['q25'],pd['1']['q25']),max(pd['0']['q75'],pd['1']['q75']),color='gray',alpha=.2)
    axs[3].set_ylim(0,1)
    fig.suptitle('Full-sheet activity and contact observations describe different aspects',fontsize=14)
    for ci,cid in enumerate(ids):fig.text(.075+(ci//4)*.48,.24-(ci%4)*.043,f'C{ci+1}  {names[cid]}',color=COLORS[ci])
    fig.text(.5,.025,'Dots: four topology/noise units; each value is an event median. Gray band: TRAIN patient TA/TB overlap IQR envelope.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(F/f'native_activity_summary.{ext}',dpi=160)
    plt.close(fig)
    # One first eligible example per mode per condition, without fit-based selection.
    ex=[examples[(cid,lab)] for cid in ids[:2] for lab in [1,0]]
    cap=max(float(np.quantile(e['raw'][e['raw']>0],.99)) for e in ex)
    fig,axs=plt.subplots(4,4,figsize=(13,11),gridspec_kw={'width_ratios':[1,1,1,1.45]})
    fig.subplots_adjust(left=.055,right=.965,bottom=.08,top=.93,hspace=.72,wspace=.29)
    for ri,e in enumerate(ex):
        k=e['peak'];maps=[e['raw'][k],e['parts'][0,k],e['parts'][1:,k].sum(0)]
        for ax,ma in zip(axs[ri,:3],maps):
            im=ax.imshow(ma,origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=cap,interpolation='none')
            plot_geometry(ax,e['contacts'],e['centers'],e['radius'])
        for j,t in enumerate(['All native activity','Largest family only','Other / unassigned activity']):axs[ri,j].set_title(t,fontsize=9)
        ax=axs[ri,3]
        for j,(color,label) in enumerate(zip(['#d38b30','#269ca8','#6b6577'],['Core 1 bins','Core 2 bins','Outside bins'])):
            rate=e['regional'][j]/e['den'][e['reg']==j].sum()
            ax.plot(e['time'],rate,c=color,label=label,lw=1.2)
        ax.axvline(e['time'][k],ls='--',c='gray');ax.set(xlabel='Time in observation window (ms)',ylabel='Active E fraction / 2 ms',xlim=(0,250),ylim=(0,None))
        ev=e['event'];label='TA-labelled' if ev['mode']==1 else 'TB-labelled'
        axs[ri,0].text(0,1.24,f'{ev["condition"]} | {label}',transform=axs[ri,0].transAxes,fontsize=11,fontweight='bold')
    axs[0,3].legend(fontsize=8)
    cb=fig.colorbar(im,ax=axs[:,:3],orientation='horizontal',fraction=.02,pad=.05);cb.set_label('Active E cells / 1 mm bin / 2 ms (same scale for every map)')
    fig.text(.5,.01,'Maps at full-window global peak. Orange/cyan circles: core boundaries; white circles: fixed SEEG contacts. Largest family is a diagnostic partition.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(F/f'full_activity_partition_examples.{ext}',dpi=150)
    plt.close(fig)
    # Animated whole-window comparison with shared native scale, no spatial smoothing.
    for lab in ([] if '--skip-movies' in sys.argv else [1,0]):
        pair=[examples[(cid,lab)] for cid in ids[:2]]
        fig,axes=plt.subplots(2,4,figsize=(13,6.4),gridspec_kw={'width_ratios':[1,1,1,1.4]})
        fig.subplots_adjust(left=.055,right=.96,bottom=.15,top=.85,hspace=.65,wspace=.43)
        ims=[];marks=[]
        for ri,e in enumerate(pair):
            for j in range(3):
                ax=axes[ri,j];ims.append(ax.imshow(e['raw'][0]*0,origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=cap,interpolation='none'))
                plot_geometry(ax,e['contacts'],e['centers'],e['radius']);ax.set_title(['All native activity','Largest family','Other / unassigned'][j],fontsize=9)
            ax=axes[ri,3];norm=e['mass']/np.maximum(e['mass'].max(1,keepdims=True),1e-12)
            ax.imshow(norm,aspect='auto',origin='upper',extent=[0,250,15-.5,-.5],vmin=0,vmax=1,cmap='magma')
            ax.set(yticks=range(15),yticklabels=e['names'],xlabel='Time in window (ms)');ax.set_title('Full contact envelopes',fontsize=9);ax.tick_params(axis='y',labelsize=7)
            marks.append(ax.axvline(0,c='#37e4d0',lw=1.4));axes[ri,0].text(0,1.24,e['event']['condition'],transform=axes[ri,0].transAxes,fontweight='bold')
        title=fig.suptitle('',fontsize=13)
        fig.text(.5,.035,'Native maps: active E cells / 1 mm / 2 ms, shared 0–%.0f scale, unsmoothed. Contacts: own full-window peak = 1; no lineage filtering.'%cap,ha='center',fontsize=8)
        frames=[]
        for k in range(0,125,2):
            for ri,e in enumerate(pair):
                for j,ma in enumerate([e['raw'][k],e['parts'][0,k],e['parts'][1:,k].sum(0)]):ims[ri*3+j].set_data(ma)
                marks[ri].set_xdata([e['time'][k]]*2)
            title.set_text(('TA' if lab==1 else 'TB')+'-labelled examples | time in each event window: %.0f ms'%pair[0]['time'][k])
            fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        frames[0].save(F/('full_partition_'+('TA' if lab==1 else 'TB')+'.gif'),save_all=True,append_images=frames[1:],duration=90,loop=0)
        plt.close(fig)
    write(OUT/'example_selection.json',[dict(**e['event'],selection='first chronological event of this label in first sorted G3 unit') for e in ex])
    verification=[]
    for p in sorted(F.iterdir()):
        if p.suffix in ['.png','.gif']:
            with Image.open(p) as im:
                for k in range(getattr(im,'n_frames',1)):im.seek(k);im.load()
                verification.append(dict(file=p.name,frames=getattr(im,'n_frames',1),size=list(im.size)))
        elif p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF-')
    write(OUT/'verification.json',dict(status='PASS',native_mass_partition_exact=True,
        scores_reproduced=True,centroids_reproduced=True,decoded_images=verification,human_visual_review='PENDING'))
    (F/'README.md').write_text('''### native_activity_summary.png / .pdf
七个候选各四次确认运行，每点是一个运行的事件中位数，短横线为四点中位数。原生活动量为每 2 ms 至少放电一次的 E 细胞数跨帧求和，不是未限制的 spike 总数。核外峰值密度份额为 r外/(r外+r核)，其中 r 按 E 神经元数归一化，0.5 为同密度；核区在峰时无发放也保留为 1，不丢掉这些事件。灰带仅为时间 pilot TRAIN 患者 TA/TB 包络区间重叠 IQR 的包络；图中核区按 1 mm bin 中心近似，另存纯核内/纯核外 bin 敏感性。
**关注点**：最大活动家族占比和 core 峰值间隔不能证明独立源或因果传播；核外活动不能直接视为异常。

### full_activity_partition_examples.png / .pdf
前两候选各取第一次 TA 标签、第一次 TB 标签事件，展示全局峰时的完整发放、最大活动家族和其余发放，右列是三个空间区域按 E 神经元数归一化的活动。保留原 SEEG 位置，所有原生场共享计数色尺，不进行空间平滑。
**关注点**：这里的最大家族只用于诊断分解，正式训练用完整接触点包络；核边界 bin 无法从现有文件精确拆到逐神经元成员。

### full_partition_TA.gif
最佳特征候选与参考位置 A 各一个 TA 标签事件的完整 250 ms 观察窗，四列为全部原生发放、最大家族、其他/未分配发放、完整接触点包络。原生场每帧来自一个真实 2 ms bin，每隔 4 ms 播放一帧；接触点颜色各自按完整窗口峰值归一化，归一化不随帧改变。
**关注点**：观看异地活动是否连续招募；两个标签不是患者传播恢复的验收。

### full_partition_TB.gif
与 TA 动画相同的选例及显示规则，展示两个候选的 TB 标签事件。两图均未重跑模拟、未改变原损失，也未打开时间 pilot 的患者留后波形。
**关注点**：保留独立亮斑，不用选家族图替代完整场；接触点包络与原生计数有不同单位和空间采样范围。
''')
    print(json.dumps(dict(status='COMPLETE',units=len(units),events=len(events),out=str(OUT))))

if __name__=='__main__':main()

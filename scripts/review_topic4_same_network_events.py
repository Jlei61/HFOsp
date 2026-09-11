#!/usr/bin/env python3
"""All-event, same-realization diagnostic; never changes the frozen fit or labels."""
from pathlib import Path
import csv, gc, json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.qualify_topic4_observation_repair import OUT as INPUT, OLD, read, write, sha
from scripts.report_topic4_observation_repair import table
from scripts.audit_topic4_xy_raw_propagation_video import PATIENT, META, VIDEO, comparison
OUT=ROOT/'results/topic4_sef_hfo/multievent_same_network_propagation_review'
SELECTED=['support_rank__vth_low','old_joint__tau_d_GABA_ms_high']
GROUPS={'upper_SCL':['SCL6','SCL7','SCL8','SCL9'],
        'right_ICL':['ICL1','ICL2'],'middle_ICL':['ICL5','ICL6','ICL7','ICL8'],
        'left_ICL':['ICL9','ICL10','ICL11']}

def group_times(row,names):
    result={}
    for key,group in GROUPS.items():
        values=row[[names.index(n) for n in group]];valid=values[np.isfinite(values)]
        result[key+'_n']=len(valid)
        result[key+'_ms']=float(np.median(valid)) if len(valid) else None
    u,r,m,l=[result[k+'_ms'] for k in GROUPS]
    # Posthoc coarse route descriptions, NOT new acceptance labels or a loss.
    result['middle_before_both_ICL_ends']=None if any(x is None for x in (r,m,l)) else m+2<min(r,l)
    result['right_before_upper_and_left']=None if any(x is None for x in (r,u,l)) else r+2<min(u,l)
    result['upper_and_left_before_right']=None if any(x is None for x in (r,u,l)) else max(u,l)+2<r
    return result

def load_worker(w):
    meta=read(w['worker_path']);p=Path(meta['arrays']['path'])
    if sha(p)!=w['arrays_sha256']:raise RuntimeError('saved simulation changed')
    with np.load(p) as z:return {k:z[k] for k in ['contact_names','contact_xy_mm','sheet_activity_counts','sheet_activity_frame_ms','contact_envelope','contact_envelope_dt_ms','topology_seed']}

def audit(workers,ev,patient,names):
    po=[list(patient['contact_order'].astype(str)).index(n) for n in names]
    exemplars=np.array([np.where(patient[k+'_participant'],patient[k+'_fig1a_centroid_ms'],np.nan)[po] for k in ['TA','TB']])
    lab,st,_=ev.classify(exemplars)
    reference={'exemplar_labels':dict(zip(['TA','TB'],lab.tolist())), 'exemplar_support_states':st.tolist(),
        'mapping_scope':'Only these two displayed examples; cluster labels are not certification of their complete routes.',
        'contacts':names,'centroid_ms':exemplars.tolist(),
        'groups':{k:group_times(t,names) for k,t in zip(['TA','TB'],exemplars)}}
    def summarize_routes(records):
        summary={'n_events':len(records),'no_SCL':sum(r['upper_SCL_n']==0 for r in records)}
        for key in ['middle_before_both_ICL_ends','right_before_upper_and_left','upper_and_left_before_right']:
            values=[r[key] for r in records if r[key] is not None]
            summary[key]={'n_matching':int(sum(values)),'n_estimable':len(values),'fraction':float(np.mean(values)) if values else None}
        return summary
    reference['patient_fit_route_distributions']={str(m):summarize_routes([group_times(t,names) for t in ev.fit[ev.fit_labels==m]]) for m in range(ev.k)}
    rows=[];units=[]
    for w in workers:
        obs=w['observation'];t=table(w);ll,ss,_=ev.classify(t)
        units.append({'candidate_id':w['candidate_id'],'seed':w['seed'],'n_primary':len(t),
            'n_detected':len(obs['windows_ms']), 'n_assigned_by_mode':[int(sum(ll==k)) for k in range(ev.k)],
            'n_supported_by_mode':[int(sum((ll==k)&(ss==1))) for k in range(ev.k)],
            'both_labels_within_network':bool(all(np.any(ll==k) for k in range(ev.k))),
            'both_supported_within_network':bool(all(np.any((ll==k)&(ss==1)) for k in range(ev.k)))})
        for index,trow,label,state in zip(obs['primary_event_indices'],t,ll,ss):
            rec={'candidate_id':w['candidate_id'],'seed':w['seed'],'detected_event_number':index+1,
                'window_start_ms':obs['windows_ms'][index][0], 'mode':int(label),'support_state':int(state),
                **group_times(trow,names)}
            for key,ref in zip(['TA','TB'],exemplars):rec.update({key+'_'+k:v for k,v in comparison(trow,ref).items()})
            rows.append(rec)
    write(OUT/'within_network_audit.json',{'networks':units,'patient_reference':reference,
        'route_groups':GROUPS,'route_tolerance_ms':2.,'route_status':'posthoc descriptive, conditional on available contacts; not calibrated patient equivalence',
        'pooled_presence_does_not_require_same_network':True})
    with (OUT/'all_primary_event_paths.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    route_summary={cid:{str(m):summarize_routes([r for r in rows if r['candidate_id']==cid and r['mode']==m]) for m in range(ev.k)} for cid in SELECTED}
    write(OUT/'patient_and_model_route_distributions.json',{'patient_FIT':reference['patient_fit_route_distributions'],'model_four_seeds':route_summary,
        'scope':'Posthoc group-centroid diagnostic, same frozen contact identities. Conditional denominators explicit; no significance or calibrated acceptance threshold; not a continuous anatomical wave measurement.'})
    return reference,rows

def contact_map(ax,xy,t,title,names,labels=False):
    valid=np.isfinite(t);rel=t-np.nanmin(t)
    ax.scatter(*xy[~valid].T,s=45,facecolors='none',edgecolors='.65',linewidth=.8)
    p=ax.scatter(*xy[valid].T,c=rel[valid],s=50,cmap='viridis',vmin=0,vmax=100,edgecolors='.25',linewidth=.4)
    ax.set(xlim=(-21,23),ylim=(-13,24),aspect='equal',title=title);ax.set_xticks([]);ax.set_yticks([])
    ax.title.set_fontsize(8)
    if labels:
        for (x,y),n in zip(xy,names):ax.text(x,y+1,n,fontsize=5,ha='center')
    return p

def montage(w,z,patient,ev,folder):
    names=z['contact_names'].astype(str).tolist();pn=patient['contact_order'].astype(str).tolist()
    order=[pn.index(n) for n in names];xy=patient['points_mm'][order]
    obs=w['observation'];t=np.asarray(obs['centroid_ms'],float);ll,ss,_=ev.classify(t)
    n=len(t)+2;fig,axes=plt.subplots(int(np.ceil(n/5)),5,figsize=(13,2.3*np.ceil(n/5)))
    for ax in axes.ravel():ax.axis('off')
    for a,key in enumerate(['TA','TB']):
        pt=np.where(patient[key+'_participant'],patient[key+'_fig1a_centroid_ms'],np.nan)[order]
        contact_map(axes.ravel()[a],xy,pt,'Patient '+key,names,True)
    for i,row in enumerate(t):
        status='primary' if i in obs['primary_event_indices'] else 'excluded from fit'
        contact_map(axes.ravel()[i+2],xy,row,f"#{i+1}  {obs['windows_ms'][i][0]/1000:.2f}s | M{ll[i]} | {status}",names)
    fig.subplots_adjust(top=.90,bottom=.10,hspace=.32,wspace=.08)
    fig.suptitle(w['candidate_id']+f" | one fixed network, seed {w['seed']}\nAll detected events, chronological. Color = centroid lag 0–100 ms; hollow = absent. M labels do not certify a route.",fontsize=12)
    sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,100),cmap='viridis')
    fig.colorbar(sm,cax=fig.add_axes([.3,.045,.4,.009]),orientation='horizontal',label='Relative centroid (ms); values above 100 clipped only in color')
    fig.savefig(folder/'all_event_contact_maps.png',dpi=130);fig.savefig(folder/'all_event_contact_maps.pdf');plt.close(fig)

def strobograms(w,z,folder):
    t=np.asarray(w['observation']['centroid_ms'],float);movie=z['sheet_activity_counts'];dt=float(z['sheet_activity_frame_ms'])
    offsets=np.array([-16,0,16,32,64]);vmax=float(movie.max())
    for page,start in enumerate(range(0,len(t),6)):
        count=min(6,len(t)-start);fig,axes=plt.subplots(count,5,figsize=(10,count*1.8),squeeze=False)
        for row,i in enumerate(range(start,start+count)):
            t0=np.nanmin(t[i]);indices=np.clip(np.round((t0+offsets)/dt-.5).astype(int),0,len(movie)-1)
            for col,frame in enumerate(indices):
                ax=axes[row,col];ax.imshow(movie[frame],origin='lower',cmap='magma',vmin=0,vmax=vmax,interpolation='nearest',extent=(0,20,0,20))
                ax.scatter(*z['contact_xy_mm'].T,s=5,facecolors='none',edgecolors='white',lw=.3)
                ax.set(xticks=[],yticks=[],title=f'#{i+1} | {offsets[col]:+d} ms');ax.title.set_fontsize(8)
        fig.suptitle(w['candidate_id']+' | seed 2511\nNative sheet: fixed scale, all detected events in chronological order',fontsize=11)
        fig.subplots_adjust(top=1-.7/(count*1.8),bottom=.02,hspace=.25,wspace=.02)
        fig.savefig(folder/f'native_chronological_page_{page+1:02d}.png',dpi=120);plt.close(fig)

def movie(w,z,patient,ev,candidate,folder):
    names=z['contact_names'].astype(str).tolist();pn=patient['contact_order'].astype(str).tolist()
    order=[pn.index(n) for n in names];pxy=patient['points_mm'][order]
    obs=w['observation'];table_all=np.asarray(obs['centroid_ms'],float);ll,ss,_=ev.classify(table_all)
    field=z['sheet_activity_counts'];env=z['contact_envelope'];dt=float(z['sheet_activity_frame_ms'])
    assert dt==float(z['contact_envelope_dt_ms'])==2.
    centers=(np.arange(len(field))+.5)*dt;vmax=max(1,float(field.max()));emax=max(1e-12,float(env.max()))
    fig,axs=plt.subplots(2,3,figsize=(13.5,7.7),dpi=80)
    fig.subplots_adjust(left=.045,right=.975,bottom=.085,top=.83,hspace=.53,wspace=.36)
    raw,model,heat,ta,tb,timeline=axs.ravel()
    raw.set(title='Native sheet: unsmoothed active E counts / 2 ms',xlabel='Model x (mm)',ylabel='Model y (mm)',xlim=(0,20),ylim=(0,20))
    im=raw.imshow(field[0],origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=vmax,interpolation='nearest',animated=True)
    for center in candidate['node_field']['centers_mm']:
        raw.add_patch(Circle(center,candidate['geometry']['distance_cutoff_mm'],fill=False,edgecolor='cyan',lw=1))
    raw_contacts=raw.scatter(*z['contact_xy_mm'].T,s=12,facecolors='none',edgecolors='white',lw=.5,animated=True)
    sc=model.scatter(*pxy.T,c=np.zeros(15),s=72,cmap='magma',norm=PowerNorm(.5,0,emax),edgecolors='.5',animated=True)
    model.set(title='Model contacts at patient positions\nIdentity projection; not a tissue warp',xlim=(-21,23),ylim=(-13,24),aspect='equal')
    for (x,y),name in zip(pxy,names):model.text(x,y+1,name,fontsize=6,ha='center')
    heat.set(title='Model density proxy + participating centroids',xlabel='Time from first centroid (ms)',yticks=range(15),yticklabels=names)
    heat.tick_params(axis='y',labelsize=6)
    him=heat.imshow(env[:,:125],aspect='auto',origin='lower',cmap='magma',norm=PowerNorm(.5,0,emax),extent=(-125,125,-.5,14.5))
    marks=heat.scatter(np.zeros(15),np.arange(15),s=12,c='cyan',marker='x')
    cursor=heat.axvline(0,c='white',ls='--',lw=.8,animated=True)
    patient_dynamic=[];meta=read(META)
    for key,ax in [('TA',ta),('TB',tb)]:
        part=patient[key+'_participant'];points=patient['points_mm']
        ax.scatter(*points[~part].T,s=45,facecolors='none',edgecolors='.65')
        pp=ax.scatter(*points[part].T,c=np.zeros(sum(part)),s=72,cmap='GnBu',norm=PowerNorm(.5,0,1),edgecolors='.5',animated=True)
        ax.set(title='Patient '+key+': same Fig. 2C event\nActual contacts; fixed window q99',xlim=(-21,23),ylim=(-13,24),aspect='equal')
        for (x,y),name in zip(points,pn):ax.text(x,y+1,name,fontsize=6,ha='center')
        patient_dynamic.append((key,pp,part,meta['movie_contract']['normalization_scales_robust_z'][key]))
    rate=field.sum((1,2));timeline.plot(centers/1000,rate,lw=.6,c='.2')
    for i,(start,stop) in enumerate(obs['windows_ms']):
        timeline.axvspan(start/1000,stop/1000,color='#5ab4ac' if i in obs['primary_event_indices'] else '#d8a35a',alpha=.2)
    tcursor=timeline.axvline(0,c='red',lw=1,animated=True)
    timeline.set(xlim=(0,centers[-1]/1000),title='Complete original trajectory (including gaps)\nGreen: fit events; amber: excluded windows',xlabel='Original simulation time (s)',ylabel='Active E neurons / 2 ms')
    title=fig.text(.5,.98,'',ha='center',va='top',fontsize=11,animated=True)
    stamp=raw.text(.02,.98,'',transform=raw.transAxes,va='top',color='white',fontsize=9,animated=True)
    fig.text(.5,.012,'Same model, same cores, same noise law throughout. Gaps skipped and timed explicitly. Model density is not patient HFO amplitude.',ha='center',fontsize=8)
    for ax in axs.ravel():ax.title.set_fontsize(9);ax.tick_params(labelsize=7)
    dynamic=[im,*raw.patches,raw_contacts,sc,cursor,tcursor,title,stamp]+[x[1] for x in patient_dynamic]
    frames=[];duration=[];records=[];selected_frames=[]
    for i,row in enumerate(table_all):
        start,stop=obs['windows_ms'][i];ix=np.flatnonzero((centers>=start)&(centers<stop))[::2]
        t0=float(np.nanmin(row));him.set_data(env[:,int(start/dt):int(stop/dt)]);him.set_extent((start-t0,stop-t0,-.5,14.5));heat.set_xlim(start-t0,stop-t0)
        marks.set_offsets(np.column_stack([row-t0,np.arange(15)]))
        fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox)
        first_frame=len(frames);valid=i in obs['primary_event_indices'];state={1:'supported',0:'uncertain',-1:'OOD'}[int(ss[i])]
        gap=None if i==0 else start-obs['windows_ms'][i-1][1]
        timing='first detected window' if gap is None else ('overlap '+f'{-gap:.0f} ms' if gap<0 else 'skipped gap '+f'{gap:.0f} ms')
        for j,f in enumerate(ix):
            relative=centers[f]-t0;fig.canvas.restore_region(background)
            im.set_data(field[f]);sc.set_array(env[:,f]);cursor.set_xdata([relative]*2);tcursor.set_xdata([centers[f]/1000]*2)
            title.set_text(f"{w['candidate_id']} | fixed seed {w['seed']} | event {i+1}/{len(table_all)}\nM{ll[i]} / {state} / {'primary fit event' if valid else 'excluded from fit'} | {timing}")
            stamp.set_text(f't = {centers[f]/1000:.3f} s\ncentroid lag {relative:+.0f} ms')
            for key,pp,part,scale in patient_dynamic:
                pt=patient[key+'_envelope_time_from_first_centroid_ms'];take=abs(pt-relative)<=1.5
                if not take.any():raise ValueError('reference movie outside stored patient signal')
                pp.set_array(np.maximum(patient[key+'_envelope_robust_z'][:,take].mean(1),0)[part]/scale)
            for artist in dynamic:
                (artist.axes if artist.axes is not None else fig).draw_artist(artist)
            rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            frame=Image.fromarray(rgb).quantize(colors=128,method=Image.Quantize.FASTOCTREE)
            frames.append(frame);duration.append(400 if j==0 else 40)
            if j==int(np.argmin(abs(centers[ix]-t0-16))):
                frame.convert('RGB').save(folder/f'event_{i+1:02d}_frame_plus16.png');selected_frames.append(len(frames)-1)
        records.append({'detected_event_number':i+1,'window_ms':[start,stop],'primary':valid,'mode':int(ll[i]),'support_state':int(ss[i]),'first_gif_frame':first_frame,'n_frames':len(ix),'relative_ms':(centers[ix]-t0).tolist()})
        print('rendered',w['candidate_id'],i+1,'/',len(table_all),flush=True)
    plt.close(fig);path=folder/'all_chronological_events.gif'
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=duration,loop=0,optimize=False,disposal=2)
    preview_end=records[min(6,len(records))-1]['first_gif_frame']+records[min(6,len(records))-1]['n_frames']
    previews=[f.copy() for f in frames[:preview_end]]
    previews[0].save(folder/'first_six_chronological_events.gif',save_all=True,append_images=previews[1:],duration=duration[:preview_end],loop=0,optimize=False,disposal=2)
    del previews
    expected=len(frames);del frames;gc.collect()
    with Image.open(path) as gif:
        assert gif.n_frames==expected
        for k in range(gif.n_frames):gif.seek(k);gif.load();assert gif.info['duration']==duration[k]
    result={'path':str(path),'sha256':sha(path),'frames':expected,'all_frames_decoded':True,'events':records,
        'biological_step_ms':4,'frame_duration_ms':40,'first_frame_duration_ms':400,
        'normalization':'model scales fixed over complete simulation; patient fixed exemplar full-window q99, gamma .5; native linear full-simulation max',
        'selection':'Two previously reviewed candidates; lowest seed 2511; ALL detected windows chronological, including excluded windows; no likeness selection.',
        'continuous_simulation':True,'continuous_playback':False,'gaps_explicit':True,'no_event_specific_model_changes':True,
        'patient_times':'Same relative centroid milliseconds, unpaired repeated examples, no time warp. Extends beyond published -8..50 ms using same cache.',
        'patient_measure':'HFO envelope versus model firing-density proxy; amplitudes not equated','agent_visual_review':False,'human_acceptance':False}
    write(folder.parent/'movie_metadata.json',result)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    workers=read(INPUT/'reobserved_workers.json')['workers'];ev=pickle.load(open(INPUT/'evaluator.pkl','rb'))
    assert sha(INPUT/'evaluator.pkl')==read(INPUT/'qualification.json')['evaluator_sha256']
    assert sha(VIDEO)==read(META)['sha256']
    with np.load(PATIENT) as p:patient={k:p[k] for k in p.files}
    names=read(INPUT/'observation_contract.json')['contact_names']
    reference,rows=audit(workers,ev,patient,names)
    design={r['candidate_id']:r for r in read(OLD/'design.json')['candidates']};movies=[]
    for cid in SELECTED:
        w=min((w for w in workers if w['candidate_id']==cid),key=lambda w:w['seed']);z=load_worker(w)
        assert z['contact_names'].astype(str).tolist()==names
        folder=OUT/cid/'figures';folder.mkdir(parents=True,exist_ok=True)
        montage(w,z,patient,ev,folder)
        strobograms(w,z,folder)
        movies.append(movie(w,z,patient,ev,design[cid],folder))
        (folder/'README.md').write_text('### all_chronological_events.gif\n同一固定网络 seed 2511 的全部已检测事件，按原始时间顺序播放，未根据与 TA/TB 的相似度挑选。保留原生网格、按通道身份投影到患者平面的读出、完整事件窗和连续 12 秒轨迹概览，跳过的间隔及不纳入拟合的窗口明确标注；患者参考重复播放同一对 Fig. 2C 示例，来自相同原始缓存但展示窗口更长。\n**关注点**：不同事件是否恢复不同参与和传播路径，而非仅产生两个聚类标签；网格与患者读出的幅度和物理含义不同。\n\n### all_event_contact_maps.png / .pdf\n全部检测事件的逐通道相对质心时序，患者 TA/TB 示例列于最前，所有图共用 0–100 ms 色标；空心通道保持不参与。模式标签保留冻结评价器的编号，不等于路径复现已通过。\n**关注点**：SCL 缺失、ICL 中部先于两端，以及患者示例的逐点顺序；需结合动画检查。\n\n### event_*_frame_plus16.png\n每个事件最早有效质心后约 16 ms 的固定规则抽帧，未按相似度挑选。\n**关注点**：抽帧仅作导航，传播顺序应结合完整动画与逐点时间判断。\n')
    write(OUT/'summary.json',{'status':'MULTIEVENT_RENDER_COMPLETE_PENDING_VISUAL_REVIEW','movies':movies,'new_simulations':0,'frozen_evaluator_modified':False,'final_model_accepted':False})

if __name__=='__main__':main()

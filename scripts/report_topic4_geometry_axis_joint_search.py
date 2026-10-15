"""Post-score diagnostics and the two agreed visual acceptance artifacts.

This module may read frozen labels. The proposal process never imports it.
Patient mean is the fixed 32+32 morphology packet, not the full FIT distribution.
"""
from pathlib import Path
import argparse, csv, fcntl, json, sys, time, warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_topic4_geometry_axis_joint_search as run
from scripts import analyze_topic4_three_observable_bo as patient_source
from scripts import topic4_patient_mean_spectra as patient_mean
from scripts.report_topic4_label_free_dense import describe, metrics
from scripts.paper_figures import plot_topic4_recovery_review as figure_source
rt,OUT=run.rt,run.OUT
patient_mean.OUT=OUT/'analysis/patient_display'
DISPLAY=figure_source.display
MODES=[('ALL',None),('TA',1),('TB',0)]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
    'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})


def gather(rows):
    ev,names,_=patient_source.patient();path=OUT/'analysis/diagnostic_observables.json'
    if path.exists():data=rt.read(path)
    else:
        data=dict(patient={},runs={})
        for old in [run.SOURCE,run.XY]:
            oldpath=old/'analysis/diagnostic_observables.json'
            if oldpath.exists():
                q=rt.read(oldpath);data['patient']=q['patient'];data['runs'].update(q['runs'])
    if not data['patient']:
        data['patient']={mode:describe(ev.fit if k is None else ev.fit[ev.fit_labels==k],names) for mode,k in MODES}
    for sc in rows:
        cid=sc['candidate']
        if cid in data['runs']:continue
        src=Path(sc['source']);r,t,cn=run.dense.load_times(src);assert cn==names
        with np.load(src.with_suffix('.npz')) as z:
            ids=np.array([int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
            labels=z['event_mode'][ids]
        data['runs'][cid]={mode:describe(t if k is None else t[labels==k],names) for mode,k in MODES}
    rt.write(path,data)
    return ev,names,data


def save_observables(rows,data,names):
    result=[]
    for sc in rows:
        for mode,_ in MODES:
            q=data['runs'][sc['candidate']][mode];p=data['patient'][mode];v=metrics(q,p,names)
            errors=[abs(a['order_probability']-p['pairs'][k]['order_probability']) for k,a in q['pairs'].items()
                if a['shaft']=='SCL' and a['order_probability'] is not None and p['pairs'][k]['order_probability'] is not None]
            result.append(dict(candidate=sc['candidate'],mode=mode,N=q['N'],J=sc['J'],J_joint=sc['J_joint'],
                rank_error=v[0],within_rod_order_error=v[1],rod_lag_ms=v[2],participation_error=v[3],
                both_rods=v[4],SCL_order_error=float(np.mean(errors)) if errors else None,reused=sc.get('reused',False)))
    with (OUT/'analysis/observables.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    return result


def mean_peak_centroids(values,grid):
    out=[]
    for v in values:
        if not np.isfinite(v).any() or np.nanmax(v)<=0:out.append(np.nan);continue
        peak=int(np.nanargmax(v));lo=hi=peak;threshold=.7*v[peak]
        while lo>0 and v[lo-1]>=threshold:lo-=1
        while hi+1<len(v) and v[hi+1]>=threshold:hi+=1
        out.append(float(np.average(grid[lo:hi+1],weights=v[lo:hi+1])))
    return np.asarray(out)


def model_mean(r,a,ids,k,grid):
    selected=np.asarray([int(i) for i in ids if a['event_mode'][i]==k],int)
    values=[];centroids=[];windows=[];dt=float(a['contact_envelope_dt_ms'])
    for i in selected:
        lo,hi=r['events'][i]['window_ms'];t=a['centroid_ms'][i];zero=float(np.nanmin(t));part=np.isfinite(t)
        first,last=round(lo/dt),round(hi/dt)
        mass=a['contact_envelope'][first:last].T.astype(float).copy()
        mass/=np.maximum(mass.max(1,keepdims=True),1e-20)
        native=np.arange(first,last)*dt-zero
        assert len(native)>1 and np.all(np.diff(native)>0), 'nonmonotone envelope time grid'
        val=np.full((len(t),len(grid)),np.nan)
        for ch in np.flatnonzero(part):val[ch]=np.interp(grid,native,mass[ch],left=np.nan,right=np.nan)
        values.append(val);centroids.append(t-zero);windows.append([native[0],native[-1]])
    if not values:
        return dict(N=0,ids=selected,mean=np.full((15,len(grid)),np.nan),counts=np.zeros((15,len(grid)),int),
            centroid_quantiles=np.full((2,15),np.nan),mean_peak=np.full(15,np.nan),common_ms=None)
    values=np.stack(values);windows=np.asarray(windows);counts=np.isfinite(values).sum(0)
    common=(grid>=windows[:,0].max())&(grid<=windows[:,1].min())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(values,axis=0);q=np.nanquantile(np.array(centroids),[.05,.95],axis=0)
    mean[:,~common]=np.nan
    return dict(N=len(selected),ids=selected,mean=mean,counts=counts,centroid_quantiles=q,
        mean_peak=mean_peak_centroids(mean,grid),common_ms=grid[common][[0,-1]].tolist() if common.any() else None)


def contact_reindex(source_names,target_names):
    assert len(set(source_names))==len(source_names) and set(source_names)==set(target_names)
    return np.array([source_names.index(n) for n in target_names],int)


def patient_packet_quantiles(names):
    manifest=rt.read(patient_mean.PACKET/'packet_manifest.json');by={1:[],0:[]}
    for r in manifest['readable_events']:
        with np.load(r['arrays_path']) as z:
            ix=contact_reindex(z['contact_names'].astype(str).tolist(),names)
            t=z['fig1a_centroid_ms'].astype(float)[ix];part=z['participation_mask'].astype(bool)[ix]
            t[~part]=np.nan;t-=np.nanmin(t);by[int(r['mode'])].append(t)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return {k:np.nanquantile(np.asarray(v),[.05,.95],axis=0) for k,v in by.items()}


def contact_image(ax,mass,grid,centroids,*,xlim,title,cmap='coolwarm',quantiles=None):
    cm=plt.get_cmap(cmap).copy();cm.set_bad('#777777')
    edges=np.r_[grid[0]-1,(grid[:-1]+grid[1:])/2,grid[-1]+1]
    ax.pcolormesh(edges,np.arange(16)-.5,mass,cmap=cm,vmin=0,vmax=1,shading='flat',rasterized=True)
    if quantiles is not None:
        for j,(l,h) in enumerate(quantiles.T):
            if np.isfinite([l,h]).all():ax.plot([l,h],[j,j],color='#f5f5f5',lw=.8,alpha=.85,zorder=4)
    DISPLAY.centroid_lines(ax,centroids,color='#202020')
    DISPLAY.contact_axis(ax);ax.set(xlim=xlim,xlabel='相对时间 (ms)',title=title,facecolor='#aaa')
    ax.tick_params(labelsize=6);ax.set_xticks([0,100])


def comparison(sc,c,r,a,ids,physics,folder,names,*,display_name=None,field_vmax=None):
    from scripts.render_topic4_shape_output_gifs import field_tile
    with np.load(patient_mean.OUT/'mean_envelopes.npz') as z:
        patient={key:z[key] for key in z.files}
    pix=contact_reindex(patient['contact_names'].astype(str).tolist(),names)
    for mode in ['TA','TB']:
        for kind in ['envelope','centroid_ms','counts']:
            patient[f'{mode}_{kind}']=patient[f'{mode}_{kind}'][pix]
    order=DISPLAY.contact_indices(names);grid=patient['time_ms'];xlim=(-125,200)
    patq=patient_packet_quantiles(names);dt=float(a['contact_envelope_dt_ms']);fd=float(a['sheet_activity_frame_ms'])
    assert fd==2, 'native frame contract changed'
    vmax=max(float(np.quantile(a['sheet_activity_counts'][round(1500/fd):],.999)),1) if field_vmax is None else float(field_vmax)
    assert np.isfinite(vmax) and vmax>0
    # Scientific image plate: matched event-average panels plus the SAME event's field.
    fig=plt.figure(figsize=(183/25.4,125/25.4))
    gs=fig.add_gridspec(2,6,width_ratios=[1.3,1.3,1.3,1,1,1],wspace=.42,hspace=.65)
    saved={};metadata=[]
    for row,(mode,k) in enumerate([('TA',1),('TB',0)]):
        m=model_mean(r,a,ids,k,grid)
        patient_values=patient[f'{mode}_envelope'].copy();pc=patient[f'{mode}_centroid_ms'].copy()
        shared=np.isfinite(patient_values).any(0)&np.isfinite(m['mean']).any(0)
        # No events removed. Both means show only their shared complete window.
        patient_values[:,~shared]=np.nan;model_values=m['mean'].copy();model_values[:,~shared]=np.nan
        pc=mean_peak_centroids(patient_values,grid);mc=mean_peak_centroids(model_values,grid)
        ax=fig.add_subplot(gs[row,0]);contact_image(ax,patient_values[order],grid,pc[order],xlim=xlim,
            title=f'患者 {mode} · n=32\n平均HFO包络',quantiles=patq[k][:,order])
        ax=fig.add_subplot(gs[row,1]);contact_image(ax,model_values[order],grid,mc[order],xlim=xlim,
            title=f'模型 {mode} · n={m["N"]}\n平均发放包络',quantiles=m['centroid_quantiles'][:,order])
        ax.tick_params(labelleft=False)
        eligible=m['ids'];ax=fig.add_subplot(gs[row,2])
        if not len(eligible):
            DISPLAY.contact_axis(ax);ax.text(.5,.5,'未观察到该类',ha='center',transform=ax.transAxes);ax.set_xlim(xlim)
            metadata.append(dict(mode=mode,N=0,status='NO_OBSERVED_EVENTS'));continue
        i=int(eligible[0]);lo,hi=r['events'][i]['window_ms'];ct=a['centroid_ms'][i];zero=float(np.nanmin(ct));part=np.isfinite(ct)
        first,last=round(lo/dt),round(hi/dt);mass=a['contact_envelope'][first:last].T.astype(float).copy()
        mass/=np.maximum(mass.max(1,keepdims=True),1e-20);mass[~part]=np.nan
        native=np.arange(first,last)*dt-zero
        cm=plt.get_cmap('magma').copy();cm.set_bad('#777777')
        ax.imshow(mass[order],extent=[lo-zero,hi-zero,14.5,-.5],aspect='auto',vmin=0,vmax=1,cmap=cm,interpolation='nearest')
        DISPLAY.centroid_lines(ax,ct[order]-zero,color='#00cfff');DISPLAY.contact_axis(ax)
        ax.set(xlim=xlim,xlabel='相对时间 (ms)',title=f'模型单事件 {i}\n与右侧原生场同步',facecolor='#aaa');ax.tick_params(labelleft=False,labelsize=6);ax.set_xticks([0,100])
        frames=[]
        for col,offset in enumerate([0,30,60],3):
            ax=fig.add_subplot(gs[row,col]);now=zero+offset
            if lo<=now<hi:
                fi=round(now/fd);ax.imshow(field_tile(a['sheet_activity_counts'][fi],c,a,physics,vmax,size=220),extent=[0,20,0,20])
                frames.append(dict(relative_ms=offset,absolute_ms=now,frame=fi))
            else:ax.text(.5,.5,'窗外',ha='center',transform=ax.transAxes)
            ax.set(title=f'+{offset} ms',xlabel='x (mm)',xticks=[0,20],yticks=[0,20]);ax.tick_params(labelsize=6)
            if col==3:ax.set_ylabel('y (mm)')
        saved.update({f'{mode}_model_mean':model_values,f'{mode}_patient_mean':patient_values,
            f'{mode}_model_counts':m['counts'],f'{mode}_model_centroid_quantiles':m['centroid_quantiles'],
            f'{mode}_patient_centroid_quantiles':patq[k],f'{mode}_model_event_ids':m['ids']})
        metadata.append(dict(mode=mode,N_model=m['N'],patient_packet_N=32,model_mean_event_ids=m['ids'].tolist(),
            displayed_shared_complete_domain_ms=grid[shared][[0,-1]].tolist() if shared.any() else None,
            representative_event=i,frames=frames))
    title=display_name if display_name is not None else f'{sc["candidate"]} · 同一网络 / 噪声'
    fig.suptitle(f'{title} · 患者与模型传播对照',fontsize=8,x=.51,y=.985)
    fig.text(.055,.036,'细线：单事件质心5–95%；点：平均包络主峰质心。患者为冻结形态小包，非频谱；模型为发放包络。',fontsize=6)
    fig.text(.055,.014,f'原生场共享色标：0–{vmax:.0f} 发放 / 2 ms / 网格；白圈为core，青色为固定触点。',fontsize=6)
    fig.subplots_adjust(left=.075,right=.992,bottom=.13,top=.86)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:outside.append(t.get_text())
    if outside:raise RuntimeError(f'text outside figure: {outside}')
    dest=folder/'patient_model_mean_event_field.png'
    fig.savefig(dest,dpi=400);fig.savefig(dest.with_suffix('.svg'));plt.close(fig)
    np.savez_compressed(folder/'comparison_source_data.npz',time_ms=grid,contact_names=np.array(names),**saved)
    rt.write(folder/'comparison_manifest.json',dict(candidate=sc['candidate'],display_name=display_name,events=metadata,
        patient=patient_mean.identity(),source=sc['source'],source_sha256=sc['source_sha256'],field_vmax=vmax,
        template='conditional mean over participating events; no individual contact shifts; no time warping; no post-mean renormalization',
        spread='5–95 percentiles of individual event centroids; not confidence intervals or centroid of the mean envelope',
        selection='first eligible event per frozen label; no patient-distance selection',
        labels='post-score comparison only',layout_mm=[183,125],svg_editable_text=True,human_review='PENDING',
        contact_alignment='explicit name reindex from patient packet and cached mean to model contact order'))
    return dest


def media(sc,names):
    from scripts.analyze_topic4_core_connectivity_search import load_unit
    from scripts.media_topic4_three_observable_bo import render_mean_template_gif
    folder=OUT/'key_results'/sc['candidate'];folder.mkdir(exist_ok=True)
    manifest=folder/'manifest.json'
    if manifest.exists():
        saved=rt.read(manifest)
        if saved.get('report_sha256')==rt.sha(__file__) and all(Path(saved[k]).exists() for k in ['gif','snapshot']):return saved
    src=Path(sc['source']);r,a,ids=load_unit(src,1500.);c=rt.read(OUT/'candidates'/f"{sc['candidate']}.json");c['topology']=run.TOPOLOGY
    # load_unit is inherited; retain the exact same primary + burn-in + end-window contract.
    ids=np.asarray([int(i) for i in ids if r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
    physics=rt.read(src.parents[1]/'applied_physics.json')
    render_mean_template_gif(c,run.NOISE,r,a,ids,physics,folder,display_xlim=(-125,200),
        patient_mean_draw=patient_mean.draw,patient_mean_identity=patient_mean.identity())
    gif=folder/'patient_mean_native_multievent.gif'
    if not gif.exists():return dict(candidate=sc['candidate'],status='NO_ELIGIBLE_MEDIA_EVENTS')
    snapshot=comparison(sc,c,r,a,ids,physics,folder,names)
    from PIL import Image
    with Image.open(gif) as im:
        n=im.n_frames
        for i in range(n):im.seek(i);im.load()
    with Image.open(snapshot) as im:im.load()
    import xml.etree.ElementTree as ET
    ET.parse(snapshot.with_suffix('.svg'))
    saved=dict(candidate=sc['candidate'],gif=str(gif),snapshot=str(snapshot),J=sc['J'],J_joint=sc['J_joint'],
        N=len(ids),mode_counts={mode:int((a['event_mode'][ids]==k).sum()) for mode,k in [('TA',1),('TB',0)]},
        source=sc['source'],source_sha256=sc['source_sha256'],report_sha256=rt.sha(__file__),gif_frames_decoded=n,
        human_visual_review='PENDING',agent_visual_review='PENDING',scientific_acceptance='NOT_ESTABLISHED',time=time.time())
    rt.write(manifest,saved)
    (folder/'README.md').write_text('### patient_mean_native_multievent.gif\n同一网络与噪声下，每类最早三个合格事件按真实时间顺序展示。患者为固定32+32事件的平均HFO包络，模型显示全部E细胞原生场与同步发放包络；均非伪造频谱。\n**关注点**：多次传播能否对应两类患者模式，是否仍有平行热点；三加三展示不估计自然比例。\n\n### patient_model_mean_event_field.png\n两行分别对照TA/TB的患者平均包络、模型所有该类事件的平均包络、预定单事件及同一事件的三个场时刻。前两列显示共同完整时间域，未参与和域外为灰色；细线是单事件质心5–95%范围，点是平均包络主峰质心，两者不等同。\n**关注点**：SCL内部顺序、杆间时间和参与是否共同改善；患者小包用于形态展示，完整FIT用于数值评价。\n')
    return saved


def main(final_media=False,check_media=False):
    for path,sha in rt.read(OUT/'analysis/figure_contract.json')['files'].items():
        assert rt.sha(path)==sha, 'frozen patient display changed'
    p=rt.read(OUT/'plan.json');rows=run.scores();selected=run.nominees(rows,p['objective']['normalization'])
    ev,names,data=gather(rows);table=save_observables(rows,data,names)
    byid={r['candidate']:r for r in rows};lookup={(r['candidate'],r['mode']):r for r in table}
    numerical=dict(complete=len(rows),new_complete=sum(not r.get('reused') for r in rows),
        nominees=selected,labels_used_for_nomination=False,patient=data['patient'],
        nominee_observables=[lookup[n['candidate'],mode] for n in selected for mode,_ in MODES],
        controls=[lookup[cid,mode] for cid in [p['reference_id'],p['joint_reference_id'],p['historical_visual_control']] for mode,_ in MODES],
        scientific_status='NOT_ACCEPTED_BY_SCORE; VISUAL_REVIEW_REQUIRED',time=time.time())
    rt.write(OUT/'analysis/current.json',numerical)
    # Save paired factorial outputs, not a misleading projection of mixed-parameter BO points.
    factorial=[dict(**cell,observables={mode:lookup.get((cell['candidate'],mode)) for mode,_ in MODES}) for cell in p['interaction_cells']]
    rt.write(OUT/'analysis/interaction_responses.json',factorial)
    if final_media or check_media:
        choices=selected if final_media else [dict(candidate=p['reference_id'],roles=['historical_pipeline_validation'])]
        outputs=[]
        for item in choices:outputs.append(dict(roles=item['roles'],**media(byid[item['candidate']],names)))
        rt.write(OUT/'key_results'/('current.json' if final_media else 'pipeline_validation.json'),dict(
            stage='FINAL_NOMINATIONS' if final_media else 'HISTORICAL_REFERENCE_PIPELINE_CHECK',
            results=outputs,labels_used_for_selection=False,scientific_review='PENDING',time=time.time()))
    text='# 当前科学审阅\n\n'
    text+=f'已完成{numerical["new_complete"]}条新轨迹，另有184条相容历史轨迹。双目标提名不读取TA/TB标签；原始仿真与诊断类别始终分开。\n\n'
    text+='|提名|原目标|同事件联合目标|TA / TB事件数|TB SCL先后误差|TB杆间时差(ms)|\n|---|---:|---:|---:|---:|---:|\n'
    for item in selected:
        cid=item['candidate'];a=lookup[cid,'TA'];b=lookup[cid,'TB']
        text+=f'|{cid}|{item["J"]:.4f}|{item["J_joint"]:.4f}|{a["N"]} / {b["N"]}|{b["SCL_order_error"]}|{b["rod_lag_ms"]}|\n'
    text+='\n仅同一网络/噪声的一条轨迹，事件不是独立网络重复。低分、标签计数、GIF解码均不等于双模式恢复；最终审阅仍看两种图。当前方向改变会重建EE伙伴及距离时延，固定拓扑seed不等于相同EE图。\n'
    text+='\n[两类验收图](../key_results/)；[逐条件观测](observables.csv)；[交互条件配对数据](interaction_responses.json)。\n'
    (OUT/'analysis/scientific_review.md').write_text(text)
    print(json.dumps(dict(new_complete=numerical['new_complete'],nominees=selected,final_media=final_media,check_media=check_media),ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final-media',action='store_true');ap.add_argument('--check-media',action='store_true')
    a=ap.parse_args()
    with (OUT/'analysis/helper.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main(a.final_media,a.check_media)

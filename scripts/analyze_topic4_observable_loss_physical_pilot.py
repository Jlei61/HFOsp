#!/usr/bin/env python3
"""Automatic training plots and independent native-field confirmation audit."""
from pathlib import Path
import argparse,csv,json,pickle,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from PIL import Image
from scripts import run_topic4_observable_loss_physical_pilot as run
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from scripts.paper_figures.audit_topic4_native_activity_shortcuts import regions
from src.topic4_envelope_joint_pilot import model_events
from src.topic4_interictal_repaired_evaluation import rank_features

OUT=run.OUT;F=OUT/'figures';COLORS=['#9b4265','#347fa5','#bb8542']
PLABELS=['Threshold-field gain','E to E strength','E to I strength','I to E strength','GABA decay (ms)']
BLABELS=['Participation','Centroid order / lag','Local envelope shape','Relative 50% times','All-contact envelope']
METRICS=['local_width_ms','contact_overlap','simultaneous_secondary_fraction','disconnected_mass_020',
         'largest_family_fraction','absolute_core_lag_ms','outside_peak_density_share']
MLABELS=['Local width (ms)','Contact overlap','Concurrent other-family mass','Disconnected activity mass',
         'Largest family mass','Core peak lag (ms)','Outside peak density share']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(v):
    a=np.array([x for x in v if x is not None],float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,mean=None,median=None,sd=None,q05=None,q95=None)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),sd=float(a.std()),
                q05=float(np.quantile(a,.05)),q95=float(np.quantile(a,.95)))


def disconnected_fraction(raw,den,fraction):
    density=np.divide(raw,den[None],out=np.zeros_like(raw,dtype=float),where=den[None]>0)
    threshold=float(density.max()*fraction);total=other=0.
    for frame,d in zip(raw,density):
        mask=(d>threshold)&(frame>0);labels,n=label(mask,structure=np.ones((3,3)))
        if not n:continue
        masses=np.bincount(labels.ravel(),weights=(frame*mask).ravel())[1:]
        total+=masses.sum();other+=masses.sum()-masses.max()
    return float(other/total) if total else None


def native_unit(path,candidate,old):
    model=model_events(path,engine.repaired_observation);labels=old.km.predict(rank_features(model['centroids'])) if len(model['centroids']) else []
    centers=np.array(candidate['node_field']['centers_mm']);radius=model['worker']['xy_geometry_audit']['distance_cutoff_mm']
    with np.load(model['worker']['arrays']['path']) as z:
        raw=z['sheet_activity_counts'].astype(float);family=z['directed_lineage_labels'];env=z['contact_envelope'].astype(float)
        dt=float(z['sheet_activity_frame_ms']);assert dt==float(z['contact_envelope_dt_ms'])==2.
        xy=z['contact_xy_mm'];_,reg,den,_,_=regions(z,centers,radius)
    rows=[];clips=[];grids=[]
    for k,(event,lab) in enumerate(zip(model['info'],labels)):
        idx=event['event_id'];meta=model['metadata']['events'][idx];lo,hi=np.rint(np.array(meta['window_ms'])/dt).astype(int)
        a=raw[lo:hi];fam=family[lo:hi];total=float(a.sum())
        if total<=0:raise ValueError('primary event has no native mass')
        fl=np.bincount(np.maximum(fam,0).ravel(),weights=a.ravel());largest=float(fl[1:].max()/total) if len(fl)>1 else 0.
        secondary=0.
        for frame,ids in zip(a,fam):
            v=np.bincount(np.maximum(ids,0).ravel(),weights=frame.ravel())[1:]
            if len(v):secondary+=v.sum()-v.max()
        regional=np.array([a[:,reg==j].sum(1) for j in range(3)])
        valid_cores=bool(np.all(regional[:2].sum(1)>0))
        lag=float((regional[1].argmax()-regional[0].argmax())*dt) if valid_cores else None
        peak=int(a.sum((1,2)).argmax());density=[]
        for mask in [reg<2,reg==2]:density.append(float(a[peak,mask].sum()/den[mask].sum()))
        outside=float(density[1]/sum(density)) if sum(density)>0 else None
        mass=np.maximum(env[:,lo:hi]-np.array(meta['local_baseline'])[:,None],0.)
        row=dict(event_id=idx,window_start_ms=float(lo*dt),mode=int(lab),
            local_width_ms=event['contact_width_ms'],contact_overlap=event['overlap'],
            centroid_span_ms=event['centroid_span_ms'],simultaneous_secondary_fraction=float(secondary/total),
            largest_family_fraction=largest,unknown_or_collision_fraction=float(fl[0]/total),
            core2_minus_core1_peak_ms=lag,absolute_core_lag_ms=abs(lag) if lag is not None else None,
            outside_peak_density_share=outside,native_active_cell_frames=total,
            **{f'disconnected_mass_{int(q*100):03d}':disconnected_fraction(a,den,q) for q in [.1,.2,.3]})
        rows.append(row)
        payload=dict(event=row,mass=mass,raw=a,names=model['names'],xy=xy,centers=centers,radius=radius,
                     mask=np.isfinite(model['centroids'][k]))
        grids.append(payload)
        if sum(x['event']['mode']==lab for x in clips)<2:clips.append(payload)
    return rows,clips,grids


def save(fig,name,entries,text):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=155,bbox_inches='tight')
    plt.close(fig);entries.append((name+'.png / .pdf',text))


def training_figures(d,records,entries):
    rows=[]
    for r in records:
        ai=run.anchor_index(r['candidate'],d['anchors']);parameters=run.prior.vector(r['candidate'])
        for uid,u in r['units'].items():
            row=dict(candidate_id=r['candidate_id'],anchor=ai+1,unit=uid,N=u['new']['N'],new_loss=u['new']['loss'],
                     old_loss=u['old']['loss'],**dict(zip(run.prior.PARAMETERS,parameters.tolist())))
            row.update({k:u['new'].get('blocks',{}).get(k,{}).get('D_off') for k in run.KEYS});rows.append(row)
    csvwrite(OUT/'training_parameter_scores.csv',rows)
    fig,axs=plt.subplots(5,3,figsize=(11,12),squeeze=False)
    for pi,p in enumerate(run.prior.PARAMETERS):
        for j,metric in enumerate(['new_loss','local_shape','joint_envelope']):
            ax=axs[pi,j]
            for r in rows:
                if r[metric] is not None:ax.scatter(r[p],r[metric],c=COLORS[r['anchor']-1],marker='o' if '6101' in r['unit'] else '^',s=17,alpha=.8)
            ax.set(xlabel=PLABELS[pi],ylabel=['New total','Local shape distance','Joint envelope distance'][j]);ax.grid(alpha=.15)
    fig.suptitle('Physical parameter changes and observable errors',fontsize=14)
    fig.tight_layout(rect=(0,.05,1,.96));fig.text(.5,.015,'Colors: fixed placements 1 / 2 / 3. Circle / triangle: topology 6101 / 6102. Joint parameter proposals; not isolated causal effects.',ha='center',fontsize=9)
    save(fig,'training_parameter_response',entries,'物理参数与新总分、局部形态和完整包络误差的关系，每点一个网络。三个颜色是三个固定双核布局，圆/三角是两张拓扑；每个提案同时改变五个参数。**关注点**：这是联合扫参关联，不能当成单参数因果曲线。')
    fig,ax=plt.subplots(figsize=(10,max(4,len(records)*.28)))
    mat=np.array([[np.mean([u['new']['blocks'][k]['D_off'] for u in r['units'].values()]) if r['ranking_eligible'] else np.nan for k in run.KEYS] for r in records])
    im=ax.imshow(np.ma.masked_invalid(mat),aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(1.5,float(np.nanmax(mat))))
    ax.set(xticks=range(5),xticklabels=BLABELS,yticks=range(len(records)),yticklabels=[str(i+1) for i in range(len(records))],ylabel='Condition index (CSV mapping)')
    fig.colorbar(im,ax=ax,label='Separate distance; lower is better');ax.set_title('Five errors stay visible beside the total score')
    save(fig,'training_five_errors',entries,'同一训练候选池的五项误差，每个条件先对两个网络等权平均。空白行表示至少一个网络不满足评分条件，条件序号对应 training_condition_index.csv。**关注点**：总分下降可能掩盖个别项恶化，五项都要检查。')
    csvwrite(OUT/'training_condition_index.csv',[dict(index=i+1,candidate_id=r['candidate_id'],new_loss=r['loss'],old_loss=r['old_loss']) for i,r in enumerate(records)])


def contact_grid_pdf(grids,path):
    with PdfPages(path) as pdf:
        for start in range(0,len(grids),8):
            subset=grids[start:start+8];fig,axs=plt.subplots(4,2,figsize=(10,10),squeeze=False)
            for ax,g in zip(axs.ravel(),subset):
                names=g['names'];order=sorted(range(len(names)),key=lambda i:(not names[i].startswith('SCL'),-int(''.join(filter(str.isdigit,names[i])))))
                mass=g['mass'];relative=mass/max(float(mass.max()),1e-12)
                ax.imshow(relative[order],aspect='auto',extent=(0,250,14.5,-.5),cmap='magma',vmin=0,vmax=1)
                ax.set(title=f'Event {g["event"]["event_id"]} | {"TA label" if g["event"]["mode"] else "TB label"}',xlabel='Window time (ms)',yticks=range(15),yticklabels=[names[i] for i in order]);ax.tick_params(labelsize=6)
            for ax in axs.ravel()[len(subset):]:ax.axis('off')
            fig.suptitle('Every primary event: all contacts, event-wide normalization');fig.tight_layout(rect=(0,0,1,.96));pdf.savefig(fig);plt.close(fig)


def render_movie(clips,patient_entries,cid,mode,index,entries,manifest):
    selected=[(uid,g) for uid,g in clips if g['event']['mode']==mode]
    if not selected:
        manifest.append(dict(candidate_id=cid,mode=mode,status='NO_OBSERVED_EVENT_NOT_PROOF_OF_ABSENCE'));return
    positive=np.concatenate([g['raw'][g['raw']>0] for _,g in selected]);cap=max(1.,float(np.quantile(positive,.99)))
    frames=[];fig,axs=plt.subplots(1,4,figsize=(12,3.5),gridspec_kw={'width_ratios':[1,1,1,1.2]})
    fig.subplots_adjust(top=.78,bottom=.25,wspace=.35)
    for ci,(uid,g) in enumerate(selected):
        pe=patient_entries[ci%len(patient_entries)]
        with np.load(pe['arrays_path']) as z:
            names=z['contact_names'].astype(str).tolist();ix=[names.index(n) for n in g['names']];win=z['packed_window_mask'].astype(bool)
            pm=z['positive_envelope_mass'][ix][:,win].astype(float);pt=z['time_ms'][win].astype(float);pt=pt-pt[0]
        mm=g['mass'];mt=(np.arange(mm.shape[1])+.5)*2
        pp=pm/max(float(pm.max()),1e-12);mp=mm/max(float(mm.max()),1e-12)
        for ax in axs:ax.clear()
        scat=[]
        for j in range(3):
            ax=axs[j];ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)' if j==0 else '',title=['Patient TRAIN','Model: all contacts','Model: native activity'][j]);ax.set_facecolor('black')
            for shaft in ['SCL','ICL']:
                ix=[i for i,n in enumerate(g['names']) if n.startswith(shaft)];ix=sorted(ix,key=lambda i:g['xy'][i,0]);ax.plot(g['xy'][ix,0],g['xy'][ix,1],color='gray',lw=.8)
            if j<2:scat.append(ax.scatter(*g['xy'].T,c=np.zeros(15),s=50,cmap='magma',vmin=0,vmax=1,edgecolors='white',lw=.4))
            else:
                native=ax.imshow(g['raw'][0],origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=cap,interpolation='none')
                ax.scatter(*g['xy'].T,s=8,facecolors='none',edgecolors='white',lw=.5)
                for center,color in zip(g['centers'],['#ef9d38','#3ebac0']):ax.add_patch(Circle(center,g['radius'],fill=False,ec=color,lw=1))
        order=sorted(range(15),key=lambda i:(not g['names'][i].startswith('SCL'),-int(''.join(filter(str.isdigit,g['names'][i])))))
        axs[3].imshow(mp[order],aspect='auto',extent=(0,250,14.5,-.5),cmap='magma',vmin=0,vmax=1)
        axs[3].set(title='Model contact time course',xlabel='Window time (ms)',yticks=range(15),yticklabels=[g['names'][i] for i in order]);axs[3].tick_params(labelsize=6)
        cursor=axs[3].axvline(0,color='cyan',lw=1)
        for text in list(fig.texts):
            if text is not fig._suptitle:text.remove()
        fig.text(.5,.055,f'Contacts: envelope / one event-wide maximum; all contacts retained. Native: 0–{cap:.0f} active E cells / 1 mm bin / 2 ms.\nFrozen 250 ms windows; no temporal warping or added smoothing. Labels organize examples, not route recovery.',ha='center',fontsize=8)
        for t in range(0,250,10):
            scat[0].set_array(np.array([np.interp(t,pt,a,left=0,right=0) for a in pp]));scat[1].set_array(np.array([np.interp(t,mt,a,left=0,right=0) for a in mp]))
            native.set_data(g['raw'][min(int(t/2),len(g['raw'])-1)]);cursor.set_xdata([t,t])
            fig.suptitle(f'Condition {index} | {"TA-labelled" if mode else "TB-labelled"} | example {ci+1}/{len(selected)} | {uid} | {t} ms',fontsize=10)
            fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
        manifest.append(dict(candidate_id=cid,mode=mode,unit=uid,event_id=g['event']['event_id'],patient_event_id=pe['raw_global_event_index'],status='DISPLAYED'))
    name=f'condition{index}_{"TA" if mode else "TB"}_multiple_events.gif'
    frames[0].save(F/name,save_all=True,append_images=frames[1:],duration=110,loop=0,optimize=False);plt.close(fig)
    entries.append((name,'每个确认网络依时间顺序取该标签最早两个事件，连续展示患者接触点、模型全部接触点、原生场与接触时间图。布局保持固定 SEEG 坐标，两个 core 仅以轮廓提示，不隐藏 core 外或非参与接触点。**关注点**：多事件而非一幅模板；患者与模型不是逐事件配对；标签和动画不证明因果机制。'))


def main(final=False):
    d,obj=run.frozen();F.mkdir(exist_ok=True);entries=[]
    records=[r for phase in ['initial','C','D'] if (OUT/f'{phase}_scores.json').exists() for r in run.base.read(OUT/f'{phase}_scores.json')['candidates']]
    training_figures(d,records,entries)
    if final:
        nomination=run.base.read(OUT/'nomination.json');confirm=run.base.read(OUT/'confirmation_scores.json')['candidates']
        old=pickle.load((run.prior.OLD/'training_objective_v2_1.pkl').open('rb'));prior_design=run.base.read(run.prior.OUT/'design.json')
        allrows=[];summaries=[];movies=[];unitrows=[];support=[];time_blocks=[];identities=[]
        for ci,r in enumerate(confirm,1):
            cid=r['candidate_id'];clips=[]
            for uid,u in r['units'].items():
                train_record=next(x for x in records if x['candidate_id']==cid)
                topo=uid.split('_dyn_')[0];train_unit=next(v for key,v in train_record['units'].items() if key.startswith(topo+'_dyn_'))
                tw=run.base.read(train_unit['worker_path']);cw=run.base.read(u['worker_path'])
                same=tw['static_array_identity']==cw['static_array_identity']
                identities.append(dict(candidate_id=cid,unit=uid,static_identity_equal=same,
                    training_dynamics_seed=tw['dynamics_seed'],confirmation_dynamics_seed=cw['dynamics_seed']))
                if not same:raise RuntimeError('same-condition same-topology static identity changed')
                rows,selected,grids=native_unit(u['worker_path'],r['candidate'],old)
                allrows.extend([dict(candidate_id=cid,condition_index=ci,unit=uid,**e) for e in rows]);clips.extend([(uid,g) for g in selected])
                unitrow=dict(candidate_id=cid,index=ci,unit=uid,N=len(rows),new_loss=u['new']['loss'],old_loss=u['old']['loss'])
                for met in METRICS:unitrow[met]=stats([e[met] for e in rows])['median']
                unitrows.append(unitrow)
                for lab in [-1,0,1]:
                    ee=[e for e in rows if lab==-1 or e['mode']==lab]
                    for met in METRICS:summaries.append(dict(candidate_id=cid,unit=uid,mode=lab,metric=met,**stats([e[met] for e in ee])))
                    leads=[e['core2_minus_core1_peak_ms'] for e in ee if e['core2_minus_core1_peak_ms'] is not None]
                    support.append(dict(candidate_id=cid,unit=uid,mode=lab,n_events=len(ee),n_both_cores_active=len(leads),
                        core1_leads_fraction=float(np.mean(np.array(leads)>0)) if leads else None,
                        ties_fraction=float(np.mean(np.array(leads)==0)) if leads else None))
                for start in [0,6000,12000,18000]:
                    ee=[e for e in rows if start<=e['window_start_ms']<start+6000]
                    time_blocks.append(dict(candidate_id=cid,unit=uid,start_ms=start,n_events=len(ee),
                        mode0_count=sum(e['mode']==0 for e in ee),mode1_count=sum(e['mode']==1 for e in ee)))
                contact_grid_pdf(grids,F/f'condition{ci}_{uid}_all_events.pdf')
            for lab in [1,0]:render_movie(clips,[p for p in prior_design['patient_training'] if p['mode']==lab],cid,lab,ci,entries,movies)
        csvwrite(OUT/'confirmation_events.csv',allrows);csvwrite(OUT/'confirmation_unit_medians.csv',unitrows);csvwrite(OUT/'confirmation_distributions.csv',summaries)
        csvwrite(OUT/'confirmation_mode_support.csv',support);csvwrite(OUT/'confirmation_six_second_blocks.csv',time_blocks)
        run.write(OUT/'confirmation_static_identity.json',identities)
        run.write(OUT/'movie_event_manifest.json',movies)
        fig,axs=plt.subplots(2,4,figsize=(13,7));axs=axs.ravel()
        for ax,metric,title in zip(axs,METRICS,MLABELS):
            for ci,r in enumerate(confirm,1):
                vals=[u for u in unitrows if u['candidate_id']==r['candidate_id']]
                for j,u in enumerate(vals):
                    if u[metric] is not None:ax.scatter(ci+(j-.5)*.12,u[metric],c=COLORS[run.anchor_index(r['candidate'],d['anchors'])],marker=['o','^'][j],s=28)
            ax.set(title=title,xlabel='Condition index',xticks=range(1,len(confirm)+1));ax.grid(alpha=.15)
        axs[-1].axis('off');axs[-1].text(0,1,'Native diagnostics were not optimized.\n\nFragmentation: 8-neighbor components;\n10/20/30% density threshold sensitivity.\n\nFamily IDs are segmentation dependent.\nOutside activity alone is not failure.\n\nNeither a lower score nor two labels\nis automatic mechanism acceptance.',va='top',fontsize=9)
        fig.suptitle('Fresh-noise confirmation: does observable improvement remove native shortcuts?');fig.tight_layout(rect=(0,0,1,.95))
        save(fig,'independent_native_confirmation',entries,'同一新噪声下，比较旧新损失各自提名及起始参考的时间形态、并行活动族、空间分散性与双核错相。每点是一个网络内事件中位数，圆/三角是两张固定拓扑。**关注点**：这些原生指标没有进入优化；族标签和像素连通性都不是完整因果证据。')
        # Paired changes retain each topology rather than pooling all events.
        refid=next(cid for cid,roles in nomination['roles'].items() if 'starting_reference' in roles)
        changes=[]
        for u in unitrows:
            b=next(v for v in unitrows if v['candidate_id']==refid and v['unit']==u['unit'])
            changes.append(dict(candidate_id=u['candidate_id'],unit=u['unit'],roles=nomination['roles'][u['candidate_id']],
                **{met:None if u[met] is None or b[met] is None else u[met]-b[met] for met in METRICS}))
        run.write(OUT/'paired_native_changes.json',changes)
        text='# 五项损失物理 pilot：确认结果待科学审阅\n\n'
        text+='已完成预定物理运行和独立原生场分析。新损失降低不自动等于消除机制捷径；需要结合下面逐网络结果、全事件接触热图与多事件 GIF。新旧提名来自同一候选池，但候选池由新目标引导探索，不能据此证明同预算优化算法更优。\n\n'
        text+='|条件|提名来源|新噪声新损失|局部宽度，两网络 ms|并行次要活动族质量，两网络|\n|---|---|---|---|---|\n'
        for ci,r in enumerate(confirm,1):
            uu=[u for u in unitrows if u['candidate_id']==r['candidate_id']]
            fmt=lambda met:' / '.join('不可估计' if u[met] is None else f'{u[met]:.3f}' for u in uu)
            text+=f'|{ci}: {r["candidate_id"]}|{", ".join(nomination["roles"][r["candidate_id"]])}|{r["loss"]}|{fmt("local_width_ms")}|{fmt("simultaneous_secondary_fraction")}|\n'
        text+='\n全部 primary 事件均进入 CSV 和逐事件 PDF。每网络每标签最早两个事件进入 GIF；未观察到某标签报告支持不足，不自动判定机制缺失。患者图使用 TRAIN 波形，无原生全场患者真值。\n\n原生指标是 1 mm / 2 ms 的活动细胞质量和分段族诊断。没有额外平滑，没有把 core 外活动全部视为错误。所有图待人工目视审阅，本轮不自动扩搜、不冻结工作点、不进入 Fig. 5。\n'
        (OUT/'scientific_review.md').write_text(text)
        entries.append(('condition*_all_events.pdf','逐确认网络保存全部 primary 事件的全部接触点时间热图；每页八个事件，统一真实 250 ms 时间窗。每事件共用一个幅度归一化，不把非参与接触点置零。**关注点**：应检查整个事件集合，避免只看 GIF 的少数示例。'))
    (F/'README.md').write_text('# 物理 pilot 图件\n\n'+'\n\n'.join('### '+name+'\n\n'+desc for name,desc in entries)+'\n')
    qa=[]
    for p in F.glob('*.png'):
        with Image.open(p) as im:im.verify()
        qa.append(dict(file=p.name,verified=True))
    for p in F.glob('*.gif'):
        with Image.open(p) as im:
            for i in range(im.n_frames):im.seek(i);im.load()
            qa.append(dict(file=p.name,frames=im.n_frames,verified=True))
    run.write(OUT/'figure_validation.json',dict(stage='final' if final else 'training',files=qa,human_visual_acceptance='PENDING'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--training-only',action='store_true');a=p.parse_args();main(a.final)

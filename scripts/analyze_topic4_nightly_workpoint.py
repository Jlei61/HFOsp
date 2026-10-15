#!/usr/bin/env python3
"""Nightly per-network observables, all-event envelopes and native validation."""
from pathlib import Path
import argparse,csv,json,pickle,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scripts.run_topic4_nightly_workpoint import OUT,SOURCE,write
from scripts import run_topic4_xy_research as base
from scripts import analyze_topic4_observable_loss_physical_pilot as visuals
from scripts.analyze_topic4_local_width_mechanism_pilot import score
from src.topic4_contact_event_objective_v2 import patient_event
from scripts import run_topic4_multievent_distribution_v2_1 as engine
KEYS=['participation','centroid_structure','local_shape','recruitment','joint_envelope']
KLABELS=['Participation','Centroid order / lag','Local envelope shape','Recruitment timing','All-contact envelope']
METRICS=['local_width_ms','recruitment_span_ms','centroid_span_ms','n_contacts']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def csvwrite(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(p,'w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def stats(x,w=None):
    x=np.asarray(x,float)
    if not len(x):return dict(n=0,mean=None,median=None,variance=None,q05=None,q95=None)
    if w is None:
        return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=float(x.var()),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))
    w=np.asarray(w,float)/np.sum(w)
    order=np.argsort(x);v=x[order];cw=np.cumsum(w[order]);mu=float(w@x)
    return dict(n=len(x),mean=mu,median=float(np.interp(.5,cw,v)),variance=float(w@((x-mu)**2)),q05=float(np.interp(.05,cw,v)),q95=float(np.interp(.95,cw,v)))

def geometry_figure(batch):
    folder=OUT/'batches'/batch;f=folder/'figures';f.mkdir(exist_ok=True)
    d=base.read(OUT/'design.json');cc=base.read(folder/'candidate_manifest.json')['candidates'];xy=np.array(d['contact_xy_mm']);names=d['contact_names']
    cols=3;rows=(len(cc)+cols-1)//cols;fig,axs=plt.subplots(rows,cols,figsize=(11,3.8*rows),squeeze=False)
    for ax,c in zip(axs.ravel(),cc):
        for shaft,col in [('SCL','#eea147'),('ICL','#35a4b8')]:
            ix=[i for i,n in enumerate(names) if n.startswith(shaft)];ax.plot(xy[ix,0],xy[ix,1],'.-',color=col,lw=1,ms=5)
        for j,center in enumerate(c['node_field']['centers_mm'],1):
            ax.add_patch(Circle(center,c['geometry']['distance_cutoff_mm'],facecolor='#ce8665',edgecolor='#84543d',alpha=.35));ax.text(*center,str(j),ha='center',va='center')
        ax.scatter(*d['shaft_balanced_center_mm'],marker='+',c='black',s=55)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=c['candidate_id'].replace('central_A_',''))
    for ax in axs.ravel()[len(cc):]:ax.axis('off')
    fig.suptitle('Initial central geometry proposals' if batch == 'A' else 'Batch B: fixed central parent and broader geometry control',fontsize=13);fig.tight_layout(rect=(0,0,1,.96))
    for ext in ['png','pdf']:fig.savefig(f/f'geometry_proposals.{ext}',dpi=155,bbox_inches='tight')
    plt.close(fig)
    (f/'README.md').write_text('### geometry_proposals.png / .pdf\n\n各面板给出该条件实际使用的双核位置；同位置的重复面板对应不同动力学干预。十字是两根电极杆中心的等权平均，只用于提案优先级；圆为拓扑 6101 的实际双核选点半径，数字是 core 编号，不是 TA/TB 起点。\n\n**关注点**：保留真实 SEEG 布局，比较两杆共同覆盖区与历史边缘布局，不把中心位置作为硬性验收条件。\n')

def patient_rows(obj):
    m=base.read(ROOT/'results/topic4_sef_hfo/contact_event_objective_revision_v2/manifest.json');records=[]
    for e in m['patient_training']:
        r=patient_event(e['arrays_path'],obj.names);mask=r['participation']>0
        records.append(dict(mode=e['mode'],local_width_ms=r['local_width_ms'],recruitment_span_ms=float(np.ptp(r['recruitment'][mask,0])),centroid_span_ms=float(np.ptp(r['centroid'][mask])),n_contacts=int(mask.sum())))
    out=[]
    for mode in ['all',0,1]:
        ix=[i for i,r in enumerate(records) if mode=='all' or r['mode']==mode]
        for k in METRICS:out.append(dict(source='patient_TRAIN',candidate_id='',unit='',mode=mode,metric=k,**stats([records[i][k] for i in ix],obj.weights[ix])))
    return out,m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',default='A');ap.add_argument('--geometry-only',action='store_true');a=ap.parse_args()
    geometry_figure(a.batch)
    if a.geometry_only:return
    folder=OUT/'batches'/a.batch;f=folder/'figures';visentries=[];movie_manifest=[]
    obj=pickle.load(open(SOURCE/'objective.pkl','rb'))
    old=pickle.load(open(ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1/training_objective_v2_1.pkl','rb'))
    dist,patmanifest=patient_rows(obj)
    manifest=base.read(folder/'candidate_manifest.json');candidate_rows=[];native_rows=[];event_rows=[];param_rows=[];movie_data={}
    cc=manifest['candidates'];references=base.read(OUT/'reference_controls.json')['candidates']
    if a.batch != 'A':
        prior=base.read(OUT/'batches/A/scores.json')['candidates']
        parent_ids={c.get('parent_candidate_id') for c in cc}
        present={r['candidate_id'] for r in references}
        references += [r for r in prior if r['candidate_id'] in parent_ids-present]
    inputs=[(c,{f'topo_{t}_dyn_{n}':str(folder/'workers'/f'{engine._stem(c["candidate_id"],t,n)}.json') for t,n in manifest['pairs']},False) for c in cc]
    inputs += [(r['candidate'],{u:v['worker_path'] for u,v in r['units'].items()},True) for r in references]
    visuals.F=f;visuals.OUT=folder
    for c,paths,control in inputs:
        units={};clips=[];allgrids=[]
        for uid,p in paths.items():
            p=Path(p)
            if not p.exists():continue
            u,op=score(p,obj);units[uid]=u
            nr,cl,gr=visuals.native_unit(p,c,old);labels={r['event_id']:r['mode'] for r in nr}
            clips.extend((uid,g) for g in cl);allgrids.extend(gr)
            for r in nr:native_rows.append(dict(candidate_id=c['candidate_id'],unit=uid,**r))
            grids_by_id={g['event']['event_id']:g for g in gr}
            for e in u['events']:
                g=grids_by_id[e['event_id']];names=np.array(g['names']);mask=np.asarray(g['mask'],bool)
                event_rows.append(dict(candidate_id=c['candidate_id'],unit=uid,mode=labels[e['event_id']],n_SCL=int(np.sum(mask & np.char.startswith(names,'SCL'))),n_ICL=int(np.sum(mask & np.char.startswith(names,'ICL'))),**e))
            for mode in ['all',0,1]:
                ee=[e for e in u['events'] if mode=='all' or labels[e['event_id']]==mode]
                for k in METRICS:dist.append(dict(source='historical_control' if control else 'new_model',candidate_id=c['candidate_id'],unit=uid,mode=mode,metric=k,**stats([e[k] for e in ee])))
            pp=dict(candidate_id=c['candidate_id'],unit=uid,control=control,physical_status=u['physical_status'],N=u['observation']['N'],loss=u['observation']['loss'],primary_fraction=u['primary_fraction'],
                x1=c['node_field']['centers_mm'][0][0],y1=c['node_field']['centers_mm'][0][1],x2=c['node_field']['centers_mm'][1][0],y2=c['node_field']['centers_mm'][1][1],
                node_gain=c['node_mapping']['node_gain'],**c['dynamic_parameters'],angle_deg=c['mechanisms']['ellipse_angle_deg'],aspect_ratio=c['mechanisms']['ellipse_aspect_ratio'])
            pp.update({k:u['observation'].get('blocks',{}).get(k,{}).get('D_off') for k in KEYS});param_rows.append(pp)
            # Keep every primary event; never choose only best-looking events.
            visuals.contact_grid_pdf(gr,f/f'{c["candidate_id"]}_{uid}_all_events.pdf')
        good=len(units)==len(manifest['pairs']) and all(u['observation']['loss'] is not None for u in units.values())
        row=dict(candidate_id=c['candidate_id'],candidate=c,control=control,units=units,ranking_eligible=good,loss=float(np.mean([u['observation']['loss'] for u in units.values()])) if good else None)
        candidate_rows.append(row);movie_data[c['candidate_id']]=clips
    write(folder/'scores.json',dict(candidates=candidate_rows,formal_acceptance=False,model_labels_used_for_training=False))
    csvwrite(folder/'parameter_observables.csv',param_rows);csvwrite(folder/'event_observables.csv',event_rows);csvwrite(folder/'native_diagnostics.csv',native_rows);csvwrite(folder/'conditional_distributions.csv',dist)
    fig,axs=plt.subplots(1,5,figsize=(15,max(4,len(candidate_rows)*.35)),sharey=True)
    for j,k in enumerate(KEYS):
        for i,r in enumerate(candidate_rows):
            for off,(uid,u) in zip([-.09,.09],r['units'].items()):
                v=u['observation'].get('blocks',{}).get(k,{}).get('D_off')
                if v is not None:axs[j].scatter(v,i+off,c='#777777' if r['control'] else '#a55448',marker='o' if '6101' in uid else '^',s=25)
        axs[j].set_title(KLABELS[j]);axs[j].set_xlabel('Separate error (lower is better)');axs[j].grid(axis='x',alpha=.2)
    axs[0].set(yticks=range(len(candidate_rows)),yticklabels=[r['candidate_id'].replace('central_A_','') for r in candidate_rows]);axs[0].invert_yaxis();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(f/f'five_observables.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    # Fixed training score nominates views; it does not establish visual acceptance.
    new=sorted([r for r in candidate_rows if not r['control'] and r['ranking_eligible']],key=lambda r:r['loss'])
    show=new[:2] if new else [r for r in candidate_rows if not r['control'] and any(u['events'] for u in r['units'].values())][:2]
    for i,r in enumerate(show,1):
        for mode in [1,0]:visuals.render_movie(movie_data[r['candidate_id']],[e for e in patmanifest['patient_training'] if e['mode']==mode],r['candidate_id'],mode,i,visentries,movie_manifest)
    write(folder/'movie_event_manifest.json',movie_manifest)
    md='# 批次 '+a.batch+'：待科学审阅\n\n总分仅用于排列搜索候选，不能证明患者传播恢复。需分别看 TB/TA、完整包络与原生全场；模型标签只组织展示。本表保留每网络事件数与物理状态，未给出自动通过结论。\n\n|候选|网络|事件数|宽度中位数 ms|物理状态|\n|---|---|---:|---:|---|\n'
    for r in candidate_rows:
        for uid,u in r['units'].items():md+=f'|{r["candidate_id"]}|{uid}|{u["observation"]["N"]}|{u["distributions"]["local_width_ms"]["median"]}|{u["physical_status"]}|\n'
    md+='\n每网络全事件 PDF、按固定规则选取的多事件 GIF、参数与各项观测表、TA/TB 条件分布和原生场诊断均已生成。下一批需记录残差判断与提案依据；未通过不能自行进入参数响应或冻结模型。\n'
    (folder/'scientific_review.md').write_text(md)
    with (f/'README.md').open('a') as stream:
        stream.write('\n### five_observables.png / .pdf\n\n五项冻结观测误差分别展示；圆和三角表示不同拓扑，灰色为历史对照。**关注点**：空值是不可评分，不是零误差。\n\n### *_all_events.pdf\n\n每张网络的所有 primary 事件按原时间顺序展示，不按事件重排接触点，不做时间拉伸；每事件使用共同幅度分母。TA/TB 为旧分类器组织标签。**关注点**：标签不代表路径已恢复；特别检查 TB 的衔接与局部持续时间。\n')
        for name,desc in visentries:stream.write('\n### '+name+'\n\n'+desc+'\n')
    from scripts.analyze_topic4_nightly_contacts import run as contact_diagnostics
    contact_diagnostics(folder,obj,old)
    qa=[]
    for p in f.glob('*.png'):
        with Image.open(p) as im:im.load();qa.append(dict(file=p.name,size=list(im.size)))
    write(folder/'figure_validation.json',dict(png_decoded=qa,human_visual_review='PENDING'))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare frozen rev22 candidates; keep old results separate from repaired graphs."""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src/snn_engine'))
from src.topic4_axis_aligned_core import axis_aligned_core_field
from src.topic4_manual_dual_core import budget_matched_dual_core_h

ART=Path('/home/honglab/leijiaxin/HFOsp')
STAGE=ART/'results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability'
IDS=['dci_p000','dci_p089','dci_p030','dci_p066','dci_p075']
COLORS=['#777777','#198C82','#A63A79','#878787','#A0A0A0']
NAMES=['Reference','Geometry only','Full four-parameter','Angle locked','Aspect ratio locked']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest_path=STAGE/'response_fit/final_execution_candidate_manifest.json'
    frozen_path=STAGE/'response_fit/frozen_candidates.json'
    aggregate_path=STAGE/'frozen_stage_aggregate/frozen_stage_aggregate.json'
    structure_path=STAGE/'connectivity_design_audit/connectivity_design_audit.json'
    manifest=json.loads(manifest_path.read_text());frozen=json.loads(frozen_path.read_text())
    aggregate=json.loads(aggregate_path.read_text());structure=json.loads(structure_path.read_text())
    for key,p in [('candidate_manifest',manifest_path),('frozen_candidates',frozen_path)]:
        if sha(p)!=aggregate['input_hashes'][key]['sha256']:raise RuntimeError('aggregate source drift')
    table_path=Path(structure['table']['path'])
    if sha(table_path)!=structure['table']['sha256']:raise RuntimeError('structure table drift')
    if set(frozen['candidate_ids'])!=set(IDS):raise RuntimeError('frozen candidate set changed')
    with open(table_path) as f:table=list(csv.DictReader(f))
    by_id={x['candidate_id']:x for x in manifest['candidates']}
    sources=[manifest_path,frozen_path,aggregate_path,structure_path,table_path]
    report=[]
    for cid in IDS:
        c=by_id[cid];r=aggregate['phases']['confirmation']['candidates'][cid]
        geom=[x for x in table if x['candidate_id']==cid]
        if len(geom)!=4:raise RuntimeError('four fit topologies are required')
        angles=np.array([float(x['achieved_EE_angle_deg']) for x in geom])-180
        ars=np.array([float(x['achieved_EE_aspect_ratio']) for x in geom])
        report.append({'candidate_id':cid,'physical':c['physical'],'kernel_angle_deg':c['mechanisms']['ellipse_angle_deg'],
                       'achieved_angle_mean_deg':float(angles.mean()),'achieved_angle_range_deg':[float(angles.min()),float(angles.max())],
                       'achieved_AR_mean':float(ars.mean()),'conditional_estimable_networks':r['conditional_estimable_units'],
                       'confirmation_networks':r['expected_units'],'primary_pooled_estimable':r['primary_estimable'],
                       'components':r['pooled']['components'],'bootstrap_90':r['bootstrap_90'],
                       'returned_families':r['pooled']['n_events'],'training_target_only':True})
    # Position identity was verified when constructing the corrected graph.
    rebuild_path=ROOT/'results/topic4_sef_hfo/substrate_autapse_correction/graph_rebuild_audit.json'
    rebuild=json.loads(rebuild_path.read_text())
    if rebuild['status']!='FULL_SIZE_CORRECTED_GRAPH_VERIFIED':raise RuntimeError('corrected graph is not verified')
    with open(rebuild['new_cache']['path'],'rb') as f:payload=pickle.load(f)
    positions=np.asarray(payload['net']['pos'])[:int(payload['NE'])]
    del payload
    centers=np.array(by_id['dci_p030']['node_field']['centers_mm'])
    h,field=budget_matched_dual_core_h(positions,centers,target_count=1499)
    contact_path=ART/'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/contact_shaft_contract.json'
    contacts=np.array([x['sheet_xy_mm'] for x in json.loads(contact_path.read_text())['contacts']])
    sources += [rebuild_path,contact_path,Path(__file__)]
    out=ROOT/'results/topic4_sef_hfo/rev22_candidate_axis_review';figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    summary={'status':'FROZEN_CANDIDATES_REVIEWED_NOT_RESELECTED','candidates':report,
             'source_hashes':{str(p):sha(p) for p in sources},
             'scope':'Four fit topologies for achieved structure; twelve confirmation topologies against the original training target for response. No held-out patient validation or repaired-network dynamics claimed.',
             'common_core_centers_mm':centers.tolist(),'figure_ellipse_scale':'constant major diameter; shape and angle reflect weighted-edge second moments, not anatomical spatial footprint'}
    (out/'candidate_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(17,8.6));gs=fig.add_gridspec(2,1,height_ratios=[1.15,1],left=.06,right=.98,top=.86,bottom=.14,hspace=.63)
    top=gs[0].subgridspec(1,5,wspace=.30);bottom=gs[1].subgridspec(1,4,wspace=.40)
    fig.suptitle('rev22 candidates: achieved connection axes and response trade-offs',fontsize=19,weight='bold',y=.97)
    fig.text(.06,.915,'Structure: original graph, 4 fit topologies. Responses: 12 confirmation topologies, original training target.',fontsize=11)
    for i,r in enumerate(report):
        ax=fig.add_subplot(top[i]);color=COLORS[i]
        ax.scatter(contacts[:,0],contacts[:,1],s=13,facecolors='none',edgecolors='#BBBBBB')
        for c in centers:ax.add_patch(Circle(c,field['distance_cutoff_mm'],color='#B7A0C7',alpha=.65))
        ax.plot(centers[:,0],centers[:,1],ls=':',color='#7E668E',lw=1)
        angle=r['achieved_angle_mean_deg'];ar=r['achieved_AR_mean'];center=np.array([10.,12.])
        ax.add_patch(Ellipse(center,10,10/ar,angle=angle,facecolor=color,edgecolor=color,alpha=.16))
        v=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        kv=np.array([np.cos(np.deg2rad(r['kernel_angle_deg'])),np.sin(np.deg2rad(r['kernel_angle_deg']))])
        ax.plot(*np.array([center-5*v,center+5*v]).T,color=color,lw=2)
        ax.plot(*np.array([center-5*kv,center+5*kv]).T,color='black',ls='--',lw=1)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if i==0:ax.set_ylabel('y (mm)')
        ax.set_title(f"{r['candidate_id'].replace('dci_','')} · {NAMES[i]}\nMeasured {angle:.1f}°  |  AR {ar:.2f}",fontsize=10,color=color,pad=10)
    metrics=[('D_support','Participation-pattern distance'),('D_order','Order distance'),('D_lag','Lag distance (ms)'),('D_cover','Coverage distance')]
    for i,(metric,title) in enumerate(metrics):
        ax=fig.add_subplot(bottom[i])
        for x,r in enumerate(report):
            b=r['bootstrap_90'][metric];y=r['components'][metric]
            ax.vlines(x,b['q05'],b['q95'],color=COLORS[x],lw=1.5)
            ax.scatter(x,y,s=38,color=COLORS[x],zorder=3)
        ax.set_xticks(range(5),[x.replace('dci_','') for x in IDS],rotation=35)
        ax.set_title(title+' ↓',fontsize=11,loc='left')
        ax.set_xlim(-.5,4.5);ax.grid(axis='y',alpha=.13)
    fig.text(.06,.045,'Solid: achieved global weighted axis. Dashed: requested kernel axis. Ellipse size normalized. Error bars: topology-bootstrap 90% intervals.',fontsize=10)
    stem=figures/'rev22_candidate_axes_and_tradeoffs'
    for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    # Concrete interior proposals, tied to existing kernel parameters. These have
    # not been optimized or assigned any of the old candidates' performance.
    historical_centers=np.array([[4.19921432,9.12890135],[16.47920304,3.96551153]])
    sep=float(np.linalg.norm(historical_centers[1]-historical_centers[0]))
    proposals=[]
    specifications=[('historical_axis_budget_matched',historical_centers.mean(0),-22.80538396505847,'dci_p000')]
    specifications += [(f'centered_{cid}',[10.,10.],by_id[cid]['mechanisms']['ellipse_angle_deg'],cid) for cid in ('dci_p000','dci_p030','dci_p089')]
    fig,axes=plt.subplots(1,5,figsize=(17,4.8));fig.subplots_adjust(left=.045,right=.99,bottom=.20,top=.78,wspace=.26)
    fig.suptitle('Interior core alternatives: structural proposals, dynamics not yet tested',fontsize=17,weight='bold',y=.96)
    items=[('Edge reference',h,{'centers_mm':centers.tolist(),**field},None)]
    for name,mid,angle,cid in specifications:
        hh,aa=axis_aligned_core_field(positions,midpoint_mm=mid,separation_mm=sep,axis_deg=angle,target_count=1499,boundary_clearance_mm=1.5)
        proposals.append({'proposal_id':name,'connection_candidate':cid,'geometry':aa,'status':'STRUCTURE_VALID_DYNAMICS_UNTESTED'})
        items.append((name,hh,aa,angle))
    for ax,(name,hh,aa,angle) in zip(axes,items):
        cc=np.array(aa['centers_mm'])
        ax.scatter(positions[hh>.5,0],positions[hh>.5,1],s=2,color='#8757A0',alpha=.5,rasterized=True)
        ax.scatter(contacts[:,0],contacts[:,1],s=12,facecolors='none',edgecolors='#999999')
        ax.plot(cc[:,0],cc[:,1],color='#245C3F',ls='--')
        title={'historical_axis_budget_matched':'Historical placement\nsame 1499-cell budget','centered_dci_p000':'Centered · reference axis','centered_dci_p030':'Centered · p030 axis','centered_dci_p089':'Centered · p089 axis'}.get(name,name)
        ax.set_title(title,fontsize=10)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
    axes[0].set_ylabel('y (mm)')
    fig.text(.045,.065,'All panels retain 1499 core E cells. Four interior proposals use the historical separation and at least 1.5 mm clearance beyond each full core disk.',fontsize=10)
    stem=figures/'interior_core_geometry_proposals'
    for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    (out/'interior_geometry_proposals.json').write_text(json.dumps({'status':'PROPOSED_NOT_FINAL_FROZEN_SUBSTRATE','proposals':proposals,'budget':1499,'separation_mm':sep,'boundary_margin_role':'1.5 mm is a declared design constraint, not a measured absence of boundary effects','source_hashes':summary['source_hashes']},indent=2)+'\n')
    (figures/'README.md').write_text('### rev22_candidate_axes_and_tradeoffs.png / .pdf / .svg\n上排同时显示五个已冻结候选的设定核方向与真实加权图的全局主轴；椭圆大小统一，只比较方向和轴比，不能解释为局部作用范围。下排读取12个确认网络针对原训练目标的既有聚合，点为合并事件后重算的距离，线为网络bootstrap的90%区间，四项都是越小越好。\n**关注点**：p030与p089体现时间顺序和空间参与/覆盖之间的取舍；这些结果来自修正自连接之前，且不是患者留出验证结果。\n\n### interior_core_geometry_proposals.png / .pdf / .svg\n将旧边缘几何与四个有明确边界约束的内部几何并排显示，均使用1499个核内E细胞。新候选保留历史核间距，分别采用历史位置或平面中心，并将核连线与相应候选的设定连接核方向一致；尚未赋予任何新动力学结果。\n**关注点**：位置、轴对齐、预算和边界是独立可审计的约束，不能沿用旧候选的性能为新位置背书。\n')
    print(json.dumps({'output':str(out),'candidates':len(report),'interior_proposals':len(proposals)},indent=2),flush=True)


if __name__=='__main__':main()

"""Post-hoc core-line geometry summaries; no new fitting or simulations."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=R/'core_line_geometry_review';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
MAN={'role':'posthoc_frozen_output_geometry_association_not_single_parameter_causal_response','figures':[]}
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=155,bbox_inches='tight')
 plt.close(fig);MAN['figures'].append(name)
def geometry(p):
 a,b=np.asarray(p[:4]).reshape(2,2);v=b-a;d=np.linalg.norm(v);mid=(a+b)/2;theta=-22.80538396505847+p[9];u=np.array([np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))]);parallel=abs(v@u);perp=abs(np.linalg.det(np.stack([u,v])));phi=(np.degrees(np.arctan2(v[1],v[0]))+90)%180-90
 return {'dx_mm':abs(v[0]),'dy_mm':abs(v[1]),'distance_mm':d,'mid_x_mm':mid[0],'mid_y_mm':mid[1],'line_angle_deg':phi,'EE_axis_deg':theta,'EE_misalignment_deg':np.degrees(np.arctan2(perp,parallel)),'along_EE_mm':parallel,'across_EE_mm':perp}
g3=read(R/'g3_scores.json')['candidates'];train=read(R/'g2_all_64_scores.json')['candidates'];top=sorted(g3,key=lambda c:c['score']['loss_off'])[:3]
diagnostics={r['candidate_id']:r for r in csv.DictReader((R/'visual_review_delivery/search_diagnostic_metrics.csv').open())}
rows=[]
for c in train:
 row={'candidate_id':c['candidate_id'],'ranking_eligible':c['ranking_eligible'],**geometry(c['parameters']),'loss':c['score']['loss_off'],'stage':'DE-B' if '_de_b_' in c['candidate_id'] else ('DE-A' if '_de_a_' in c['candidate_id'] else 'Initial')}
 for k in ['participation','order','lag','direction','OOD']:
  v=diagnostics[c['candidate_id']][k];row[k]=float(v) if v not in ['','None'] else np.nan
 rows.append(row)
with (OUT/'geometry_observation_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
geo_top=[{'candidate_number':i+1,'candidate_id':c['candidate_id'],**geometry(c['parameters'])} for i,c in enumerate(top)]
wp=Path(next(iter(top[0]['units'].values()))['worker_path'])
with np.load(wp.with_suffix('.npz')) as z:xy=z['contact_xy_mm'];names=z['contact_names'].astype(str)
fig,axs=plt.subplots(1,3,figsize=(12,5.7));fig.subplots_adjust(top=.83,bottom=.30,wspace=.28)
for i,(c,ax,g) in enumerate(zip(top,axs,geo_top)):
 a,b=np.array(c['parameters'][:4]).reshape(2,2);mid=(a+b)/2
 for shaft,color in [('SCL','#cf8038'),('ICL','#289dba')]:
  ix=[j for j,n in enumerate(names) if n.startswith(shaft)];ix=sorted(ix,key=lambda j:xy[j,0]);ax.plot(xy[ix,0],xy[ix,1],c=color,lw=1)
 ax.scatter(*xy.T,s=18,c='white',edgecolor='.4',zorder=3)
 ax.plot([a[0],b[0]],[a[1],b[1]],color='#cc613c',lw=2.5);ax.scatter([a[0],b[0]],[a[1],b[1]],s=80,c='#cc613c',zorder=4)
 ax.plot([a[0],b[0],b[0]],[a[1],a[1],b[1]],color='#3988bb',ls=':',lw=1.5)
 ax.scatter(*mid,marker='x',c='black',s=45,zorder=5)
 u=np.array([np.cos(np.deg2rad(g['EE_axis_deg'])),np.sin(np.deg2rad(g['EE_axis_deg']))]);q=np.array([mid-3*u,mid+3*u]);ax.plot(*q.T,color='#8261a5',ls='--',lw=2)
 ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=f'Candidate {i+1}')
 ax.set_xticks([0,5,10,15,20]);ax.set_yticks([0,5,10,15,20]);ax.text(.5,-.23,f'dx = {g["dx_mm"]:.2f} mm | d = {g["distance_mm"]:.2f} mm\nmidpoint y = {g["mid_y_mm"]:.2f} mm\nline / EE angle = {g["EE_misalignment_deg"]:.1f} deg',transform=ax.transAxes,ha='center',va='top',fontsize=10)
fig.suptitle('Core separation is not the same as horizontal separation',fontsize=16)
fig.text(.5,.025,'Orange: core-to-core line | blue dotted: dx / dy | black x: midpoint | purple dashed: nominal EE axis\nCurrent search: center distance >= 4 mm; no dx penalty and no axis-alignment penalty.',ha='center',fontsize=10)
save(fig,'top3_core_line_definition')
metrics=[('loss','Joint loss'),('participation','Participation error'),('order','Order error'),('lag','Time-lag error (ms)'),('direction','Direction error'),('OOD','Out-of-distribution fraction')]
pages=[('distance_position',[('dx_mm','Horizontal separation (mm)'),('distance_mm','Core-to-core distance (mm)'),('line_angle_deg','Core line angle (deg)'),('mid_y_mm','Midpoint height (mm)')]),('axis_projection',[('mid_x_mm','Midpoint x (mm)'),('EE_misalignment_deg','Line / EE angle (deg)'),('along_EE_mm','Separation along EE axis (mm)'),('across_EE_mm','Separation across EE axis (mm)')])]
for name,defs in pages:
 fig,axs=plt.subplots(4,6,figsize=(18,10));fig.subplots_adjust(top=.88,bottom=.13,wspace=.42,hspace=.63)
 for ri,(key,label) in enumerate(defs):
  for col,(metric,title) in enumerate(metrics):
   ax=axs[ri,col]
   for stage,color,marker in [('Initial','#7c8797','o'),('DE-A','#3186a5','s'),('DE-B','#d06c38','^')]:
    sub=[r for r in rows if r['ranking_eligible'] and r['stage']==stage];ax.scatter([r[key] for r in sub],[r[metric] for r in sub],c=color,marker=marker,s=25,alpha=.75)
   for number,c in enumerate(top,1):
    r=next(r for r in rows if r['candidate_id']==c['candidate_id']);ax.scatter(r[key],r[metric],facecolors='none',edgecolors='black',s=80,zorder=5);ax.annotate(str(number),(r[key],r[metric]),xytext=(4,3),textcoords='offset points',fontsize=8)
   failed=[r for r in rows if not r['ranking_eligible']];ax.scatter([r[key] for r in failed],[-.08]*len(failed),transform=ax.get_xaxis_transform(),c='#c65858',marker='|',s=35,clip_on=False)
   ax.tick_params(axis='x',pad=10,labelsize=8);ax.tick_params(axis='y',labelsize=8);ax.set_xlabel(label,fontsize=9);ax.grid(alpha=.16)
   if ri==0:ax.set_title(title+' (lower)',fontsize=10)
   if key=='distance_mm':ax.axvline(4,c='.4',ls='--',lw=.8)
 fig.suptitle('Core-line geometry versus observed fit: existing 64-condition search',fontsize=15)
 fig.text(.5,.92,'Gray: initial | blue: DE-A | orange: DE-B | numbered circles: confirmation top 3 (shown at their training scores)',ha='center',fontsize=10)
 fig.text(.5,.025,'22 rankable conditions shown; red rail: 42 unranked conditions, not assigned a numerical loss.\nOther parameters and geometry coordinates vary together. These are associations, not isolated geometric effects or evidence of a unique optimum.',ha='center',fontsize=10)
 save(fig,'geometry_metrics_'+name)

# Existing matched controls isolate the full placement change, not distance alone.
ids=['v2_anchor_historical__baseline','v2_anchor_support_rank__baseline','v2_anchor_old_joint__baseline']
labels=['Farther diagonal','Upper, closer','Lower, horizontal']
base=[next(c for c in g3 if c['candidate_id']==cid) for cid in ids]
assert all(np.array_equal(c['parameters'][4:],base[0]['parameters'][4:]) for c in base)
confirm=list(csv.DictReader((R/'visual_review_delivery/confirmation_metrics.csv').open()));units=sorted(base[0]['units']);table=[]
for label,c in zip(labels,base):
 row={'placement':label,'candidate_id':c['candidate_id'],**geometry(c['parameters'])}
 for k in ['loss','participation','order','lag','direction','support']:
  row[k]=float(np.mean([float(r[k]) for r in confirm if r['candidate_id']==c['candidate_id']]))
 table.append(row)
fig,axs=plt.subplots(2,3,figsize=(12,7));fig.subplots_adjust(top=.82,bottom=.15,hspace=.53,wspace=.3)
for ax,(k,title) in zip(axs.flat,[('loss','Joint loss'),('participation','Participation error'),('order','Order error'),('lag','Time-lag error (ms)'),('direction','Direction error'),('support','Patient-supported fraction')]):
 for unit,color in zip(units,['#3672a4','#84a9ca','#b87344','#dcb391']):
  ys=[float(next(r[k] for r in confirm if r['candidate_id']==cid and r['unit']==unit)) for cid in ids];ax.plot(range(3),ys,'o-',lw=.8,ms=4,c=color,alpha=.8)
 ax.scatter(range(3),[r[k] for r in table],marker='_',s=260,c='black',linewidths=2,zorder=5)
 ax.set_title(title+(' (higher)' if k=='support' else ' (lower)'));ax.set_xticks(range(3),['Farther\ndiagonal','Upper,\ncloser','Lower,\nhorizontal'],fontsize=9);ax.grid(axis='y',alpha=.15)
fig.suptitle('Three existing placements with the same other model parameters',fontsize=15)
fig.text(.5,.87,'Threshold gain 1 | EE / EI / IE gains 1 | GABA decay 18 ms | nominal EE axis -22.8 deg / ratio 2',ha='center',fontsize=10)
fig.text(.5,.035,'Each thin line pairs the same topology and dynamics seeds; black marks: four-run means.\nThis compares full placements: midpoint, distance and orientation change together. It does not isolate an x-distance effect.',ha='center',fontsize=10)
save(fig,'matched_placement_comparison')
MAN.update(top3_geometry=geo_top,matched_placement_summary=table,hard_geometry_rules={'coordinate_bounds_mm':[.75,19.25],'minimum_center_distance_mm':4,'disjoint_disks':True,'minimum_nodes_per_core_fraction':.25,'dx_penalty':False,'axis_alignment_penalty':False},source_paths=[str(R/'g2_all_64_scores.json'),str(R/'g3_scores.json'),str(R/'visual_review_delivery/search_diagnostic_metrics.csv'),str(R/'visual_review_delivery/confirmation_metrics.csv')],parameterization={'independent':'midpoint x, midpoint y, center distance, unoriented center-line angle','derived':'dx, dy, separation along/across candidate nominal EE axis, line-to-EE acute angle','axis_note':'nominal candidate EE axis used consistently across all 64; not fitted patient propagation direction'})
(OUT/'manifest.json').write_text(json.dumps(MAN,indent=2)+'\n')
with (OUT/'matched_placement_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
desc={'top3_core_line_definition':('当前前三候选的双核连线、横纵距离、中点和名义EE轴，保持SEEG布局。标注的夹角是无向轴之间的锐角，横向距离与实际直线距离分开计算。','当前只有直线距离至少4 mm等硬几何约束，没有x距离惩罚，也没有连线对齐惩罚。'),'matched_placement_comparison':('三个已有布局对照的其他七个参数完全相同，细线配对同一拓扑和动力学种子，黑标为四次运行均值。对照改变的是整个双核布局，中点、距离和方向共同改变。','它支持比较整个位置方案，不能单独归因于x距离或夹角。')}
lines=[]
for name in MAN['figures']:
 a,b=desc.get(name,('将已有64条件的几何重表达与六个观测指标对应；仅展示22个可排名条件，42个不可排名条件放在红色底轨。数字圈标出确认期前三候选在训练期的位置与分数，未混用训练与确认损失。','多个几何量与生理参数同时变化；不能从此图直接读取单变量因果效应，也不把缺事件或runaway补为数值loss。'))
 lines.append(f'### {name}.png / .pdf\n\n{a}\n\n**关注点**：{b}\n')
(F/'README.md').write_text('\n'.join(lines))
print(json.dumps({'top3':geo_top,'matched':table},indent=2))

"""Descriptive matched-run state effects, frozen labels and saved events only."""
from pathlib import Path
import sys,csv,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import topic4_initial_state_runtime as rt
D=rt.read(ROOT/'config/topic4_continuous_core_state_r1.json');B=Path(D['output_root']);O=B/'representative_review';F=O/'figures'
rows=[];sources=[]
for j in D['jobs']:
 p=B/'workers'/(j['id']+'.json');r=rt.read(p);sources.append(dict(path=str(p),sha256=rt.sha(p)))
 hi=6000 if j['kind']=='fixed' else 30000
 ev=[e for e in r['events'] if e['window_ms'][0]>=1500 and e['window_ms'][1]<=hi and e['mode']>=0]
 if j['kind']=='fixed':
  for selection in ['all_classifiable_detections','primary']:
   take=[e for e in ev if selection!='primary' or e['primary_eligible']]
   rows.append(dict(job=j['id'],seed=j['dynamics_seed'],condition=j['z'],selection=selection,n=len(take),m0=sum(e['mode']==0 for e in take)))
 else:
  for sign in [-1,1]:
   take=[e for e in ev if e['primary_eligible'] and (e['z_at_detection_start']<0 if sign<0 else e['z_at_detection_start']>=0)]
   rows.append(dict(job=j['id'],seed=j['dynamics_seed'],condition=sign,selection='OU_primary_sign',n=len(take),m0=sum(e['mode']==0 for e in take)))
with (O/'state_effect_counts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(12.6,4.4));fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.79,wspace=.24)
seeds=sorted({r['seed'] for r in rows});colors=['#2878b5','#e58b39','#5a9a63']
for ax,selection,title in zip(axs,['all_classifiable_detections','primary','OU_primary_sign'],['Fixed s: all classifiable detections','Fixed s: qualified events','Continuous OU: qualified events']):
 for seed,color in zip(seeds,colors):
  rr=sorted([r for r in rows if r['seed']==seed and r['selection']==selection],key=lambda x:x['condition']);x=np.array([r['condition'] for r in rr],float);y=np.array([r['m0']/r['n'] if r['n'] else np.nan for r in rr])
  # Small display offsets separate coincident lines without changing the data.
  offset=(seeds.index(seed)-1)*(.035 if selection!='OU_primary_sign' else .065)
  ax.plot(x+offset,y,'o-',color=color,lw=1.4,ms=4,label=str(seed))
  if selection=='OU_primary_sign':
   for xx,yy,r in zip(x+offset,y,rr):ax.annotate(f'{r["m0"]}/{r["n"]}',(xx,yy),xytext=((seeds.index(seed)-1)*12,8),textcoords='offset points',ha='center',color=color,fontsize=8)
 ax.set(title=title,ylim=(-.05,1.13),yticks=[0,.25,.5,.75,1]);ax.grid(axis='y',alpha=.15)
 if selection=='OU_primary_sign':ax.set(xticks=[-1,1],xticklabels=['s < 0','s >= 0'],xlim=(-1.5,1.5),xlabel='State at detection start')
 else:ax.set(xticks=[-1,-.5,0,.5,1],xlabel='Imposed fixed state s')
axs[0].set_ylabel('M0 fraction');axs[0].legend(title='Paired noise replay',loc='upper left',bbox_to_anchor=(0,-.27),ncol=3,frameon=False,fontsize=8,title_fontsize=8)
fig.suptitle('State changes mode preference | one fixed graph, three matched noise replays',fontsize=13,y=.98)
fig.text(.62,.055,'Left includes overlapping / prolonged detections: sensitivity only.\nMiddle and right retain the frozen primary-event filter. Lines connect noise runs; events are not independent repeats.',ha='center',fontsize=8)
for ext in ['png','pdf']:fig.savefig(F/('state_effect_evidence.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
rt.write(O/'state_effect_metadata.json',dict(sources=sources,producer=str(Path(__file__).resolve()),producer_sha256=rt.sha(__file__),counts=rows,statistical_unit='noise replay on one topology',fixed_window_ms=[1500,6000],ou_window_ms=[1500,30000],ou_sign_comparison='posthoc descriptive grouping; does not isolate instantaneous state from its history',inference='no significance test; no new fit; all-detection sensitivity does not replace primary acceptance',human_visual_review_pending=True))
p=F/'README.md';s=p.read_text();entry='''\n### state_effect_evidence.png / .pdf

左、中两图是在同图、同背景输入重演中主动改变固定 s，分别统计全部可分类检测与冻结合格事件的 M0 比例；右图对三个连续 OU 重演按检测起点 s 的符号分组，点旁为 M0/全部合格事件数。线代表噪声重复，显示偏移仅用于分开重合的点；每组仍只有三个重复。
**关注点**：固定状态比较提供模型内部干预证据，OU 分组是包含历史效应的条件关联；全部检测包含重叠/延长事件，只作筛选敏感性检查，不能代替患者传播验收。\n'''
if '### state_effect_evidence.png' not in s:p.write_text(s+entry)
print('saved',F/'state_effect_evidence.png')

"""Descriptive variance partition and label-conditioned prototype reconstruction.

No simulation, classifier refit, objective change, or full-generative R2 claim.
"""
from pathlib import Path
import sys,json,pickle,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.topic4_joint_xy_kernel import event_kernel_features,kernel_map
from src.topic4_interictal_repaired_evaluation import rank_features
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1';OUT=R/'mode_variance_review';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
EP=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl'
with EP.open('rb') as f:ev=pickle.load(f)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def features(t):
 f=event_kernel_features(t,ev.xy,ev.groups,ev.scale);f['kernel_joint']=kernel_map(f['joint'],ev.maps['joint']).astype(float);return f
def partition(x,l):
 mu=x.mean(0);v=float(np.mean(np.sum((x-mu)**2,axis=1)));parts={}
 for k in [1,0]:
  a=x[l==k];p=len(a)/len(x);m=a.mean(0);w=float(np.mean(np.sum((a-m)**2,axis=1)));b=float(np.sum((m-mu)**2))
  parts[str(k)]={'n':len(a),'frequency':p,'within_variance':w,'within_contribution':p*w/v,'between_contribution':p*b/v}
 between=sum(z['between_contribution'] for z in parts.values());within=sum(z['within_contribution'] for z in parts.values());assert np.isclose(between+within,1,atol=1e-9)
 return {'n':len(x),'total_variance':v,'between_fraction':between,'modes':parts,'closure_error':abs(between+within-1)}
def reconstruct(x,l,means,patient_partition):
 mu=x.mean(0);v=patient_partition['total_variance'];r2=float(1-np.mean(np.sum((x-means[l])**2,axis=1))/v);con={}
 for k in [1,0]:
  a=x[l==k];p=len(a)/len(x);con[str(k)]=float(p*(np.sum((a.mean(0)-mu)**2)-np.sum((a.mean(0)-means[k])**2))/v)
 assert np.isclose(r2,sum(con.values()),atol=1e-9);assert r2<=patient_partition['between_fraction']+1e-9
 return {'conditional_prototype_R2':r2,'TA_contribution':con['1'],'TB_contribution':con['0']}
P={};PF={};PL={}
for split,ix in [('ALL',np.arange(len(ev.patient))),('FIT',ev.index['FIT']),('PROBE',ev.index['PROBE'])]:
 t=ev.patient[ix];PL[split]=ev.km.predict(rank_features(t));PF[split]=features(t);P[split]={k:partition(x,PL[split]) for k,x in PF[split].items()}
scores=json.loads((R/'g3_scores.json').read_text())['candidates'];cs=sorted(scores,key=lambda c:c['score']['loss_off'])[:3];model=[]
for i,c in enumerate(cs,1):
 mus={k:[] for k in PF['PROBE']};runrows={}
 for unit,u in c['units'].items():
  p=Path(u['worker_path']);op=p.parent.parent/'repaired_observation'/p.with_suffix('.npz').name
  with np.load(op) as z:t=z['centroid_ms'][z['primary_event_indices']]
  l=ev.km.predict(rank_features(t));assert np.bincount(l,minlength=2).tolist()==u['mode_counts'];ff=features(t);runrows[unit]={}
  for k,x in ff.items():
   means=np.array([x[l==m].mean(0) for m in [0,1]]);mus[k].append(means);runrows[unit][k]=reconstruct(PF['PROBE'][k],PL['PROBE'],means,P['PROBE'][k])
 summary={k:reconstruct(PF['PROBE'][k],PL['PROBE'],np.mean(ms,axis=0),P['PROBE'][k]) for k,ms in mus.items()}
 model.append({'number':i,'candidate_id':c['candidate_id'],'summary':summary,'per_run':runrows,'aggregation':'equal-unit average of mode means; separately retain each unit result; ensemble R2 is not mean per-unit R2'})
fit_reference={}
for k,x in PF['FIT'].items():fit_reference[k]=reconstruct(PF['PROBE'][k],PL['PROBE'],np.array([x[PL['FIT']==m].mean(0) for m in [0,1]]),P['PROBE'][k])
labels={'support':'Participation','rank_space':'Rank + space','timing_space':'Timing + space','joint':'Joint features','kernel_joint':'Frozen joint kernel'}
colors=['#918d98','#c75049','#3f84af']
fig,axs=plt.subplots(1,2,figsize=(12,6));fig.subplots_adjust(top=.79,bottom=.22,wspace=.48)
part=P['ALL']['joint'];ax=axs[0];freq=[part['modes']['1']['frequency'],part['modes']['0']['frequency']];var=[part['between_fraction'],part['modes']['1']['within_contribution'],part['modes']['0']['within_contribution']]
for x,vals,cols in [(0,freq,colors[1:]),(1,var,colors)]:
 bottom=0
 for val,col in zip(vals,cols):ax.bar(x,val*100,bottom=bottom*100,width=.55,color=col);ax.text(x,(bottom+val/2)*100,f'{val*100:.1f}%',ha='center',va='center',color='white',weight='bold');bottom+=val
ax.set(ylim=(0,100),ylabel='Percent',title='Frequency is not variance contribution');ax.set_xticks([0,1],['Event frequency','Total variance'])
names=['joint','rank_space','timing_space','support'];ax=axs[1]
for j,k in enumerate(names):
 d=P['ALL'][k];v=[d['between_fraction'],d['modes']['1']['within_contribution'],d['modes']['0']['within_contribution']];left=0
 for val,col in zip(v,colors):ax.barh(j,val*100,left=left,color=col);left+=val*100
 ax.text(d['between_fraction']*100+1,j,f'between = {d["between_fraction"]*100:.1f}%',va='center',fontsize=8)
ax.set(yticks=range(4),yticklabels=[labels[k] for k in names],xlim=(0,100),xlabel='Fraction of variance (%)',title='Each feature space has its own total');ax.invert_yaxis()
fig.suptitle('Patient event variability: between TA/TB and within each mode\n30,049 eligible events / 15 contacts / frozen mode labels',fontsize=15)
fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],['Between TA and TB means','Within TA','Within TB'],loc='lower center',bbox_to_anchor=(.5,.11),ncol=3,frameon=False)
fig.text(.5,.045,'Variation is across single-event participation / rank / timing / spatial feature vectors, not raw waveform variance.\nTA/TB labels are learned on FIT. This is descriptive; the between-mode fraction is not model-explained variance.',ha='center',fontsize=9)
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
save(fig,'patient_TA_TB_variance_partition')
fig,axs=plt.subplots(1,2,figsize=(13,6));fig.subplots_adjust(top=.80,bottom=.25,wspace=.33)
names=['support','rank_space','timing_space','joint','kernel_joint'];x=np.arange(len(names));ax=axs[0]
for off,row,color in [(-.14,fit_reference,'#777777'),(.14,model[0]['summary'],'#ab6c40')]:ax.bar(x+off,[row[k]['conditional_prototype_R2']*100 for k in names],width=.28,color=color,label='Patient FIT templates' if off<0 else 'Best model templates')
ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[labels[k].replace(' ','\n',1) for k in names],fontsize=9);ax.set(ylabel='Conditional template R2 (%)',title='Known TA/TB label; reconstruct patient PROBE');ax.legend(fontsize=9,frameon=False)
ax=axs[1]
for i,c in enumerate(model):
 for off,k,col in [(-.16,'TA_contribution',colors[1]),(.16,'TB_contribution',colors[2])]:ax.bar(i+off,c['summary']['joint'][k]*100,width=.3,color=col,label=k[:2] if i==0 else None)
 ax.plot(i,c['summary']['joint']['conditional_prototype_R2']*100,'k_',ms=18,mew=2)
ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['Candidate 1','Candidate 2','Candidate 3']);ax.set(ylabel='Contribution to joint template R2 (pp)',title='TA / TB contributions; black = total');ax.legend(frameon=False)
fig.suptitle('How much do model mean templates reconstruct?\n5,298 patient PROBE events; no event-to-event matching or model refit',fontsize=15)
fig.text(.5,.06,'R2 compares a label-specific mean template against the single PROBE grand mean. Negative values mean worse reconstruction.\nModel templates average 4 confirmation units equally; per-unit scores are retained. PROBE has been reused for development.\nThis does not measure the full generative distribution or mechanism variance explained; mode labels are supplied, not predicted here.',ha='center',fontsize=9)
save(fig,'model_TA_TB_template_reconstruction')
record={'role':'posthoc_descriptive_analysis_no_training_change','definition':'V_total = V_between_modes + p_TA*V_within_TA + p_TB*V_within_TB','feature_space':'frozen event_kernel_features, with original weights and patient FIT time scale; joint RFF sensitivity retained','time_scale_ms':ev.scale,'variance_unit':'between-event variation of within-event feature vectors; not raw waveform amplitude','patient_partitions':P,'model_prototypes':model,'patient_FIT_template_reference_on_PROBE':fit_reference,'restrictions':['Not full-generative variance explained','No independent validation claim: PROBE reused','No causal attribution of patient variance to modeled noise or network','No event-iid confidence intervals','Negative reconstruction R2 retained'],'source_files':[str(EP),str(R/'g3_scores.json')],'producer':str(Path(__file__).resolve())}
(OUT/'variance_decomposition.json').write_text(json.dumps(record,indent=2)+'\n')
rows=[]
for split,dd in P.items():
 for k,d in dd.items():rows.append({'partition':split,'features':k,'n':d['n'],'TA_frequency':d['modes']['1']['frequency'],'TB_frequency':d['modes']['0']['frequency'],'between_TA_TB':d['between_fraction'],'within_TA':d['modes']['1']['within_contribution'],'within_TB':d['modes']['0']['within_contribution']})
with (OUT/'patient_variance_partition.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(F/'README.md').write_text('''### patient_TA_TB_variance_partition.png / .pdf\n\n当前患者30049个合格事件在冻结观测特征中的方差分解，区别模式频率与方差贡献。总体方差精确分为TA/TB均值间差异、按自然频率加权的TA内部变化和TB内部变化；各特征空间单独归一化。\n\n**关注点**：这是单事件内部特征在事件之间的变化，不是原始波形方差；两类标签解释的部分不等于模型解释的部分。\n\n### model_TA_TB_template_reconstruction.png / .pdf\n\n以患者PROBE总体均值为基准，已知每个患者事件的冻结标签，用患者FIT均值模板或模型同标签均值模板重构事件特征。模型按四个运行等权平均条件均值，并在JSON保留逐运行结果；负R2不截断。\n\n**关注点**：这只评价平均模板重构，不能冒充生成分布或机制解释率；PROBE已有开发复用，TA/TB标签是给定的。\n''')
print('patient ALL joint',P['ALL']['joint']);print('model best joint',model[0]['summary']['joint']);print('DONE')

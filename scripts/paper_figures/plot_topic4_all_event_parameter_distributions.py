"""All confirmed parameters: natural-mixture and conditional observable views.

No changes to training, physical parameters, event selection, or frozen labels.
The ALL stratum is computed directly from the full primary-event table.
"""
from pathlib import Path
import sys,json,pickle,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic4_interictal_repaired_evaluation import rank_features

R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=R/'all_event_parameter_distributions';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
EP=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl'
with EP.open('rb') as f:ev=pickle.load(f)
with (R/'training_objective_v2_1.pkl').open('rb') as f:objective=pickle.load(f)
cs=sorted(json.loads((R/'g3_scores.json').read_text())['candidates'],key=lambda c:c['score']['loss_off'])
config=json.loads((ROOT/'config/topic4_multievent_distribution_search_v2_1.json').read_text())
parameter_names=list(config['parameters'])
titles=['Jointly optimized parameters','Reference placement A','Placement B: threshold offset x0.7','Reference placement B','Placement A: GABA decay 24 ms','Second joint parameter candidate','Historical placement']
colors=['#256a9c','#76b7dd','#975426','#d5a173'];runlabels=['Network 1 / noise 1','Network 1 / noise 2','Network 2 / noise 1','Network 2 / noise 2']
with np.load(Path(next(iter(cs[0]['units'].values()))['worker_path']).with_suffix('.npz')) as z:
    names=z['contact_names'].astype(str).tolist();xy=z['contact_xy_mm'].copy()
assert np.allclose(xy,ev.xy)
order=[i for i,n in enumerate(names) if n.startswith('SCL')]+[i for i,n in enumerate(names) if n.startswith('ICL')]
strata=['ALL','TA','TB'];stats=[];files=[];models=[];parameters=[];score_components=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def summary(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,**{k:None for k in ['mean','median','variance','sd','q05','q25','q75','q95']})
    q=np.quantile(a,[.05,.25,.5,.75,.95]);assert np.all(np.diff(q)>=0)
    return dict(n=len(a),mean=float(a.mean()),median=float(q[2]),variance=float(a.var()),sd=float(a.std()),q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4]))

def features(t):
    t=np.asarray(t,dtype=float);mask=np.isfinite(t);n=mask.sum(1);assert np.all(n>=2)
    rank=np.full(t.shape,np.nan)
    for j,row in enumerate(t):rank[j,mask[j]]=(rankdata(row[mask[j]])-1)/(n[j]-1)
    assert np.array_equal(np.isfinite(rank),mask)
    lag=t-np.nanmin(t,axis=1)[:,None];early=[];late=[]
    for rr,mm in zip(rank,mask):
        q=np.quantile(rr[mm],[1/3,2/3]);early.append(xy[mm&(rr<=q[0])].mean(0));late.append(xy[mm&(rr>=q[1])].mean(0))
    early=np.array(early);late=np.array(late);d=late-early
    result={'contact_count':n.astype(float),'centroid_span_ms':np.nanmax(lag,axis=1),'displacement_x_mm':d[:,0],'displacement_y_mm':d[:,1]}
    for j,name in enumerate(names):
        result['participation:'+name]=mask[:,j].astype(float);result['rank:'+name]=rank[:,j];result['relative_centroid_ms:'+name]=lag[:,j]
    return result

def split_and_record(t,source):
    # ALL does not read labels or use a target mode mixture. Classifier only organizes the additional rows.
    ff={'ALL':features(t)};labels=ev.km.predict(rank_features(t))
    for label,k in [('TA',1),('TB',0)]:ff[label]=features(t[labels==k])
    for s,data in ff.items():
        for key,a in data.items():stats.append(dict(source=source,stratum=s,observable=key,**summary(a)))
    assert len(ff['ALL']['contact_count'])==len(ff['TA']['contact_count'])+len(ff['TB']['contact_count'])
    # All-event mean must equal the natural-frequency conditional mixture, using per-feature participation counts.
    for key,a in ff['ALL'].items():
        m=summary(a);aa=summary(ff['TA'][key]);bb=summary(ff['TB'][key])
        assert m['n']==aa['n']+bb['n']
        if aa['n'] and bb['n']:assert np.isclose(m['mean'],(aa['mean']*aa['n']+bb['mean']*bb['n'])/m['n'])
    return ff,labels

patient,_=split_and_record(ev.patient,'patient_ALL')
for ci,c in enumerate(cs,1):
    parameters.append(dict(candidate=ci,title=titles[ci-1],candidate_id=c['candidate_id'],**dict(zip(parameter_names,c['parameters']))))
    runs=[]
    for unit,u in sorted(c['units'].items()):
        wp=Path(u['worker_path'])
        with np.load(wp.with_suffix('.npz')) as z:assert z['contact_names'].astype(str).tolist()==names and np.allclose(z['contact_xy_mm'],xy)
        op=wp.parent.parent/'repaired_observation'/wp.with_suffix('.npz').name
        with np.load(op) as z:t=np.asarray(z['centroid_ms'][z['primary_event_indices']],float)
        ff,l=split_and_record(t,c['candidate_id']+'/'+unit);assert np.bincount(l,minlength=2).tolist()==u['mode_counts']
        score=objective.score_network(t);assert np.isclose(score['loss_off'],u['score']['loss_off'],atol=1e-10)
        score_components.append(dict(candidate=ci,unit=unit,n=len(t),global_half=.5*score['D_off']['global']/objective.normalizers['global'],modes_half=.5*score['D_off']['balanced_modes']/objective.normalizers['balanced_modes'],loss=score['loss_off']))
        runs.append(dict(unit=unit,f=ff))
    models.append(runs)
print('ALL computed without labels; natural mixtures and 28 scores verified',flush=True)

def save(fig,name,text):
    for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig);files.append((name,text))

metrics=[('contact_count','Participating contacts'),('centroid_span_ms','Centroid time span (ms)'),('displacement_x_mm','Early-to-late x displacement (mm)'),('displacement_y_mm','Early-to-late y displacement (mm)')]
for ci,(c,runs) in enumerate(zip(cs,models),1):
    pp=c['parameters'];subtitle=f'Core centers ({pp[0]:.2f}, {pp[1]:.2f}), ({pp[2]:.2f}, {pp[3]:.2f}) mm | threshold offset x{pp[4]:.2f}\nEE x{pp[5]:.2f} | E to I x{pp[6]:.2f} | I to E x{pp[7]:.2f} | GABA {pp[8]:.2f} ms | EE angle offset {pp[9]:+.2f} deg | aspect {pp[10]:.2f}'
    fig,axs=plt.subplots(3,4,figsize=(17,10));fig.subplots_adjust(left=.15,right=.985,top=.82,bottom=.12,wspace=.3,hspace=.57)
    for ri,s in enumerate(strata):
        dd=[patient[s]]+[r['f'][s] for r in runs]
        for j,(key,title) in enumerate(metrics):
            ax=axs[ri,j]
            for z,data in enumerate(dd):
                d=summary(data[key]);col='black' if z==0 else colors[z-1]
                ax.plot([d['q05'],d['q95']],[z,z],color=col,lw=1.2);ax.plot([d['q25'],d['q75']],[z,z],color=col,lw=5,solid_capstyle='butt');ax.plot(d['median'],z,'o',color=col,ms=4);ax.plot(d['mean'],z,'|',color='#cf185e',ms=10,mew=1.5)
            ax.set_title(title,fontsize=10);ax.set_ylim(4.6,-.6);ax.grid(axis='x',alpha=.15)
            ax.set_yticks(range(5),[('Patient' if z==0 else runlabels[z-1])+f' n={len(d[key])}' for z,d in enumerate(dd)] if j==0 else [])
            if j==0:ax.set_ylabel('All events' if s=='ALL' else s,fontsize=13,weight='bold')
        for j,(key,_) in enumerate(metrics):
            allvals=np.concatenate([d[key] for d in [patient[q] for q in strata]+[r['f'][q] for r in runs for q in strata]])
            # Fixed axis across ALL / TA / TB for each observable within candidate.
            a,b=np.quantile(allvals,[0,1]);pad=max((b-a)*.04,.05)
            for ax in axs[:,j]:ax.set_xlim(a-pad,b+pad)
    fig.suptitle(f'C{ci}: {titles[ci-1]}\n{subtitle}',fontsize=12)
    fig.text(.5,.025,'All events: natural generated mixture; no TA/TB filtering or balancing. Each model run is separate.\nDot: median; magenta tick: mean; thick: 25-75%; thin: 5-95% event range (not confidence intervals).\nSpatial displacement is a contact-rank summary, not a continuous propagation path.',ha='center',fontsize=9)
    save(fig,f'c{ci}_ranges_all_and_modes',f'C{ci}：{titles[ci-1]}。第一行直接统计全部合格事件，不使用TA/TB标签筛选或重配比；后两行保留条件分布用于定位差距，四次运行分别列出。**关注点**：参数实际值写在图顶，所有行使用相同量的共同横轴；区间是事件范围，不是置信区间。')

    fig,axs=plt.subplots(3,3,figsize=(17,11.3));fig.subplots_adjust(top=.84,bottom=.18,hspace=.65,wspace=.24)
    for ri,s in enumerate(strata):
        for j,(prefix,title) in enumerate([('participation:','Participation probability'),('rank:','Participating-contact rank (0-1)'),('relative_centroid_ms:','Centroid delay from earliest contact (ms)')]):
            ax=axs[ri,j];x=np.arange(15);p=[summary(patient[s][prefix+names[k]]) for k in order]
            if j:ax.fill_between(x,[d['q25'] for d in p],[d['q75'] for d in p],color='gray',alpha=.22)
            ax.plot(x,[d['mean'] if j==0 else d['median'] for d in p],color='black',lw=2)
            for z,r in enumerate(runs):
                d=[summary(r['f'][s][prefix+names[k]]) for k in order];xx=x+(z-1.5)*.13
                ax.plot(xx,[v['mean'] if j==0 else v['median'] for v in d],'.-',color=colors[z],lw=.7,ms=3)
                if j:ax.vlines(xx,[v['q25'] for v in d],[v['q75'] for v in d],color=colors[z],lw=.8,alpha=.7)
            ax.set_title(title);ax.set_xticks(x,[names[k] for k in order],rotation=65,ha='right');ax.axvline(3.5,color='gray',ls=':',lw=.7);ax.grid(axis='y',alpha=.15)
            if j<2:ax.set_ylim(-.03,1.03)
            if j==0:ax.set_ylabel('All events' if s=='ALL' else s,fontsize=13,weight='bold')
    lim=max(ax.get_ylim()[1] for ax in axs[:,2])
    for ax in axs[:,2]:ax.set_ylim(0,lim)
    fig.suptitle(f'C{ci}: {titles[ci-1]}\n{subtitle}',fontsize=12)
    handles=[Line2D([],[],color='black',lw=2,label='Patient')]+[Line2D([],[],color=c,lw=2,label=l) for c,l in zip(colors,runlabels)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=5,frameon=False)
    fig.text(.5,.02,'First row: all events at their observed natural frequencies, no mode labels needed. Conditional rows are additional diagnostics.\nParticipation: mean. Rank/timing: median and IQR among participating contacts; absence remains missing. All moments saved in CSV.',ha='center',fontsize=9)
    save(fig,f'c{ci}_contacts_all_and_modes',f'C{ci}：{titles[ci-1]}。第一行是整体逐通道参与概率、顺序与时差分布，下两行是TA/TB条件分布；没有把TA/TB各取一半混合。**关注点**：整体均值接近仍可能掩盖多峰分布差异；顺序/时差只在通道参与时统计，标准差、方差与全部分位数另存CSV。')
    print('Plotted C'+str(ci),flush=True)

# Non-label CDFs distinguish spread/mixture shape from mean and quartile agreement.
fig,axs=plt.subplots(3,4,figsize=(16,9));fig.subplots_adjust(top=.9,bottom=.13,hspace=.5,wspace=.3)
cdfmetrics=[('contact_count','Participating contacts'),('rank:ICL1','ICL1 rank (0-1)'),('centroid_span_ms','Centroid time span (ms)'),('displacement_y_mm','Early-to-late y displacement (mm)')]
for ri,ci in enumerate([2,5,1]):
    for j,(key,title) in enumerate(cdfmetrics):
        ax=axs[ri,j]
        for z,data in enumerate([patient['ALL']]+[r['f']['ALL'] for r in models[ci-1]]):
            a=np.asarray(data[key]);a=np.sort(a[np.isfinite(a)]);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',color='black' if z==0 else colors[z-1],lw=1.8 if z==0 else 1)
        ax.set(title=title,ylim=(0,1),ylabel=f'C{ci}\nFraction of events');ax.grid(alpha=.15)
fig.suptitle('All-event distributions without TA/TB labels\nC2: reference A | C5: same parameters except GABA decay 18 to 24 ms | C1: jointly optimized',fontsize=14)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=5,frameon=False)
save(fig,'all_event_cumulative_distributions','不使用TA/TB标签，比较参考位置A、仅延长GABA衰减以及联合优化候选的经验累积分布；每种条件的四次运行分别画线。ICL1顺序仍以该通道参与为条件。**关注点**：曲线显示全部分布形状，均值与分位范围相似并不足以保证混合分布相似；C1对C2不是单参数干预。')

def writecsv(name,rows):
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('observable_statistics_all_and_modes.csv',stats);writecsv('candidate_parameters.csv',parameters);writecsv('loss_component_audit.csv',score_components)
(F/'README.md').write_text('\n\n'.join('### '+n+'.png / .pdf\n\n'+d for n,d in files)+'\n')
contrasts=[]
for a,b in [(2,5),(4,3)]:
    changed=[name for name,x,y in zip(parameter_names,cs[a-1]['parameters'],cs[b-1]['parameters']) if x!=y]
    assert len(changed)==1
    contrasts.append(dict(reference=a,intervention=b,changed_parameter=changed[0],common_units=list(cs[a-1]['units'])))
audit=dict(patient_n=len(ev.patient),patient_mode_mapping={'TA':1,'TB':0},patient_FIT_mode_proportions=objective.proportions.tolist(),normalizers=objective.normalizers,
    feature_shape=objective.maps['joint']['weights'].shape,objective_class=type(objective).__name__,all_event_statistics_use_labels=False,training_score_uses_frozen_labels=True,
    all_event_mixture='natural observed frequencies within each run, no balancing, no cross-run event pooling',conditional_strata='descriptive and training-informed, not independent route validation',
    early_late_definition='participating-contact rank quantiles 1/3 and 2/3, inclusive ties, equal-coordinate means, late minus early; descriptive summary added posthoc',
    training_spatial_definition='participation centroid, rank-weighted early (1-r) and late r centroids, ICL/SCL participation; not the tercile diagnostic',
    matched_parameter_contrasts=contrasts,run_count=28,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EP,R/'g3_scores.json',R/'training_objective_v2_1.pkl']})
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
report=['# 全部事件、模式条件与参数对照',
    '**纠正前提：生成器没有TA/TB输入，但优化评分确实使用冻结的患者模式分类器。** 因此应先看模型按自身自然比例生成的全部合格事件，再用TA/TB条件行定位差距；条件图不是完全独立于训练的验证。此次新增7个确认条件×2图，共14张，加1张不分标签的累积分布图。未重新训练、重跑物理仿真或更改历史评分。',
    '## 早晚接触点群的定义与边界',
    '上一轮诊断图对每个事件分别操作：只保留参与的接触点，按各点活动时间质心排序，得到0（最早）到1（最晚）的归一化rank；取该事件参与点rank的1/3和2/3分位边界，分别保留不晚于前一边界、不早于后一边界的点，边界并列全部保留。两群各取接触点坐标的等权平均，晚群中心减早群中心得到x/y位移。例如12个点且无并列时，是前4个和后4个。群内点不要求空间相邻，也没有预先指定SCL、ICL或双核成员；中心可以落在两根电极之间的无采样区域。这个三分之一划分是新增的描述性选择，不是患者已证明的源区或训练约束。它用的是活动时间质心，不是波形首次起始，不能单独证明点火或弯折路径。',
    '训练空间量采用另一种定义：所有参与点位置中心、按(1-rank)加权的偏早位置中心、按rank加权的偏晚位置中心，以及ICL/SCL参与比例。这些空间摘要连同逐通道参与、rank和相对时间进入训练。故上述三分之一图可用于直观审阅，但不能说它就是损失直接拟合的同一个量。',
    '## 当前损失实际奖励什么',
    '每个运行先将逐事件的参与、顺序、时差和空间摘要组成53维表征，再映射为冻结的512维核特征。评分是0.5×整体去自配对距离/固定尺度，加0.5×模式平衡去自配对距离/固定尺度，最后对运行等权平均。整体项比较全部事件的核特征分布；模式项将生成事件交给冻结患者分类器，利用患者FIT比例归一化的类别指示及核特征，同时奖励正确比例和类别内表征。FIT比例为TA 66.59%、TB 33.41%，不是强制各半；“平衡”指两类在该分量中的权重，不是人为修改生成事件数。',
    '去自配对统计量为均值嵌入距离平方减去该特征总方差/(N-1)，用于去掉自配对贡献；不是要求压低原始事件方差，也不是显式匹配每一项原始均值、方差和分位数。有限样本和连续事件相关性仍会影响排序。它没有逐事件指定TA/TB刺激，也没有明确奖励某条命名的右下→SCL→左下路径，不学习事件排列或模式切换动力学。相同数据中出现两个分类标签也不能证明真实的两条路径均已恢复。',
    '## 整体视图的新读数',
    '全部事件直接统计，模型四次运行各自保留自然比例，不跨网络合并、不把TA/TB各取一半。患者全部事件的质心跨度均值52.31ms，最佳候选四次为75.27、78.33、86.56、78.54ms；ICL11参与率患者84.36%，模型57.14%–65.22%。因此时差偏长与参与点偏差在不分标签的整体视图中仍存在，不能解释为分类后才出现的问题。',
    'C1相对C2的确认平均总分从1.079降至0.979；其中已乘0.5权重和固定归一化的整体项仅从0.871降至0.855，模式项从0.208降至0.124。因此这两个候选之间的评分改善主要来自模式项，不能把总分变化全部解释为不分标签的整体恢复。精确逐运行分量在loss_component_audit.csv；没有据此修改排名规则。',
    '## 其他参数条件的两张图',
    '每张图第一行都是全部事件，第二、三行是TA和TB；黑色患者，深浅蓝为网络1两次噪声，深浅棕为网络2两次噪声。参数实际值列在图顶。',
    '|编号|条件|中心与范围|逐通道分布|','|---|---|---|---|']
cn=['联合优化候选','参考位置A','位置B：阈值偏移倍率0.7','参考位置B','位置A：GABA衰减24ms','另一联合优化候选','历史双核位置']
for i,title in enumerate(cn,1):report.append(f'|C{i}|{title}|[图](figures/c{i}_ranges_all_and_modes.png)|[图](figures/c{i}_contacts_all_and_modes.png)|')
report+=['只有C2→C5、C4→C3是这里明确的一项参数对照：前者GABA衰减18→24ms，后者阈值偏移倍率1→0.7；其他条件不应据此归因于某一参数。C2→C5四次运行的平均参与点数都增加，平均质心跨度都缩短，但跨度中位数并不都缩短，且GABA变化还改变积分抑制剂量，不能单独归因为时间尺度机制。C1与C6同时改变位置、EE/EI/IE权重及其他量；当前7个确认条件不是覆盖所有全局参数的单因素扫描。',
    '[不分标签的累积分布图](figures/all_event_cumulative_distributions.png)额外保留分布形状，避免仅用均值和区间掩盖混合模式。所有均值、中位数、方差、SD、5/25/75/95%分位和有效样本量保存在observable_statistics_all_and_modes.csv；11个参数完整值在candidate_parameters.csv。',
    '这是同一患者、既有开发数据上的观测分布审阅；没有新增独立盲测，也没有以整体摘要替代原生场时间分辨路径验证。少数模式每次事件数较小，分位区间是描述范围，不是机制能力的硬性通过门槛。']
(OUT/'scientific_review.md').write_text('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n')
print('DONE',OUT,flush=True)

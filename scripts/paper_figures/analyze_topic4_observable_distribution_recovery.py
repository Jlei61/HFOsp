"""Posthoc patient/model marginal distributions; no refit or overall recovery score."""
from pathlib import Path
import sys, json, pickle, csv, hashlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic4_interictal_repaired_evaluation import rank_features

R = ROOT / 'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT = R / 'observable_distribution_recovery'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
EP = ROOT / 'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl'
with EP.open('rb') as f:
    ev = pickle.load(f)
cs = sorted(json.loads((R/'g3_scores.json').read_text())['candidates'], key=lambda c:c['score']['loss_off'])[:3]
with np.load(Path(next(iter(cs[0]['units'].values()))['worker_path']).with_suffix('.npz')) as z:
    names = z['contact_names'].astype(str).tolist()
    xy = z['contact_xy_mm'].copy()
assert np.allclose(xy, ev.xy)
order = np.array([i for i,n in enumerate(names) if n.startswith('SCL')] + [i for i,n in enumerate(names) if n.startswith('ICL')])
pair_i, pair_j = np.triu_indices(len(names), 1)
colors = ['#256a9c', '#76b7dd', '#975426', '#d5a173']
runlabels = ['Network 1 / noise 1', 'Network 1 / noise 2', 'Network 2 / noise 1', 'Network 2 / noise 2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
rows, comparisons, pairrows, covariance_rows, references = [], [], [], [], {}
files, checks = [], {'model_units_checked':0, 'mode_counts_match':True, 'contact_mapping_match':True}

def dist(x):
    a = np.asarray(x); a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, **{k:None for k in ['mean','median','variance','sd','q05','q25','q75','q95']})
    q = np.quantile(a,[.05,.25,.5,.75,.95])
    assert np.all(np.diff(q)>=0)
    return dict(n=int(len(a)),mean=float(a.mean()),median=float(q[2]),variance=float(a.var()),sd=float(a.std()),q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4]))

def observables(t):
    mask = np.isfinite(t)
    n = mask.sum(1)
    assert np.all(n>=2)
    rk = np.full(t.shape, np.nan)
    for j,row in enumerate(t):
        rk[j,mask[j]] = (rankdata(row[mask[j]], method='average')-1)/(n[j]-1)
    assert np.array_equal(np.isfinite(rk), mask)
    relative = t - np.nanmin(t,axis=1)[:,None]
    early, late = [], []
    for r,m in zip(rk,mask):
        # Include all ties at the tercile boundary; no arbitrary contact tie-breaking.
        lo,hi = np.quantile(r[m],[1/3,2/3])
        early.append(xy[m & (r<=lo)].mean(0))
        late.append(xy[m & (r>=hi)].mean(0))
    early,late = np.array(early), np.array(late)
    displacement = late-early
    lag = t[:,pair_j]-t[:,pair_i]
    f = {'contact_count':n.astype(float), 'centroid_span_ms':np.nanmax(relative,axis=1),
         'early_x_mm':early[:,0], 'early_y_mm':early[:,1], 'late_x_mm':late[:,0], 'late_y_mm':late[:,1],
         'displacement_x_mm':displacement[:,0], 'displacement_y_mm':displacement[:,1]}
    for j,name in enumerate(names):
        f['participation:'+name] = mask[:,j].astype(float)
        f['rank:'+name] = rk[:,j]
        f['relative_centroid_ms:'+name] = relative[:,j]
    for j,(a,b) in enumerate(zip(pair_i,pair_j)):
        f['lag_ms:'+names[a]+'->'+names[b]] = lag[:,j]
    return dict(f=f,mask=mask.astype(float),rank=rk,relative=relative,early=early,late=late,displacement=displacement,lag=lag,t=t)

def domain(key):
    if key=='contact_count' or key.startswith('participation:'):return 'participation'
    if key.startswith('rank:'):return 'order'
    if key=='centroid_span_ms' or key.startswith(('relative_centroid_ms:','lag_ms:')):return 'timing'
    return 'spatial_contact_summary'

def record(obs, source, mode, ref=None):
    for key,a in obs['f'].items():
        d=dist(a)
        rows.append(dict(source=source,mode=mode,domain=domain(key),observable=key,**d))
        if ref is not None:
            p=ref['f'][key];p=p[np.isfinite(p)];m=a[np.isfinite(a)];pd=dist(p)
            # Binary interval overlap is usually vacuous; probabilities remain in the raw table.
            eligible=len(m)>=5 and len(p)>=5 and not key.startswith('participation:')
            comparisons.append(dict(source=source,mode=mode,domain=domain(key),observable=key,n_model=len(m),n_patient=len(p),
                mean_difference=None if not len(m) else d['mean']-pd['mean'],
                median_difference=None if not len(m) else d['median']-pd['median'],
                variance_ratio=None if not len(m) or pd['variance']==0 else d['variance']/pd['variance'],
                model_in_patient_90pct_interval=None if not eligible else float(np.mean((m>=pd['q05'])&(m<=pd['q95']))),
                patient_in_model_90pct_interval=None if not eligible else float(np.mean((p>=d['q05'])&(p<=d['q95']))),
                interval_status='DESCRIPTIVE_MARGINAL' if eligible else ('BINARY_USE_PROBABILITY' if key.startswith('participation:') else 'FEWER_THAN_5_OBSERVED')))
    for j,(a,b) in enumerate(zip(pair_i,pair_j)):
        v=obs['lag'][:,j];v=v[np.isfinite(v)]
        probs = [float(np.mean(v>2)),float(np.mean(np.abs(v)<=2)),float(np.mean(v < -2))] if len(v) else [None]*3
        if len(v):assert np.isclose(sum(probs),1)
        pairrows.append(dict(source=source,mode=mode,contact_i=names[a],contact_j=names[b],n_both=len(v),p_i_before_j=probs[0],p_tie_2ms=probs[1],p_j_before_i=probs[2]))
    # Complete observations for participation and displacement: no missing-value imputation.
    for label,x,keys in [('participation',obs['mask'],names),('displacement',obs['displacement'],['dx_mm','dy_mm'])]:
        cv=(x-x.mean(0)).T@(x-x.mean(0))/len(x)
        assert np.allclose(cv,cv.T) and np.allclose(np.diag(cv),x.var(0))
        if label=='participation':assert np.allclose(np.diag(cv),x.mean(0)*(1-x.mean(0)))
        for i,a in enumerate(keys):
            for j,b in enumerate(keys):covariance_rows.append(dict(source=source,mode=mode,domain=label,variable_i=a,variable_j=b,n=len(x),covariance=float(cv[i,j])))

patient_labels=ev.km.predict(rank_features(ev.patient))
for split,ix in [('ALL',np.arange(len(ev.patient))),('FIT',ev.index['FIT']),('PROBE',ev.index['PROBE'])]:
    references[split]={}
    for mode,k in [('TA',1),('TB',0)]:
        o=observables(ev.patient[ix][patient_labels[ix]==k]); references[split][mode]=o; record(o,'patient_'+split,mode)
print('Patient reference complete',flush=True)
models=[]
for c in cs:
    runs=[]
    for unit,u in sorted(c['units'].items()):
        wp=Path(u['worker_path'])
        with np.load(wp.with_suffix('.npz')) as z:
            assert z['contact_names'].astype(str).tolist()==names and np.allclose(z['contact_xy_mm'],xy)
        op=wp.parent.parent/'repaired_observation'/wp.with_suffix('.npz').name
        with np.load(op) as z:t=z['centroid_ms'][z['primary_event_indices']]
        l=ev.km.predict(rank_features(t));assert np.bincount(l,minlength=2).tolist()==u['mode_counts']
        oo={}
        for mode,k in [('TA',1),('TB',0)]:
            oo[mode]=observables(t[l==k]);record(oo[mode],c['candidate_id']+'/'+unit,mode,references['ALL'][mode])
        runs.append(dict(unit=unit,obs=oo));checks['model_units_checked']+=1
    models.append(dict(candidate_id=c['candidate_id'],runs=runs))
print('12 model units checked',flush=True)

def save(fig,name,description):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=165,bbox_inches='tight')
    plt.close(fig);files.append((name,description))

def footer(fig,text):fig.text(.5,.02,text,ha='center',va='bottom',fontsize=9)
def legend(fig):
    handles=[Line2D([],[],color='black',lw=2,label='Patient')]+[Line2D([],[],color=c,lw=2,label=l) for c,l in zip(colors,runlabels)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=5,frameon=False)

for ci,c in enumerate(models,1):
    metrics=[('contact_count','Participating contacts'),('centroid_span_ms','Centroid time span (ms)'),('displacement_x_mm','Early-to-late displacement x (mm)'),('displacement_y_mm','Early-to-late displacement y (mm)')]
    fig,axs=plt.subplots(2,4,figsize=(17,7.7),layout=None);fig.subplots_adjust(left=.15,right=.985,top=.86,bottom=.19,wspace=.3,hspace=.6)
    for ri,mode in enumerate(['TA','TB']):
        datasets=[references['ALL'][mode]]+[r['obs'][mode] for r in c['runs']]
        for j,(key,title) in enumerate(metrics):
            ax=axs[ri,j]
            for s,o in enumerate(datasets):
                d=dist(o['f'][key]);color='black' if s==0 else colors[s-1]
                ax.plot([d['q05'],d['q95']],[s,s],color=color,lw=1.3)
                ax.plot([d['q25'],d['q75']],[s,s],color=color,lw=5,solid_capstyle='butt')
                ax.plot(d['median'],s,'o',color=color,ms=5);ax.plot(d['mean'],s,'|',color='#cf185e',ms=11,mew=1.5)
            ax.set_title(title);ax.set_ylim(4.7,-.7);ax.grid(axis='x',alpha=.15)
            ax.set_yticks(range(5),[('Patient' if s==0 else runlabels[s-1])+f"  n={len(o['t'])}" for s,o in enumerate(datasets)] if j==0 else [])
            if j==0:ax.set_ylabel(mode,fontsize=16,weight='bold',labelpad=14)
    fig.suptitle(f'Candidate {ci}: distribution centers and ranges\nTA and TB shown separately; four model runs remain separate',fontsize=16)
    footer(fig,'Dot: median  |  magenta tick: mean  |  thick line: 25-75%  |  thin line: 5-95% of events (not confidence intervals)\nSpatial displacement joins early/late contact groups; it does not locate wave ignition or reconstruct a continuous path.')
    save(fig,f'candidate_{ci}_distribution_ranges','按 TA/TB 比较参与数、活动时间质心跨度、早晚接触点群位置的 x/y 位移。点为中位数，粉色竖线为均值，粗线为四分位区间，细线为5%–95%事件范围；四次运行分别显示样本量。**关注点**：范围不是置信区间；时间跨度不是波形持续时间，位移不是完整传播路径。')

    fig,axs=plt.subplots(2,3,figsize=(17,9));fig.subplots_adjust(top=.86,bottom=.22,hspace=.55,wspace=.25)
    for ri,mode in enumerate(['TA','TB']):
        p=references['ALL'][mode]
        for j,(prefix,title) in enumerate([('participation:','Probability of participation'),('rank:','Rank among participating contacts (0-1)'),('relative_centroid_ms:','Centroid delay from earliest contact (ms)')]):
            ax=axs[ri,j];pd=[dist(p['f'][prefix+names[k]]) for k in order];x=np.arange(15)
            if j:
                ax.fill_between(x,[d['q25'] for d in pd],[d['q75'] for d in pd],color='gray',alpha=.23,label='Patient IQR')
            ax.plot(x,[d['mean'] if j==0 else d['median'] for d in pd],color='black',lw=2)
            for z,r in enumerate(c['runs']):
                d=[dist(r['obs'][mode]['f'][prefix+names[k]]) for k in order];xx=x+(z-1.5)*.13
                y=np.array([v['mean'] if j==0 else v['median'] for v in d],float)
                ax.plot(xx,y,'.-',color=colors[z],lw=.7,ms=4)
                if j:ax.vlines(xx,[v['q25'] for v in d],[v['q75'] for v in d],color=colors[z],lw=.9,alpha=.65)
            ax.set_title(title);ax.set_xticks(x,[names[k] for k in order],rotation=65,ha='right');ax.axvline(3.5,color='gray',ls=':',lw=.8);ax.grid(axis='y',alpha=.15)
            if j==0:ax.set(ylim=(-.03,1.03),ylabel=mode)
            if j==1:ax.set_ylim(-.03,1.03)
    fig.suptitle(f'Candidate {ci}: which contacts match, and which do not?',fontsize=16)
    legend(fig);footer(fig,'Participation: means. Rank and timing: medians and interquartile ranges, conditional on that contact participating.\nAbsence is retained as missing; no zero timing/rank imputation. Patient reference: all 30,049 eligible events.')
    save(fig,f'candidate_{ci}_contact_distributions','逐通道展示参与概率、参与时的归一化顺序和相对最早活动质心的时差。患者为黑线与灰色四分位带，模型保留两张网络各两次噪声运行；顺序/时差仅统计该通道参与的事件。**关注点**：某通道时差相似不能弥补它的参与概率错误；通道分布均值、中位数、方差与分位数全部另存表格。')

    fig,axs=plt.subplots(2,3,figsize=(15,10));fig.subplots_adjust(top=.88,bottom=.17,hspace=.32,wspace=.3)
    for ri,mode in enumerate(['TA','TB']):
        p=references['ALL'][mode]
        for j,key in enumerate(['early','late','displacement']):
            ax=axs[ri,j];pts=p[key];ii=np.linspace(0,len(pts)-1,min(1500,len(pts)),dtype=int)
            ax.scatter(pts[ii,0],pts[ii,1],s=9,color='#777777',alpha=.08,rasterized=True)
            ax.scatter(*np.median(pts,axis=0),s=65,color='black',marker='x',lw=2)
            for z,r in enumerate(c['runs']):
                v=r['obs'][mode][key];ax.scatter(v[:,0],v[:,1],s=22,alpha=.6,color=colors[z]);ax.scatter(*np.median(v,axis=0),s=70,color=colors[z],marker='x',lw=2)
            if j<2:
                for group,col in [('SCL','#e39c42'),('ICL','#45b5ca')]:
                    ind=[k for k,n in enumerate(names) if n.startswith(group)];ax.plot(xy[ind,0],xy[ind,1],'.-',color=col,ms=4,lw=1)
                ax.set(xlim=(0,20),ylim=(0,20),xlabel='x (mm)',ylabel='y (mm)')
            else:
                ax.axhline(0,c='gray',lw=.6);ax.axvline(0,c='gray',lw=.6);ax.set(xlim=(-16,16),ylim=(-16,16),xlabel='late x - early x (mm)',ylabel='late y - early y (mm)')
            ax.set_aspect('equal');ax.set_title(mode+' | '+['Early contact group','Late contact group','Early-to-late displacement'][j])
    fig.suptitle(f'Candidate {ci}: spatial distributions in the fixed SEEG layout',fontsize=16)
    legend(fig);footer(fig,'Gray: up to 1,500 patient events per mode for display; all model events shown. Crosses: coordinate-wise medians.\nEarly/late groups: bottom/top thirds of participating contact ranks, including boundary ties. These are contact summaries, not wave sources.')
    save(fig,f'candidate_{ci}_spatial_distributions','固定SEEG布局上展示每个事件较早、较晚三分之一参与接触点的位置中心及二者位移。灰点只为显示抽取最多1500个患者事件；数值统计仍使用全部事件，彩点为每次模型运行的全部事件。**关注点**：叉号是逐坐标中位数；这些位置是质心时间排序得到的接触点群，不是起燃位置。')

    # Full pairwise ordering structure, preserving same-contact conditioning and ties.
    fig,axs=plt.subplots(2,3,figsize=(14,10));fig.subplots_adjust(top=.85,bottom=.14,wspace=.35,hspace=.5)
    for ri,mode in enumerate(['TA','TB']):
        matrices=[]
        for o in [references['ALL'][mode]]+[r['obs'][mode] for r in c['runs']]:
            a=np.full((15,15),np.nan)
            for k,(i,j) in enumerate(zip(pair_i,pair_j)):
                v=o['lag'][:,k];v=v[np.isfinite(v)]
                if len(v)>=5:a[i,j]=np.mean(v>2);a[j,i]=np.mean(v < -2)
            matrices.append(a[np.ix_(order,order)])
        stack=np.stack(matrices[1:]);count=np.isfinite(stack).sum(0)
        model=np.divide(np.nansum(stack,axis=0),count,out=np.full((15,15),np.nan),where=count>0)
        for j,(a,title) in enumerate([(matrices[0],'Patient: P(row before column)'),(model,'Model: mean of four run probabilities'),(model-matrices[0],'Model minus patient')]):
            ax=axs[ri,j];im=ax.imshow(a,vmin=-.5 if j==2 else 0,vmax=.5 if j==2 else 1,cmap='RdBu_r' if j==2 else 'viridis')
            ax.set_title(mode+' | '+title,fontsize=10);ax.set_xticks(range(15),np.array(names)[order],rotation=90,fontsize=7);ax.set_yticks(range(15),np.array(names)[order],fontsize=7);fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle(f'Candidate {ci}: pairwise order probabilities\nOnly events in which both contacts participate; earlier means more than 2 ms earlier',fontsize=15)
    footer(fig,'Ties within 2 ms are retained separately in the CSV, so opposite probabilities need not sum to 1.\nModel map averages available runs equally; pairs with fewer than 5 joint observations are blank. Per-run counts and all 105 pairs are saved.')
    save(fig,f'candidate_{ci}_pairwise_order','展示所有通道对共同参与时的先后概率，右列是模型减患者的概率差。模型图等权平均四次运行的可估计概率，逐运行概率、2ms内并列比例及共同参与数保留在表中。**关注点**：这不是单一方向误差；两方向概率之和可小于1，因为仍有并列事件。')
    print('Figures complete',ci,flush=True)

# Show spread in original observable units, independently of mean/median agreement.
c=models[0]
fig,axs=plt.subplots(2,3,figsize=(17,8.5));fig.subplots_adjust(top=.86,bottom=.23,hspace=.58,wspace=.26)
for ri,mode in enumerate(['TA','TB']):
    for j,(prefix,title) in enumerate([('participation:','Participation standard deviation'),('rank:','Within-mode rank standard deviation'),('relative_centroid_ms:','Within-mode delay standard deviation (ms)')]):
        ax=axs[ri,j];x=np.arange(15)
        for z,o in enumerate([references['ALL'][mode]]+[r['obs'][mode] for r in c['runs']]):
            d=[dist(o['f'][prefix+names[k]]) for k in order]
            ax.plot(x,[v['sd'] for v in d],'.-',color='black' if z==0 else colors[z-1],lw=2 if z==0 else 1,ms=3)
        ax.set_title(title);ax.set_xticks(x,[names[k] for k in order],rotation=65,ha='right');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
        if j==0:ax.set_ylabel(mode)
fig.suptitle('Candidate 1: is event-to-event variation also reproduced?',fontsize=16)
legend(fig);footer(fig,'Descriptive SD within each mode and each run; variance (SD squared) is saved in the table.\nParticipation is binary: variance = p(1-p). Rank/timing variation is conditional on that contact participating; no event-iid confidence claims.')
save(fig,'candidate_1_contact_spread','独立于均值相似度，比较每个接触点在TA/TB内部的参与、顺序和时差标准差。标准差保持原单位，方差原值在统计表；二元参与的方差由参与概率决定。**关注点**：散布更小不自动更好，应与患者散布相容；少量模型事件的散布估计仍有观测不确定性。')

def writecsv(name,data):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
writecsv('observable_statistics.csv',rows)
writecsv('marginal_interval_comparison.csv',comparisons)
writecsv('pairwise_order_probabilities.csv',pairrows)
writecsv('covariance.csv',covariance_rows)
manifest=dict(role='posthoc_observable_distribution_comparison',producer=str(Path(__file__).resolve()),patient_reference='ALL 30049 eligible events; FIT and reused PROBE separately tabulated; not independent validation',
    modes={'TA':1,'TB':0},classifier='frozen patient FIT rank classifier; labels organize comparison, not certify routes',
    candidates=[dict(candidate_id=c['candidate_id'],runs=[r['unit'] for r in c['runs']]) for c in models],
    contacts=names,contact_xy_mm=xy.tolist(),variance='population descriptive variance, ddof=0; units squared; no causal explained-variance claim',
    intervals='empirical event quantiles, not confidence intervals; marginal overlap only, not joint-distribution recall or OOD',
    low_counts='interval overlap omitted below 5 participating observations; all raw descriptive statistics retained; small model quantile ranges are uncertain',
    conditioning='rank/timing by participating contacts; pair lag/order by both participating; no imputation; normalized rank=(average rank - 1)/(n_participating - 1)',
    spatial='centers of bottom/top thirds of participating ranks, boundary ties included; not ignition/continuous path; fixed same-patient projected geometry',
    covariance='complete participation and 2D displacement covariance only; rank and lag dependencies shown as conditional pairwise observables, no imputed covariance',
    model_observer='firing-density-derived centroid proxy; patient HFO-derived centroid; discrepancy can arise from physical propagation or observation mapping',
    checks=checks,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EP,R/'g3_scores.json']})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(FIG/'README.md').write_text('\n\n'.join('### '+name+'.png / .pdf\n\n'+desc for name,desc in files)+'\n')
compact={}
for mode in ['TA','TB']:
    keys=['contact_count','centroid_span_ms','displacement_x_mm','displacement_y_mm']
    compact[mode]={key:{'patient':dist(references['ALL'][mode]['f'][key]),'model_runs':[dist(r['obs'][mode]['f'][key]) for r in models[0]['runs']],
        'interval_comparison':[d for d in comparisons if d['source'].startswith(models[0]['candidate_id']+'/') and d['mode']==mode and d['observable']==key]} for key in keys}
(OUT/'best_candidate_summary.json').write_text(json.dumps(compact,indent=2)+'\n')
report=['# 患者—模型分维度分布恢复审阅',
    '本次仅重分析冻结输出，不重跑仿真、不改变损失或标签。当前最佳候选保留了一部分接触点排序和横向组织，但参与分布、时间尺度和纵向散布仍有明确差距；不能据此接受患者传播路径已恢复。',
    '## 先说明比较前提',
    '患者参考是同一患者全部30,049个合格事件：TA 19,563，TB 10,486；标签来自冻结的患者FIT聚类器。另存FIT、PROBE分布用于查看参考分区敏感性，PROBE已经开发复用，不是新盲测。模型保留每个候选两张拓扑×两次噪声的四个运行；最佳候选每次TA 23–30个、TB 12–17个事件。没有把这些事件当成独立患者，也没有据此计算独立事件置信区间。',
    '一次事件的观测是15个接触点是否参与，以及参与点的活动时间质心。参与概率使用全部该模式事件；rank和时差仅在通道参与时统计，通道对仅在两点共同参与时统计。患者来自HFO时间观测，模型来自放电密度代理读出；这里首先评价观测吻合，不能把差异直接归因于某个突触机制。',
    '## 读数定义',
    '- 中心：均值与中位数分别列出。散布：方差、标准差、25%–75%和5%–95%分位区间。方差以总体描述性公式除以N，不是去偏方差估计；SD保持原单位。\n- 参与：事件参与点数和逐通道参与概率；二元变量的方差是p(1-p)，不把通常为[0,1]的范围当有用覆盖率。\n- 顺序：参与点内归一化rank，0最早、1最晚；保存所有105个通道对的先后/2ms内并列概率。\n- 时差：逐通道相对最早质心的时差、所有通道对有符号时差、最晚减最早质心的跨度，单位ms。跨度不是波形持续时间或物理传播耗时。\n- 空间：早/晚三分之一参与点的位置中心及其位移，固定SEEG投影坐标、单位mm；保留位置散点和位移协方差。这是接触点路径摘要，尚不是连续原生场路径的恢复检验。',
    '## 最佳候选的中心与散布',
    '下表模型列是四次运行各自统计量的最小–最大，不是先合并事件再统计；同一行不同列的两端未必来自同一运行。',
    '|模式 / 观测量|患者均值|患者中位数|患者SD|模型均值范围|模型中位数范围|模型SD范围|',
    '|---|---:|---:|---:|---:|---:|---:|']
def fmt(v):return f'{v:.2f}'
selected=[('contact_count','参与点数'),('rank:ICL1','ICL1顺序（0–1）'),('centroid_span_ms','质心时间跨度（ms）'),('displacement_x_mm','横向位移（mm）'),('displacement_y_mm','纵向位移（mm）')]
for mode in ['TA','TB']:
    for key,label in selected:
        p=dist(references['ALL'][mode]['f'][key]);mm=[dist(r['obs'][mode]['f'][key]) for r in models[0]['runs']]
        report.append('|'+mode+' / '+label+'|'+'|'.join(fmt(p[k]) for k in ['mean','median','sd'])+'|'+'|'.join(fmt(min(d[k] for d in mm))+'–'+fmt(max(d[k] for d in mm)) for k in ['mean','median','sd'])+'|')
report += ['## 怎样判断恢复效果',
    '1. **参与数大体落在患者范围，参与位置仍不匹配。** TA的ICL11患者参与率86.4%，模型四次仅50.0%–65.2%；TB的SCL9患者61.2%，模型82.4%–100%。这些是完整逐通道表中的具体差距，不能用总参与数相近掩盖。',
    '2. **TA的ICL沿线顺序有部分相似，TB右端偏早的关系恢复不足。** TB中ICL1的患者顺序中位数0.167，模型0.364–0.618；ICL11患者中位数0.571，模型0.778–1.000。完整图保留各通道分布及通道对先后概率，不把两个挑出的点当完整路径证明；样本数和模式分类条件都必须同时看。',
    '3. **时差尺度偏长，部分运行散布也过宽。** TA的跨度均值由患者54.54ms变成模型75.13–86.89ms，TB由48.15ms变成69.45–86.11ms。TA跨度方差为患者1.40–1.84倍，TB为1.40–3.60倍。这与其他包络起始/尾部指标并不矛盾：质心跨度和波形时程是不同观测量。',
    '4. **横向组织有所保留，纵向组织与范围不够相容。** TA横向位移均值接近患者，但纵向位移方差为患者2.44–3.45倍；TB纵向位移方差为1.96–2.57倍，且四次运行的均值可正可负。这说明方向摘要相近仍可能同时存在二维位置和散布偏差；不能把此处的早接触点中心称为起燃位置。',
    '## 覆盖范围必须双向读',
    '定义一：模型事件有多少落在患者该观测量的5%–95%区间；定义二：患者事件有多少落在模型该观测量的5%–95%区间。两者都是单维经验区间重叠，不是多维OOD率、精确率或路径召回率；小样本分位数不稳定，区间也可能跨过没有事件的内部空隙。每项都保留原始事件数，低于5个有效事件不报告区间重叠。',
    '例如TA纵向位移，患者落在模型区间的比例约98%–99%，看起来很高；但模型落在患者区间的比例只有50%–63%。结合方差比可见模型过宽，不能把第一个数字解释成优秀的恢复率。参与点数为离散变量、区间端点有大量事件，因此名义5%–95%区间也可能实际覆盖超过90%的患者事件。',
    '## 交付与边界',
    '三候选分别为：\n'+ '\n'.join(f'- Candidate {i}: `{c["candidate_id"]}`' for i,c in enumerate(models,1)),
    '图件每候选四张：中心与分位范围、逐通道分布、空间分布、全通道对先后概率；最佳候选另有逐通道散布图。`observable_statistics.csv`保存全部均值、中位数、方差、SD和分位数；`marginal_interval_comparison.csv`保存双向区间重叠；`pairwise_order_probabilities.csv`保留共同参与数及并列；`covariance.csv`保存参与及二维位移协方差。没有把异单位观测拼成一个Overall百分比。',
    '这轮尚未完成连续波形/原生场路径的分布级恢复证明。空间图可以定位接触点时序组织的偏差，不能确认TB是否逐事件完整经历右下→SCL→左下；该问题仍需保留时间分辨包络与原生场逐事件核对。候选比较为开发诊断，不能从少量运行推断患者机制或网络噪声的因果贡献。图件还需用户目视审阅。']
(OUT/'scientific_review.md').write_text('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n')
print('DONE',OUT,flush=True)

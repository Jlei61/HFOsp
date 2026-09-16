"""Raw units and true paired responses complement, but never alter, training loss."""
from pathlib import Path
import csv,warnings
import numpy as np
from itertools import combinations
from scripts.topic4_three_observable_plot_labels import plot_label, parameter_table

def describe(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return dict(n=len(v),mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,variance=float(v.var()) if len(v) else None,q05=float(np.quantile(v,.05)) if len(v) else None,q95=float(np.quantile(v,.95)) if len(v) else None)

def observables(t,names):
    from src.topic4_three_observable_objective import normalized_ranks
    r,m=normalized_ranks(t);r=np.where(m,r,np.nan);scl=[i for i,n in enumerate(names) if n.startswith('SCL')];icl=[i for i,n in enumerate(names) if n.startswith('ICL')]
    both=m[:,scl].any(1)&m[:,icl].any(1);lag=np.nanmedian(t[both][:,scl],axis=1)-np.nanmedian(t[both][:,icl],axis=1) if both.any() else np.array([])
    pairs={}
    for shaft,ix in [('SCL',scl),('ICL',icl)]:
        for i,j in combinations(ix,2):
            delta=t[:,j]-t[:,i];valid=np.isfinite(delta);v=delta[valid]
            pairs[f'{names[i]}→{names[j]}']=dict(shaft=shaft,joint_fraction=float(valid.mean()) if len(t) else None,order_probability=float(np.mean((v>0)+.5*(v==0))) if len(v) else None,**describe(v))
    contacts={}
    for i,name in enumerate(names):contacts[name]=dict(participation=float(m[:,i].mean()) if len(t) else None,**describe(r[:,i]))
    return dict(N=len(t),contacts=contacts,pairs=pairs,rod_lag_ms=describe(lag),both_rods=float(both.mean()) if len(t) else None,
        SCL_only=float((m[:,scl].any(1)&~m[:,icl].any(1)).mean()) if len(t) else None,ICL_only=float((~m[:,scl].any(1)&m[:,icl].any(1)).mean()) if len(t) else None),lag

def report(a,rr,top):
    import matplotlib.pyplot as plt
    ev,names,_=a.patient();obj=a.load_objective();F=a.F;folder=a.A/'raw';folder.mkdir(exist_ok=True)
    refs={};cache={};lags={};flat=[];pairs=[]
    def store(key,mode,x):
        m,l=observables(x,names);cache[(key,mode)]=m;lags[(key,mode)]=l
        flat.append(dict(unit=key,mode=mode,**{f'lag_{k}':v for k,v in m['rod_lag_ms'].items()},N=m['N'],both_rods=m['both_rods'],SCL_only=m['SCL_only'],ICL_only=m['ICL_only']))
        pairs.extend(dict(unit=key,mode=mode,pair=p,**v) for p,v in m['pairs'].items());return m
    for mode,k in [('ALL',None),('TA',1),('TB',0)]:refs[mode]=store('patient_FIT',mode,ev.fit if k is None else ev.fit[ev.fit_labels==k])
    for row in rr:
        key=f"{row['stage']}/{row['candidate']}/{row['topology']}_{row['noise']}";row['_raw_key']=key
        _,t,labels,ids,_=a.load_small(Path(row['source']))
        for mode,k in [('ALL',None),('TA',1),('TB',0)]:store(key,mode,t[ids] if k is None else t[ids[labels[ids]==k]])
    a.rt.write(folder/'observables.json',dict(patient=refs,runs={r['_raw_key']:{m:cache[(r['_raw_key'],m)] for m in refs} for r in rr},units='one simulation is a comparison unit; event ranges describe within-run variation; time in ms; variance in ms squared'))
    for name,rows in [('rod_lag_summary.csv',flat),('within_rod_pairs.csv',pairs)]:
        with (folder/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    selected=[r for r in rr if r['candidate'] in top and r['topology']==2511 and r['stage'] in ['initial','adaptive']]
    colors=dict(zip(top,['#276eae','#cb6539','#4a975c']))
    fig,axes=plt.subplots(3,3,figsize=(17,13));disp=a.DISPLAY
    for row,mode in enumerate(['ALL','TA','TB']):
        ax=axes[row,0]
        for r in [None]+selected:
            key='patient_FIT' if r is None else r['_raw_key'];v=lags[(key,mode)]
            if len(v):ax.step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',color='black' if r is None else colors[r['candidate']],ls='-' if r is None or r['noise']==847401 else '--',lw=2 if r is None else 1,label='患者FIT' if r is None else f"{plot_label(r['candidate'])} / 噪声{r['noise']}")
        ax.set(xlabel='SCL−ICL 质心中位差 (ms)',ylabel=f'{mode}：累计事件比例',ylim=(0,1));ax.grid(alpha=.15)
        for col,items in [(1,[k for k,v in refs[mode]['pairs'].items() if v['shaft']=='SCL']),(2,disp)]:
            ax=axes[row,col]
            for r in [None]+selected:
                key='patient_FIT' if r is None else r['_raw_key'];m=cache[(key,mode)]
                y=[m['pairs'][k]['order_probability'] if col==1 else m['contacts'][k]['participation'] for k in items]
                ax.plot(range(len(items)),y,'o-',color='black' if r is None else colors[r['candidate']],ls='-' if r is None or r['noise']==847401 else '--',ms=3,lw=2 if r is None else 1)
            ax.set(xticks=range(len(items)),xticklabels=items,ylim=(-.02,1.02),ylabel='后一触点晚于前一触点的概率' if col==1 else '触点参与概率');ax.tick_params(axis='x',rotation=70,labelsize=7);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7);fig.suptitle('患者原量对照：杆间毫秒分布、SCL内部先后、两杆触点参与\n黑线患者；每条模型线为一个固定拓扑/噪声，不以事件数代替网络重复',fontsize=13);fig.tight_layout(rect=(0,.14,1,.95));parameter_table(fig,top,height=.1)
    for ext in ['png','pdf']:fig.savefig(F/f'patient_model_raw_observables.{ext}',dpi=140)
    plt.close(fig)
    # Acquisition-stage progress: paired runs are the condition score; incumbent
    # component curves always refer to the same current minimum-J condition.
    train=[r for r in rr if r['stage'] in ['initial','adaptive'] and r['topology']==2511]
    grouped=[]
    for cid in dict.fromkeys(r['candidate'] for r in train):
        rs=[r for r in train if r['candidate']==cid]
        if len(rs)==2 and all(r['J'] is not None for r in rs):grouped.append(dict(candidate=cid,J=float(np.mean([r['J'] for r in rs])),parts=np.mean([r['components'] for r in rs],axis=0),time=max(r['updated_unix'] for r in rs)))
    grouped.sort(key=lambda r:r['time']);fig,axes=plt.subplots(1,4,figsize=(17,4));best=[]
    for i in range(len(grouped)):best.append(min(grouped[:i+1],key=lambda r:r['J']))
    xx=2*np.arange(1,len(grouped)+1)
    for j,ax in enumerate(axes):
        val=lambda r:r['J'] if j==0 else r['parts'][j-1]
        ax.scatter(xx,[val(r) for r in grouped],s=18,label='该条件两噪声均值');ax.plot(xx,[val(r) for r in best],color='#ca603b',label='当时最低J条件')
        ax.set(xlabel='已评分条件的物理槽位数（含旧轨迹复用）',ylabel=['联合目标J ↓','整体rank组 ↓','杆内/杆间时序组 ↓','参与组 ↓'][j]);ax.grid(alpha=.15)
    axes[0].legend(fontsize=7);fig.suptitle('优化进展；初始历史重评分与新增物理实验分别记账，确认结果不进入训练曲线');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(F/f'optimization_progress.{ext}',dpi=140)
    plt.close(fig)
    legacy=a.A/'legacy_diagnostics/comparison.json'
    if legacy.exists():
        old=a.rt.read(legacy)['rows'];paired=[]
        for r in grouped:
            vals=[v['old_loss'] for v in old if v['candidate']==r['candidate'] and v['topology']==2511 and v['noise'] in [847401,847402]]
            if len(vals)==2 and all(v is not None for v in vals):paired.append(dict(candidate=r['candidate'],new_J=r['J'],old_loss=float(np.mean(vals))))
        a.rt.write(folder/'new_old_ranking.json',dict(rows=paired,new_ranking=[r['candidate'] for r in sorted(paired,key=lambda r:r['new_J'])],old_ranking=[r['candidate'] for r in sorted(paired,key=lambda r:r['old_loss'])],interpretation='Same candidate/event pool, score-choice comparison only; no attribution to optimization algorithm'))
    # All 55 ICL pairs, same fixed contact ordering for every mode/condition.
    icl=[n for n in disp if n.startswith('ICL')];fig,axes=plt.subplots(3,1+len(top),figsize=(4*(1+len(top)),12),squeeze=False)
    for row,mode in enumerate(['ALL','TA','TB']):
        for col,cid in enumerate([None]+top):
            mm=[refs[mode]] if cid is None else [cache[(r['_raw_key'],mode)] for r in selected if r['candidate']==cid]
            matrix=np.full((11,11),np.nan)
            for i in range(11):
                for j in range(i+1,11):
                    vals=[m['pairs'][f'{icl[i]}→{icl[j]}']['order_probability'] for m in mm];vals=[v for v in vals if v is not None]
                    if vals:matrix[i,j]=np.mean(vals)
            ax=axes[row,col];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='coolwarm');ax.set(xticks=range(11),xticklabels=icl,yticks=range(11),yticklabels=icl,title=f'{mode}\n'+('患者FIT' if cid is None else plot_label(cid)));ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7)
    fig.subplots_adjust(bottom=.18,top=.9,hspace=.4);fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='列触点晚于行触点的概率');fig.suptitle('ICL全部55对：患者与逐运行等权模型摘要；灰白缺格为不可读或未展示三角',fontsize=12);parameter_table(fig,top,height=.1)
    for ext in ['png','pdf']:fig.savefig(F/f'ICL_all_pair_order.{ext}',dpi=140)
    plt.close(fig)
    plan=a.rt.read(a.OUT/'plan.json');ref=plan['reference_id'];lookup={(r['candidate'],r['topology'],r['noise']):r for r in rr}
    # Paired probes have one scalar change, unlike BO joint candidates.
    def values(cid,topology,noise,mode):
        r=lookup.get((cid,topology,noise))
        if r is None:return [np.nan]*6
        m=cache[(r['_raw_key'],mode)];p=refs[mode];original=r['raw'][mode]
        mae=[]
        for shaft in ['SCL','ICL']:
            valid=[abs(v['order_probability']-p['pairs'][k]['order_probability']) for k,v in m['pairs'].items() if v['shaft']==shaft and v['order_probability'] is not None and p['pairs'][k]['order_probability'] is not None]
            mae.append(np.mean(valid) if valid else np.nan)
        return [original.get('rank_correlation'),*mae,m['rod_lag_ms']['median'],m['both_rods'],original.get('participation_mae')]
    designs=[dict(prefix='',center=ref,meta=plan['initial_meta'],pairs=[(2511,n) for n in plan['seeds']])]
    response=a.OUT/'proposals/response.json'
    if response.exists():
        rp=a.rt.read(response);designs.append(dict(prefix='response_',center=rp['center'],meta=rp['meta'],pairs=[(2511,n) for n in plan['seeds']]+[(t,n) for t in plan['confirmation_seeds']['topology'] for n in plan['confirmation_seeds']['dynamics']]))
    for design,mode in [(d,m) for d in designs for m in ['ALL','TA','TB']]:
        fig,axes=plt.subplots(6,6,figsize=(21,17),squeeze=False)
        labels=['平均rank模板相关 ↑','SCL成对顺序概率误差 ↓','ICL成对顺序概率误差 ↓','SCL−ICL 时差中位数 (ms)','两杆共同参与比例','触点参与概率误差 ↓'];p=refs[mode];patient_values=[1,0,0,p['rod_lag_ms']['median'],p['both_rods'],0]
        for axis in range(6):
            ids=[design['center']]+[m['candidate'] for m in design['meta'] if m['axis']==axis];ids=sorted(set(ids),key=lambda c:a.run.vector(a.rt.read(a.OUT/'candidates'/f'{c}.json'))[axis])
            x=[a.run.vector(a.rt.read(a.OUT/'candidates'/f'{c}.json'))[axis] for c in ids]
            for (topology,noise),col in zip(design['pairs'],['#2875a9','#d7772f','#47955f','#bd4e54','#8866a9','#866249']):
                yy=np.asarray([values(c,topology,noise,mode) for c in ids],dtype=float)
                for j in range(6):axes[j,axis].plot(x,yy[:,j],'o-',color=col,label=f'网络{topology}/噪声{noise}',ms=4)
            for j in range(6):
                ax=axes[j,axis];ax.axhline(patient_values[j],color='black',ls='--',lw=1,label='患者FIT参考');ax.grid(alpha=.15);ax.set(xlabel=['左核X (mm)','左核Y (mm)','右核X (mm)','右核Y (mm)','核向外EE倍率','EE轴偏移 (°)'][axis],ylabel=labels[j] if axis==0 else '')
        axes[0,0].legend(fontsize=7);fig.suptitle(f'{mode} 参数→观测：共用拓扑/噪声种子，只改变横轴标量；方向改变会重建EE边和时延\n黑虚线为患者参考；缺失运行不填补；只在共同可读触点对计算顺序误差，参与缺失另外呈现',fontsize=13);fig.tight_layout(rect=(0,0,1,.95))
        for ext in ['png','pdf']:fig.savefig(F/f'{design["prefix"]}raw_paired_response_{mode}.{ext}',dpi=120)
        plt.close(fig)
    with (F/'README.md').open('a') as f:
        f.write('\n### optimization_progress.png\n按成对噪声完成后的条件排列训练进展；四图橙线始终对应同一个当时最低J条件，确认不进入曲线。\n**关注点**：旧轨迹重评分与新增仿真分开，分数改善不等于传播验收。\n')
        for name,desc in [('patient_model_raw_observables','ALL/TA/TB的杆间时差分布、SCL成对时序与15触点参与；黑线为患者，同一候选的两条噪声分别实/虚线。'),('ICL_all_pair_order','固定ICL身份展示55对先后概率，模型按运行等权；单元原始毫秒均值、中位数、方差和范围保存在raw/within_rod_pairs.csv。')]+[(f'{d["prefix"]}raw_paired_response_{m}',f'{m}的六个单标量配对响应；患者参考每格显示，缺失不补值，完整条件是变量物理作用的证据。') for d in designs for m in ['ALL','TA','TB']]:
            f.write(f'\n### {name}.png\n{desc}\n**关注点**：原量与损失分开；选择过的训练数据不冒称独立验证。\n')

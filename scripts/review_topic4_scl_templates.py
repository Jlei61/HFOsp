"""Patient mean timing templates and SCL-only shape audit, frozen outputs only.

Shape codes describe signs of three adjacent centroid differences, not new clusters
or causal wave paths. Complete-four and pairwise participation supports stay separate.
"""
from pathlib import Path
import sys, json, hashlib, time, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
from scripts import analyze_topic4_shape_output_response as an
from src.topic4_pdf_font_guard import install

BASE = Path('/data/hfosp/topic4_sef_hfo')
N = BASE/'core_multiseed_response_curves_20260913'
OUT = BASE/'scl_internal_template_review_20260914'
CID = 'bridge_circle_out125_xminus075'
CONTACTS = ['SCL9', 'SCL8', 'SCL7', 'SCL6']
MODES = [('ALL', None), ('TA', 1), ('TB', 0)]
COLORS = {2511:'#666666', 2711:'#2878b5', 2712:'#d47e18'}
PATTERNS = ['+++','---','+--','++-','-++','--+','+-+','-+-','ties']
PLABELS = ['9→6\n单向', '6→9\n单向', '8最晚\n单折', '7最晚\n单折',
           '8最早\n单折', '7最早\n单折', '8晚/7早\n双折', '8早/7晚\n双折', '精确\n并列']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stat(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,median=None,variance=None,q05=None,q95=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                variance=float(x.var(ddof=1)) if len(x)>1 else None,
                q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))


def signs(d, eps):
    return np.array([''.join(s) for s in np.where(d>eps,'+',np.where(d < -eps,'-','0'))])


def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    parent=an.rt.read(an.run.base.PARENT);ev=an.rt.load_evaluator(parent)
    names=np.asarray(an.rt.load_observation_contract(parent)['contact_names'])
    order=[list(names).index(n) for n in CONTACTS]
    data=[dict(id='patient_FIT',topology=None,noise=None,x=np.asarray(ev.fit)[:,order],labels=np.asarray(ev.fit_labels))]
    sources=[];event_ids={}
    for f in sorted((N/'analysis/units').glob('*/result.json')):
        r=an.rt.read(f);c=r['counts']
        if c['candidate'] != CID:continue
        assert c['physical_status']=='COMPLETE_NO_RUNAWAY'
        raw=Path(r['source']).with_suffix('.npz');assert sha(raw)==r['source_sha256']
        ids=[e['event'] for e in r['events'] if e['primary']]
        with np.load(raw) as a:
            idx=[list(a['contact_names'].astype(str)).index(s) for s in CONTACTS]
            x=a['centroid_ms'][ids][:,idx];labs=a['event_mode'][ids]
        assert len(ids)==c['primary'] and (labs==1).sum()==c['TA'] and (labs==0).sum()==c['TB']
        key=f"t{c['topology']}_n{c['noise']}"
        data.append(dict(id=key,topology=c['topology'],noise=c['noise'],x=x,labels=labs))
        event_ids[key]=ids;sources.append(dict(result=str(f),result_sha256=sha(f),arrays=str(raw),arrays_sha256=r['source_sha256']))
    data[1:]=sorted(data[1:],key=lambda d:(d['topology'],d['noise']))
    assert len(data)==7 and len(data[0]['x'])==19770
    cached={};support=[];profiles=[];patterns=[];pairs=[];spans=[];sensitivity=[];permutations=[];covariances=[]
    for ds in data:
        for mode,k in MODES:
            x=ds['x'] if k is None else ds['x'][ds['labels']==k]
            meta={z:ds[z] for z in ['id','topology','noise']};meta['mode']=mode
            ok=np.isfinite(x);complete=ok.all(1);full=x[complete]
            centered=full-full.mean(1,keepdims=True);d=np.diff(full,axis=1)
            cache=dict(full=full,centered=centered,d=d,n=len(x),complete_n=len(full),codes=signs(d,0))
            cached[(ds['id'],mode)]=cache
            support.append(dict(**meta,n=len(x),complete4_n=len(full),complete4_fraction=float(complete.mean()),
                                **{f'participating_{i}_n':int((ok.sum(1)==i).sum()) for i in range(5)},
                                **{n+'_participation':float(ok[:,j].mean()) for j,n in enumerate(CONTACTS)}))
            for j,n in enumerate(CONTACTS):profiles.append(dict(**meta,contact=n,**stat(centered[:,j])))
            spans.append(dict(**meta,**stat(np.ptp(full,axis=1))))
            for code in PATTERNS:
                hit=np.array(['0' in z for z in cache['codes']]) if code=='ties' else cache['codes']==code
                patterns.append(dict(**meta,shape=code,n=int(hit.sum()),denominator=len(full),fraction=float(hit.mean())))
            for eps in [0,2,5]:
                code=signs(d,eps)
                for s in [''.join(z) for z in itertools.product('+-0',repeat=3)]:
                    sensitivity.append(dict(**meta,tolerance_ms=eps,shape=s,n=int((code==s).sum()),denominator=len(full),fraction=float((code==s).mean())))
            for perm in itertools.permutations(range(4)):
                code='>'.join(CONTACTS[j] for j in perm)
                ranked=np.argsort(full,axis=1)
                tie_free=np.array([len(np.unique(row))==4 for row in full])
                hit=(ranked==np.array(perm)).all(1)&tie_free
                permutations.append(dict(**meta,centroid_order=code,n=int(hit.sum()),denominator=len(full),fraction=float(hit.mean())))
            for i,j in itertools.combinations(range(4),2):
                for scope,xx in [('pairwise_coparticipating',x),('complete4',full)]:
                    delta=xx[:,j]-xx[:,i];delta=delta[np.isfinite(delta)]
                    z=stat(delta)
                    pairs.append(dict(**meta,scope=scope,earlier_test=CONTACTS[i],later_test=CONTACTS[j],
                                      p_i_before_j=float(((delta>0)+.5*(delta==0)).mean()) if len(delta) else None,**z))
            if len(full)>1:
                cov=np.cov(d,rowvar=False)
                for i,j in itertools.product(range(3),repeat=2):covariances.append(dict(**meta,i=i,j=j,covariance_ms2=float(cov[i,j])))
    tables={'support':support,'centered_templates':profiles,'shape_frequencies':patterns,'pair_differences':pairs,
            'scl_span':spans,'shape_tolerance_sensitivity':sensitivity,'all_24_orders':permutations,'adjacent_covariances':covariances}
    for key,rows in tables.items():pd.DataFrame(rows).to_csv(OUT/f'{key}.csv',index=False)
    np.savez_compressed(OUT/'scl_events.npz',**{ds['id']+'_'+k:ds[k] for ds in data for k in ['x','labels']})
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11})
    # SCL timing averages use the same four contacts in each event.
    fig,axes=plt.subplots(2,2,figsize=(12,9));fig.subplots_adjust(left=.09,right=.98,top=.82,bottom=.19,wspace=.25,hspace=.65)
    for row,mode in enumerate(['TA','TB']):
        p=cached[('patient_FIT',mode)]['centered'];mean=p.mean(0);med=np.median(p,axis=0);lo,hi=np.quantile(p,[.05,.95],axis=0)
        a,b=axes[row];a.fill_betweenx(range(4),lo,hi,color='.85');a.plot(mean,range(4),'o-',c='black',label='均值');a.plot(med,range(4),'s--',c='.5',label='中位数')
        a.set_title(f'患者 {mode} 平均模板\n四点共同参与 n={len(p):,}')
        b.plot(mean,range(4),'o:',c='black',lw=2,label='患者均值')
        for ds in data[1:]:
            z=cached[(ds['id'],mode)]['centered'];seed=ds['noise'];m=z.mean(0)
            b.plot(m,range(4),color=COLORS[ds['topology']],marker='o' if seed==847401 else '^',ls='-' if seed==847401 else '--',alpha=.6 if len(z)<8 else 1,lw=1.3)
        ns='；'.join(f"{t}:"+'/'.join(str(cached[(f't{t}_n{s}',mode)]['complete_n']) for s in [847401,847402]) for t in COLORS)
        b.set_title(f'当前候选 {mode}：六次运行各自均值\n四点共同参与 n（两噪声） {ns}',fontsize=10)
        for ax in [a,b]:
            ax.axvline(0,c='.75',lw=.8);ax.set(yticks=range(4),yticklabels=CONTACTS,ylim=(3.4,-.4),xlim=(-35,35),xlabel='相对本事件四个SCL质心均值 (ms)');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9,loc='lower left')
    handles=[Line2D([],[],c=c,label=f'网络{t}') for t,c in COLORS.items()]+[
        Line2D([],[],c='black',marker='o',label='噪声847401'),Line2D([],[],c='black',marker='^',ls='--',label='噪声847402')]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.54,.93),frameon=False)
    fig.suptitle('用患者多事件平均模板看：SCL内部时序是否相同？',y=.995,fontsize=17)
    fig.text(.08,.035,'每事件只减一个共同时间零点，保留SCL内部毫秒时差；点/线表示质心统计，灰带为患者事件5–95%范围，不是平均HFO频谱。\n'
             '仅四点共同参与事件进入完整形态比较；患者TA 8,779/13,165、TB 3,109/6,605。每条模型线来自一条60秒运行，不跨网络混池。\n'
             '网络2712的TA完整事件仅7/1个，淡线用于呈现实际观察，不能据此声称稳定模板。缺失参与另见support.csv和相邻触点对图。\n'
             '患者FIT参与过训练；此为冻结输出的局部拟合诊断。core位置/连接/噪声与训练目标均不变。',fontsize=9)
    save(fig,'patient_mean_scl_templates')
    # Eight coarse timing shapes; exact ties retained separately.
    fig,axes=plt.subplots(2,1,figsize=(13,8));fig.subplots_adjust(left=.085,right=.985,top=.85,bottom=.22,hspace=.4)
    df=pd.DataFrame(patterns)
    for ax,mode in zip(axes,['TA','TB']):
        v=df[(df.id=='patient_FIT')&(df['mode']==mode)].set_index('shape').loc[PATTERNS,'fraction']*100
        ax.bar(np.arange(9)-.18,v,width=.27,color='.25',label='患者FIT')
        for j,ds in enumerate(data[1:]):
            v=df[(df.id==ds['id'])&(df['mode']==mode)].set_index('shape').loc[PATTERNS,'fraction']*100
            ax.plot(np.arange(9)+.02+j*.055,v,ls='none',marker='o' if ds['noise']==847401 else '^',color=COLORS[ds['topology']],ms=5,
                    alpha=.45 if cached[(ds['id'],mode)]['complete_n']<8 else 1)
        ax.set(xticks=range(9),xticklabels=PLABELS,ylabel='四点共同参与事件中的比例 (%)',ylim=(0,105),title=f'{mode}：相邻质心先后组合');ax.grid(axis='y',alpha=.18)
    fig.legend(handles=[Line2D([],[],c='.25',lw=7,label='患者FIT')]+handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False,fontsize=10)
    fig.suptitle('SCL有哪些局部时序形态？按固定触点排列计数',y=.992,fontsize=17)
    fig.text(.075,.035,'按SCL9→8→7→6排列，统计三个相邻质心差的正负：两类单向、四类单折、两类双折；精确并列另列。\n'
             '这是八种描述性形态，不是发现了八个自然簇或八种因果传播机制；不要求每次事件有一条连续组织前沿。\n'
             '每个彩点是一条运行。TA网络2712仅7/1个完整事件，比例可能极端；不能把少量观察当作机制能力定论。\n'
             '主图保留全部实际先后差；另保存2/5ms容差敏感性：小幅折线可能改变类别。模式标签、shape均未新增到训练或刺激。',fontsize=9)
    save(fig,'scl_timing_shape_frequencies')
    # Pairwise analysis includes incomplete-four events when the pair is present.
    pdf=pd.DataFrame(pairs);sp=pd.DataFrame(support)
    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(left=.10,right=.985,top=.83,bottom=.18,hspace=.4,wspace=.25)
    pair_ids=list(zip(CONTACTS[:-1],CONTACTS[1:]))
    for row,mode in enumerate(['TA','TB']):
        for ds in data:
            col='black' if ds['id']=='patient_FIT' else COLORS[ds['topology']];style=':' if ds['id']=='patient_FIT' else '-' if ds['noise']==847401 else '--'
            marker='s' if ds['id']=='patient_FIT' else 'o' if ds['noise']==847401 else '^'
            rr=pdf[(pdf.id==ds['id'])&(pdf['mode']==mode)&(pdf.scope=='pairwise_coparticipating')]
            vals=[float(rr[(rr.earlier_test==i)&(rr.later_test==j)].p_i_before_j.iloc[0]) for i,j in pair_ids]
            axes[row,0].plot(range(3),vals,color=col,ls=style,marker=marker,lw=1.6)
            rr=sp[(sp.id==ds['id'])&(sp['mode']==mode)].iloc[0]
            axes[row,1].plot(range(4),[rr[n+'_participation'] for n in CONTACTS],color=col,ls=style,marker=marker,lw=1.6)
        axes[row,0].set(title=mode+'：只要求该触点对同时参与',xticks=range(3),xticklabels=['SCL9早于8','SCL8早于7','SCL7早于6'],ylabel='先后概率',ylim=(-.03,1.03))
        axes[row,1].set(title=mode+'：各触点参与概率',xticks=range(4),xticklabels=CONTACTS,ylim=(-.03,1.03))
        for ax in axes[row]:ax.grid(alpha=.18)
    fig.legend(handles=[Line2D([],[],c='black',ls=':',marker='s',label='患者FIT')]+handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False,fontsize=10)
    fig.suptitle('不要求SCL四点都参与：先后顺序与参与缺口分开看',y=.997,fontsize=17)
    fig.text(.085,.035,'左图各对仅使用两个触点同时参与的事件，精确并列记0.5；每个点的实际分母见pair_differences.csv。右图分母为该类全部合格事件。\n'
             '这样保留不完整SCL事件，不以缺失代替某一先后顺序；固定3网络×2噪声，各运行单独显示。患者FIT并非独立验证集。\n'
             '完整四点的均值/中位数/方差/5–95%范围、六触点对时差、三维相邻时差协方差及全部24种严格排序均已保存。',fontsize=9)
    save(fig,'scl_pair_order_and_participation')
    manifest=dict(created_unix=time.time(),candidate=CID,patient_fit_n=len(ev.fit),patient_parent=str(an.run.base.PARENT),
                  sources=sources,event_ids=event_ids,producer=str(Path(__file__)),producer_sha256=sha(Path(__file__)),
                  complete_four_template='subtract per-event mean of four SCL centroids; then event mean, median, q05/q95; no waveform reconstruction',
                  shape_definition='signs of three adjacent centroid differences; no inferred K; exact tie category; sensitivity epsilon=2/5ms',
                  pairwise_scope='both contacts participating regardless of other SCL participation',new_physical_runs=0,loss_changes=0)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT/'figures/README.md').write_text('### patient_mean_scl_templates.png\n左列为患者完整SCL事件的均值/中位数与5–95%事件范围，右列为当前候选六次运行各自均值。每事件只减一个共同时间零点，Y轴SCL9至6固定；不是合成平均频谱。\n**关注点**：模型是否恢复均值形态，及低参与网络的模板证据是否足够。\n\n### scl_timing_shape_frequencies.png\n按三个相邻质心时差正负，呈现八种描述性形态与精确并列的比例；患者黑柱、逐网络噪声彩点。主图没有人为近同时容差，另存2/5ms敏感性；不将八形态解释成八自然模式。\n**关注点**：模型是否集中于患者少见的SCL单向顺序。\n\n### scl_pair_order_and_participation.png\n左侧保留所有相邻触点共同参与的事件，右侧独立展示参与概率。每条模型线固定一张网络和一条噪声。\n**关注点**：完整四点筛选是否掩盖部分参与事件，以及时序和参与是否都匹配。\n')
    print(OUT)
    print(pd.DataFrame(patterns).query("mode == 'TB' and shape == '---'").to_string(index=False))


def save(fig,stem):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{stem}.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()

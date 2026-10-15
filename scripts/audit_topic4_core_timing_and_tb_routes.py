"""Frozen-output audit: core timing association versus contact propagation recovery.

No simulation, relabeling, new route loss, or pooled-network significance test.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scripts import analyze_topic4_shape_output_response as an
rt=an.rt
BASE=Path('/data/hfosp/topic4_sef_hfo')
SOURCES=['core_shape_output_response_20260911','core_recruitment_tradeoff_followup_20260912','core_multiseed_response_curves_20260913']
ORDER=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]
CASES=[('up3__circle','圆形基线'),('up3__circle__EE_core_to_out_scale_1.25','向外EE增强25%'),('up3__circle__EE_core_to_out_scale_1.25__x_minus075','向外EE增强25%＋左移0.75mm'),('follow_circle_out1.125_EI0.875','向外EE×1.125＋核内EI×0.875')]
SEEDS=[847101,847102];MC={'TA':'#c95a4e','TB':'#397fb0'}
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3,'axes.spines.top':False,'axes.spines.right':False})


def collect(out):
    records=[];seen=set();manifest=[]
    for series in SOURCES:
      for f in sorted((BASE/series/'analysis/units').glob('*/result.json')):
        r=rt.read(f);resolved=str(Path(r['source']).resolve())
        if resolved in seen:continue
        seen.add(resolved);r['series']=series;records.append(r)
        manifest.append(dict(series=series,analysis_file=str(f),analysis_sha256=rt.sha(f),trajectory=r['source'],resolved_trajectory=resolved,arrays_sha256=r['source_sha256']))
    rt.write(out/'sources.json',dict(time=datetime.datetime.now().astimezone().isoformat(),unique_runs=len(records),deduplication='resolved trajectory identity; reused parent units are counted once',records=manifest))
    return records


def patient():
    design=rt.read(an.run.base.PARENT);ev=rt.load_evaluator(design);x=np.asarray(ev.fit);label=np.asarray(ev.fit_labels)
    names=np.asarray(rt.load_observation_contract(design)['contact_names']);scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL')
    refs={};lags={}
    for mode,m in [('ALL',np.ones(len(x),bool)),('TA',label==1),('TB',label==0)]:
        v=x[m];joint=np.isfinite(v[:,scl]).any(1)&np.isfinite(v[:,icl]).any(1);w=v[joint]
        lags[mode]=np.nanmedian(w[:,scl],axis=1)-np.nanmedian(w[:,icl],axis=1)
        refs[mode]=dict(events=len(v),joint_events=len(w),pairs={(names[i],names[j]):(n,p) for (i,j),(n,p) in an.an.pair_table(v).items()})
    return refs,lags


def tabulate(records,out):
    rows=[];events=[]
    for r in records:
        c=r['counts'];ev=pd.DataFrame(r['events'])
        for e in r['events']:events.append(dict(e,series=r['series']))
        if ev.empty:continue
        for layer in ['primary','all_detected']:
         for mode in ['ALL','TA','TB']:
            d=ev if layer=='all_detected' else ev[ev.primary]
            if mode!='ALL':d=d[d['mode']==mode]
            delta=d.B_minus_A_t10_ms.to_numpy(float);valid=np.isfinite(delta);b=delta[valid]
            share=d.coreA_mass/(d.coreA_mass+d.coreB_mass)
            row=dict(candidate=c['candidate'],series=r['series'],topology=c['topology'],noise=c['noise'],layer=layer,mode=mode,n=len(d),core_pair_valid_n=int(valid.sum()),
                A_t10_earlier_fraction=float((b>0).mean()) if len(b) else None,B_t10_earlier_fraction=float((b<0).mean()) if len(b) else None,t10_tie_fraction=float((b==0).mean()) if len(b) else None)
            for key,v in [('B_minus_A_t10_ms',b),('coreA_mass_fraction',share),('rod_centroid_lag_ms',d.centroid_SCL_minus_ICL_ms),('rod_t10_lag_ms',d.t10_SCL_minus_ICL_ms),('local_width_ms',d.local_width_ms),('recruitment_span_ms',d.recruitment_span_ms)]:
                a=np.asarray(v,float);a=a[np.isfinite(a)]
                for q,suffix in [(.05,'q05'),(.5,'median'),(.95,'q95')]:row[key+'_'+suffix]=float(np.quantile(a,q)) if len(a) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'core_contact_timing_by_run.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(events)


def save(fig,path):
    for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def core_contact_scatter(events,out):
    fig,axes=plt.subplots(2,4,figsize=(16.5,8));fig.subplots_adjust(left=.065,right=.99,top=.81,bottom=.19,wspace=.35,hspace=.43)
    for i,seed in enumerate(SEEDS):
      for j,(cid,label) in enumerate(CASES):
        ax=axes[i,j];d=events[(events.candidate==cid)&(events.topology==2511)&(events.noise==seed)&events.primary]
        for mode,color in MC.items():
            z=d[d['mode']==mode];ax.scatter(z.B_minus_A_t10_ms,z.centroid_SCL_minus_ICL_ms,c=color,s=13,alpha=.55,label=f'{mode} n={len(z)}')
        ax.axvline(0,c='#777',ls=':',lw=.8);ax.axhline(0,c='#777',ls=':',lw=.8)
        ax.set(xlim=(-180,180),ylim=(-45,130),xlabel='右核t10 − 左核t10 (ms)',ylabel='SCL − ICL 质心时差 (ms)',title=label+f'\n噪声{seed}')
        ax.legend(fontsize=7,frameon=False,loc='upper left');ax.grid(alpha=.12)
    fig.suptitle('核内活动先后与传播路径是不同层：TA常伴左核较早，但杆间时序仍可完全错误',fontsize=15,y=.98)
    fig.text(.065,.88,'横轴正值=左核窗口内累计10%发放较早；纵轴正值=SCL质心晚于ICL。红/蓝仅为冻结患者模板分配的TA/TB标签。',fontsize=10)
    fig.text(.065,.045,'每点为一次合格事件；每格来自一条60秒运行，不能把点当作独立网络。只展示两核t10和两杆质心均可估计的点，完整支持量见CSV。\nt10在既有250ms事件窗内按各核自身发放质量定义，包含递归活动；它不是因果起源时间，也不证明先活动的核驱动了另一个核。\n分类器未读取core的t10；但标签来自接触时序，因此颜色与纵轴并非独立验证。应结合不分标签分布和原生场。',fontsize=9)
    save(fig,out/'figures/core_timing_contact_timing')


def paired_probabilities(records,refs,out):
    lookup={(r['counts']['candidate'],r['counts']['topology'],r['counts']['noise']):r for r in records}
    ids=[CASES[1][0],CASES[2][0]];labels=['原位置／噪声847101','原位置／噪声847102','左移／噪声847101','左移／噪声847102']
    spec=[(cid,s) for cid in ids for s in SEEDS];fig,axes=plt.subplots(2,5,figsize=(19.5,8.5));fig.subplots_adjust(left=.052,right=.94,top=.83,bottom=.20,wspace=.35,hspace=.55)
    rows=[];cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d3d3d3')
    for row,mode in enumerate(['TA','TB']):
      for col,item in enumerate([None]+spec):
        ax=axes[row,col];mat=np.full((15,15),np.nan);support=np.zeros((15,15),int)
        if item is None:ps=refs[mode]['pairs'];n=refs[mode]['events']
        else:
            r=lookup[(item[0],2511,item[1])];ps={(z['contact_i'],z['contact_j']):(z['model_joint_n'],z['model_i_precedes_j']) for z in r['pairs'] if z['mode']==mode and z['layer']=='primary'};n=r['counts'][mode]
        for (ci,cj),(n_joint,p) in ps.items():
            i,j=ORDER.index(ci),ORDER.index(cj);support[i,j]=support[j,i]=n_joint
            if p is not None:mat[i,j]=p;mat[j,i]=1-p
            pn,pr=refs[mode]['pairs'][(ci,cj)]
            rows.append(dict(candidate='patient' if item is None else item[0],noise=None if item is None else item[1],mode=mode,contact_i=ci,contact_j=cj,joint_n=n_joint,probability_i_earlier=p,patient_joint_n=pn,patient_probability_i_earlier=pr,residual=None if p is None or pr is None else p-pr))
        im=ax.imshow(mat,cmap=cmap,vmin=0,vmax=1,interpolation='nearest');ax.set(xticks=range(15),xticklabels=ORDER,yticks=range(15),yticklabels=ORDER,title=(('患者FIT' if item is None else labels[col-1])+f'｜{mode} n={n}'))
        ax.tick_params(axis='x',labelrotation=90,labelsize=6);ax.tick_params(axis='y',labelsize=6);ax.axhline(3.5,c='black',lw=.6);ax.axvline(3.5,c='black',lw=.6)
    cax=fig.add_axes([.955,.28,.012,.48]);fig.colorbar(im,cax=cax,label='P(行接触点质心早于列接触点 | 两者参与)')
    fig.suptitle('完整接触顺序结构：左移改善部分TA，但TB的ICL上端顺序和分布仍有残差',fontsize=15,y=.985)
    fig.text(.052,.90,'固定分杆顺序：SCL9→6，然后ICL11→1。红=行较早，蓝=列较早；灰=对角或无共同参与支持。每个模型格独立显示一次运行。',fontsize=10)
    fig.text(.052,.035,'展示全部105个触点对，不只选ICL11/ICL9。零/一概率仍是有限观测，不宣称真实概率精确为零/一；逐格实际共同参与数在pair_probability_residuals.csv。\n患者参考是完整FIT自然事件，模型为全部合格事件条件分布；标签用于组织比较，不是另一套独立传播验收。两类都出现不等于患者两类都恢复。',fontsize=9)
    save(fig,out/'figures/contact_order_probability');pd.DataFrame(rows).to_csv(out/'pair_probability_residuals.csv',index=False)


def unconditional_lags(events,patient_lags,out):
    fig,axes=plt.subplots(2,4,figsize=(16.5,8));fig.subplots_adjust(left=.065,right=.99,top=.80,bottom=.19,wspace=.34,hspace=.72);bins=np.arange(-250,252,4);width=4.;rows=[]
    for i,seed in enumerate(SEEDS):
      for j,(cid,label) in enumerate(CASES):
        ax=axes[i,j];d=events[(events.candidate==cid)&(events.topology==2511)&(events.noise==seed)&events.primary];values=d.centroid_SCL_minus_ICL_ms.dropna().to_numpy()
        for name,vals,color in [('患者FIT',patient_lags['ALL'],'#555555'),('模型',values,'#467fa8')]:
            hist,_=np.histogram(vals,bins=bins);density=hist/(max(len(vals),1)*width);ax.stairs(density,bins,color=color,fill=True,alpha=.3,label=f'{name} n={len(vals)}')
            rows.append(dict(candidate=cid,noise=seed,sample=name,n=len(vals),outside_display=int(((vals<-75)|(vals>130)).sum()),outside_histogram=int(((vals<bins[0])|(vals>bins[-1])).sum())))
        ax.axvline(0,c='#888',ls=':',lw=.7);ax.set(xlim=(-75,130),ylim=(0,.16),xlabel='SCL − ICL 质心时差 (ms)',ylabel='事件比例 / ms',title=label+f'\n噪声{seed}');ax.legend(fontsize=7,frameon=False)
    fig.suptitle('不分TA/TB：模型仍在约+35ms附近形成窄峰，与患者分布不同',fontsize=15,y=.985)
    fig.text(.065,.88,'每个格使用全部合格、两杆均参与的事件；相同4ms直方图分箱，以全部可估计事件数归一化。没有重排、时间缩放或按标签筛选。',fontsize=10)
    fig.text(.065,.045,'每条运行单独与患者FIT比较；图外事件仍计入分母并记录在histogram_support.csv，不将截断窗口重新归一化。\n该图回答跨杆质心差的无条件分布，不能替代逐接触路径，也不比较HFO局部宽度；两杆未同时参与的事件另在原参与表保留。',fontsize=9)
    save(fig,out/'figures/unconditional_rod_lag_distribution');pd.DataFrame(rows).to_csv(out/'histogram_support.csv',index=False)


def main(out):
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    records=collect(out);refs,lags=patient();table,events=tabulate(records,out)
    core_contact_scatter(events,out);paired_probabilities(records,refs,out);unconditional_lags(events,lags,out)
    sel=table[(table.topology==2511)&table.noise.isin(SEEDS)&table.candidate.isin([x[0] for x in CASES])&(table.layer=='primary')&(table['mode']!='ALL')]
    sel.to_csv(out/'selected_core_mode_correspondence.csv',index=False)
    lo,hi=sel[sel['mode']=='TB'].B_t10_earlier_fraction.min(),sel[sel['mode']=='TB'].B_t10_earlier_fraction.max()
    text=f'''# 核内活动先后与患者传播恢复：两个不同环节

问题：当前TB不相似，是否仍是两核没有反向先后，还是已有相反活动先后但电极路径不对？本审计只读取冻结输出，不新仿真、不重标标签、不修改loss。去重后共{len(records)}条运行，历史复用按实际轨迹路径只计一次。

## 可支持的结论

在四个明确列出的工作点、网络2511、两条噪声中，全部合格TA事件的左核窗口t10均较早；TB中右核较早比例为{lo:.1%}–{hi:.1%}。因此当前模型已有明显的核内活动先后与模板标签对应，不能再笼统说完全没有相反时序。它不同于旧版本约五五开的历史审计，不能跨模型身份混用。

这还不是因果起源证据。t10是各核在原250ms窗内达到自身总发放10%的时间，可受小幅前驱、持续放电、窗口截断和递归招募影响；两核在窗口里的总发放份额也已列出。没有通过干预证明一个核驱动另一个核。

更关键的是：左核较早的TA仍可能出现SCL反而晚约92ms，改变连接后才变为较早。TB多数右核较早时，跨杆质心差仍常在+35ms附近，ICL上端的条件先后不符。核先后对应与患者完整路径恢复是两个不同验收环节。

不按TA/TB分类的全部事件直方图仍显示模型在少数杆间时差附近集中，所以不能把该时间残差完全解释为分类器分组制造的现象。直方图只验证这一观测层，不自动证明传播速度、起源或内部机制。全部触点对矩阵和实际支持量一同交付，不靠单个示例判断。

## 比较前提与下一步

每格为一条60秒运行，前1.5秒排除；事件是运行内样本，不做事件伪重复显著性。患者来自冻结FIT自然分布。图中的标签由接触表征分配，未使用核t10，但也不能算独立于接触时序的验证。core/t10与接触质心不是相同物理量；完整原生场GIF仍需结合查看。

先等待新噪声位置对照和后续网络重演，判断核间先后、参与和具体路径的对应是否保留。若核间相反先后可重演而TB路径残差不变，后续物理探针应检验向外连接的空间分配/分支招募，不能仅因TA改善而继续同向增加总兴奋量。该方向是残差指导的待检假设，尚未另行增加物理条件。
'''
    (out/'scientific_note.md').write_text(text)
    captions={'core_timing_contact_timing.png':'核窗口t10之差与接触杆间质心差的逐事件联合散点，四个条件各两次噪声。**关注点**：核先后对应并不保证电极顺序正确，t10不等于因果起点。','contact_order_probability.png':'全部触点对的条件先后概率，患者与模型采用固定分杆15行，同一候选两噪声分别展示。**关注点**：具体残差和共同参与支持，不能把无支持当零概率。','unconditional_rod_lag_distribution.png':'不区分TA/TB的跨杆时差分布，固定4ms分箱和总可估计事件分母。**关注点**：分类标签之外是否仍存在时差分布偏移和过度集中。'}
    (out/'figures/README.md').write_text('# 核时序与接触路径审计\n\n'+''.join('### '+k+'\n'+v+'\n\n' for k,v in captions.items()))
    with PdfPages(out/'core_timing_and_tb_routes.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.065,.91,'核间先后已出现，患者TB路径仍待恢复',fontsize=19)
        fig.text(.065,.81,f'冻结输出审计：{len(records)}条唯一运行；重点图为四个工作点×同一网络×两条噪声。\n\nTA均伴左核窗口t10较早；TB中右核较早占{lo:.1%}–{hi:.1%}。\n但TA曾在相同核先后下SCL仍晚约92ms；TB仍常晚约35ms。\n\n核活动先后、模板标签、接触路径和因果传播需分别解释。\n不分TA/TB的全部事件时差分布也有残差，不只依赖分类条件。\n\n每次运行是复制单位，图上事件不是独立网络。\n窗口t10不是起源证明；全部接触对支持量保留，原生GIF需联合审阅。\n\n完整前提和逐运行数据见scientific_note.md及CSV。',va='top',fontsize=12,linespacing=1.8);pdf.savefig(fig);plt.close(fig)
        for file in captions:
            with Image.open(out/'figures'/file) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=plt.figure(figsize=(17,17*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=140);plt.close(fig)
    print(str(out/'scientific_note.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)

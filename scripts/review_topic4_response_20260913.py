"""Immutable completed-140 and ongoing-followup scientific snapshot; no simulation."""
from pathlib import Path
import sys,json,datetime,hashlib,shutil
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
P=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
F=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
O=Path('/data/hfosp/topic4_sef_hfo/core_response_review_20260913');FIG=O/'figures'
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3,'axes.spines.right':False,'axes.spines.top':False})
CAP={};SEEDS=[847101,847102]


def snapshot():
    O.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    if not (O/'snapshot.json').exists():
        meta=dict(time=datetime.datetime.now().astimezone().isoformat(),source_script=str(Path(__file__).resolve()),sources={})
        for label,base in [('completed_140',P),('followup',F)]:
            records=[rt.read(p) for p in sorted((base/'analysis/units').glob('*/result.json'))]
            for k in ['counts','observations','contacts','pairs','events','segments']:
                rows=[z for r in records for z in (r[k] if isinstance(r[k],list) else [r[k]])];pd.DataFrame(rows).to_csv(O/(label+'_'+k+'.csv'),index=False)
            meta['sources'][label]=dict(status=rt.read(base/'status.json'),analyzed_runs=len(records),trajectories=[dict(source=r['source'],arrays_sha256=r['source_sha256']) for r in records])
        shutil.copy2(P/'analysis/patient_reference.json',O/'patient_reference.json');rt.write(O/'snapshot.json',meta)
    tables={label:{k:pd.read_csv(O/(label+'_'+k+'.csv')) for k in ['counts','observations','pairs']} for label in ['completed_140','followup']}
    return rt.read(O/'snapshot.json'),tables,rt.read(O/'patient_reference.json')


def save(fig,name,caption):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=165,bbox_inches='tight')
    plt.close(fig);CAP[name+'.png']=caption


def setup_plot(title,cols=3):
    fig,axs=plt.subplots(2,cols,figsize=(4.2*cols+1,7.7));fig.subplots_adjust(left=.075,right=.99,top=.79,bottom=.16,hspace=.38,wspace=.30)
    fig.suptitle(title,fontsize=14,y=.99);return fig,axs


def annotate(ax,key,ref,mode):
    v=ref['modes'][mode]['n']/ref['fit_n'] if key=='fraction' else (0 if key=='pair_order_probability_mae' else ref['modes'][mode][key])
    ax.axhline(v,c='black',ls=':',lw=1.3);ax.text(.99,v,('完全匹配 0' if key=='pair_order_probability_mae' else f'患者 {v:.2f}'),transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
    if key in ['fraction','SCL_upper_participation']:ax.set_ylim(-.025,1.08)
    elif key=='SCL_minus_ICL_lag_median_ms':ax.set_ylim(-65,115);ax.axhline(0,c='#bbb',lw=.5)
    else:ax.set_ylim(-.02,.45)
    ax.grid(axis='y',alpha=.17);ax.set_title(mode,loc='left',fontweight='bold')


def result_value(d,count,cid,seed,mode,key):
    if key=='fraction':
        z=count.loc[(cid,seed)];return z[mode]/z.primary
    z=d[(d.candidate==cid)&(d.noise==seed)&(d['mode']==mode)]
    return float(z.iloc[0][key]) if len(z) else np.nan


def factorial(t,ref):
    plan=rt.read(F/'plan.json');cs={c['id']:c for c in plan['candidates']};mapping={}
    for cid,c in cs.items():
        if c.get('factorial'):
            x=c['factorial'];mapping[(x['shape'],x['EE_out'],x['EI'])]=cid
    d=t['observations'].query("topology==2511 and layer=='primary'");count=t['counts'].query('topology==2511').set_index(['candidate','noise'])
    metrics=[('SCL_upper_participation','SCL9/8 平均参与概率'),('SCL_minus_ICL_lag_median_ms','SCL − ICL 质心时差 (ms)'),('fraction','本类 / 合格事件')]
    for shape in ['circle','ellipse4']:
        fig,axs=setup_plot(('圆核' if shape=='circle' else '左核椭圆4:1')+'：连接参数有相互作用，TB 时差尚未恢复')
        for i,mode in enumerate(['TA','TB']):
         for ax,(key,label) in zip(axs[i],metrics):
            for ei,col in [(1,'#777777'),(.875,'#3478b8'),(.75,'#b44b69')]:
             for seed,ls,mk in zip(SEEDS,['-','--'],['o','^']):
                xs=[1,1.125,1.25];ys=[result_value(d,count,mapping[(shape,x,ei)],seed,mode,key) for x in xs]
                ax.plot(xs,ys,c=col,ls=ls,marker=mk,lw=1.5,ms=5)
            annotate(ax,key,ref,mode);ax.set(xlabel='核向外 E→E 权重倍率',ylabel=label,xticks=[1,1.125,1.25])
        fig.legend(handles=[Line2D([],[],c=c,label=f'核内 E→I ×{e:g}') for e,c in [(1,'#777777'),(.875,'#3478b8'),(.75,'#b44b69')]]+[Line2D([],[],c='#333',marker='o',label='噪声847101'),Line2D([],[],c='#333',ls='--',marker='^',label='噪声847102')],loc='upper center',bbox_to_anchor=(.5,.90),ncol=5,frameon=False,fontsize=9)
        fig.text(.075,.035,'固定同一网络2511和核位置；颜色=核内EI倍率，横轴=向外EE倍率，线型/点形=噪声。每点为独立60秒运行的统计。\n时差：每事件两杆内参与触点质心中位数之差，再取该类事件中位数；负值表示SCL较早。\n患者参考来自完整自然比例FIT。连线表示参数响应，不代表单个事件轨迹；只有一张网络，尚不能称为全局效应。',fontsize=9)
        save(fig,'factorial_'+shape,'固定同一拓扑和噪声的组合响应，灰/蓝/红表示核内EI倍率1/0.875/0.75，横轴为向外EE。虚线三角/实线圆点保留两次噪声。**关注点**：招募从不足到饱和，TA时序部分改善，但TB与患者的毫秒时差仍明显不符。')


def geometry(t,ref):
    base='up3__circle__EE_core_to_out_scale_1.25';ids=[base+'__x_minus075',base,base+'__x_plus075']
    d=t['observations'].query("topology==2511 and layer=='primary'");count=t['counts'].query('topology==2511').set_index(['candidate','noise'])
    metrics=[('SCL_upper_participation','SCL9/8 平均参与概率'),('SCL_minus_ICL_lag_median_ms','SCL − ICL 质心时差 (ms)'),('pair_order_probability_mae','成对顺序概率误差'),('fraction','本类 / 合格事件')]
    fig,axs=setup_plot('在向外EE增强25%的圆核工作点附近，左移有局部改善；TB仍有缺口',4)
    rows=[]
    for i,mode in enumerate(['TA','TB']):
     for ax,(key,label) in zip(axs[i],metrics):
        for seed,col,ls,mk in zip(SEEDS,['#7562a8','#19897f'],['-','--'],['o','^']):
            ys=[result_value(d,count,cid,seed,mode,key) for cid in ids];ax.plot([-.75,0,.75],ys,c=col,ls=ls,marker=mk,lw=1.8)
            rows.extend(dict(candidate=cid,noise=seed,mode=mode,metric=key,value=v) for cid,v in zip(ids,ys))
        annotate(ax,key,ref,mode);ax.set(xlabel='左核中心 x 偏移 (mm)',xticks=[-.75,0,.75],ylabel=label)
    fig.legend(handles=[Line2D([],[],c='#7562a8',marker='o',label='噪声847101'),Line2D([],[],c='#19897f',ls='--',marker='^',label='噪声847102'),Line2D([],[],c='black',ls=':',label='患者FIT参考')],loc='upper center',bbox_to_anchor=(.5,.89),ncol=3,frameon=False)
    fig.text(.075,.035,'同一网络2511、两次噪声分别配对；y、半径、阈值和连接参数不变，但平移会改变实际成员与邻接。\n顺序误差：两触点共同参与时，先后概率与患者之差的绝对值，再在可比较触点对上等权平均；0只表示该摘要完全匹配。\n这些是开发数据中的响应，不是已确认的最优位置。下一批将用两张新网络和两条新噪声复测左移及直接对照。',fontsize=9)
    pd.DataFrame(rows).to_csv(O/'x_shift_response.csv',index=False)
    save(fig,'left_core_position_response','紫圆实线和绿三角虚线分别为同一网络下两条配对噪声；0是已增强向外EE的直接对照。四项观测分别显示，不合为总体恢复率。**关注点**：左移改善TA参与和顺序，却没有解决TB的跨杆时差。')


def radius_confirmation(t,ref):
    cs=t['counts'].set_index(['candidate','topology','noise']);d=t['observations'].query("layer=='primary'").set_index(['candidate','topology','noise','mode'])
    ids=['up3__circle','up3__radius25','up3__radius25_dose_matched'];combos=[(2611,847201),(2611,847202),(2612,847201),(2612,847202)]
    fig,axs=setup_plot('完整新拓扑确认：扩大半径的收益依赖阈值总量，仍不能等同于传播恢复')
    rows=[]
    for i,mode in enumerate(['TA','TB']):
     for ax,(key,label) in zip(axs[i],[('SCL_upper_participation','SCL9/8 平均参与概率'),('SCL_minus_ICL_lag_median_ms','SCL − ICL 质心时差 (ms)'),('fraction','本类 / 合格事件')]):
        for topo,seed in combos:
            ys=[]
            for cid in ids:
                z=cs.loc[(cid,topo,seed)] if key=='fraction' else d.loc[(cid,topo,seed,mode)];v=z[mode]/z.primary if key=='fraction' else z[key];ys.append(v)
                rows.append(dict(candidate=cid,topology=topo,noise=seed,mode=mode,metric=key,value=v))
            ax.plot(range(3),ys,c='#3478b8' if topo==2611 else '#cf7b2b',ls='-' if seed%2 else '--',marker='o' if seed%2 else '^',lw=1.4)
        annotate(ax,key,ref,mode);ax.set(xticks=range(3),xticklabels=['原半径\n1.75mm','扩大半径\n2.5mm','扩大半径\n阈值总量匹配'],ylabel=label)
    fig.legend(handles=[Line2D([],[],c='#3478b8',label='网络2611'),Line2D([],[],c='#cf7b2b',label='网络2612'),Line2D([],[],c='#333',marker='o',label='噪声847201'),Line2D([],[],c='#333',ls='--',marker='^',label='噪声847202')],loc='upper center',bbox_to_anchor=(.5,.90),ncol=4,frameon=False)
    fig.text(.075,.035,'原140条中的4个确认单元，全部完成后纳入；每条线固定网络和噪声。横轴是离散干预，不是连续梯度。\n半径扩大同时改变易激神经元数、随机输入支持和连接分块；匹配阈值降低总量仍未匹配随机输入总量。\n少数模式事件数较少的运行仅显示观测支持，不把未出现或不稳定摘要直接判为机制不可能。',fontsize=9)
    pd.DataFrame(rows).to_csv(O/'radius_confirmation_by_run.csv',index=False);save(fig,'completed_radius_confirmation','蓝/橙为两个确认网络，线型/点形为两条噪声；每条线在三个离散干预间配对。左核大小变化不等于只改变一个总剂量。**关注点**：扩大core改善部分分数，并未稳定纠正TA/TB传播；匹配阈值总量后少数模式支持更少。')


def media(meta):
    cid='up3__circle__EE_core_to_out_scale_1.25__x_minus075';folder=FIG/'left_shift_same_network';folder.mkdir(exist_ok=True)
    source=F/'analysis/figures'/f'{cid}_topology2511'
    for p in source.glob('*.png'):shutil.copy2(p,folder/p.name)
    if (source/'manifest.json').exists():shutil.copy2(source/'manifest.json',folder/'source_manifest.json')
    # Use plain parameter names in the new comparison; source example selection
    # and channel order are inherited unchanged, not edited as raster pixels.
    rr=[r for r in meta['sources']['followup']['trajectories'] if '/'+cid+'/' in r['source'] and '/2511_' in r['source']]
    units={int(Path(z['source']).parents[1].name.split('_')[-1]):an.an.load_unit(Path(z['source']),1500.) for z in rr}
    c=rt.read(F/'candidates'/f'{cid}.json');c.update(topology=2511,display_name='圆核：向外E→E增强25%，左核向左移动0.75mm')
    title_manifest=an.figreview.spectral_comparison(c,units,sorted(units),folder,an.figreview.patient_payloads(),'primary');rt.write(folder/'comparison_manifest.json',title_manifest)
    if not (folder/'gif_manifest.json').exists():
        rr=[r for r in meta['sources']['followup']['trajectories'] if '/'+cid+'/' in r['source'] and '/2511_847101/' in r['source']][0]
        path=Path(rr['source']);r,a,ids=an.an.load_unit(path,1500.);c=rt.read(F/'candidates'/f'{cid}.json');c['topology']=2511;ap=rt.read(path.parents[1]/'applied_physics.json');c['_applied_threshold']=ap['threshold']
        from scripts.render_topic4_shape_output_gifs import render
        m=render(c,847101,r,a,ids,ap,folder,an.figreview.patient_payloads());rt.write(folder/'gif_manifest.json',m)
    (folder/'README.md').write_text('# 左移候选：同一网络与患者\n\n'+''.join(f'### {p.name}\n来自左移0.75mm条件、网络2511的真实轨迹，患者为固定Fig2C真实STFT，模型为发放密度包络。分杆固定15行、不改毫秒轴，示例取模型自身均值附近；GIF按时间选每类前三例及固定连续片段。\n**关注点**：训练分数降低是否对应两种传播都恢复；原生场全部E活动不得由lineage筛选隐藏。\n\n' for p in sorted(folder.iterdir()) if p.suffix in ['.png','.gif']))
    return sorted(folder.glob('*patient_spectra_model_envelopes.png'))


def main():
    meta,t,ref=snapshot();factorial(t['followup'],ref);geometry(t['followup'],ref);radius_confirmation(t['completed_140'],ref);images=media(meta)
    counts=t['completed_140']['counts'];follow=t['followup']['counts'];reuse={x['candidate'] for x in rt.read(F/'plan.json')['reuse']};nr=int(((follow.topology==2511)&follow.candidate.isin(reuse)).sum())
    diag=t['followup']['observations'].query("topology==2511 and layer=='primary'")
    keys=['L_search','primary','TA','TB'];means=follow.groupby('candidate')[keys].mean().sort_values('L_search')
    means['n_runs']=follow.groupby('candidate').size();means['scorable_runs']=follow.groupby('candidate').L_search.count()
    means['complete_two_noise_development_summary']=(means.n_runs==2)&(means.scorable_runs==2)
    means.to_csv(O/'followup_condition_means.csv')
    base='up3__circle__EE_core_to_out_scale_1.25';chosen=base+'__x_minus075'
    lines=[]
    for cid in [base,chosen]:
        z=means.loc[cid];ta=diag[(diag.candidate==cid)&(diag['mode']=='TA')];tb=diag[(diag.candidate==cid)&(diag['mode']=='TB')]
        lines.append(f'|{"原位置、向外EE增强25%" if cid==base else "同条件左移0.75mm"}|{z.L_search:.3f}|{ta.SCL_upper_participation.mean():.1%}|{ta.SCL_minus_ICL_lag_median_ms.mean():.2f}|{ta.pair_order_probability_mae.mean():.3f}|{tb.SCL_minus_ICL_lag_median_ms.mean():.2f}|')
    note=f'''# 结果报告：完整140条与当前续跑

快照时间：{meta['time']}。原批次140/140全部完成，合格事件{int(counts.primary.sum()):,}；物理状态为{counts.physical_status.value_counts().to_dict()}。续跑已分析{len(follow)-nr}条新运行＋{nr}条复用直接对照；未完成条件不填值、不混作新重复。

## 当前判断

参数—观测关系已得到具体响应证据，但患者两种传播模式尚未完整恢复。可以推进配对复测与细化曲线，不能将低分候选称为机制恢复或冻结Fig4/5基底。下一批用户已授权，正式108条将在当前续跑物理运行完成后自动接上。

本次优先展示两类模式的参与、时差、顺序和比例。患者FIT共{ref['fit_n']:,}个事件，TA{ref['modes']['TA']['n']:,}、TB{ref['modes']['TB']['n']:,}；样本是事件，但参数效应的实验单位是固定网络×噪声的一次60秒运行。排除前1.5秒，患者和模型沿用原资格与标签表征。

## 三个可据数据讨论的结果

1. **几何微调有局部收益。** 同一网络2511、两次噪声等权，左移使TA上部SCL参与和顺序同时改善；TB时差基本没有接近患者约+1.19ms。以下为两次运行统计量的平均，不是事件混池，也不是置信区间。

|条件|冻结训练分数↓|TA上部SCL参与|TA跨杆时差ms|TA顺序概率误差↓|TB跨杆时差ms|
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

2. **EE与EI存在相互作用。** 固定形状、拓扑、噪声，减弱核内E→I与增加向外E→E可把SCL参与从不足推到接近100%。是否有利必须同时看患者TA约84.8%、TB约75.4%的上部SCL参与及毫秒时差，不能以参与越多越好。这组因子网格全部完成，但只有一张网络；尚不能宣称普遍或显著交互作用。TB时差仍集中在约+34至+36ms附近。

3. **完整确认不支持把扩大core当作解决方案。** 两张确认网络×两噪声中，圆形基线、2.5mm半径、匹配降阈值总量的2.5mm半径，平均训练分数依次为{counts[counts.candidate.eq('up3__circle') & counts.stage.eq('confirmation')].L_search.mean():.3f}、{counts[counts.candidate.eq('up3__radius25') & counts.stage.eq('confirmation')].L_search.mean():.3f}、{counts[counts.candidate.eq('up3__radius25_dose_matched') & counts.stage.eq('confirmation')].L_search.mean():.3f}。较低分不是两模式传播恢复；匹配阈值总量后TA每条仅{counts[counts.candidate.eq('up3__radius25_dose_matched') & counts.stage.eq('confirmation')].TA.tolist()}例。半径扩大还改变噪声支持人数和连接分块，所以不能由这三点唯一归因于某个生物机制。

## 图件与剩余缺口

图中明确颜色和噪声，TA/TB分行，患者参考为完整自然分布。黑底对照按杆固定15行，患者为实际STFT，模型是发放密度包络，二者不宣称物理信号等价；原生场GIF包含全部E活动。位置候选的好分数不能消除TB折返及时间过程的缺口。局部宽度、空间面积、core先后、并行活动和旋转均在原始逐事件表保留；旋转为操作性候选，可能有多波源叠加假阳性。

具体到左移候选：TB中ICL11质心早于ICL9的共同参与条件概率为20.8%/30.3%，患者为69.4%；直接对照为21.3%/13.0%，说明这一局部顺序改善也尚未跨两条噪声重复。两次噪声的TA/TB局部宽度中位数均仍为16ms；TA招募跨度42ms，TB约78ms。原生场抽查仍有多个并行亮区，不能将这次参与和分数改善表述为单一连续前沿已恢复。

全部局部参数单干预仍见[上一份参数总表]({P}/scientific_review_20260912/figures/all_local_parameters.png)，其56条响应已完整，本次没有重定义。当前快照的全部counts、observations、contacts、pairs、events、segments分表可独立复核。

## 继续执行的界限

见[下一批合同](/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/execution_plan.md)：18条件×3网络×2新噪声=108条。三条细曲线用于区分单点偶然、参数过渡和网络依赖；左移与增边/增权直接对照保持。固定损失和限核随机输入，Z/M与空间OU关闭，不为了获得好结果更换资格或加入指定TB路线。完整运行后停在科学审阅点。

本报告及图已由Agent检查后交付；用户人工验图尚未完成。
'''
    (O/'scientific_report.md').write_text(note)
    (FIG/'README.md').write_text('# 参数与传播结果图\n\n'+''.join(f'### {n}\n{c}\n\n' for n,c in CAP.items())+'\n同一网络黑底/原生场见left_shift_same_network/README.md。\n')
    with PdfPages(O/'result_report.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.91,'完整140条与续跑：当前结论',fontsize=20)
        text=f'已完成140/140，合格事件{int(counts.primary.sum()):,}。续跑快照：{len(follow)-nr}条新运行＋{nr}条复用。\n\n参数—观测响应成立；患者双模式恢复尚未成立。\n\n左移0.75mm改善TA参与和顺序，但TB跨杆时差仍约35ms，患者约1ms。\nEE×EI组合从漏招募走向过度招募，值得在中间区间补点，并跨网络复测。\n扩大core可降低部分训练分数，但不能由此认定两模式恢复。\n\n每条线固定网络和噪声；每点为60秒运行统计。标签比例与模式内分布分开。\n患者黑底为真实STFT，模型为发放包络；原生场GIF另列，15行按杆固定。\n\n下一批：18条件×3网络×2新噪声=108条，当前续跑结束后自动接上。\n保持既定阈值、噪声和loss定义；不自动冻结模型或进入Fig5。\n\n完整数值、前提和来源：同目录scientific_report.md与逐运行CSV。'
        fig.text(.07,.82,text,va='top',fontsize=12,linespacing=1.8);pdf.savefig(fig);plt.close(fig)
        for p in [FIG/(n) for n in CAP]+images:
            with Image.open(p) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=plt.figure(figsize=(14,14*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=140);plt.close(fig)
    rt.write(O/'report_manifest.json',dict(pdf_sha256=rt.sha(O/'result_report.pdf'),figures={str(p.relative_to(O)):rt.sha(p) for p in FIG.rglob('*.png')},created_unix=__import__('time').time()))
    print(str(O/'scientific_report.md'))


if __name__=='__main__':main()

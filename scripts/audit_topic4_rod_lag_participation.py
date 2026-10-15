"""Separate contact-composition effects from timing residuals on frozen outputs.

Every model event is crossed with every patient event in the specified comparison.
This is a descriptive common-contact calculation, not event matching or new loss.
"""
from pathlib import Path
import argparse, datetime, json, sys, warnings
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scripts import audit_topic4_core_timing_and_tb_routes as source
rt=source.rt
MODES=[('ALL',None),('TA',1),('TB',0)]
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3,'axes.spines.top':False,'axes.spines.right':False})


def rod_lag(x,scl,icl):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(x[...,scl],axis=-1)-np.nanmedian(x[...,icl],axis=-1)


def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,median=None,q05=None,q95=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))


def common_contact_distribution(model,patient,scl,icl):
    """Exact empirical cross product in small model batches; never nearest matching."""
    ml=rod_lag(model,scl,icl);pl=rod_lag(patient,scl,icl)
    arrays={k:[] for k in ['model_original_restricted','patient_original_restricted','model_common','patient_common']}
    support=[]
    for start in range(0,len(model),8):
        m=model[start:start+8];mask=np.isfinite(m[:,None,:])&np.isfinite(patient[None,:,:])
        ok=mask[...,scl].any(-1)&mask[...,icl].any(-1)
        mv=rod_lag(np.where(mask,m[:,None,:],np.nan),scl,icl)
        pv=rod_lag(np.where(mask,patient[None,:,:],np.nan),scl,icl)
        arrays['model_common'].append(mv[ok]);arrays['patient_common'].append(pv[ok])
        arrays['model_original_restricted'].append(np.broadcast_to(ml[start:start+len(m),None],ok.shape)[ok])
        arrays['patient_original_restricted'].append(np.broadcast_to(pl[None,:],ok.shape)[ok])
        support.extend(mask[ok].sum(-1).tolist())
    arrays={k:np.concatenate(v) if v else np.array([]) for k,v in arrays.items()}
    out={k:stats(v) for k,v in arrays.items()}
    out.update(model_original=stats(ml),patient_original=stats(pl),model_events=len(model),patient_events=len(patient),
               all_event_cross_pairs=len(model)*len(patient),common_valid_pairs=len(arrays['model_common']),common_contact_count=stats(support))
    for key,m,p in [('original_gap','model_original','patient_original'),('restricted_gap','model_original_restricted','patient_original_restricted'),('common_gap','model_common','patient_common')]:
        out[key]=None if out[m]['mean'] is None or out[p]['mean'] is None else out[m]['mean']-out[p]['mean']
    if out['common_gap'] is not None:
        out['support_selection_shift']=out['restricted_gap']-out['original_gap']
        out['contact_set_shift']=out['common_gap']-out['restricted_gap']
        assert abs(out['common_gap']-(out['original_gap']+out['support_selection_shift']+out['contact_set_shift']))<1e-8
    return out


def pairs(x,names):
    rows=[]
    for i in range(len(names)):
      for j in range(i+1,len(names)):
        valid=np.isfinite(x[:,i])&np.isfinite(x[:,j]);s=stats(x[valid,j]-x[valid,i])
        rows.append(dict(contact_i=names[i],contact_j=names[j],group='跨杆' if names[i].startswith('SCL')!=names[j].startswith('SCL') else 'SCL内' if names[i].startswith('SCL') else 'ICL内',**s))
    return rows


def load_models(records,names):
    wanted={c for c,_ in source.CASES}
    out=[]
    for r in records:
        c=r['counts']
        if c['candidate'] not in wanted or c['topology']!=2511 or c['noise'] not in source.SEEDS:continue
        path=Path(r['source']);meta=rt.read(path)
        with np.load(path.with_suffix('.npz')) as z:
            order=z['contact_names'].astype(str);idx=[list(order).index(n) for n in names]
            a={k:z[k] for k in ['centroid_ms','primary_event_indices','event_mode']}
        ids=source.an.an.analysis_ids(meta,a,1500.)
        out.append((c,a['centroid_ms'][ids][:,idx],a['event_mode'][ids]))
    assert len(out)==8,len(out)
    return out


def composition_plot(table,out):
    fig,axes=plt.subplots(3,4,figsize=(16.8,10.8));fig.subplots_adjust(left=.06,right=.99,top=.84,bottom=.15,wspace=.35,hspace=.66)
    keys=['original_gap','restricted_gap','common_gap'];colors=['#8965a5','#258d96']
    for row,(mode,_) in enumerate(MODES):
      for col,(cid,label) in enumerate(source.CASES):
        ax=axes[row,col]
        for seed,color in zip(source.SEEDS,colors):
            d=table[(table.candidate==cid)&(table.noise==seed)&(table['mode']==mode)].iloc[0]
            ax.plot(range(3),[d[k] for k in keys],'-o',color=color,lw=1.3,ms=4,label=str(seed))
        ax.axhline(0,c='#666',lw=.7,ls=':');ax.set(xticks=range(3),xticklabels=['原指标','保留共同支持','统一接触点'],ylim=(-50,110),ylabel='模型 − 患者均值 (ms)',title=f'{label}\n{mode}')
        ax.tick_params(axis='x',labelsize=8)
        if row==0:ax.legend(frameon=False,title='噪声种子',fontsize=7,title_fontsize=7)
    fig.suptitle('杆间延迟是否只是参与触点不同？先分开接触集合与剩余时序误差',fontsize=15,y=.985)
    fig.text(.06,.895,'全部事件交叉比较，无相似度配对。每对事件仅使用双方共同参与的触点计算SCL−ICL；中间一项保留相同有效事件对、仍用原触点。',fontsize=10)
    fig.text(.06,.04,'每条线为同一网络的一条60秒运行；各栏为观测计算方式，不是三次物理干预。用均值使接触集合与支持选择的改变量可以相加，\n不称为总体方差解释率。未共同支持的事件对保留数量；共同触点减少不等于参与问题修复。TA/TB仅组织比较，ALL不分标签。\n原分布及共同触点分布的中位数和5–95%范围同时见CSV；完整触点对时差图用于检查固定空间位置上的剩余偏差。',fontsize=9)
    source.save(fig,out/'figures/common_contact_mean_gap')


def pair_plot(pair_table,out):
    order=source.ORDER;spec=[('patient',None,'患者FIT')]+[(cid,seed,('原位置' if cid==source.CASES[1][0] else '左移')+f'／{seed}') for cid in [source.CASES[1][0],source.CASES[2][0]] for seed in source.SEEDS]
    fig,axes=plt.subplots(3,5,figsize=(19.5,12));fig.subplots_adjust(left=.05,right=.94,top=.87,bottom=.13,wspace=.35,hspace=.63)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d0d0d0');clipped=[]
    for row,(mode,_) in enumerate(MODES):
      for col,(cid,seed,label) in enumerate(spec):
        d=pair_table[(pair_table.candidate==cid)&(pair_table['mode']==mode)]
        if seed is not None:d=d[d.noise==seed]
        mat=np.full((15,15),np.nan)
        for z in d.to_dict('records'):
            i,j=order.index(z['contact_i']),order.index(z['contact_j']);mat[i,j]=z['median'];mat[j,i]=-z['median']
            if abs(z['median'])>160:clipped.append(z)
        ax=axes[row,col];im=ax.imshow(mat,cmap=cmap,vmin=-160,vmax=160,interpolation='nearest')
        ax.set(xticks=range(15),xticklabels=order,yticks=range(15),yticklabels=order,title=f'{label}｜{mode}')
        ax.tick_params(axis='x',labelrotation=90,labelsize=6);ax.tick_params(axis='y',labelsize=6);ax.axhline(3.5,c='black',lw=.6);ax.axvline(3.5,c='black',lw=.6)
    cax=fig.add_axes([.955,.23,.012,.5]);fig.colorbar(im,cax=cax,label='列接触点 − 行接触点质心：中位数 (ms)')
    fig.suptitle('固定触点对的毫秒时差：每格比较相同的两个接触点',fontsize=15,y=.985)
    fig.text(.05,.92,'固定分杆15行／列；红=列较晚，蓝=列较早。只纳入该对共同参与事件，灰表示对角或无支持；不同格的事件集合可不同。',fontsize=10)
    fig.text(.05,.035,'患者与模型没有按距离挑选事件、没有时间拉伸。模型格各自为一次运行，不以事件数量替代网络重复；共同参与数及5–95%范围见contact_pair_lag_distributions.csv。\n该图避免将“杆中位数使用不同触点”直接当作时序效应，但仍是条件分布比较，不能排除参与条件的潜在事件选择，也不等于因果连接。',fontsize=9)
    source.save(fig,out/'figures/fixed_contact_pair_lags');rt.write(out/'color_saturation.json',dict(limit_ms=160,saturated_pair_entries=clipped))


def main(out):
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    records=source.collect(out);p=rt.read(source.an.run.base.PARENT);ev=rt.load_evaluator(p)
    patient=np.asarray(ev.fit);plabel=np.asarray(ev.fit_labels);names=np.asarray(rt.load_observation_contract(p)['contact_names'])
    scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL')
    models=load_models(records,names);rows=[];dists=[];pairrows=[]
    for mode,label in MODES:
        ref=patient if label is None else patient[plabel==label]
        for z in pairs(ref,names):pairrows.append(dict(candidate='patient',topology=None,noise=None,mode=mode,**z))
        for c,x,l in models:
            m=x if label is None else x[l==label];identity={k:c[k] for k in ['candidate','topology','noise']};identity['mode']=mode
            result=common_contact_distribution(m,ref,scl,icl)
            for k,v in result.items():
                if isinstance(v,dict):dists.append(dict(**identity,distribution=k,**v))
            rows.append(dict(**identity,**{k:v for k,v in result.items() if not isinstance(v,dict)}))
            for z in pairs(m,names):pairrows.append(dict(**identity,**z))
            print(json.dumps(dict(**identity,original_gap=result['original_gap'],common_gap=result['common_gap']),ensure_ascii=False),flush=True)
    table=pd.DataFrame(rows);dist=pd.DataFrame(dists);pair_table=pd.DataFrame(pairrows)
    table.to_csv(out/'common_contact_gap_decomposition.csv',index=False);dist.to_csv(out/'rod_lag_distribution_summaries.csv',index=False);pair_table.to_csv(out/'contact_pair_lag_distributions.csv',index=False)
    composition_plot(table,out);pair_plot(pair_table,out)
    note='''# 杆间时差与接触参与集合：冻结输出分解

问题：模型TB的SCL−ICL质心差约35ms，而患者约1ms；这个差距是否部分由双方参与接触点不同造成？本审计不更改物理、loss、事件资格或标签。原中位数指标仍保留；这里引入均值只为了使计算方式导致的改变量可加，不把它称为Overall恢复率。

每个模型事件与该模式下每个患者事件交叉。保留双方共同触点中SCL、ICL至少各一个的事件对。在同一有效事件对集合上，先使用各自原参与触点算杆间差，再使用双方共同触点算杆间差。不存在寻找最相似患者、重新调时间轴或按结果选择配对。ALL交叉全部事件，不区分标签。跨乘产生的事件对不是新增独立样本，不作显著性或机制估计。

差距分成三个可核查数值：原始模型均值减患者均值；限制到共同支持事件对后的均值差；相同事件对上统一接触点后的均值差。第二减第一是支持选择效应，第三减第二是接触集合改变效应。它们只解释该观测计算的敏感性，不是因果归因；减少接触点后更接近患者，也不能抵消模型参与分布不正确。

contact_pair_lag_distributions.csv另列全部105对、ALL/TA/TB、每条运行的共同参与数、均值、中位数及5–95%范围。固定触点对时差矩阵比较相同两个空间位置，可检查不依赖杆内触点构成的剩余时序偏差。每格仍条件于双方参与，不能认为已经控制了所有潜在事件类型。

患者为冻结FIT自然分布，TA13,165/TB6,605；模型来自四个固定工作点×网络2511×两次噪声的全部合格事件，排除前1.5秒。当前结果属于开发审计，不是新拓扑确认或独立患者验证。患者HFO质心与模型发放包络质心的物理差别仍保留。
'''
    (out/'scientific_note.md').write_text(note)
    (out/'figures/README.md').write_text('''# 接触集合与时序残差

### common_contact_mean_gap.png
四个条件、两条噪声，分别展示原始杆间均值差、相同有效事件对上的差、统一接触点后的差。每条线为一条运行；三点是计算方式变化，不是仿真参数变化。
**关注点**：参与集合是否改变时差解释，残差是否在统一接触点后仍保留。

### fixed_contact_pair_lags.png
固定分杆15行，患者与原位置／左移两次噪声逐格比较固定触点对的质心时差中位数。无参与支持用灰色，实际支持及范围在CSV。
**关注点**：避免把杆内参与构成变化当作速度效应；仍需注意每格条件事件不同。
''')
    with PdfPages(out/'rod_lag_participation_audit.pdf') as pdf:
        for name in ['common_contact_mean_gap.png','fixed_contact_pair_lags.png']:
            with Image.open(out/'figures'/name) as im:a=np.asarray(im.convert('RGB'))
            h,w=a.shape[:2];fig=plt.figure(figsize=(17,17*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(a);ax.axis('off');pdf.savefig(fig,dpi=140);plt.close(fig)
    rt.write(out/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',new_simulations=0,model_runs=8,comparison_rows=len(table),pair_rows=len(pair_table),created=datetime.datetime.now().astimezone().isoformat(),producer=str(Path(__file__).resolve()),producer_sha256=rt.sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)

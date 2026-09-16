"""Describe all 14 predeclared local perturbations; no new score or simulation.

The unit is a paired 60 s trajectory, conditional event counts remain explicit.
This is a fixed completed-140 snapshot, not a replacement for live N108 curves.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from src.topic4_pdf_font_guard import install

P=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
O=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/completed_parameter_effects')
F=O/'figures'
LABELS={
 ('EE_same_core_scale',.75):'两核内 E→E：0.85→0.75',
 ('EE_same_core_scale',1.):'两核内 E→E：0.85→1',
 ('EE_core_to_out_scale',.75):'两核向外 E→E：1→0.75',
 ('EE_core_to_out_scale',1.25):'两核向外 E→E：1→1.25',
 ('EI_same_core_scale',.75):'两核内 E→I：1→0.75',
 ('EI_same_core_scale',1.25):'两核内 E→I：1→1.25',
 ('IE_same_core_scale',1.25):'两核内 I→E：1→1.25',
 ('II_same_core_scale',.75):'两核内 I→I：1→0.75',
 ('depth_A_scale',.75):'左核阈值降低量：1→0.75',
 ('depth_A_scale',1.25):'左核阈值降低量：1→1.25',
 ('core_mean_rate_scale',1.):'两核平均外部输入：0.95→1',
 ('core_ou_correlation',.5):'两核OU相关系数：0→0.5',
 ('core_ou_correlation',1.):'两核OU相关系数：0→1',
 ('EE_core_to_out_degree_scale',1.5):'两核向外 E→E 入边数：1→1.5',
}
GROUPS={
 'contact_response':[
  ('SCL_upper_participation','SCL9/8参与\n变化（百分点）',100),
  ('ICL_contact_participation','ICL平均参与\n变化（百分点）',100),
  ('SCL_minus_ICL_lag_median_ms','SCL−ICL质心时差\n中位数变化（ms）',1),
  ('pair_order_probability_mae','成对顺序概率误差\n变化（百分点）',100),
  ('fraction','该类事件比例\n变化（百分点）',100)],
 'native_response':[
  ('local_width_ms_median','接触局部宽度\n中位数变化（ms）',1),
  ('recruitment_span_ms_median','跨触点t10跨度\n中位数变化（ms）',1),
  ('B_minus_A_t10_ms_median','右核−左核t10\n中位数变化（ms）',1),
  ('union_active_area_mm2_median','活动并集面积\n中位数变化（mm²）',1),
  ('peak_largest_component_fraction_median','峰值最大连通域份额\n中位数变化（百分点）',100)]}

def collect():
    plan=json.loads((P/'plan.json').read_text());counts=pd.read_csv(P/'analysis/counts.csv');obs=pd.read_csv(P/'analysis/observations.csv')
    assert len(counts)==140 and not counts.duplicated(['candidate','topology','noise']).any()
    counts=counts.set_index(['candidate','topology','noise']);obs=obs.query("layer=='primary'").set_index(['candidate','topology','noise','mode'])
    rows=[]
    for j,(key,new) in enumerate(plan['response']['axes']):
      for shape in ['circle','ellipse4']:
        base='up3__'+shape;cid=base+'__'+key+'_'+str(new)
        # Use the actual stored string representation, not inferred formatting.
        candidate=json.loads((P/'candidates'/f'{cid}.json').read_text())
        assert candidate['comparison']==base
        for noise in plan['seeds']:
          count=counts.loc[(cid,2511,noise)];bc=counts.loc[(base,2511,noise)]
          for mode in ['ALL','TA','TB']:
            r=obs.loc[(cid,2511,noise,mode)];b=obs.loc[(base,2511,noise,mode)]
            for metric,label,scale in sum(GROUPS.values(),[]):
              x=(np.nan if not count.primary else (1. if mode=='ALL' else count[mode]/count.primary)) if metric=='fraction' else r[metric]
              y=(np.nan if not bc.primary else (1. if mode=='ALL' else bc[mode]/bc.primary)) if metric=='fraction' else b[metric]
              rows.append(dict(axis_index=j,parameter=key,new_parameter_value=new,label=LABELS[(key,new)],shape=shape,candidate=cid,reference=base,topology=2511,noise=noise,mode=mode,metric=metric,display_scale=scale,value=x,reference_value=y,difference=x-y,model_n=int(r['n']),reference_n=int(b['n']),model_total_n=int(count.primary),reference_total_n=int(bc.primary)))
    d=pd.DataFrame(rows);d.to_csv(O/'paired_parameter_effects.csv',index=False)
    summary=[]
    for keys,x in d.groupby(['axis_index','parameter','label','shape','mode','metric']):
        assert len(x)==2
        a=dict(zip(['axis_index','parameter','label','shape','mode','metric'],keys));v=x.difference.dropna().to_numpy()
        a.update(noise_runs=len(v),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),mean_change=float(v.mean()) if len(v) else None,min_change=float(v.min()) if len(v) else None,max_change=float(v.max()) if len(v) else None,reference_mean=float(x.reference_value.mean()),new_mean=float(x.value.mean()),minimum_class_support=int(min(x.model_n.min(),x.reference_n.min())))
        summary.append(a)
    pd.DataFrame(summary).to_csv(O/'effect_summary.csv',index=False)
    return d,plan

def figures(d,plan):
    captions=[]
    for mode in ['ALL','TA','TB']:
      for group,metrics in GROUPS.items():
        metrics=[x for x in metrics if not (mode=='ALL' and x[0]=='fraction')]
        fig,axes=plt.subplots(1,len(metrics),figsize=(18,9),sharey=True,squeeze=False)
        fig.subplots_adjust(left=.24,right=.98,top=.79,bottom=.17,wspace=.3)
        subset=d[d['mode']==mode]
        for ax,(key,label,scale) in zip(axes[0],metrics):
            ax.axvline(0,c='#bbb',lw=.8,zorder=0)
            for shape,col,offset in [('circle','#3478b8',-.16),('ellipse4','#c17c37',.16)]:
              for row in range(14):
                z=subset[(subset.metric==key)&(subset['shape']==shape)&(subset.axis_index==row)].sort_values('noise')
                vals=z.difference.to_numpy()*scale
                yy=row+offset
                ax.plot(vals,[yy]*len(vals),c=col,alpha=.45,lw=1)
                for i,v in enumerate(vals):ax.plot(v,yy,marker='o' if i==0 else '^',mfc=col if i==0 else 'white',mec=col,ms=4.5,ls='none')
            limit=np.nanmax(np.abs(d.loc[d.metric==key,'difference'].to_numpy()*scale))
            ax.set_xlim(-max(limit*1.15,1e-3),max(limit*1.15,1e-3));ax.set_title(label,fontsize=10,pad=12);ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',labelsize=8)
            ax.set_ylim(13.6,-.6);ax.set_yticks(range(14));ax.set_yticklabels([LABELS[tuple(x)] for x in plan['response']['axes']],fontsize=9)
            for row in range(14):
                vals=subset.loc[(subset.metric==key)&(subset.axis_index==row),'difference']
                if not np.isfinite(vals).any():ax.text(0,row,'不可估计',ha='center',va='center',color='#999',fontsize=7)
        handles=[Line2D([],[],c='#3478b8',label='圆核基底'),Line2D([],[],c='#c17c37',label='左核椭圆4:1基底'),Line2D([],[],c='#444',marker='o',ls='none',label='噪声847101'),Line2D([],[],c='#444',marker='^',mfc='white',ls='none',label='噪声847102')]
        fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.6,.915),frameon=False)
        fig.suptitle(f'14种参数干预各自改变了什么？｜{mode}｜同基础网络2511、同噪声配对',fontsize=16,y=.98)
        fig.text(.06,.035,'每点 = 改动后60秒运行统计 − 同形状直接对照；横线只连接两次噪声的差值，不代表置信区间。各列使用原始量纲，不合成总体恢复率。\n零表示无变化；参与上升、时差下降都不自动等于更接近患者。TA/TB按冻结分类器组织，比例改变与类内时序改变分开解释。\n共56条局部干预＋4条复用基底，全部来自原140条；事件嵌套于运行，每类实际事件数、原值及对照值保留在CSV。两种形状不是两张独立随机网络。\nOU输入仅在核内；全局EE轴、Z/M、空间OU、GABA18ms不变。核内连接参数作用于两个core；阈值深度参数只作用于左核，入边数量干预改变实际邻接。',fontsize=9)
        name=f'{mode}_{group}'
        for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=155)
        plt.close(fig)
        captions.append(f'### {name}.png\n逐行显示全部14个已预定局部干预，蓝/棕是圆核/左椭圆核基底，实圆/空三角为两次噪声。各点为固定基础拓扑及噪声下的改动后减对照，横线不是置信区间。\n**关注点**：同一参数是否在两种形状、两次噪声中方向一致；参与、类内传播与模式比例的变化必须分别判断。\n')
    (F/'README.md').write_text('# 完成批次的逐参数作用图\n\n'+'\n'.join(captions))

def main():
    O.mkdir(exist_ok=True);F.mkdir(exist_ok=True);install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3,'axes.spines.top':False,'axes.spines.right':False})
    d,plan=collect();figures(d,plan)
    with PdfPages(O/'parameter_effect_atlas.pdf') as pdf:
      for p in sorted(F.glob('*.png')):
        with Image.open(p) as im:arr=np.asarray(im.convert('RGB'))
        fig=plt.figure(figsize=(18,9));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=155);plt.close(fig)
    inputs={str(P/'analysis'/x):hashlib.sha256((P/'analysis'/x).read_bytes()).hexdigest() for x in ['counts.csv','observations.csv']}
    (O/'manifest.json').write_text(json.dumps(dict(inputs=inputs,producer=str(Path(__file__).resolve()),formal_new_simulations=0,paired_trajectory_units=56,unique_control_trajectory_units=4,conditional_labels='frozen; descriptive only',comparison='one perturbation versus its own shape baseline under identical base topology seed and noise seed'),indent=2))
    print(f'Wrote {len(d)} metric rows and six figures to {O}')

if __name__=='__main__':main()

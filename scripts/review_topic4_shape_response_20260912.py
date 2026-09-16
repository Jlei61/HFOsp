"""Read-only scientific snapshot of completed shape/connection responses.

Never invokes simulation or the streaming observer. Outputs live in a separate review
directory. Runs, rather than their individual events, are the paired experimental units.
"""
from pathlib import Path
import sys, json, hashlib, shutil, datetime, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scripts import analyze_topic4_shape_output_response as obs
from scripts import render_topic4_shape_output_gifs as gifs

P = obs.OUT
O = P/'scientific_review_20260912'
F = O/'figures'
rt, an, fr = obs.rt, obs.an, obs.figreview
SEEDS = [847101, 847102]
SHAPE_COLORS = ['#7540a5', '#bf870a']
CASES = {
    'up3__circle': '圆形基线',
    'up3__ellipse4': '左核改为椭圆',
    'up3__circle__EI_same_core_scale_0.75': '圆形＋核内 E→I 减25%',
    'up3__ellipse4__EI_same_core_scale_0.75': '椭圆＋核内 E→I 减25%',
    'up3__circle__EE_core_to_out_scale_1.25': '圆形＋向外 E→E 增25%',
    'up3__ellipse4__EE_core_to_out_scale_1.25': '椭圆＋向外 E→E 增25%',
    'up3__circle__EE_core_to_out_degree_scale_1.5': '圆形＋向外 E→E 边数增50%',
}
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
CAPTIONS = {}


def save(fig, name, caption):
    for ext in ['png','pdf']:
        fig.savefig(F/f'{name}.{ext}', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    CAPTIONS[name+'.png'] = caption


def snapshot():
    O.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
    snapshot_path = O/'snapshot.json'
    if snapshot_path.exists():
        meta = rt.read(snapshot_path)
    else:
        records=[]; sources=[]
        for path in sorted((P/'analysis/units').glob('*/result.json')):
            r=rt.read(path); rotpath=P/'rotation'/hashlib.sha256(r['source'].encode()).hexdigest()[:20]/'result.json'
            if rotpath.exists():
                rot=rt.read(rotpath)
                r['counts'].update(rotation_time_fraction=rot['candidate_time_fraction'],
                    max_fixed_ring_turns=rot['maximum_fixed_ring_turns'])
            records.append(r)
            sources.append(dict(result=str(path),result_sha256=rt.sha(path),trajectory=r['source'],arrays_sha256=r['source_sha256'],
                rotation=str(rotpath) if rotpath.exists() else None,rotation_sha256=rt.sha(rotpath) if rotpath.exists() else None))
        for key in ['counts','observations','contacts','pairs','events','segments']:
            rows=[item for r in records for item in (r[key] if isinstance(r[key],list) else [r[key]])]
            pd.DataFrame(rows).to_csv(O/(key+'.csv'),index=False)
        shutil.copy2(P/'analysis/patient_reference.json',O/'patient_reference.json')
        meta=dict(snapshot_time=datetime.datetime.now().astimezone().isoformat(),analyzed_runs=len(records),formal_budget=140,
            status=rt.read(P/'status.json'),sources=sources,selection='All completed immutable unit analyses at snapshot time; incomplete runs not scored or imputed.',
            source_script=str(Path(__file__).resolve()))
        rt.write(snapshot_path,meta)
    tables={k:pd.read_csv(O/(k+'.csv')) for k in ['counts','observations','contacts','pairs','events']}
    return meta,tables,rt.read(O/'patient_reference.json')


def interaction(t, ref, pathway, suffix, xvalues, name):
    d=t['observations']; d=d[(d.topology==2511)&(d.layer=='primary')]
    counts=t['counts']; counts=counts[counts.topology==2511].set_index(['candidate','noise'])
    keys=['SCL_upper_participation','SCL_minus_ICL_lag_median_ms','fraction']
    ylabs=['SCL9/8 平均参与概率','SCL − ICL 质心时差 (ms)','本类事件 / 全部合格事件']
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.5))
    fig.subplots_adjust(left=.075,right=.99,top=.80,bottom=.14,hspace=.38,wspace=.28)
    for row,mode in enumerate(['TA','TB']):
      for col,(key,ylab) in enumerate(zip(keys,ylabs)):
        ax=axes[row,col]
        pr=ref['modes'][mode]['n']/ref['fit_n'] if key=='fraction' else ref['modes'][mode][key]
        ax.axhline(pr,c='#222',ls=':',lw=1.5)
        ax.text(.99,pr,f'患者 {pr:.2f}',ha='right',va='bottom',transform=ax.get_yaxis_transform(),fontsize=8)
        for shape,color in zip(['circle','ellipse4'],SHAPE_COLORS):
          for s,ls,mk in zip(SEEDS,['-','--'],['o','s']):
            y=[]
            for tail in ['',suffix]:
                cid='up3__'+shape+tail
                if key=='fraction':
                    rr=counts.loc[(cid,s)]; value=rr[mode]/rr.primary
                else:
                    rr=d[(d.candidate==cid)&(d.noise==s)&(d['mode']==mode)].iloc[0];value=rr[key]
                y.append(value)
            ax.plot([0,1],y,c=color,ls=ls,marker=mk,lw=1.7,ms=5)
        ax.set(xticks=[0,1],xticklabels=xvalues,ylabel=ylab,xlim=(-.08,1.08))
        ax.set_title(mode,color=an.MODE_COLOR[mode],loc='left',fontweight='bold')
        if key!='SCL_minus_ICL_lag_median_ms':ax.set_ylim(0,1.08)
        else:ax.set_ylim(-35,115);ax.axhline(0,c='#999',lw=.6)
        ax.grid(axis='y',alpha=.15)
    handles=[Line2D([],[],c=SHAPE_COLORS[0],lw=2,label='左核圆形'),Line2D([],[],c=SHAPE_COLORS[1],lw=2,label='左核椭圆 4:1'),
        Line2D([],[],c='#555',marker='o',label='噪声 847101'),Line2D([],[],c='#555',ls='--',marker='s',label='噪声 847102'),Line2D([],[],c='#222',ls=':',label='患者 FIT 参考')]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.51,.89),frameon=False,fontsize=9)
    fig.suptitle(name+'\n同一拓扑 2511；两条噪声逐条配对；左核中心固定上移 3 mm，右核固定',fontsize=14,y=.985)
    fig.text(.07,.027,'时差：每事件两杆内参与触点质心的中位数之差，再取该类事件中位数；负值表示 SCL 较早。\n标签仅组织分布比较；每次运行 60 s，剔除前 1.5 s。线段连接参数条件，不代表追踪同一个事件。',fontsize=9)
    save(fig,pathway+'_shape_interaction',name+'。颜色区分左核形状，线型/点形区分两次噪声；每条线固定网络与噪声，点为运行内统计。患者参考来自全部自然比例 FIT 事件。**关注点**：TA参与和时差能否同时改善，TB与自然模式比例是否仍有残差。')


def distributions(t, ref):
    selected=['up3__circle','up3__circle__EI_same_core_scale_0.75','up3__ellipse4__EI_same_core_scale_0.75','up3__circle__EE_core_to_out_scale_1.25']
    colors=['#777777','#b479d1','#b38715','#388b78']
    labels=['患者 FIT','圆形基线','圆形\nE→I 减25%','椭圆\nE→I 减25%','圆形\n向外 E→E 增25%']
    ob=t['observations'];ob=ob[(ob.topology==2511)&(ob.layer=='primary')]
    fig,axes=plt.subplots(2,2,figsize=(14,8.2));fig.subplots_adjust(left=.06,right=.99,top=.86,bottom=.17,hspace=.36,wspace=.19)
    for row,mode in enumerate(['TA','TB']):
      for j,(family,lab) in enumerate([('SCL_minus_ICL_lag','SCL − ICL 质心时差 (ms)'),('local_width_ms','模型：单触点 10–90% 活动宽度 (ms)')]):
        ax=axes[row,j]
        if j==0:
            q=[ref['modes'][mode][family+'_'+p+'_ms'] for p in ['q05','median','q95']]
            ax.plot([0,0],[q[0],q[2]],c='black',lw=2);ax.scatter(0,q[1],c='black',s=45,zorder=3)
        else:
            ax.text(0,35,'未做信号等价\n不填患者值',ha='center',fontsize=8,color='#555')
        for k,(cid,color) in enumerate(zip(selected,colors),1):
          for s,off,mk in zip(SEEDS,[-.12,.12],['o','s']):
            r=ob[(ob.candidate==cid)&(ob.noise==s)&(ob['mode']==mode)].iloc[0]
            q=[r[family+'_'+p+'_ms'] if j==0 else r[family+'_'+p] for p in ['q05','median','q95']]
            x=k+off;ax.plot([x,x],[q[0],q[2]],c=color,lw=1.7);ax.scatter(x,q[1],c=color,marker=mk,s=30,zorder=3)
            if j==0:ax.text(x,q[2]+5,str(int(r.lag_joint_n)),color=color,ha='center',fontsize=7)
        ax.set(xticks=range(5),xticklabels=labels,ylabel=lab,title=mode,xlim=(-.45,4.45))
        ax.set_ylim((-60,160) if j==0 else (0,45));ax.grid(axis='y',alpha=.15)
        if j==0:ax.axhline(0,c='#bbb',lw=.8)
    fig.suptitle('分布仍不相同：看事件间散布，不能只看一个中位数',fontsize=15)
    fig.text(.06,.055,'点 = 每次运行的中位数；竖线 = 事件分布 5–95% 范围（不是置信区间）；圆/方 = 两次噪声。时差上方数字 = 两杆均参与的事件数。\n左列患者来自全部 FIT；右列只比较模型发放密度包络，不能将其等同于患者 HFO 频谱宽度。\n宽度定义：每事件对参与接触点的 t90−t10 取中位数，再展示这些事件值的分布。',fontsize=9)
    save(fig,'event_distribution_ranges','逐运行显示患者/模型的时差中位数与事件5–95%范围，另列模型局部活动宽度。区间是事件散布，不是网络均值置信区间。**关注点**：TB的正时差集中偏移，以及局部宽度仍集中在约16ms；患者HFO与模型活动信号不能直接等同。')


def confirmation(t):
    c=t['counts'].set_index(['candidate','topology','noise'])
    combos=[(2511,847101),(2511,847102),(2611,847201),(2611,847202),(2612,847201),(2612,847202)]
    cols={2511:'#555555',2611:'#336ba7',2612:'#cf7928'}
    fig,axes=plt.subplots(1,3,figsize=(13,5.6));fig.subplots_adjust(left=.07,right=.99,top=.78,bottom=.20,wspace=.34)
    for ax,pair,metric,title in zip(axes,[('up3__circle','up3__ellipse4'),('up3__out_reference','up3__out_perp20'),('up3__out_reference','up3__out_perp20')],
        ['L_search','L_search','rotation_time_fraction'],['只改变左核形状','左核输出横向范围 ×2','同一范围干预：旋转候选']):
        for topo,s in combos:
            if any((cid,topo,s) not in c.index for cid in pair):continue
            vals=[c.loc[(cid,topo,s),metric] for cid in pair]
            if metric=='rotation_time_fraction':vals=np.asarray(vals)*100
            ax.plot([0,1],vals,c=cols[topo],ls='-' if s%2 else '--',marker='o' if s%2 else 's',ms=5,lw=1.5)
        ax.set(xticks=[0,1],xticklabels=['圆核','椭圆4:1'] if pair[0]=='up3__circle' else ['原范围\n重采样对照','横向范围\n×2'],
            ylabel='冻结训练分数（越低越接近目标）' if metric=='L_search' else '旋转候选占观察时间 (%)',title=title,xlim=(-.1,1.1));ax.grid(axis='y',alpha=.2)
    fig.legend(handles=[Line2D([],[],c=color,label=('训练' if k==2511 else '确认')+'拓扑 '+str(k)) for k,color in cols.items()],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.89))
    fig.suptitle('跨网络检查：拟合改善不总能重复；旋转候选的响应可单独观察',fontsize=14)
    fig.text(.07,.045,'每条线连接同网络、同噪声的两个条件；实线/圆与虚线/方区分该阶段两次噪声。这里使用已完整的配对条件，不混入未完成运行。\n范围比较匹配各源出度和总输出权重，但目标、入度及距离时延可变。旋转计数含叠加波假阳性，不能称为确证螺旋波。',fontsize=9)
    save(fig,'topology_noise_confirmation','三个拓扑各两次噪声的配对结果，训练与新拓扑确认分色；仅使用本图条件已完成的全部配对。范围比较以同规则重采样为对照。**关注点**：形状单改使分数变差，范围效果随拓扑改变；旋转候选减少不能直接推为真实螺旋减少。')


def outgoing_density(t,ref):
    ids=['up3__circle','up3__circle__EE_core_to_out_scale_1.25','up3__circle__EE_core_to_out_degree_scale_1.5']
    d=t['observations'];d=d[(d.topology==2511)&(d.layer=='primary')].set_index(['candidate','noise','mode'])
    fig,axes=plt.subplots(3,3,figsize=(13.2,9.4));fig.subplots_adjust(left=.075,right=.99,top=.87,bottom=.14,hspace=.42,wspace=.32)
    for row,mode in enumerate(['ALL','TA','TB']):
      for col,(key,label) in enumerate([('SCL_upper_participation','SCL9/8 平均参与概率'),('SCL_minus_ICL_lag_median_ms','SCL − ICL 质心时差 (ms)'),('pair_order_probability_mae','成对顺序概率平均误差')]):
        ax=axes[row,col];pr=0 if col==2 else ref['modes'][mode][key]
        for seed,color,marker,style in zip(SEEDS,['#7562a8','#19897f'],['o','s'],['-','--']):
            y=[d.loc[(cid,seed,mode),key] for cid in ids]
            # Discrete interventions, each contrasted with the common baseline.
            ax.plot(range(3),y,c=color,marker=marker,ls=style)
        ax.axhline(pr,c='#333',ls=':',lw=1.3)
        ax.set(xticks=range(3),xticklabels=['基线','原边权重\n增加25%','连接条数\n增加约50%'],ylabel=label,title=mode,xlim=(-.12,2.12))
        if col==0:ax.set_ylim(0,1.08)
        elif col==1:ax.set_ylim(-35,115)
        else:ax.set_ylim(-.02,.6)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('向外连接的“强度”和“条数”产生不同响应；总体还受到模式比例影响',fontsize=14)
    fig.legend(handles=[Line2D([],[],c='#7562a8',marker='o',label='噪声847101'),Line2D([],[],c='#19897f',ls='--',marker='s',label='噪声847102'),Line2D([],[],c='#333',ls=':',label='患者参考')],loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.text(.06,.035,'同一拓扑2511、同一圆形core位置；横轴是不同干预，不是连续参数梯度。连接条数和总权重同时增加，不能据此分离纯数量作用。\n增加连接后，两次噪声的TA时差为−19.52/−19.58ms，TB为34.54/34.46ms；TA占比52%/39%却使总体中位数变为−14.62/+11.36ms。\n总体中位数变号不等于两种模式内部的传播方向都变了。顺序误差在两触点均参与的事件上计算，每个触点对等权。',fontsize=9)
    save(fig,'outgoing_weight_vs_edge_count','共同圆形基线下对比原边增权与实际增边，ALL/TA/TB分别展示参与、时差和顺序误差。增边同时增加总输入，并非剂量匹配实验。**关注点**：增边缓解TA漏招募但造成过度参与；噪声改变模式占比足以使总体中位数变号。')


def all_parameter_map(t):
    d=t['observations'];d=d[(d.topology==2511)&(d.layer=='primary')].set_index(['candidate','noise','mode'])
    candidates=sorted(t['counts'].query("stage=='response'").candidate.unique())
    metrics=[('SCL_upper_participation','SCL9/8参与\n百分点',100),('ICL_contact_participation','ICL平均参与\n百分点',100),
        ('pair_order_probability_mae','顺序概率误差\n变化',1),('SCL_minus_ICL_lag_median_ms','跨杆时差\nms',1),('recruitment_span_ms_median','招募跨度\nms',1),('n','该类事件数\n变化',1)]
    short={'EE_same_core_scale':'核内 E→E','EE_core_to_out_scale':'向外 E→E','EE_core_to_out_degree_scale':'向外 E→E 边数','EI_same_core_scale':'核内 E→I',
        'IE_same_core_scale':'核内 I→E','II_same_core_scale':'核内 I→I','depth_A_scale':'左核降阈值','core_mean_rate_scale':'核内输入均值','core_ou_correlation':'两核输入相关'}
    fig,axes=plt.subplots(1,2,figsize=(17,12));fig.subplots_adjust(left=.19,right=.99,top=.90,bottom=.105,wspace=.50)
    labels=[];raw=[]
    for cid in candidates:
        c=rt.read(P/'candidates'/f'{cid}.json');k,v=c['contrast'].split('=');shape='圆形' if c['shape'].get('aspect_A',1)==1 else '椭圆'
        labels.append(shape+'｜'+short.get(k,k)+(' =' if k=='core_ou_correlation' else ' ×')+v)
    for ax,mode in zip(axes,['TA','TB']):
        values=np.empty((len(candidates),len(metrics)));stars=np.zeros_like(values,dtype=bool)
        for i,cid in enumerate(candidates):
            c=rt.read(P/'candidates'/f'{cid}.json');base=c['comparison']
            for j,(key,label,scale) in enumerate(metrics):
                delta=np.array([d.loc[(cid,s,mode),key]-d.loc[(base,s,mode),key] for s in SEEDS])*scale
                values[i,j]=delta.mean();stars[i,j]=delta.min()<0<delta.max()
                raw.append(dict(candidate=cid,comparison=base,mode=mode,metric=key,unit_scale=scale,noise847101_delta=delta[0],noise847102_delta=delta[1],mean_delta=delta.mean()))
        denom=np.maximum(np.nanmax(abs(values),axis=0),1e-9)
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#bcbcbc')
        ax.imshow(values/denom,aspect='auto',cmap=cmap,vmin=-1,vmax=1)
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,(f'{v:+.2f}'+('*' if stars[i,j] else '')) if np.isfinite(v) else '支持不足',fontsize=7.5,ha='center',va='center',color='white' if abs(v/denom[j])>.73 else 'black')
        ax.set(xticks=range(len(metrics)),xticklabels=[v[1] for v in metrics],yticks=range(len(labels)),yticklabels=labels,title=mode)
        ax.tick_params(labelsize=8);ax.xaxis.tick_top();ax.xaxis.set_label_position('top')
    fig.suptitle('全部局部参数响应：相对各自圆形／椭圆基线的变化',fontsize=15)
    fig.text(.08,.021,'每格为同拓扑 2511、两次噪声配对变化的平均；* 表示两条噪声方向相反。红=增加，蓝=减少，颜色不是“好/坏”。\n色阶按列缩放；灰色表示至少一次噪声无该类有效观测，不能合并；数字保留真实单位，未汇总成联合分数。\n行标签为实际配置倍率：核内EE基线为0.85，输入均值0.95，输入相关系数0；不是各自相对变化率。\n所有条件为局部单参数干预，圆形与椭圆为不同背景；逐噪声数据见 paired_parameter_snapshot.csv。',fontsize=10)
    pd.DataFrame(raw).to_csv(O/'paired_parameter_snapshot.csv',index=False)
    save(fig,'all_local_parameters','完整展示本轮局部参数条件在TA/TB上的配对响应；数字保留各自量纲，两噪声均值用于概览，星号提示方向相反。颜色按列缩放，不表示拟合优劣。**关注点**：不同参数改变参与、顺序、时差、范围和事件量的不同组合，而非一条轴改善所有观测。')


def tb_branch(t):
    ids=['up3__circle','up3__circle__EE_core_to_out_scale_1.25','up3__circle__EE_core_to_out_degree_scale_1.5','up3__ellipse4__EI_same_core_scale_0.75']
    labels=['圆形基线','圆形\n向外边权增25%','圆形\n向外边数增50%','椭圆\n核内E→I减25%']
    d=t['pairs'];d=d[(d.topology==2511)&(d.layer=='primary')&(d['mode']=='TB')&d.contact_i.eq('ICL11')&d.contact_j.eq('ICL9')]
    fig,ax=plt.subplots(figsize=(10,5));fig.subplots_adjust(left=.10,right=.98,top=.81,bottom=.26)
    patient=float(d.patient_i_precedes_j.iloc[0]);pn=int(d.patient_joint_n.iloc[0])
    ax.axhline(patient,c='black',ls=':',label=f'患者 FIT：{patient:.1%}，共同参与 n={pn}')
    rows=[]
    for i,cid in enumerate(ids):
      for seed,off,mk,color in zip(SEEDS,[-.09,.09],['o','s'],['#7562a8','#19897f']):
        r=d[(d.candidate==cid)&(d.noise==seed)].iloc[0];value=r.model_i_precedes_j
        ax.scatter(i+off,value,c=color,marker=mk,s=55);ax.text(i+off,value+.025,f'n={int(r.model_joint_n)}',ha='center',fontsize=8,color=color)
        rows.append(dict(candidate=cid,noise=seed,model_joint_n=int(r.model_joint_n),patient_joint_n=pn,model_probability=float(value),patient_probability=patient))
    ax.set(xticks=range(4),xticklabels=labels,ylim=(-.035,.95),ylabel='P(ICL11 的质心早于 ICL9 | 两者参与)',title='TB 左侧折返的具体残差：不是单个示例看起来不像')
    ax.legend(loc='upper right',frameon=False);ax.grid(axis='y',alpha=.15)
    fig.text(.10,.045,'冻结FIT和每次运行全部合格TB事件；该触点对用于解释已有图的残差，未加入损失或提名条件。\n紫圆=噪声847101，绿方=847102；点是逐运行比例，没有把这些事件当作独立网络复制。\n这是一个已选触点对的诊断，不能独自概括完整空间路径；全部105个触点对保存在 pairs.csv。',fontsize=9)
    pd.DataFrame(rows).to_csv(O/'tb_branch_pair_diagnostic.csv',index=False)
    save(fig,'TB_branch_residual','TB全部合格事件中，比较ICL11与ICL9的条件质心先后概率，同时列实际共同参与事件数。该触点对仅用于事后解释图像残差，不进入训练或提名。**关注点**：患者约69%为ICL11较早，而这些模型为0–21%，说明折返残差不能归咎于单个GIF选例。')


def media(meta):
    selected=['up3__circle__EE_core_to_out_scale_1.25','up3__ellipse4__EI_same_core_scale_0.75','up3__circle__EE_core_to_out_degree_scale_1.5']
    patient=fr.patient_payloads(); manifests=[]
    for cid in selected:
        c=rt.read(P/'candidates'/f'{cid}.json');c.update(topology=2511,display_name=CASES[cid]+'；左核上移 3 mm')
        units={}
        for s in SEEDS:
            source=next(x['trajectory'] for x in meta['sources'] if '/'+cid+'/' in x['trajectory'] and '/2511_'+str(s)+'/' in x['trajectory'])
            units[s]=an.load_unit(Path(source),1500.)
        physics=rt.read(Path(source).parents[1]/'applied_physics.json');c['_applied_threshold']=physics['threshold']
        manifests.append(fr.spectral_comparison(c,units,SEEDS,F,patient,'primary'))
        CAPTIONS[cid+'_patient_spectra_model_envelopes.png']=CASES[cid]+'。左列是Fig2C固定患者示例，右侧两列是两次噪声的模型类内均值附近事件，不按患者相似度选例；完整毫秒轴和固定15行。**关注点**：SCL参与和TB的ICL次序，以及模型包络比真实STFT明显短。'
        r,a,ids=units[SEEDS[0]]
        mm=fr.four_panel(c,SEEDS[0],r,a,ids,F,physics,'primary')
        fields=[fr.native_timing(r,a,x['event'])[0] for x in fr.representatives(a,ids).values()]
        valid=np.concatenate([q[np.isfinite(q)] for q in fields])
        mm['native_shared_color_limits_ms']=[float(valid.min()),max(float(valid.max()),float(valid.min())+1)]
        manifests.append(mm)
        CAPTIONS[cid+'_847101_same_network.png']='同一网络的阈值场、两类事件原生累计活动时间和完整60秒读出。真实core形状及电极位置保持；累计质量时间不解释为因果起源。**关注点**：空间活动是否支持接触时序，是否存在平行活动或折返。'
        g=gifs.render(c,SEEDS[0],r,a,ids,physics,F,patient);manifests.append(g)
        CAPTIONS[g['multievent_file']]='每类最早三个合格事件按实际发生时间组织的6事件动画，左为全部原生E活动，中为模型包络，右为固定Fig2C患者示例。窗口间有拼接，不代表连续模式切换；同一动画固定原生场色标。**关注点**：亮区是否沿电极招募，TA/TB是否只是接触质心分类。'
        CAPTIONS[cid+'_847101_continuous_field_readout.gif']='固定1.5–7.5秒连续原生活动与电极读出，未按类别或好坏选时间段；原生2ms帧每40ms抽一帧。**关注点**：多事件完整背景；快速传播细节请看4ms步进的6事件GIF。'
        # Contact/native storyboard for the same central exemplars: fixed +20 ms offsets.
        rep=fr.representatives(a,ids);fig,axes=plt.subplots(2,7,figsize=(16,5.8))
        fig.subplots_adjust(left=.025,right=.995,top=.85,bottom=.10,wspace=.08,hspace=.23)
        vmax=max(float(np.quantile(a['sheet_activity_counts'][750:],.999)),1)
        chosen=[]
        for row,lab in enumerate(['TA','TB']):
            i=rep[lab]['event'];lo,hi=r['events'][i]['window_ms'];movie=a['sheet_activity_counts'][round(lo/2):round(hi/2)]
            cum=np.cumsum(movie.sum((1,2)));start=lo+2*np.argmax(cum>=.05*cum[-1]);chosen.append(dict(mode=lab,event=i,global_mass5_ms=float(start)))
            for col,offset in enumerate([0,20,40,60,80,100,120]):
                tt=min(start+offset,hi-2);axes[row,col].imshow(gifs.field_tile(a['sheet_activity_counts'][round(tt/2)],c,a,physics,vmax,240))
                axes[row,col].axis('off');axes[row,col].set_title(f'{lab} {i}\n+{tt-start:.0f} ms',fontsize=9)
        fig.suptitle(CASES[cid]+'：正常选例的原生活动逐帧图',fontsize=14)
        fig.text(.04,.025,'同一原生色标；白线=core，青点=电极。每行从窗口全场5%累计活动时间开始，原生2ms帧每20ms展示；5%质量时间不是因果起源。',fontsize=9)
        stem=cid+'_native_storyboard';save(fig,stem,'两类自身均值附近示例的全部原生E活动，时间与色标固定，不做空间插值。**关注点**：双核活动、大片同时招募及沿电极传播之间的关系。')
        manifests.append(dict(file=stem,selection=chosen,field_color_limits=[0,vmax]))
        del units,r,a
    rt.write(O/'media_manifest.json',manifests)


def small_pdf(meta):
    files=['EI_shape_interaction.png','outgoing_weight_vs_edge_count.png','TB_branch_residual.png','event_distribution_ranges.png','topology_noise_confirmation.png','all_local_parameters.png',
        'up3__circle__EE_core_to_out_scale_1.25_patient_spectra_model_envelopes.png',
        'up3__ellipse4__EI_same_core_scale_0.75_patient_spectra_model_envelopes.png',
        'up3__ellipse4__EI_same_core_scale_0.75_847101_same_network.png',
        'up3__ellipse4__EI_same_core_scale_0.75_native_storyboard.png',
        'up3__circle__EE_core_to_out_degree_scale_1.5_patient_spectra_model_envelopes.png']
    # CJK OTF fonts in this environment need Type 3 embedding for reliable PDF footers.
    old_pdf_fonttype=plt.rcParams['pdf.fonttype'];plt.rcParams['pdf.fonttype']=3
    with PdfPages(O/'key_results_review.pdf') as pdf:
      for idx,name in enumerate(files):
        if not (F/name).exists():continue
        with Image.open(F/name) as im:
            w,h=im.size;fig=plt.figure(figsize=(14,max(4.8,14*h/w+.75)))
            ax=fig.add_axes([.015,.075,.97,.91]);ax.imshow(im);ax.axis('off')
            fig.text(.03,.025,f'{idx+1} | 完成输出快照 {meta["analyzed_runs"]}/140；实验单位=运行 | {meta["snapshot_time"]}',fontsize=9)
            pdf.savefig(fig);plt.close(fig)
    plt.rcParams['pdf.fonttype']=old_pdf_fonttype
    (F/'README.md').write_text('# 2026-09-12 关键结果：未完成轮次的独立快照\n\n'+''.join('### '+name+'\n'+caption+'\n\n' for name,caption in CAPTIONS.items()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-media',action='store_true');args=p.parse_args()
    meta,t,ref=snapshot()
    interaction(t,ref,'EI','__EI_same_core_scale_0.75',['E→I 原权重','E→I 权重 ×0.75'],'核内抑制招募 × core 形状：TA 时序改善是否牺牲 SCL？')
    interaction(t,ref,'EE','__EE_core_to_out_scale_1.25',['向外 E→E 原权重','向外 E→E 权重 ×1.25'],'向外兴奋输出 × core 形状：增强输出也存在参与—时序取舍')
    distributions(t,ref);confirmation(t);outgoing_density(t,ref);all_parameter_map(t);tb_branch(t)
    print(json.dumps({'snapshot_runs':meta['analyzed_runs'],'static_response_plots':'complete'}),flush=True)
    if not args.skip_media:media(meta)
    small_pdf(meta)
    qa=[]
    for path in sorted(F.iterdir()):
        if path.suffix not in ['.png','.gif']:continue
        with Image.open(path) as im:
            for i in range(getattr(im,'n_frames',1)):im.seek(i);im.load()
            qa.append(dict(file=path.name,size=im.size,frames=getattr(im,'n_frames',1),sha256=rt.sha(path)))
    rt.write(O/'artifact_qa.json',dict(images=qa,review_pdf_sha256=rt.sha(O/'key_results_review.pdf'),
        automatic_decode='PASS',human_visual_acceptance='PENDING',producer_sha256=rt.sha(Path(__file__))))
    print(json.dumps({'output':str(O),'decoded_images':len(qa)}),flush=True)


if __name__=='__main__':main()

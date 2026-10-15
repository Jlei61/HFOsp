"""Existing 32 confirmation runs: geometry effects paired within graph and noise."""
from pathlib import Path
import hashlib,json,sys,time
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
install()
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
P=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
OUT=P.parent/'overnight_exploration_20260913/geometry_confirmation_effects'
F=OUT/'figures'
IDS=['endpoint__circle','endpoint__ellipse4','up3__circle','up3__ellipse4',
     'up3__radius25','up3__radius25_dose_matched','up3__out_reference','up3__out_perp20']
LABELS=['端点\n圆核','端点\n椭圆核','上移3mm\n圆核','上移3mm\n椭圆核',
        '半径\n2.5mm','半径2.5mm\n匹配降阈值总量','离核边\n重采样对照','离核横向\n范围×2']
CONTRASTS=[
 ('circle_y3','圆核：左核上移3 mm','endpoint__circle','up3__circle'),
 ('ellipse_y3','椭圆核：左核上移3 mm','endpoint__ellipse4','up3__ellipse4'),
 ('shape','上移后：左核圆→等面积椭圆4:1','up3__circle','up3__ellipse4'),
 ('radius','上移后：左核半径1.75→2.5 mm','up3__circle','up3__radius25'),
 ('radius_dose','扩大半径＋匹配降阈值总量','up3__circle','up3__radius25_dose_matched'),
 ('outgoing','离核边：横向范围倍率1→2','up3__out_reference','up3__out_perp20')]
PAIRS=[(t,s) for t in [2611,2612] for s in [847201,847202]]
COLORS={2611:'#3675a9',2612:'#c67535'}
METRICS=[('TA_rod','TA：杆间时差变化\n(ms)',1),
         ('TA_upper','TA：SCL9/8参与变化\n(百分点)',100),
         ('TA_fraction','TA标签比例变化\n(百分点)',100),
         ('TB_rod','TB：杆间时差变化\n(ms)',1),
         ('TB_return','TB：ICL11早于ICL9的概率变化\n(百分点)',100)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    F.mkdir(parents=True,exist_ok=True);lookup={};sources=[];rows=[]
    for p in sorted((P/'analysis/units').glob('*/result.json')):
        r=json.loads(p.read_text());c=r['counts']
        if c['stage']!='confirmation' or c['candidate'] not in IDS:continue
        key=(c['candidate'],c['topology'],c['noise']);assert key not in lookup
        assert c['duration_ms']==60000 and c['physical_status']=='COMPLETE_NO_RUNAWAY'
        obs={o['mode']:o for o in r['observations'] if o['layer']=='primary'}
        pairs=[v for v in r['pairs'] if v['layer']=='primary' and v['mode']=='TB' and
               {v['contact_i'],v['contact_j']}=={'ICL11','ICL9'}]
        assert len(pairs)==1;pair=pairs[0]
        prob=pair['model_i_precedes_j'] if pair['contact_i']=='ICL11' else 1-pair['model_i_precedes_j']
        z=dict(candidate=c['candidate'],topology=c['topology'],noise=c['noise'],TA_n=c['TA'],TB_n=c['TB'],
               primary_n=c['primary'],TA_fraction=c['TA']/c['primary'],TA_upper=obs['TA']['SCL_upper_participation'],
               TA_rod=obs['TA']['SCL_minus_ICL_lag_median_ms'],TB_rod=obs['TB']['SCL_minus_ICL_lag_median_ms'],
               TA_pair_error=obs['TA']['pair_order_probability_mae'],TB_pair_error=obs['TB']['pair_order_probability_mae'],
               TB_return=prob,TB_return_joint_n=pair['model_joint_n'],L_search=c['L_search'])
        lookup[key]=z;rows.append(z)
        source=Path(r['source']);sources.append(dict(analysis=str(p),analysis_sha256=sha(p),trajectory=str(source),trajectory_json_sha256=sha(source)))
    assert set(lookup)=={(cid,t,s) for cid in IDS for t,s in PAIRS}
    configs={cid:json.loads((P/'candidates'/f'{cid}.json').read_text()) for cid in IDS}
    # These comparisons are explicit and do not borrow different EI/EE settings.
    for cid,c in configs.items():
        assert c['parameters']['EI_same_core_scale']==1 and c['parameters']['EE_core_to_out_scale']==1
    paired=[]
    for name,label,reference,candidate in CONTRASTS:
        for topo,noise in PAIRS:
            a,b=lookup[(reference,topo,noise)],lookup[(candidate,topo,noise)]
            for metric,_,scale in METRICS:
                paired.append(dict(contrast=name,label=label,reference=reference,candidate=candidate,topology=topo,noise=noise,
                   metric=metric,reference_value=a[metric],candidate_value=b[metric],difference=b[metric]-a[metric],
                   plotted_difference=scale*(b[metric]-a[metric]),reference_TA_n=a['TA_n'],candidate_TA_n=b['TA_n'],
                   reference_TB_n=a['TB_n'],candidate_TB_n=b['TB_n']))
    df=pd.DataFrame(rows);de=pd.DataFrame(paired);df.to_csv(OUT/'run_observations.csv',index=False);de.to_csv(OUT/'paired_effects.csv',index=False)
    refs=json.loads((P/'analysis/patient_reference.json').read_text())
    write('sources.json',dict(created_unix=time.time(),source_round=str(P),runs=sources,candidates=configs,
          contrasts=CONTRASTS,patient_reference=str(P/'analysis/patient_reference.json'),patient_reference_sha256=sha(P/'analysis/patient_reference.json'),
          scope='32 existing geometry/range confirmation units; zero new physical runs; no selection or loss changes'))
    fig,axes=plt.subplots(1,5,figsize=(19.5,6.4),sharey=True);fig.subplots_adjust(left=.225,right=.99,top=.8,bottom=.2,wspace=.38)
    for ax,(metric,label,scale) in zip(axes,METRICS):
        ax.axvline(0,c='.55',lw=.8);ax.grid(axis='x',alpha=.15)
        for i,(name,_,_,_) in enumerate(CONTRASTS):
            for topo,offset in [(2611,-.13),(2612,.13)]:
                q=de[(de.contrast==name)&(de.metric==metric)&(de.topology==topo)].sort_values('noise')
                vals=q.plotted_difference.to_numpy();ax.plot(vals,[i+offset]*2,c=COLORS[topo],lw=1,alpha=.55)
                for value,noise in zip(vals,q.noise):
                    ax.scatter(value,i+offset,c=COLORS[topo] if noise==847201 else 'white',edgecolors=COLORS[topo],marker='o' if noise==847201 else '^',s=45,zorder=3)
        ax.set_xlabel(label);ax.set_yticks(range(6));ax.set_yticklabels([c[1] for c in CONTRASTS]);ax.set_ylim(5.55,-.55)
    handles=[Line2D([],[],c=c,lw=2,label=f'网络 {t}') for t,c in COLORS.items()]+[
       Line2D([],[],marker='o',c='.3',ls='',label='噪声847201'),Line2D([],[],marker='^',mfc='white',mec='.3',ls='',label='噪声847202')]
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.59,.92),frameon=False)
    fig.suptitle('同一几何改动，在不同网络上是否有相同作用？',fontsize=18,y=.99)
    fig.text(.02,.045,'每点 = 同一基础网络、同一噪声种子下“改动后 − 各自直接对照”；两点之间的细线仅连接两次噪声，不是置信区间。\n杆间时差 = 事件中参与SCL接触点质心中位数 − 参与ICL接触点质心中位数，再取该类事件中位数。TA/TB按冻结分类器组织。\n32条既有确认运行，2张网络×2噪声；均固定核内EI=1、向外EE=1，不能替代新EE/EI候选的复测。条件事件组成也可随参数改变。',fontsize=10)
    save(fig,'paired_geometry_effects')
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True,sharey='row');fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.17,hspace=.2,wspace=.12)
    for i,mode in enumerate(['TA','TB']):
        ref=refs['modes'][mode];median=ref['SCL_minus_ICL_lag_median_ms'];q05=ref['SCL_minus_ICL_lag_q05_ms'];q95=ref['SCL_minus_ICL_lag_q95_ms']
        for j,topo in enumerate([2611,2612]):
            ax=axes[i,j];ax.axhspan(q05,q95,color='.93');ax.axhline(median,c='black',ls=':',lw=1.2)
            for k,cid in enumerate(IDS):
                for noise,dx,marker in [(847201,-.12,'o'),(847202,.12,'^')]:
                    r=lookup[(cid,topo,noise)];value=r[mode+'_rod']
                    ax.scatter(k+dx,value,c=COLORS[topo] if noise==847201 else 'white',edgecolors=COLORS[topo],marker=marker,s=40,zorder=3)
                    ax.annotate(str(r[mode+'_n']),(k+dx,value),xytext=(0,7 if noise==847201 else -12),textcoords='offset points',fontsize=7,ha='center')
            ax.set_title(f'{mode}｜网络 {topo}');ax.set_ylabel('SCL − ICL 质心时差 (ms)');ax.set_xticks(range(8));ax.set_xticklabels(LABELS,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('绝对时差与患者参考：图上数字是本类实际事件数',fontsize=17,y=.99)
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.53,.965),frameon=False)
    fig.text(.07,.045,'黑虚线：患者FIT该类事件中位数；灰区：患者该类事件5–95%范围，不是模型误差条。点：一条60秒运行，排除前1.5秒。\n同一模式的纵轴固定；圆点/三角为两噪声，颜色区分基础网络。TA事件很少的条件仍如实展示，不自动判定缺乏模式能力。\n不能将两张网络上的相反变化平均后写成普遍改善，也不能把相近杆间摘要等同于完整传播路径恢复。',fontsize=10)
    save(fig,'absolute_timing_by_network')
    fig,axes=plt.subplots(1,3,figsize=(15,6.3));fig.subplots_adjust(left=.19,right=.98,top=.82,bottom=.16,wspace=.4)
    for ax,(key,title,fmt) in zip(axes,[('TA_n','TA实际事件数','d'),('TB_n','TB实际事件数','d'),('TA_fraction','TA占合格事件的比例','.1%')]):
        arr=np.array([[lookup[(cid,t,s)][key] for t,s in PAIRS] for cid in IDS]);im=ax.imshow(arr,cmap='YlGnBu',aspect='auto',vmin=0,vmax=1 if key=='TA_fraction' else None)
        for (i,j),v in np.ndenumerate(arr):ax.text(j,i,format(int(v) if fmt=='d' else v,fmt),ha='center',va='center',fontsize=9,color='white' if v>im.norm.vmax*.6 else 'black')
        ax.set_xticks(range(4));ax.set_xticklabels([f'{t}\n{s}' for t,s in PAIRS],fontsize=8);ax.set_title(title);ax.set_yticks(range(8));ax.set_yticklabels([x.replace('\n',' ') for x in LABELS] if ax is axes[0] else [])
        fig.colorbar(im,ax=ax,fraction=.05,pad=.025)
    fig.suptitle('每个条件有多少观测支持？',fontsize=18,y=.98)
    fig.text(.19,.035,f'每列上行=网络种子，下行=噪声种子。患者FIT自然比例：TA {refs["modes"]["TA"]["n"]/refs["fit_n"]:.1%}。\n32条确认均完整，但并不具有相同的少数模式证据量；事件数不是网络重复数。\n这批结果没有新模拟；TA/TB标签本身不证明恢复了患者路径。',fontsize=10)
    save(fig,'event_support')
    note='''# 几何作用的跨网络确认：32条既有运行

问题是同一个core几何或离核范围改动，在不同网络中是否保留同方向的参与、时序与模式支持变化。这里重读原140条中的全部32条几何确认：8条件×2基础拓扑（2611/2612）×2噪声（847201/847202），每条60秒，排除前1.5秒，没有新增物理运行。

每个差值在同一基础拓扑和噪声身份下取“改动−直接对照”，不是将事件逐一配对。网络数是2；四个点不等于四张独立网络，点间细线不代表置信区间。参数改变会同时改变core成员、随机输入支持和局部连接分块；等面积或匹配降阈值总量并不固定这些成分。离核范围对照重新采样实际边，不能称为同一物理图。

当前最重要的结果是网络与几何之间存在明显的条件依赖：在上移后的圆核→等面积椭圆对照中，网络2611的TA杆间时差从约88ms降到−26/−17ms；网络2612却从−4.5/+1.0ms变为+4.2/+10.2ms。两次噪声在各网络内方向较一致，但跨网络方向相反。这是TA标签条件摘要的变化，可能同时包含类内事件组成改变；不是证明同一个事件的传播被反转。

所有32条运行的TB杆间时差中位数仍约36–38ms，ICL11先于ICL9的概率仍远低于患者；不能用较好TA结果宣布双模式恢复。扩大并匹配降阈值总量的条件仅有6、1、6、2个TA事件，某些参与摘要达到1也缺少充分模式内支持。支持不足与机制不可能应分开。

这批几何确认固定核内EI=1、向外EE=1。它能说明几何效应不能仅凭一个网络推广，却不能替代正在运行的左移、EE/EI及连接数量候选自己的多网络复测；不因此修改当前候选、损失或预算。

患者黑色参考为FIT事件在同一观测定义下的统计。TA/TB来自同一冻结分类器，其标签分布也参与现有训练目标，因此条件分布是拟合诊断；既有新拓扑/噪声确认只检验这些运行中的重复性，不能作为未见患者上的外推证据。杆间摘要不能替代固定接触对、Fig2C时序或模型原生场的检查。
'''
    (OUT/'scientific_note.md').write_text(note)
    readme='# 几何确认图\n\n'
    descriptions={
      'paired_geometry_effects':'六种直接对照分别在两张网络、两噪声下计算五种观测的变化。颜色是网络，符号是噪声；细线只连接同一网络的两个噪声点。\n**关注点**：TA时差效应可能跨网络反向，TB固定延迟仍存在。',
      'absolute_timing_by_network':'TA/TB分行、网络分列，统一同模式纵轴，显示绝对杆间时差与患者中位数及5–95%事件范围。点旁数字为本类事件数。\n**关注点**：差值小不代表已经恢复，少数模式摘要必须结合实际支持。',
      'event_support':'列出全部32条确认的TA/TB事件数与TA比例。没有将事件数量当作网络重复数。\n**关注点**：扩大且匹配降阈值总量时，TA仅1–6例，参与率不能单独判断恢复。'}
    for name,desc in descriptions.items():readme+=f'### {name}.png\n{desc}\n\n'
    (F/'README.md').write_text(readme)
    with PdfPages(OUT/'geometry_confirmation_report.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.94,'几何参数：两张网络上的作用与边界',fontsize=18)
        import textwrap
        cover='\n\n'.join('\n'.join(textwrap.wrap(p,58)) for p in note.split('\n\n')[1:5]);fig.text(.06,.86,cover,fontsize=10,va='top',linespacing=1.6);pdf.savefig(fig);plt.close(fig)
        for name in descriptions:
            with Image.open(F/f'{name}.png') as im:arr=np.array(im.convert('RGB'))
            h,w=arr.shape[:2];fig=plt.figure(figsize=(17,17*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig);plt.close(fig)
    write('computational_checks.json',dict(status='PASS_PENDING_VISUAL_REVIEW',runs=len(rows),contrasts=len(CONTRASTS),paired_metric_rows=len(paired),new_simulations=0,
       TB_rod_range_ms=[float(df.TB_rod.min()),float(df.TB_rod.max())],TB_return_range=[float(df.TB_return.min()),float(df.TB_return.max())],
       shape_effects=[r for r in paired if r['contrast']=='shape' and r['metric']=='TA_rod']))
    print(json.dumps(dict(runs=len(rows),effects=len(paired),output=str(OUT))))


if __name__=='__main__':main()

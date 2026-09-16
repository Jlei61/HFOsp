"""Immutable, concise review snapshots of the authorized eight-hour window.

Reads completed analysis units; never changes scores, nominations or simulations.
Missing conditions remain missing. An event is not a network replicate.
"""
from pathlib import Path
import sys,json,datetime,hashlib,shutil,argparse
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

BASE=Path('/data/hfosp/topic4_sef_hfo');W=BASE/'overnight_exploration_20260913'
FOLLOW=BASE/'core_recruitment_tradeoff_followup_20260912';N=BASE/'core_multiseed_response_curves_20260913';G=BASE/'global_axis_residual_probe_20260913'
ANCHORS=[('up3__circle__EE_core_to_out_scale_1.25','圆核／向外EE×1.25','#3478b8'),('up3__ellipse4__EI_same_core_scale_0.75','椭圆／核内EI×0.75','#bd7939')]
PROBES=[('x_minus075','左核 x −0.75mm'),('x_plus075','左核 x +0.75mm'),('y_minus10','左核 y −1mm'),('y_plus10','左核 y +1mm'),('radius205','左核基准半径→2.05mm'),('radius235','左核基准半径→2.35mm')]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def geometry_review(out):
    records=[read(p) for p in sorted((FOLLOW/'analysis/units').glob('*/result.json'))]
    lookup={(r['counts']['candidate'],r['counts']['topology'],r['counts']['noise']):r for r in records}
    metrics=[('TA_upper','TA：上部SCL参与\n变化（百分点）',100),('TA_order','TA：成对顺序误差\n变化（百分点）',100),('TB_rod','TB：SCL−ICL时差\n中位数变化（ms）',1),('TB_return','TB：ICL11早于ICL9\n概率变化（百分点）',100),('TA_fraction','TA事件占比\n变化（百分点）',100)]
    def measures(r):
        modes={x['mode']:x for x in r['observations'] if x['layer']=='primary'};cnt=r['counts']
        pair=next(x for x in r['pairs'] if x['layer']=='primary' and x['mode']=='TB' and x['contact_i']=='ICL11' and x['contact_j']=='ICL9')
        return dict(TA_upper=modes['TA']['SCL_upper_participation'],TA_order=modes['TA']['pair_order_probability_mae'],TB_rod=modes['TB']['SCL_minus_ICL_lag_median_ms'],TB_return=pair['model_i_precedes_j'],TA_fraction=cnt['TA']/cnt['primary'] if cnt['primary'] else None,TA_n=cnt['TA'],TB_n=cnt['TB'],TB_pair_n=pair['model_joint_n'],primary_n=cnt['primary'])
    rows=[];availability=[];sources={}
    for ai,(anchor,label,col) in enumerate(ANCHORS):
      for pi,(suffix,_) in enumerate(PROBES):
       cid=anchor+'__'+suffix;c=read(FOLLOW/'candidates'/f'{cid}.json');assert c['comparison']==anchor and c['origin']=='new_geometry'
       for noise in [847101,847102]:
        r=lookup.get((cid,2511,noise));b=lookup.get((anchor,2511,noise));available=r is not None and b is not None
        availability.append(dict(candidate=cid,anchor=anchor,noise=noise,analyzed_complete=available))
        if not available:continue
        x,y=measures(r),measures(b)
        for metric,_,scale in metrics:
            xv,yv=x[metric],y[metric];diff=None if xv is None or yv is None else xv-yv
            rows.append(dict(anchor=anchor,anchor_label=label,candidate=cid,probe=suffix,probe_index=pi,topology=2511,noise=noise,metric=metric,value=xv,reference_value=yv,difference=diff,display_scale=scale,TA_n=x['TA_n'],TB_n=x['TB_n'],TB_pair_n=x['TB_pair_n'],reference_TA_n=y['TA_n'],reference_TB_n=y['TB_n'],reference_TB_pair_n=y['TB_pair_n']))
        for z in [r,b]:sources[z['source']]=z['source_sha256']
    d=pd.DataFrame(rows);d.to_csv(out/'geometry_paired_effects.csv',index=False);write(out/'geometry_availability.json',availability)
    fig,axes=plt.subplots(1,5,figsize=(18,6.8),sharey=True);fig.subplots_adjust(left=.205,right=.98,top=.76,bottom=.22,wspace=.35)
    for ax,(key,title,scale) in zip(axes,metrics):
        ax.axvline(0,c='#999',lw=.8)
        for ai,(anchor,_,col) in enumerate(ANCHORS):
         for pi,(_,label) in enumerate(PROBES):
            z=d[(d.anchor==anchor)&(d.metric==key)&(d.probe_index==pi)].sort_values('noise');yy=pi+(-.16 if ai==0 else .16)
            values=z.difference.to_numpy(float)*scale
            if len(values):ax.plot(values,[yy]*len(values),c=col,lw=.8,alpha=.5)
            for (_,row),value in zip(z.iterrows(),values):
                ax.plot(value,yy,marker='o' if row.noise==847101 else '^',ms=5,mfc=col if row.noise==847101 else 'white',mec=col,ls='none')
        lim=max(float(np.nanmax(abs(d.loc[d.metric==key,'difference'].to_numpy(float)*scale)))*1.15,1.)
        ax.set(xlim=(-lim,lim),ylim=(5.6,-.6),yticks=range(6),yticklabels=[x[1] for x in PROBES],title=title);ax.grid(axis='y',alpha=.17)
    handles=[Line2D([],[],c=c,label=s) for _,s,c in ANCHORS]+[Line2D([],[],c='#444',marker='o',ls='none',label='噪声847101'),Line2D([],[],c='#444',marker='^',mfc='white',ls='none',label='噪声847102')]
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.6,.91),frameon=False)
    ncomplete=sum(x['analyzed_complete'] for x in availability)
    fig.suptitle(f'中心／范围微调改变了什么？｜已分析{ncomplete}/24条几何探针',fontsize=16,y=.98)
    fig.text(.045,.055,'每点为同基础拓扑2511、同噪声的改动后减直接对照；两种颜色代表不同参数组合，不是独立网络。连线不是置信区间。\n每列保留原始量纲；零代表没有改变，正负不自动等于患者相似度提高。实际事件数及触点对共同参与数见CSV；缺失点不填0。\n圆核背景：向外EE×1.25、核内EI×1；椭圆背景：向外EE×1、核内EI×0.75。移动中心会改变实际core成员及连接分块。\n范围探针匹配降阈值总量，但核心内随机输入支持人数也会改变；不能解释成只有几何面积变化。未完成条件保留在availability表中。',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/'figures'/('geometry_parameter_response.'+ext),dpi=155)
    plt.close(fig);return dict(completed_geometry=ncomplete,planned_geometry=24,sources=sources)

def build(label=None, window_close=False):
    stamp=datetime.datetime.now().astimezone();tag=label or stamp.strftime('%H%M%S');out=W/f'review_{tag}'
    if window_close and stamp.timestamp()<read(W/'window.json')['review_due_unix']:
        raise RuntimeError('The authorized eight-hour review window has not elapsed')
    if out.exists():raise RuntimeError(f'Immutable snapshot already exists: {out}')
    (out/'figures').mkdir(parents=True);install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    states={name:read(p/'status.json') for name,p in [('followup',FOLLOW),('multiseed',N),('global_axis',G)]}
    followup_cumulative={stage:sum(read(p).get('status')=='COMPLETE' for p in (FOLLOW/stage).glob('units/*/*/workers/trajectory.json')) for stage in ['response','confirmation']}
    selection_path=FOLLOW/'replication_selection.json'
    followup_cumulative['confirmation_planned']=len(read(selection_path)['candidates'])*4 if selection_path.exists() else None
    geometry=geometry_review(out);files=[]
    selection=[('geometry_parameter_response.png',None),
      ('all_parameters_TA.png',W/'completed_parameter_effects/figures/TA_contact_response.png'),
      ('all_parameters_TB.png',W/'completed_parameter_effects/figures/TB_contact_response.png')]
    radius=W/'radius_tradeoff_review/figures/radius_participation_and_activity.png'
    if radius.exists():selection.append(('radius_participation_and_activity.png',radius))
    for name in ['paired_geometry_effects','absolute_timing_by_network']:
        p=W/'geometry_confirmation_effects/figures'/f'{name}.png'
        if p.exists():selection.append(('prior_confirmation_'+name+'.png',p))
    p=W/'geometry_confirmation_effects/core_contact_timing/figures/core_phase_and_rod_timing.png'
    if p.exists():selection.append(('prior_confirmation_core_contact_timing.png',p))
    for short in ['circle','ellipse']:
        matches=list((W/'workpoint_tradeoffs/figures'/short).glob('*patient_spectra_model_envelopes.png'));assert len(matches)==1
        selection.append((short+'_patient_comparison.png',matches[0]))
    complete_pair=W/'ellipse_yplus1_two_noise'
    early=complete_pair if (complete_pair/'manifest.json').exists() else W/'ellipse_yplus1_first_noise'
    if (early/'manifest.json').exists():
        p=next((early/'figures').glob('*patient_spectra_model_envelopes.png'))
        name='ellipse_yplus1_two_noise_patient_comparison.png' if early==complete_pair else 'ellipse_yplus1_first_noise_patient_comparison.png'
        selection.append((name,p))
    for mode in ['TA','TB']:
        p=N/'analysis/figures'/f'bridge_pair0_{mode}_participation_timing.png'
        if p.exists():selection.append(('new_network_'+mode+'.png',p))
    replay=W/'position_new_noise_replay/figures/position_effect_old_and_new_noise.png'
    if replay.exists():selection.append(('position_development_and_new_noise.png',replay))
    distributions=W/'bridge_event_distributions/figures/new_noise_event_delay_distributions.png'
    if distributions.exists():selection.append(('new_noise_event_delay_distributions.png',distributions))
    # Include actual new-network media as soon as the first-noise consumer has
    # completed a unit. The still-missing second noise is explicit in its panel.
    for topo in [2511,2711]:
        folder=N/'analysis/first_noise_review'/f'bridge_circle_out125_xminus075_topology{topo}'
        if (folder/'manifest.json').exists():
            p=next(folder.glob('*patient_spectra_model_envelopes.png'))
            selection.append((f'first_new_noise_topology{topo}_patient_comparison.png',p))
    axis_records=[read(p) for p in (G/'analysis/units').glob('*/result.json')]
    if any(r['counts']['candidate']!='global_axis_+0' for r in axis_records):
        for name in ['contact_timing_order','sampling_support']:
            p=G/'analysis/figures'/(name+'.png')
            if p.exists():selection.append(('global_axis_'+name+'.png',p))
        p=G/'analysis/figures/global_axis_-15/global_axis_-15_patient_spectra_model_envelopes.png'
        if (p.parent/'manifest.json').exists():selection.append(('global_axis_minus15_patient_comparison.png',p))
        p=G/'analysis/figures/global_axis_+15/global_axis_+15_patient_spectra_model_envelopes.png'
        if (p.parent/'manifest.json').exists():selection.append(('global_axis_plus15_patient_comparison.png',p))
    for name,source in [('core_contact_sampling.png',W/'core_contact_sampling/figures/core_contact_sampling.png'),('TB_core_timing.png',W/'core_phase_tb_fold/figures/core_t10_conditional_observations.png'),('TB_native_example.png',W/'tb_core_timing_native_review/figures/tb_event_60_field_core_readout_native_peak.png')]:selection.append((name,source))
    for name,source in selection:
        dest=out/'figures'/name
        if source is not None:shutil.copyfile(source,dest)
        files.append(dict(file=str(dest),source=str(source or Path(__file__)),sha256=sha(dest)))
    tables=[]
    for name,p in [('followup',FOLLOW),('multiseed',N),('global_axis',G)]:
        records=[read(q) for q in sorted((p/'analysis/units').glob('*/result.json'))]
        for r in records:tables.append(dict(series=name,**r['counts'],source=r['source'],source_arrays_sha256=r['source_sha256']))
    pd.DataFrame(tables).to_csv(out/'completed_run_summary.csv',index=False)
    status_lines=[]
    translations={'RESPONSE_RUNNING':'响应运行中','CONFIRMATION_RUNNING':'确认运行中','WAITING_FOR_SHARED_RESOURCES':'等待共享资源','ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW':'整轮完成待科学审阅','WINDOW_ELAPSED_DRAINING':'窗口结束，收尾已派发运行','WINDOW_ELAPSED_NOT_DISPATCHED':'窗口结束，保留未派发条件'}
    for name,title in [('followup','续跑当前阶段'),('multiseed','多网络响应'),('global_axis','方向探针')]:
        s=states[name];status_lines.append(f'{title}：{translations.get(s["status"],s["status"])}；当前阶段已完成 {s.get("complete","见状态表")}；运行中 {len(s.get("active",[]))}。')
    status_lines.append(f'续跑累计：响应 {followup_cumulative["response"]}/60；确认 {followup_cumulative["confirmation"]}/{followup_cumulative["confirmation_planned"] if followup_cumulative["confirmation_planned"] is not None else "待冻结提名"}。')
    window_statement='8小时自主窗口已结束；以下为实际完成与仍运行状态，不代表全部实验批次完成。' if window_close else '本图集是时间快照，不是8小时目标完成声明。'
    text='\n'.join(status_lines)+'\n\n'+window_statement+'''原140条已经完成；续跑中的12条历史复用不计为新增实验。
前1.5秒排除，单条正式运行是统计单位。冻结TA/TB聚类也参与损失的模式分布项；同标签评价不是独立验证。
目前接受部分参数—观测对应关系，尚不接受患者完整双模式恢复。SCL参与、时差和模式比例存在权衡。
局部宽度仍不足，TB多数事件的跨杆与ICL局部时序仍偏离患者；少数较好的接触顺序可伴两前沿相遇。
方向参数的实际图改变已验证，完整传播结果仍以该批实际结果为准。新拓扑重复的结论只使用实际完成的配对。
全部保持降阈值core、限核随机输入、Z/M与空间OU关闭、GABA18ms和冻结loss；本图集没有新提名或新物理。
损失设计系数：全体联合特征0.25、模式条件特征0.25、参与组合0.50；先各自正尺度归一化，不是方差解释率。
神经元输入不接收TA/TB标签、指定路线或事件时序；这与参数筛选使用患者模式信息是两回事。
患者真实STFT与模型发放密度包络不是同一信号，显示同一毫秒尺度不表示等价；15个触点始终固定分杆。
旋转候选仅为筛查；两个核心前沿相遇不能自动称为螺旋。
'''
    pdf=out/'overnight_review.pdf'
    with PdfPages(pdf) as book:
        fig=plt.figure(figsize=(14,9));fig.text(.06,.94,'患者双传播恢复与参数作用：'+('8小时探索结果' if window_close else '夜间审阅快照'),fontsize=19,va='top');fig.text(.06,.87,stamp.strftime('%Y-%m-%d %H:%M:%S %Z')+'；自主窗口 01:32–09:32',fontsize=11,va='top');fig.text(.06,.8,text,fontsize=10,va='top',linespacing=1.8);book.savefig(fig);plt.close(fig)
        for row in files:
            with Image.open(row['file']) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=plt.figure(figsize=(16,16*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');book.savefig(fig,dpi=155);plt.close(fig)
    write(out/'manifest.json',dict(created=stamp.isoformat(),window_elapsed_review=window_close,states=states,followup_cumulative=followup_cumulative,geometry=geometry,files=files,pdf_sha256=sha(pdf),producer=str(Path(__file__)),producer_sha256=sha(Path(__file__)),physical_runs_added_by_builder=0,acceptance='REVIEW_SNAPSHOT_NOT_MODEL_ACCEPTANCE'))
    note='# 夜间审阅快照\n\n'+stamp.isoformat()+'\n\n'+text.replace('\n','\n\n')+'\n\n[当前图集](overnight_review.pdf)按中心/范围、全参数、患者对照、新网络（若已有结果）、采样和核心时序排序。GIF见[两组合按时间选取的多事件动画](../workpoint_tradeoffs/figures/circle/README.md)及[同一TB标签内部的原生过程诊断](../tb_core_timing_native_review/figures/README.md)。\n\n全部完成运行和来源见completed_run_summary.csv，缺失几何条件见geometry_availability.json。用户人工目视验收待定。\n'
    (out/'scientific_review.md').write_text(note)
    captions=[]
    for name,source in selection:
        if name=='geometry_parameter_response.png':desc='固定基础网络和噪声，中心/范围探针减各自直接对照；横线只是连接两次噪声结果。各列独立量纲，未完成点不填0。'
        elif name=='radius_participation_and_activity.png':desc='范围参数的三个离散点在两工作点内分别配对，显示TA出现、参与、两类时序与左核群体发放量。匹配降阈值总量并未匹配随机输入支持或连接分块；缺失点不填0。'
        elif 'all_parameters' in name:desc='原140条中的14种局部参数干预，按固定形状和噪声配对。颜色为形状组合，点形为噪声，不把事件数视为网络重复。'
        elif name=='prior_confirmation_core_contact_timing.png':desc='原8条圆核／椭圆确认的TA事件，核心累计先后与电极杆间时差并排比较。160个TA均为左核窗口t10较早，154个可测杆间差；领先core未交换，电极时序仍改变。'
        elif name.startswith('prior_confirmation'):desc='原32条几何确认按两张网络、两次噪声拆开比较；某些TA条件摘要的几何效应跨网络反向。此处固定EI=1、向外EE=1，不能替代新局部参数候选的复测。'
        elif name=='position_development_and_new_noise.png':desc='圆核向外EE×1.25下，左核左移0.75mm的作用，在开发噪声和已完整的新噪声中逐条配对。固定基础网络2511；灰线是开发参照，彩线是新噪声，同位置静态数组已核对一致。'
        elif name=='new_noise_event_delay_distributions.png':desc='首条新噪声的原位置与左移条件，分别呈现ALL、TA、TB全部合格事件的杆间质心差和固定ICL接触对时差。每条线是一条运行，没有跨网络混池；杆内使用参与触点质心中位数，完整统计范围及方差见对应CSV。'
        elif name.startswith('first_new_noise'):desc='已完成的首条新噪声中，左移工作点与患者Fig2C的两类过程对照。第二条噪声尚未纳入本页并明确留空；单次重演不代表完整双噪声确认。'
        elif name.startswith('ellipse_yplus1_first_noise'):desc='新完成的椭圆左核再上移1mm条件，仅有第一噪声，第二列明确待完成。TA上部SCL招募更多但偏离患者参与参考，TB具体顺序仍未恢复；较低总分不是整体验收。'
        elif name.startswith('ellipse_yplus1_two_noise'):desc='椭圆左核再上移1mm条件的两条完整噪声，与患者固定Fig2C真实STFT同列比较。例子按模型自身均值选取；两噪声是同一网络重演，不是两张独立网络。'
        elif 'patient_comparison' in name:desc='患者固定Fig2C真实STFT与模型自身模式均值附近的事件比较。SCL/ICL按固定15行展示，未参与保留灰色，不拉伸时间。'
        elif 'new_network' in name:desc='已完整分析的新噪声和拓扑中的左移效应，各线固定基础网络与噪声。未完成条件保持缺失，不能用部分结果提前宣称全网一致。'
        elif name.startswith('global_axis'):desc='仅展示已完成20秒方向条件，保持原三点横轴与缺失状态。角度改变实际EE图和距离时延；开发噪声下的响应不等于新拓扑确认。'
        else:desc='冻结输出的核心时序、空间采样或全场活动诊断。核心后验时序不等于因果起源，几何权重不等于事件信号贡献。'
        captions.append(f'### {name}\n{desc}\n**关注点**：参数是否同时改善参与和实际传播，及改善是否跨运行保留；不能仅凭较低loss或两个标签接受双模式恢复。\n')
    (out/'figures/README.md').write_text('\n'.join(captions));print(out,flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label');p.add_argument('--window-close',action='store_true');args=p.parse_args();build(args.label,args.window_close)

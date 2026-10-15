"""Streaming paired parameter observables. Events never substitute for network replicates."""
from pathlib import Path
import sys,time,hashlib,json,argparse,copy,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
from scripts import run_topic4_shape_output_response as run
from scripts import analyze_topic4_core_connectivity_search as an
from scripts import analyze_topic4_propagation_recovery_night as rev
from scripts.paper_figures import plot_topic4_recovery_review as figreview
rt=run.rt;OUT=run.OUT;A=OUT/'analysis';F=A/'figures';VERSION='shape_response_observer_v1'
PARAM_NAMES=dict(an.PARAM_ZH,core_mean_rate_scale='核内平均外部输入 ×',core_ou_correlation='两核 OU 输入相关系数')
SHAPE_NAMES=dict(circle='圆形对照',ellipse4='沿端点轴拉长 4:1',ellipse9='沿端点轴拉长 9:1',ellipse4_orthogonal='转90°拉长 4:1',radius25='左核半径 2.5 mm',
    radius25_dose_matched='半径 2.5 mm＋降阈值总量匹配',out_reference='离核重采样对照',out_perp15='左核输出横向范围 ×1.5',out_perp20='左核输出横向范围 ×2',
    out_parallel15='左核输出纵向范围 ×1.5',out_angle_m15='左核输出方向 −15°',out_angle_p15='左核输出方向 +15°',ellipse4_out_perp20='拉长 4:1＋输出横向 ×2')
METRICS=[('SCL_upper_participation','SCL9/8 参与概率'),('ICL_contact_participation','ICL 平均参与概率'),('both_rods','两杆共同参与概率'),
    ('participation_mae','通道参与概率误差'),('pair_order_probability_mae','成对顺序概率误差'),('SCL_minus_ICL_lag_median_ms','SCL−ICL 质心时差 (ms)'),
    ('local_width_ms_median','接触点局部宽度 (ms)'),('recruitment_span_ms_median','跨接触点 t10 跨度 (ms)'),('B_minus_A_t10_ms_median','右−左核 t10 (ms)'),
    ('first_10pct_mass_core_share_median','早期10%活动的核内份额'),('peak_active_components_median','峰值并行连通域数'),('n','本类合格事件数')]
SPACE_METRICS=[('union_active_area_mm2_median','窗口内活动并集面积 (mm²)'),('peak_active_area_mm2_median','峰值活动面积 (mm²)'),
    ('native_spatial_rms_mm_median','累计活动空间离散度 (mm)'),('peak_largest_component_fraction_median','峰值最大连通域活动份额')]


def title(c):
    prefix={'endpoint':'端点原位','up3':'左核上移3 mm'}[c['layout']]
    text=c['contrast']
    if '=' in text:
        k,v=text.split('=');text=PARAM_NAMES.get(k,k)+' '+v
        prefix+='｜'+SHAPE_NAMES[c['comparison'].split('__')[-1]]
    else:text=SHAPE_NAMES.get(text,text)
    return prefix+'｜'+text


def load_reference():
    parent=rt.read(run.base.PARENT);ev=rt.load_evaluator(parent);obj=rt.load_objective(parent)
    patient=np.asarray(ev.fit);plabel=np.asarray(ev.fit_labels);names=np.asarray(rt.load_observation_contract(parent)['contact_names'])
    maskref=an.jm.MaskReference(np.isfinite(patient));cal=rt.read(run.OLD/'analysis/mask_score_calibration.json');maskref.a_mask=cal['a_mask']
    refs={mode:patient if label is None else patient[plabel==label] for mode,label in [('ALL',None),('TA',1),('TB',0)]}
    pairs={m:an.pair_table(x) for m,x in refs.items()}
    rt.write(A/'patient_reference.json',dict(fit_n=len(patient),contact_order=names.tolist(),modes={m:dict(an.measures(x,x,names),pairs=[dict(i=int(i),j=int(j),n=n,p=p) for (i,j),(n,p) in pairs[m].items()]) for m,x in refs.items()},
        a_mask=cal['a_mask'],source=str(run.base.PARENT),definition='Frozen FIT natural event distribution; TA/TB labels organize conditional comparisons after simulation. Geometry prior is not independent validation.'))
    return obj,maskref,names,refs,pairs


def summarize(v):
    x=np.asarray(v,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,median=None,variance=None,q05=None,q95=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=float(x.var(ddof=1)) if len(x)>1 else None,q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))


def process(path,reference):
    obj,maskref,names,refs,refpairs=reference;r,a,ids=an.load_unit(path,1500.);allids=an.all_detected_ids(r,1500.);job=r['job'];c=rt.read(OUT/'candidates'/f'{job["candidate"]}.json')
    key=hashlib.sha256(str(path).encode()).hexdigest()[:20];dest=A/'units'/key;dest.mkdir(parents=True,exist_ok=True)
    root=dict(candidate=c['id'],stage=job['stage'],topology=job['topology_seed'],noise=job['dynamics_seed'],comparison=c['comparison'],description=title(c))
    events=[];byevent={};nc=len(a['group_coreAE'])+len(a['group_coreBE']);no=len(a['group_surroundE'])
    for i in allids:
        rec=dict(root,event=int(i),primary=bool(i in ids),mode='TA' if a['event_mode'][i]==1 else 'TB',event_time_ms=float(a['event_time_ms'][i]),**rev.core_timing(r,a,i))
        rec.update(an.event_timing(r,a,i,names) or {});rec.update(an.native_diag(r,a,i,nc,no) or {})
        lo,hi=r['events'][i]['window_ms'];movie=a['sheet_activity_counts'][round(lo/2):round(hi/2)].astype(float)
        activity=movie>=2;mass=movie.sum(0);xx,yy=np.meshgrid(np.arange(20)+.5,np.arange(20)+.5)
        rec['union_active_area_mm2']=float(activity.any(0).sum());rec['peak_active_area_mm2']=float(activity[movie.sum((1,2)).argmax()].sum())
        if mass.sum()>0:
            pts=np.stack([xx.ravel(),yy.ravel()],1);center=np.average(pts,weights=mass.ravel(),axis=0);dd=pts-center
            cov=(dd.T*mass.ravel())@dd/mass.sum();vals,vec=np.linalg.eigh(cov)
            rec.update(native_spatial_rms_mm=float(np.sqrt(np.trace(cov))),native_activity_axis_deg=float(np.rad2deg(np.arctan2(vec[1,-1],vec[0,-1]))%180),native_axis_ratio=float(np.sqrt(vals[-1]/max(vals[0],1e-12))))
        events.append(rec);byevent[int(i)]=rec
    observations=[];contacts=[];pairrows=[]
    for layer,baseids in [('primary',ids),('all_detected',allids)]:
      for mode,label in [('ALL',None),('TA',1),('TB',0)]:
        selected=baseids if label is None else baseids[a['event_mode'][baseids]==label];x=a['centroid_ms'][selected]
        rec=dict(root,layer=layer,mode=mode,**an.measures(x,refs[mode],names));pp=an.pair_table(x);errs=[]
        for (i,j),(n,p) in pp.items():
            pn,pr=refpairs[mode][(i,j)];err=None if p is None or pr is None else abs(p-pr)
            if err is not None:errs.append(err)
            ok=np.isfinite(x[:,i])&np.isfinite(x[:,j]);lags=summarize(x[ok,j]-x[ok,i])
            pairrows.append(dict(root,layer=layer,mode=mode,contact_i=names[i],contact_j=names[j],model_joint_n=n,patient_joint_n=pn,model_i_precedes_j=p,patient_i_precedes_j=pr,absolute_probability_error=err,**{'lag_'+k:v for k,v in lags.items()}))
        rec.update(pair_order_probability_mae=float(np.mean(errs)) if errs else None,pair_order_supported_pairs=len(errs))
        for name in ['local_width_ms','recruitment_span_ms','B_minus_A_t10_ms','B_minus_A_t50_ms','first_10pct_mass_core_share','peak_active_components','peak_largest_component_fraction','union_active_area_mm2','peak_active_area_mm2','native_spatial_rms_mm']:
            stats=summarize([byevent[int(i)].get(name) for i in selected])
            rec.update({name+'_'+k:v for k,v in stats.items()})
        rank=an.ranks(x)
        for j,name in enumerate(names):
            contacts.append(dict(root,layer=layer,mode=mode,contact=name,events=len(x),participation=float(np.isfinite(x[:,j]).mean()) if len(x) else None,
                patient_participation=float(np.isfinite(refs[mode][:,j]).mean()),**{'rank_'+k:v for k,v in summarize(rank[:,j]).items()}))
        observations.append(rec)
    score=obj.score_network(a['centroid_ms'][ids]) if len(ids) else {};ms=an.jm.score_times(a['centroid_ms'][ids],maskref) if len(ids) else {}
    combined=an.jm.combined_search_loss(score.get('loss_off'),ms.get('D_mask_off'),maskref.a_mask)
    if len(ids)<16:combined=None
    counts=dict(root,physical_status=r['physical_status'],duration_ms=r['actual_duration_ms'],primary=len(ids),TA=int((a['event_mode'][ids]==1).sum()),TB=int((a['event_mode'][ids]==0).sum()),
        **an.detection_layer(r,a,1500.),**an.core_activity(a,1500.,r['actual_duration_ms']),L_search=combined,L_off=score.get('loss_off'),L_off_A=score.get('loss_off_A_component'),L_off_B=score.get('loss_off_B_subtraction'),
        L_D16=score.get('loss_D16'),D_mask_off=ms.get('D_mask_off'))
    # Six-second temporal bins use actual observation time and retain empty segments.
    segments=[]
    for lo in np.arange(1500,r['actual_duration_ms'],6000):
        hi=min(lo+6000,r['actual_duration_ms']);ix=ids[(a['event_time_ms'][ids]>=lo)&(a['event_time_ms'][ids]<hi)]
        segments.append(dict(root,start_ms=lo,end_ms=hi,n=len(ix),TA=int((a['event_mode'][ix]==1).sum()),TB=int((a['event_mode'][ix]==0).sum()),intervals_ms=np.diff(a['event_time_ms'][ix]).tolist()))
    rt.write(dest/'result.json',dict(version=VERSION,source=str(path),source_sha256=r['arrays_sha256'],counts=counts,observations=observations,contacts=contacts,pairs=pairrows,events=events,segments=segments,score=score,mask_score=ms))
    print(json.dumps(dict(candidate=c['id'],topology=root['topology'],noise=root['noise'],primary=len(ids),TA=counts['TA'],TB=counts['TB'],L_search=combined)),flush=True)


def aggregate():
    results=[rt.read(p) for p in (A/'units').glob('*/result.json')];rotations={r['source']:r for p in (OUT/'rotation').glob('*/result.json') for r in [rt.read(p)]}
    tables={k:[r2 for r in results for r2 in (r[k] if isinstance(r[k],list) else [r[k]])] for k in ['counts','observations','contacts','pairs','events','segments']}
    for r in results:
        z=rotations.get(r['source']);row=r['counts']
        row['rotation_time_fraction']=None if z is None else z['candidate_time_fraction'];row['rotation_tracks_per_minute']=None if z is None else z['candidate_tracks_per_minute']
        row['max_fixed_ring_turns']=None if z is None else z['maximum_fixed_ring_turns']
    for k,rows in tables.items():an.writecsv(A/(k+'.csv'),rows)
    lookup={(r['candidate'],r['topology'],r['noise'],r['layer'],r['mode']):r for r in tables['observations']};diffs=[]
    for row in tables['observations']:
        if row['comparison']==row['candidate']:continue
        ref=lookup.get((row['comparison'],row['topology'],row['noise'],row['layer'],row['mode']))
        if ref is None:continue
        out={k:row[k] for k in ['candidate','comparison','stage','topology','noise','layer','mode','description']}
        out.update(n_condition=row['n'],n_reference=ref['n'])
        for key,label in METRICS+SPACE_METRICS:
            x,y=row.get(key),ref.get(key);out[key]=None if x is None or y is None else x-y
        diffs.append(out)
    an.writecsv(A/'paired_differences.csv',diffs)
    clook={(r['candidate'],r['topology'],r['noise']):r for r in tables['counts']};cd=[]
    for row in tables['counts']:
        ref=clook.get((row['comparison'],row['topology'],row['noise']))
        if ref is None or row['candidate']==row['comparison']:continue
        d={k:row[k] for k in ['candidate','comparison','topology','noise','description']}
        for key in ['L_search','TA','TB','primary','rotation_time_fraction','rotation_tracks_per_minute','max_fixed_ring_turns','coreAE_rate_hz','coreBE_rate_hz','surroundE_rate_hz']:
            x,y=row.get(key),ref.get(key);d[key]=None if x is None or y is None else x-y
        cd.append(d)
    an.writecsv(A/'paired_run_differences.csv',cd)
    evidence=[]
    groups={}
    for row in diffs:
        if row['layer']=='primary':groups.setdefault((row['candidate'],row['mode']),[]).append(row)
    for (cid,mode),rows in groups.items():
        for key,label in METRICS+SPACE_METRICS:
            valid=[r for r in rows if r.get(key) is not None];values=np.asarray([r[key] for r in valid],float)
            if not len(valid):continue
            evidence.append(dict(candidate=cid,description=rows[0]['description'],comparison=rows[0]['comparison'],mode=mode,observable=key,observable_zh=label,
                paired_runs=len(valid),topologies=len({r['topology'] for r in valid}),networks_noise_min_event_support=min(min(r['n_condition'],r['n_reference']) for r in valid),
                positive_pairs=int((values>0).sum()),negative_pairs=int((values<0).sum()),zero_pairs=int((values==0).sum()),mean_paired_change=float(values.mean()),median_paired_change=float(np.median(values)),
                minimum_paired_change=float(values.min()),maximum_paired_change=float(values.max()),interpretation='descriptive paired effects; events not independent network replicates; direction consistency is not significance'))
    an.writecsv(A/'response_evidence.csv',evidence)
    for mode in ['ALL','TA','TB']:
      for stage in ['shape_range','response','confirmation']:
        subset=[r for r in diffs if r['mode']==mode and r['layer']=='primary' and r['stage']==stage]
        if not subset:continue
        candidates=list(dict.fromkeys(r['candidate'] for r in subset))
        fig,axes=plt.subplots(3,4,figsize=(18,max(10,len(candidates)*.28)),layout='constrained')
        for ax,(key,label) in zip(axes.flat,METRICS):
            for j,cid in enumerate(candidates):
                for row in [r for r in subset if r['candidate']==cid]:
                    v=row.get(key)
                    if v is None:continue
                    color={2511:'#222222',2611:'#5268ae',2612:'#d17c36'}[row['topology']];marker='o' if row['noise']%2 else '^'
                    ax.scatter(v,j+(.11 if marker=='o' else -.11),c=color,marker=marker,s=18,alpha=.8)
            ax.axvline(0,color='gray',lw=.7);ax.set(yticks=range(len(candidates)),yticklabels=[next(r['description'] for r in subset if r['candidate']==cid) for cid in candidates],xlabel='相对直接对照的变化：'+label)
            ax.invert_yaxis();ax.tick_params(labelsize=6);ax.grid(axis='x',alpha=.2)
        fig.suptitle(f'{stage}｜{mode}｜每点为同拓扑、同噪声的一对运行之差\n黑/蓝/橙=拓扑2511/2611/2612；圆/三角=两次噪声。事件数与指标并列，少数模式证据有限时不作机制结论。',fontsize=11)
        for ext in ['png','pdf']:fig.savefig(F/f'{stage}_{mode}_paired_response.{ext}',dpi=130)
        plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(18,max(6,len(candidates)*.28)),layout='constrained')
        for ax,(key,label) in zip(axes,SPACE_METRICS):
            for row in subset:
                value=row.get(key)
                if value is None:continue
                ax.scatter(value,candidates.index(row['candidate'])+(.1 if row['noise']%2 else -.1),color={2511:'black',2611:'#5268ae',2612:'#d17c36'}[row['topology']],marker='o' if row['noise']%2 else '^',s=18)
            ax.axvline(0,color='gray',lw=.7);ax.set(yticks=range(len(candidates)),yticklabels=[next(r['description'] for r in subset if r['candidate']==cid) for cid in candidates],xlabel='相对直接对照的变化：'+label);ax.invert_yaxis();ax.tick_params(labelsize=6)
        fig.suptitle(f'{stage}｜{mode}｜原生场空间范围与集中程度\n活动面积阈值固定为每1mm网格、2ms内≥2个活跃E神经元；面积和空间离散不等于因果传播距离。',fontsize=11)
        for ext in ['png','pdf']:fig.savefig(F/f'{stage}_{mode}_native_space_response.{ext}',dpi=130)
        plt.close(fig)
    if cd:
        fig,axes=plt.subplots(1,4,figsize=(17,max(6,len(cd)*.13)),layout='constrained')
        ids=list(dict.fromkeys(r['candidate'] for r in cd))
        for ax,(key,label) in zip(axes,[('TA','TA事件数'),('TB','TB事件数'),('rotation_time_fraction','旋转候选占用时间比例'),('rotation_tracks_per_minute','旋转候选轨迹/分钟')]):
            for row in cd:
                if row.get(key) is None:continue
                ax.scatter(row[key],ids.index(row['candidate'])+(.1 if row['noise']%2 else -.1),marker='o' if row['noise']%2 else '^',color={2511:'black',2611:'#5268ae',2612:'#d17c36'}[row['topology']],s=16)
            ax.axvline(0,c='gray',lw=.6);ax.set(yticks=range(len(ids)),yticklabels=[next(r['description'] for r in cd if r['candidate']==cid) for cid in ids],xlabel='相对直接对照变化：'+label);ax.invert_yaxis();ax.tick_params(labelsize=6)
        fig.suptitle('旋转是独立观测：相位＋原生活动环检出的候选；轨迹可能分段，不能当成独立螺旋事件',fontsize=11)
        for ext in ['png','pdf']:fig.savefig(F/f'rotation_and_mode_response.{ext}',dpi=130)
        plt.close(fig)
    rt.write(A/'status.json',dict(analyzed_runs=len(results),rotation_analyzed_new=sum(r['source'] in rotations for r in results),paired_observation_rows=len(diffs),updated_unix=time.time()))
    (A/'scientific_note.md').write_text('# core形状与离核连接：持续更新的实验报告\n\n'+f'已分析正式运行 {len(results)}/140。当前为阶段结果，尚未完成科学验收。\n\n'+
        '问题是：在固定端点先验下，改变局部易激区的形状和连接范围，能否同时改善患者两杆参与和两种条件传播，而不是仅增加TA/TB标签。每次参数效应以同一拓扑、同一噪声的直接对照差表示；确认阶段使用两个新拓扑×两个新噪声。事件是运行内观测，不能当作独立网络。\n\n'+
        'counts.csv保留物理状态、全部检测/合格事件、TA/TB数和原冻结loss；observations.csv逐模式、逐观测保留均值/方差或中位数/5–95%范围；contacts.csv与pairs.csv保留各通道参与、rank分布与成对时差/顺序支持量。events.csv保留原生活动和时间分解；segments.csv检查6秒段内模式比例。paired_differences.csv使用真实量纲，未合成总体恢复率。\n\n'+
        '旋转候选采用相位绕转、原生活动固定环、持续时间和方向一致性联合筛查；两个非旋转波源叠加也可能通过，因而这些数值不能证明稳定螺旋波。原生场GIF和灵敏度应一起审阅。该观测不进入患者拟合loss。\n\n'+
        'core拉长保持E成员数与阈值降低总量，但成员身份、I邻域和空间分布随形状改变；扩大半径同时扩大随机输入支持，即便匹配阈值总量也不等于匹配总随机驱动。局部离核范围比较应使用同规则重采样对照，它保留各A源的出度与总权重，目标入度与时延可以变化。\n\n'+
        '患者Fig2C为真实STFT；模型仍为发放密度包络，并非模型HFO频谱。相同毫秒轴和固定15行、分杆质心线不改变这一信号层区别。所有传播与旋转结论待完整分布和原生场审阅；不自动冻结模型或进入Fig5。\n')
    if results:
        bycondition={}
        for row in tables['counts']:
            if row['topology']==2511:bycondition.setdefault(row['candidate'],[]).append(row)
        ranked=[(np.mean([r['L_search'] for r in rr]),cid,rr) for cid,rr in bycondition.items() if len(rr)==2 and all(r['L_search'] is not None for r in rr)]
        ranked.sort(key=lambda x:x[0])
        with (A/'scientific_note.md').open('a') as f:
            f.write('\n## 本次自动汇总的实际证据\n\n'+f'现有 {len(evidence)} 条逐指标配对响应摘要，完整数据见response_evidence.csv；每条注明成对运行数、拓扑数、正/负/零变化数量及最少事件支持。方向相同只是可重复性线索，不是显著性检验。\n\n')
            for value,cid,rr in ranked[:3]:
                f.write(f'- 当前可评分候选：{rr[0]["description"]}。同拓扑两噪声平均冻结联合分数 {value:.3f}；TA数 {[r["TA"] for r in rr]}，TB数 {[r["TB"] for r in rr]}。这是分布评分提名，不是传播或螺旋验收。\n')
            f.write('\n后续审阅应先看两种模式各自的患者残差是否同时减小，再检查原生场是否支持该解释；若仅参与或标签比例增加，仍不能接受完整传播恢复。新拓扑确认按照预定条件执行，不因这里的排序而更换。\n')


def report_pdf():
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    status=rt.read(A/'status.json');figs=list(sorted(F.glob('*.png')))
    keys=rt.read(OUT/'plan.json')['confirmation']['candidates']
    for cid in keys:
        for folder in sorted(F.glob(cid+'_topology*')):figs.extend(sorted(folder.glob('*same_network.png')));figs.extend(sorted(folder.glob('*patient_spectra_model_envelopes.png')))
    temp=A/'parameter_propagation_report.tmp.pdf'
    with PdfPages(temp) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.08,.9,'core空间形状与离核输出：参数—观测实验',fontsize=20)
        lines=[f'阶段报告：已分析 {status["analyzed_runs"]}/140 条正式运行；尚待完整科学审阅。',
            '问题：形状/空间范围是否改善两杆与两模式传播？哪些参数效应跨噪声、拓扑保留？',
            '实验：52条形状/范围；56条两种形状下的参数响应；32条新拓扑/新噪声确认。',
            '每条60秒；先排除1.5秒。运行是配对单位，事件是运行内样本。',
            '图中黑/蓝/橙分别为拓扑2511/2611/2612；圆/三角为两次噪声。',
            '逐项显示参与、顺序、时差、局部宽度、空间范围和旋转候选；不合成为总体恢复率。',
            '患者真实STFT与模型发放密度包络保持相同毫秒轴；它们不是相同物理信号。',
            '核心、TA、TB、完整读出沿用同一网络；各类示例取自身均值附近。',
            '旋转候选存在双波源叠加假阳性；需结合原生场GIF，不能直接认定稳定螺旋。',
            '自动更新本报告及CSV/JSON；GIF、完整分布与实际配置见同目录。',
            '当前未冻结患者双模式工作点，未进入Fig5。']
        for i,line in enumerate(lines):fig.text(.08,.80-i*.058,line,fontsize=11)
        pdf.savefig(fig);plt.close(fig)
        for path in figs:
            with Image.open(path) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=plt.figure(figsize=(14,14*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=130);plt.close(fig)
    temp.replace(A/'parameter_propagation_report.pdf');rt.write(A/'report_manifest.json',dict(figures=[str(p) for p in figs],analysis_status=status,updated_unix=time.time(),pdf_sha256=rt.sha(A/'parameter_propagation_report.pdf')))


def geometry(ax,c,a,contact_labels=False):
    # Use the canonical contact layout and replace only the core outline by actual applied geometry.
    before=len(ax.patches);original_geometry(ax,c,a,contact_labels)
    for patch in list(ax.patches)[before:]:patch.remove()
    ap=c['_applied_threshold'];aa,bb=ap['ellipse_A_semiaxes_mm'];ax.add_patch(Ellipse(c['centers_mm'][0],2*aa,2*bb,angle=ap['ellipse_A_angle_deg'],fill=False,ec='#c94a44',lw=.95,zorder=7))
    ax.add_patch(Circle(c['centers_mm'][1],c['radii_mm'][1],fill=False,ec='#c94a44',lw=.95,zorder=7))
original_geometry=figreview.geometry;figreview.geometry=geometry


def review_figures():
    results=[rt.read(p) for p in (A/'units').glob('*/result.json')];groups={}
    for r in results:
        j=r['counts'];groups.setdefault((j['candidate'],j['topology']),[]).append(r)
    patient=None;manifests=[]
    for (cid,topo),rr in groups.items():
        if len(rr)<2:continue
        folder=F/f'{cid}_topology{topo}';folder.mkdir(exist_ok=True)
        if (folder/'manifest.json').exists() and not any(x.get('status')=='PENDING_ROTATION' for x in rt.read(folder/'manifest.json')):continue
        c=rt.read(OUT/'candidates'/f'{cid}.json');c.update(topology=topo,display_name=title(c));units={}
        for rec in rr:
            path=Path(rec['source']);units[rec['counts']['noise']]=an.load_unit(path,1500.)
        seeds=sorted(units);path=Path(rr[0]['source']);physics=rt.read(path.parents[1]/'applied_physics.json');c['_applied_threshold']=physics['threshold']
        if patient is None:patient=figreview.patient_payloads()
        manifest=[figreview.spectral_comparison(c,units,seeds,folder,patient,'primary')]
        for seed in seeds:
            r,a,ids=units[seed];manifest.append(figreview.four_panel(c,seed,r,a,ids,folder,physics,'primary'))
        if cid in rt.read(OUT/'plan.json')['confirmation']['candidates']:
            from scripts.render_topic4_shape_output_gifs import render,rotation_clip
            seed=seeds[0];r,a,ids=units[seed];manifest.append(render(c,seed,r,a,ids,physics,folder,patient))
            source=next(rec['source'] for rec in rr if rec['counts']['noise']==seed)
            rotation=OUT/'rotation'/hashlib.sha256(source.encode()).hexdigest()[:20]/'result.json'
            if rotation.exists():manifest.append(rotation_clip(c,seed,r,a,physics,folder,rt.read(rotation)))
            elif r['actual_duration_ms']>=20000:manifest.append(dict(status='PENDING_ROTATION'))
        rt.write(folder/'manifest.json',manifest);manifests.extend(manifest)
        (folder/'README.md').write_text('# 患者与模型逐运行审阅\n\n'+''.join('### '+p.name+'\n患者使用固定Fig2C真实STFT，模型为原始发放密度包络；示例及GIF选择规则见manifest.json。固定15行按杆分组，未参与行保留；原生场叠加实际core形状，包含全部E活动。\n**关注点**：同时检查两杆招募、接触时序、局部时间宽度与原生传播；标签一致不等于恢复。\n\n' for p in sorted(folder.iterdir()) if p.suffix in ['.png','.gif']))
    if manifests:print(json.dumps(dict(new_review_panels=len(manifests))),flush=True)


def main(once=False):
    A.mkdir(exist_ok=True);F.mkdir(exist_ok=True);(A/'units').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':8,'pdf.fonttype':42})
    with (A/'observer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reference=load_reference();last=0
        while True:
            paths=[p for stage in ['shape_range','response','confirmation'] for p in (OUT/stage).glob('units/*/*/workers/trajectory.json')]
            changed=False
            for path in sorted(paths):
                key=hashlib.sha256(str(path).encode()).hexdigest()[:20]
                if (A/'units'/key/'result.json').exists():continue
                process(path,reference);changed=True
            if changed or time.time()-last>300:
                aggregate();review_figures();report_pdf();last=time.time()
                if list(F.glob('*.png')):
                    (F/'README.md').write_text('# 参数—观测图\n\n'+''.join('### '+p.name+'\n图中各点为同拓扑、同噪声相对直接对照的变化，量纲保留在横轴。原位与上移、圆核与椭圆核使用明确对照，缺失观测不填零。\n**关注点**：方向能否跨噪声和新拓扑重复，以及事件支持是否足够。\n\n' for p in sorted(F.glob('*.png'))))
            if (OUT/'simulation_complete.json').exists() and len(paths)==140:
                # GPU consumers finish first; keep final state honest about pending rotation analysis.
                done=sum((OUT/'rotation'/hashlib.sha256(str(p).encode()).hexdigest()[:20]/'result.json').exists() for p in paths if rt.read(p)['actual_duration_ms']>=20000)
                expected=sum(rt.read(p)['actual_duration_ms']>=20000 for p in paths)
                if done==expected:
                    aggregate();review_figures();report_pdf();rt.write(OUT/'status.json',dict(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW',formal_runs=140,analysis_runs=len(paths),rotation_runs=done,short_rotation_not_estimable=140-expected,updated_unix=time.time()));break
            if once:break
            time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args();main(a.once)

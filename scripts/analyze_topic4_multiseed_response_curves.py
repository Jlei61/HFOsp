"""Run-level, fully crossed parameter curves. No redefinition of patient fitting."""
from pathlib import Path
import argparse,fcntl,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scripts import run_topic4_multiseed_response_curves as run
from scripts import analyze_topic4_shape_output_response as an
OUT=run.OUT;rt=run.rt;A=OUT/'analysis';F=A/'figures'
COLORS={2511:'#666666',2711:'#3478b8',2712:'#cf7b2b'}


def configure():
    from src.topic4_pdf_font_guard import install
    install()
    an.OUT=OUT;an.A=A;an.F=F;an.run.OUT=OUT;an.title=lambda c:c['display_name']
    an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    for d in [A,F,A/'units']:d.mkdir(parents=True,exist_ok=True)


def aggregate():
    results=[rt.read(p) for p in sorted((A/'units').glob('*/result.json'))]
    tables={k:[z for r in results for z in (r[k] if isinstance(r[k],list) else [r[k]])] for k in ['counts','observations','contacts','pairs','events','segments']}
    physics=[]
    for r in results:
        path=Path(r['source']);h=hashlib.sha256(str(path).encode()).hexdigest()[:20];rot=OUT/'rotation'/h/'result.json'
        for key in ['rotation_time_fraction','rotation_tracks_per_minute','max_fixed_ring_turns']:r['counts'][key]=None
        if rot.exists():
            z=rt.read(rot);r['counts'].update(rotation_time_fraction=z['candidate_time_fraction'],rotation_tracks_per_minute=z['candidate_tracks_per_minute'],max_fixed_ring_turns=z['maximum_fixed_ring_turns'])
        ap=rt.read(path.parents[1]/'applied_physics.json')
        physics.append(dict(candidate=r['counts']['candidate'],topology=r['counts']['topology'],noise=r['counts']['noise'],identity=ap['identity'],threshold=ap['threshold'],group_counts=ap['group_counts'],input=ap['input'],degree=ap['graph']['stage_audits']['degree'],blocks=ap['graph']['block_summary']))
    # A fixed graph replay holds all actual static arrays fixed, not the trajectory.
    identities={}
    for p in physics:
        key=(p['candidate'],p['topology'])
        if key in identities and identities[key]!=p['identity']:raise RuntimeError(f'static replay identity mismatch: {key}')
        identities[key]=p['identity']
    rt.write(A/'applied_physics_by_run.json',physics)
    for k,v in tables.items():an.an.writecsv(A/(k+'.csv'),v)
    plan=rt.read(OUT/'plan.json');lookup={(r['candidate'],r['topology'],r['noise'],r['layer'],r['mode']):r for r in tables['observations']}
    contrasts=[]
    for fam in plan['families']:
        baseline=fam['candidates'][-1] if fam['axis']=='EI' else fam['candidates'][0]
        contrasts.extend((fam['id'],cid,baseline) for cid in fam['candidates'] if cid!=baseline)
    contrasts.extend((p['question'],p['candidate'],p['reference']) for p in plan['pairs'])
    diffs=[]
    for family,cid,base in contrasts:
      for topo in plan['topology_seeds']:
       for noise in plan['seeds']:
        for layer in ['primary','all_detected']:
         for mode in ['ALL','TA','TB']:
            r=lookup.get((cid,topo,noise,layer,mode));b=lookup.get((base,topo,noise,layer,mode))
            if r is None or b is None:continue
            d=dict(family=family,candidate=cid,reference=base,topology=topo,noise=noise,layer=layer,mode=mode,n=r['n'],reference_n=b['n'])
            for k,_ in an.METRICS+an.SPACE_METRICS:
                x,y=r.get(k),b.get(k);d[k]=None if x is None or y is None else x-y
            diffs.append(d)
    an.an.writecsv(A/'paired_curve_differences.csv',diffs)
    evidence=[]
    for family,cid,base in contrasts:
      for mode in ['ALL','TA','TB']:
       for metric,label in an.METRICS+an.SPACE_METRICS:
        rows=[r for r in diffs if r['family']==family and r['candidate']==cid and r['mode']==mode and r['layer']=='primary' and r.get(metric) is not None]
        if not rows:continue
        v=np.asarray([r[metric] for r in rows]);evidence.append(dict(family=family,candidate=cid,reference=base,mode=mode,metric=metric,metric_zh=label,paired_runs=len(rows),topologies=len({r['topology'] for r in rows}),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),median_change=float(np.median(v)),min_change=float(v.min()),max_change=float(v.max()),minimum_support=min(min(r['n'],r['reference_n']) for r in rows)))
    an.an.writecsv(A/'response_repeatability.csv',evidence)
    return tables


def curves(tables):
    if not tables['counts']:return
    plan=rt.read(OUT/'plan.json');ref=rt.read(A/'patient_reference.json');count={(r['candidate'],r['topology'],r['noise']):r for r in tables['counts']}
    lookup={(r['candidate'],r['topology'],r['noise'],r['mode']):r for r in tables['observations'] if r['layer']=='primary'}
    def value(cid,topo,noise,mode,key):
        if key=='mode_fraction':
            r=count.get((cid,topo,noise));return np.nan if not r or not r['primary'] else r[mode]/r['primary']
        r=(count if key.startswith('rotation_') or key=='L_search' else lookup).get((cid,topo,noise) if key.startswith('rotation_') or key=='L_search' else (cid,topo,noise,mode))
        v=None if r is None else r.get(key);return np.nan if v is None else v
    groups=[('participation_timing',an.METRICS[:6]),('native_time_space',an.METRICS[6:11]+an.SPACE_METRICS),('support_rotation',[('n','本类合格事件数'),('mode_fraction','本类 / 合格事件'),('L_search','冻结分布训练分数'),('rotation_time_fraction','旋转候选时间比例')])]
    captions=[]
    for fam in plan['families']:
      if not any(r['candidate'] in fam['candidates'] for r in tables['counts']):continue
      for mode in ['ALL','TA','TB']:
       for group,metrics in groups:
        use=[m for m in metrics if mode!='ALL' or m[0]!='mode_fraction'];nc=3;nr=int(np.ceil(len(use)/nc))
        fig,axes=an.plt.subplots(nr,nc,figsize=(14,3.35*nr+1.4),squeeze=False);fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.13,hspace=.48,wspace=.33)
        for ax,(key,label) in zip(axes.flat,use):
            for topo in plan['topology_seeds']:
             for i,noise in enumerate(plan['seeds']):
                ys=[value(cid,topo,noise,mode,key) for cid in fam['candidates']]
                ax.plot(fam['values'],ys,c=COLORS[topo],ls='-' if i==0 else '--',marker='o' if i==0 else '^',lw=1.4,ms=4)
            patient=None if key=='n' else ref['modes'][mode].get(key)
            if key=='mode_fraction':patient=ref['modes'][mode]['n']/ref['fit_n']
            if patient is not None:ax.axhline(patient,c='black',ls=':',lw=1)
            ax.set(xlabel='核内 E→I 倍率' if fam['axis']=='EI' else '核向外 E→E 权重倍率',ylabel=label,xticks=fam['values']);ax.tick_params(axis='x',labelsize=7);ax.grid(alpha=.18)
            if key in ['SCL_upper_participation','ICL_contact_participation','both_rods','mode_fraction']:ax.set_ylim(-.02,1.04)
            if key=='SCL_minus_ICL_lag_median_ms':ax.set_ylim(-40,130)
        for ax in list(axes.flat)[len(use):]:ax.axis('off')
        handles=[Line2D([],[],c=COLORS[x],label='网络 '+str(x)) for x in plan['topology_seeds']]+[Line2D([],[],c='#333',marker='o',label='噪声 847401'),Line2D([],[],c='#333',ls='--',marker='^',label='噪声 847402'),Line2D([],[],c='black',ls=':',label='患者 FIT 参考')]
        fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False,fontsize=8)
        fig.suptitle(f'{fam["label"]}｜{mode}｜{fam["fixed"]}',fontsize=14,y=.985)
        fig.text(.075,.025,'每点为一条60秒运行的统计；同一条线固定网络和噪声。缺失/不可估计不连线，不把事件当网络重复。\n患者参考保留自然比例；局部宽度为模型发放包络，旋转为含叠加假阳性的操作性候选。',fontsize=8)
        name=f'{fam["id"]}_{mode}_{group}'
        for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=150)
        an.plt.close(fig);captions.append(f'### {name}.png\n颜色为网络，实线圆点/虚线三角为两次噪声，每条线固定网络与噪声。各观测保留实际量纲，患者参考只用于可比的接触统计。\n**关注点**：参数曲线形状是否跨网络重演；参与改善是否伴随时差、模式支持或原生场的代价。\n')
    # The predeclared bridge pairs are not part of the three continuous-axis
    # families. Plot them explicitly so early cross-network runs yield a paired
    # position/edge-response figure, rather than only a CSV and example movies.
    for pair_index,pair in enumerate(plan['pairs']):
      ids=[pair['reference'],pair['candidate']]
      if not any(r['candidate'] in ids for r in tables['counts']):continue
      title='左核位置：原位置与左移0.75mm' if pair_index==0 else '离核输出：增权50%与增边50%'
      labels=['原位置','左移0.75mm'] if pair_index==0 else ['增权50%','增边50%']
      note='固定向外EE×1.25、EI×1；平移改变core成员及对应连接分块。' if pair_index==0 else '两者仅名义总剂量接近；增边改变实际邻接和时延，实际权重总量见应用记录。'
      for mode in ['ALL','TA','TB']:
       for group,metrics in groups:
        use=[m for m in metrics if mode!='ALL' or m[0]!='mode_fraction'];nr=int(np.ceil(len(use)/3))
        fig,axes=an.plt.subplots(nr,3,figsize=(14,3.35*nr+1.4),squeeze=False);fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.13,hspace=.48,wspace=.33)
        for ax,(key,label) in zip(axes.flat,use):
            for topo in plan['topology_seeds']:
             for i,noise in enumerate(plan['seeds']):
                ys=[value(cid,topo,noise,mode,key) for cid in ids]
                ax.plot([0,1],ys,c=COLORS[topo],ls='-' if i==0 else '--',marker='o' if i==0 else '^',lw=1.4,ms=4)
            patient=ref['modes'][mode].get(key)
            if key in ['pair_order_probability_mae','participation_mae']:patient=0.
            if key=='mode_fraction':patient=ref['modes'][mode]['n']/ref['fit_n']
            if key in ['n','L_search'] or key.startswith('rotation_'):patient=None
            if patient is not None:ax.axhline(patient,c='black',ls=':',lw=1)
            ax.set(xticks=[0,1],xticklabels=labels,ylabel=label);ax.grid(alpha=.18)
            if key in ['SCL_upper_participation','ICL_contact_participation','both_rods','mode_fraction']:ax.set_ylim(-.02,1.04)
            if key=='SCL_minus_ICL_lag_median_ms':ax.set_ylim(-40,130)
        for ax in list(axes.flat)[len(use):]:ax.axis('off')
        handles=[Line2D([],[],c=COLORS[x],label='基础网络 '+str(x)) for x in plan['topology_seeds']]+[Line2D([],[],c='#333',marker='o',label='噪声 847401'),Line2D([],[],c='#333',ls='--',marker='^',label='噪声 847402'),Line2D([],[],c='black',ls=':',label='患者 FIT 参考')]
        fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False,fontsize=8)
        fig.suptitle(title+f'｜{mode}｜逐网络配对',fontsize=14,y=.985)
        fig.text(.075,.025,note+'\n每条线固定基础网络与噪声；两端是离散条件，不是事件轨迹。每点为60秒运行统计，未完成或不可估计的点不填补。\n颜色=网络，实圆/虚三角=噪声；标签比例、类内传播与事件支持分开，不能把事件当作独立网络。',fontsize=8)
        name=f'bridge_pair{pair_index}_{mode}_{group}'
        for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=150)
        an.plt.close(fig)
        captions.append(f'### {name}.png\n{title}；每条线为固定基础拓扑种子和噪声下的两个条件。{note}\n**关注点**：相同参数改变是否跨网络重演，参与和时序有无代价；缺失点不代表零响应。\n')
    (F/'README.md').write_text('# 多网络参数响应曲线\n\n'+'\n'.join(captions))


def media_one():
    """At most one predeclared candidate/network group per polling pass."""
    plan=rt.read(OUT/'plan.json');records=[rt.read(p) for p in (A/'units').glob('*/result.json')];fr=an.figreview
    for cid in plan['confirmation']['candidates']:
     for topo in plan['topology_seeds']:
        rr=[r for r in records if r['counts']['candidate']==cid and r['counts']['topology']==topo]
        folder=F/f'{cid}_topology{topo}'
        if len(rr)!=2 or (folder/'manifest.json').exists():continue
        folder.mkdir(exist_ok=True);c=rt.read(OUT/'candidates'/f'{cid}.json');c['topology']=topo
        units={r['counts']['noise']:an.an.load_unit(Path(r['source']),1500.) for r in rr};seeds=sorted(units);physics=rt.read(Path(rr[0]['source']).parents[1]/'applied_physics.json');c['_applied_threshold']=physics['threshold']
        patient=fr.patient_payloads();manifest=[fr.spectral_comparison(c,units,seeds,folder,patient,'primary')]
        for seed in seeds:
            r,a,ids=units[seed];m=fr.four_panel(c,seed,r,a,ids,folder,physics,'primary')
            arrays=[fr.native_timing(r,a,v['event'])[0] for v in fr.representatives(a,ids).values()];finite=np.concatenate([x[np.isfinite(x)] for x in arrays]) if arrays else np.array([])
            m['native_shared_color_limits_ms']=[float(finite.min()),max(float(finite.max()),float(finite.min())+1)] if len(finite) else None;manifest.append(m)
        from scripts.render_topic4_shape_output_gifs import render
        r,a,ids=units[seeds[0]];manifest.append(render(c,seeds[0],r,a,ids,physics,folder,patient));rt.write(folder/'manifest.json',manifest)
        (folder/'README.md').write_text('# 固定网络患者与模型比较\n\n'+''.join(f'### {p.name}\n患者为Fig2C原始STFT，模型为发放密度包络，15行分杆且固定，真实毫秒轴。示例靠近模型自身模式均值；GIF按时间取各类前三例和固定连续片段，均含全部E活动。\n**关注点**：两类传播是否都恢复，而非只获得两个标签；不按患者相似程度挑示例。\n\n' for p in sorted(folder.iterdir()) if p.suffix in ['.png','.gif']))
        return True
    return False


def report(tables):
    counts=tables['counts'];plan=rt.read(OUT/'plan.json');nrot=sum(r.get('rotation_time_fraction') is not None for r in counts)
    note=f'''# 参数曲线与网络/噪声重复：持续结果

已分析 {len(counts)}/108 条正式新运行，旋转诊断完成 {nrot} 条；3条500ms参数应用检查不计入科学样本。

18条件×3网络(2511、2711、2712)×2新噪声(847401、847402)，每条60秒、排除前1.5秒。2511为开发网络，另外两张是本系列新网络。每个参数效应用同网络、同噪声的直接对照差；不把事件当作独立网络。

两种core形状下细扫核内E→I；圆核固定EI=0.875细扫向外E→E。另复测左移0.75mm，并比较向外增边与名义总权重相近的增权。后者不是精确总剂量匹配，实际边数和权重见applied_physics_by_run.json。曲线比较基线由各自family指定。

全部观测保留ALL/TA/TB，包含事件数、参与、rank、成对顺序/时差、局部宽度、跨区招募、原生面积/集中度和旋转候选；contacts/pairs/events/segments.csv保留分布及支持量。response_repeatability.csv只作方向一致性的描述，不给伪重复显著性。L_search、资格和输入物理保持冻结，不加入TA/TB细路线或旋转损失。

几何先验已经参与设计；患者参考为冻结FIT。患者HFO-STFT与模型发放包络不是同一信号。旋转仍可能来自多波源叠加，不等于确证螺旋。全部108条完成后停止在科学审阅点，不自动冻结模型或进入Fig5。
'''
    (A/'scientific_report.md').write_text(note)
    temp=A/'parameter_response_report.tmp.pdf'
    with PdfPages(temp) as pdf:
        fig=an.plt.figure(figsize=(11.7,8.3));fig.text(.07,.92,'多网络参数响应｜进度与解释边界',fontsize=18)
        import textwrap
        lines='\n\n'.join('\n'.join(textwrap.wrap(p,width=61)) for p in note.split('\n\n')[1:]);fig.text(.07,.84,lines,va='top',fontsize=10,linespacing=1.6);pdf.savefig(fig);an.plt.close(fig)
        for path in sorted(F.glob('*.png'))+sorted(F.glob('*/*patient_spectra_model_envelopes.png')):
            with Image.open(path) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=an.plt.figure(figsize=(14,14*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=130);an.plt.close(fig)
    temp.replace(A/'parameter_response_report.pdf');rt.write(A/'status.json',dict(analyzed_runs=len(counts),rotation_runs=nrot,updated_unix=time.time()))


def observer(once=False):
    configure()
    with (A/'observer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reference=an.load_reference();last=-1;last_rot=-1
        while True:
            paths=sorted((OUT/'response').glob('units/*/*/workers/trajectory.json'))
            for path in paths:
                key=hashlib.sha256(str(path).encode()).hexdigest()[:20]
                if not (A/'units'/key/'result.json').exists():an.process(path,reference)
            nrot=len(list((OUT/'rotation').glob('*/result.json')))
            changed=len(paths)!=last or nrot!=last_rot
            if changed:
                tables=aggregate();curves(tables);report(tables);last=len(paths);last_rot=nrot
            more=media_one()
            # Short physical runaway is a completed simulation, but cannot supply
            # the existing >=20s rotational observation. Never await an impossible job.
            eligible=[p for p in paths if rt.read(p).get('actual_duration_ms',0)>=20000]
            rotation_complete=all((OUT/'rotation'/hashlib.sha256(str(p).encode()).hexdigest()[:20]/'result.json').exists() for p in eligible)
            if (OUT/'simulation_complete.json').exists() and len(paths)==108 and rotation_complete and not more:
                tables=aggregate();curves(tables);report(tables);rt.write(OUT/'status.json',dict(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW',new_formal_runs=108,analysis_runs=108,rotation_runs=nrot,rotation_ineligible_short_duration=len(paths)-len(eligible),updated_unix=time.time()));break
            if once:break
            time.sleep(20)


def rotation(gpu):
    from scripts import analyze_topic4_rotation_response as rot
    rot.OUT=OUT;rot.OLD=OUT/'no_legacy_inputs';rot.main(gpu)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['observer','rotation']);p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.action=='observer':observer(a.once)
    else:rotation(a.gpu)

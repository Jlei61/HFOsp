"""Versioned analysis adapter for the bounded post-140 continuation."""
from pathlib import Path
import argparse,copy,csv,fcntl,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_recruitment_tradeoff_followup as run
from scripts import analyze_topic4_shape_output_response as an
OUT=run.OUT;rt=run.rt


def configure():
    from src.topic4_pdf_font_guard import install
    install()
    an.OUT=OUT;an.A=OUT/'analysis';an.F=an.A/'figures';an.run.OUT=OUT
    an.title=lambda c:c.get('display_name',c['id'])
    # The parent plot is correct; its metadata reuses a later raster vmax variable.
    # Correct the metadata locally without editing frozen shared producers.
    original=an.figreview.four_panel
    def panel(c,seed,r,a,ids,*args,**kwargs):
        result=original(c,seed,r,a,ids,*args,**kwargs);rep=an.figreview.representatives(a,ids)
        arrays=[an.figreview.native_timing(r,a,v['event'])[0] for v in rep.values()]
        finite=np.concatenate([x[np.isfinite(x)] for x in arrays]) if arrays else np.array([0.,250.])
        result['native_shared_color_limits_ms']=[float(finite.min()),max(float(finite.max()),float(finite.min())+1)]
        return result
    an.figreview.four_panel=panel


def factorial_effects():
    p=OUT/'analysis/observations.csv'
    if not p.exists():return []
    rows=list(csv.DictReader(p.open()));plan=rt.read(OUT/'plan.json');cs={c['id']:c for c in plan['candidates']}
    data={}
    for row in rows:
        c=cs.get(row['candidate']);f=None if c is None else c.get('factorial')
        if f and int(row['topology'])==2511 and row['layer']=='primary':data[(f['shape'],f['EE_out'],f['EI'],int(row['noise']),row['mode'])]=row
    effects=[]
    for (shape,ee,ei,seed,mode),row in data.items():
      for axis,refkey in [('EE_out',(shape,1.,ei,seed,mode)),('EI',(shape,ee,1.,seed,mode)),('shape',('circle',ee,ei,seed,mode))]:
        ref=data.get(refkey)
        if ref is None or ref['candidate']==row['candidate']:continue
        z=dict(candidate=row['candidate'],reference=ref['candidate'],axis=axis,shape=shape,EE_out=ee,EI=ei,noise=seed,mode=mode,n=int(row['n']),reference_n=int(ref['n']))
        for k,label in an.METRICS+an.SPACE_METRICS:
            x,y=row.get(k),ref.get(k);z[k]=float(x)-float(y) if x and y else None
        effects.append(z)
    an.an.writecsv(OUT/'analysis/factorial_simple_effects.csv',effects)
    for mode in ['ALL','TA','TB']:
      for shape in ['circle','ellipse4']:
        fig,axes=an.plt.subplots(2,3,figsize=(13,8),layout='constrained');found=False
        for ax,(metric,label) in zip(axes.flat,[an.METRICS[i] for i in [0,2,4,5,6,11]]):
            for ei,color in [(1.,'#666666'),(.875,'#4477aa'),(.75,'#bb5566')]:
             for seed in plan['seeds']:
                xs=[];ys=[]
                for ee in [1.,1.125,1.25]:
                    row=data.get((shape,ee,ei,seed,mode));v=None if row is None else row.get(metric)
                    xs.append(ee);ys.append(float(v) if v else np.nan)
                if np.isfinite(ys).any():found=True
                ax.plot(xs,ys,color=color,marker='o' if seed%2 else '^',ls='-' if seed%2 else '--',lw=1,label=f'EI×{ei:g}' if seed==plan['seeds'][0] else None)
            ax.set(xlabel='核→外 EE 权重倍数',ylabel=label,xticks=[1,1.125,1.25]);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle(f'{shape}｜{mode}｜形状×EE×EI组合响应\n颜色=EI倍数；圆实线/三角虚线=两条配对噪声。事件数和参与结构同时保留。')
        if found:
            for ext in ['png','pdf']:fig.savefig(an.F/f'factorial_{shape}_{mode}.{ext}',dpi=150)
        an.plt.close(fig)
    return effects


def additional_media():
    spec=OUT/'replication_selection.json'
    if not spec.exists():return
    chosen=rt.read(spec)['candidates'];results=[rt.read(p) for p in (an.A/'units').glob('*/result.json')];patient=None
    for cid in chosen:
      for topo in sorted({r['counts']['topology'] for r in results if r['counts']['candidate']==cid}):
        rr=[r for r in results if r['counts']['candidate']==cid and r['counts']['topology']==topo]
        folder=an.F/f'{cid}_topology{topo}'
        if len(rr)<2 or not (folder/'manifest.json').exists() or (folder/'continuation_media.json').exists():continue
        row=min(rr,key=lambda r:r['counts']['noise']);path=Path(row['source']);r,a,ids=an.an.load_unit(path,1500.);seed=row['counts']['noise']
        c=rt.read(OUT/'candidates'/f'{cid}.json');c['topology']=topo;physics=rt.read(path.parents[1]/'applied_physics.json')
        if patient is None:patient=an.figreview.patient_payloads()
        from scripts.render_topic4_shape_output_gifs import render,rotation_clip
        z=[render(c,seed,r,a,ids,physics,folder,patient)]
        rotation=OUT/'rotation'/hashlib.sha256(str(path).encode()).hexdigest()[:20]/'result.json'
        if rotation.exists():z.append(rotation_clip(c,seed,r,a,physics,folder,rt.read(rotation)))
        rt.write(folder/'continuation_media.json',z)
        with (folder/'README.md').open('a') as f:
            for gif in sorted(folder.glob('*.gif')):f.write(f'\n### {gif.name}\n来自真实完整轨迹，选择规则见continuation_media.json；固定分杆显示。\n**关注点**：原生场、接触时序和患者真实STFT是否共同支持传播解释，旋转仍为候选。\n')


def summary_pdf():
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    status=rt.read(an.A/'status.json');counts=list(csv.DictReader((an.A/'counts.csv').open())) if (an.A/'counts.csv').exists() else []
    plan=rt.read(OUT/'plan.json');reused={x['candidate'] for x in plan['reuse']};nreused=sum(r['candidate'] in reused and int(r['topology'])==2511 for r in counts)
    text=(f'已分析 {len(counts)-nreused} 条新运行，另复用 {nreused}/12 条历史直接对照。新预算上限64。\n\n'
        '固定比较：圆核/椭圆4:1 × 离核EE 1/1.125/1.25 × 同核EI 1/0.875/0.75；6个已有条件复用。\n'
        '另在两个已观察到时序改善的工作点附近比较x±0.75mm、y±1mm与半径2.05/2.35mm。\n'
        '半径探针匹配降阈值总量，但随机输入支持人数随范围变，不能说全部输入剂量相同。\n\n'
        '新确认条件仅按冻结训练分数提名，连同其直接对照，在2611/2612已有拓扑上用847301/847302新噪声重演。\n'
        '这不是未见拓扑测试；旧140条确认结果已进入开发历史。少数模式支持量逐运行保留。\n\n'
        '所有loss、资格、读出、Z/M关闭和限核噪声定义不变；TA/TB细路线及旋转不进入输入或新损失。\n'
        '患者STFT、模型发放包络、固定分杆15行与原生场多事件GIF并列。\n'
        '旋转候选可能由双波源叠加产生，不能直接证明稳定螺旋。\n')
    (an.A/'continuation_status.md').write_text('# 140条之后的有界续跑\n\n'+text+'\n完整原始均值/分布与支持量见CSV；factorial_simple_effects.csv分别比较固定EI时的EE、固定EE时的EI和相同参数时的形状效应。\n')
    # Replace the inherited hard-coded 140-run prose with the actual follow-up contract.
    (an.A/'scientific_note.md').write_text((an.A/'continuation_status.md').read_text())
    temp=an.A/'continuation_report.tmp.pdf'
    with PdfPages(temp) as pdf:
        fig=an.plt.figure(figsize=(11.7,8.3));fig.text(.06,.94,'core形状×离核EE×局部EI：有界续跑',fontsize=18,va='top');fig.text(.06,.87,text,fontsize=10,va='top',linespacing=1.8);pdf.savefig(fig);an.plt.close(fig)
        paths=sorted(an.F.glob('*.png'))
        for cid in plan['geometry_anchors']:
            paths.extend(sorted((an.F/f'{cid}_topology2511').glob('*patient_spectra_model_envelopes.png')))
        for path in paths:
            with Image.open(path) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=an.plt.figure(figsize=(14,14*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=130);an.plt.close(fig)
    temp.replace(an.A/'continuation_report.pdf')
    rt.write(an.A/'report_manifest.json',dict(analyzed_new=len(counts)-nreused,reused=nreused,pdf_sha256=rt.sha(an.A/'continuation_report.pdf'),updated_unix=time.time()))


def observer(once=False):
    configure();an.A.mkdir(exist_ok=True);an.F.mkdir(exist_ok=True);(an.A/'units').mkdir(exist_ok=True)
    an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':8,'pdf.fonttype':3})
    with (an.A/'observer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reference=an.load_reference();last=0
        while True:
            paths=[p for stage in ['response','confirmation'] for p in (OUT/stage).glob('units/*/*/workers/trajectory.json')];changed=False
            for path in sorted(paths):
                key=hashlib.sha256(str(path).encode()).hexdigest()[:20]
                if (an.A/'units'/key/'result.json').exists():continue
                an.process(path,reference);changed=True
            if changed or time.time()-last>300:
                an.aggregate();factorial_effects();an.review_figures();additional_media();summary_pdf();last=time.time()
                if list(an.F.glob('*.png')):
                    (an.F/'README.md').write_text('# 参数—观测响应\n\n'+''.join(f'### {p.name}\n固定拓扑与噪声配对，参数含义与复用身份见计划和CSV。组合图颜色区分EI倍数，线型区分噪声；其余响应图沿用原比较量纲。\n**关注点**：参与和时序能否同时改善；事件数与新噪声支持并列。\n\n' for p in sorted(an.F.glob('*.png'))))
            complete=OUT/'simulation_complete.json'
            if complete.exists():
                final=rt.read(complete);expected=final['expected_analyzed_runs']
                needed=[p for p in paths if rt.read(p)['actual_duration_ms']>=20000]
                rotations=all((OUT/'rotation'/hashlib.sha256(str(p).encode()).hexdigest()[:20]/'result.json').exists() for p in needed)
                if len(paths)==expected and rotations:
                    an.aggregate();factorial_effects();an.review_figures();additional_media();summary_pdf()
                    rt.write(OUT/'status.json',dict(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW',new_runs=final['new_runs'],reused_runs=12,analysis_runs=len(paths),updated_unix=time.time()));break
            if once:break
            time.sleep(20)


def rotation(gpu):
    from scripts import analyze_topic4_rotation_response as rot
    rot.OUT=OUT;rot.OLD=run.PARENT;rot.main(gpu)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['observer','rotation']);p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.action=='observer':observer(a.once)
    else:rotation(a.gpu)

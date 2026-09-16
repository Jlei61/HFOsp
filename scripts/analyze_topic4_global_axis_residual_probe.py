"""Paired global-axis diagnostics, with explicit actual-graph changes and support."""
from pathlib import Path
import fcntl,hashlib,json,time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scripts import run_topic4_global_axis_residual_probe as run
from scripts import analyze_topic4_shape_output_response as an
OUT=run.OUT;rt=run.rt;A=OUT/'analysis';F=A/'figures'


def configure():
    from src.topic4_pdf_font_guard import install
    install()
    an.OUT=OUT;an.A=A;an.F=F;an.run.OUT=OUT;an.title=lambda c:c['display_name']
    an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})


def aggregate():
    records=[rt.read(p) for p in (A/'units').glob('*/result.json')]
    for r in records:
        p=OUT/'rotation'/hashlib.sha256(r['source'].encode()).hexdigest()[:20]/'result.json'
        rot=rt.read(p) if p.exists() else None
        r['counts']['rotation_time_fraction']=None if rot is None else rot['candidate_time_fraction']
    tables={k:[z for r in records for z in (r[k] if isinstance(r[k],list) else [r[k]])] for k in ['counts','observations','contacts','pairs','events','segments']}
    for k,rows in tables.items():
        tmp=A/(k+'.tmp.csv');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(A/(k+'.csv'))
    refs=rt.read(A/'patient_reference.json');plan=rt.read(OUT/'plan.json');lookup={c['id']:c for c in plan['candidates']}
    counts={(r['candidate'],r['noise']):r for r in tables['counts']};obs={(r['candidate'],r['noise'],r['mode']):r for r in tables['observations'] if r['layer']=='primary'}
    effects=[]
    for (cid,noise,mode),r in obs.items():
        b=obs.get(('global_axis_+0',noise,mode))
        if cid=='global_axis_+0' or b is None:continue
        z=dict(candidate=cid,noise=noise,mode=mode,model_n=r['n'],control_n=b['n'])
        for key,_ in an.METRICS+an.SPACE_METRICS:
            x,y=r.get(key),b.get(key);z[key]=None if x is None or y is None else x-y
        effects.append(z)
    pd.DataFrame(effects).to_csv(A/'paired_differences.csv',index=False)
    if not records:return records
    groups=[('contact_participation',an.METRICS[:3]),('contact_timing_order',an.METRICS[3:6]),('native_timing',an.METRICS[6:9]),('native_support',an.METRICS[9:11]+an.SPACE_METRICS[:1]),('native_space',an.SPACE_METRICS[1:]),('sampling_support',[('n','该类合格事件数'),('fraction','该类 / 合格事件'),('L_search','整条运行训练分数\n（三行重复同值）'),('rotation_time_fraction','整条运行旋转候选时间比例\n（三行重复同值）')])]
    cs=sorted(plan['candidates'],key=lambda c:c['parameters']['EE_angle_offset_deg']);xs=[c['parameters']['EE_angle_offset_deg'] for c in cs]
    for name,metrics in groups:
        fig,axes=an.plt.subplots(3,len(metrics),figsize=(4.2*len(metrics),11),squeeze=False);fig.subplots_adjust(top=.85,bottom=.15,left=.035,right=.99,wspace=.34,hspace=.58)
        for i,mode in enumerate(['ALL','TA','TB']):
          for ax,(key,label) in zip(axes[i],metrics):
            for seed,col,ls,marker in [(847101,'#8965a5','-','o'),(847102,'#218e98','--','^')]:
                vals=[]
                for c in cs:
                    count=counts.get((c['id'],seed));row=obs.get((c['id'],seed,mode))
                    if key=='fraction':v=None if count is None or not count['primary'] else (1 if mode=='ALL' else count[mode]/count['primary'])
                    elif key in ['L_search','rotation_time_fraction']:v=None if count is None else count.get(key)
                    else:v=None if row is None else row.get(key)
                    vals.append(np.nan if v is None else v)
                ax.plot(xs,vals,c=col,ls=ls,marker=marker,ms=5)
            reference=refs['modes'][mode].get(key)
            if key=='fraction':reference=refs['modes'][mode]['n']/refs['fit_n']
            if key in ['pair_order_probability_mae','participation_mae']:reference=0
            if key in ['n','L_search']:reference=None
            if reference is not None:ax.axhline(reference,c='black',ls=':',lw=.9)
            ax.set(xlabel='全局EE轴角偏移 (°)',ylabel=label,xticks=xs,title=mode);ax.grid(alpha=.12)
            if key in ['SCL_upper_participation','ICL_contact_participation','both_rods','fraction']:ax.set_ylim(-.02,1.04)
        handles=[Line2D([],[],c='#8965a5',marker='o',label='噪声847101'),Line2D([],[],c='#218e98',ls='--',marker='^',label='噪声847102'),Line2D([],[],c='black',ls=':',label='患者FIT参考')]
        fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False)
        fig.suptitle('全局连接方向是否改变TB固定接触时序？｜20秒有界探针',fontsize=16,y=.985)
        fig.text(.035,.025,'固定神经元位置、基础拓扑种子2511及噪声；旋转EE核会重新采样实际E→E边和距离时延，每目标入度保留，不能称为同一张物理图。\n每点为一条20秒运行，排除前1.5秒；角度0的两条为历史长轨迹前缀重演，不充当新增独立重复。少数模式支持不足不自动判为机制缺失。\n横轴仅三个离散取值，连接线不表示已测连续曲线；不按分数停批或添加TA/TB路线损失。完整contacts/pairs/events表保留分布与支持。',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=140,bbox_inches='tight')
        an.plt.close(fig)
    rt.write(A/'status.json',dict(analyzed_runs=len(records),formal_budget=6,updated_unix=time.time()))
    return records


def media(records):
    plan=rt.read(OUT/'plan.json')
    for c in plan['candidates']:
        cid=c['id'];rr=[r for r in records if r['counts']['candidate']==cid];folder=F/cid
        if not rr:continue
        signature=sorted((r['counts']['noise'],r['source'],rt.sha(Path(r['source']))) for r in rr)
        state=folder/'media_state.json'
        if state.exists() and rt.read(state)['source_signature']==[list(x) for x in signature]:continue
        folder.mkdir(exist_ok=True);c=dict(c,topology=2511);units={r['counts']['noise']:an.an.load_unit(Path(r['source']),1500.) for r in rr};seeds=plan['seeds']
        physics=rt.read(Path(rr[0]['source']).parents[1]/'applied_physics.json');c['_applied_threshold']=physics['threshold'];fr=an.figreview;patient=fr.patient_payloads()
        # Keep both frozen noise columns. Missing outputs are explicitly pending,
        # rather than delaying all native/patient review until both runs finish.
        manifest=[fr.spectral_comparison(c,units,seeds,folder,patient,'primary')]
        for seed in sorted(units):
            r,a,ids=units[seed];m=fr.four_panel(c,seed,r,a,ids,folder,physics,'primary')
            arrays=[fr.native_timing(r,a,v['event'])[0] for v in fr.representatives(a,ids).values()];finite=np.concatenate([x[np.isfinite(x)] for x in arrays]) if arrays else np.array([])
            m['native_shared_color_limits_ms']=[float(finite.min()),max(float(finite.max()),float(finite.min())+1)] if len(finite) else None;manifest.append(m)
        from scripts.render_topic4_shape_output_gifs import render
        if seeds[0] in units:
            r,a,ids=units[seeds[0]];manifest.append(render(c,seeds[0],r,a,ids,physics,folder,patient))
        rt.write(folder/'manifest.json',manifest)
        rt.write(state,dict(source_signature=signature,available_noises=sorted(units),pending_noises=[s for s in seeds if s not in units],gif_noise=seeds[0] if seeds[0] in units else None,updated_unix=time.time(),scope='Complete physical runs only; one-noise media do not establish replay consistency'))
        (folder/'README.md').write_text('# 患者与20秒方向探针\n\n'+f'已完成噪声：{sorted(units)}；待完成：{[s for s in seeds if s not in units]}。缺失列明确标注等待，不能以一条噪声轨迹判断可重复性。\n\n'+''.join(f'### {p.name}\n患者为固定Fig2C原始STFT，模型为全部E发放包络／原生场，分杆固定15行。示例取模型自身均值附近，GIF固定使用噪声847101、按每类前三个事件与固定连续片段选取。\n**关注点**：两种接触路径及未恢复部分；同一角度内两次噪声分开，跨角度实际EE图改变。\n\n' for p in sorted(folder.iterdir()) if p.suffix in ['.png','.gif']))
        return True
    return False


def baseline_replay(records):
    if (A/'baseline_prefix_replay.json').exists():return
    rr=[r for r in records if r['counts']['candidate']=='global_axis_+0']
    if len(rr)!=2:return
    rows=[]
    for rec in rr:
        s=rec['counts']['noise'];p=Path(rec['source']);old=run.FOLLOW/'response/units'/run.ANCHOR/f'2511_{s}/workers/trajectory.npz'
        with np.load(p.with_suffix('.npz')) as a,np.load(old) as b:
            checks={k:bool(np.array_equal(a[k],b[k][:len(a[k])])) for k in ['sheet_activity_counts','trace_coreAE_spikes','trace_coreBE_spikes','core_ou_mixture_values']}
        rows.append(dict(noise=s,checks=checks,source=str(old),result=str(p)))
    rt.write(A/'baseline_prefix_replay.json',dict(status='PASS' if all(all(r['checks'].values()) for r in rows) else 'MISMATCH_REQUIRES_REVIEW',runs=rows,interpretation='Prefix replays are not independent scientific replicates; comparisons within this pilot use all new runs with the same20s duration and unchanged observer'))


def report(records):
    note=f'''# 全局EE轴与TB残差：有界方向探针

已分析{len(records)}/6条20秒运行。固定较好左移圆核工作点，比较全局EE角度偏移−15°、0°、+15°，基础拓扑种子2511与噪声847101/847102。所有运行使用相同20秒长度，排除前1.5秒；每次运行是实验单位。

角度改变会重新采样整张E→E图并重算距离时延，保持每目标入度及其他突触类型、阈值和随机输入定义。不能称为跨角度物理图完全相同。横向尺度倍率1.5、纵向1、核内EE0.85、离核EE1.25、其他参数继承原候选。

这是开发工作点上的机制敏感性探针；不是新拓扑确认，也不由患者指定TB路线、单例顺序或定向刺激驱动。loss、事件资格及TA/TB分配不变。角度0的20秒基线重演旧60秒轨迹前缀，完整输出另检查，不能算新增独立复制。

联合查看参与、固定接触对顺序／毫秒时差、类别支持和原生场；只改善杆间摘要而固定对仍错误，不能称为TB恢复。只有一张基础网络，方向效应即便出现仍须后续独立复测。无论传播是否改善，所有已派发条件完整收尾；09:32之后不再新增派发，未运行条件保持明确状态。

患者Fig2C实际STFT与模型发放密度包络并排，15行按杆固定。模型自身模式均值附近选例、每类最早三例和连续原生场片段可审阅，不按患者相似度挑例。旋转为已有操作性诊断，可能受多波源叠加影响，不等于稳定螺旋。
'''
    (A/'scientific_report.md').write_text(note)
    (F/'README.md').write_text('# 方向探针参数响应\n\n'+''.join(f'### {p.name}\n三点为全局EE轴角度偏移，颜色和线型区分两噪声，逐运行和逐模式展示。其他参数固定，实际EE图随方向重采样。\n**关注点**：TB固定触点时序是否改善，TA和参与结构是否付出代价；事件数与原生场不可省略。\n\n' for p in sorted(F.glob('*.png'))))
    temp=A/'axis_probe_report.tmp.pdf'
    with PdfPages(temp) as pdf:
        fig=an.plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,'全局EE连接方向：当前证据与边界',fontsize=18)
        import textwrap
        text='\n\n'.join('\n'.join(textwrap.wrap(p,60)) for p in note.split('\n\n')[1:]);fig.text(.06,.86,text,fontsize=10,va='top',linespacing=1.55);pdf.savefig(fig);an.plt.close(fig)
        for path in sorted(F.glob('*.png'))+sorted(F.glob('*/*patient_spectra_model_envelopes.png')):
            with Image.open(path) as im:arr=np.asarray(im.convert('RGB'))
            h,w=arr.shape[:2];fig=an.plt.figure(figsize=(17,17*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=140);an.plt.close(fig)
    temp.replace(A/'axis_probe_report.pdf')


def observer():
    configure()
    with (A/'observer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reference=an.load_reference();last=-1;last_rot=-1
        while True:
            paths=sorted((OUT/'response').glob('units/*/*/workers/trajectory.json'))
            for p in paths:
                key=hashlib.sha256(str(p).encode()).hexdigest()[:20]
                if not (A/'units'/key/'result.json').exists():an.process(p,reference)
            nrot=len(list((OUT/'rotation').glob('*/result.json')))
            if len(paths)!=last or nrot!=last_rot:
                records=aggregate();baseline_replay(records);report(records);last=len(paths);last_rot=nrot
            more=media(records)
            if more:report(records)
            eligible=[p for p in paths if rt.read(p)['actual_duration_ms']>=20000]
            rot_done=all((OUT/'rotation'/hashlib.sha256(str(p).encode()).hexdigest()[:20]/'result.json').exists() for p in eligible)
            if (OUT/'simulation_complete.json').exists() and rot_done and not more:
                records=aggregate();report(records);done=rt.read(OUT/'simulation_complete.json')
                rt.write(A/'closeout.json',dict(analyzed_runs=len(records),formal_budget=6,physical_completion=done,scientific_acceptance='PENDING_REVIEW',updated_unix=time.time()))
                rt.write(OUT/'status.json',dict(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW' if done['all_six_complete'] else 'WINDOW_ELAPSED_PARTIAL_PENDING_REVIEW',formal_complete=len(records),formal_budget=6,rotation_runs=nrot,updated_unix=time.time()));return
            time.sleep(20)


if __name__=='__main__':observer()

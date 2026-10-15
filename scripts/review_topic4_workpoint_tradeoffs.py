"""Compare two completed composite workpoints without re-nominating candidates."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from src.topic4_pdf_font_guard import install
from scripts import analyze_topic4_shape_output_response as an

BASE=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
OUT=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/workpoint_tradeoffs')
CASES=[('circle','up3__circle__EE_core_to_out_scale_1.25__x_minus075','圆核＋向外E→E增强25%＋左核左移0.75mm'),
       ('ellipse','up3__ellipse4__EI_same_core_scale_0.75__x_minus075','左核椭圆4:1＋两核内E→I减弱25%＋左核左移0.75mm')]

def main():
    OUT.mkdir(exist_ok=True);figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3})
    refs=json.loads((BASE/'analysis/patient_reference.json').read_text())
    counts=pd.read_csv(BASE/'analysis/counts.csv').set_index(['candidate','noise'])
    obs=pd.read_csv(BASE/'analysis/observations.csv').query("layer=='primary'").set_index(['candidate','noise','mode'])
    pair=pd.read_csv(BASE/'analysis/pairs.csv').query("layer=='primary' and mode=='TB' and contact_i=='ICL11' and contact_j=='ICL9'").set_index(['candidate','noise'])
    records=[];manifests=[];patient=an.figreview.patient_payloads()
    for short,cid,title in CASES:
        folder=figdir/short;folder.mkdir(exist_ok=True);c=json.loads((BASE/'candidates'/f'{cid}.json').read_text());c.update(topology=2511,display_name=title)
        units={};physics=None
        for noise in [847101,847102]:
            path=BASE/'response/units'/cid/f'2511_{noise}/workers/trajectory.json'
            r,a,ids=an.an.load_unit(path,1500.);units[noise]=(r,a,ids)
            current=json.loads((path.parents[1]/'applied_physics.json').read_text())
            if physics is not None:assert physics['identity']==current['identity']
            physics=current;co=counts.loc[(cid,noise)];ta=obs.loc[(cid,noise,'TA')];tb=obs.loc[(cid,noise,'TB')];pa=pair.loc[(cid,noise)]
            records.append(dict(case=short,candidate=cid,noise=noise,total_events=int(co.primary),TA=int(co.TA),TB=int(co.TB),L_search=float(co.L_search),TA_fraction=float(co.TA/co.primary),TA_SCL_upper=float(ta.SCL_upper_participation),TA_rod_lag_ms=float(ta.SCL_minus_ICL_lag_median_ms),TA_pair_order_error=float(ta.pair_order_probability_mae),TB_rod_lag_ms=float(tb.SCL_minus_ICL_lag_median_ms),TB_ICL11_before_ICL9=float(pa.model_i_precedes_j),TB_ICL_pair_n=int(pa.model_joint_n),source=str(path),source_sha256=r['arrays_sha256']))
        c['_applied_threshold']=physics['threshold']
        manifest=an.figreview.spectral_comparison(c,units,[847101,847102],folder,patient,'primary')
        r,a,ids=units[847101]
        m=an.figreview.four_panel(c,847101,r,a,ids,folder,physics,'primary')
        vv=np.concatenate([an.figreview.native_timing(r,a,v['event'])[0].ravel() for v in an.figreview.representatives(a,ids).values()]);vv=vv[np.isfinite(vv)]
        m['native_shared_color_limits_ms']=[float(vv.min()),max(float(vv.max()),float(vv.min())+1.)]
        if not (folder/'gif_manifest.json').exists():
            from scripts.render_topic4_shape_output_gifs import render
            # Event selection remains chronological first-three per label; no
            # selection by matching the patient's particular example.
            gm=render(c,847101,r,a,ids,physics,folder,patient)
            (folder/'gif_manifest.json').write_text(json.dumps(gm,indent=2))
        manifests.append(dict(case=short,comparison=manifest,mechanism=m,applied_threshold=physics['threshold']))
        (folder/'README.md').write_text('# 组合工作点：'+title+'\n\n'+''.join(f'### {p.name}\n患者为固定Fig2C真实STFT，模型为全部E活动及发放密度包络；15行固定按SCL、ICL分杆。示例来自本运行模式均值附近；GIF按每类最早三个事件与固定连续片段选取。\n**关注点**：参与和路径的恢复是否一致；该组合与另一组合有多个参数差异，不能将全部差异归因于形状。\n\n' for p in sorted(folder.iterdir()) if p.suffix in ['.png','.gif']))
    d=pd.DataFrame(records);d.to_csv(OUT/'workpoints_by_run.csv',index=False)
    lines=[]
    metrics=[('TA_fraction','TA事件比例',refs['modes']['TA']['n']/refs['fit_n'],100,'%'),('TA_SCL_upper','TA：SCL9/8平均参与概率',refs['modes']['TA']['SCL_upper_participation'],100,'%'),('TA_rod_lag_ms','TA：SCL−ICL时差中位数',refs['modes']['TA']['SCL_minus_ICL_lag_median_ms'],1,'ms'),('TA_pair_order_error','TA：成对先后概率误差',0.,1,''),('TB_rod_lag_ms','TB：SCL−ICL时差中位数',refs['modes']['TB']['SCL_minus_ICL_lag_median_ms'],1,'ms'),('TB_ICL11_before_ICL9','TB：ICL11早于ICL9的比例',float(pair.iloc[0].patient_i_precedes_j),100,'%')]
    for key,label,pv,scale,unit in metrics:
        cols=[]
        for case,_,_ in CASES:
            z=d[d['case']==case].sort_values('noise')[key]*scale;cols.append(' / '.join(f'{v:.2f}' for v in z)+' '+unit)
        lines.append(f'|{label}|{pv*scale:.2f} {unit}|{cols[0]}|{cols[1]}|')
    note='''# 两个较有希望的组合，改善的是不同方面

这里比较同一基础网络2511的两个已完成组合，各两次噪声847101/847102；每条60秒，排除前1.5秒。表中模型两数始终依次是这两次噪声，不混池事件；患者参考为完整冻结FIT自然分布。黑色图保留患者Fig2C原始STFT及模型发放密度包络的信号区别，15个接触点固定分杆，不拉伸时间。

两个组合的左核中心相同，均已左移0.75mm。圆核组合使用向外EE×1.25、核内EI×1；椭圆组合左核4:1、向外EE×1、核内EI×0.75。它们同时改变形状和连接参数，是候选工作点比较，不是单独形状效应。

|观测|患者FIT|圆核＋向外EE增强|椭圆＋核内EI减弱|
|---|---:|---:|---:|
'''+ '\n'.join(lines)+'''

观察与判断：椭圆组合的TA上杆参与更接近患者，圆核组合的TA自然占比及杆间时差更接近患者；它们不是所有观测上互相支配的两个点。目视可见椭圆例的上杆参与较完整，但两个组合的TB仍主要沿下杆顺次向左、再到上杆，没有稳定恢复患者示例及分布中的ICL局部折返。上表的固定ICL11/ICL9先后概率支持这个缺口，不只是单例视觉印象。

不能因两个标签都出现、参与更完整或冻结训练分数更低而接受完整双模式恢复。本比较不改变续跑控制器的原loss提名，也不把该固定接触对加入训练；正在运行的新网络配对和有界方向探针用于进一步判断响应能否保留、TB缺口是否可被连接几何改变。

模式内实际事件数、固定接触对共同参与数及来源保存在workpoints_by_run.csv；完整事件图与每类前三例GIF均在figures下。图件经Agent自查，用户人工验收仍待进行。
'''
    (OUT/'scientific_note.md').write_text(note)
    (OUT/'manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2))
    with PdfPages(OUT/'workpoint_comparison.pdf') as pdf:
        for case,_,_ in CASES:
            for p in sorted((figdir/case).glob('*patient_spectra_model_envelopes.png'))+sorted((figdir/case).glob('*same_network.png')):
                with Image.open(p) as im:arr=np.asarray(im.convert('RGB'))
                h,w=arr.shape[:2];fig=an.plt.figure(figsize=(17,17*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=150);an.plt.close(fig)
    print('Completed two composite workpoint comparisons, fixed contact layout, four-page PDF and two multi-event GIFs.')

if __name__=='__main__':main()

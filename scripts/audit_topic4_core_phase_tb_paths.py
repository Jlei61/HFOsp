"""Within-label core-timing stratification; descriptive, not a route loss.

Core t10/t50 are cumulative mass times in existing windows, not causal onsets.
"""
from pathlib import Path
import sys,json,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from src.topic4_pdf_font_guard import install
from scripts import analyze_topic4_shape_output_response as an
BASE=Path('/data/hfosp/topic4_sef_hfo');OUT=BASE/'overnight_exploration_20260913/core_phase_tb_fold';F=OUT/'figures'
CASES=[('up3__circle','圆核基线'),('up3__circle__EE_core_to_out_scale_1.25','圆核＋向外EE增强25%'),('up3__circle__EE_core_to_out_scale_1.25__x_minus075','向外EE增强＋左移'),('follow_circle_out1.125_EI0.875','圆核＋EE1.125／EI0.875'),('up3__ellipse4__EI_same_core_scale_0.75__x_minus075','椭圆＋EI减弱＋左移')]
SHOW=[CASES[i] for i in [1,2,4]]

def collect():
    design=an.rt.read(an.run.base.PARENT);ev=an.rt.load_evaluator(design)
    patient=np.asarray(ev.fit)[np.asarray(ev.fit_labels)==0];names=np.asarray(an.rt.load_observation_contract(design)['contact_names']);i,j=list(names).index('ICL11'),list(names).index('ICL9')
    pp=an.an.pair_table(patient);pt=an.an.measures(patient,patient,names);pdelta=patient[:,j]-patient[:,i];pdelta=pdelta[np.isfinite(pdelta)]
    ref=dict(TB_events=len(patient),pair_n=len(pdelta),pair_probability=float(np.mean((pdelta>0)+.5*(pdelta==0))),pair_lag_quantiles_ms=np.quantile(pdelta,[.05,.5,.95]).tolist(),rod_lag_median_ms=pt['SCL_minus_ICL_lag_median_ms'])
    rows=[];summaries=[];sources=[];seen=set()
    for series in ['core_shape_output_response_20260911','core_recruitment_tradeoff_followup_20260912']:
      for file in sorted((BASE/series/'analysis/units').glob('*/result.json')):
        r=an.rt.read(file);c=r['counts'];key=(c['candidate'],c['topology'],c['noise'])
        if key in seen or c['candidate'] not in dict(CASES) or c['topology']!=2511:continue
        seen.add(key);path=Path(r['source'])
        with np.load(path.with_suffix('.npz')) as a:mu=a['centroid_ms'];assert list(a['contact_names'])==list(names)
        events=[]
        for e in r['events']:
            if not e['primary']:continue
            n=e['event'];row=dict(candidate=key[0],topology=key[1],noise=key[2],event=n,mode=e['mode'],core_t10=e['B_minus_A_t10_ms'],core_t50=e['B_minus_A_t50_ms'],ICL9_minus_ICL11=mu[n,j]-mu[n,i],coreA_mass=e['coreA_mass'],coreB_mass=e['coreB_mass'],rod_lag=e.get('centroid_SCL_minus_ICL_ms'))
            events.append(row);rows.append(row)
        d=pd.DataFrame(events);tb=d[d['mode']=='TB']
        for timing in ['core_t10','core_t50']:
          for group,z in [('all',tb),('left earlier',tb[tb[timing]>0]),('right earlier',tb[tb[timing]<0]),('tie',tb[tb[timing]==0])]:
            x=mu[z.event.astype(int)];m=an.an.measures(x,patient,names);pairs=an.an.pair_table(x);errors=[abs(p-pp[k][1]) for k,(_,p) in pairs.items() if p is not None and pp[k][1] is not None]
            lag=z.ICL9_minus_ICL11.dropna();qs=np.quantile(lag,[.05,.5,.95]) if len(lag) else [None]*3
            summaries.append(dict(candidate=key[0],topology=key[1],noise=key[2],timing=timing,group=group,**m,TB_total_n=len(tb),core_timing_valid_n=int(tb[timing].notna().sum()),pair_order_error=float(np.mean(errors)) if errors else None,pair_order_supported_pairs=len(errors),ICL11_first_probability=float(np.mean((lag>0)+.5*(lag==0))) if len(lag) else None,ICL_joint_n=len(lag),ICL_lag_q05=qs[0],ICL_lag_median=qs[1],ICL_lag_q95=qs[2],coreA_mass_median=z.coreA_mass.median(),coreB_mass_median=z.coreB_mass.median()))
        sources.append(dict(candidate=key[0],topology=key[1],noise=key[2],analysis=str(file),analysis_sha256=an.rt.sha(file),trajectory=str(path),arrays_sha256=r['source_sha256']))
    assert len(sources)==10
    d=pd.DataFrame(rows);s=pd.DataFrame(summaries);d.to_csv(OUT/'events.csv',index=False);s.to_csv(OUT/'conditional_observations.csv',index=False)
    an.rt.write(OUT/'patient_reference.json',ref);an.rt.write(OUT/'sources.json',dict(records=sources,created=datetime.datetime.now().astimezone().isoformat(),selection='five named completed development workpoints, two noises each, no candidate re-nomination'))
    return d,s,ref

def plots(d,s,ref):
    fig,axes=an.plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True);fig.subplots_adjust(top=.82,bottom=.17,left=.07,right=.99,wspace=.25,hspace=.38)
    for row,noise in enumerate([847101,847102]):
      for col,(cid,title) in enumerate(SHOW):
        ax=axes[row,col];z=d[(d.candidate==cid)&(d.noise==noise)&(d['mode']=='TB')].dropna(subset=['core_t10','ICL9_minus_ICL11'])
        ax.axhspan(ref['pair_lag_quantiles_ms'][0],ref['pair_lag_quantiles_ms'][2],fc='#ddd',alpha=.5,label='患者TB 5–95%')
        ax.axhline(ref['pair_lag_quantiles_ms'][1],c='black',ls=':',lw=1)
        for group,mask,color in [('左核累计10%较早',z.core_t10>0,'#b97c36'),('右核累计10%较早',z.core_t10<0,'#3478b8')]:
            v=z[mask];ax.scatter(v.core_t10,v.ICL9_minus_ICL11,c=color,s=19,alpha=.65,label=f'{group} n={len(v)}')
        ax.axvline(0,c='#888',lw=.8);ax.axhline(0,c='#888',lw=.8);ax.set(xlim=(-180,180),ylim=(-60,65),xlabel='右核t10 − 左核t10 (ms)',ylabel='ICL9 − ICL11质心时间 (ms)',title=title+f'\n噪声{noise}');ax.legend(fontsize=7,frameon=False,loc='upper left');ax.grid(alpha=.1)
    fig.suptitle('同样标为TB，ICL局部先后仍随两核活动时序而分开',fontsize=16,y=.98)
    fig.text(.07,.035,'每格是一条60秒运行，点为本类且两接触点共同参与的事件；横轴正值=左核较早，纵轴正值=ICL11较早。患者参照为完整TB分布。\n核t10取既有250ms窗内各核自身累计10%发放的时间，不是起燃时间或因果驱动；标签、接触时序与分层并非独立验收。\n这一关联不能区分双前沿叠加、局部准备状态或真实区域间驱动；未改变loss、噪声或任何物理参数。',fontsize=9)
    save(fig,'tb_core_timing_pair_lag')
    for timing in ['core_t10','core_t50']:
        fig,axes=an.plt.subplots(3,3,figsize=(15,11));fig.subplots_adjust(top=.84,bottom=.18,left=.07,right=.99,wspace=.3,hspace=.62)
        metrics=[('ICL11_first_probability','P(ICL11质心早于ICL9)',ref['pair_probability']),('pair_order_error','全部可比较触点对的顺序概率误差',0.),('SCL_minus_ICL_lag_median_ms','SCL−ICL质心时差中位数 (ms)',ref['rod_lag_median_ms'])]
        for col,(cid,title) in enumerate(SHOW):
          for row,(key,label,pval) in enumerate(metrics):
            ax=axes[row,col]
            for noise,color,marker in [(847101,'#8965a5','o'),(847102,'#218e98','^')]:
                z=s[(s.candidate==cid)&(s.noise==noise)&(s.timing==timing)].set_index('group').loc[['all','left earlier','right earlier']]
                ax.plot(range(3),z[key],c=color,marker=marker,ls='-' if noise==847101 else '--',lw=1,label=f'噪声{noise}')
                if row==0:
                    for k,(_,r) in enumerate(z.iterrows()):ax.annotate(f'n={r.ICL_joint_n}',(k,r[key]),xytext=(0,6 if noise==847101 else -13),textcoords='offset points',fontsize=7,color=color,ha='center')
            ax.axhline(pval,c='black',ls=':',lw=1);ax.set(xticks=range(3),xticklabels=['全部TB','左核较早','右核较早'],ylabel=label,title=title if row==0 else None);ax.grid(alpha=.13)
            if row==0:ax.set_ylim(-.08,1.14)
            if row==1:ax.set_ylim(0,.46)
            if row==2:ax.set_ylim(-35,50)
        axes[0,0].legend(fontsize=8,frameon=False,loc='lower center')
        fig.suptitle('核'+('10%' if timing=='core_t10' else '50%')+'累计时间分层：局部折返可出现，但总体TB分布仍未恢复',fontsize=16,y=.98)
        fig.text(.07,.055,'同一轨迹在三个统计集合中展示，连线不是参数干预，也不是独立样本。首行n为实际共同参与该触点对的事件数；其余指标的支持量见CSV。\n黑虚线是完整患者TB参考，患者没有对应core状态标签，不能要求每个模型亚组单独等于患者全部TB。第二行显示全部可比较触点对，避免只看一个有利接触对。\n两种累计时间定义都保留，未把分层结果加入loss或提名；即便接触时序较接近，也须结合原生场判断是否为两个前沿相位叠加。',fontsize=9)
        save(fig,timing+'_conditional_observations')

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=150)
    an.plt.close(fig)

def window_and_trace_checks(d):
    """Move only the core-timing diagnostic window; keep labels and readout fixed."""
    rows=[];selected=None
    for source in an.rt.read(OUT/'sources.json')['records']:
        if not source['candidate'].endswith('__x_minus075'):continue
        path=Path(source['trajectory']);r=an.rt.read(path)
        with np.load(path.with_suffix('.npz')) as a:
            tt=a['trace_time_ms'];aa=a['trace_coreAE_spikes'];bb=a['trace_coreBE_spikes']
            events=d[(d.candidate==source['candidate'])&(d.noise==source['noise'])&(d['mode']=='TB')]
            for _,e in events.iterrows():
                lo,hi=r['events'][int(e.event)]['window_ms']
                for shift in [-25.,0.,25.]:
                    mask=(tt>=lo+shift)&(tt<hi+shift);times=[]
                    for x in [aa[mask],bb[mask]]:
                        times.append(float(np.interp(.1*x.sum(),np.cumsum(x),tt[mask])) if x.sum()>0 else np.nan)
                    delta=times[1]-times[0]
                    if shift==0:assert np.isclose(delta,e.core_t10,equal_nan=True)
                    rows.append(dict(candidate=source['candidate'],noise=source['noise'],event=int(e.event),shift_ms=shift,core_delta=delta,original_core_delta=e.core_t10,ICL_lag=e.ICL9_minus_ICL11,boundary_truncated=bool(lo+shift<0 or hi+shift>r['actual_duration_ms'])))
            if source['candidate']==CASES[2][0] and source['noise']==847101:
                # The canonical loader checks the output hash and converts raw
                # contact-by-time envelopes to the plotter's time-by-contact.
                loaded_r,loaded_a,_=an.an.load_unit(path,1500.)
                selected=(source,loaded_r,loaded_a,events)
    pd.DataFrame(rows).to_csv(OUT/'window_shift_diagnostic.csv',index=False)
    source,r,a,events=selected;names=list(a['contact_names']);order=an.figreview.display.contact_indices(names)
    fig,axes=an.plt.subplots(2,3,figsize=(17,10),gridspec_kw={'width_ratios':[1.25,1,1.3]});fig.subplots_adjust(left=.07,right=.99,top=.84,bottom=.17,wspace=.3,hspace=.4);manifest=[]
    for row,(group,eligible) in enumerate([('左核累计10%较早',events[events.core_t10>0]),('右核累计10%较早',events[events.core_t10<0])]):
        rep=an.figreview.representatives(a,eligible.event.to_numpy(int))['TB'];i=rep['event'];lo,hi=r['events'][i]['window_ms'];mu=a['centroid_ms'][i];zero=float(np.nanmin(mu));tt=a['trace_time_ms'];full=(tt>=max(0,lo-80))&(tt<min(r['actual_duration_ms'],hi+80));inside=(tt>=lo)&(tt<hi);qt=[]
        for key,label,color in [('coreAE','左核','#bc7939'),('coreBE','右核','#397fb0')]:
            counts=a['trace_'+key+'_spikes'].astype(float);mass=counts[inside].sum();n=len(a['group_'+key]);trace_ms=float(np.median(np.diff(tt)));rate=counts[full]*1000/(n*trace_ms)
            axes[row,0].plot(tt[full]-zero,rate,c=color,lw=.8,label=label)
            cumulative=np.cumsum(counts[inside])/mass;q10=float(np.interp(.1,cumulative,tt[inside]))-zero;qt.append(q10)
            axes[row,1].plot(tt[inside]-zero,cumulative,c=color,lw=1.2,label=label);axes[row,1].axvline(q10,c=color,ls=':',lw=.7)
        axes[row,0].axvspan(lo-zero,hi-zero,fc='#eee',zorder=-1);axes[row,0].set(ylabel='核内E群体放电率（Hz/细胞）',title=group+f'｜TB事件{i}，亚组n={len(eligible)}');axes[row,0].legend(frameon=False,fontsize=8)
        axes[row,1].axhline(.1,c='#aaa',lw=.7);axes[row,1].set(ylabel='既有事件窗内累计发放份额',title=f'右核t10 − 左核t10 = {qt[1]-qt[0]:.1f} ms',ylim=(-.02,1.03))
        dt=float(a['contact_envelope_dt_ms']);env=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T.copy();part=np.isfinite(mu[order]);env/=np.maximum(env.max(1,keepdims=True),1e-20);env[~part]=np.nan
        cmap=an.plt.get_cmap('magma').copy();cmap.set_bad('#777');ax=axes[row,2];ax.imshow(env,aspect='auto',extent=[lo-zero,hi-zero,14.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest');an.figreview.display.centroid_lines(ax,mu[order]-zero,color='#397fb0');an.figreview.display.contact_axis(ax);ax.set_facecolor('#b6b6b6');ax.set_title('同一事件的固定分杆接触包络');ax.tick_params(axis='y',labelsize=7)
        for ax in axes[row]:ax.axvline(0,c='#999',lw=.7,ls='--');ax.set(xlim=(-130,160),xlabel='相对本事件最早参与质心 (ms)')
        manifest.append(dict(group=group,event=int(i),eligible_n=len(eligible),selection='closest to this subgroup own event feature mean via existing representative selector; not patient-nearest',window_ms=[lo,hi],zero_absolute_ms=zero,core_t10_relative_ms=qt))
    shared_rate_max=max(ax.get_ylim()[1] for ax in axes[:,0])
    for ax in axes[:,0]:ax.set_ylim(0,shared_rate_max)
    fig.suptitle('同一固定网络、同一条记录、同样TB标签：原始两核发放与接触时序',fontsize=16,y=.98)
    fig.text(.07,.055,'圆核＋向外EE增强25%＋左移0.75mm，基础网络2511／噪声847101。两个事件分别取各亚组自身均值附近，不按患者相似度挑选。\n左图为原始1ms核内群体发放率，灰底仅标既有250ms窗口；中图在该窗内分别归一化两核发放质量。右图保留固定15行和未参与接触点。\n这里直接检查窗口与波形，不是证明核间因果方向。部分接触顺序更接近患者仍可能来自双前沿叠加；患者对应图及原生多事件GIF见同级workpoint_tradeoffs。',fontsize=9)
    save(fig,'selected_tb_core_traces');an.rt.write(OUT/'selected_tb_core_trace_manifest.json',dict(source=source,events=manifest,display_order=an.figreview.display.CONTACT_ORDER))

def main():
    OUT.mkdir(exist_ok=True);F.mkdir(exist_ok=True);install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3,'font.size':9})
    d,s,ref=collect();plots(d,s,ref);window_and_trace_checks(d)
    with PdfPages(OUT/'core_phase_tb_paths.pdf') as pdf:
      for p in sorted(F.glob('*.png')):
        with Image.open(p) as im:arr=np.asarray(im.convert('RGB'))
        h,w=arr.shape[:2];fig=an.plt.figure(figsize=(15,15*h/w));ax=fig.add_axes([0,0,1,1]);ax.imshow(arr);ax.axis('off');pdf.savefig(fig,dpi=150);an.plt.close(fig)
    (F/'README.md').write_text('# TB内部的core活动时序与接触路径\n\n'+''.join(f'### {p.name}\n固定工作点与噪声逐运行展示，分层依据已有事件窗内core累计发放的10%或50%时间，不是因果起源。患者参考为完整TB事件的接触统计。\n**关注点**：TB标签内部仍可能混合不同原生过程；局部触点先后改善与完整传播恢复应分开判断。\n\n' for p in sorted(F.glob('*.png'))))
    agreement=[]
    for (cid,noise),z in d[d['mode']=='TB'].groupby(['candidate','noise']):
        valid=z[['core_t10','core_t50']].notna().all(axis=1);v=z[valid]
        agreement.append(dict(candidate=cid,noise=noise,n=len(v),sign_agreement=float(np.mean(np.sign(v.core_t10)==np.sign(v.core_t50))) if len(v) else None))
    pd.DataFrame(agreement).to_csv(OUT/'timing_definition_agreement.csv',index=False)
    print('Completed 10-run descriptive stratification, window sensitivity and four scientific figures.')

if __name__=='__main__':main()

"""Dense one-parameter responses; labels are only read in this reporting module."""
from pathlib import Path
import json, warnings, time
import numpy as np
from scripts import run_topic4_label_free_dense_search as run
from scripts import analyze_topic4_three_observable_bo as old_analysis
from scripts.report_topic4_three_observable_raw import observables
from src.topic4_three_observable_objective import normalized_ranks
OUT=run.OUT;rt=run.rt
DISPLAY=old_analysis.DISPLAY

def describe(t,names):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        raw,_=observables(t,names)
        r,m=normalized_ranks(t) if len(t) else (np.empty_like(t),np.empty_like(t,dtype=bool))
        ranks=np.where(m,r,np.nan)
        raw['mean_rank']=np.nanmean(ranks,axis=0).tolist() if len(t) else [None]*len(names)
        raw['rank_q05']=np.nanquantile(ranks,.05,axis=0).tolist() if len(t) else [None]*len(names)
        raw['rank_q95']=np.nanquantile(ranks,.95,axis=0).tolist() if len(t) else [None]*len(names)
        raw['recruitment_span_ms']=float(np.nanmedian(np.nanmax(t,axis=1)-np.nanmin(t,axis=1))) if len(t) else None
    return raw

def mean_valid(values):
    x=np.array(values,dtype=float);return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan

def metrics(q,p,names):
    rank_delta=abs(np.array(q['mean_rank'],float)-np.array(p['mean_rank'],float))
    rank=mean_valid([mean_valid(rank_delta[[i for i,n in enumerate(names) if n.startswith(shaft)]]) for shaft in ['SCL','ICL']])
    order=[];part=[]
    for shaft in ['SCL','ICL']:
        order.append(mean_valid([abs(float(v['order_probability'])-float(p['pairs'][key]['order_probability'])) for key,v in q['pairs'].items() if v['shaft']==shaft and v['order_probability'] is not None and p['pairs'][key]['order_probability'] is not None]))
        part.append(mean_valid([abs(q['contacts'][n]['participation']-p['contacts'][n]['participation']) for n in names if n.startswith(shaft) and q['contacts'][n]['participation'] is not None]))
    return [rank,mean_valid(order),q['rod_lag_ms']['median'],mean_valid(part),q['both_rods'],q['N']]

def report_legacy():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from src.topic4_pdf_font_guard import install
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    F=OUT/'figures';F.mkdir(exist_ok=True)
    p=rt.read(OUT/'plan.json');rows=run.score_rows();byid={r['candidate']:r for r in rows}
    ev,names,_=old_analysis.patient();cache=OUT/'analysis/diagnostic_observables.json'
    data=rt.read(cache) if cache.exists() else dict(patient={},runs={})
    modes=[('ALL',None),('TA',1),('TB',0)]
    if not data['patient']:
        for mode,k in modes:data['patient'][mode]=describe(ev.fit if k is None else ev.fit[ev.fit_labels==k],names)
    for row in rows:
        cid=row['candidate']
        if cid in data['runs']:continue
        src=Path(row['source']);r,t,cn=run.load_times(src);assert names==cn
        # Classification happens after score creation; these outputs are never
        # consumed by the proposal generator or the label-free scorer.
        with np.load(src.with_suffix('.npz')) as z:
            ids=np.array([int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
            labels=z['event_mode'][ids]
        data['runs'][cid]={mode:describe(t if k is None else t[labels==k],names) for mode,k in modes}
    rt.write(cache,data)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    titles=['整体rank分布误差 ↓','杆内顺序/杆间时差分布误差 ↓','参与结构分布误差 ↓']
    x0=run.old.vector(rt.read(OUT/'candidates'/f"{p['reference_id']}.json"))
    pdf_tmp=F/'dense_parameter_response.tmp.pdf'
    with PdfPages(pdf_tmp) as pdf:
        fig,axs=plt.subplots(6,3,figsize=(14,19),sharex='row')
        for line in p['response_lines']:
            j=line['axis'];xx=np.array(line['values']);ids=line['ids']
            for k in range(3):
                yy=[byid[c]['components'][k] if c in byid and byid[c]['J'] is not None else np.nan for c in ids]
                ax=axs[j,k];ax.plot(xx,yy,'o-',ms=4,color=colors[j]);ax.axvline(x0[j],color='gray',ls=':',lw=1)
                ax.set(xlabel=line['label'],ylabel=titles[k],xlim=(xx[0],xx[-1]));ax.grid(alpha=.2)
                ax.text(.03,.94,f'{np.isfinite(yy).sum()}/13个实测点',transform=ax.transAxes,va='top',fontsize=8)
        fig.suptitle('无标签三组分布目标：每条线只改变一个参数\n固定拓扑2511 / 噪声847401；缺失点表示尚未完成或不可评分，不插值；灰竖线为共同工作点',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(F/'label_free_loss_response.png',dpi=120);pdf.savefig(fig);plt.close(fig)
        metric_names=['平均rank绝对误差 ↓','杆内先后概率误差 ↓','SCL−ICL中位时差 (ms)','触点参与概率误差 ↓']
        order=[names.index(n) for n in DISPLAY]
        table=[]
        for line in p['response_lines']:
            j=line['axis'];xx=np.array(line['values']);fig,axs=plt.subplots(3,5,figsize=(20,12))
            for row,(mode,_) in enumerate(modes):
                patient=data['patient'][mode];vals=[];qlo=[];qhi=[];nn=[]
                for cid in line['ids']:
                    q=data['runs'].get(cid,{}).get(mode)
                    v=metrics(q,patient,names) if q else [np.nan]*6;vals.append(v)
                    qlo.append(q['rod_lag_ms']['q05'] if q else np.nan);qhi.append(q['rod_lag_ms']['q95'] if q else np.nan);nn.append(q['N'] if q else None)
                    table.append(dict(axis=j,value=float(xx[len(vals)-1]),candidate=cid,mode=mode,rank_mean_error=v[0],within_rod_order_error=v[1],rod_lag_median_ms=v[2],participation_error=v[3],both_rods=v[4],N=v[5]))
                arr=np.array(vals,dtype=float)
                for k in range(4):
                    ax=axs[row,k];ax.plot(xx,arr[:,k],'o-',color=colors[j],ms=4)
                    if k==2:
                        ax.fill_between(xx,np.array(qlo,dtype=float),np.array(qhi,dtype=float),color=colors[j],alpha=.12,label='模型事件5–95%')
                        ax.vlines(xx,np.array(qlo,dtype=float),np.array(qhi,dtype=float),color=colors[j],alpha=.3,lw=1)
                        lag=patient['rod_lag_ms'];ax.axhspan(lag['q05'],lag['q95'],color='black',alpha=.07,label='患者事件5–95%');ax.axhline(lag['median'],color='black',ls='--',label='患者中位数')
                    else:ax.axhline(0,color='black',ls='--',lw=1)
                    ax.axvline(x0[j],color='gray',ls=':',lw=1);ax.grid(alpha=.18)
                    ax.set(xlabel=line['label'],ylabel=f'{mode}：{metric_names[k]}',xlim=(xx[0],xx[-1]))
                    if k==0:ax.text(.02,.96,'各点事件数：'+','.join(str(n) if n is not None else '—' for n in nn),transform=ax.transAxes,va='top',fontsize=6)
                ax=axs[row,4];mu=np.array(patient['mean_rank'],float)[order];low=np.array(patient['rank_q05'],float)[order];high=np.array(patient['rank_q95'],float)[order];yy=np.arange(15)
                ax.fill_betweenx(yy,low,high,color='black',alpha=.08);ax.plot(mu,yy,'o-',color='black',ms=3,label='患者平均rank')
                anchor=data['runs'][p['reference_id']][mode];v=np.array(anchor['mean_rank'],float)[order]
                for indices in [np.arange(4),np.arange(4,15)]:ax.plot(v[indices],yy[indices],'o--',color='#0072B2',ms=3,label='共同工作点' if indices[0]==0 else None)
                ax.set(yticks=yy,yticklabels=DISPLAY,ylim=(14.5,-.5),xlim=(-.05,1.05),xlabel='相对rank（早→晚）',title=f'{mode} 模板对照')
                # No line connects SCL to ICL, including the patient template.
                ax.lines[0].remove()
                for indices in [np.arange(4),np.arange(4,15)]:ax.plot(mu[indices],yy[indices],'o-',color='black',ms=3,label='患者平均rank' if indices[0]==0 else None)
                ax.axhline(3.5,color='gray',lw=.8)
                if row==0:ax.legend(fontsize=7)
            axs[0,2].legend(fontsize=7)
            fig.suptitle(f'{line["label"]}：单参数实测响应；其余五个标量及背景固定\nALL进入无标签拟合；TA/TB仅分组诊断。色带为事件分布范围，不是网络/噪声重复的置信区间。',fontsize=13)
            fig.tight_layout(rect=(0,0,1,.95));fig.savefig(F/f'axis{j}_observables.png',dpi=120);pdf.savefig(fig);plt.close(fig)
        rt.write(OUT/'analysis/parameter_metric_table.json',dict(rows=table,statistics='one fixed topology and noise, full 60-second simulation; events describe within-run distributions',labels_used_for_optimization=False))
        # Winner is selected from J only. Displaying labels follows nomination.
        good=sorted([r for r in rows if r['J'] is not None],key=lambda r:r['J']);top=list(dict.fromkeys([p['reference_id'],good[0]['candidate']]))
        fig,axs=plt.subplots(2,1+len(top),figsize=(5*(1+len(top)),10),sharey=True,squeeze=False)
        for row,(mode,_) in enumerate(modes[1:]):
            for col,cid in enumerate([None]+top):
                q=data['patient'][mode] if cid is None else data['runs'][cid][mode]
                mu=np.array(q['mean_rank'],float)[order];part=np.array([q['contacts'][n]['participation'] for n in DISPLAY],float)
                ax=axs[row,col];ax.barh(np.arange(15),part,color='#d0d0d0',height=.75,label='参与概率')
                for indices in [np.arange(4),np.arange(4,15)]:ax.plot(mu[indices],indices,'o-',color='#0072B2',label='平均rank' if indices[0]==0 else None)
                ax.set(yticks=np.arange(15),yticklabels=DISPLAY,ylim=(14.5,-.5),xlim=(0,1.02),xlabel='概率 / 归一化rank',title=('患者FIT' if cid is None else ('共同工作点' if cid==p['reference_id'] else '当前无标签最低分'))+f' · {mode} · N={q["N"]}')
                ax.axhline(3.5,color='gray',lw=.8)
        axs[0,0].legend(fontsize=8);fig.suptitle('提名后的患者TA/TB模板与参与结构；最低分不等于传播已恢复',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(F/'patient_best_templates.png',dpi=130);pdf.savefig(fig);plt.close(fig)
    pdf_tmp.replace(F/'dense_parameter_response.pdf')
    rt.write(OUT/'analysis/ranking.json',dict(ranking=[dict(candidate=r['candidate'],J=r['J'],N=r['N'],reused=r.get('reused',False)) for r in good],time=time.time(),labels_used=False))
    (F/'README.md').write_text('### label_free_loss_response.png\n六个参数各13个位置的三组无标签分布误差，所有曲线围绕相同工作点。缺值不连接，不以预测值补点。**关注点**：实测点是否增加，哪个观测组随参数变化。\n\n'+''.join(f'### axis{j}_observables.png\n{label}的整体及TA/TB分组响应，附患者模板与共同工作点。色带是同一次轨迹内事件5–95%范围，非跨seed置信区间。**关注点**：误差、毫秒差和参与之间的权衡；TA/TB未进入优化。\n\n' for j,label in enumerate(run.AXES))+'### patient_best_templates.png\n无标签目标提名后的TA/TB模板与患者对照，Y轴按杆固定。灰条为参与概率，蓝线为平均rank。**关注点**：低loss是否同时改善两种条件传播，而不只改善频率或平均。\n\n### dense_parameter_response.pdf\n上述实时实测图的合订版本。未完成的点保留为空。**关注点**：本轮仅一个拓扑及一个噪声，不能推断跨网络稳定性。\n',encoding='utf-8')
    (OUT/'analysis/scientific_progress.md').write_text(f'# 无标签单seed加密实验\n\n更新：{time.strftime("%Y-%m-%d %H:%M:%S")}。已有{len(rows)}个评分，其中{sum(not r.get("reused") for r in rows)}个新物理结果；其余为32条已验证旧轨迹的单噪声复用。\n\n当前最低无标签J：`{good[0]["candidate"]}`，J={good[0]["J"]:.6f}。此处没有把TA/TB诊断反馈进提案。\n\n[实时参数×观测PDF](../figures/dense_parameter_response.pdf)。不接受仅凭最低分或两类标签宣称患者双模式恢复。\n',encoding='utf-8')

def report():
    # User changed the recurring delivery contract on 2026-09-15. Keep the
    # previous producer for provenance, but never call it from the live loop.
    from scripts.report_topic4_label_free_key_results import report as key_report
    return key_report()

if __name__=='__main__':report()

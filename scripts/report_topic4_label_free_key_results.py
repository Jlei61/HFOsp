"""Live key results only: best multievent GIF, paired snapshot, three figures.

No PDF; no candidate galleries. Full numerical diagnostics remain machine data.
"""
from pathlib import Path
import time, warnings
import numpy as np
from scripts import run_topic4_label_free_dense_search as run
from scripts import analyze_topic4_three_observable_bo as old_analysis
from scripts.report_topic4_label_free_dense import describe,metrics
from scripts import topic4_patient_mean_spectra as patient_mean
rt=run.rt;OUT=run.OUT
KEY=OUT/'key_results'

def gather(rows):
    ev,names,_=old_analysis.patient();path=OUT/'analysis/diagnostic_observables.json'
    data=rt.read(path) if path.exists() else dict(patient={},runs={})
    if not data['patient']:
        for name,k in [('ALL',None),('TA',1),('TB',0)]:data['patient'][name]=describe(ev.fit if k is None else ev.fit[ev.fit_labels==k],names)
    for row in rows:
        cid=row['candidate']
        if cid in data['runs']:continue
        src=Path(row['source']);r,t,cn=run.load_times(src);assert cn==names
        with np.load(src.with_suffix('.npz')) as z:
            ids=np.array([int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
            labels=z['event_mode'][ids]
        data['runs'][cid]={name:describe(t if k is None else t[labels==k],names) for name,k in [('ALL',None),('TA',1),('TB',0)]}
    rt.write(path,data);return ev,names,data

def parameter_figures(plan,data,names):
    import matplotlib.pyplot as plt
    cols=[('位置微调',list(range(4))),('核向外EE强度',[4]),('全局EE方向',[5])]
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#B87B00','#7851A9']
    labels=['左核X','左核Y','右核X','右核Y','核向外EE','EE轴方向']
    baseline=run.old.vector(rt.read(OUT/'candidates'/f"{plan['reference_id']}.json"))
    descriptions=[('rank',0,'平均传播顺序','平均归一化rank误差 ↓'),('local_order',1,'杆内相对顺序','节点对先后概率误差 ↓'),('participation',3,'两杆触点参与','触点参与概率误差 ↓')]
    paths=[];table=[]
    for name,index,title,ylabel in descriptions:
        fig,axes=plt.subplots(3,3,figsize=(12.8,8.8),sharey=True)
        for row,mode in enumerate(['ALL','TA','TB']):
            for col,(colname,js) in enumerate(cols):
                ax=axes[row,col]
                for j in js:
                    line=plan['response_lines'][j];xx=np.array(line['values']);xx=xx-baseline[j] if col==0 else xx
                    values=[];counts=[]
                    for x,cid in zip(xx,line['ids']):
                        q=data['runs'].get(cid,{}).get(mode)
                        v=metrics(q,data['patient'][mode],names) if q else [np.nan]*6
                        values.append(v[index]);counts.append(q['N'] if q else None)
                        table.append(dict(observable=name,mode=mode,axis=j,parameter_value=float(x),candidate=cid,value=v[index],N=counts[-1],raw_metrics=v))
                    ax.plot(xx,np.array(values,float),'o-',color=colors[j],ms=3.5,lw=1.4,label=labels[j])
                    # This invisible span fixes the full planned x-axis even
                    # while most trajectories are still pending.
                    ax.update_datalim(np.array([[xx[0],0],[xx[-1],0]]));ax.autoscale_view()
                ax.axhline(0,color='black',ls='--',lw=.8)
                ax.axvline(0 if col==0 else baseline[js[0]],color='gray',ls=':',lw=.8)
                ax.grid(alpha=.16);ax.set_xlabel(['相对同一工作点位移 (mm)','向外EE权重倍数','相对参考EE轴偏移 (°)'][col])
                if col==0:ax.set_ylabel(f'{mode}\n{ylabel}')
                if row==0:ax.set_title(colname);ax.legend(fontsize=8,ncol=2 if col==0 else 1)
        fig.suptitle(f'{title}：参数 × 观测\n固定网络种子2511 / 噪声847401；总体用于无标签拟合，TA/TB仅诊断；空缺处不补点',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.93));path=KEY/f'parameter_{name}.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    rt.write(OUT/'analysis/parameter_metric_table.json',dict(rows=table,display_metrics='three raw summary errors; distinct from distributional training loss; SCL/ICL equally weighted',labels_used_for_optimization=False))
    return paths

def comparison(c,noise,r,a,ids,physics,folder,ev,names,chosen,xlim):
    import matplotlib.pyplot as plt
    from scripts.paper_figures import plot_topic4_recovery_review as fr
    from scripts.render_topic4_shape_output_gifs import field_tile
    order=[names.index(n) for n in old_analysis.DISPLAY];dt=float(a['contact_envelope_dt_ms']);fd=float(a['sheet_activity_frame_ms'])
    vmax=max(float(np.quantile(a['sheet_activity_counts'][round(1500/fd):],.999)),1)
    offsets=[-20,0,40,80];selection=[]
    fig=plt.figure(figsize=(18,8));grid=fig.add_gridspec(2,6,width_ratios=[2,2,1,1,1,1],wspace=.30,hspace=.35)
    for row,(mode,k) in enumerate([('TA',1),('TB',0)]):
        ax=fig.add_subplot(grid[row,0]);patient_mean.draw(ax,mode,xlim,fr.display)
        events=[e for e in chosen if e['mode']==mode]
        ax=fig.add_subplot(grid[row,1]);fr.display.contact_axis(ax);ax.set(xlim=xlim,xlabel='相对最早质心 (ms)')
        if not events:
            ax.text(.5,.5,f'本运行未观察到{mode}',transform=ax.transAxes,ha='center');continue
        i=events[0]['event'];lo,hi=r['events'][i]['window_ms'];t=a['centroid_ms'][i];zero=float(np.nanmin(t));part=np.isfinite(t)[order]
        mass=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T.copy();mass/=np.maximum(mass.max(axis=1,keepdims=True),1e-20);mass[~part]=np.nan
        cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777')
        ax.set_facecolor('#aaa');ax.imshow(mass,aspect='auto',extent=[lo-zero,hi-zero,14.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        fr.display.centroid_lines(ax,t[order]-zero,color='cyan');fr.display.contact_axis(ax)
        ax.set(xlim=xlim,xlabel='相对最早质心 (ms)',title=f'模型{mode}，事件{i}\n发放包络（逐触点归一化）')
        selected=[]
        for col,offset in enumerate(offsets,2):
            ax=fig.add_subplot(grid[row,col]);now=zero+offset
            if lo<=now<hi:
                fi=round(now/fd);ax.imshow(field_tile(a['sheet_activity_counts'][fi],c,a,physics,vmax,size=250),extent=[0,20,0,20]);selected.append(dict(relative_ms=offset,absolute_ms=now,frame_index=fi))
            else:ax.text(.5,.5,'观察窗外',transform=ax.transAxes,ha='center')
            ax.set(title=f'{offset:+d} ms',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)');ax.tick_params(labelsize=7)
            if col==2:ax.set_ylabel('y (mm)')
        selection.append(dict(mode=mode,event=i,window_ms=[lo,hi],snapshot_frames=selected))
    fig.suptitle(f'{c["id"]} · 网络{c["topology"]} / 噪声{noise}\n患者平均模板 → 模型同一事件包络 → 全部原生E活动；白线core，青色触点；场共享色标0–{vmax:.0f}发放/2 ms/网格',fontsize=12)
    fig.subplots_adjust(left=.045,right=.99,bottom=.09,top=.86);dest=folder/'patient_model_snapshots.png';fig.savefig(dest,dpi=140);plt.close(fig)
    rt.write(folder/'snapshot_selection.json',dict(selection='first displayed GIF event of each mode; no patient similarity selection',events=selection,native_frame_ms=fd,field_color_limits=[0,vmax],time_alignment='subtract the earliest participating centroid once per event; no stretching',patient=patient_mean.identity()))
    return dest

def best_media(best,ev,names):
    cid=best['candidate'];folder=KEY/cid;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'manifest.json'
    patient_identity=patient_mean.identity()
    if manifest.exists() and rt.read(manifest).get('media_schema')=='key_delivery_v3_patient_mean' and rt.read(manifest).get('patient_template')==patient_identity:return rt.read(manifest)
    from scripts.analyze_topic4_core_connectivity_search import load_unit
    from scripts.media_topic4_three_observable_bo import render_mean_template_gif
    src=Path(best['source']);r,a,ids=load_unit(src,1500.);c=rt.read(OUT/'candidates'/f'{cid}.json');c['topology']=run.TOPOLOGY
    physics=rt.read(src.parents[1]/'applied_physics.json')
    shown=[int(i) for k in [1,0] for i in ids[a['event_mode'][ids]==k][:3]]
    relative=[np.array(r['events'][i]['window_ms'])-np.nanmin(a['centroid_ms'][i]) for i in shown]
    xlim=(min(-100.,float(np.floor(min(v[0] for v in relative)/25)*25)),max(200.,float(np.ceil(max(v[1] for v in relative)/25)*25))) if relative else (-100.,200.)
    render_mean_template_gif(c,run.NOISE,r,a,ids,physics,folder,display_xlim=xlim,patient_mean_draw=patient_mean.draw,patient_mean_identity=patient_identity)
    gif=folder/'patient_mean_native_multievent.gif'
    if gif.exists():
        selected=rt.read(folder/'patient_mean_native_multievent.json')['events'];snapshot=comparison(c,run.NOISE,r,a,ids,physics,folder,ev,names,selected,xlim)
    else:selected=[];snapshot=None
    result=dict(media_schema='key_delivery_v3_patient_mean',patient_template=patient_identity,display_xlim_ms=xlim,candidate=cid,J=best['J'],N=best['N'],reused=best.get('reused',False),source=str(src),source_sha256=rt.sha(src),gif=str(gif) if gif.exists() else None,snapshot=str(snapshot) if snapshot else None,events=selected,selection='minimum label-free J on topology2511/noise847401; first three eligible events per mode, chronological',mode_counts={name:int((a['event_mode'][ids]==k).sum()) for name,k in [('TA',1),('TB',0)]},human_visual_review='PENDING',agent_visual_review='PENDING',time=time.time())
    rt.write(manifest,result)
    (folder/'README.md').write_text('### patient_mean_native_multievent.gif\n当前无标签首名的多事件动画，每模式最早三个合格事件按真实时间先后排列。左为患者每类32个冻结事件的平均HFO包络（原始EEG当前不可读，未冒充平均频谱），中为全体E原生场，右为模型发放密度包络。**关注点**：三加三展示不代表自然模式占比；患者平均图的宽度包含事件间变异，模型包络也不是HFO频谱。\n\n### patient_model_snapshots.png\n每模式使用GIF中最早的同一事件，与患者平均模板及四个原生时刻并列。患者图沿用Fig2C的时间横轴、触点分层和配色，SCL/ICL按固定行序分组；每事件仅统一平移，未逐触点对齐，未拉伸时间。**关注点**：患者点为平均包络主峰区质心，不是全FIT事件平均质心或起燃点；原始频谱版本仍待原始数据可读。\n',encoding='utf-8')
    return result

def report():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.topic4_pdf_font_guard import install
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9})
    KEY.mkdir(exist_ok=True);plan=rt.read(OUT/'plan.json');rows=run.score_rows()
    identity=[(r['candidate'],r['J'],r['N'],r.get('source_sha256')) for r in rows]
    report_sha=rt.sha(__file__)
    if (KEY/'current.json').exists():
        saved=rt.read(KEY/'current.json')
        if saved.get('score_identity')==rt.json_safe(identity) and saved.get('report_sha256')==report_sha and saved.get('best',{}).get('patient_template')==patient_mean.identity() and all(Path(f).exists() for f in saved['parameter_figures']):return
    ev,names,data=gather(rows)
    figures=parameter_figures(plan,data,names)
    good=sorted([r for r in rows if r['J'] is not None],key=lambda r:r['J'])
    rt.write(OUT/'analysis/ranking.json',dict(ranking=[{k:r.get(k) for k in ['candidate','J','N','reused']} for r in good],time=time.time(),labels_used=False))
    best=best_media(good[0],ev,names)
    rt.write(KEY/'current.json',dict(best=best,parameter_figures=figures,completed=len(rows),new_complete=sum(not r.get('reused') for r in rows),automatically_generate_pdf=False,score_identity=identity,report_sha256=report_sha,time=time.time()))
    (KEY/'README.md').write_text('# 本轮只交付关键结果\n\n'+f'当前无标签评分首名：`{best["candidate"]}`；N={best["N"]}，TA={best["mode_counts"]["TA"]}，TB={best["mode_counts"]["TB"]}。来源：'+('复用既有真实轨迹，新轮轨迹仍待完成。' if best['reused'] else '本轮新完成轨迹。')+'\n\n'+f'[多事件GIF]({best["gif"]}) · [患者平均模板、模型时序与field快照]({best["snapshot"]})\n\n'+''.join(f'### {Path(f).name}\n每图对应一个观测，列为位置/EE/方向，行为总体/TA/TB。各点只变一个参数，位置四条线以颜色区分两核X/Y。**关注点**：与患者的均值/先后概率/参与误差如何随参数变化；各杆等权，未完成点留空，TA/TB不进入优化。\n\n' for f in figures)+'旧PDF和图册保留为历史产物，不再自动更新。全部原始统计仍在analysis，三张展示图的摘要误差不同于训练中比较完整分布的核距离。\n',encoding='utf-8')
    (OUT/'analysis/scientific_progress.md').write_text(f'# 当前关键结果\n\n已有{len(rows)}个评分，其中{sum(not r.get("reused") for r in rows)}个新物理结果。当前最低无标签J：`{best["candidate"]}`，J={best["J"]:.6f}。\n\n[关键结果：GIF、模板与field快照、三张参数图](../key_results/README.md)。只根据无标签J提名，TA/TB展示不反馈优化；尚未接受双模式恢复。\n',encoding='utf-8')

if __name__=='__main__':report()

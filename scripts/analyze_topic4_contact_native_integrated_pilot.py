#!/usr/bin/env python3
"""Deliver separate observables, native-prior tradeoffs and full event movies."""
from pathlib import Path
import argparse, csv, json, pickle, subprocess, sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scripts import run_topic4_contact_native_integrated_pilot as run
from scripts import analyze_topic4_observable_loss_physical_pilot as visuals

OUT = run.OUT
F = OUT/'figures'
KEYS = run.KEYS
LABELS = ['Contact participation', 'Centroid order / lag', 'Local duration / shape',
          'Recruitment timing', 'All-contact envelope']
COLORS = ['#a44b69', '#377faa', '#b3833f']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42,
                     'axes.spines.top':False, 'axes.spines.right':False})


def csvwrite(path, rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display(row, d):
    i = run.execution.anchor_index(row['candidate'], d['anchors'])+1
    cid = row['candidate_id']
    if cid in [c['candidate_id'] for c in d['anchors']]:
        return f'Place {i}: original'
    parts = cid.split('_')
    prefix = 'timing' if cid.startswith('tshape') else ('native' if cid.startswith('native') else 'integrated')
    return f'Place {i}: {prefix} {parts[-2]} {"+" if parts[-1]=="plus" else "-"}'


def save(fig, name, entries, description):
    for ext in ['png', 'pdf']:
        fig.savefig(F/f'{name}.{ext}', dpi=155, bbox_inches='tight')
    plt.close(fig)
    entries.append((name+'.png / .pdf', description))


def training(d, rows, entries):
    units = []
    for i, r in enumerate(rows, 1):
        for uid, u in r['units'].items():
            obs = u['observation']
            units.append(dict(condition_index=i, candidate_id=r['candidate_id'], name=display(r,d),
                              placement=run.execution.anchor_index(r['candidate'],d['anchors'])+1,
                              unit=uid, N=obs['N'], observation_loss=obs['loss'],
                              R_unsupported=u['regularizer'].get('mean_unsupported_fraction'),
                              total_loss=None if obs['loss'] is None or u['regularizer'].get('mean_unsupported_fraction') is None else obs['loss']+d['lambda_contract']['primary']*u['regularizer']['mean_unsupported_fraction'],
                              **dict(zip(run.prior.PARAMETERS, run.prior.vector(r['candidate']).tolist())),
                              **{k:obs.get('blocks',{}).get(k,{}).get('D_off') for k in KEYS}))
    csvwrite(OUT/'training_parameter_scores.csv', units)
    csvwrite(OUT/'condition_index.csv', [dict(index=i+1, candidate_id=r['candidate_id'], name=display(r,d),
              eligible=r['ranking_eligible'], observation_loss=r['observation_loss'], R_unsupported=r['R_unsupported'], total_loss=r['loss']) for i,r in enumerate(rows)])
    fig, axs = plt.subplots(1, 6, figsize=(15, max(5,len(rows)*.29)), sharey=True)
    for j, key in enumerate(KEYS+['R_unsupported']):
        ax=axs[j]
        for i,r in enumerate(rows):
            color=COLORS[run.execution.anchor_index(r['candidate'],d['anchors'])]
            for offset,(uid,u) in zip([-.10,.10],r['units'].items()):
                v=u['regularizer'].get('mean_unsupported_fraction') if key=='R_unsupported' else u['observation'].get('blocks',{}).get(key,{}).get('D_off')
                if v is not None:
                    ax.scatter(v,i+offset,c=color,marker='o' if u['topology_seed']==6101 else '^',s=23)
        ax.set_title((LABELS+['Unsupported activity\n(structural prior)'])[j],fontsize=9)
        ax.set_xlabel('Separate error; lower is better' if j<5 else 'Activity fraction')
        ax.grid(axis='x',alpha=.16)
    axs[0].set(yticks=range(len(rows)),yticklabels=[display(r,d) for r in rows]);axs[0].invert_yaxis()
    fig.suptitle('Five patient-observation errors and the native-activity prior remain separate',fontsize=13)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.text(.5,.015,'Colors: fixed placements 1 / 2 / 3. Circle / triangle: topology 6101 / 6102. Two-label appearance is not a ranking gate.',ha='center')
    save(fig,'five_errors_and_native_prior',entries,'五项患者观测误差与原生场正则分别展示，每点是一张固定拓扑上的一次 24 秒运行。颜色是三个固定布局，圆与三角是拓扑 6101 与 6102。**关注点**：分项恶化不能被总分掩盖；正则比例不是患者机制恢复率。')
    fig,axs=plt.subplots(5,7,figsize=(18,12),squeeze=False)
    for pi,p in enumerate(run.prior.PARAMETERS):
        for j,key in enumerate(KEYS+['observation_loss','R_unsupported']):
            ax=axs[pi,j]
            for u in units:
                if u[key] is not None:
                    ax.scatter(u[p],u[key],c=COLORS[u['placement']-1],marker='o' if '6101' in u['unit'] else '^',s=15,alpha=.8)
            ax.set_xlabel(visuals.PLABELS[pi],fontsize=8)
            if pi==0:ax.set_title((LABELS+['Five-term mean','Native prior'])[j],fontsize=8)
            ax.grid(alpha=.12)
    fig.suptitle('Parameter changes versus each observable error',fontsize=14)
    fig.tight_layout(rect=(0,.04,1,.96));fig.text(.5,.012,'Joint parameter proposals; these associations are not isolated parameter interventions. Colors = placements; symbols = topology.',ha='center')
    save(fig,'parameter_by_observable',entries,'五种物理参数分别对应五项误差、观测平均误差和原生场正则。所有网络点保留，颜色与形状含义沿用分项图。**关注点**：每个提案联合改动多个参数，图展示响应关联而非单参数因果效应。')
    fig,ax=plt.subplots(figsize=(8,5))
    for i,r in enumerate(rows,1):
        if r['ranking_eligible']:
            ax.scatter(r['observation_loss'],r['R_unsupported'],c=COLORS[run.execution.anchor_index(r['candidate'],d['anchors'])],s=35)
            ax.annotate(str(i),(r['observation_loss'],r['R_unsupported']),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Five observation terms (mean)',ylabel='Unsupported activity prior',title='Patient-observation fit and native support: inspect the tradeoff')
    ax.grid(alpha=.2)
    save(fig,'observation_native_tradeoff',entries,'每点先对两个网络等权，横轴为五项观测平均误差，纵轴为缺少前驱支持的活动比例。序号与 condition_index.csv 一致。**关注点**：同时向左下变化才表示两个方向共同改善；不能只凭正则下降接受模型。')
    # Keep native-prior and pure-observation rankings on the identical pool.
    tradeoffs=[]
    for lam in d['lambda_contract']['grid']:
        eligible=sorted([r for r in rows if r['ranking_eligible']],key=lambda r:(r['observation_loss']+lam*r['R_unsupported'],r['candidate_id']))
        for rank,r in enumerate(eligible,1):
            tradeoffs.append(dict(lambda_value=lam,rank=rank,candidate_id=r['candidate_id'],
                                  observation_loss=r['observation_loss'],R_unsupported=r['R_unsupported'],
                                  combined=r['observation_loss']+lam*r['R_unsupported']))
    csvwrite(OUT/'same_pool_lambda_rankings.csv',tradeoffs)


def final_delivery(d, train_rows, entries):
    rows=run.base.read(OUT/'confirmation_scores.json')['candidates']
    nomination=run.base.read(OUT/'nomination.json')
    old=pickle.load((run.prior.OLD/'training_objective_v2_1.pkl').open('rb'))
    patients=run.base.read(run.prior.OUT/'design.json')['patient_training']
    visuals.F=F
    all_events=[];summaries=[];movies=[];blocks=[]
    for ci,r in enumerate(rows,1):
        clips=[]
        for uid,u in r['units'].items():
            native_rows,selected,grids=visuals.native_unit(u['worker_path'],r['candidate'],old)
            exact={e['event_id']:e for e in u['events']}
            for e in native_rows:
                e['local_width_ms']=exact[e['event_id']]['local_width_ms']
                e['recruitment_span_ms']=exact[e['event_id']]['recruitment_span_ms']
                all_events.append(dict(candidate_id=r['candidate_id'],unit=uid,**e))
            clips.extend([(uid,g) for g in selected])
            if grids:
                visuals.contact_grid_pdf(grids,F/f'condition{ci}_{uid}_all_events.pdf')
            else:
                run.write(OUT/f'condition{ci}_{uid}_no_primary_events.json',
                          dict(status='NO_PRIMARY_EVENTS', worker_path=u['worker_path']))
            for label in [-1,0,1]:
                subset=[e for e in native_rows if label==-1 or e['mode']==label]
                for metric in visuals.METRICS+['recruitment_span_ms','centroid_span_ms']:
                    summaries.append(dict(candidate_id=r['candidate_id'],unit=uid,mode=label,metric=metric,
                                          **run.distribution([e[metric] for e in subset if e[metric] is not None])))
            for start in [0,6000,12000,18000]:
                subset=[e for e in native_rows if start<=e['window_start_ms']<start+6000]
                times=sorted(e['window_start_ms'] for e in subset)
                blocks.append(dict(candidate_id=r['candidate_id'],unit=uid,start_ms=start,n_events=len(subset),
                                   TA_count=sum(e['mode']==1 for e in subset),TB_count=sum(e['mode']==0 for e in subset),
                                   within_block_interval_mean_ms=float(np.mean(np.diff(times))) if len(times)>1 else None))
        for lab in [1,0]:
            visuals.render_movie(clips,[p for p in patients if p['mode']==lab],r['candidate_id'],lab,ci,entries,movies)
    csvwrite(OUT/'confirmation_events.csv',all_events)
    csvwrite(OUT/'confirmation_distributions.csv',summaries)
    csvwrite(OUT/'confirmation_six_second_blocks.csv',blocks)
    run.write(OUT/'movie_event_manifest.json',movies)
    reference_id=next(cid for cid,roles in nomination['roles'].items() if 'starting_reference' in roles)
    reference=next(r for r in rows if r['candidate_id']==reference_id)
    paired=[]
    for r in rows:
        for uid,u in r['units'].items():
            ref=reference['units'][uid]
            values={k:u['observation'].get('blocks',{}).get(k,{}).get('D_off') for k in KEYS}
            baseline={k:ref['observation'].get('blocks',{}).get(k,{}).get('D_off') for k in KEYS}
            for k in ['mean_unsupported_fraction','primary_window_fraction','mean_background_mass_per_frame']:
                values[k]=u['regularizer'].get(k);baseline[k]=ref['regularizer'].get(k)
            values['N']=u['observation']['N'];baseline['N']=ref['observation']['N']
            for k,v in values.items():
                b=baseline[k]
                paired.append(dict(candidate_id=r['candidate_id'],unit=uid,metric=k,value=v,reference=b,
                                   difference=None if v is None or b is None else v-b))
    csvwrite(OUT/'confirmation_paired_components.csv',paired)
    with np.load(run.REV/'patient_training_representation.npz') as z:
        masks=z['participation']>0;local=z['local_shape'];rec=z['recruitment'][:,:,0];weights=z['weights']
        patient_width=np.array([np.median((l[:,17]-l[:,1])[m]) for l,m in zip(local,masks)])
        patient_rec=np.array([np.ptp(x[m]) for x,m in zip(rec,masks)])
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,metric,pat,title in zip(axs,['local_width_ms','recruitment_span_ms'],[patient_width,patient_rec],['Within-contact duration','Between-contact recruitment span']):
        order=np.argsort(pat);ax.step(pat[order],np.cumsum(weights[order]),color='black',lw=2.2,label='Patient TRAIN',where='post')
        for ci,r in enumerate(rows,1):
            for j,(uid,u) in enumerate(r['units'].items()):
                vals=np.sort([e[metric] for e in u['events']])
                if len(vals):ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=plt.get_cmap('tab10')(ci-1),ls=['-','--'][j],alpha=.8,label=f'Condition {ci}' if j==0 else None)
        ax.set(xlabel='ms',ylabel='Event cumulative probability',title=title,ylim=(0,1.01));ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.tight_layout()
    save(fig,'patient_model_duration_and_recruitment',entries,'患者 TRAIN 与每个确认网络分别比较事件内接触点宽度中位数、接触点 50% 质量时间跨度的分布，不区分 TA/TB。患者按冻结训练权重加权，模型各网络单独绘制，时间单位均为 ms。**关注点**：局部宽度与跨区招募分开；分布尾部及散布同样需要恢复。')
    text='# 整合 pilot：确认结果待科学审阅\n\n'
    text+='本轮用五项观测损失与原生场支持先验共同引导一个自适应批次。首批 12 次继承旧目标选出的参数，不能倒称为新版优化；同池比较只能说明排序选择的改变，不能唯一归因算法或机制。\n\n'
    text+='|条件|提名来源|五项观测平均|原生场先验|局部宽度，两网络中位数 ms|\n|---|---|---|---|---|\n'
    for ci,r in enumerate(rows,1):
        widths=' / '.join(str(u['distributions']['local_width_ms']['median']) for u in r['units'].values())
        text+=f'|{ci}: {display(r,d)}|{", ".join(nomination["roles"][r["candidate_id"]])}|{r["observation_loss"]}|{r["R_unsupported"]}|{widths}|\n'
    text+='\n对原始工作点 1 的同拓扑、新噪声配对比较：\n\n'
    for ci,r in enumerate(rows,1):
        if r['candidate_id']==reference_id:continue
        details=[]
        for key,label in [('local_shape','局部形态'),('recruitment','招募'),('joint_envelope','完整包络'),('mean_unsupported_fraction','原生场先验')]:
            pp=[p for p in paired if p['candidate_id']==r['candidate_id'] and p['metric']==key and p['difference'] is not None]
            details.append(f'{label} {sum(p["difference"]<0 for p in pp)}/{len(pp)} 个可估计网络改善')
        text+=f'- 条件 {ci}：'+ '；'.join(details)+'。\n'
    text+='\n五项分量、均值/中位数/方差/5–95% 范围、逐网络与逐模式支持分别保留。没有把 pooled events 当独立网络重复，也没有因为两个标签都有就宣称恢复传播。原生场正则依赖 1 mm / 2 ms 粗粒度图支持，是机制先验，不是患者全场测量。\n\n'
    text+='训练全程未调用模型 TA/TB 分类器；确认后的标签仅用于组织动画与分布。低事件数或未观察到少数模式按实际支持报告。原 v1 的 18 条 review 波形已被其审阅使用，本轮不会将其称为新的盲法留后数据。\n\n'
    text+='全部 primary 事件进入 CSV 与逐事件 PDF；GIF 每网络每标签取最早两个事件，保留 SEEG 布局、全接触点与原生场，不增加平滑或拉伸。所有图待人工目视检查；本轮不继续扩搜、不冻结模型、不进入 Fig. 5。\n'
    (OUT/'scientific_review.md').write_text(text)
    entries.append(('condition*_all_events.pdf','逐确认网络的全部 primary 事件接触点时间图，非参与接触点保留实际活动。每页八个事件，统一真实 250 ms 时间窗。**关注点**：不能只从 GIF 少数例子判断全体事件分布。'))


def main(final=False):
    d,obj=run.frozen();F.mkdir(exist_ok=True);entries=[]
    rows=[r for phase in ['history','inherited_A','adaptive_B'] if (OUT/f'{phase}_scores.json').exists()
          for r in run.base.read(OUT/f'{phase}_scores.json')['candidates']]
    training(d,rows,entries)
    if final:final_delivery(d,rows,entries)
    (F/'README.md').write_text('# 整合 pilot 图件\n\n'+'\n\n'.join('### '+name+'\n\n'+desc for name,desc in entries)+'\n')
    qa=[]
    for p in F.glob('*'):
        if p.suffix in ['.png','.gif']:
            with Image.open(p) as im:
                for i in range(getattr(im,'n_frames',1)):im.seek(i);im.load()
                qa.append(dict(file=p.name,frames=getattr(im,'n_frames',1),sha256=run.base.sha(p)))
        elif p.suffix=='.pdf':
            info=subprocess.check_output(['pdfinfo',str(p)],text=True)
            pages=int(next(line.split(':')[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
            with tempfile.TemporaryDirectory(prefix='topic4_pdf_qa_') as tmp:
                subprocess.run(['pdftoppm','-scale-to','128','-png',str(p),str(Path(tmp)/'page')],check=True,capture_output=True)
                rasters=list(Path(tmp).glob('page-*.png'))
                if len(rasters)!=pages:raise RuntimeError(f'PDF page decode incomplete: {p}')
                for raster in rasters:
                    with Image.open(raster) as im:im.load()
            qa.append(dict(file=p.name,pages=pages,sha256=run.base.sha(p)))
    run.write(OUT/'figure_validation.json',dict(stage='final' if final else 'training',files=qa,human_visual_acceptance='PENDING'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)

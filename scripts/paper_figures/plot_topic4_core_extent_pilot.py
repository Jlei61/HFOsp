"""Frozen pilot outputs: paired observations and multi-event native movies."""
from pathlib import Path
import argparse
import csv
import warnings
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import rankdata, spearmanr
from PIL import Image
from scripts import pilot_topic4_core_extent as p
rt=p.rt;OUT=p.OUT;F=OUT/'figures'
ANALYSIS_END_MS=30000.
SHOW_ALL_GEOMETRIES=False
RUN_ROLE='training'
MODE_NAMES={0:'M0',1:'M1'}
MODE_ORDER=[0,1]
CONDITION_LABELS={}


def ranks(table):
    result=np.full(np.asarray(table).shape,np.nan)
    for i,row in enumerate(table):
        ix=np.isfinite(row)
        if ix.sum()>1:result[i,ix]=(rankdata(row[ix])-1)/(ix.sum()-1)
    return result


def avg(x,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(x,axis=axis)


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as stream:
        wr=csv.DictWriter(stream,fieldnames=keys);wr.writeheader();wr.writerows(rows)


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)


def load_unit(c,seed):
    path=p.path_for(c,seed)
    if not path.exists():return None
    r=rt.read(path)
    with np.load(path.with_suffix('.npz')) as z:
        ar={k:z[k] for k in ['centroid_ms','event_mode','primary_event_indices','contact_names','contact_xy_mm',
            'contact_envelope','contact_envelope_dt_ms','sheet_activity_counts','event_phi','h','positions_E','vtheta']}
    if ar['contact_envelope'].shape[0] != len(ar['contact_names']):
        raise RuntimeError('R1 envelope must be contacts x time')
    ar['contact_envelope']=ar['contact_envelope'].T
    ids=np.array([i for i in ar['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=min(ANALYSIS_END_MS,r['actual_duration_ms'])],int)
    return r,ar,ids


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    F.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42})
    plan=rt.read(OUT/'plan.json')
    cs=rt.read(OUT/'all_candidates.json') if (OUT/'all_candidates.json').exists() else plan['candidates']
    ev=rt.load_evaluator(plan['parent_design']);patient=np.asarray(ev.fit);pl=np.asarray(ev.fit_labels)
    names=np.asarray(rt.load_observation_contract(plan['parent_design'])['contact_names'])
    scl=np.array([str(n).startswith('SCL') for n in names]);icl=np.array([str(n).startswith('ICL') for n in names])
    display_order=np.r_[np.flatnonzero(scl),np.flatnonzero(icl)]
    patient_rank=ranks(patient)
    rows=[];events=[];units={}
    observation_seeds=list(dict.fromkeys(plan.get('observation_seeds',plan['training_seeds']+[plan['confirmation_seed']])))
    for c in cs:
        for seed in observation_seeds:
            unit=load_unit(c,seed)
            if unit is None:continue
            units[(c['id'],seed)]=unit;r,ar,ids=unit
            assert np.array_equal(names,ar['contact_names'])
            table=ar['centroid_ms'][ids];labels=ar['event_mode'][ids];active=np.isfinite(table)
            score=p.score_candidate(c,[seed])['units'][0]
            row=dict(candidate=c['id'],seed=seed,role=RUN_ROLE if seed in plan['training_seeds'] else 'confirmation',
                radius_A=c['radii_mm'][0],radius_B=c['radii_mm'][1],n=len(ids),status=r['physical_status'],loss=score['loss_off'])
            for mode in [0,1]:
                model=table[labels==mode];pat=patient[pl==mode];ma=np.isfinite(model);pa=np.isfinite(pat)
                row[f'M{mode}_n']=len(model)
                row[f'M{mode}_no_SCL']=float(np.mean(~ma[:,scl].any(1))) if len(model) else None
                row[f'M{mode}_no_ICL']=float(np.mean(~ma[:,icl].any(1))) if len(model) else None
                row[f'M{mode}_participation_MAE']=float(np.mean(abs(ma.mean(0)-pa.mean(0)))) if len(model) else None
                mr=avg(ranks(model)) if len(model) else np.full(len(names),np.nan)
                pr=avg(patient_rank[pl==mode]);valid=np.isfinite(mr)&np.isfinite(pr)
                row[f'M{mode}_mean_rank_rho']=float(spearmanr(mr[valid],pr[valid]).statistic) if valid.sum()>2 else None
            for i in ids:
                e=r['events'][i];lo,hi=e['window_ms'];dt=float(ar['contact_envelope_dt_ms'])
                env=ar['contact_envelope'][max(0,int(lo/dt)):int(hi/dt)]
                if env.shape[-1]!=len(names):raise RuntimeError('unexpected envelope axes')
                cumulative=np.cumsum(env,axis=0);mass=cumulative[-1];ok=np.isfinite(ar['centroid_ms'][i])&(mass>0)
                qt=np.stack([np.argmax(cumulative>=q*mass,axis=0)*dt for q in [.1,.5,.9]])
                events.append(dict(candidate=c['id'],seed=seed,event_index=int(i),mode=int(ar['event_mode'][i]),
                    time_ms=e['event_time_ms'],n_contacts=int(ok.sum()),no_SCL=bool(~ok[scl].any()),no_ICL=bool(~ok[icl].any()),
                    median_local_width_ms=float(np.median((qt[2]-qt[0])[ok])) if ok.any() else None,
                    early_recruitment_span_ms=float(np.ptp(qt[0,ok])) if ok.any() else None,
                    centroid_span_ms=float(np.ptp(ar['centroid_ms'][i,ok])) if ok.any() else None))
            rows.append(row)
    csv_write(OUT/'per_run_observations.csv',rows);csv_write(OUT/'per_event_observations.csv',events)
    score_rows=[p.score_candidate(c,plan['training_seeds']) for c in cs]
    labels_c=[CONDITION_LABELS.get(c['id'],c['id'].replace('expand_','').replace('selected_','')) for c in cs]
    fig,axs=plt.subplots(2,3,figsize=(max(11,len(cs)*.9),8),layout='constrained')
    metrics=[('loss','Frozen joint loss'),('M0_participation_MAE','M0 participation error'),('M1_participation_MAE','M1 participation error'),
             ('M0_mean_rank_rho','M0 mean-rank correlation'),('M1_mean_rank_rho','M1 mean-rank correlation'),('M1_no_SCL','M1 events without SCL')]
    metrics=[(key,title.replace('M0',MODE_NAMES[0]).replace('M1',MODE_NAMES[1])) for key,title in metrics]
    for ax,(metric,title) in zip(axs.flat,metrics):
        palette=['#B55A30','#2B79A7'] if len(plan['training_seeds'])==2 else plt.get_cmap('tab10')(np.linspace(0,.9,len(plan['training_seeds'])))
        for seed,color in zip(plan['training_seeds'],palette):
            vals=[]
            for c in cs:
                row=next((r for r in rows if r['candidate']==c['id'] and r['seed']==seed),{})
                v=row.get(metric);vals.append(np.nan if v is None else v)
            ax.plot(range(len(cs)),vals,'o-',color=color,label=f'Noise {seed}',lw=1)
        if metric=='M1_no_SCL':ax.axhline(np.mean(~np.isfinite(patient[pl==1])[:,scl].any(1)),color='black',ls='--',label='Patient FIT')
        ax.set(title=title,xticks=range(len(cs)),xticklabels=labels_c);ax.tick_params(axis='x',rotation=60);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8);fig.suptitle('One fixed network; paired noise. Missing score = insufficient events / incomplete trajectory.')
    save(fig,'parameter_observation_response')
    # Fixed patient groups plus all model events: missing contacts stay gray.
    plotted=[c for c in cs if any((c['id'],s) in units for s in plan['training_seeds'])]
    fig,axs=plt.subplots(len(plotted)+1,2,figsize=(10,2*(len(plotted)+1)),squeeze=False,layout='constrained')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#aaaaaa')
    for rowid,c in enumerate([None]+plotted):
        for column,m in enumerate(MODE_ORDER):
            if c is None:
                cloud=patient_rank[pl==m];ix=np.linspace(0,len(cloud)-1,min(100,len(cloud))).astype(int);data=cloud[ix];title=f'Patient FIT {MODE_NAMES[m]}: evenly indexed 100 events'
            else:
                tables=[u[1]['centroid_ms'][u[2]][u[1]['event_mode'][u[2]]==m] for s in plan['training_seeds'] if (u:=units.get((c['id'],s))) is not None]
                data=ranks(np.concatenate(tables)) if tables else np.empty((0,len(names)))
                title=f'{c["id"]} {MODE_NAMES[m]}: all {len(data)} events'
            ax=axs[rowid,column]
            if len(data):ax.imshow(data[:,display_order].T,origin='upper',aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            ax.set(title=title,yticks=range(len(names)),yticklabels=names[display_order],xlabel='Events (within runs in time order)')
    save(fig,'all_event_rank_and_participation')
    nomination=rt.read(OUT/'nomination.json') if (OUT/'nomination.json').exists() else None
    chosen=nomination['candidate'] if nomination else cs[0]
    show_cases=plotted if SHOW_ALL_GEOMETRIES else [cs[0]]+([] if chosen['id']=='baseline' else [chosen])
    for c in show_cases:
        unit=units.get((c['id'],plan['training_seeds'][0]))
        if unit is None:continue
        r,ar,ids=unit;xy=ar['contact_xy_mm'];representatives={}
        for m in [0,1]:
            ix=ids[ar['event_mode'][ids]==m]
            if len(ix):
                phi=ar['event_phi'][ix];representatives[m]=int(ix[np.argmin(np.sum((phi-phi.mean(0))**2,axis=1))])
        fig,axs=plt.subplots(1,4,figsize=(17,3.7),gridspec_kw={'width_ratios':[1,1,1,2]},layout='constrained')
        for ax in axs[:3]:
            ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
            for ci,rad in zip(c['centers_mm'],c['radii_mm']):ax.add_patch(Circle(ci,rad,fill=False,color='#d34a42',lw=1))
        ix=np.arange(0,len(ar['h']),4);axs[0].scatter(*ar['positions_E'][ix].T,c=ar['h'][ix],s=2,cmap='plasma',vmin=0,vmax=1)
        axs[0].scatter(*xy.T,s=12,facecolor='white',edgecolor='black');axs[0].set_title('Threshold modulation support')
        timing_max=max([float(np.ptp(ar['centroid_ms'][i][np.isfinite(ar['centroid_ms'][i])])) for i in representatives.values()]+[1.])
        timing_map=None
        for m,ax in zip(MODE_ORDER,axs[1:3]):
            i=representatives.get(m)
            ax.scatter(*xy.T,s=25,facecolors='none',edgecolors='gray')
            if i is not None:
                times=ar['centroid_ms'][i];ok=np.isfinite(times);rel=times[ok]-min(times[ok])
                timing_map=ax.scatter(*xy[ok].T,c=rel,cmap='viridis',s=35,vmin=0,vmax=timing_max)
                ax.set_title(f'{MODE_NAMES[m]}: event {i}; centroid time')
            else:ax.set_title(f'{MODE_NAMES[m]}: no observed event')
        dt=float(ar['contact_envelope_dt_ms']);env=ar['contact_envelope'];vmax=max(1,np.percentile(env[env>0],99)) if np.any(env>0) else 1
        if timing_map is not None:fig.colorbar(timing_map,ax=list(axs[1:3]),shrink=.65,label='Relative centroid time (ms)')
        axs[3].set_facecolor('black');axs[3].imshow(env[:,display_order].T,aspect='auto',cmap='inferno',vmin=0,vmax=vmax,extent=[0,len(env)*dt/1000,len(names),0],interpolation='nearest')
        axs[3].set(yticks=np.arange(len(names))+.5,yticklabels=names[display_order],xlabel='Time (s)',title='Full contact firing-density envelope')
        save(fig,c['id']+'_same_network')
        # Multiple chronological examples per label, no patient-similarity selection.
        picks=[]
        for m in [0,1]:picks.extend(ids[ar['event_mode'][ids]==m][:3].tolist())
        picks=sorted(set(picks),key=lambda i:r['events'][i]['event_time_ms'])
        frames=[];movie=ar['sheet_activity_counts'];windows=[]
        for i in picks:
            lo,hi=r['events'][i]['window_ms'];a=max(0,round(lo/2));b=min(len(movie),round(hi/2));windows.append((i,a,b))
        vmax=max([float(movie[a:b].max()) for _,a,b in windows if b>a]+[1.])
        for i,a,b in windows:
            lo,hi=r['events'][i]['window_ms'];envwin=env[int(lo/dt):int(hi/dt)]
            for frame in range(a,b,2):
                fig,axes=plt.subplots(1,2,figsize=(8,3.7),layout='constrained')
                axes[0].imshow(movie[frame],origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
                axes[0].scatter(*xy.T,s=10,facecolors='none',edgecolors='cyan')
                for ci,rad in zip(c['centers_mm'],c['radii_mm']):axes[0].add_patch(Circle(ci,rad,fill=False,color='white',lw=.8))
                axes[0].set(xlabel='x (mm)',ylabel='y (mm)',title='Native 2 ms counts; fixed movie scale',xlim=(0,20),ylim=(0,20))
                axes[1].imshow(envwin[:,display_order].T,aspect='auto',cmap='inferno',interpolation='nearest',extent=[lo,hi,len(names),0])
                axes[1].axvline(frame*2,color='cyan',lw=1);axes[1].set(xlabel='Absolute time (ms)',yticks=np.arange(len(names))+.5,yticklabels=names[display_order],title='Contact firing-density envelope')
                fig.suptitle(f'{c["id"]}, noise {plan["training_seeds"][0]}, {MODE_NAMES[int(ar["event_mode"][i])]}, event {i}')
                fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE));plt.close(fig)
        if frames:
            frames[0].save(F/(c['id']+'_multiple_events.gif'),save_all=True,append_images=frames[1:],duration=75,loop=0)
            for frame in frames:frame.close()
        rt.write(OUT/(c['id']+'_visual_selection.json'),dict(representatives=representatives,gif_events=picks,
            rule='same first noise; medoid relative to model own class mean; GIF first three eligible events of each label in chronological order; no patient-distance selection'))
    additional_seeds=[s for s in observation_seeds if s not in plan['training_seeds']]
    notes=['# Core 范围 pilot：完成的观测与科学审阅点','',
        f'各条件固定同一张拓扑。配对评估噪声为 {plan["training_seeds"]}，按运行等权评分；额外展示噪声为 {additional_seeds}。额外历史运行不自动成为新候选的独立确认。逐事件表不作为独立网络样本作显著性推断。',
        '模型质心时间并非起燃时间；黑底图是实际接触发放密度包络，不是 HFO 频谱。完整多事件图与原生动画共同检查传播，分类标签不能证明两种患者机制恢复。','',
        '| 条件 | 配对噪声合格事件数 | 原冻结损失 |','|---|---|---|']
    for s in score_rows:
        ns='/'.join(str(u.get('n_events','缺失')) for u in s['units']);loss=s['loss_off']
        notes.append(f'|{s["candidate"]["id"]}|{ns}|{loss:.4f}|' if loss is not None else f'|{s["candidate"]["id"]}|{ns}|不可评分|')
    notes+=['','提名：'+(chosen['id'] if nomination else '尚无完整提名；当前为阶段输出'),
            '重点检查：SCL 缺杆是否下降，同时保持 ICL 参与及两类相对顺序；原生场是否新增并行热点或持续高活动。半径增加会增加阈值调制总量，剂量对照需单独解释。',
            '图件已自动生成，尚待 Agent 和用户目视科学审阅，不能从最低损失自动宣布成功。']
    (OUT/'scientific_report.md').write_text('\n'.join(notes)+'\n')
    desc=[]
    for path in sorted(F.glob('*.png')):
        name=path.stem
        if name=='parameter_observation_response':meaning=f'各条件下 {len(plan["training_seeds"])} 条配对评估噪声的损失、参与误差、平均顺序和 SCL 缺杆比例。颜色区分噪声，编号见图例，网络拓扑相同；缺值保留为缺值。'
        elif name=='all_event_rank_and_participation':meaning=f'患者各模式按事件索引等间隔展示最多 100 例，模型保留 {len(plan["training_seeds"])} 条配对评估轨迹的全部合格事件。紫色早、黄色晚、灰色不参与；每行接触点顺序完全一致。'
        else:meaning='按机制、两类代表事件与连续读出布局，保留 SEEG 几何。代表事件取距该运行自身类均值最近者；时序颜色为活动质心，不是原生起燃位置。'
        desc.append(f'### {path.name}\n{meaning} **关注点**：参与、时序与原生过程是否同时改善，不能只看分数或标签。\n')
    for path in sorted(F.glob('*.gif')):desc.append(f'### {path.name}\n首条噪声中每类最先出现的最多三个合格事件，按实际时间顺序播放，事件间为剪辑跳转。左侧为未平滑原生计数，右侧为接触活动包络；这不是连续全程动画或真实 HFO 频谱。 **关注点**：是否发生有序招募，是否只是多处同时变亮。\n')
    (F/'README.md').write_text('\n'.join(desc))
    # Decode products before reporting analysis completion.
    checks=[]
    for path in list(F.glob('*.png'))+list(F.glob('*.gif')):
        with Image.open(path) as im:
            for f in range(getattr(im,'n_frames',1)):im.seek(f);im.load()
            checks.append(dict(path=str(path),frames=getattr(im,'n_frames',1)))
    rt.write(OUT/'figure_decode_checks.json',checks)


if __name__=='__main__':main()

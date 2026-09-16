"""Full-output review of the bounded core-input pilot, including empty runs."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label as component_labels
from scipy.stats import spearmanr
from PIL import Image,ImageDraw
from scripts import run_topic4_core_driven_input_pilot as run
from scripts.analyze_topic4_geometry_threshold_refinement import patient_examples
from scripts.plot_topic4_all_condition_time_review import render
rt=run.rt
OUT=run.OUT/'analysis';F=OUT/'figures'


def avg(x):
    n=np.isfinite(x).sum(0)
    return np.divide(np.nansum(x,0),n,out=np.full(x.shape[1],np.nan),where=n>0)
def ranks(x):
    from src.topic4_d6_natural_kmeans import normalize_event_ranks
    from scipy.stats import rankdata
    out=np.full_like(x,np.nan)
    for i,row in enumerate(x):
        ok=np.isfinite(row);out[i,ok]=rankdata(row[ok])
    return normalize_event_ranks(out)
def writecsv(name,rows):
    if rows:
        with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);F.mkdir(exist_ok=True)
    plan=rt.read(run.OUT/'plan.json');cs=plan['candidates'];seeds=plan['seeds'];units={};stats=[];counts=[];native=[];timing=[];selections=[]
    evaluator=rt.load_evaluator(plan['parent_design']);patient=np.asarray(evaluator.fit);pl=np.asarray(evaluator.fit_labels)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    for c in cs:
        for seed in seeds:
            path=run.result_path(c['id'],seed);r=rt.read(path)
            with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
            assert rt.sha(path.with_suffix('.npz'))==r['arrays_sha256']
            names=a['contact_names'].astype(str);scl=np.char.startswith(names,'SCL')
            a['contact_envelope']=a['contact_envelope'].T
            ids=np.asarray([i for i in a['primary_event_indices'] if r['events'][i]['window_ms'][0]>=plan['analysis_burnin_ms'] and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
            units[(c['id'],seed)]=(r,a,ids)
            counts.append(dict(candidate=c['id'],label=c['label'],seed=seed,physical_status=r['physical_status'],duration_ms=r['actual_duration_ms'],detected=r['n_detected'],primary=r['n_primary'],analysis_n=len(ids)))
            for m in [1,0]:
                idx=ids[a['event_mode'][ids]==m];x=a['centroid_ms'][idx];p=patient[pl==m]
                mr=avg(ranks(x));pr=avg(ranks(p));ok=np.isfinite(mr)&np.isfinite(pr)
                stats.append(dict(candidate=c['id'],label=c['label'],seed=seed,mode='TA' if m==1 else 'TB',n=len(idx),
                    participation_mae=float(abs(np.isfinite(x).mean(0)-np.isfinite(p).mean(0)).mean()) if len(idx) else None,
                    rank_correlation=float(spearmanr(mr[ok],pr[ok]).statistic) if ok.sum()>=3 else None,
                    no_scl=float((~np.isfinite(x)[:,scl].any(1)).mean()) if len(idx) else None,
                    fraction=len(idx)/len(ids) if len(ids) else None))
            tt=a['trace_time_ms'];nc=len(a['group_coreAE'])+len(a['group_coreBE']);no=len(a['group_surroundE'])
            picks=sorted([int(i) for m in [1,0] for i in ids[a['event_mode'][ids]==m][:3]],key=lambda i:r['events'][i]['window_ms'][0])
            for i in ids:
                lo,hi=r['events'][i]['window_ms'];mask=(tt>=lo)&(tt<hi)
                dt=float(a['contact_envelope_dt_ms']);env=a['contact_envelope'][round(lo/dt):round(hi/dt)]
                cumulative=np.cumsum(env,axis=0);mass=cumulative[-1]
                participated=np.isfinite(a['centroid_ms'][i])&(mass>0)
                if participated.any():
                    q=np.stack([np.argmax(cumulative>=fraction*mass,axis=0)*dt for fraction in [.1,.5,.9]])
                    timing.append(dict(candidate=c['id'],seed=seed,event=int(i),mode='TA' if a['event_mode'][i]==1 else 'TB',
                        n_contacts=int(participated.sum()),local_width_ms=float(np.median((q[2]-q[0])[participated])),
                        recruitment_span_ms=float(np.ptp(q[0,participated])),centroid_span_ms=float(np.ptp(a['centroid_ms'][i,participated]))))
                cc=a['trace_coreAE_spikes'][mask]+a['trace_coreBE_spikes'][mask];oo=a['trace_surroundE_spikes'][mask];total=cc+oo
                if not total.sum():continue
                end=np.searchsorted(np.cumsum(total),.1*total.sum())+1
                denom=float(total[:end].sum());coremass=float(cc[:end].sum());outmass=float(oo[:end].sum())
                frames=a['sheet_activity_counts'][round(lo/2):round(hi/2)].astype(float)
                peak=frames[frames.sum((1,2)).argmax()];active=peak>=2
                labels,num=component_labels(active,structure=np.ones((3,3)))
                mass=np.bincount(labels.ravel(),weights=(peak*active).ravel())
                native.append(dict(candidate=c['id'],seed=seed,event=int(i),mode='TA' if a['event_mode'][i]==1 else 'TB',
                    first_10pct_mass_core_share=coremass/denom,first_10pct_core_to_outside_per_neuron_density=(coremass/nc)/(outmass/no) if outmass>0 else None,
                    peak_active_components=int(num),peak_largest_component_fraction=float(mass[1:].max(initial=0)/mass[1:].sum()) if mass[1:].sum() else None))
            # Continuous raster preserves every episode, even when no event passed the observer.
            fig,ax=plt.subplots(figsize=(12,4),layout='constrained');order=[list(names).index(n) for n in [f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]]
            env=a['contact_envelope'][:,order].T
            im=ax.imshow(env/max(env.max(),1e-20),aspect='auto',cmap='magma',extent=[0,len(env[0])*2/1000,14.5,-.5],vmin=0,vmax=1)
            ax.axhline(3.5,c='cyan',lw=.6);ax.set(yticks=range(15),yticklabels=names[order],xlabel='时间 (s)',title=f'{c["label"]} | seed {seed} | 完整读出；合格事件 {len(ids)} | {r["physical_status"]}')
            fig.colorbar(im,ax=ax,label='整段共同归一化包络');fig.savefig(F/f'{c["id"]}_{seed}_continuous.png',dpi=140);plt.close(fig)
            if seed==seeds[0] and picks:
                # First 3 per mode, chronological. Native 2ms counts, sampled every 4ms in GIF.
                vmax=max(float(a['sheet_activity_counts'][round(r['events'][i]['window_ms'][0]/2):round(r['events'][i]['window_ms'][1]/2)].max()) for i in picks)
                vmax=max(vmax,1);movieframes=[];cmap=plt.get_cmap('inferno')
                fig,axes=plt.subplots(len(picks),6,figsize=(12,2*len(picks)),squeeze=False,layout='constrained')
                for row,i in enumerate(picks):
                    lo,hi=r['events'][i]['window_ms'];mode='TA' if a['event_mode'][i]==1 else 'TB'
                    selections.append(dict(candidate=c['id'],seed=seed,event=i,mode=mode,window_ms=[lo,hi],rule='first three eligible events per mode, chronological'))
                    for col,offset in enumerate([0,40,80,120,160,200]):
                        ax=axes[row,col];frame=a['sheet_activity_counts'][round((lo+offset)/2)]
                        im=ax.imshow(frame,origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
                        ax.scatter(*a['contact_xy_mm'].T,s=7,facecolors='none',edgecolors='cyan',linewidths=.5)
                        for xy,rad in zip(c['centers_mm'],c['radii_mm']):ax.add_patch(Circle(xy,rad,fill=False,color='white',lw=.6))
                        ax.set(xticks=[],yticks=[],title=f'{offset} ms' if row==0 else None)
                        if col==0:ax.set_ylabel(f'{mode} · {i}')
                    for offset in range(0,250,4):
                        fr=a['sheet_activity_counts'][round((lo+offset)/2)]
                        rgb=(cmap(np.clip(fr[::-1]/vmax,0,1))[:,:,:3]*255).astype('uint8')
                        canvas=Image.new('RGB',(450,500),'white');nativeim=Image.fromarray(rgb).resize((400,400),Image.Resampling.NEAREST);canvas.paste(nativeim,(25,65));draw=ImageDraw.Draw(canvas)
                        draw.text((15,10),f'{c["id"]} | seed {seed}',fill='black');draw.text((15,30),f'{mode} event {i} | t={lo+offset:.0f} ms | frame={offset} ms',fill='black')
                        for xy,rad in zip(c['centers_mm'],c['radii_mm']):
                            x,y=25+xy[0]*20,65+(20-xy[1])*20;rr=rad*20;draw.ellipse((x-rr,y-rr,x+rr,y+rr),outline='white',width=1)
                        for x,y in a['contact_xy_mm']:
                            px,py=25+x*20,65+(20-y)*20;draw.ellipse((px-2,py-2,px+2,py+2),outline='cyan',width=1)
                        draw.text((15,478),'Native 2 ms activity; shared scale within this run',fill='black');movieframes.append(canvas)
                fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='原生2ms活动神经元数 / 1mm网格')
                fig.suptitle(c['label']+'：每类最早三个合格事件，无逐帧选优');fig.savefig(F/f'{c["id"]}_native_stills.png',dpi=140);plt.close(fig)
                movieframes[0].save(F/f'{c["id"]}_native_multievent.gif',save_all=True,append_images=movieframes[1:],duration=65,loop=0)
    writecsv('per_run_counts.csv',counts);writecsv('per_mode_observations.csv',stats);writecsv('per_event_native_diagnostics.csv',native);writecsv('per_event_timing.csv',timing)
    rt.write(OUT/'native_selection.json',selections)
    render(units,cs,seeds,patient_examples(),OUT,lambda c:c['label'])
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained');xx=np.arange(len(cs))
    for ri,m in enumerate(['TA','TB']):
        for ax,k,title in zip(axes[ri],['participation_mae','rank_correlation','no_scl'],['参与误差 ↓','平均顺序相关 ↑','整杆SCL缺失 ↓']):
            for seed in seeds:
                vals=[next(x[k] for x in stats if x['candidate']==c['id'] and x['seed']==seed and x['mode']==m) for c in cs]
                ax.plot(xx,[np.nan if x is None else x for x in vals],'-o',label=f'噪声 {seed}',ms=4)
            ax.set(xticks=xx,xticklabels=['混合全场','全降全场','局部限核','全部限核','核OU无局部','再关I状态'],title=m+' · '+title);ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
    fig.suptitle('固定几何和连接下的输入机制对照；缺值表示事件不足，不填零');fig.savefig(F/'parameter_observations.png',dpi=150);fig.savefig(F/'parameter_observations.pdf');plt.close(fig)
    # Offline observables only: no new training penalty, eligibility rule, or physics.
    summaries=[]
    for name,source,features in [('timing',timing,[('local_width_ms','局部10–90%宽度 (ms)'),('recruitment_span_ms','跨触点10%招募跨度 (ms)'),('centroid_span_ms','跨触点质心跨度 (ms)')]),
                                 ('native',native,[('first_10pct_mass_core_share','前10%活动质量中core占比'),('first_10pct_core_to_outside_per_neuron_density','早期core/核外每神经元活动密度'),('peak_active_components','峰值空间分离分量数')])]:
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        for row,mode in enumerate(['TA','TB']):
            for ax,(key,title) in zip(axes[row],features):
                for si,seed in enumerate(seeds):
                    med=[];lower=[];upper=[]
                    for ci,c in enumerate(cs):
                        values=np.asarray([r[key] for r in source if r['candidate']==c['id'] and r['seed']==seed and r['mode']==mode and r[key] is not None],float)
                        values=values[np.isfinite(values)]
                        q=np.quantile(values,[.05,.5,.95]) if len(values) else np.full(3,np.nan)
                        med.append(q[1]);lower.append(q[0]);upper.append(q[2])
                        summaries.append(dict(candidate=c['id'],seed=seed,mode=mode,observable=key,n=len(values),mean=float(values.mean()) if len(values) else None,
                            variance=float(values.var()) if len(values) else None,q05=float(q[0]) if len(values) else None,median=float(q[1]) if len(values) else None,q95=float(q[2]) if len(values) else None))
                    pos=xx+(-.035 if si==0 else .035)
                    ax.plot(pos,med,'-o',ms=3,label=f'噪声 {seed}');ax.vlines(pos,lower,upper,color=f'C{si}',alpha=.35,lw=2)
                ax.set(xticks=xx,xticklabels=['混合全场','全降全场','局部限核','全部限核','核OU无局部','再关I状态'],title=mode+' · '+title)
                ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
        fig.suptitle(('时间观测' if name=='timing' else '原生空间活动')+'：每运行中位数及事件5–95%范围；不是置信区间；无事件不填零')
        fig.savefig(F/f'parameter_{name}_distributions.png',dpi=150);fig.savefig(F/f'parameter_{name}_distributions.pdf');plt.close(fig)
    writecsv('observable_distribution_summary.csv',summaries)
    text='# Core 驱动输入 pilot：完成记录\n\n本次固定12条20秒运行，未按结果追加或选择。参与、顺序与SCL是多事件条件观测；原生图和完整包络用于判断残余断裂，分数与标签不是恢复结论。\n\n'
    text+='原生诊断的 first_10pct_mass_core_share 是窗口累计活动前10%中的core占比，不是因果起源；密度比校正内外神经元数量。峰值连通分量以每1mm格至少2个活跃神经元、八邻接定义，只是并行空间分离诊断，不等于独立因果root。两种mode均来自冻结患者分类器，不使用命名路径训练。\n\n'
    text+='|条件|噪声|TA数|TB数|物理状态|\n|---|---:|---:|---:|---|\n'
    for x in counts:
        mm=[next(y['n'] for y in stats if y['candidate']==x['candidate'] and y['seed']==x['seed'] and y['mode']==m) for m in ['TA','TB']]
        text+=f'|{x["label"]}|{x["seed"]}|{mm[0]}|{mm[1]}|{x["physical_status"]}|\n'
    text+='\n时间分布沿用前批全接触包络定义：每事件参与接触点的10–90%质量宽度中位数、跨触点10%招募跨度和质心跨度分别计算，不以平均rank替代。参数×时间及参数×原生空间图给逐运行中位数和事件5–95%范围，后者是事件散布，不是置信区间。只有一个参与触点时跨度为0是定义结果，应结合n_contacts解释。\n'
    text+='\n所有模式为零或很少时，完整读出仍交付，不能称为噪声消除后传播恢复。20秒可能不足以估计少数模式，需先审阅原始轨迹再决定是否延长。\n\n本报告为自动观测汇总，尚待Agent与用户目视科学审阅；不自动接受机制或进入下一搜索。\n'
    (OUT/'scientific_review.md').write_text(text)
    entries=[]
    for f in sorted(F.iterdir()):
        if f.suffix not in ['.png','.gif']:continue
        with Image.open(f) as im:
            for frame in range(getattr(im,'n_frames',1)):im.seek(frame);im.load()
        entries.append(f'### {f.name}\n\n固定条件的完整接触读出、原生过程或配对观测；输入掩码只使用core几何。原生动画使用每类最早三个合格事件，未按患者相似性选优；无事件条件仅显示完整轨迹。\n\n**关注点**：core外并行活动、SCL/ICL时序与TA/TB取舍；图件可读不代表传播恢复。')
    (F/'README.md').write_text('\n\n'.join(entries)+'\n')
    rt.write(OUT/'delivery_checks.json',dict(status='GENERATED_DECODE_CHECKED_PENDING_VISUAL_REVIEW',runs=len(counts),mode_rows=len(stats),native_events=len(native),user_accepted=False))


if __name__=='__main__':main()

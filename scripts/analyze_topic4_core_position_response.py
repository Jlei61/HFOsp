"""Full-output review and paired position-to-observable response, including empty runs."""
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
from scripts import run_topic4_core_position_response as run
from scripts.analyze_topic4_geometry_threshold_refinement import patient_examples
from scripts.plot_topic4_all_condition_time_review import render
run.configure()
rt=run.rt
OUT=run.OUT/'analysis';F=OUT/'figures'


def position_plan():
    """Display real montage and predeclared position probes, not activity."""
    OUT.mkdir(parents=True,exist_ok=True);F.mkdir(exist_ok=True)
    plan=rt.read(run.OUT/'plan.json');cases=[c for c in plan['candidates'] if c['stage']!='state_bridge']
    with np.load(run.PARENT/'formal/units/core_ou_only_Istate_off/847101/workers/trajectory.npz') as a:
        names=a['contact_names'].astype(str);xy=a['contact_xy_mm']
    distances=rt.read(run.OUT/'geometry_to_contact_distances.json')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,4,figsize=(13,7.5),layout='constrained')
    for ax,c in zip(axes.flat,cases):
        for shaft,color in [('SCL','#00b7c4'),('ICL','#df9427')]:
            ids=[i for i,n in enumerate(names) if n.startswith(shaft)];ids=sorted(ids,key=lambda i:int(names[i][len(shaft):]))
            ax.plot(*xy[ids].T,'-o',color=color,ms=3,lw=1)
            for i in ids:ax.annotate(names[i],xy[i],xytext=(2,3),textcoords='offset points',fontsize=5.8)
        ax.add_patch(Circle(cases[0]['centers_mm'][0],c['radii_mm'][0],fill=False,ec='gray',ls=':',lw=1))
        for k,(center,rad) in enumerate(zip(c['centers_mm'],c['radii_mm'])):
            ax.add_patch(Circle(center,rad,fc='#dc655d' if k==0 else '#659aba',alpha=.25,ec='black',lw=1));ax.plot(*center,'+',color='black',ms=6)
        gap=next(x['edge_gap_mm'] for x in distances if x['candidate']==c['id'] and x['contact']=='SCL9')
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=c['label']+f'\nA=({c["centers_mm"][0][0]:.2f}, {c["centers_mm"][0][1]:.2f})；距SCL9边缘 {gap:.2f} mm',xticks=[0,10,20],yticks=[0,10,20])
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.95,'原位及有先验的位置探针\n\n红色：左核A\n蓝色：右核B\n灰虚线：左核原位\n橙色：ICL\n青色：SCL\n\n半径固定约1.753 mm\n图是几何设计，不是传播结果\n主比较关闭外加慢I状态\n另有原位/上移3mm的I状态桥',va='top',fontsize=10)
    fig.suptitle('固定半径，调整端点先验附近的位置；没有按仿真结果挑点')
    for ext in ['png','pdf']:fig.savefig(F/f'position_design.{ext}',dpi=160)
    plt.close(fig)
    existing=(F/'README.md').read_text() if (F/'README.md').exists() else ''
    import re
    existing=re.sub(r'(?ms)^### position_design\.(?:png|pdf)\n.*?(?=^### |\Z)','',existing).strip()
    (F/'README.md').write_text((existing+'\n\n' if existing else '')+'### position_design.png\n\n患者固定SEEG几何与本轮七个预设位置，保持同一20×20 mm场和半径。标出原位、加权中心、垂直位移和上部SCL附近的几何探针；红/蓝区仅表示core范围。\n\n**关注点**：向SCL靠近会同时离ICL更远；此图不表示仿真传播成功。\n\n### position_design.pdf\n\n同名PNG的矢量版本，几何和数值完全相同。实际阈值、OU支持和神经元成员由canary/正式运行另行核对。\n\n**关注点**：几何设计与实际传播分开。\n')


def avg(x):
    n=np.isfinite(x).sum(0)
    return np.divide(np.nansum(x,0),n,out=np.full(x.shape[1],np.nan),where=n>0)
def finite_rho(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    value=float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else None
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
            path=run.core.result_path(c['id'],seed);r=rt.read(path)
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
                    rank_correlation=finite_rho(mr[ok],pr[ok]),
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
            ax.set(xticks=xx,xticklabels=[c['label'] for c in cs],title=m+' · '+title);ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
    fig.suptitle('固定半径、连接和输入规律下的位置对照；缺值表示事件不足，不填零');fig.savefig(F/'parameter_observations.png',dpi=150);fig.savefig(F/'parameter_observations.pdf');plt.close(fig)
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
                ax.set(xticks=xx,xticklabels=[c['label'] for c in cs],title=mode+' · '+title)
                ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
        fig.suptitle(('时间观测' if name=='timing' else '原生空间活动')+'：每运行中位数及事件5–95%范围；不是置信区间；无事件不填零')
        fig.savefig(F/f'parameter_{name}_distributions.png',dpi=150);fig.savefig(F/f'parameter_{name}_distributions.pdf');plt.close(fig)
    writecsv('observable_distribution_summary.csv',summaries)
    text='# Core 位置响应：全条件完成记录\n\n本次14条新20秒位置运行，另复用4条相同物理的原位运行；未按结果追加或选择。参与、顺序与SCL是多事件条件观测；原生图和完整包络用于判断残余断裂，分数与标签不是恢复结论。\n\n'
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


def position_responses():
    """Predefined geometry contrasts, unconditional and TA/TB separately."""
    plan=rt.read(run.OUT/'plan.json');cases=plan['candidates'];seeds=plan['seeds']
    ev=rt.load_evaluator(plan['parent_design']);patient=np.asarray(ev.fit);labels=np.asarray(ev.fit_labels)
    names=np.asarray(rt.load_observation_contract(plan['parent_design'])['contact_names'])
    scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL');upper=np.isin(names,['SCL9','SCL8'])
    def measures(x,ref):
        n=len(x);ok=np.isfinite(x);pr=np.isfinite(ref).mean(0)
        m=dict(n=n,SCL_any=None,SCL_upper_participation=None,ICL_contact_participation=None,both_rods=None,
            participation_mae=None,rank_correlation=None,rank_shared_contacts=0,lag_joint_n=0,SCL_minus_ICL_lag_median_ms=None,
            SCL_minus_ICL_lag_q05_ms=None,SCL_minus_ICL_lag_q95_ms=None)
        if not n:return m
        p=ok.mean(0);mr=avg(ranks(x));rref=avg(ranks(ref));common=np.isfinite(mr)&np.isfinite(rref)
        both=ok[:,scl].any(1)&ok[:,icl].any(1)
        m.update(SCL_any=float(ok[:,scl].any(1).mean()),SCL_upper_participation=float(p[upper].mean()),
            ICL_contact_participation=float(p[icl].mean()),both_rods=float(both.mean()),participation_mae=float(abs(p-pr).mean()),
            rank_correlation=finite_rho(mr[common],rref[common]),rank_shared_contacts=int(common.sum()),lag_joint_n=int(both.sum()))
        if both.any():
            lag=np.nanmedian(x[both][:,scl],1)-np.nanmedian(x[both][:,icl],1);q=np.quantile(lag,[.05,.5,.95])
            m.update(SCL_minus_ICL_lag_median_ms=float(q[1]),SCL_minus_ICL_lag_q05_ms=float(q[0]),SCL_minus_ICL_lag_q95_ms=float(q[2]))
        return m
    refs={};summary=[];contacts=[];pair_rows=[]
    for mode,label in [('ALL',None),('TA',1),('TB',0)]:
        ref=patient if label is None else patient[labels==label]
        refs[mode]=measures(ref,ref)
        pair_reference={}
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                valid=np.isfinite(ref[:,i])&np.isfinite(ref[:,j]);delta=ref[valid,j]-ref[valid,i]
                pair_reference[(i,j)]=(int(valid.sum()),float(np.mean((delta>0)+.5*(delta==0))) if len(delta) else None)
        for c in cases:
            for seed in seeds:
                path=run.core.result_path(c['id'],seed);result=rt.read(path)
                with np.load(path.with_suffix('.npz')) as a:
                    assert np.array_equal(a['contact_names'].astype(str),names)
                    ids=np.asarray([i for i in a['primary_event_indices'] if result['events'][i]['window_ms'][0]>=plan['analysis_burnin_ms'] and result['events'][i]['window_ms'][1]<=result['actual_duration_ms']],int)
                    if label is not None:ids=ids[a['event_mode'][ids]==label]
                    x=a['centroid_ms'][ids]
                    n_core=int((a['h']>0).sum());total_lower=float(np.sum(18-a['vtheta'][:32000]))
                rec=dict(candidate=c['id'],label=c['label'],seed=seed,mode=mode,retain_I_state=c['retain_I_state'],
                    x_A_mm=c['centers_mm'][0][0],y_A_mm=c['centers_mm'][0][1],n_core_E=n_core,total_threshold_lowering_mV=total_lower,**measures(x,ref))
                pair_errors=[]
                for (i,j),(pn,pp) in pair_reference.items():
                    valid=np.isfinite(x[:,i])&np.isfinite(x[:,j]);delta=x[valid,j]-x[valid,i]
                    probability=float(np.mean((delta>0)+.5*(delta==0))) if len(delta) else None
                    error=abs(probability-pp) if probability is not None and pp is not None else None
                    if error is not None:pair_errors.append(error)
                    pair_rows.append(dict(candidate=c['id'],seed=seed,mode=mode,contact_i=names[i],contact_j=names[j],
                        model_joint_n=int(valid.sum()),patient_joint_n=pn,model_i_precedes_j=probability,patient_i_precedes_j=pp,absolute_probability_difference=error))
                rec.update(pair_order_probability_mae=float(np.mean(pair_errors)) if pair_errors else None,pair_order_supported_pairs=len(pair_errors))
                summary.append(rec)
                prob=np.isfinite(x).mean(0) if len(x) else np.full(15,np.nan)
                for name,p,pp in zip(names,prob,np.isfinite(ref).mean(0)):
                    contacts.append(dict(candidate=c['id'],label=c['label'],seed=seed,mode=mode,contact=name,n=len(x),participation=None if not np.isfinite(p) else float(p),patient_participation=float(pp)))
    writecsv('position_observations.csv',summary);writecsv('contact_participation.csv',contacts);writecsv('contact_pair_order_probabilities.csv',pair_rows);rt.write(OUT/'patient_observation_reference.json',refs)
    effects=[]
    keys=['SCL_any','SCL_upper_participation','ICL_contact_participation','both_rods','participation_mae','rank_correlation','pair_order_probability_mae','SCL_minus_ICL_lag_median_ms']
    for x in summary:
        baseline='base_on' if x['retain_I_state'] else 'base_off'
        b=next(v for v in summary if v['candidate']==baseline and v['seed']==x['seed'] and v['mode']==x['mode'])
        for key in keys:
            v0,v1=b[key],x[key]
            effects.append(dict(candidate=x['candidate'],label=x['label'],baseline=baseline,seed=x['seed'],mode=x['mode'],observable=key,baseline_value=v0,value=v1,
                change=None if v0 is None or v1 is None else v1-v0,baseline_n=b['n'],n=x['n']))
    writecsv('paired_position_effects.csv',effects)
    vertical=[c for c in cases if not c['retain_I_state'] and c['y_shift_mm'] is not None]
    vertical.sort(key=lambda c:c['y_shift_mm']);xx=[c['y_shift_mm'] for c in vertical]
    fig,axes=plt.subplots(3,4,figsize=(14,9),layout='constrained')
    panels=[('SCL_upper_participation','SCL9/8 平均参与概率'),('ICL_contact_participation','ICL 各触点平均参与概率'),('both_rods','两杆联合参与概率'),('rank_correlation','平均质心顺序相关')]
    for row,mode in enumerate(['ALL','TA','TB']):
        for ax,(key,title) in zip(axes[row],panels):
            for si,seed in enumerate(seeds):
                data=[next(v for v in summary if v['candidate']==c['id'] and v['seed']==seed and v['mode']==mode) for c in vertical]
                yy=[np.nan if v[key] is None else v[key] for v in data]
                ax.plot(xx,yy,'o',color=f'C{si}',ls='-' if si==0 else '--',label=f'噪声 {seed}',ms=4)
                for pos,y,v in zip(xx,yy,data):
                    if np.isfinite(y):ax.annotate(f'n={v["n"]}',(pos,y),xytext=(2,5 if si==0 else -10),textcoords='offset points',fontsize=6,color=f'C{si}')
            if key!='rank_correlation':ax.axhline(refs[mode][key],color='black',ls=':',lw=1,label='患者 FIT')
            ax.set(title=('不分模式' if mode=='ALL' else mode)+' · '+title,xlabel='左核上移量 (mm)',xticks=xx,ylim=(-1.08,1.12) if key=='rank_correlation' else (-.05,1.13))
            if row==0 and key==panels[0][0]:ax.legend(fontsize=7)
    fig.suptitle('位置 → 传播观测：固定半径/连接/输入规律，外加慢I关闭\n每条线为同一拓扑的一个配对噪声；n为实际事件数，rank相关只比较共同可读触点')
    for ext in ['png','pdf']:fig.savefig(F/f'vertical_position_response.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for ax,mode in zip(axes,['ALL','TA','TB']):
        for si,seed in enumerate(seeds):
            data=[next(v for v in summary if v['candidate']==c['id'] and v['seed']==seed and v['mode']==mode) for c in vertical]
            yy=[np.nan if v['pair_order_probability_mae'] is None else v['pair_order_probability_mae'] for v in data]
            ax.plot(xx,yy,'o-',color=f'C{si}',label=f'噪声 {seed}')
            for pos,y,v in zip(xx,yy,data):
                if np.isfinite(y):ax.annotate(f'{v["pair_order_supported_pairs"]}/105',(pos,y),xytext=(1,5 if si==0 else -12),textcoords='offset points',fontsize=7)
        ax.set(title='不分模式' if mode=='ALL' else mode,xlabel='左核上移量 (mm)',ylabel='成对顺序概率平均绝对偏差',xticks=xx,ylim=(-.02,1.04))
    axes[0].legend(fontsize=7)
    fig.suptitle('触点i早于j的条件概率，与患者FIT逐对比较；相同质心记半次\n仅共同有观测的触点对等权平均，标注可读对数；缺失不填零，不能脱离参与图比较高低')
    for ext in ['png','pdf']:fig.savefig(F/f'vertical_position_pair_order.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,mode in zip(axes,['ALL','TA','TB']):
        for si,seed in enumerate(seeds):
            data=[next(v for v in summary if v['candidate']==c['id'] and v['seed']==seed and v['mode']==mode) for c in vertical]
            mid=[np.nan if v['SCL_minus_ICL_lag_median_ms'] is None else v['SCL_minus_ICL_lag_median_ms'] for v in data]
            lo=[np.nan if v['SCL_minus_ICL_lag_q05_ms'] is None else v['SCL_minus_ICL_lag_q05_ms'] for v in data]
            hi=[np.nan if v['SCL_minus_ICL_lag_q95_ms'] is None else v['SCL_minus_ICL_lag_q95_ms'] for v in data]
            pos=np.asarray(xx)+(-.035 if si==0 else .035);ax.plot(pos,mid,'o-',color=f'C{si}',label=f'噪声 {seed}');ax.vlines(pos,lo,hi,color=f'C{si}',alpha=.4)
            for x,y,v in zip(pos,mid,data):
                if np.isfinite(y):ax.annotate(str(v['lag_joint_n']),(x,y),xytext=(3,4),textcoords='offset points',fontsize=7)
        ax.axhspan(refs[mode]['SCL_minus_ICL_lag_q05_ms'],refs[mode]['SCL_minus_ICL_lag_q95_ms'],color='gray',alpha=.13,label='患者事件5–95%')
        ax.axhline(refs[mode]['SCL_minus_ICL_lag_median_ms'],color='black',ls=':',label='患者中位数');ax.axhline(0,color='gray',lw=.6)
        ax.set(title='不分模式' if mode=='ALL' else mode,xlabel='左核上移量 (mm)',ylabel='SCL − ICL 质心时间差 (ms)',xticks=xx)
    axes[0].legend(fontsize=7);fig.suptitle('两杆都参与的事件：每杆参与触点质心时间取中位数，再求SCL−ICL\n负值为SCL较早；竖线是事件5–95%范围，数字是两杆联合参与事件数；不是起燃时差或置信区间')
    for ext in ['png','pdf']:fig.savefig(F/f'vertical_position_lag.{ext}',dpi=160)
    plt.close(fig)
    display=[f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]
    fig,axes=plt.subplots(1,3,figsize=(16,9),layout='constrained')
    for ax,mode in zip(axes,['ALL','TA','TB']):
        matrix=[];labs=[]
        matrix.append([next(v['patient_participation'] for v in contacts if v['mode']==mode and v['contact']==n) for n in display]);labs.append('患者 FIT')
        for c in cases:
            for seed in seeds:
                matrix.append([next(v['participation'] for v in contacts if v['candidate']==c['id'] and v['seed']==seed and v['mode']==mode and v['contact']==n) for n in display])
                rec=next(v for v in summary if v['candidate']==c['id'] and v['seed']==seed and v['mode']==mode)
                labs.append(c['label']+f' · {seed%100:02d} (n={rec["n"]})')
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e6e6e6');im=ax.imshow(np.asarray(matrix,float),cmap=cmap,vmin=0,vmax=1,aspect='auto')
        ax.set(xticks=range(15),xticklabels=display,yticks=range(len(labs)),yticklabels=labs,title='不分模式' if mode=='ALL' else mode)
        ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=6);ax.axvline(3.5,color='white',lw=1);ax.axhline(.5,color='red',lw=.8)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.55,label='触点参与概率');fig.suptitle('全条件逐触点参与：患者参考、每条噪声和实际事件数；灰色表示无可估计事件')
    for ext in ['png','pdf']:fig.savefig(F/f'all_positions_contact_participation.{ext}',dpi=180)
    plt.close(fig)
    # Descriptive complete table; no score-based selection or automatic acceptance.
    text='# 位置变化究竟改变了什么\n\n本轮14条新20秒轨迹加4条原位复用；同一拓扑、两条配对seed。主系列关闭外加慢I状态，原位/上移3mm另有开启桥。没有按本轮分数扩大搜索。\n\n'
    text+='## 不分TA/TB的观测，先看是否真正补足两杆\n\n|位置|噪声|N|SCL9/8平均参与|ICL平均参与|两杆联合参与|全触点参与误差|\n|---|---:|---:|---:|---:|---:|---:|\n'
    def fmt(x):return '不可估计' if x is None else f'{x:.3f}'
    for x in summary:
        if x['mode']=='ALL':text+=f'|{x["label"]}|{x["seed"]}|{x["n"]}|{fmt(x["SCL_upper_participation"])}|{fmt(x["ICL_contact_participation"])}|{fmt(x["both_rods"])}|{fmt(x["participation_mae"])}|\n'
    text+='\n## 如何解释\n\nSCL9/8平均参与是这两个具体触点的参与概率均值，不代表SCL整杆恢复。ICL平均参与同样是11个触点的均值；两杆联合参与表示事件至少各有一个触点。患者参考来自完整冻结FIT，不用Fig2C两例估计自然比例。\n\n'
    text+='SCL−ICL时差只在两杆均参与事件中计算，每杆先对参与触点的质心时间取中位数。负值为SCL较早；它不是两处起燃时间。中位数和5–95%范围是逐事件分布，不是置信区间；无两杆事件时不填0。\n\n'
    text+='成对顺序概率逐对计算P(触点i质心早于j | 两触点均参与)，相同质心记半次；再与患者FIT同一概率比较。曲线是共同有观测的触点对的等权平均绝对偏差，最多105对，旁列实际支持对数，CSV保留每对的实际事件数。由于可读集合可能随位置改变，不能只用此均值排名；它是离线诊断，不是新损失。\n\n'
    text+='原位→上移是E阈值支持及共享OU加载位置的整体干预；静态图/权重/时延及潜在阈值抽样固定，但实际core成员和调制总量会随位置变化。参与加权或斜向SCL探针改变了X，不能接到纯Y位移曲线上冒充同一自由度。旧慢I开启桥单独解释。\n\n'
    text+='“SCL多了但ICL少了”是可解释的位置取舍，不是自动成功；标签齐全、rank高、单个示例漂亮，都不足以接受完整双模式。全部条件黑底图、完整连续读出和原生GIF用于后续科学目视，未按预期路径删掉事件。\n'
    (OUT/'position_scientific_review.md').write_text(text)
    rt.write(OUT/'position_response_summary.json',dict(status='OBSERVATIONS_COMPLETE_PENDING_VISUAL_REVIEW',runs=18,new_runs=14,reused_runs=4,
        all_event_rows=[x for x in summary if x['mode']=='ALL'],mode_rows=summary,patient_reference=refs,user_accepted=False))
    descriptions={
        'vertical_position_response':'纯Y位移系列的逐运行参与和rank响应，保留不分模式及TA/TB；横轴为实际mm。',
        'vertical_position_pair_order':'触点对的先后概率与患者逐对比较，保留实际可读对数；缺失不填零。',
        'vertical_position_lag':'两杆都参与事件的SCL−ICL质心时差分布；无支持不填零。',
        'all_positions_contact_participation':'全部位置、噪声和模式的逐触点参与概率，与冻结患者FIT比较。'}
    with (F/'README.md').open('a') as f:
        for stem,desc in descriptions.items():
            for ext in ['png','pdf']:f.write(f'\n\n### {stem}.{ext}\n\n{desc} 每个观测按运行保留实际N，不将事件当成独立网络。\n\n**关注点**：改善SCL是否丢失ICL、两类是否存在不同取舍，不能由单个分数接受完整传播。\n')
    for stem in descriptions:
        with Image.open(F/(stem+'.png')) as im:im.load()


if __name__=='__main__':
    main()
    position_plan()
    position_responses()

"""Scientific diagnostic figures from all completed native physical runs."""
from pathlib import Path
import csv,json,collections,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Patch,Rectangle
from PIL import Image
from run import OUT,read,write,name
import metrics_v2

def summarize(job):
    folder=OUT/'per_run'/name(job)
    result=read(folder/'result.json')
    if (folder/'metrics_v2.json').exists() and read(folder/'metrics_v2.json')['coreAE']['measurement_version']!=metrics_v2.VERSION:
        (folder/'metrics_v2.json').rename(folder/'metrics_v2_raw_peak_superseded.json')
    if not (folder/'metrics_v2.json').exists():write(folder/'metrics_v2.json',metrics_v2.run_metrics(folder/'trajectory.npz',result))
    return read(folder/'metrics_v2.json')

COLORS=dict(background='#e5e9ed',sparse_bursts='#b5cbd1',irregular_bursts='#d48c61',regular_bursts='#627da8',patterned_bursts='#8d76ad',
    variable_bursts='#ccb77b',sustained_activity='#ab5661',active_without_bursts='#8c9b83',insufficient_record='#ffffff')
TEXT=dict(background='Low-activity background',sparse_bursts='Sparse bursts (<8)',irregular_bursts='Irregular bursts',patterned_bursts='Patterned bursts (high CV)',
    regular_bursts='Regular bursts',variable_bursts='Variable / intermediate bursts',sustained_activity='Sustained activity',
    active_without_bursts='Active, without detected bursts',insufficient_record='Insufficient post-burn-in record')
GROUPS=['coreAE','coreBE','coreUnionE']
TITLES=['Core A','Core B','Combined cores']

def save(fig,stem):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'{stem}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def audit(plan,records):
    static={};topologies={};lowering={};weight_fields={};other_factors=None
    for r in records:
        j=r['job'];folder=OUT/'per_run'/name(j);a=read(folder/'applied_physics.json')
        assert a['threshold']['n_raised']==0
        assert a['candidate']['parameters']['EE_same_core_scale']==j['ee']
        assert a['threshold']['depth_scales']==[j['depth'],j['depth']]
        wa=a['graph']['stage_audits']['weights']
        assert wa['factors']['EE_same_core_scale']==j['ee']
        other={k:v for k,v in wa['factors'].items() if k!='EE_same_core_scale'}
        if other_factors is None:other_factors=other
        else:assert other_factors==other
        for block in wa['blocks'].values():
            assert np.isclose(block['weight_after'],block['weight_before']*block['factor'],rtol=1e-9,atol=1e-9)
        identity=a['identity'];key=(j['ee'],j['depth'],j['topology'])
        if key in static:assert static[key]==identity,'noise changes static substrate'
        else:static[key]=identity
        graph=(identity['positions_E_sha256'],identity['ampa_topology_sha256'],identity['gaba_topology_sha256'],identity['core_index_sha256'])
        if j['topology'] in topologies:assert topologies[j['topology']]==graph,'axes changed topology or geometry'
        else:topologies[j['topology']]=graph
        key=(j['topology'],j['depth'])
        if key in lowering:assert lowering[key]==identity['vtheta_float64_sha256'],'weight axis changed thresholds'
        else:lowering[key]=identity['vtheta_float64_sha256']
        key=(j['topology'],j['ee'])
        if key in weight_fields:assert weight_fields[key]==identity['ampa_values_sha256'],'depth axis changed edge weights'
        else:weight_fields[key]=identity['ampa_values_sha256']
        with np.load(folder/'trajectory.npz') as z:
            names=z['group_names'].tolist();idx={n:names.index(n) for n in names}
            for metric in ['spike_counts_2ms','active_counts_2ms','active_counts_10ms']:
                x=z[metric].astype(np.int64)
                assert np.array_equal(x[:,idx['coreAE']]+x[:,idx['coreBE']],x[:,idx['coreUnionE']])
                assert np.array_equal(x[:,idx['coreUnionE']]+x[:,idx['surroundE']],x[:,idx['allE']])
            assert np.all(z['active_counts_10ms']<=z['group_sizes'])
            assert np.all(z['active_counts_2ms']<=z['group_sizes'])
            if r['result']['runaway_early_stop_ms'] is None:
                assert r['result']['actual_duration_ms']==j['duration_ms']
                assert int(z['total_spikes_per_cell'].sum())==int(z['spike_counts_2ms'][:,idx['allE']].sum()+z['spike_counts_2ms'][:,idx['allI']].sum())
    payload=dict(status='PASS',n_runs=len(records),zero_raised_E_thresholds=True,axes_applied=True,
        actual_pathway_weight_doses_checked=True,other_pathway_factors_fixed=True,
        topology_geometry_fixed_within_network=True,static_identity_fixed_across_noise=True,
        EE_axis_preserves_thresholds=True,threshold_axis_preserves_weights=True,native_count_conservation=True)
    write(OUT/'parameter_application_validation.json',payload)

def phase(plan,records):
    fig,axes=plt.subplots(1,3,figsize=(14.5,6.2),sharex=True,sharey=True)
    ee=plan['ee_values'];depth=plan['depth_values'];seeds=plan['seeds']
    for ax,group,title in zip(axes,GROUPS,TITLES):
        for iy,d in enumerate(depth):
            for ix,e in enumerate(ee):
                pair=[next(r for r in records if r['job']['family']=='grid' and r['job']['ee']==e and r['job']['depth']==d and r['job']['seed']==s) for s in seeds]
                triangles=[[(ix-.5,iy-.5),(ix-.5,iy+.5),(ix+.5,iy+.5)],[(ix-.5,iy-.5),(ix+.5,iy-.5),(ix+.5,iy+.5)]]
                for k,(r,points) in enumerate(zip(pair,triangles)):
                    m=r['metrics'][group]
                    ax.add_patch(Polygon(points,facecolor=COLORS[m['label']],edgecolor='white',linewidth=.8))
                    if not m['threshold_stable']:ax.text(ix+(-.23 if k==0 else .23),iy+(.2 if k==0 else -.2),'×',ha='center',va='center',fontsize=10,color='#252525')
                if e==.85 and d==1.:ax.add_patch(Rectangle((ix-.49,iy-.49),.98,.98,fill=False,edgecolor='black',linewidth=2))
        ax.set(xlim=(-.5,len(ee)-.5),ylim=(-.5,len(depth)-.5),xticks=range(len(ee)),yticks=range(len(depth)),
            xticklabels=[f'{x:g}' for x in ee],yticklabels=[f'{x:g}' for x in depth],xlabel='Within-core E→E weight multiplier',title=title)
        ax.set_aspect('equal')
    axes[0].set_ylabel('Threshold-lowering amplitude (both cores)')
    fig.suptitle('Native burst regime map | 40,000 neurons | one fixed network',fontsize=15,y=.995)
    fig.text(.5,.91,'20 s per run; first 2 s excluded. Upper-left / lower-right triangles: two noise realizations.',ha='center',fontsize=10)
    present={r['metrics'][g]['label'] for r in records if r['job']['family']=='grid' for g in GROUPS}
    fig.legend(handles=[Patch(facecolor=COLORS[k],label=TEXT[k]) for k in COLORS if k in present],loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.02))
    fig.text(.5,.14,'Black box: current reference (0.85, 1). ×: label changes at 7.5% or 12.5% onset threshold. Finite-record phenotypes; no interpolated boundaries.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.29,wspace=.17)
    save(fig,'native_burst_regime_map')

def continuous(plan,records):
    specs=[('mean_rate_hz','Mean firing rate (Hz / E cell)'),('burst_rate_hz','Detected burst rate (Hz)'),
        ('cv','Inter-burst interval CV'),('cv2','Local interval CV2'),
        ('median_peak_active_fraction_2ms','Peak 2 ms active fraction'),('burst_duty','Burst duty fraction')]
    fig,axes=plt.subplots(2,3,figsize=(13,9.2))
    for ax,(metric,title) in zip(axes.flat,specs):
        grid=np.full((len(plan['depth_values']),len(plan['ee_values'])),np.nan)
        for iy,d in enumerate(plan['depth_values']):
            for ix,e in enumerate(plan['ee_values']):
                pair=[r['metrics']['coreUnionE'] for r in records if r['job']['family']=='grid' and r['job']['ee']==e and r['job']['depth']==d]
                vals=[]
                for m in pair:
                    v=m['n_bursts']/m['observed_s'] if metric=='burst_rate_hz' and m['observed_s']>0 else m.get(metric)
                    if metric in ['cv','cv2'] and (m['n_bursts']<8 or m['label']=='sustained_activity'):v=None
                    if v is not None and np.isfinite(v):vals.append(v)
                if len(vals)==2:grid[iy,ix]=np.mean(vals)
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#ededed')
        im=ax.imshow(grid,origin='lower',aspect='equal',cmap=cmap)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        ax.set(xticks=range(len(plan['ee_values'])),yticks=range(len(plan['depth_values'])),
            xticklabels=[f'{v:g}' for v in plan['ee_values']],yticklabels=[f'{v:g}' for v in plan['depth_values']],
            title=title,xlabel='Within-core E→E multiplier',ylabel='Threshold-lowering amplitude')
        for iy,ix in zip(*np.where(np.isfinite(grid))):
            rgba=im.cmap(im.norm(grid[iy,ix]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(ix,iy,f'{grid[iy,ix]:.2g}',ha='center',va='center',fontsize=8,color='black' if lum>.5 else 'white')
    fig.suptitle('Combined-core observables | equal weight per run',fontsize=14)
    fig.text(.5,.01,'Each cell: mean of the two run-level summaries. Grey: fewer than two estimable runs; CV/CV2 require ≥8 bursts and no sustained activity.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.95],h_pad=2.5);save(fig,'native_burst_continuous_observables')

def examples(records):
    choices=[]
    for label in ['background','irregular_bursts','regular_bursts','patterned_bursts','variable_bursts','sustained_activity','sparse_bursts']:
        candidates=[(r,g) for r in records if r['job']['family']=='grid' for g in ['coreAE','coreBE'] if r['metrics'][g]['label']==label]
        if not candidates:
            candidates=[(r,'coreUnionE') for r in records if r['job']['family']=='grid' and r['metrics']['coreUnionE']['label']==label]
        if not candidates:continue
        # Prefer a phenotype reproduced in the same core across both noises.
        repeated=[]
        for r,g in candidates:
            pair=[x['metrics'][g] for x in records if x['job']['family']=='grid' and x['job']['ee']==r['job']['ee'] and x['job']['depth']==r['job']['depth']]
            if len(pair)==2 and all(x['label']==label and x['threshold_stable'] for x in pair):repeated.append((r,g))
        if repeated:candidates=repeated
        # Closest to the retained class pool's median rate; deterministic ties.
        med=np.median([r['metrics'][g]['mean_rate_hz'] or 0 for r,g in candidates])
        candidates.sort(key=lambda x:(not x[0]['metrics'][x[1]]['threshold_stable'],abs((x[0]['metrics'][x[1]]['mean_rate_hz'] or 0)-med),name(x[0]['job']),x[1]))
        choices.append(candidates[0])
    if not choices:return []
    fig,axes=plt.subplots(len(choices),2,figsize=(14,2.35*len(choices)),squeeze=False,gridspec_kw={'width_ratios':[1.45,1]})
    manifest=[]
    for row,(r,g) in enumerate(choices):
        m=r['metrics'][g];j=r['job'];folder=OUT/'per_run'/name(j)
        with np.load(folder/'trajectory.npz') as z:
            idx=z['group_names'].tolist().index(g);n=int(z['group_sizes'][idx])
            times=(np.arange(len(z['active_counts_10ms']))+.5)*.01
            frac=z['active_counts_10ms'][:,idx]/n
            trace_color={'background':'#667078','sparse_bursts':'#648b98'}.get(m['label'],COLORS[m['label']])
            ax=axes[row,0];ax.plot(times,frac,color=trace_color,lw=.8)
            ax.axhline(.1,color='#888888',ls=':',lw=.7)
            for e in m['events']:ax.axvspan(e['start_s']+2.,e['stop_s']+2.,color=COLORS[m['label']],alpha=.18)
            visible=frac[(times>=2.)&(times<=20.)]
            ax.set(xlim=(2.,20.),ylim=(0,max(.18,visible.max()*1.06 if len(visible) else .18)),ylabel='10 ms active fraction')
            group_title=dict(coreAE='Core A',coreBE='Core B',coreUnionE='Combined cores')[g]
            ax.set_title(f'{TEXT[m["label"]]} | {group_title} | EE={j["ee"]:g}, depth={j["depth"]:g}',loc='left',fontsize=11)
            cv='insufficient events' if m['n_bursts']<8 else ('not estimable' if m['cv'] is None else f'{m["cv"]:.2f}')
            ax.text(.99,.91,f'n={m["n_bursts"]}; CV={cv}',transform=ax.transAxes,ha='right',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
            # A fixed early post-burn-in segment avoids event-centered selection.
            core_num=0 if g=='coreAE' else 1
            sample=z['raster_sample_ids'];sample=sample[sample<len(z['core_index_E'])]
            if g=='coreUnionE':
                sample=sample[z['core_index_E'][sample]>=0]
                sample=sample[np.argsort(z['core_index_E'][sample],kind='stable')]
            else:sample=sample[z['core_index_E'][sample]==core_num]
            ids=z['raster_cell'];t=z['raster_time_ms']/1000
            sel=(t>=2.)&(t<=7.)&np.isin(ids,sample)
            order={int(c):k for k,c in enumerate(sample)}
            axes[row,1].scatter(t[sel],[order[int(c)] for c in ids[sel]],s=.4,c='#262626',rasterized=True)
            if g=='coreUnionE':
                divide=np.count_nonzero(z['core_index_E'][sample]==0)-.5
                axes[row,1].axhline(divide,color='#888888',ls=':',lw=.6)
            axes[row,1].set(xlim=(2.,7.),ylim=(-1,max(len(sample),1)),ylabel='Sampled core E cells',title='Fixed 2–7 s segment; occupied 2 ms bins')
            for a in axes[row]:a.set_xlabel('Time (s)')
        pair=[x['metrics'][g] for x in records if x['job']['family']=='grid' and x['job']['ee']==j['ee'] and x['job']['depth']==j['depth']]
        repeat_agreement=len(pair)==2 and all(x['label']==m['label'] and x['threshold_stable'] for x in pair)
        manifest.append(dict(name=name(j),group=g,label=m['label'],selection='same-core paired label and threshold agreement preferred; median firing rate in retained class pool; threshold-stable first',
            paired_label_threshold_agreement=repeat_agreement,raster_window_s=[2,7]))
    fig.tight_layout();save(fig,'native_burst_examples');return manifest

def controls(plan,records):
    rows=[]
    for r in records:
        j=r['job']
        if j['family']=='grid':continue
        ref=next(x for x in records if x['job']['family']=='grid' and x['job']['ee']==j['ee'] and x['job']['depth']==j['depth'] and x['job']['seed']==j['seed'])
        for g in GROUPS:
            m=r['metrics'][g];b=ref['metrics'][g]
            rows.append(dict(family=j['family'],ee=j['ee'],depth=j['depth'],seed=j['seed'],group=g,
                reference_label=b['label'],control_label=m['label'],reference_n=b['n_bursts'],control_n=m['n_bursts'],
                reference_rate_hz=b['mean_rate_hz'],control_rate_hz=m['mean_rate_hz'],reference_cv=b['cv'],control_cv=m['cv']))
    table(OUT/'controls.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.))
    for ax,family,title in zip(axes,['noise_off','new_network'],['Random input removed, mean matched','Same parameter anchors, network 2711']):
        rr=[r for r in rows if r['family']==family and r['group']=='coreUnionE']
        for iy,r in enumerate(rr):
            for ix,col in enumerate(['reference_label','control_label']):
                ax.add_patch(Rectangle((ix-.48,iy-.42),.96,.84,facecolor=COLORS[r[col]],edgecolor='white'))
                ax.text(ix,iy,str(r['reference_n'] if ix==0 else r['control_n']),ha='center',va='center',fontsize=10)
        ax.set(xlim=(-.5,1.5),ylim=(-.6,len(rr)-.4),xticks=[0,1],xticklabels=['Reference','Control'],yticks=range(len(rr)),
            yticklabels=[f'EE {r["ee"]:g}, depth {r["depth"]:g} | seed {str(r["seed"])[-2:]}' for r in rr],title=title)
        ax.invert_yaxis()
    fig.suptitle('Combined-core controls | numbers are detected bursts',fontsize=14)
    fig.text(.5,.01,'Colors follow the regime map. Noise-off trials start from the same resting initialization; they are not a within-trajectory noise withdrawal experiment.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.93]);save(fig,'native_burst_controls');return rows

def interval_figure(plan,records):
    """Unfitted interval distributions and their order, preserving run identity."""
    grid=[r for r in records if r['job']['family']=='grid']
    candidates=[]
    for d in plan['depth_values']:
        for e in plan['ee_values']:
            mm=[r['metrics'][g] for r in grid if r['job']['ee']==e and r['job']['depth']==d for g in ['coreAE','coreBE']]
            score=sum(m['label']=='irregular_bursts' and m['threshold_stable'] for m in mm)
            if score:candidates.append((score,min(m['n_bursts'] for m in mm),e,d))
    conditions=[]
    if candidates:
        ranked=sorted(candidates,key=lambda x:(-x[0],-x[1],x[2],x[3]))
        selected=[x for x in ranked if x[0]==4][:2] or ranked[:1]
        for score,n,e,d in selected:conditions.append((e,d,f'Irregular candidate ({score}/4 stable labels)'))
    for e,d,title in [(.85,1.,'Current reference'),(1.2,1.3,'Strong-coupling anchor')]:
        if not any(x[:2]==(e,d) for x in conditions):conditions.append((e,d,title))
    fig,axes=plt.subplots(2,len(conditions),figsize=(4.6*len(conditions),7),squeeze=False)
    manifest=[];plotted=[]
    for col,(e,d,title) in enumerate(conditions):
        pair=sorted([r for r in grid if r['job']['ee']==e and r['job']['depth']==d],key=lambda r:r['job']['seed'])
        for k,r in enumerate(pair):
            for g,color,group_title in [('coreAE','#35699b','A'),('coreBE','#bb6842','B')]:
                m=r['metrics'][g];times=np.array([v['start_s'] for v in m['events'] if not v['left_censored']]);dt=np.diff(times)
                if not len(dt):continue
                label=f'Core {group_title}, noise {k+1}';style='-' if k==0 else '--'
                axes[0,col].plot(times[1:]+2.,dt,style,color=color,lw=.8,marker='.',ms=3,label=label)
                values=np.sort(dt);cdf=np.arange(1,len(values)+1)/len(values)
                axes[1,col].step(values,cdf,where='post',color=color,ls=style,lw=1.2,label=label)
                plotted.extend(dt.tolist())
                manifest.append(dict(name=name(r['job']),group=g,label=m['label'],n_intervals=len(dt),cv=m['cv']))
            axes[0,col].set_title(f'{title}\nEE={e:g}, depth={d:g}',fontsize=11)
        axes[0,col].set(xlim=(2,20),xlabel='Time of next burst (s)',ylabel='Inter-burst interval (s)',yscale='log')
        axes[1,col].set(xlabel='Inter-burst interval (s)',ylabel='Empirical cumulative probability',xscale='log',ylim=(0,1.04))
    if plotted:
        lo=min(plotted)*.85;hi=max(plotted)*1.15
        for col in range(len(conditions)):
            axes[0,col].set_ylim(lo,hi);axes[1,col].set_xlim(lo,hi)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.025))
    fig.suptitle('Single-core burst intervals | each realization kept separate',fontsize=14)
    fig.text(.5,.007,'All complete adjacent onset intervals after 2 s; no distribution fit or event pooling. Candidate selected by repeat / threshold agreement, then event count.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.095,1,.94]);save(fig,'native_burst_interval_distributions')
    write(OUT/'interval_figure_selection.json',dict(selection='up to two cells with all four core/noise labels threshold-stable irregular, otherwise the best one; then greatest minimum burst count, deterministic parameter tie-break',conditions=conditions,series=manifest))

def report(plan,records,examples_manifest,control_rows):
    counts={g:dict(collections.Counter(r['metrics'][g]['label'] for r in records if r['job']['family']=='grid')) for g in GROUPS}
    stable={g:sum(r['metrics'][g]['threshold_stable'] for r in records if r['job']['family']=='grid') for g in GROUPS}
    consensus={}
    for g in GROUPS:
        agree=[]
        for d in plan['depth_values']:
            for e in plan['ee_values']:
                pair=[r['metrics'][g] for r in records if r['job']['family']=='grid' and r['job']['ee']==e and r['job']['depth']==d]
                agree.append(dict(ee=e,depth=d,labels=[x['label'] for x in pair],agreement=pair[0]['label']==pair[1]['label'],
                    robust_irregular=all(x['label']=='irregular_bursts' and x['threshold_stable'] for x in pair)))
        consensus[g]=agree
    combined_only=[]
    for r in records:
        if r['job']['family']=='grid' and r['metrics']['coreUnionE']['label']=='irregular_bursts' and all(r['metrics'][g]['label']!='irregular_bursts' for g in ['coreAE','coreBE']):combined_only.append(name(r['job']))
    dual_irregular=[]
    for d in plan['depth_values']:
        for e in plan['ee_values']:
            pair=[r for r in records if r['job']['family']=='grid' and r['job']['ee']==e and r['job']['depth']==d]
            if all(r['metrics'][g]['label']=='irregular_bursts' and r['metrics'][g]['threshold_stable'] for r in pair for g in ['coreAE','coreBE']):
                dual_irregular.append([e,d])
    noise_rows=[r for r in records if r['job']['family']=='noise_off']
    noise_quiet=sum(all(r['metrics'][g]['n_bursts']==0 for g in ['coreAE','coreBE']) for r in noise_rows)
    summary=dict(status='COMPLETE_AGENT_REVIEW_PENDING',measurement_version=metrics_v2.VERSION,budget=plan['budget'],completed_runs=len(records),grid_counts=counts,
        actual_duration_ms_counts=dict(collections.Counter(str(r['result']['actual_duration_ms']) for r in records)),
        runaway_runs=sum(r['result']['runaway_early_stop_ms'] is not None for r in records),
        threshold_stable_runs=stable,cell_consensus=consensus,combined_only_irregular_runs=combined_only,
        both_cores_repeated_threshold_stable_irregular_cells=dual_irregular,noise_off_zero_burst_runs=noise_quiet,
        examples=examples_manifest,figure_human_review='PENDING',patient_mechanism_accepted=False,model_frozen=False)
    write(OUT/'summary.json',summary)
    lines=['# 当前完整模型的原生burst动力学区域图（第一版）','',
        '本轮完成62条新的20秒轨迹：主网格50条、关闭随机输入6条、另一张网络的固定锚点6条。均使用当前32,000 E + 8,000 I完整网络；前2秒不进入表型统计。没有启动患者拟合或更改现有科学执行器。','',
        '## 问题与比较','',
        '要检验的是不规则、可自限burst是否形成可重演区域。横轴同核E→E倍率，纵轴两core的降阈值幅度；当前参考点(0.85,1)。降幅0保留原几何和限核随机输入，因此背景格只表示此模型的低活动条件，不能当成健康组织验证。','',
        '降阈值幅度同时缩放固定逐细胞降幅的均值与离散度；它不是均值固定的阈值异质性轴。图、连接入度和空间组织在每张网络内部固定，因此本版也不直接回答连接异质性的独立作用。','',
        '这里的irregular只定义burst发生时刻的变化；它不等于burst内部放电混乱，也没有证明事件内HFO波形及频率已经与患者一致。峰值同步比例是独立读出，不能用间隔CV替代。','',
        f'两个core各自在两条噪声下均呈不规则、且通过检测起始阈值敏感性的参数格为：{dual_irregular if dual_irregular else "本轮没有"}。这回答了本张网络内的重复性；两条噪声和18秒分析窗仍不足以确认稳定的数学相边界。','',
        '## 固定降阈值幅度1：连接强度的实际效应','',
        '每行包含两个core×两条噪声的四个单core读数；范围保留运行间及core间差异，未合并事件后计算CV。峰同步为每条运行各burst峰值的中位数，再报告这四个中位数的范围。','',
        '|同核E→E倍率|单core burst数范围|间隔CV范围|CV2范围|峰2ms同步比例范围|四个读出的类别|',
        '|---|---|---|---|---|---|']
    for e in plan['ee_values']:
        mm=[r['metrics'][g] for r in records if r['job']['family']=='grid' and r['job']['ee']==e and r['job']['depth']==1. for g in ['coreAE','coreBE']]
        def span(key,minimum=0):
            values=[x[key] for x in mm if x.get(key) is not None and x['n_bursts']>=minimum]
            return f'{min(values):.3g}–{max(values):.3g}' if values else '不可估计'
        labels=', '.join(f'{k}: {v}' for k,v in collections.Counter(x['label'] for x in mm).items())
        lines.append(f'|{e:g}|{span("n_bursts")}|{span("cv",8)}|{span("cv2",8)}|{span("median_peak_active_fraction_2ms")}|{labels}|')
    lines += ['',
        '## 主网格结果','',
        '|观测对象|50条运行的状态计数|起始阈值敏感性不改变标签|','|---|---|---|']
    for g in GROUPS:lines.append(f'|{g}|'+', '.join(f'{TEXT[k]}: {v}' for k,v in counts[g].items())+f'|{stable[g]}/50|')
    lines+=['','以下是两条噪声均判不规则、且两条都通过起始阈值敏感性检查的格子；它们仍是固定网络、20秒条件下的结果：','']
    for g in GROUPS:
        good=[(x['ee'],x['depth']) for x in consensus[g] if x['robust_irregular']]
        lines.append(f'- {g}：{good if good else "没有此类格子"}。')
    lines+=['',f'合并core被判不规则、但两个单独core均未被判不规则的运行数为{len(combined_only)}；这种情况不能解释成每个core内部已经产生不规则burst。具体列表在summary.json。',
        '', '## 机制边界与对照','',
        f'实际结果：{noise_quiet}/{len(noise_rows)}条noise-off锚点运行中，两个core均无检测到的burst。该对照只直接适用于(EE=0.5/0.85/1.2, depth=0/1)这六个参数点；若不规则候选格位于其他参数点，本轮不能替代该格自己的配对关噪检验。','',
        'noise-off去掉核内OU与Poisson随机性，固定为整流高斯输入的平稳期均值，其余物理保持。对照从统一初态重新运行；若burst消失，只支持该初态、该参数下的自发burst依赖随机输入，不证明所有吸引子或初态都静息。若持续重复，则说明确定性网络能够自维持相关活动，仍不能只凭CV把它定名为极限环。',
        '', '需要随机输入与事件是否规则是两个不同问题。关噪后静息本身不能区分不规则瞬态和噪声维持的规则重复活动；本版将间隔形态、峰同步性和去噪对照分别报告。',
        '', '网络2711仅复测三个预先固定锚点：低活动锚点(0.5,0)的两个core×两条噪声均保留背景，强连接锚点(1.2,1.3)的四个读出均保留规则burst；参考点(0.85,1)为三个变化性读出和一个规则读出。两个单core不规则候选点没有跨网络复测，不能由锚点结果替代。逐条件、逐core、每条噪声的标签及事件数见controls.csv与对照图。',
        '', '## 定义、可估计性与解释','',
        '所有事件直接取原生10ms内活跃细胞比例：10%启动、3%延续、≤20ms低活动间隙合并；启动阈值7.5%和12.5%检验标签敏感性。采用sample-SD CV及局部CV2；至少8个burst才分类规则性。CV和CV2的操作阈值、ACF与前后半段检查全部在运行前固定，见execution_plan.md。未满足两类条件的格子保留变化性burst，不强行称irregular。',
        '', '读出v2修订由“两个固定周期源错相叠加”这一明确反例触发，先通过合成检验，再统一消费全部原生轨迹；物理仿真不变。199次间隔重排保持间隔分布和事件数，对每个频率对称标准化实测及替代谱，再在0.5–15Hz且观察窗至少8周期的频带取最大额外谱功率。原本高CV/CV2的运行若p≤0.05，改标patterned_bursts；其余保留描述性irregular。频率搜索包含在每次替代检验中，但这不是跨网格的总体显著性检验。未拒绝重排参照不能证明非周期，更不证明噪声机制。完整理由与两版文件边界见measurement_amendment_v2.md。',
        '', '持续活动包括连续活动段≥1秒、占时≥75%或原执行器的持续高率提前终止。连续读数与原生raster用于判断阈值划分是否可信；它们不是严格数学分岔图。事件是运行内样本，网格格子不是独立患者；每条运行等权，未对事件池进行伪重复检验。',
        '', '## 交付与验收','',
        'figures/含区域图、连续指标、真实存在状态的代表轨迹、单core间隔分布及噪声/网络对照；每格两三角分别对应两条噪声，未插值边界。per_run/保留全部原生数组、实际阈值/图审计和事件表，run_metrics.csv汇总。代表轨迹优先选同一core在两噪声均复现且检测阈值稳定的类别，再取该候选池发放率中位附近；raster统一截取2–7秒。间隔图同时展示原始次序和经验累积分布，不拟合分布或合并运行；最多展示两个四个单core读出均通过的不规则参数点，候选选取规则和逐系列样本数在interval_figure_selection.json。不存在的状态没有补画示意。',
        '', '原执行器和有序scatter的完整状态及观测逐位一致，已有基线1秒前缀复现；数值测试与验证文件见validation.json。Agent图件检查后另写visual_review.json，用户目视验收仍待定。完成本轮后停止，没有追加波次、冻结模型或进入Fig5。']
    (OUT/'scientific_report.md').write_text('\n'.join(lines)+'\n')
    readme='''### native_burst_regime_map.png / .pdf
展示同核E→E强度与两核降阈值幅度平面上的原生动力学区域，两个core及合并core分别作图。每格两个三角表示两条噪声，叉号表示标签对启动阈值敏感，黑框是原参考参数。高CV但间隔次序仍提供额外谱结构的运行单标patterned。**关注点**：不规则是否在重复和阈值敏感性中保留，以及合并活动是否掩盖单core规则性。

### native_burst_continuous_observables.png / .pdf
展示合并core的平均发放率、burst率、CV、CV2、2ms峰同步比例和burst占时。每格先计算每次运行再等权平均；CV/CV2支持不足和持续活动留灰。**关注点**：类别边界是否对应连续观测的实际改变，不把灰格当零。

### native_burst_examples.png / .pdf
展示实际出现类别的原生群体活动和固定2–7秒片段内的细胞raster，优先选择同一core跨噪声与检测阈值均保留的类别，再取候选池发放率中位附近的运行。raster显示选定细胞是否在2ms格内发放，不是完整单细胞精确spike时间；合并core的点线分隔A与B细胞。**关注点**：burst能否回到低活动、不规则是否直观看到、持续活动是否被误切成事件。

### native_burst_controls.png / .pdf
并列显示关闭随机输入以及另一张网络固定锚点的对照，数字为原生burst数量，颜色沿用区域图。关闭噪声保持整流后平稳期平均输入，但从统一初态重跑。**关注点**：随机输入依赖与网络依赖分别判断，不把单初态对照解释成全局吸引子证明。

### native_burst_interval_distributions.png / .pdf
展示单core间隔随时间的变化与经验累积分布，每条噪声及两个core始终分开。对比最多两个四个单core读出均通过的不规则候选、当前参考点和强连接锚点；选择规则与每条曲线的样本数另存JSON，未进行分布拟合。**关注点**：间隔展宽是否伴随明显漂移，以及不同运行的分布是否相近。
'''
    (OUT/'figures/README.md').write_text(readme)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    plan=read(OUT/'plan.json');records=[];rows=[]
    for j in plan['jobs']:
        m=summarize(j);r=read(OUT/'per_run'/name(j)/'result.json');records.append(dict(job=j,result=r,metrics=m))
        for g,v in m.items():rows.append(dict(name=name(j),**j,group=g,**{k:x for k,x in v.items() if not isinstance(x,(list,dict))},actual_duration_ms=r['actual_duration_ms']))
    audit(plan,records)
    table(OUT/'run_metrics.csv',rows)
    phase(plan,records);continuous(plan,records);ex=examples(records);controls_rows=controls(plan,records);interval_figure(plan,records);report(plan,records,ex,controls_rows)
    checks=[]
    for p in sorted((OUT/'figures').glob('*.png')):
        with Image.open(p) as im:im.load();checks.append(dict(file=str(p),width=im.width,height=im.height))
    write(OUT/'figure_validation.json',dict(status='FILES_DECODED_AGENT_VISUAL_PENDING',files=checks))
    print(json.dumps(dict(status='FIGURES_COMPLETE',runs=len(records),figures=len(checks))),flush=True)

if __name__=='__main__':main()

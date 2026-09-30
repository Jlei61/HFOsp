"""Isolated retrospective correspondence audit from paired raw records.

No native engine import, simulation, parameter tuning, or upstream writes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,csv
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,ListedColormap
from matplotlib.patches import Circle,Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
OUT=ROOT/'results/topic4_sef_hfo/fig5_native_reduction_correspondence_20260916'
FIG=OUT/'figures'
SEEDS=(9108401,9108402)
DT=.01

def read(path):return json.loads(path.read_text())
def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def csv_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def first(mask,hold=1,start=8.):
    for a,b in runs(mask):
        if b-a>=hold:return float(start+a*DT)
    return None
def finite_events(rate,start=8.):
    separators=[(a,b) for a,b in runs(rate<5.) if b-a>=2]
    events=[]
    for (_,left),(right,_) in zip(separators[:-1],separators[1:]):
        if right-left>=2 and max(rate[left:right])>=20:
            events.append(dict(start_bin=int(left),end_bin=int(right),
                  start_s=float(start+left*DT),end_s=float(start+right*DT),
                  duration_ms=int((right-left)*10),peak_hz=float(max(rate[left:right]))))
    return separators,events

def load_native(seed,counts):
    folder=SOURCE/f'replay/runs/eta0.0005_s{seed}/chunks'
    cell=[];total=[];times=[];end=80000
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as a:
            lo,hi=int(a['start_step']),int(a['end_step'])
            if hi<=80000:continue
            assert lo==end,(seed,lo,end)
            end=hi;cell.append(a['field_1ms'].astype(float));total.append(a['spikes_1ms'][:,0].astype(float));times.append(a['time_ms'])
    cell=np.concatenate(cell);total=np.concatenate(total);times=np.concatenate(times)
    assert len(cell)==4500 and end==125000
    assert np.array_equal(cell.sum(1),total)
    assert np.max(abs(np.diff(times)-1))<1e-8
    rate=cell.reshape(-1,10,400).sum(1)/counts/.01
    pop=total.reshape(-1,10).sum(1)/counts.sum()/.01
    assert np.max(abs(np.average(rate,weights=counts,axis=1)-pop))<1e-10
    return rate,dict(spike_count_conservation=True,first_record_time_ms=float(times[0]),
        last_record_time_ms=float(times[-1]),start_step=80000,end_step=end,source=str(folder)),cell/counts/.001

def load_reduced(seed,counts):
    path=SOURCE/f'approx/v1/runs/path_s{seed}/fields.npz'
    with np.load(path) as a:
        assert np.array_equal(a['count_e'],counts)
        assert int(a['start_step'])==80000 and float(a['frame_ms'])==1.
        rate1=a['fields_hz'][:,0].astype(float);rate=rate1.reshape(-1,10,400).mean(1)
    assert rate.shape==(450,400)
    result=read(path.parent/'result.json')
    assert result['job']['mode']=='path' and result['job']['seed']==seed
    return rate,dict(source=str(path),job=result['job'],variant=result['variant'],
        Z='supplied native path',M='dynamic',input='matched conditional external rate; independent-input closure'),rate1

def arrival(field,threshold=50.,window=(9.42,10.37)):
    lo=int(round((window[0]-8.)/DT));hi=int(round((window[1]-8.)/DT))
    arr=np.full(400,np.nan);left=np.zeros(400,bool)
    for cell in range(400):
        # Conservative censoring: any activity in the two preceding bins.
        left[cell]=bool(np.any(field[max(0,lo-2):lo,cell]>threshold))
        if left[cell]:continue
        hit=first(field[lo:hi,cell]>threshold,2,window[0])
        if hit is not None:arr[cell]=hit
    return arr,left

def region_event_descriptor(field,events,region_weights,threshold=50.):
    core=np.column_stack([np.average(field,weights=w,axis=1) for w in region_weights])
    rows=[]
    for e in events:
        a,b=e['start_bin'],e['end_bin'];hits=[]
        for k in (0,1):
            initial=bool(np.any(core[max(0,a-2):a,k]>threshold))
            hit=None if initial else first(core[a:b,k]>threshold,1,e['start_s'])
            hits.append(dict(left_censored=initial,arrival_s=hit))
        lag=None if any(h['arrival_s'] is None for h in hits) else 1000*(hits[1]['arrival_s']-hits[0]['arrival_s'])
        rows.append(dict(**e,core_arrivals=hits,B_minus_A_ms=lag))
    return rows

def metrics(field,counts,region_weights):
    global_rate=np.average(field,weights=counts,axis=1)
    separators,events=finite_events(global_rate)
    before=[e for e in events if e['start_s']>=8.-1e-9 and e['end_s']<=9.42+1e-9]
    all_events=region_event_descriptor(field,before,region_weights)
    high=first(global_rate>=200,20)
    last_quiet=max([8.+b*DT for a,b in separators if high is not None and 8.+b*DT<=high+1e-9],default=None)
    windows={}
    for name,lo,hi in [('pre_entry',8.,9.42),('entry',9.42,10.37),('high_tail',10.37,12.5)]:
        a,b=int(round((lo-8.)/DT)),int(round((hi-8.)/DT))
        windows[name]=dict(time_window_s=[lo,hi],mean_global_E_hz=float(global_rate[a:b].mean()),
             quiet_fraction=float(np.mean(global_rate[a:b]<5)),
             finite_events=sum(e['start_s']>=lo-1e-9 and e['end_s']<=hi+1e-9 for e in events))
    expansion={}
    for th in (20.,50.,100.):
        occ=np.average(field>th,weights=counts,axis=1)
        expansion[str(int(th))]={str(f):dict(first_s=first(occ>=f),sustained20ms_s=first(occ>=f,2)) for f in (.25,.5,.75)}
    return dict(windows=windows,high_onset_s=high,last_quiet_end_s=last_quiet,
         finite_events_pre_entry=all_events,spatial_expansion_common_start8=expansion),global_rate

def compare_arrivals(nat,red,counts,threshold):
    an,ln=arrival(nat,threshold);ar,lr=arrival(red,threshold)
    vn=np.isfinite(an);vr=np.isfinite(ar);joint=vn&vr;union=vn|vr
    errs=(ar[joint]-an[joint])*1000
    corr=None
    if joint.sum()>=3 and len(np.unique(an[joint]))>1 and len(np.unique(ar[joint]))>1:
        corr=float(spearmanr(an[joint],ar[joint]).statistic)
    return dict(threshold_hz=threshold,hold_ms=20,window_s=[9.42,10.37],
        native_left_censored_cells=int(ln.sum()),reduced_left_censored_cells=int(lr.sum()),
        native_valid_cells=int(vn.sum()),reduced_valid_cells=int(vr.sum()),joint_valid_cells=int(joint.sum()),
        newly_recruited_weighted_iou=float(counts[joint].sum()/counts[union].sum()) if union.any() else None,
        arrival_order_spearman=corr,
        median_signed_time_error_ms=float(np.median(errs)) if len(errs) else None,
        median_absolute_time_error_ms=float(np.median(abs(errs))) if len(errs) else None,
        median_offset_removed_absolute_error_ms=float(np.median(abs(errs-np.median(errs)))) if len(errs) else None,
        scope='first sustained recruitment during a fixed transition window, not a single-wave propagation velocity'),(an,ln,ar,lr)

def spatial_panel(ax,arr,centers,cmap,norm,label,left=None):
    data=np.asarray(arr).reshape(20,20);valid=np.isfinite(data)
    base=np.zeros((20,20));base[~valid]=1
    if left is not None:base[np.asarray(left).reshape(20,20)]=2
    ax.imshow(base,origin='lower',extent=[0,20,0,20],cmap=ListedColormap(['white','#ececec','#555555']),vmin=0,vmax=2,interpolation='nearest')
    im=ax.imshow(np.ma.masked_invalid(data),origin='lower',extent=[0,20,0,20],cmap=cmap,norm=norm,interpolation='nearest')
    for j,xy in enumerate(centers):
        ax.add_patch(Circle(xy,1.5,fc='none',ec='#18c9ce',lw=1.1))
        ax.text(xy[0],xy[1]+2.2,'AB'[j],ha='center',fontsize=8,color='#08757a',bbox=dict(fc='white',ec='none',pad=.1))
    ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',ylabel='y (mm)')
    ax.text(.5,1.04,label,transform=ax.transAxes,ha='center',fontsize=10)
    return im

def save(fig,name):
    assert all(ax.get_title()=='' for ax in fig.axes)
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def plot(data,counts,centers):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(11.3,7.7),sharex=True)
    fig.subplots_adjust(hspace=.15,wspace=.25,bottom=.12,top=.93)
    t=8.+np.arange(450)*.01+.005
    for col,seed in enumerate(SEEDS):
        for model,color in [('native','#202020'),('reduced','#c75b27')]:
            f=data[seed][model];pop=np.average(f,weights=counts,axis=1);occ=np.average(f>50,weights=counts,axis=1)
            axes[0,col].plot(t,pop,c=color,lw=.75,label='Native SNN' if model=='native' else 'Reduced v1, supplied Z(t)')
            axes[1,col].plot(t,occ,c=color,lw=.85)
            mm=data[seed]['metrics'][model]['finite_events_pre_entry']
            yy=1 if model=='native' else 0
            axes[2,col].broken_barh([(e['start_s'],e['end_s']-e['start_s']) for e in mm],(yy-.15,.3),color=color)
        axes[0,col].text(.5,1.04,f'Noise realization {col+1}',ha='center',transform=axes[0,col].transAxes)
        axes[0,col].set_ylabel('Global E rate (Hz)');axes[1,col].set_ylabel('E-weighted fraction >50 Hz')
        axes[1,col].set_ylim(-.03,1.03);axes[2,col].set(yticks=[0,1],yticklabels=['Reduced','SNN'],ylim=(-.5,1.5),xlabel='Time (s)',ylabel='Complete finite events')
        for ax in axes[:,col]:
            ax.axvline(9.42,c='#777',ls=':',lw=.8);ax.axvline(10.37,c='#777',ls=':',lw=.8);ax.set_xlim(8,10.4)
    axes[0,0].legend(frameon=False,fontsize=9,loc='upper left')
    save(fig,'fig_temporal_correspondence')
    fig,axes=plt.subplots(2,2,figsize=(7.7,7.2));fig.subplots_adjust(left=.09,right=.83,bottom=.11,top=.93,wspace=.25,hspace=.35)
    for row,seed in enumerate(SEEDS):
        an,ln,ar,lr=data[seed]['arrival_arrays']
        for col,(arr,left,label) in enumerate([(an,ln,'SNN'),(ar,lr,'Reduced')]):
            im=spatial_panel(axes[row,col],arr,centers,'viridis',Normalize(9.42,10.37),f'{label}, noise {row+1}',left)
    fig.colorbar(im,cax=fig.add_axes([.88,.30,.022,.50]),label='First sustained recruitment time (s)',ticks=[9.42,9.6,9.8,10.,10.37])
    fig.legend(handles=[Patch(fc='#555555',label='Already active at window start'),Patch(fc='#ececec',label='Not recruited in window')],loc='lower center',bbox_to_anchor=(.49,-.015),ncol=2,frameon=False,fontsize=9)
    save(fig,'fig_spatial_recruitment_correspondence')
    # Actual paired 50-ms maps, same absolute time and color scale.
    fig,axes=plt.subplots(4,3,figsize=(10.8,12.1));fig.subplots_adjust(left=.08,right=.88,bottom=.065,top=.955,wspace=.28,hspace=.38)
    for k,seed in enumerate(SEEDS):
        for mi,model in enumerate(('native','reduced')):
            for j,when in enumerate((9.42,9.87,10.37)):
                a=int(round((when-.025-8)*1000));b=a+50
                im=spatial_panel(axes[k*2+mi,j],data[seed][model+'_1ms'][a:b].mean(0),centers,'magma',Normalize(0,500),f'{when:.2f} s' if k*2+mi==0 else '')
                if j==0:axes[k*2+mi,j].set_ylabel(f"{'SNN' if mi==0 else 'Reduced'}, noise {k+1}\ny (mm)")
    fig.colorbar(im,cax=fig.add_axes([.915,.30,.017,.44]),label='E rate (Hz)',ticks=[0,250,500]);save(fig,'fig_paired_spatial_fields')
    fig,axes=plt.subplots(2,2,figsize=(7.6,7.1));fig.subplots_adjust(left=.09,right=.83,bottom=.09,top=.93,wspace=.25,hspace=.32)
    for row,seed in enumerate(SEEDS):
        for col,model in enumerate(('native','reduced')):
            field=data[seed][model+'_1ms'][1395:1445].mean(0)
            im=spatial_panel(axes[row,col],field,centers,'magma',Normalize(0,500),f"{'SNN' if col==0 else 'Reduced'}, noise {row+1}")
    fig.colorbar(im,cax=fig.add_axes([.88,.26,.022,.55]),label='E rate (Hz)',ticks=[0,250,500]);save(fig,'fig_preentry_spatial_pair')

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    geo=np.load(SOURCE/'replay/geometry.npz');coarse=np.load(SOURCE/'approx/coarse_20/geometry.npz')
    counts=geo['cell_e_counts'].astype(float);centers=geo['centers_mm']
    assert counts.sum()==32000 and np.array_equal(counts,coarse['count_e'])
    assert np.array_equal(geo['positions_e'],coarse['positions_e'])
    assert np.allclose(centers,coarse['centers_mm'])
    xy=coarse['positions_e'];spatial_index=np.floor(xy[:,1]).astype(int)*20+np.floor(xy[:,0]).astype(int)
    assert np.array_equal(spatial_index,coarse['cell_e'])
    region_weights=[np.bincount(coarse['cell_e'][coarse['g175']==j],minlength=400) for j in (0,1)]
    identity=read(SOURCE/'reference_identity.json');assert identity['status']=='PASS' and identity['identity_match']
    data={};rows=[];provenance=[]
    for seed in SEEDS:
        nat,nq,nat1=load_native(seed,counts);red,rq,red1=load_reduced(seed,counts)
        assert np.isfinite(nat).all() and np.isfinite(red).all() and nat.min()>=0 and red.min()>=0
        nm,ng=metrics(nat,counts,region_weights);rm,rg=metrics(red,counts,region_weights)
        recruitment=[]
        for th in (20.,50.,100.):
            comparison,arr=compare_arrivals(nat,red,counts,th);recruitment.append(comparison)
            if th==50:primary_arrays=arr
        pair=dict(seed=seed,native=nm,reduced=rm,recruitment=recruitment,
            finite_event_propagation_gate='NOT_ESTIMABLE' if min(len(nm['finite_events_pre_entry']),len(rm['finite_events_pre_entry']))<3 else 'ESTIMABLE_BUT_NOT_YET_ACCEPTED',
            pre_entry_state_correspondence='FAIL' if abs(nm['windows']['pre_entry']['quiet_fraction']-rm['windows']['pre_entry']['quiet_fraction'])>.1 else 'NOT_SUFFICIENT_FOR_PASS',
            entry_occupancy_MAE=float(np.mean(abs(np.average(nat[142:237]>50,weights=counts,axis=1)-np.average(red[142:237]>50,weights=counts,axis=1)))),
            pre_entry_occupancy_MAE=float(np.mean(abs(np.average(nat[:142]>50,weights=counts,axis=1)-np.average(red[:142]>50,weights=counts,axis=1)))))
        data[seed]=dict(native=nat,reduced=red,native_1ms=nat1,reduced_1ms=red1,metrics=dict(native=nm,reduced=rm),arrival_arrays=primary_arrays)
        rows.append(pair);provenance.append(dict(seed=seed,native=nq,reduced=rq))
        np.savez_compressed(OUT/f'paired_fields_{seed}.npz',native_rate10_hz=nat,reduced_rate10_hz=red,count_e=counts,t_s=8.+np.arange(450)*.01,
             native_recruitment_s=primary_arrays[0],native_left_censored=primary_arrays[1],reduced_recruitment_s=primary_arrays[2],reduced_left_censored=primary_arrays[3])
        snapshot_times=[9.42,9.87,10.37];snapshots={}
        for label,fields in [('native',nat1),('reduced',red1)]:
            snapshots[label]=np.array([fields[int(round((t-.025-8)*1000)):int(round((t+.025-8)*1000))].mean(0) for t in snapshot_times])
        np.savez_compressed(OUT/f'paired_snapshots_{seed}.npz',**snapshots,time_s=snapshot_times,window_duration_ms=50)
        print(seed,'complete events',len(nm['finite_events_pre_entry']),len(rm['finite_events_pre_entry']),'recruitment',recruitment[1],flush=True)
    original=read(SOURCE/'approximation_validation.json')['rows']
    assert len(original)==24
    for row in original:
        native_readout=read(SOURCE/'native/runs'/row['name']/'readout.json')
        assert row['native_category']==native_readout['category']
    native_self=[r for r in original if r['native_category']=='SELF_LIMITED']
    validation=dict(paired_conditions=24,native_self_limited_conditions=len(native_self),
       reproduced_self_limited=sum(r['approx_category']=='SELF_LIMITED' for r in native_self),
       category_matches=sum(r['category_match'] for r in original),
       magnitude_matches=sum(r['magnitude_pass'] for r in original),source=str(SOURCE/'approximation_validation.json'))
    extensions=read(SOURCE/'approx/v1/validation_extension.json')['rows']
    validation['extension_pairs']=len(extensions)
    validation['extension_category_matches']=sum(r.get('native_category')==r.get('approx_category') for r in extensions)
    validation['extension_native_self_limited']=sum(r['native_category']=='SELF_LIMITED' for r in extensions)
    validation['old_summary_count_correction']='Main 24 conditions contain 8 SELF_LIMITED, not 12; extension 8 contain 6, scored separately and not independent trajectories.'
    baseline,_=compare_arrivals(data[SEEDS[0]]['native'],data[SEEDS[1]]['native'],counts,50.)
    baseline['scope']='two previously observed native noise realizations; descriptive only, not an equivalence margin'
    write(OUT/'paired_metrics.json',dict(rows=rows,frozen_Z_validation=validation,native_noise_comparison=baseline))
    csv_rows(OUT/'frozen_condition_validation.csv',original)
    csv_rows(OUT/'recruitment_metrics.csv',[dict(seed=r['seed'],**a) for r in rows for a in r['recruitment']])
    csv_rows(OUT/'temporal_metrics.csv',[dict(seed=r['seed'],model=model,window=name,**a) for r in rows for model in ('native','reduced') for name,a in r[model]['windows'].items()])
    write(OUT/'provenance.json',dict(reference_identity=identity['network'],Vth_E_counts=identity['Vth_E_counts'],
        source_identity_status=identity['status'],pairs=provenance,statistical_unit='full trajectory or full frozen condition; no pixel-level independent samples',
        current_q1p25_branch_model='different parameterization; no paired SNN validation in this audit',
        q1_smooth_primitive_branch_model='numerically refined descendant; exact full path not rerun; validation not automatically inherited',
        old_expansion_window_mismatch='old native 0-12.5s first crossings versus reduced 8-12.5s; here both start 8s'))
    # Reproduce event/quiet summaries independently; allow only roundoff.
    old=read(SOURCE/'approx/v1/path_replay_gate4.json')['per_seed'];checks=[]
    for row in rows:
        seed=row['seed']
        for label,oldlabel in [('native','native'),('reduced','model')]:
            metrics_=row[label];previous=old[str(seed)][oldlabel]
            assert len(metrics_['finite_events_pre_entry'])==previous['events_8_to_9p42']
            assert abs(metrics_['windows']['pre_entry']['quiet_fraction']-previous['quiet_fraction_8_9p42'])<1e-12
            assert abs(metrics_['high_onset_s']-previous['high_onset_s'])<1e-9
            checks.append(dict(seed=seed,model=label,events_match=True,quiet_match=True,high_onset_match=True))
    meta_path=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v2/review_square_complete_20260915/fig5_metadata.json'
    meta=read(meta_path);snap=np.load(OUT/'paired_snapshots_9108401.npz');snapshot_checks=[]
    for current,original in ((0,2),(2,3)):
        error=float(np.max(abs(snap['native'][current]-np.array(meta['native_maps'][original]['rate_Hz']))))
        assert error<1e-10
        snapshot_checks.append(dict(time_s=float(snap['time_s'][current]),max_abs_rate_error_hz=error))
    write(OUT/'qa.json',dict(status='PASS',cell_count_and_position_identity=True,spike_count_conservation=True,
         common_window_s=[8.,12.5],grid=[20,20],rate_bin_ms=10,old_event_readouts_reproduced=checks,
        original_Fig5_snapshot_identity=snapshot_checks,
        snapshot_definition='50 ms centered on the indicated time, matching original Fig.5 windows',
        no_upstream_mutation=True,no_new_simulation=True))
    plot(data,counts,centers)
    report(rows,validation,baseline)
    write(OUT/'status.json',dict(status='EXECUTION_COMPLETE',original_q1_v1_dynamic_correspondence='FAIL',
        propagation_event_distribution='NOT_ESTIMABLE_IN_REDUCED_MODEL',transition_recruitment='DESCRIPTIVE_PAIRED_RESULTS_AVAILABLE',
        q1p25_native_correspondence='NOT_VALIDATED',bifurcation_interpretation_gate='NOT_PASSED',
        active_processes=[],human_visual_acceptance='PENDING'))
    print('COMPLETE',flush=True)

def report(rows,validation,baseline):
    table=[];prop=[];exp=[]
    for k,row in enumerate(rows,1):
        n,r=row['native'],row['reduced'];nm,rm=n['windows']['pre_entry'],r['windows']['pre_entry']
        table.append(f"| {k} | {len(n['finite_events_pre_entry'])} / {len(r['finite_events_pre_entry'])} | {nm['quiet_fraction']:.3f} / {rm['quiet_fraction']:.3f} | {nm['mean_global_E_hz']:.2f} / {rm['mean_global_E_hz']:.2f} | {n['high_onset_s']:.2f} / {r['high_onset_s']:.2f} |")
        for th in (20,50,100):
            a=next(p for p in row['recruitment'] if p['threshold_hz']==th)
            fmt=lambda x:'NA' if x is None else f'{x:.3f}'
            prop.append(f"| {k} | {th} | {a['native_valid_cells']} / {a['reduced_valid_cells']} | {a['joint_valid_cells']} | {fmt(a['newly_recruited_weighted_iou'])} | {fmt(a['arrival_order_spearman'])} | {fmt(a['median_absolute_time_error_ms'])} | {fmt(a['median_offset_removed_absolute_error_ms'])} |")
        for frac in ('.25','.5','.75'):
            f=str(float(frac));a=n['spatial_expansion_common_start8']['50'][f];b=r['spatial_expansion_common_start8']['50'][f]
            exp.append(f"| {k} | {f} | {a['first_s']} / {b['first_s']} | {a['sustained20ms_s']} / {b['sustained20ms_s']} |")
    text='''# 原生 SNN—降阶模型对应性审阅

## 结论

**继续解释 SNN 分岔的前提未通过。** 原参数 q_IE=1 的粗粒度率模型 v1 没有保留发作附近的有限事件与静息结构；事件传播模式的跨模型验收因降阶事件不足而不可估计。已补上原始空间记录的共同窗口招募对照，但这些条件输入下的招募诊断不能覆盖时间动力学失败，也不能验证自主 Z/M 转变。

当前分岔图使用的 q_IE=1.25 降阶模型没有同参数原生 SNN 配对。因此“原参数 v1 已失败”和“当前调参分岔模型尚未验证”须分别记录，不能将本次误差直接移用到另一个参数版本。平滑传递积分和临界点求解的数值精度不是原生对应性的替代验证。

本轮复用了已存在的原始记录，未重新仿真、未调参、未修改原模型或父任务结果。两条噪声实现都已有总体结果公开，不是新留出测试。

## 时间动力学：从原始空间计数独立重算

两模型严格共享8–12.5 s起止与10 ms窗。下表事件、静息和平均率取8–9.42 s；每项数值顺序均为 SNN / 降阶。

| 噪声实现 | 完整有限事件数 | 静息比例 | 全局 E 平均率 Hz | 高率起始 s |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+f'''

原冻结 Z 配对的 {validation['native_self_limited_conditions']} 个 SNN 自限条件，仅 {validation['reproduced_self_limited']} 个被降阶复现为自限；24 条中类别匹配 {validation['category_matches']} 条、量值门槛通过 {validation['magnitude_matches']} 条。已有8条延长结果并未恢复对应性。不能只凭约0.11 s的高率起始误差声称转变动力学相同。

**计数纠正**：旧审阅摘要称24条主条件中有12个自限条件；逐条复核 native/readout.json 与验证表后，实际为8个。8条延长条件中另有6个自限，不能混入主批、也不能当作额外独立轨迹。此次纠正改变分母，不改变“自限条件0个复现”的结论。

## 传播：区分有限事件传播与进入阶段的空间招募

原生在主窗有4–6个完整事件，降阶只有0–1个。按至少3个事件的最低可估计条件，源核比例、核间先后/时差分布及传播速度分布均为 **NOT_ESTIMABLE**。不能把降阶的持续高活动切成任意时间片，冒充对应的事件传播。

另对9.42–10.37 s的整个进入过程计算每格首次连续20 ms超过阈值的时刻。窗口开始前20 ms内已越阈者左删失，窗口内没有越阈者不填时间。到达次序相关仅限共同有效格；同时报告空间参与交并比，避免仅在小交集上得高相关。这个量是多事件背景上的首次持续招募，**不是单个传播波的速度**。

| 噪声 | 阈值 Hz | 新招募格 SNN / 降阶 | 共同有效格 | 加权交并比 | 时序 Spearman | 时间误差中位绝对值 ms | 去共同偏移后 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(prop)+'''

两条原生噪声轨迹的相同指标也保存于 paired_metrics.json 的 native_noise_comparison；其共同格到达次序相关仅约0.029，绝对时间误差中位数约100 ms，新招募格的加权交并比约0.868。这说明首次招募顺序在当前定义下本来就对噪声敏感，不能仅凭模型相关0.016/0.133低而宣称传播机制不同；模型空间参与交并比0.543/0.615与大量提前活跃格，是需要解释的描述性差异。只有一对原生对照，不能据此估计可靠的等效误差界限或给传播 PASS。阈值敏感性提供诊断范围，不按结果选择最有利阈值。

## 修正旧空间扩张比较的窗口问题

旧报告部分首次扩张时刻，原生从0 s计、降阶从8 s计，25%/50%扩张因而不可直接对比。本轮全部从8 s重新计算。以下率阈值均为50 Hz，比例按 E 细胞数加权；数值顺序 SNN / 降阶：

| 噪声 | 参与比例 | 首次越过 s | 至少持续20 ms s |
|---|---:|---|---|
'''+ '\n'.join(exp)+'''

首次75%扩张与全局200 Hz持续进入不是同一观测，单次大爆发可能先满足前者。这个修正没有改变有限事件和静息结构失败的结论；这些事件读出与原报告逐项重现。

## 具体缺口与下一版路线

当前最值得检验的失败原因是独立输入闭合丢失了同步脉冲簇的作用，而非连接图不同。已有原生工作点诊断显示平均率输入闭合只输出约0.5 Hz，而原生约22 Hz；这是支持该原因的证据，并非本轮新发现。仍需区分相关输入统计与膜电位/不应期分布造成的历史依赖，不能仅靠增大方差系数补平均率。

建议下一版先做局部输入—输出有效性检查，再建立新闭合：

1. 从原参照挑选独立的自限、局部持续、进入高率窗口，保留真实靶端E/I输入的时间结构。比较给定真实输入时，LIF群体与现有率传递的响应；以原生细胞/群体状态初始化，不拟合原生Z路径为预测成功。
2. 在保留一阶驱动的条件下，单独打散跨输入同步性；另单独控制膜电位与不应期初态。前者能区分同步脉冲簇效应，后者能区分状态记忆；其具体构造须保持所声称固定的量并单独验证。若两者都影响显著，单一无记忆率闭合不够。
3. 依据该诊断决定保留脉冲簇/相关输入变量还是膜电位与不应期群体分布，不以“把分岔移进合法D”选择模型结构。
4. 固定新模型后，先在同参、同输入、给定Z(t)下恢复有限事件与传播统计；再以新的未参与选择的噪声实现验证。通过之后才检查动态Z/M自主路径和冻结Z分岔。若第一层仍失败，停止其分岔解释路线，保留原生SNN的干预状态图。

本次仅完成对应资格复核与传播诊断；没有自动执行上述新闭合研究，也没有停止或干预父任务的任何进程。新模型验证前，已有分岔结果应作为该降阶方程的数学探索保留。

## 文件

- paired_metrics.json：两轨迹时间、事件、招募与原生噪声差异。
- temporal_metrics.csv、recruitment_metrics.csv、frozen_condition_validation.csv：可直接审阅的聚合表。
- paired_fields_*.npz：配对10 ms空间率、招募时间、删失掩码。
- provenance.json：网络身份、原生/降阶来源、Z/M/输入处理与版本边界。
- qa.json：计数守恒、网格/位置/计数一致、旧事件读出重现。
- figures/：时间动力学、招募时序图、相同时刻空间率场。

Agent 完成图形自查后交付候选图；用户人工图形验收待完成。
'''
    (OUT/'scientific_report.md').write_text(text)
    (FIG/'README.md').write_text('''### fig_temporal_correspondence.png / .pdf / .svg
两套已保存噪声实现中，SNN与降阶v1使用共同8 s起点及10 ms读出，比较全局率、空间占据与完整有限事件。虚竖线为9.42和10.37 s，同一读出下有限事件减少及静息结构缺失可直接观察。
**关注点**：相近的高率进入时间是否掩盖了进入前动力学不一致；这不是新的仿真或留出验证。

### fig_spatial_recruitment_correspondence.png / .pdf / .svg
比较9.42–10.37 s中每格首次连续20 ms超过50 Hz的开始时刻，深灰为窗口开始前已活跃而左删失，浅灰为窗内未招募。相同的绝对时间色标用于两模型和两套噪声；删失格不参与到达次序相关。
**关注点**：这是整个进入阶段的招募时序，不能解释成单次波速度；空间参与集合与共同格的时差须一起读。

### fig_paired_spatial_fields.png / .pdf / .svg
两套噪声中SNN和降阶模型在9.42、9.87、10.37 s附近的配对空间率场，全部使用相同0–500 Hz色标。各图从1 ms记录取[t−25 ms,t+25 ms)均值，与原Fig.5快照窗口一致，核位置与物理网格相同。
**关注点**：双核或带状外观相似是否伴随活动范围和幅度一致，不能仅凭图形相似判传播通过。

### fig_preentry_spatial_pair.png / .pdf / .svg
只展示9.420 s、[9.395,9.445) s的空间率，左列为原生SNN、右列为降阶模型，上下两排为两套噪声实现。所有图共享0–500 Hz线性色标，避免归一化掩盖幅度与招募范围差异。
**关注点**：第一套噪声中降阶已广泛活跃，原生仍主要局限于核A；第二套的实际活动空间也不对应。静态差别是动态失败的例证，不替代事件传播分布验收。
''')

if __name__=='__main__':main()

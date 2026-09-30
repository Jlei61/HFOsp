"""Independent read-only-source audit of the selected g40 mean full trajectory."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr,wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo'
NATIVE=BASE/'fig5_current_network_z_state_v1'
CAND=BASE/'fig5_spatial_kinetic_full_trajectory_20260916'
OUT=BASE/'fig5_kinetic_acceptance_review_20260916'
OUT.mkdir(exist_ok=True)
GEO=dict(np.load(NATIVE/'approx/coarse_20/geometry.npz'))
WEIGHTS=GEO['count_e'];REGION=np.bincount(GEO['g175'],minlength=3)

def write(path,value):
    def safe(x):
        if isinstance(x,np.ndarray):return safe(x.tolist())
        if isinstance(x,np.generic):return safe(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [safe(v) for v in x]
        return x
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stretches(mask):
    x=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(x==1),np.flatnonzero(x==-1)))

def first(mask,hold):
    return next((int(a) for a,b in stretches(mask) if b-a>=hold),None)

def load():
    data={}
    for seed in (9108401,9108402):
        arrays={k:[] for k in ('field_1ms','regions_1ms','spikes_1ms','slow_time_ms','Z','M')};end=0
        for path in sorted((NATIVE/f'replay/runs/eta0.0005_s{seed}/chunks').glob('*.npz')):
            with np.load(path) as a:
                assert int(a['start_step'])==end;end=int(a['end_step'])
                for k in arrays:arrays[k].append(a[k])
        assert end==125000
        row={k:np.concatenate(v) for k,v in arrays.items()};row['regions_1ms']=row['regions_1ms'][:,:3]
        data[f'SNN_{seed}']=row
    with np.load(CAND/'run/trajectory.npz') as a:
        data['candidate_9108401']={k:a[k] for k in ('field_1ms','regions_1ms','spikes_1ms','slow_time_ms','Z','M')}
    for row in data.values():
        assert row['field_1ms'].shape==(12500,400)
        assert np.array_equal(row['field_1ms'].sum(1),row['spikes_1ms'][:,0])
        assert np.array_equal(row['regions_1ms'].sum(1),row['spikes_1ms'][:,0])
        row['rate']=row['spikes_1ms'][:,0].reshape(-1,10).sum(1)/WEIGHTS.sum()/.01
        row['field10']=row['field_1ms'].reshape(-1,10,400).sum(1)/WEIGHTS/.01
        field=np.c_[row['field_1ms']/WEIGHTS/.001,row['regions_1ms']/REGION/.001]
        cs=np.vstack([np.zeros((1,403)),np.cumsum(field,axis=0)])
        hi=np.arange(len(field))+1;lo=np.maximum(0,hi-5)
        row['trailing5']=(cs[hi]-cs[lo])/(hi-lo)[:,None]
    return data

def events(row,window):
    lo,hi=map(lambda x:round(x*100),window);rate=row['rate'][lo:hi]
    quiet=[(a,b) for a,b in stretches(rate<5) if b-a>=2]
    result=[]
    for (_,a),(b,_) in zip(quiet[:-1],quiet[1:]):
        if b-a<2 or rate[a:b].max()<20:continue
        aa,bb=(lo+a)*10,(lo+b)*10;sm=row['trailing5']
        left=np.any(sm[max(0,aa-10):aa]>=50,axis=0)
        arrival=np.full(403,np.nan)
        for cell in range(403):
            if left[cell]:continue
            index=first(sm[aa:bb,cell]>=50,5)
            if index is not None:arrival[cell]=index
        early=(row['regions_1ms'][aa:min(aa+30,bb)]/REGION/.001).mean(0)
        label='A' if early[0]>2*early[1] else ('B' if early[1]>2*early[0] else 'both')
        valid=np.isfinite(arrival[:400]);lag=float(arrival[401]-arrival[400]) if np.isfinite(arrival[400:402]).all() else None
        result.append(dict(start_s=(lo+a)*.01,end_s=(lo+b)*.01,duration_ms=(b-a)*10,
            peak_hz=float(rate[a:b].max()),early_core=label,B_minus_A_ms=lag,
            recruited_E_fraction=float(WEIGHTS[valid].sum()/WEIGHTS.sum()),
            left_censored_E_fraction=float(WEIGHTS[left[:400]].sum()/WEIGHTS.sum()),
            arrival_ms=arrival[:400],core_arrivals_ms=arrival[400:402]))
    return result

def summary(row,window):
    ev=events(row,window);lo,hi=map(lambda x:round(x*100),window)
    iei=np.diff([e['start_s'] for e in ev])*1000
    durations=[e['duration_ms'] for e in ev];lag=[e['B_minus_A_ms'] for e in ev if e['B_minus_A_ms'] is not None]
    med=lambda x:float(np.median(x)) if len(x) else None
    return dict(window_s=window,events=ev,event_count=len(ev),quiet_fraction=float(np.mean(row['rate'][lo:hi]<5)),
        mean_E_hz=float(row['rate'][lo:hi].mean()),duration_median_ms=med(durations),
        duration_IQR_ms=np.quantile(durations,[.25,.75]).tolist() if durations else None,
        peak_median_hz=med([e['peak_hz'] for e in ev]),IEI_median_ms=med(iei),
        IEI_CV=float(iei.std()/iei.mean()) if len(iei)>1 else None,
        early_core_counts=dict(Counter(e['early_core'] for e in ev)),
        dual_core_arrival_events=len(lag),B_minus_A_median_ms=med(lag),
        core_lead_counts=dict(A=sum(l>0 for l in lag),B=sum(l<0 for l in lag),tie=sum(l==0 for l in lag)),
        recruited_E_fraction_median=med([e['recruited_E_fraction'] for e in ev]))

def compare(one,two):
    pairs=[]
    for i,a in enumerate(one['events']):
        for j,b in enumerate(two['events']):
            if a['early_core']!=b['early_core']:continue
            x,y=np.array(a['arrival_ms']),np.array(b['arrival_ms']);joint=np.isfinite(x)&np.isfinite(y);union=np.isfinite(x)|np.isfinite(y)
            rho=None;mae=None
            if joint.sum()>=5:
                delta=y[joint]-x[joint];mae=float(np.median(abs(delta-np.median(delta))))
                if len(np.unique(x[joint]))>1 and len(np.unique(y[joint]))>1:rho=float(spearmanr(x[joint],y[joint]).statistic)
            pairs.append(dict(event1=i,event2=j,early_core=a['early_core'],joint_cells=int(joint.sum()),
                IoU=float(WEIGHTS[joint].sum()/WEIGHTS[union].sum()) if union.any() else None,
                arrival_rho=rho,offset_removed_error_ms=mae))
    def med(key):
        v=[p[key] for p in pairs if p[key] is not None and np.isfinite(p[key])];return float(np.median(v)) if v else None
    return dict(conditioned_pairs=len(pairs),IoU=med('IoU'),arrival_rho=med('arrival_rho'),
        offset_removed_error_ms=med('offset_removed_error_ms'),pairs=pairs,
        scope='All cross-event pairs of the same early-core category, no best-pair selection; dependent pairs, descriptive only.')

def transition(row):
    at=first(row['rate']>=200,20);assert at is not None
    onset=at*.01;last=max(b*.01 for a,b in stretches(row['rate']<5) if b-a>=2 and b*.01<onset)
    before=np.flatnonzero(row['slow_time_ms']<onset*1000-1e-6)[-1]
    aligned=slice(at+50,at+150)
    return dict(high_onset_s=onset,last_quiet_end_s=last,last_quiet_to_high_ms=(onset-last)*1000,
        resource_time_s=float(row['slow_time_ms'][before]/1000),D_before_onset=float(1-row['Z'][before,0]),
        M_before_onset=float(row['M'][before,0]),
        late_mean_E_hz=float(row['rate'][1150:1250].mean()),
        late_occupied_E_fraction=float(np.average(row['field10'][1150:1250]>=50,weights=WEIGHTS,axis=1).mean()),
        aligned_high_window_s=[onset+.5,onset+1.5],aligned_high_mean_E_hz=float(row['rate'][aligned].mean()),
        aligned_high_occupied_E_fraction=float(np.average(row['field10'][aligned]>=50,weights=WEIGHTS,axis=1).mean()))

def plot(data,stats,trans):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    keys=['SNN_9108401','candidate_9108401'];colors=['#252525','#c06b28'];labels=['Native SNN','Spatial candidate']
    fig,axes=plt.subplots(3,2,figsize=(11.5,7.2));fig.subplots_adjust(hspace=.35,wspace=.29,bottom=.10,top=.95)
    for key,c,label in zip(keys,colors,labels):
        row=data[key];t=np.arange(1250)*.01+.005
        axes[0,0].plot(t,row['rate'],c=c,lw=.7,label=label)
        axes[1,0].plot(row['slow_time_ms']/1000,1-row['Z'][:,0],c=c,lw=1.2)
        axes[2,0].plot(row['slow_time_ms']/1000,.0005*row['M'][:,0],c=c,lw=1.2)
        ev=stats['early_interictal'][key]['events']
        for ax,values,xlabel in [(axes[0,1],[e['duration_ms'] for e in ev],'Event duration (ms)'),
            (axes[1,1],np.diff([e['start_s'] for e in ev])*1000,'Onset interval (ms)'),
            (axes[2,1],[e['recruited_E_fraction'] for e in ev],'Recruited E fraction')]:
            a=np.sort(values);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',color=c,lw=1.5);ax.set(xlabel=xlabel,ylabel='Empirical CDF',ylim=(0,1.03))
    axes[0,0].set(ylabel='Global E rate (Hz)',ylim=(0,500));axes[0,0].legend(frameon=False)
    axes[1,0].set_ylabel(r'$D=1-\langle Z_E\rangle$');axes[2,0].set_ylabel(r'$\eta_M\langle M_E\rangle$ (mV)')
    for ax in axes[:,0]:ax.set(xlim=(0,12.5),xlabel='Time (s)')
    for i,ax in enumerate(axes.ravel()):ax.text(-.14,1.04,'ABCDEF'[i],transform=ax.transAxes,fontsize=15,weight='bold')
    for ext in ('png','pdf'):fig.savefig(dest/f'fig_full_trajectory_correspondence.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Every complete event contributes; no selected favorable propagation movie.
    fig,axes=plt.subplots(1,3,figsize=(10,3.4));fig.subplots_adjust(left=.07,right=.88,bottom=.20,top=.84,wspace=.25)
    for ax,key,label in zip(axes,keys+['SNN_9108402'],labels+['Native SNN, noise 2']):
        ev=stats['early_interictal'][key]['events'];p=np.mean([np.isfinite(e['arrival_ms']) for e in ev],axis=0)
        im=ax.imshow(p.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=1,cmap='magma')
        for xy in GEO['centers_mm']:ax.add_patch(Circle(xy,1.5,fc='none',ec='#29cfcc',lw=1))
        ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20]);ax.text(.5,1.05,label,ha='center',transform=ax.transAxes)
    axes[0].set_ylabel('y (mm)');fig.colorbar(im,cax=fig.add_axes([.91,.22,.016,.55]),label='Recruitment probability / event',ticks=[0,.5,1])
    for ext in ('png','pdf'):fig.savefig(dest/f'fig_interictal_recruitment_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (dest/'README.md').write_text('''### fig_full_trajectory_correspondence.png / .pdf
左列直接对照同一外部随机输入下原 SNN 与候选从零到 12.5 s 的全局放电率、D 和有效 M 电流。右列为 0.5–8 s 全部完整自限事件的时长、起始间隔和招募范围经验分布，事件并非独立网络重复。**关注点**：事件数接近是否同时伴随时长/范围接近，以及慢变量积累和进入时刻的偏差；待用户人工验图。

### fig_interictal_recruitment_comparison.png / .pdf
三图分别是原 SNN、同输入候选和第二条原 SNN 输入下，0.5–8 s 各格被完整事件招募的比例。每个事件均纳入，使用 5 ms 尾随率达到 50 Hz 并持续 5 ms 的统一规则，已活跃格左删失。**关注点**：候选的二维支持是否与原生事件族相容；这不是传播速度或同步程度的证明。
''')

def main():
    data=load();windows=dict(early_interictal=[.5,8.],early=[.5,3.],middle=[3.,6.],late_interictal=[6.,8.],pre_entry=[8.,9.42])
    stats={name:{key:summary(row,window) for key,row in data.items()} for name,window in windows.items()}
    comparisons={name:dict(native_candidate=compare(v['SNN_9108401'],v['candidate_9108401']),
        native_noise_baseline=compare(v['SNN_9108401'],v['SNN_9108402'])) for name,v in stats.items()}
    trans={key:transition(row) for key,row in data.items()}
    # Arrival across each model's own persistent-entry window, not a single wave.
    entry={}
    for key,row in data.items():
        lo=round(trans[key]['last_quiet_end_s']*1000);hi=min(12500,round((trans[key]['high_onset_s']+.2)*1000));sm=row['trailing5'];arr=np.full(400,np.nan)
        left=np.any(sm[max(0,lo-10):lo,:400]>=50,axis=0)
        for cell in range(400):
            if left[cell]:continue
            i=first(sm[lo:hi,cell]>=50,20)
            if i is not None:arr[cell]=i
        entry[key]=dict(window_s=[lo/1000,hi/1000],arrival_ms=arr,left_censored_cells=int(left.sum()))
    entry_compare={}
    for label,key in [('native_candidate','candidate_9108401'),('native_noise_baseline','SNN_9108402')]:
        arr1,arr2=entry['SNN_9108401']['arrival_ms'],entry[key]['arrival_ms'];valid=np.isfinite(arr1)&np.isfinite(arr2);union=np.isfinite(arr1)|np.isfinite(arr2)
        d=arr2[valid]-arr1[valid]
        entry_compare[label]=dict(IoU=float(WEIGHTS[valid].sum()/WEIGHTS[union].sum()),joint_cells=int(valid.sum()),
            arrival_rho=float(spearmanr(arr1[valid],arr2[valid]).statistic),offset_removed_error_ms=float(np.median(abs(d-np.median(d)))),
            scope='First sustained local recruitment in separately onset-aligned windows. Multi-burst recruitment, not wave speed.')
    metadata=json.loads((CAND/'figure_metadata.json').read_text());v=stats['early_interictal']
    assert v['candidate_9108401']['event_count']==metadata['interictal_statistics']['early_interictal']['candidate']['finite_events']
    assert v['SNN_9108401']['event_count']==metadata['interictal_statistics']['early_interictal']['native']['finite_events']
    results=dict(identity=dict(candidate='g40_mean',q_ie=1.,particles=40000,Z='dynamic',M='dynamic',compute_grid_mm=.5,observation_grid_mm=1.),
        windows=stats,propagation_comparisons=comparisons,transition=trans,entry_arrivals=entry,entry_comparison=entry_compare,
        statistical_unit='One complete candidate trajectory, two native reference input realizations; all are development data. No equivalence-test tolerance established.',
        qa=dict(independently_reproduced_event_counts=True,all_count_conservation=True))
    write(OUT/'independent_comparison.json',results)
    plot(data,stats,trans)
    compact={name:{key:{k:v for k,v in row.items() if k not in ('events',)} for key,row in rows.items()} for name,rows in stats.items()}
    print(json.dumps(dict(windows=compact,transitions=trans,propagation={name:{k:{kk:vv for kk,vv in x.items() if kk!='pairs'} for k,x in row.items()} for name,row in comparisons.items()},entry=entry_compare),indent=2),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Show paired noise realizations behind the completed M-on Z scan endpoints."""
from pathlib import Path
import io
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/topic4_sef_hfo/m_on_z_kinetics_20260912'
OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/Z_seed_sensitivity'


def edges(values,log=False):
    a=np.log(values) if log else np.asarray(values)
    result=np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]
    return np.exp(result) if log else result


def main():
    OUT.mkdir(exist_ok=True)
    blob=(SOURCE/'analysis_arrays.npz').read_bytes()
    with np.load(io.BytesIO(blob)) as a:arrays={k:a[k] for k in a.files}
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    tau,thresholds=arrays['tau_s'],arrays['thresholds']
    times,observed=arrays['run_restricted_time_s'],arrays['observed']
    assert times.shape==observed.shape==(7,7,3)
    finite=np.isfinite(times);assert np.array_equal(finite,np.isfinite(observed))
    assert np.all(times[finite & (observed==0)]==180)
    assert np.all(np.isnan(arrays['restricted_mean_s'][arrays['completed_count']<3]))
    assert tau[3]==5 and thresholds[3]==protocol['thresholds'][3]
    # Fixed central slices are selected from the declared baseline, not for appearance.
    comparisons=[]
    for direction in ['tau_Z','I_th']:
        for si,seed in enumerate(protocol['seeds']):
            v=times[3,:,si] if direction=='tau_Z' else times[:,3,si]
            o=observed[3,:,si] if direction=='tau_Z' else observed[:,3,si]
            params=tau if direction=='tau_Z' else thresholds
            for k in range(6):
                if not (np.isfinite(v[k]) and np.isfinite(v[k+1])):continue
                comparisons.append(dict(parameter=direction,seed=seed,low=float(params[k]),high=float(params[k+1]),
                    restricted_time_low_s=float(v[k]),restricted_time_high_s=float(v[k+1]),
                    both_observed=bool(o[k] and o[k+1]),both_censored=bool(not o[k] and not o[k+1]),
                    difference_s=float(v[k+1]-v[k]),
                    meaning='Difference of min(T_confirmation,180); exact latency difference only when both entries observed.'))
    summary=dict(status='COMPLETE_SNAPSHOT',source=str(SOURCE/'analysis_arrays.npz'),
        source_sha256=hashlib.sha256(blob).hexdigest(),completed=int(finite.sum()),
        complete_cells=int((finite.sum(-1)==3).sum()),total_runs=147,total_cells=49,
        seeds=protocol['seeds'],fixed_M={'eta_M':.02,'tau_M_s':2},
        fixed_tau_5s_restricted_times=times[:,3].tolist(),fixed_tau_5s_observed=observed[:,3].tolist(),
        fixed_Ith_baseline_restricted_times=times[3].tolist(),fixed_Ith_baseline_observed=observed[3].tolist(),
        adjacent_baseline_slice_comparisons=comparisons,
        statistical_unit='Three paired stochastic trajectories per parameter setting; bins, grid points and events are not independent replicates.',
        missing_policy='Gray/pending is not censoring. A censor marker requires a completed180s trajectory.',
        conclusion='The observed finite-horizon response has noise dependence and nonmonotonic individual-seed changes. No universal monotonic timing law is established.',
        limitations='Partial scan; three seeds only. No patient validation, bifurcation classification, autonomous termination or reset recurrence claim.',
        human_review='PENDING',agent_visual_review='PENDING')
    # Strict JSON: unfinished entries remain explicit nulls.
    def safe(value):
        if isinstance(value,list):return [safe(v) for v in value]
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    (OUT/'analysis.json').write_text(json.dumps(safe(summary),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'analysis_arrays_snapshot.npz').write_bytes(blob)
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':17,'pdf.fonttype':42})
    fig=plt.figure(figsize=(17,10));grid=fig.add_gridspec(2,1,height_ratios=[1,1],hspace=.34)
    top=grid[0].subgridspec(1,3,wspace=.32);bottom=grid[1].subgridspec(1,2,wspace=.23)
    axes=[fig.add_subplot(top[i]) for i in range(3)]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e0e0e0')
    xe,ye=edges(tau,True),edges(thresholds)
    for si,ax in enumerate(axes):
        im=ax.pcolormesh(xe,ye,np.ma.masked_invalid(times[:,:,si]),cmap=cmap,vmin=0,vmax=180,
            edgecolors=(1,1,1,.3),linewidth=.5,shading='flat')
        for y,x in zip(*np.where(finite[:,:,si]&(observed[:,:,si]==0))):
            ax.add_patch(Rectangle((xe[x],ye[y]),xe[x+1]-xe[x],ye[y+1]-ye[y],
                facecolor='none',hatch='////',edgecolor='#585858',linewidth=0))
        ax.plot(5,thresholds[3],marker='o',mfc='none',mec='white',ms=8,mew=1.5)
        ax.set_xscale('log');ax.set_xticks([2.5,5,10],['2.5','5','10']);ax.minorticks_off()
        ax.set(xlabel='τZ (s)',ylabel='Depletion threshold (mV equiv.)' if si==0 else '',
            yticks=[75,95.1985131267,120],yticklabels=['75','95.2','120'],title=f'Noise seed {si+1}')
    fig.subplots_adjust(left=.075,right=.88,bottom=.12,top=.94)
    cb=fig.add_axes([.90,.58,.018,.34]);fig.colorbar(im,cax=cb).set_label('Restricted entry time (s)')
    colors=['#277da8','#d45c38','#7553a3']
    for dim in range(2):
        ax=fig.add_subplot(bottom[dim]);params=tau if dim==0 else thresholds
        values=times[3] if dim==0 else times[:,3];flags=observed[3] if dim==0 else observed[:,3]
        for si,color in enumerate(colors):
            v=values[:,si];o=flags[:,si];ok=np.isfinite(v)
            ax.plot(params,v,color=color,alpha=.75,lw=1.4)
            q=ok&(o==1);ax.scatter(params[q],v[q],c=color,s=43,zorder=3)
            q=ok&(o==0);ax.scatter(params[q],v[q],edgecolors=color,facecolors='none',marker='^',s=95,lw=1.6,zorder=3)
        if dim==0:
            ax.set_xscale('log');ax.set_xticks([2.5,5,10],['2.5','5','10']);ax.minorticks_off()
        ax.set(ylim=[0,191],yticks=[0,60,120,180],xlabel='τZ (s)' if dim==0 else 'Depletion threshold (mV equiv.)',
            ylabel='Restricted entry time (s)',title='Ith = 95.2 mV equiv.' if dim==0 else 'τZ = 5 s')
        ax.grid(alpha=.18)
    handles=[Line2D([],[],color=c,marker='o',label=f'Seed {i+1}') for i,c in enumerate(colors)]
    handles += [Line2D([],[],color='#333333',marker='^',mfc='none',ls='none',label='No entry by 180 s'),
        Patch(fc='#e0e0e0',ec='none',label='Still running / pending')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.49,.01),ncol=5,frameon=False,fontsize=13)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'paired_noise_response.png',dpi=170);fig.savefig(folder/'paired_noise_response.pdf');plt.close(fig)
    (folder/'README.md').write_text('### paired_noise_response.png / .pdf\n'
        '上排分别显示三个配对噪声种子的Z参数响应面，下排是预先定义基线τZ=5秒与Ith=95.2的两个切片。圆点表示已观测进入，空心三角及斜线格表示真实完成180秒后仍未进入；灰色保留尚未完成的位置。\n'
        '**关注点**：这里只比较首次进入的有限窗响应，不把删失当作永不进入；连接线展示限制时间，不是连续参数拟合或分岔曲线。\n')
    lines=['# 配对噪声与Z动力学参数的关系','',
        f'本次快照完成{int(finite.sum())}/147条、{int((finite.sum(-1)==3).sum())}/49个完整格。所有曲线使用固定M强度0.02、τM=2秒，无人工reset。','',
        '基线切片在看结果前由原协议定义，未根据最佳表现选点。三个种子跨参数共享外部输入；每一参数下只有三个随机实现，不能用格点数增加样本量。','',
        '在τZ=5秒时，Ith=75的三个种子均进入（47.23、55.31、24.17秒）；Ith=95.2为73.68、136.80和180秒删失；Ith=103.47及以上在此切片中均180秒删失。这个范围支持阈值影响有限时间内的进入机会与等待。',
        '但81.73/88.47与95.2之间可见单种子的非单调变化，τZ切片也不是一致上升。因此目前不能把F画成预期的平滑单调色带，更不能说全部条件必然进入、只差早晚。',
        'Ith是原始GABA电流越过后启动Z耗竭的门槛；τZ同时改变恢复和耗竭速度。它们不是参考示意图里的独立耗竭强度与独立恢复时间常数。',
        '严格结论：该原生SNN的首次进入对Z参数和噪声实现敏感，现有有限窗样本尚不支持普适的单调时间关系。随后完整147条与M40的F应分别报告，不能混合为同一参数扫描。']
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':int(finite.sum()),'complete_cells':int((finite.sum(-1)==3).sum()),'figure':str(folder/'paired_noise_response.png')}))


if __name__=='__main__':main()

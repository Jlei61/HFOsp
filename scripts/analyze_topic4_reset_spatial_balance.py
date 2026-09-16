#!/usr/bin/env python3
"""Compare observed pre-entry and late post-reset states without causal claims."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo'
OUT=BASE/'fig5_m_overnight_exploration_20260913/reset_spatial_balance'
SOURCE=BASE/'fig5_manual_core_release_v1/m_runaway_return_v1/runs/weak_fast.npz'
LATE=BASE/'reset_state_diagnosis_20260911/runs/z_only_long'


def describe(z,m,inputs,field,ncell,dt):
    return dict(mean_Z=float(z[:,0].mean()),mean_Z_core_A_B_surround=z[:,5:8].mean(0).tolist(),
        mean_across_neuron_Z_SD=float(z[:,1].mean()),
        temporal_Z_10_50_90=np.quantile(z[:,[0,5,6,7]],[.1,.5,.9],axis=0).tolist(),
        mean_M_current=m.mean(0).tolist(),mean_raw_GABA_above_threshold_fraction=float(z[:,8].mean()),
        predicted_mean_dZ_per_s=float((1-z[:,8]-z[:,0]).mean()/5),
        mean_E_rate=float(field.sum()/ncell.sum()/dt),
        mean_input_OU_Erate_Irate=inputs[:,1:].mean(0).tolist())


def main():
    OUT.mkdir(exist_ok=True)
    # Fixed before comparison: exclude startup and the first local high entry at72.26s.
    early_window=[10.,70.];late_window=[970.,1000.]
    with np.load(SOURCE) as a:
        t=a['z_time_ms']/1000;q=(t>=early_window[0])&(t<early_window[1])
        ez=a['z_stats'][q];em=a['m_stats'][q][:,[0,5,6,7]]*.02
        ezfield=a['z_field_5ms'][q].mean(0)
        ec=a['field_e_count_1ms'][10000:70000].sum(0,dtype=np.uint64)
        ncell=a['cell_e_counts'];centers=a['centers_mm'];xy=a['positions_e'];cell=a['cell_e']
        inputs=a['input_summary'];ei=inputs[(inputs[:,0]>=10000)&(inputs[:,0]<70000)]
    assert np.array_equal(np.bincount(cell,minlength=400),ncell)
    assert xy.shape==(32000,2) and ncell.sum()==32000
    collections={key:[] for key in ['Z','M','Z_field','inputs']};lc=np.zeros(400,np.uint64);paths=[];nslow=0
    for path in sorted((LATE/'chunks').glob('*.npz')):
        lo,hi=map(int,path.stem.split('_'))
        if hi<=9700000 or lo>=10000000:continue
        with np.load(path) as a:
            t=a['slow_time_ms']/1000;q=(t>=970)&(t<1000)
            for key in ['Z','M','Z_field']:collections[key].append(a[key][q])
            inputs=a['inputs'];collections['inputs'].append(inputs[(inputs[:,0]>=970000)&(inputs[:,0]<1000000)])
            ft=a['field_time_ms']/1000;fq=(ft>=970)&(ft<1000)
            lc+=a['field_5ms'][fq].sum(0,dtype=np.uint64);nslow+=q.sum();paths.append(str(path))
    assert nslow==6000
    lz,lm,lzf,li=[np.concatenate(collections[k]) for k in ['Z','M','Z_field','inputs']]
    assert np.isclose(np.average(ezfield,weights=ncell),ez[:,0].mean(),atol=1e-10)
    assert np.isclose(np.average(lzf.mean(0),weights=ncell),lz[:,0].mean(),atol=1e-10)
    lzfield=lzf.mean(0);erate=ec/ncell/60; lrate=lc/ncell/30
    early=describe(ez,em,ei,ec,ncell,60);late=describe(lz,lm*.02,li,lc,ncell,30)
    summary=dict(status='COMPLETE',comparison='Two descriptive windows from one original noise realization; no independent replication or causal intervention comparison.',
        source_native=str(SOURCE),source_late_chunks=paths,early_window_s=early_window,late_window_s=late_window,
        first_native_global_entry_s=73.48,external_Z_refill_s=[75.5,76.5],
        postrelease_followup_s=923.5,second_entry_observed=False,
        early=early,late=late,
        Z_map_weighted_RMS_difference=float(np.sqrt(np.average((lzfield-ezfield)**2,weights=ncell))),
        rate_map_weighted_RMS_difference_Hz=float(np.sqrt(np.average((lrate-erate)**2,weights=ncell))),
        conclusion='Similar coarse state distributions do not establish equal full network states, identical event probabilities, or permanent reset-induced protection.',
        human_review='PENDING',agent_visual_review='PENDING')
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(OUT/'spatial_fields.npz',early_Z=ezfield,late_Z=lzfield,
        early_E_rate=erate,late_E_rate=lrate,cell_E_count=ncell,centers_mm=centers)
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':16,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained',sharex=True,sharey=True)
    titles=['Before first entry\n10–70 s','Late after Z refill\n970–1000 s','Late − before']
    arrays=[[ezfield,lzfield,lzfield-ezfield],[erate,lrate,lrate-erate]]
    for row in range(2):
        for col in range(3):
            ax=axs[row,col]
            if col==2:
                bound=.05 if row==0 else 10
                kwargs=dict(cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
            else:kwargs=dict(cmap='viridis' if row==0 else 'magma',vmin=.65 if row==0 else 0,vmax=1 if row==0 else 50)
            im=ax.imshow(arrays[row][col].reshape(20,20),origin='lower',extent=[0,20,0,20],interpolation='nearest',**kwargs)
            for i,center in enumerate(centers):
                ax.add_patch(Circle(center,1.5,fc='none',ec='#40dfce',lw=1.5))
                ax.text(center[0],center[1]+2,'AB'[i],ha='center',color='#137b75',fontweight='bold',bbox=dict(fc='white',alpha=.8,ec='none',pad=.5))
            if row==0:ax.set_title(titles[col])
            ax.set(xticks=[0,10,20],yticks=[0,10,20])
            if col==0:ax.set_ylabel('y (mm)')
            if row==1:ax.set_xlabel('x (mm)')
            if col==2:
                fig.colorbar(im,ax=ax,shrink=.85).set_label('ΔZ' if row==0 else 'Δ E rate (Hz)')
            elif col==1:
                fig.colorbar(im,ax=list(axs[row,:2]),shrink=.85).set_label('Mean Z' if row==0 else 'Mean E rate (Hz)')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'before_and_after_reset_fields.png',dpi=170);fig.savefig(folder/'before_and_after_reset_fields.pdf');plt.close(fig)
    (folder/'README.md').write_text('### before_and_after_reset_fields.png / .pdf\n'
        '比较同一噪声实现第一次高态前10–70秒，与外部Z补充后970–1000秒的原生空间Z与E放电率；两行使用相同1mm分箱和真实1.5mm双核边界。右列为末段减去前段的差值，色标围绕零点对称。\n'
        '**关注点**：两段时长不同，均为描述性平均；接近的平均场不代表完整状态相同，也不能单独解释再次进入的概率。\n')
    report=['# Reset前后是否是不同的粗空间状态','',
        '问题：原轨迹第一次高态前是否本来就处于Z收支近似平衡，而不是一直单调耗竭？这里固定比较10–70秒与970–1000秒，不按结果寻找最像的窗口。',
        '统计单位仍是一条噪声实现。细胞、空间分箱和连续时间采样均不能充当重复实验。','',
        '| 读出 | 第一次进入前 | Reset后末段 |','|---|---:|---:|']
    for key,label in [('mean_Z','平均Z'),('mean_E_rate','平均E放电率Hz'),('mean_raw_GABA_above_threshold_fraction','耗竭阈值以上E比例'),('predicted_mean_dZ_per_s','平均Z导数每秒')]:
        report.append(f'| {label} | {early[key]:.6f} | {late[key]:.6f} |')
    report+=['',f'空间Z的细胞数加权RMS差为{summary["Z_map_weighted_RMS_difference"]:.6f}。'
        '这类平均比较检验粗状态差异，但不会检出全部瞬时突触、延迟、局部M及噪声历史；不能据此证明两个状态在动力学上等价。',
        '原生Z在首次进入前可以长期处于大致收支平衡，随后仍进入高态。因此仅凭reset后Z处于平台，不能推出网络从此不能再次进入；应继续读取同噪声M清零、完整快状态清零和长窗结果。',
        '另一方面，1000秒Z-only未再进入依然是实际阴性结果，不能因前段曾进入就视作已满足Fig.5第⑤状态。']
    (OUT/'scientific_review.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()

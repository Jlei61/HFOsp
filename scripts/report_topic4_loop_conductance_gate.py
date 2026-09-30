#!/usr/bin/env python3
"""Freeze the failed local-response gate; no validation-selected refit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from calibrate_topic4_loop_conductance_static import OUT, StaticCorrection, base_features


def main():
    result=json.loads((OUT/'result.json').read_text())
    with np.load(OUT/'validation_predictions_locked.npz') as a:
        pars=a['pars'];g=a['g']
    with np.load(OUT/'validation_scored.npz') as a:
        data={k:a[k] for k in a.files}
    net=StaticCorrection().double().eval()
    saved=torch.load(OUT/'locked_model.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(saved['model'])
    f0,b0=base_features(pars,np.zeros(len(g)))
    with torch.no_grad():
        observed=net(torch.tensor(f0),torch.tensor(b0)).numpy()
        parent=(500*torch.sigmoid(torch.tensor(b0)+np.log(20.))).numpy()
    assert np.array_equal(observed,parent)
    (OUT/'zero_conductance_physical_input_qa.json').write_text(json.dumps(dict(status='PASS',
        recomputed_zero_conductance_features_and_base_logits=True,parent_output_bitwise=True,total=len(g)),indent=2)+'\n')
    failures=[]
    for i in np.flatnonzero(~data['within_cap']):
        failures.append(dict(index=int(i),g_over_gL=float(g[i]),mean_effective_mV=float(pars[i,0]),
            effective_sigma_E=float(np.sqrt(pars[i,2])),effective_sigma_I=float(np.sqrt(pars[i,3])),
            native_MC_mean_Hz=float(data['mean_Hz'][i]),native_MC_SEM_Hz=float(data['SEM_Hz'][i]),
            approximation_Hz=float(data['predicted_Hz'][i])))
    summary=dict(status='REJECTED_FOR_BIFURCATION_THIS_ROUND',static=result,
        broad_cap_failures=failures,validation_selected_refit=False,
        reason='Two independent low-rate, noise-driven test conditions exceed the prespecified broad error cap. Such errors can alter the recovery-state interpretation.',
        transient_validation='NOT_RUN_STATIC_GATE_FAILED',native_spatial_correspondence='NOT_RUN_STATIC_GATE_FAILED',
        formal_continuation='NOT_AUTHENTICATED',
        fallback='Complete original native SNN conditional Z/K states and matched spatial-axis controls. Keep bifurcation type unknown; do not promote unrelated older branches.',
        limitation='A failed finite calibration is not proof that no conductance-aware rate reduction is possible.')
    (OUT/'gate_review.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 电导响应闭合：本轮不进入正式分岔认证','','原问题是原生SNN怎样在同一底物上进入高活动，再由反馈和资源恢复返回间期。率模型只有先能对应这一动力学，才可用它命名分岔；静态局部响应是第一道必要检查。','',
        '简单膜时间压缩在新增电导条件只通过19/36点。一次独立训练的电导修正将验证提高到241/256点，但预定宽误差上限仍有2点不通过，因此整体不通过，未开展后续瞬态或全空间率模型认证。训练和验证噪声、设计均分开；没有用验证结果回调模型或放宽阈值。','',
        '两处反证都属于有电导时的低平均驱动、噪声触发放电：', '',
        '|g/gL|MC均值±SEM Hz|预测Hz|','|---|---|---|']
    for r in failures:
        lines.append(f"|{r['g_over_gL']:.2f}|{r['native_MC_mean_Hz']:.2f}±{r['native_MC_SEM_Hz']:.2f}|{r['approximation_Hz']:.2f}|")
    lines += ['', '这类误差会影响低活动和恢复状态判断，不能因为多数高率点拟合良好就忽略。结果只否定本次近似的认证资格，并未证明率模型原则上不可行。下一步本轮完成原生条件状态、自然漂移和匹配空间轴对照；正式SN/Hopf/周期分支类型保持未定，不以旧的不相关不稳定分支替代。', '',
        '统计单位：256个独立于训练的合成输入条件，每点1024个局部彩色LIF噪声重复；这不是256条空间SNN或患者样本。MC与原生膜更新、重置和不应期顺序一致，突触噪声为既有高斯滤波近似。g=0重新构造物理输入后仍逐位保留父模型。']
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    mean=data['mean_Hz'];pred=data['predicted_Hz'];bad=~data['within_cap']
    im=axs[0].scatter(mean,pred,c=g,cmap='viridis',s=12,vmin=0,vmax=32)
    axs[0].plot([0,500],[0,500],color='grey',lw=.8)
    axs[0].scatter(mean[bad],pred[bad],facecolors='none',edgecolors='#d62728',s=80,lw=1.4)
    axs[0].set(xlabel='Independent colored-LIF MC rate (Hz)',ylabel='Approximation (Hz)',xlim=(-8,508),ylim=(-8,508))
    axs[1].scatter(mean,pred-mean,c=g,cmap='viridis',s=12,vmin=0,vmax=32)
    axs[1].scatter(mean[bad],(pred-mean)[bad],facecolors='none',edgecolors='#d62728',s=80,lw=1.4)
    x=np.linspace(0,500,200);tol=np.maximum(2,.1*x)
    axs[1].plot(x,tol,c='grey',ls='--',lw=.7);axs[1].plot(x,-tol,c='grey',ls='--',lw=.7)
    axs[1].axhline(0,c='grey',lw=.6)
    axs[1].set(xlabel='Independent MC rate (Hz)',ylabel='Prediction − MC (Hz)',xlim=(-8,508))
    fig.colorbar(im,ax=axs,label='Total conductance / gL',shrink=.8)
    fig.suptitle('Static response gate failed: 241/256 within tolerance; 2 exceed broad cap',fontsize=11)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'conductance_static_validation.png',dpi=180)
    fig.savefig(dest/'conductance_static_validation.pdf');plt.close(fig)
    (dest/'README.md').write_text('''### conductance_static_validation.png

独立局部MC验证的均值率和误差；颜色是总电导，红圈是超过预定宽误差上限的两点。虚线仅展示2Hz或10%均值的误差尺度，逐点正式判定还包括MC标准误。该近似本轮不用于正式分岔认证。

**关注点**：低率噪声触发区的偏差，不能用总体拟合观感替代恢复态的准确性；图待人工审阅。
''')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

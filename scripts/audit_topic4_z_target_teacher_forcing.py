#!/usr/bin/env python3
"""Separate direct target averaging error from closed-loop activity divergence."""
from topic4_spatial_boundary_common import OUT,OLD,read,write,checkpoint_path
from topic4_e_only_z_rate import EOnlySystem,native_gaba_variance_factor
from checkpoint import load
from scipy.special import ndtr
import numpy as np
import matplotlib.pyplot as plt
from plot_topic4_spatial_boundary_results import save


def main():
    a=np.load(OUT/'native_boundary_moments.npz');b=np.load(OUT/'closure_audit_arrays.npz')
    src=np.load(OLD/'external_input.npz');s=EOnlySystem();m=s.m;weights=m.count_e;cells=src['cell_e']
    final=load(checkpoint_path(9400))['slow']['z'][:len(cells)]
    zend=np.bincount(cells,weights=final,minlength=100)/weights
    ztrue=np.vstack([b['z_native'],zend]);tau=float(a['tau_z_ms']);alpha=(1-s.dt/tau)**100
    exact_target=(ztrue[1:]-alpha*ztrue[:-1])/(1-alpha)
    # Float32 neuron snapshots only affect this inferred target by ~1e-5.
    assert exact_target.min()>-1e-4 and exact_target.max()<1+1e-4
    aggregation=np.zeros((100,400));aggregation[src['cell_i'],a['cell_i']]=1
    ri=(a['field_i_count_1ms']@aggregation.T)/src['count_i']
    ri_past=np.array([ri[k-50:k].mean(0) for k in range(50,1400,10)])
    variance=native_gaba_variance_factor(s)*ri_past@m.v_ei.T
    target_independent=ndtr((float(a['threshold'])-b['mean_raw_gaba'][5:])/np.sqrt(np.maximum(variance,1e-12)))
    targets={'native_ODE_interval_target':exact_target[5:],
             'native_mean_and_empirical_SD':b['target_empirical_gauss'][5:],
             'threshold_of_native_mean':b['target_mean_indicator'][5:],
             'native_mean_independent_current_variance':target_independent}
    trajectories={};rows=[];times=np.arange(8050,9401,10)
    for name,target in targets.items():
        z=ztrue[5].copy();trace=[z.copy()]
        for p in target:
            z=alpha*z+(1-alpha)*p;trace.append(z.copy())
        trace=np.array(trace);trajectories[name]=trace
        diff=trace-ztrue[5:]
        rows.append({'name':name,'end_mean_Z_error':float(np.average(diff[-1],weights=weights)),
            'end_spatial_Z_RMSE':float(np.sqrt(np.average(diff[-1]**2,weights=weights))),
            'max_absolute_cell_Z_error':float(abs(diff).max())})
    assert np.max(abs(trajectories['native_ODE_interval_target']-ztrue[5:]))<1e-12
    write(OUT/'z_target_teacher_forcing.json',{'status':'COMPLETE','interval_ms':[8050,9400],
        'initial_condition':'Exact measured native cell-mean Z at 8.05 s for every counterfactual.',
        'scope':'Native fast activity/current observations supplied throughout; this isolates target-averaging error along the observed path and excludes feedback of counterfactual Z onto firing. It is not an autonomous model.',
        'native_target':'Interval-weighted target reconstructed from exact native Euler Z law, with float32 recording tolerance.',
        'independent_variance':'Uses preceding 50-ms native I-cell rates and the independent-Poisson impulse variance; this is a variance-closure sensitivity, not exact dynamic current variance.',
        'rows':rows})
    np.savez_compressed(OUT/'z_target_teacher_forcing.npz',time_ms=times,native_z=ztrue[5:],count_e=weights,**trajectories)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels={'native_ODE_interval_target':'Native ODE interval target','native_mean_and_empirical_SD':'Gaussian: native current mean + SD',
        'threshold_of_native_mean':'Threshold of native current mean','native_mean_independent_current_variance':'Gaussian: independent-input variance'}
    for name,trace in trajectories.items():
        axs[0].plot(times/1000,np.average(trace,axis=1,weights=weights),label=labels[name],lw=1.4)
        axs[1].plot(times/1000,np.sqrt(np.average((trace-ztrue[5:])**2,axis=1,weights=weights)),label=labels[name])
    axs[0].set(xlabel='Time (s)',ylabel='Mean E-target Z',title='Native fast activity held fixed');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Time (s)',ylabel='Spatial Z RMSE against native',title='Direct error of the Z-target approximation')
    fig.suptitle('Diagnose Z averaging without allowing activity feedback to amplify the error',fontsize=14)
    save(fig,'z_target_teacher_forcing_diagnostic',
        '所有离线分支从同一个真实 Z 状态开始，持续提供原 SNN 的快速活动和电流信息，比较不同阈值占据平均化对 Z 轨迹的直接影响。原 ODE 区间目标由相邻 Z 记录和真实离散更新系数反推，重构误差小于数值容差。',
        '这项对照阻断了 Z 对放电的反馈，不能当成新的自主预测；它用于区分直接平均化误差与闭环动力学放大。')
    print(rows)


if __name__=='__main__':main()

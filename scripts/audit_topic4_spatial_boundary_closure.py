#!/usr/bin/env python3
"""Measured within-cell averaging errors, with no change to biological parameters."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write
from scipy.special import ndtr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    a=np.load(OUT/'native_boundary_moments.npz'); source=np.load(OLD/'external_input.npz')
    cells=source['cell_e']; weights=source['count_e']; n=len(weights)
    z=a['z_e'].astype(float); ii=a['raw_gaba_e'].astype(float); threshold=float(a['threshold'])
    def coarse(x):return np.array([np.bincount(cells,weights=row,minlength=n)/weights for row in x])
    mu=coarse(ii); var=np.maximum(coarse(ii*ii)-mu*mu,0); zm=coarse(z)
    target=coarse(ii<threshold); empirical_gauss=ndtr((threshold-mu)/np.sqrt(np.maximum(var,1e-12)))
    indicator=(mu<threshold).astype(float); actual_effective=coarse(z*ii); cov=actual_effective-zm*mu
    cells20=a['cell_e'];weights20=a['cell_e_counts']
    def coarse20(x):return np.array([np.bincount(cells20,weights=row,minlength=400)/weights20 for row in x])
    mu20=coarse20(ii);var20=np.maximum(coarse20(ii*ii)-mu20*mu20,0)
    target20=coarse20(ii<threshold);gauss20=ndtr((threshold-mu20)/np.sqrt(np.maximum(var20,1e-12)))
    def avg(x): return np.average(x,axis=-1,weights=weights)
    native=np.load(REFERENCE/'trajectory.npz'); auto=np.load(OLD/'rate/autonomous_gaussian_expected.npz')
    times=a['time_ms']; rate_index=times.astype(int)-1
    diff=auto['z'][rate_index]-zm
    snapshots=[]
    for k in (0,40,80,120):
        snapshots.append({'time_ms':float(times[k]),'native_mean_Z':float(avg(zm[k])),
            'autonomous_rate_mean_Z':float(avg(auto['z'][rate_index[k]])),
            'spatial_Z_rmse':float(np.sqrt(avg(diff[k]**2))),
            'cell_Z_error_range':[float(diff[k].min()),float(diff[k].max())]})
    metrics={
        'native_replay_qa':read(OUT/'replay_status.json')['status'],
        'window_ms':[8000,9400],'sampling_interval_ms':10,'cells':100,
        'statistical_unit':'140 snapshots from one trajectory; cells and times are not independent biological samples.',
        'target_empirical_gaussian_MAE':float(avg(abs(empirical_gauss-target)).mean()),
        'target_empirical_gaussian_bias':float(avg(empirical_gauss-target).mean()),
        'target_mean_indicator_MAE':float(avg(abs(indicator-target)).mean()),
        'target_mean_indicator_bias':float(avg(indicator-target).mean()),
        'mean_Z_current_covariance':float(avg(cov).mean()),
        'mean_absolute_covariance_relative_to_effective_current':float(avg(abs(cov)).mean()/avg(actual_effective).mean()),
        'native_within_cell_variance_grid10':float(avg(var).mean()),
        'native_within_cell_variance_grid20':float(np.average(var20,axis=1,weights=weights20).mean()),
        'within_grid10_current_variance_resolved_by_grid20_fraction':float(1-np.average(var20,axis=1,weights=weights20).mean()/avg(var).mean()),
        'grid20_target_empirical_gaussian_MAE':float(np.average(abs(gauss20-target20),axis=1,weights=weights20).mean()),
        'grid20_target_mean_indicator_MAE':float(np.average(abs((mu20<threshold)-target20),axis=1,weights=weights20).mean()),
        'snapshots':snapshots,
        'interpretation':[
            'Autonomous rate Z is above native early, then becomes lower; a single uniformly too-fast depletion rate is not a sufficient explanation.',
            'Empirical Gaussian diagnostic uses true native current mean and variance; this is a teacher-forced closure audit, not an autonomous prediction or fitted correction.',
            'Small aggregate covariance or target error does not rule out dynamical amplification in a sensitive local region.',
            'Near-equal global Z can conceal different spatial depletion profiles; inspect spatial state before interpreting a scalar phase diagram.'
        ]}
    write(OUT/'closure_audit.json',metrics)
    np.savez_compressed(OUT/'closure_audit_arrays.npz',time_ms=times,z_native=zm,z_rate=auto['z'][rate_index],
        mean_raw_gaba=mu,var_raw_gaba=var,target_native=target,target_empirical_gauss=empirical_gauss,
        target_mean_indicator=indicator,cov_Z_current=cov,count_e=weights)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    fig,axs=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    ax=axs[0,0]; mask=native['z_time_ms']<=10680
    ax.plot(native['z_time_ms'][mask]/1000,native['z_stats'][mask,0],c='k',label='Native SNN')
    ax.plot(np.arange(10680)/1000,avg(auto['z'][:10680]),c='#b94272',label='Autonomous rate Z')
    ax.set(xlabel='Time (s)',ylabel='Mean E-target Z',title='A  Z error reverses sign');ax.legend(fontsize=10)
    ax=axs[0,1]
    for values,label,color in [(target,'Native fraction','#222222'),(empirical_gauss,'Gaussian: native mean + SD','#4b7ba6'),(indicator,'Threshold of cell mean','#cc9035')]:
        ax.plot(times/1000,avg(values),color=color,label=label,lw=1)
    ax.set(xlabel='Time (s)',ylabel='Fraction with raw GABA < threshold',title='B  Audit the target driving Z');ax.legend(fontsize=9)
    ax=axs[1,0]
    for x,label,color in [(empirical_gauss,'Gaussian with native SD','#4b7ba6'),(indicator,'Threshold of mean','#cc9035')]:
        ax.plot(times/1000,avg(abs(x-target))*100,color=color,label=label)
    ax.set(xlabel='Time (s)',ylabel='Weighted absolute target error (pp)',title='C  Measured averaging error');ax.legend(fontsize=9)
    ax=axs[1,1];im=ax.imshow(diff[0].reshape(10,10),origin='lower',extent=[0,20,0,20],cmap='RdBu_r',vmin=-.11,vmax=.11)
    carrier=read(read(REFERENCE/'protocol.json')['carrier'])
    centers=np.array(carrier['candidate']['node_field']['centers_mm'])
    ax.scatter(centers[:,0],centers[:,1],facecolors='none',edgecolors='k',s=100)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='D  Spatial Z error at 8.0 s');fig.colorbar(im,ax=ax,label='Rate Z minus native Z')
    ax=axs[2,0];ax.plot(times/1000,np.sqrt(avg(diff*diff)),color='#b94272')
    ax.set(xlabel='Time (s)',ylabel='Neuron-weighted spatial Z RMSE',title='E  Spatial depletion diverges')
    ax=axs[2,1];ax.plot(times/1000,100*avg(abs(cov))/np.maximum(avg(actual_effective),1e-12),color='#5b6e4c')
    ax.set(xlabel='Time (s)',ylabel='Absolute covariance / effective current (%)',title='F  Lost within-cell Z-current covariance')
    fig.suptitle('Closure audit on the unchanged C substrate | measured native currents',fontsize=15)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    name='native_current_and_z_closure_audit'
    fig.savefig(folder/f'{name}.png',bbox_inches='tight');fig.savefig(folder/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    readme=folder/'README.md';previous=readme.read_text() if readme.exists() else ''
    sections=['### '+part for part in previous.split('### ') if part.strip() and not part.startswith('native_current_and_z_closure_audit')]
    readme.write_text('### native_current_and_z_closure_audit\n\n'
        '比较同一 C 双核底物的原 SNN 与既有自主 rate-Z 的时间及空间差异；8.0–9.4 秒电流来自逐步一致性验证通过的原 SNN 重放。高斯目标诊断使用真实神经元电流均值和方差，因此只检验平均化近似，不能当作自主预测或新的拟合结果。\n\n'
        '**关注点**：平均 Z 的误差先正后负，近似相同的平均 Z 仍可对应不同空间耗竭；D 中圆圈标出本次沿用的上侧双核。PNG/PDF 待用户目视审阅。\n\n'+''.join(sections))
    print(metrics)


if __name__=='__main__':main()

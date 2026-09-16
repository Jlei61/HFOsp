#!/usr/bin/env python3
"""Show the spatial error hidden by a nearly matched mean Z."""
from topic4_spatial_boundary_common import OUT,OLD,REFERENCE,write
from plot_topic4_spatial_boundary_results import save
import numpy as np
import matplotlib.pyplot as plt


def main():
    src=np.load(OLD/'external_input.npz');native=np.load(REFERENCE/'trajectory.npz');w=src['count_e']
    mapping=np.bincount(src['cell_e']*400+native['cell_e'],minlength=40000).reshape(100,400)/w[:,None]
    z=native['z_field_10ms']@mapping.T;times=(6000,7000,7500);rows=[]
    for label,path in [('Previous',OLD/'rate/autonomous_gaussian_expected.npz'),('Mixed',OUT/'mixed_timescale/autonomous_gaussian.npz')]:
        a=np.load(path)
        for tm in times:
            diff=a['z'][tm-1]-z[tm//10]
            rows.append({'model':label,'time_ms':tm,'mean_native_Z':float(np.average(z[tm//10],weights=w)),
                'mean_model_Z':float(np.average(a['z'][tm-1],weights=w)),
                'spatial_Z_RMSE':float(np.sqrt(np.average(diff**2,weights=w))),
                'weighted_fraction_Z_lower_by_more_than_005':float(np.average(diff<-.05,weights=w)),
                'minimum_cell_Z_error':float(diff.min()),'maximum_cell_Z_error':float(diff.max())})
    candidate=np.load(OUT/'mixed_timescale/autonomous_gaussian.npz')['z']
    vmin=np.floor(min(float(z[np.array(times)//10].min()),float(candidate[np.array(times)-1].min()))*20)/20
    fig,axs=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for row,tm in enumerate(times):
        nz=z[tm//10];cz=candidate[tm-1];diff=cz-nz
        for col,(label,field) in enumerate((('Native SNN',nz),('Autonomous corrected rate',cz))):
            im=axs[row,col].imshow(field.reshape(10,10),origin='lower',extent=[0,20,0,20],cmap='viridis',vmin=vmin,vmax=1)
            axs[row,col].set_title(f'{label} | {tm/1000:g} s\nmean Z = {np.average(field,weights=w):.4f}',fontsize=10)
        err=axs[row,2].imshow(diff.reshape(10,10),origin='lower',extent=[0,20,0,20],cmap='coolwarm',vmin=-.12,vmax=.12)
        axs[row,2].set_title(f'Model minus native\nspatial RMSE = {np.sqrt(np.average(diff**2,weights=w)):.4f}',fontsize=10)
        for ax in axs[row]:ax.set(xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axs[:,:2],label='E-target Z',shrink=.7)
    fig.colorbar(err,ax=axs[:,2],label='Z error (blue: more depleted in model)',shrink=.7)
    fig.suptitle('A nearly matched mean Z can conceal a different spatial resource field',fontsize=14)
    save(fig,'autonomous_spatial_z_error_before_trigger',
        '在6、7及7.5秒比较原SNN与修正自主模型的Z二维场及逐格误差，二者均投到共同10×10格。7秒时平均Z接近，但局部高低误差并未消失；这些时刻早于操作性持续高率触发终点。',
        '本图是空间误差诊断，不是致病区域归因；平均误差抵消不能证明局部抑制资源或闭环状态相同。')
    write(OUT/'autonomous_pretrigger_spatial_z_diagnostic.json',{'rows':rows,
        'scope':'Descriptive Z-field errors before the operational sustained-high trigger; no attribution of causal cells. Native Z aggregated to the same 10x10 grid.'})


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare the carried-state counterfactual to its unchanged parent trajectory."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import analyze_topic4_fixed_zm_termination as analysis

ROOT=analysis.OUT;OUT=ROOT/'state_matched_sahp_ablation'

def read_interval(folder,lo=6.,hi=12.):
    parts={}
    for f in sorted((folder/'chunks').glob('*.npz')):
        with np.load(f) as d:
            for key in ['time_ms','spikes_1ms','regions_1ms','field_time_ms','field_5ms','slow_time_ms','Z','M']:
                tk='field_time_ms' if key in ['field_time_ms','field_5ms'] else 'slow_time_ms' if key in ['slow_time_ms','Z','M'] else 'time_ms'
                keep=(d[tk]>=lo*1000)&(d[tk]<hi*1000)
                parts.setdefault(key,[]).append(d[key][keep])
    return {k:np.concatenate(v) for k,v in parts.items()}

def main():
    qa=analysis.run.carrier.base.read(OUT/'complete.json');assert qa['unchanged_restart']=='PASS'
    source=Path(qa['source']);branches=[source,OUT/'runs/removed_sahp_at_6s'];data=[read_interval(f) for f in branches]
    geo=dict(np.load(ROOT/'geometry.npz'));report=dict(branch_time_s=6.,same_full_state=True,unchanged_restart_parity=qa['field_parity'],autonomous_candidate=False,rows=[])
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'xtick.labelsize':13,'ytick.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(15,12));gs=fig.add_gridspec(4,4,height_ratios=[1.15,.95,1.35,1.35],hspace=.5,wspace=.35,left=.08,right=.96,top=.95,bottom=.08)
    ax=fig.add_subplot(gs[0,:]);zax=fig.add_subplot(gs[1,:2]);localax=fig.add_subplot(gs[1,2:])
    names=['sAHP retained','sAHP removed at 6 s'];colors=['#246d98','#bf3547']
    for b,(d,name,col) in enumerate(zip(data,names,colors)):
        n=len(d['spikes_1ms'])//10;rate=d['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/320
        local=d['regions_1ms'][:n*10,:2].reshape(n,10,2).sum(1)/geo['region_counts'][:2]/.01
        t=(np.arange(n)+.5)*.01+6.;ax.plot(t,rate,label=name,c=col,lw=1.5)
        zax.plot(d['slow_time_ms']/1000,d['Z'][:,0],c=col)
        for k,style in enumerate(['-',':']):localax.plot(t,local[:,k],c=col,ls=style,lw=1.1,label=name+' / '+str('AB'[k]))
        report['rows'].append(dict(branch=name,mean_E_Hz_10to12=rate[t>=10].mean(),mean_core_Hz_10to12=local[t>=10].mean(0),mean_Z_at12=d['Z'][-1,0],quiet_E_fraction_10to12=(rate[t>=10]<5).mean()))
        for i,tm in enumerate([6.5,7.5,8.5,10.5]):
            aa=fig.add_subplot(gs[b+2,i]);keep=(d['field_time_ms']>=1000*(tm-.025))&(d['field_time_ms']<1000*(tm+.025));assert keep.sum()==10
            field=d['field_5ms'][keep].sum(0)/geo['cell_e_counts']/.05
            im=aa.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
            for center in geo['centers_mm']:aa.add_patch(Circle(center,1.5,fill=False,ec='#44cccc',lw=1))
            aa.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20]);aa.set_title(f'{tm:g} s',fontsize=15)
            if i==0:aa.set_ylabel(name+'\ny (mm)',fontsize=14)
            else:aa.tick_params(labelleft=False)
    ax.set(xlim=(6,12),ylabel='All E rate (Hz)',xlabel='Time (s)',ylim=(0,520));ax.legend(loc='upper right',frameon=False)
    zax.set(xlim=(6,12),ylabel='Mean Z',xlabel='Time (s)',ylim=(0,1.05));localax.set(xlim=(6,12),ylabel='Core E rate (Hz)',xlabel='Time (s)',ylim=(0,520))
    localax.text(.02,.97,'Core A: solid; Core B: dotted',transform=localax.transAxes,va='top',fontsize=11,bbox=dict(fc='white',ec='none',alpha=.8))
    fig.colorbar(im,ax=fig.axes[3:],location='right',fraction=.016,pad=.02,label='E rate (Hz)')
    dest=OUT/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'carried_state_sahp_removal.png',dpi=160);fig.savefig(dest/'carried_state_sahp_removal.pdf');plt.close(fig)
    analysis.write(OUT/'causal_comparison.json',report)
    (dest/'README.md').write_text('### carried_state_sahp_removal.png\n\n两条轨迹从同一个6s完整网络状态分叉，保留原Z/M及所有历史；红线分支只撤掉新增sAHP电导。未改参数的6–8s续跑与母轨迹逐位一致，蓝线随后延用母轨迹。下方是无空间平滑的原生场，不是电极投影。\n\n**关注点**：这是有干预的因果诊断，不是另一条自主发作；用于检验慢适应对终止的贡献，不能单独排除所有边界作用。\n')
    print(report)

if __name__=='__main__':main()

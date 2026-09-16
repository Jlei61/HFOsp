"""Matched initial-state deterministic trajectories at three frozen Z fields."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_z_branch_extension_20260915'
MODEL=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/approx/coarse_20'


def main():
    m=np.load(MODEL/'model.npz');geo=np.load(MODEL/'geometry.npz');count=m['count_e'];weights=m['threshold_weights_e'];cell=geo['cell_e']
    pos=np.c_[np.bincount(cell,weights=geo['positions_e'][:,0],minlength=400)/count,np.bincount(cell,weights=geo['positions_e'][:,1],minlength=400)/count]
    reg=[np.bincount(cell[geo['g175']==k],minlength=400) for k in [0,1]]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,3,figsize=(12.8,9.2),gridspec_kw={'height_ratios':[1,1,1.2]});fig.subplots_adjust(left=.075,right=.925,top=.84,bottom=.15,wspace=.30,hspace=.48)
    names=['s0_extension','s0.1','s0.228845'];colors=['#252d37','#bd7629','#21877b']
    for col,name in enumerate(names):
        a=np.load(OUT/'deterministic'/f'{name}.npz');raw=a['r_hz'];s=float(a['s']);re=(raw[:,:3200].reshape(len(raw),400,8)*weights).sum(2);g=re@count/count.sum();cores=[re@r/r.sum() for r in reg];t=np.arange(1,len(raw)+1)/1000
        axes[0,col].plot(t,g,color=colors[0],lw=.8);axes[0,col].set_ylim(0,max(40,float(g.max())*1.07));axes[0,col].set_xlim(0,3)
        axes[0,col].set_title((('Fig.5 ③ field · ' if col==2 else '')+f's = {s:.5f}'),fontsize=12,fontweight='bold',pad=10)
        axes[0,col].set_ylabel('Global E rate (Hz)');axes[0,col].set_xlabel('Time after Z change (s)')
        axes[0,col].axvspan(2,3,color='#dddddd',alpha=.22,lw=0)
        for k,x in enumerate(cores):axes[1,col].plot(t,x,color=colors[k+1],lw=.9,label=['Core A','Core B'][k])
        axes[1,col].set_xlim(2,3);axes[1,col].set_ylim(0,max(110,max(float(x[-1000:].max()) for x in cores)*1.08));axes[1,col].set_xlabel('Time after Z change (s)');axes[1,col].set_ylabel('Core E rate (Hz)')
        if col==0:axes[1,col].legend(frameon=False,fontsize=9,loc='upper right')
        mean=re[-1000:].mean(0);im=axes[2,col].scatter(pos[:,0],pos[:,1],c=np.maximum(mean,.02),norm=LogNorm(.02,500),cmap='magma',marker='s',s=31,edgecolors='none')
        for label,(cx,cy) in zip(['A','B'],[(4.19921431597,9.12890135365),(16.47920304044,3.965511533)]):
            axes[2,col].add_patch(plt.Circle((cx,cy),1.75,fill=False,color='#48d4d5',lw=1));axes[2,col].text(cx,cy,label,color='white',ha='center',va='center',fontsize=9,fontweight='bold')
        axes[2,col].set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)');axes[2,col].set_title('Spatial mean, last 1 s',fontsize=10)
    cb=fig.colorbar(im,cax=fig.add_axes([.94,.158,.012,.20]));cb.set_label('E rate (Hz / neuron)',fontsize=9);cb.ax.tick_params(labelsize=8)
    fig.text(.075,.968,'What the same reduction does at larger Z depletion',fontsize=20,fontweight='bold',va='top')
    fig.text(.075,.921,'Identical initial fast state and M; full spatial Z fields changed; constant mean input; dynamic M',fontsize=10.5,color='#626a71')
    fig.text(.075,.892,'This is a deterministic reduced-model control, not a replay of the native SNN.',fontsize=10.5,color='#626a71')
    fig.text(.075,.074,'At the ③ field, the reduction remains at 158–165 Hz globally; it does not reproduce the target irregular bursts.',fontsize=10.5,color='#424a51')
    fig.text(.075,.044,'Ranges and means describe the final 1 s of a 3 s continuation. They are not equilibrium or periodic-branch certificates.',fontsize=10,color='#626a71')
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'fig5_reduced_dynamics_across_z.{ext}',dpi=190)
    print('PLOTTED matched-state direct comparison',flush=True)


if __name__=='__main__':main()

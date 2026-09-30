"""Compare late spatial history differences with within-trajectory variability."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyze_topic4_fig5_D_basins import OUT,SOURCE,FIG,BLUE,ORANGE,savefig,rate_summary


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    counts=np.load(SOURCE/'approx/coarse_20/model.npz')['count_e'];rows=[]
    fig,axs=plt.subplots(3,2,figsize=(11,7),layout='constrained')
    for col,D in enumerate([.22,.228844760565]):
        maps=[];tail=[]
        for h,color in [('burst',BLUE),('tonic',ORANGE)]:
            a=dict(np.load(OUT/f'D{D:.6f}_{h}.npz'))
            assert a['time_s'][-1]>=12.
            tail.append(rate_summary(a,8.,12.))
            maps.append(np.array([a['cell_E_hz'][(a['cell_time_s']>t)&(a['cell_time_s']<=t+1)].mean(0) for t in range(8,12)]))
            for i in range(3):
                axs[i,col].plot(a['time_s'],a['readouts'][:,i],color=color,lw=.7,alpha=.85)
                axs[i,col].set_xlim(6,12)
                if col==0:axs[i,col].set_ylabel([r'$r_E$',r'$r_{E,A}$',r'$r_{E,B}$'][i]+' (Hz / neuron)')
                if i==2:axs[i,col].set_xlabel('Time after transplant (s)')
        axs[0,col].text(.02,.12,fr'$D={D:.4f}$'+('  ③' if D>.225 else ''),transform=axs[0,col].transAxes)
        avg=np.array([a.mean(0) for a in maps]);diff=avg[0]-avg[1]
        between=float(np.sqrt(np.average(diff**2,weights=counts)))
        within=[float(np.sqrt(np.average(((a-a.mean(0))**2).mean(0),weights=counts))) for a in maps]
        rows.append(dict(D=D,window_s=[8,12],tail_by_history=dict(zip(['burst','tonic'],tail)),
                         between_history_mean_map_RMS_Hz=between,within_history_1s_map_RMS_Hz=within,
                         paired_1s_mean_map_RMS_Hz=[float(np.sqrt(np.average(d**2,weights=counts))) for d in maps[0]-maps[1]],
                         global_mean_by_1s_window=[[float(np.average(v,weights=counts)) for v in a] for a in maps],
                         interpretation='Descriptive finite-window comparison; neither proof of identical attractors nor a test of no spatial multistability.'))
    fig.legend(handles=[Line2D([],[],color=BLUE,label='Burst history'),Line2D([],[],color=ORANGE,label='Tonic history')],
               loc='outside upper center',ncol=2,frameon=False)
    savefig(fig,'fig5_D_high_state_extension')
    (OUT/'extension_analysis.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    for r in rows:
        print(r['D'],'between',r['between_history_mean_map_RMS_Hz'],'within',r['within_history_1s_map_RMS_Hz'],
              'global_means',[r['tail_by_history'][h]['mean_E_hz'] for h in ['burst','tonic']])


if __name__=='__main__':main()

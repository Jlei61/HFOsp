"""Finite-window diagnostics for frozen-D trajectories; never a basin certificate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import csv
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_D_separatrix_exploration_20260916'
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
FIG=OUT/'figures'
BLUE,ORANGE='#2759b3','#d48425'


def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def intervals(mask):
    edges=np.diff(np.r_[0,np.asarray(mask,int),0])
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def rate_summary(a,start,end):
    t=a['time_s'];r=a['readouts'];sel=(t>start)&(t<=end)
    # 10-ms nonoverlapping means of the 1-ms sampled continuous reduced rates.
    rr=r[sel,0];rr=rr[:len(rr)//10*10].reshape(-1,10).mean(1)
    qs=[(l,h) for l,h in intervals(rr<5.) if h-l>=2]
    events=[]
    for (_,b),(c,_) in zip(qs[:-1],qs[1:]):
        if c-b>=2 and rr[b:c].max()>=20:events.append(int((c-b)*10))
    cells=a['cell_E_hz'][(a['cell_time_s']>start)&(a['cell_time_s']<=end)]
    region10=r[sel,:4][:len(rr)*10].reshape(-1,10,4).mean(1)
    weights=np.load(SOURCE/'approx/coarse_20/model.npz')['count_e']
    return dict(start_s=start,end_s=end,mean_E_hz=float(rr.mean()),min_10ms_E_hz=float(rr.min()),
                max_10ms_E_hz=float(rr.max()),sd_E_hz=float(rr.std()),quiet_fraction=float((rr<5).mean()),
                quiet_intervals_ge20ms=len(qs),complete_events=len(events),event_duration_ms=events,
                core_A_hz=float(r[sel,1].mean()),core_B_hz=float(r[sel,2].mean()),
                region_min_10ms_hz=region10.min(0).tolist(),region_max_10ms_hz=region10.max(0).tolist(),
                region_quiet_fractions=(region10<5).mean(0).tolist(),
                persistent_E_fraction_50Hz_duty80=float(np.average((cells>50).mean(0)>=.8,weights=weights)),
                surround_hz=float(r[sel,3].mean()),mean_M=float(r[sel,5].mean()),
                cell_mean_E_hz=cells.mean(0).tolist())


def native_memory():
    rows=json.loads((SOURCE/'native_state_map.json').read_text())['rows']
    rows=[r for r in rows if r['z_source_ms']==9420]
    native={};initial={}
    for row in rows:
        name=row['name'];folder=SOURCE/'native/runs'/name
        trace=dict(np.load(folder/'readout_arrays.npz'))
        init=json.loads((folder/'continuation.json').read_text())['initial_mean_M_E']
        times=[];M=[]
        for f in sorted((folder/'fields').glob('*.npz')):
            a=np.load(f)
            times.extend(a['m_step']*.0001-10.37)
            M.extend(a['m'].mean(1))
        trace.update(m_time_s=np.array(times),mean_M=np.array(M))
        native[name]=trace;initial[name]=init
    pairs=[]
    for future in ['W1','W2']:
        lo=f'z9420_h8000_{future}';hi=f'z9420_h10370_{future}'
        t=native[lo]['m_time_s'];assert np.allclose(t,native[hi]['m_time_s'])
        actual=native[hi]['mean_M']-native[lo]['mean_M']
        direct=(initial[hi]-initial[lo])*.9999**np.rint(t/.0001)
        pairs.append(dict(future=future,times_s=t.tolist(),observed_mean_M_difference=actual.tolist(),
                          direct_initial_mean_M_remainder=direct.tolist(),
                          driven_mean_M_difference=(actual-direct).tolist(),
                          initial_mean_M_difference=initial[hi]-initial[lo],
                          direct_initial_current_difference_mV=(.0005*direct).tolist()))
    write('native_history_memory.json',dict(rows=rows,pairs=pairs,tau_M_s=1.,dt_ms=.1,eta_M=.0005,
          interpretation='Algebraic memory decomposition of existing recordings; no new native simulations. '
                         'Different later spike histories can preserve different M even when the direct initial remainder decays. '
                         'This does not identify initial M versus fast history as the trigger, or establish distinct attractors.'))
    return native,pairs


def savefig(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/(name+'.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot_native(native,pairs):
    fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for col,future in enumerate(['W1','W2']):
        for history,color in [(8000,BLUE),(10370,ORANGE)]:
            a=native[f'z9420_h{history}_{future}']
            t=(np.arange(len(a['r10']))+.5)*.01
            axs[0,col].plot(t,a['r10'],color=color,lw=.65,alpha=.8)
        axs[0,col].set(xlim=(6,10),xlabel='Time after transplant (s)',ylabel=r'$r_E$ (Hz / neuron)')
        axs[0,col].text(.03,.92,future,transform=axs[0,col].transAxes)
        p=pairs[col];t=p['times_s']
        axs[1,col].plot(t,np.abs(p['observed_mean_M_difference']),'o-',color='k',ms=3,lw=1.2)
        axs[1,col].plot(t,np.abs(p['direct_initial_mean_M_remainder']),'--',color='#7e4e9b',lw=1.5)
        axs[1,col].set(yscale='log',xlabel='Time after transplant (s)',ylabel=r'$|\Delta\langle M\rangle_E|$',xlim=(0,10))
    fig.legend(handles=[Line2D([],[],color=BLUE,label='8.00 s history'),Line2D([],[],color=ORANGE,label='10.37 s history'),
                        Line2D([],[],color='k',label='Observed M difference'),Line2D([],[],color='#7e4e9b',ls='--',label='Initial M remainder')],
               loc='outside upper center',ncol=2,frameon=False)
    savefig(fig,'fig5_D3_native_history_memory')


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    native,p=native_memory();plot_native(native,p)
    data={f.stem:dict(np.load(f)) for f in sorted(OUT.glob('D*_burst.npz'))+sorted(OUT.glob('D*_tonic.npz'))}
    if len(data)<6:
        print('Native audit complete; six reduced trajectories not yet available.');return
    duration=min(float(a['time_s'][-1]) for a in data.values())
    if duration<2.:
        print('Native audit complete; reduced trajectories below 2 s.');return
    counts=np.load(SOURCE/'approx/coarse_20/model.npz')['count_e']
    rows=[];pairs=[]
    for name,a in data.items():
        windows=[rate_summary(a,float(s),float(s+1)) for s in range(int(duration))]
        tail=rate_summary(a,max(0.,duration-2.),duration)
        rows.append(dict(name=name,D=float(a['D']),duration_s=duration,windows=windows,tail=tail))
    for D in sorted({float(a['D']) for a in data.values()}):
        lo=data[f'D{D:.6f}_burst'];hi=data[f'D{D:.6f}_tonic']
        mask=(lo['time_s']>duration-2)&(lo['time_s']<=duration)
        cells_mask=(lo['cell_time_s']>duration-2)&(lo['cell_time_s']<=duration)
        diff=lo['cell_E_hz'][cells_mask].mean(0)-hi['cell_E_hz'][cells_mask].mean(0)
        base=np.arange(round((duration-1.5)*1000),round((duration-.5)*1000))
        lags=np.arange(-500,501)
        mse=np.array([np.mean((lo['readouts'][base,0]-hi['readouts'][base+lag,0])**2) for lag in lags])
        lag=int(lags[np.argmin(mse)])
        tc=lo['cell_time_s'][(lo['cell_time_s']>=duration-1.5)&(lo['cell_time_s']<=duration-.5)]
        ac=np.column_stack([np.interp(tc,lo['cell_time_s'],lo['cell_E_hz'][:,k]) for k in range(400)])
        bc=np.column_stack([np.interp(tc+lag*.001,hi['cell_time_s'],hi['cell_E_hz'][:,k]) for k in range(400)])
        pairs.append(dict(D=D,tail_start_s=duration-2,tail_end_s=duration,
                          rate_distribution_W1_Hz=float(wasserstein_distance(lo['readouts'][mask,0],hi['readouts'][mask,0])),
                          cell_mean_difference_RMS_Hz=float(np.sqrt(np.average(diff**2,weights=counts))),
                          global_mean_difference_Hz=float(np.mean(lo['readouts'][mask,0]-hi['readouts'][mask,0])),
                          best_global_phase_shift_ms=lag,phase_aligned_global_RMS_Hz=float(np.sqrt(mse.min())),
                          phase_aligned_spatial_RMS_Hz=float(np.sqrt(np.average(((ac-bc)**2).mean(0),weights=counts))),
                          statement='Finite-window comparison, not an asymptotic basin classification.'))
    write('analysis.json',dict(duration_s=duration,rows=rows,pairs=pairs))
    with (OUT/'window_statistics.csv').open('w') as f:
        keys=['name','D','start_s','end_s','mean_E_hz','sd_E_hz','quiet_fraction','quiet_intervals_ge20ms','complete_events','core_A_hz','core_B_hz','mean_M']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader()
        for r in rows:
            for window in r['windows']:w.writerow(dict(name=r['name'],D=r['D'],**window))
    fig,axs=plt.subplots(3,2,figsize=(11,8),layout='constrained',gridspec_kw={'width_ratios':[1.65,1]})
    display_D=[.1,.16,0.228844760565]
    for row,D in enumerate(display_D):
        for history,color in [('burst',BLUE),('tonic',ORANGE)]:
            a=data[f'D{D:.6f}_{history}'];sel=a['time_s']<=duration;r=a['readouts'][sel];t=a['time_s'][sel]
            axs[row,0].plot(t,r[:,0],lw=.8,color=color,alpha=.9)
            early=t<=duration-2;late=~early
            axs[row,1].plot(r[early,0],r[early,5],lw=.7,color=color,alpha=.22)
            axs[row,1].plot(r[late,0],r[late,5],lw=1.,color=color)
        axs[row,0].text(.02,.14 if D>.2 else .94,fr'$D={D:.4f}$'+('  ③' if D>.2 else ''),transform=axs[row,0].transAxes,va='top')
        axs[row,0].set(xlim=(0,duration),ylabel=r'$r_E$ (Hz / neuron)')
        axs[row,1].set(xlabel=r'$r_E$ (Hz / neuron)',ylabel=r'$\langle M\rangle_E$')
    axs[-1,0].set_xlabel('Time after transplant (s)')
    fig.legend(handles=[Line2D([],[],color=BLUE,label='Burst history'),Line2D([],[],color=ORANGE,label='Tonic history')],
               loc='outside upper center',ncol=2,frameon=False)
    savefig(fig,'fig5_D_two_history_trajectories')
    fig,axs=plt.subplots(3,3,figsize=(9,9),layout='constrained',sharex=True,sharey=True)
    arrays=[]
    for D in display_D:
        means=[]
        for history in ['burst','tonic']:
            a=data[f'D{D:.6f}_{history}'];mask=(a['cell_time_s']>duration-2)&(a['cell_time_s']<=duration)
            means.append(a['cell_E_hz'][mask].mean(0).reshape(20,20))
        arrays.append([*means,means[1]-means[0]])
    delta=max(float(abs(a[2]).max()) for a in arrays)
    maximum=max(float(a[:2].max()) for a in map(np.asarray,arrays))
    for i,(D,aa) in enumerate(zip(display_D,arrays)):
        for j,arr in enumerate(aa):
            im=axs[i,j].imshow(arr,origin='lower',extent=(0,20,0,20),cmap='magma' if j<2 else 'RdBu_r',
                               vmin=0 if j<2 else -delta,vmax=maximum if j<2 else delta)
            axs[i,j].set(xticks=[0,10,20],yticks=[0,10,20])
            if i==2:axs[i,j].set_xlabel('x (mm)')
            if j==0:axs[i,j].set_ylabel(fr'$D={D:.4f}$'+'\ny (mm)')
            if j==0:mean_im=im
            if j==2:diff_im=im
    for j,label in enumerate(['Burst history','Tonic history','Tonic − burst']):
        axs[0,j].text(.5,1.05,label,transform=axs[0,j].transAxes,ha='center')
    fig.colorbar(mean_im,ax=axs[:,:2],shrink=.7,label='Mean E rate (Hz / neuron)')
    fig.colorbar(diff_im,ax=axs[:,2],shrink=.7,label='Difference (Hz / neuron)')
    savefig(fig,'fig5_D_two_history_spatial')
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,col,label in [(axs[0,0],0,r'$r_E$ (Hz / neuron)'),(axs[0,1],1,r'$r_{E,A}$ (Hz / neuron)'),
                         (axs[1,0],2,r'$r_{E,B}$ (Hz / neuron)')]:
        for history,color,marker in [('burst',BLUE,'o'),('tonic',ORANGE,'s')]:
            rr=sorted([r for r in rows if r['name'].endswith(history)],key=lambda r:r['D'])
            x=np.array([r['D'] for r in rr])
            # Displace the two symbols slightly for legibility; parameter values remain the ticks/data.
            shown=x+(-.00055 if history=='burst' else .00055)
            means=[r['tail'][['mean_E_hz','core_A_hz','core_B_hz'][col]] for r in rr]
            minima=[r['tail']['region_min_10ms_hz'][col] for r in rr]
            maxima=[r['tail']['region_max_10ms_hz'][col] for r in rr]
            ax.vlines(shown,minima,maxima,color=color,lw=1.3,alpha=.8)
            ax.scatter(shown,means,color=color,s=28,marker=marker,zorder=3)
        ax.set(xlabel=r'$D=1-\langle Z_E\rangle$',ylabel=label,xlim=(.095,.234))
    ax=axs[1,1]
    for history,color,marker in [('burst',BLUE,'o'),('tonic',ORANGE,'s')]:
        rr=sorted([r for r in rows if r['name'].endswith(history)],key=lambda r:r['D'])
        ax.scatter([r['D'] for r in rr],[r['tail']['quiet_fraction'] for r in rr],color=color,marker=marker,s=30)
    ax.set(xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Quiet fraction',ylim=(-.03,1.03),xlim=(.095,.234))
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='Burst history'),
                        Line2D([],[],color=ORANGE,marker='s',ls='',label='Tonic history')],
               loc='outside upper center',ncol=2,frameon=False)
    savefig(fig,'fig5_D_finite_window_scan')
    print(json.dumps(dict(duration_s=duration,pairs=pairs),indent=2))


if __name__=='__main__':main()

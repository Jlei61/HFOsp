#!/usr/bin/env python3
"""Native spikes, all-neuron rates, effective inhibition and return diagnostics."""
from run_topic4_snn_raster_transition import OUT,ROOT,read,write,gain
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

COL_E='#2876ad';COL_I='#e17c24';COL_Q='#7b3294'
ARMS=['jump','z_current_all','z_current_e','jump_ou']
NAMES={'jump':'All GABA synaptic jumps; OU off','z_current_all':'GABA current on E + I; OU off','z_current_e':'GABA current on E only; OU off','jump_ou':'All GABA jumps; original OU background','base':'Fixed original gains; OU off, neuron Poisson retained'}
NAMES['z_current_e_ou']='GABA current on E only; original OU background'
CONTROL_LABELS={
    'jump':r'I$\to$E and I$\to$I: GABA spike increments $\Delta s_I\mapsto q(t)\Delta s_I$',
    'z_current_all':r'E and I cells: $I_{\mathrm{net}}=I_{\mathrm{AMPA}}-q(t)I_{\mathrm{GABA}}$',
    'z_current_e':r'E cells: $I_{\mathrm{net}}=I_{\mathrm{AMPA}}-q(t)I_{\mathrm{GABA}}$; I cells: $q_I=1$',
    'jump_ou':r'I$\to$E and I$\to$I: GABA spike increments $\Delta s_I\mapsto q(t)\Delta s_I$',
    'base':r'All GABA pathways unchanged: $q=1$ throughout',
}
CONTROL_LABELS['z_current_e_ou']=CONTROL_LABELS['z_current_e']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def binned(a,dt,bin_ms=5):return a.reshape(-1,round(bin_ms/dt)).mean(1)


def analyze(a,meta):
    dt=float(a['dt_ms']);e=binned(a['rate_e_hz'],dt);i=binned(a['rate_i_hz'],dt);result={}
    for name,(lo,hi) in {'baseline':(.5,1.),'low_gain_hold':(2.,3.5),'late_restored':(5.5,6.5)}.items():
        y=e[round(lo*200):round(hi*200)];iy=i[round(lo*200):round(hi*200)];peaks,_=find_peaks(y,prominence=20,distance=4)
        result[name]={'time_s':[lo,hi],'mean_E_hz':float(y.mean()),'mean_I_hz':float(iy.mean()),'peak_E_5ms_hz':float(y.max()),'peak_I_5ms_hz':float(iy.max()),'E_p05_p50_p95_hz':np.quantile(y,[.05,.5,.95]).tolist(),
            'fraction_5ms_bins_E_below1Hz':float(np.mean(y<1)),'fraction_5ms_bins_E_above120Hz':float(np.mean(y>120)),'CV_5ms_E':float(y.std()/max(y.mean(),1e-12)),
            'large_peak_count_20Hz_prominence':len(peaks),'peak_interval_ms':(np.diff(peaks)*5).tolist()}
    reg=a['region_spikes_1ms']/a['region_counts'][None,:]*1000
    result['regional_E_late_mean_hz']={name:float(reg[5500:6500,k].mean()) for k,name in enumerate(['core_A','core_B','surround'])}
    result['regional_E_baseline_mean_hz']={name:float(reg[500:1000,k].mean()) for k,name in enumerate(['core_A','core_B','surround'])}
    # Coarse event readout separated from raw peak counts: brief intra-burst gaps are merged.
    active=e>5
    for k in range(1,len(active)-1):
        if not active[k] and active[k-1]:
            end=k
            while end<len(active) and not active[end]:end+=1
            if end<len(active) and end-k<=4:active[k:end]=True
    starts=np.flatnonzero(np.diff(np.r_[False,active].astype(int))==1);ends=np.flatnonzero(np.diff(np.r_[active,False].astype(int))==-1)+1
    episodes=[]
    for lo,hi in zip(starts,ends):
        if e[lo:hi].max()<20 or hi-lo<2:continue
        episodes.append({'start_s':float(lo*.005),'end_s':float(hi*.005),'peak_E_hz':float(e[lo:hi].max()),'duration_ms':int((hi-lo)*5)})
    result['descriptive_episodes']=episodes
    result['episode_definition']='5-ms all-E rate >5 Hz; merge gaps <=20 ms; keep duration >=10 ms and peak >=20 Hz. Descriptive, not patient IED detection.'
    if meta['arm']=='base':
        late=e[200:];peaks=[p for p in episodes if p['start_s']>=1]
        result['constant_control_after_1s']={'mean_E_hz':float(late.mean()),'peak_E_5ms_hz':float(late.max()),'median_E_5ms_hz':float(np.median(late)),
            'episodes':len(peaks),'episode_peak_range_hz':[float(min(p['peak_E_hz'] for p in peaks)),float(max(p['peak_E_hz'] for p in peaks))] if peaks else None,
            'fraction_5ms_bins_E_below1Hz':float(np.mean(late<1))}
    return result


def raster(ax,a):
    spikes=a['sample_spikes'];dt=float(a['dt_ms'])
    for lo,hi,color in [(0,240,COL_E),(240,300,COL_I)]:
        t,neuron=np.nonzero(spikes[:,lo:hi]);ax.scatter(t*dt/1000,neuron+lo,s=.35,color=color,marker='.',linewidths=0,rasterized=True)
    for y in [60,120,240]:ax.axhline(y-.5,c='.75',lw=.6)
    ax.set(ylim=(-1,300),yticks=[30,90,180,270],yticklabels=['Core A E','Core B E','Other E','I'],xlim=(0,6.5))


def decorate(ax):
    ax.axvspan(2,3.5,color=COL_Q,alpha=.055)
    for x in [1,2,3.5,4.5]:ax.axvline(x,c='.7',ls=':',lw=.65)


def single(arm):
    path=OUT/'runs'/f'{arm}_seed9108401.npz';a=np.load(path);meta=read(path.with_suffix('.json'));stats=analyze(a,meta)
    write(OUT/'runs'/f'{arm}_summary.json',stats)
    fig,axes=plt.subplots(3,1,figsize=(14,8),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[.55,1.4,1]})
    t=np.arange(6500)/1000;axes[0].plot(t,a['q_1ms'],c=COL_Q,lw=2);axes[0].set(ylabel='GABA multiplier q\n(dimensionless)',ylim=(.45,1.2),yticks=[.5,1])
    axes[0].set_title(CONTROL_LABELS[arm]+'\nq = 1: original coefficient; q = 0.5: half coefficient (actual current also depends on network activity)',fontsize=11)
    annotations=[(3.25,'No parameter change, no imposed stimulus')] if arm=='base' else [(.5,'Initial'),(1.5,'Decrease'),(2.75,'Hold low'),(4,'Restore'),(5.5,'Hold restored')]
    for x,label in annotations:axes[0].text(x,1.12,label,ha='center',fontsize=11)
    raster(axes[1],a);dt=float(a['dt_ms']);tt=(np.arange(1300)+.5)*.005
    for key,c,label in [('rate_e_hz',COL_E,'All E'),('rate_i_hz',COL_I,'All I')]:axes[2].plot(tt,binned(a[key],dt),color=c,lw=.85,label=label)
    axes[2].set(xlabel='Time (s)',ylabel='Rate (Hz; 5-ms bins)');axes[2].legend(loc='upper right')
    if arm=='z_current_e':
        axes[2].annotate('Sustained high-rate plateau\n(not population bursting)',xy=(2.7,455),xytext=(2.55,210),ha='center',fontsize=10,arrowprops={'arrowstyle':'->','color':'.3'})
        axes[2].annotate('Return during externally\nimposed inhibition restoration',xy=(4.08,20),xytext=(4.65,310),fontsize=10,arrowprops={'arrowstyle':'->','color':'.3'})
    if arm!='base':
        for axis in axes:decorate(axis)
    fig.suptitle('Spatial dual-core SNN — '+NAMES[arm],fontsize=15)
    footer='Constant control: no parameter cycle and no stimulation; Z/M and OU off, neuron-level Poisson retained.' if arm=='base' else 'q(t) is externally prescribed, including restoration; endogenous Z/M dynamics are OFF. GABA decay stays at 20.61 ms.'
    fig.supxlabel('Same 40,000-neuron network throughout; no state reset. Raster is a fixed stratified sample, population rates use all neurons.\n'+footer,fontsize=10)
    folder=OUT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/f'raster_{arm}.png',dpi=180);fig.savefig(folder/f'raster_{arm}.pdf');plt.close(fig)
    print(arm,stats,flush=True)


def main():
    paths={arm:OUT/'runs'/f'{arm}_seed9108401.npz' for arm in ARMS}
    assert all(p.exists() for p in paths.values())
    data={arm:np.load(path) for arm,path in paths.items()};meta={arm:read(path.with_suffix('.json')) for arm,path in paths.items()}
    # Same state, topology and external random stream before the parameter ramp.
    prefix=round(1000/float(data['jump']['dt_ms']));prefix_checks={}
    for arm in ['z_current_all','z_current_e']:
        prefix_checks[arm]={k:bool(np.array_equal(data['jump'][k][:prefix],data[arm][k][:prefix])) for k in ['sample_spikes','rate_e_hz','rate_i_hz']};assert all(prefix_checks[arm].values())
    summaries={arm:analyze(data[arm],meta[arm]) for arm in ARMS}
    base_path=OUT/'runs/base_seed9108401.npz'
    if base_path.exists():
        base=np.load(base_path);summaries['base']=analyze(base,read(base_path.with_suffix('.json')))
        prefix_checks['base']={k:bool(np.array_equal(data['jump'][k][:prefix],base[k][:prefix])) for k in ['sample_spikes','rate_e_hz','rate_i_hz']};assert all(prefix_checks['base'].values())
    write(OUT/'analysis.json',{'status':'COMPLETE','same_input_prefix_identity':prefix_checks,'arms':summaries,
        'units':'All rates from all neurons; raster fixed stratified sample. Peaks in 5 ms bins. Quiet and high-rate fractions are continuous diagnostics, not clinical labels.',
        'limitations':'One seed, finite 6.5-s protocol, prescribed inhibition not evolving Z; restored-state window 1 s only; no hysteresis or patient event-repertoire acceptance.'})
    fig=plt.figure(figsize=(15,13),layout='constrained');grid=fig.add_gridspec(5,2,height_ratios=[.65,1,1,1,1],width_ratios=[1.05,1])
    tt=np.arange(6500)/1000
    for column in [0,1]:
        top=fig.add_subplot(grid[0,column]);top.plot(tt,[gain(t*1000) for t in tt],color=COL_Q,lw=2);top.set(ylabel='GABA multiplier q\n(dimensionless)' if column==0 else '',ylim=(.43,1.18),yticks=[.5,1],xlim=(0,6.5),xticks=[])
        top.set_title('Externally prescribed q(t): 1 = original coefficient; 0.5 = half coefficient',fontsize=10)
        for x,label in [(.5,'Initial'),(1.5,'Decrease'),(2.75,'Hold low'),(4,'Restore'),(5.5,'Restored')]:top.text(x,1.12,label,ha='center',fontsize=10)
    for row,arm in enumerate(ARMS,1):
        a=data[arm];left=fig.add_subplot(grid[row,0]);right=fig.add_subplot(grid[row,1],sharex=left);raster(left,a)
        dt=float(a['dt_ms']);t=(np.arange(1300)+.5)*.005;e=binned(a['rate_e_hz'],dt);i=binned(a['rate_i_hz'],dt)
        right.plot(t,e,c=COL_E,lw=.85,label='E');right.plot(t,i,c=COL_I,lw=.85,label='I');right.set(ylabel='Population rate (Hz)',xlim=(0,6.5))
        left.set_title(f'{chr(64+row)}  {NAMES[arm]}\n'+CONTROL_LABELS[arm],loc='left',fontsize=10);right.set_title('Sustained high-rate plateau; return under external restoration' if arm=='z_current_e' else 'All 32,000 E + 8,000 I; 5-ms bins',fontsize=10)
        decorate(left);decorate(right)
        if row<4:left.tick_params(labelbottom=False);right.tick_params(labelbottom=False)
        else:left.set_xlabel('Time (s)');right.set_xlabel('Time (s)')
        if row==1:right.legend(loc='upper right',fontsize=10)
    fig.suptitle('Spatial dual-core SNN: inhibition decrease and restoration without resetting state',fontsize=15)
    fig.supxlabel('Raster: fixed 60 core-A E, 60 core-B E, 120 other E, 60 I; rates use all neurons. A/B/C share Poisson input; OU off. D: original OU noise.\nOnly the stated GABA coefficient is modulated; actual current depends on network activity. GABA decay = 20.61 ms; endogenous Z/M OFF.',fontsize=10)
    folder=OUT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'snn_raster_inhibition_cycle.png',dpi=180);fig.savefig(folder/'snn_raster_inhibition_cycle.pdf');plt.close(fig)
    # Region and current observables answer whether global averaging hid local activity.
    fig,axs=plt.subplots(4,2,figsize=(14,11),layout='constrained',sharex=True)
    for row,arm in enumerate(ARMS):
        a=data[arm];reg=a['region_spikes_1ms']/a['region_counts'][None,:]*1000;reg=reg.reshape(-1,10,3).mean(1);t=(np.arange(len(reg))+.5)*.01
        for j,(label,c) in enumerate([('Core A','#a43c8e'),('Core B','#3f83af'),('Surround','#777777')]):axs[row,0].plot(t,reg[:,j],lw=.85,c=c,label=label)
        current=a['currents_1ms'].copy();q=a['q_1ms']
        if arm=='z_current_e':current[:,1]*=q
        elif arm=='z_current_all':current[:,1]*=q;current[:,3]*=q
        axs[row,1].plot(tt,current[:,1],lw=.8,color=COL_E,label='Effective GABA on E');axs[row,1].plot(tt,current[:,3],lw=.8,color=COL_I,label='Effective GABA on I')
        axs[row,0].set(ylabel='E rate (Hz)',title=NAMES[arm]);axs[row,1].set(ylabel='Current-equivalent voltage (mV)',title='Actual inhibitory term used by membrane')
        decorate(axs[row,0]);decorate(axs[row,1])
        if row==0:axs[row,0].legend(fontsize=8);axs[row,1].legend(fontsize=8)
    for a in axs[-1]:a.set_xlabel('Time (s)')
    fig.suptitle('Regional firing and the inhibitory current actually applied to each population',fontsize=14)
    fig.savefig(folder/'regional_rates_and_effective_inhibition.png',dpi=180);fig.savefig(folder/'regional_rates_and_effective_inhibition.pdf');plt.close(fig)
    print('Analysis and figures complete',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--single',choices=ARMS+['base','z_current_e_ou']);args=parser.parse_args()
    if args.single:single(args.single)
    else:main()

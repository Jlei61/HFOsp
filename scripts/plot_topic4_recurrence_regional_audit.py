#!/usr/bin/env python3
"""Does population averaging hide pronounced local E/I burst modulation?"""
from pathlib import Path
import json
import numpy as np
from scipy.signal import detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'weak_fast_z_refill_recurrence_v2';FIG=OUT/'figures'
with np.load(BASE/'weak_fast_z_refill_v1/runs/weak_fast_z_refill.npz') as f:
    counts=f['region_spikes_1ms'];n=f['region_counts']
windows=[(68.4,68.8,'Finite event'),(74.8,75.2,'First high state'),(96.1,96.5,'After Z refill')]
plt.rcParams.update({'font.size':13,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42})
fig,axes=plt.subplots(3,3,figsize=(15,9),sharex='col',gridspec_kw={'hspace':.3,'wspace':.24})
rows=[]
for j,(lo,hi,title) in enumerate(windows):
    rate=counts[round(lo*1000):round(hi*1000)].reshape(-1,2,6).sum(1)/n[None,:]/.002
    t=np.arange(len(rate))*2+1
    for i,name in enumerate(['Core A','Core B','Surround']):
        ax=axes[i,j]
        for k,col,label in [(i,'#236b85','E'),(i+3,'#c8782f','I')]:
            x=rate[:,k];ax.plot(t,x,c=col,lw=1.15,label=label)
            rows.append(dict(window=title,region=name,population=label,n_cells=int(n[k]),
                mean_hz=float(x.mean()),p05_hz=float(np.percentile(x,5)),p95_hz=float(np.percentile(x,95)),
                fraction_bins_below5Hz=float((x<5).mean()),detrended_sd_hz=float(detrend(x).std())))
        ax.set(xlim=(0,400),ylim=(-10,610));ax.spines[['top','right']].set_visible(False)
        if j==0:ax.set_ylabel(f'{name}\nRate (Hz; 2-ms bins)')
        if i==0:ax.set_title(f'{title}\n{lo:g}–{hi:g} s',fontsize=15)
        if i==2:ax.set_xlabel('Time within window (ms)')
axes[0,0].legend(frameon=False,ncol=2)
for row in axes:
    upper=np.ceil(max(np.max(line.get_ydata()) for ax in row for line in ax.lines)*1.08/100)*100
    for ax in row:ax.set(ylim=(-.02*upper,upper),yticks=[0,upper/2,upper])
fig.subplots_adjust(left=.105,right=.98,bottom=.09,top=.92)
fig.savefig(FIG/'regional_native_rates.png',dpi=170);fig.savefig(FIG/'regional_native_rates.pdf');plt.close(fig)
result=dict(question='Are pronounced local core oscillations being hidden by global averaging?',
    observable='Native regional spike counts; same fixed Core A/Core B/Surround masks, 2-ms population-rate bins',
    unit='One continuous trajectory; regions and time bins are not independent replicates',
    selection='Same predetermined windows as the raw/band-limited readout audit; no selection by oscillator score',
    boundary='These traces assess the saved regional means, not every spatial eigenmode or single-cell periodicity.',
    all_plotted_values_within_limits=all(np.max(line.get_ydata())<=ax.get_ylim()[1] for row in axes for ax in row for line in ax.lines),results=rows)
(OUT/'regional_rate_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([r for r in rows if r['window']=='First high state'],indent=2))

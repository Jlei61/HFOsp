#!/usr/bin/env python3
"""Inspect actual raw and band-limited virtual-contact traces at matched times."""
from pathlib import Path
import json,argparse,numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'weak_fast_z_refill_recurrence_v2';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--readout',choices=['applied','legacy'],default='applied');args=parser.parse_args()
legacy=args.readout=='legacy';prefix='legacy_' if legacy else ''
a=np.load(BASE/'weak_fast_z_refill_v1/runs/weak_fast_z_refill.npz')
t=a['lfp_time_ms']/1000;raw=a['lfp_raw' if legacy else 'lfp_effective'];sos=butter(4,[30,80],btype='bandpass',fs=2000,output='sos');band=sosfiltfilt(sos,raw,axis=0)
ids=[list(a['contact_names']).index(n) for n in ['SCL9','ICL2']];cols=['#236b85','#a65d24']
windows=[(68.4,68.8,'Finite event'),(74.8,75.2,'First high state'),(96.1,96.5,'After Z refill')]
plt.rcParams.update({'font.size':13,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42})
fig,axes=plt.subplots(3,3,figsize=(15,9),gridspec_kw={'hspace':.4,'wspace':.25},sharex='col')
metrics=[]
for j,(lo,hi,title) in enumerate(windows):
 q=(t>=lo)&(t<hi);x=(t[q]-lo)*1000
 for idx,col in zip(ids,cols):
  axes[0,j].plot(x,raw[q,idx],color=col,lw=1.2,label=str(a['contact_names'][idx]))
  axes[1,j].plot(x,band[q,idx],color=col,lw=1.2)
  metrics.append(dict(window=title,contact=str(a['contact_names'][idx]),raw_mean=float(raw[q,idx].mean()),raw_sd=float(raw[q,idx].std()),band_rms=float(np.sqrt(np.mean(band[q,idx]**2)))))
 lo_i=round(lo*10000);hi_i=round(hi*10000)
 er=a['rate_e_hz'][lo_i:hi_i].reshape(-1,10).mean(1);ir=a['rate_i_hz'][lo_i:hi_i].reshape(-1,10).mean(1)
 axes[2,j].plot(np.arange(len(er))+.5,er,c='#236b85',lw=1.1,label='E')
 axes[2,j].plot(np.arange(len(ir))+.5,ir,c='#c8782f',lw=1.1,label='I')
 axes[0,j].set_title(f'{title}\n{lo:g}–{hi:g} s',fontsize=15)
 axes[2,j].set_xlabel('Time within window (ms)');axes[2,j].set_xlim(0,400)
 for k in range(3):axes[k,j].spines[['top','right']].set_visible(False)
axes[0,0].set_ylabel('Native current proxy\n(a.u.; no filtering)');axes[1,0].set_ylabel('Virtual SEEG\n30–80 Hz (same a.u.)');axes[2,0].set_ylabel('Population rate\n(Hz; 1-ms bins)')
# Preserve the very different amplitudes; common scales within each row.
for row in range(3):
 limits=np.array([ax.get_ylim() for ax in axes[row]]);lo=limits[:,0].min();hi=limits[:,1].max()
 for ax in axes[row]:ax.set_ylim(lo,hi)
axes[0,0].legend(frameon=False,ncol=2,fontsize=12);axes[2,0].legend(frameon=False,ncol=2,fontsize=12)
fig.subplots_adjust(left=.1,right=.98,bottom=.08,top=.92)
fig.savefig(FIG/f'{prefix}native_vs_band_limited_readout.png',dpi=170);fig.savefig(FIG/f'{prefix}native_vs_band_limited_readout.pdf');plt.close(fig)
(OUT/f'{prefix}readout_window_audit.json').write_text(json.dumps(dict(readout='lfp_raw' if legacy else 'lfp_effective',filter='Fourth-order30-80Hz zero-phase Butterworth; full continuous signal before cropping',metrics=metrics,window_selection='Fixed event1, first high plateau, late post-refill event from prior numbered states, no waveform-fit selection',interpretation='Band-limited wave packets are an observation; compare unfiltered current and actual E/I spikes before claiming a sustained oscillatory network state.'),indent=2)+'\n')
print(json.dumps(metrics,indent=2))

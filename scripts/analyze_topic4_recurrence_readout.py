#!/usr/bin/env python3
"""Freeze and compute a common dimensionless ER on model and fixed E1146 SZ13."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.signal import butter,sosfiltfilt,spectrogram
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.paper_figures import plot_fig3b_interictal_ictal_shared_field as clinical
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'weak_fast_z_refill_recurrence_v2'

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def power(x,fs,band=(1,150)):
 f,tt,p=spectrogram(np.asarray(x,float),fs=fs,nperseg=round(fs),noverlap=round(fs/2),scaling='density',mode='psd',axis=0)
 return p[(f>=band[0])&(f<=band[1])].sum(0).T,tt

def common_er(x,fs,t0_sample,onset):
 t=np.arange(len(x))/fs+t0_sample-onset
 def window(lo,hi):
  q=(t>=lo-1e-9)&(t<hi-1e-9);z=x[q];assert abs(len(z)/fs-(hi-lo))<1/fs+.00001;return z
 target,_=power(window(0,1),fs);base,_=power(window(-60,-30),fs)
 er=target.mean(0)/base.mean(0)
 return er,dict(target_power=target.mean(0),baseline_power=base.mean(0),baseline_frames=len(base))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--readout',choices=['applied','legacy'],default='applied');args=parser.parse_args()
 legacy=args.readout=='legacy';suffix='_legacy' if legacy else ''
 protocol=dict(status='DEFINED_BEFORE_VALUES',band_hz=[1,150],estimator='SciPy spectrogram,1s window,0.5s hop,Tukey default,constant detrend',
 target_s_relative_onset=[0,1],baseline_s_relative_onset=[-60,-30],quantity='ER = mean target band power / mean baseline band power',
 model_onset='First start of >=200Hz full-E rate lasting >=200ms:73.48s',patient_onset='Clinical onset of fixed E1146 SZ13',
 changes='Both sides use identical duration, relative windows, band and dimensionless ratio. Replaces prior model dB / patient robust-z pairing. New comparison, not the canonical Fig3 robust-z result.',
 display='One shared positive-valued logarithmic ER colorbar; ER1 unchanged, ER<1 decrease, ER>1 increase. No clipping of decreases.',
 A_display='Historical fourth-order30-80Hz zero-phase bandpass of recorded LFP; native signal retained separately. Does not prove network oscillation.',
 patient_selection='Keep the previous SZ13 example; do not maximize new ER correlation.',
 readout_source='lfp_raw: weighted abs(I_AMPA)+abs(I_GABA) before slow.apply_currents; exact old paper source' if legacy else 'lfp_effective: applied-inhibition current proxy after multiplying GABA by Z',
 readout_choice_reason='User requested original paper burst method; source verified in old kick_probe.py lines432-439, not chosen by correlation' if legacy else 'Initially inherited applied-current readout; retained as explicit comparator')
 dump(OUT/f'observation{suffix}_protocol.json',protocol)
 with np.load(BASE/'weak_fast_z_refill_v1/runs/weak_fast_z_refill.npz') as f:
  model=np.asarray(f['lfp_raw' if legacy else 'lfp_effective']);mt=f['lfp_time_ms']/1000;names=f['contact_names'].copy();xy=f['contact_xy'].copy();centers=f['centers_mm'].copy();raw=f['lfp_raw'].copy()
 onset=73.48;model_er,md=common_er(model,2000,0,onset)
 inv=clinical._inventory_for_subject('epilepsiae','1146');eeg_rel=clinical._eeg_offset_from_inventory('epilepsiae',inv[12]);pre,post=clinical._extract_bounds(eeg_rel)
 seizure=clinical.extract_seizure_window('epilepsiae/1146',12,pre_sec=pre,post_sec=post,reference=clinical.ICTAL_REFERENCE['epilepsiae'])
 raw_names=[clinical.bipolar_alias_label(n) for n in seizure.ch_names]
 ids=[raw_names.index(str(n)) for n in names];patient=seizure.signal[ids].T
 patient_er,pd=common_er(patient,seizure.fs,-seizure.pre_sec,0)
 rho=float(spearmanr(model_er,patient_er).statistic)
 # Filter full continuous recordings before choosing display windows; never filter cropped bursts.
 sos=butter(4,[30,80],btype='bandpass',fs=2000,output='sos')
 filtered=sosfiltfilt(sos,model,axis=0)
 b=(mt>=.5)&(mt<onset);scale=np.percentile(abs(filtered[b]),95,axis=0);scale=np.maximum(scale,.15*np.median(scale[scale>1e-12]))
 native_rms={}
 for label,lo,hi in [('interictal',68,71),('first_entry',73.48,74.48),('first_high',74.5,75.5),('after_release',80,83)]:
  q=(mt>=lo)&(mt<hi);native_rms[label]=dict(window_s=[lo,hi],band_rms=np.sqrt(np.mean(filtered[q]**2,0)).tolist(),unfiltered_sd=np.std(model[q],axis=0).tolist())
 np.savez_compressed(OUT/f'common_er{suffix}_arrays.npz',contact_names=names,contact_xy=xy,centers_mm=centers,model_ER=model_er,patient_ER=patient_er,
  model_baseline_power=md['baseline_power'],model_early_power=md['target_power'],patient_baseline_power=pd['baseline_power'],patient_early_power=pd['target_power'],display_scale_30_80=scale)
 # Exact raw/filtered views for checking whether cycles survive outside the chosen band.
 q=(mt>=68)&(mt<76.5)
 np.savez_compressed(OUT/f'readout{suffix}_audit_arrays.npz',time_s=mt[q],lfp_observed=model[q],lfp_raw=raw[q],band30_80=filtered[q],contact_names=names)
 result=dict(protocol=protocol,rho=rho,model_ER=model_er.tolist(),patient_ER=patient_er.tolist(),
 model_enhanced_contacts=int((model_er>1).sum()),patient_enhanced_contacts=int((patient_er>1).sum()),
 model_baseline_frames=md['baseline_frames'],patient_baseline_frames=pd['baseline_frames'],model_onset_s=onset,
 model_reference='Unreferenced positive synaptic-current proxy',patient_reference=clinical.ICTAL_REFERENCE['epilepsiae'],
 current_proxy_not_biophysical_SEEG=True,native_rms=native_rms,
 trace_producer_reference=str(ROOT/'.worktrees/figure-final-sync-20260905/scripts/paper_figures/plot_fig_mz_early_bridge_v2.py'))
 dump(OUT/f'common_er{suffix}_summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

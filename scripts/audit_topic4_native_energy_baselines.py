#!/usr/bin/env python3
"""Fixed-window sensitivity of measured native early band power, no fitting."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import analyze_topic4_native_band_energy as band

def correlation(rank,power):
    valid=np.isfinite(rank)&np.isfinite(power)
    if valid.sum()<3 or np.ptp(rank[valid])==0 or np.ptp(power[valid])==0:return None
    return float(spearmanr(-rank[valid],power[valid]).statistic)

def main(root,name):
    qa=json.loads((root/'observation_qa.json').read_text());assert qa['status']=='PASS'
    folder=root/'runs'/name;energy=folder/'native_energy'
    figure=json.loads((root/'full_fig5/figure_metadata.json').read_text())
    displayed_rank=np.array([np.nan if v is None else v for v in figure['early_energy']['rank']],float)
    family_ranks={family:np.array([np.nan if v is None else v for v in group['rank']],float)
        for family,group in figure['early_energy'].get('all_family_correspondence',{}).items()}
    rank=family_ranks.get('B',displayed_rank)
    saved=np.load(energy/'spectral_power.npz')
    early={k:saved['early_'+k] for k in ['effective_proxy','native_effective_grid']}
    baseline=band.window(folder,.5,7.)
    windows=[];powers={key:[] for key in early}
    for i in range(7):
        for key in powers:
            powers[key].append(band.spectral_power(baseline[key][i*10000:(i+1)*10000]))
        delta=early['effective_proxy']-powers['effective_proxy'][-1]
        windows.append(dict(baseline_s=[.5+i,1.5+i],
            B_contact_rho=correlation(rank,delta),positive_contacts=int((delta>0).sum()),
            all_family_contact_rho={family:correlation(values,delta) for family,values in family_ranks.items()},
            contact_delta=delta.tolist()))
    pooled={key:early[key]-np.mean(powers[key],axis=0) for key in powers}
    original=saved['delta_effective_proxy']
    assert np.allclose(np.asarray(windows[0]['contact_delta']),original,rtol=1e-12,atol=1e-8)
    assert np.array_equal(original,np.asarray(figure['early_energy']['contact_power']))
    report=dict(source=str(folder),fixed_early_window_s=figure['early_energy']['spectral_method']['early_s'],
        baseline_windows=windows,pooled_baseline_s=[.5,7.5],
        pooled_B_contact_rho=correlation(rank,pooled['effective_proxy']),
        pooled_all_family_contact_rho={family:correlation(values,pooled['effective_proxy']) for family,values in family_ranks.items()},
        pooled_positive_contacts=int((pooled['effective_proxy']>0).sum()),
        original_figure_rho=figure['early_energy']['model_contact_spearman_minus_rank_vs_band_change'],
        original_displayed_family=figure['early_energy']['displayed_family'],
        original_B_contact_rho=correlation(rank,original),
        original_figure_power_verified=True,
        method='Seven fixed nonoverlapping1s baseline windows,0.5–7.5s. Every window uses native10kHz Hann periodogram, mean removed,1–150Hz. The early window and B template are held fixed. Pooled baseline averages seven spectral-power estimates; no baseline is selected to improve correlation.',
        statistical_unit='Descriptive baseline-window sensitivity within one model trajectory, not independent networks or patient evidence.',
        power_interpretation='Positive and negative changes retained. Slow onset ramps contribute low-frequency power; a power increase alone is not proof of sustained oscillations.')
    (energy/'baseline_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(energy/'pooled_baseline_power.npz',**{f'delta_{k}':v for k,v in pooled.items()})
    print({k:report[k] for k in ['original_figure_rho','pooled_B_contact_rho','pooled_positive_contacts']})
    print('fixed_window_rhos',[v['B_contact_rho'] for v in windows])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    args=p.parse_args();main(args.root,args.name)

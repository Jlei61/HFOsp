from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(ROOT/'scripts/paper_figures'))
import build_fig5_transition_focus as b
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[1]
m=json.loads((p/'fig5_metadata.json').read_text())
a,r=b.previous.audit.load_small(Path(m['source']),keys=('lfp_time_ms','lfp_raw','spikes_1ms'))
n=round(m['source_loaded_time_window_s'][1]*1000)
for key in ['lfp_time_ms','lfp_raw']: a[key]=a[key][:2*n]
a['spikes_1ms']=a['spikes_1ms'][:n]
fig,ax=plt.subplots()
w=b.draw_readout(ax,a,m['snapshots'],m['first_entry']['onset_s'],9.42)
t=a['lfp_time_ms']/1000;take=(t>=8)&(t<=11)
ids=[list(a['contact_names']).index(n) for n in w['contact_order']]
raw=a['lfp_raw'][:,ids]
errors=[]
for i,line in enumerate(ax.lines[:15]):
 assert np.array_equal(line.get_xdata(),t[take])
 recovered=(line.get_ydata()-(14-i))*w['gain_per_contact'][i]/w['display_gain_per_row_spacing']
 errors.append(float(np.max(abs(recovered-raw[take,i]))))
 assert np.allclose(recovered,raw[take,i],rtol=1e-12,atol=1e-9)
 assert min(line.get_ydata())>=ax.get_ylim()[0] and max(line.get_ydata())<=ax.get_ylim()[1]
plt.close(fig)
windows=[[8,9.37],[10.2,11]]
means=[];sds=[];rates=[]
for lo,hi in windows:
 x=raw[(t>=lo)&(t<hi)];means.append(x.mean(0));sds.append(x.std(0))
 c=a['spikes_1ms'][round(lo*1000):round(hi*1000)]
 rates.append((c.sum(0)/np.array([32000,8000])/(hi-lo)).tolist())
prior=ROOT/'results/paper-ready-figure/archive/2026-09-30_pre_unfiltered_readout_fig5/fig5'
unchanged={k:(p/f'figures/fig5-panel{k}.png').read_bytes()==(prior/f'figures/fig5-panel{k}.png').read_bytes() for k in 'bcdef'}
assert all(unchanged.values())
assert m['waveform']['raw_lfp_sha256']==hashlib.sha256(a['lfp_raw'].tobytes()).hexdigest()
out=dict(status='PASS',source=m['source'],windows_s=windows,contact_order=w['contact_order'],raw_mean_per_window=[x.tolist() for x in means],raw_SD_per_window=[x.tolist() for x in sds],median_within_contact_late_early_mean_ratio=float(np.median(means[1]/means[0])),median_within_contact_late_early_SD_ratio=float(np.median(sds[1]/sds[0])),population_order=['All E','All I'],mean_rates_Hz_per_window=rates,recovered_raw_samples_max_absolute_error=max(errors),all_stored_samples_in_window_plotted=True,samples_per_contact=int(take.sum()),DC_preserved=True,B_to_F_PNG_bitwise_unchanged=unchanged,unit='Same trajectory; within-contact temporal comparison, not independent replications.')
(p/'validation/unfiltered_readout_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['raw_mean_per_window','raw_SD_per_window','contact_order']}))

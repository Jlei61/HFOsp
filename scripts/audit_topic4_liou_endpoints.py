#!/usr/bin/env python3
"""Source spatial extent and native current balance, from existing trajectories."""
import json
import numpy as np
from scipy.ndimage import uniform_filter1d
from run_topic4_liou_original_reference import OUT
import analyze_topic4_autonomous_recovery as common
import analyze_topic4_liou_spatial_feedback as native

rows=[]
for p in sorted((native.OUT/'runs').glob('liou*/result.json')):
    result=json.loads(p.read_text());job=result['job']
    a=common.load(p.parent,['slow_time_ms','Z','M','currents','time_ms','spikes_1ms'])
    q=native.feedback(p.parent);ix=a['slow_time_ms']>=50000;qi=q['time_ms']>=50000
    rows.append(dict(name=p.parent.name,gamma=job['spatial_gamma'],seed=job['seed'],
        observed_s=result['end_s'],entries=result['tracker']['entries'],recoveries=result['tracker']['recoveries'],
        late_window_s=[50,60],E_mean_Hz=float(a['spikes_1ms'][-10000:,0].mean()*1000/32000),
        mean_Z=float(a['Z'][ix,0].mean()),final_Z=float(a['Z'][-1,0]),
        mean_E_input=float(a['currents'][ix,0].mean()),mean_raw_total_I=float(a['currents'][ix,3].mean()),
        mean_applied_I=float(a['currents'][ix,2].mean()),mean_M_current=float(job['eta_m']*a['M'][ix,0].mean()),
        expected_M_current_at500Hz=job['eta_m']*job['tau_M_s']*500,
        fraction_E_with_Z_recovery_target=float(q['Z_recovery_drive_fraction'][qi].mean()),
        late_raw_global_current=float(q['raw_global_current'][qi].mean()),
        late_effective_global_current=float(q['effective_global_current'][qi].mean())))
assert len(rows)==6
(OUT/'native_endpoint_balance.json').write_text(json.dumps({'rows':rows,'current_unit':'mV equivalent in the current-LIF voltage equation','statistical_unit':'paired noise trajectory; two noise seeds, one frozen topology','scope':'Observed late-state balance, not a proof of perpetual behavior or of a bifurcation'},indent=2)+'\n')

extent=[]
names=['exp2a_s20260915','exp2_global_removed','exp2_global_redistributed_local_extended200',
       'exp4a_s20260915_extended150','exp4a_s20260916_extended150','exp4b_s20260915_extended150','exp4b_s20260916']
for name in names:
    folder=OUT/'reference_runs'/name;pr=json.loads((folder/'protocol.json').read_text())
    if pr['spiking']:
        s=np.load(folder/'spikes.npy',mmap_mode='r');nt=len(s)//10
        f=s[:nt*10].reshape(nt,10,100,20).sum((1,3))/.2
        position=(np.arange(100)+.5)/100
    else:
        s=np.load(folder/'output.npy',mmap_mode='r');nt=len(s)//10
        f=s[:nt*10].reshape(nt,10,500).mean(1);position=(np.arange(500)+1)/500
    # Average100ms to avoid calling isolated single spikes a recruited front.
    f=uniform_filter1d(f.astype(float),10,axis=0,mode='nearest')
    active=(f>=20)&((np.arange(nt)*.01)[:,None]>=2)
    cum=active.any(0);distal=active[:,position>=.95].any(1)
    extent.append(dict(name=name,threshold_Hz=20,temporal_window_s=.1,
        farthest_active_position=float(position[cum].max()) if cum.any() else None,
        first_distal95_activation_s=float(np.flatnonzero(distal)[0]*.01) if distal.any() else None,
        last_local_activation_s=float(np.flatnonzero(active.any(1))[-1]*.01) if active.any() else None,
        boundary='Finite linear domain. Distal recruitment is a description; reaching it before termination does not prove boundary-caused termination.'))
(OUT/'source_extent_audit.json').write_text(json.dumps({'rows':extent},indent=2)+'\n')
print(json.dumps({'native_rows':len(rows),'source_extent_rows':len(extent)},indent=2))

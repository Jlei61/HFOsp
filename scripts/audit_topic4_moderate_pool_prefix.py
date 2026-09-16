#!/usr/bin/env python3
"""Compare actual paired moderate-M runs with the historical native prefix."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
import analyze_topic4_autonomous_recovery as analysis

ROOT=analysis.run.ROOT
OUT=analysis.OUT/'activity_global_pool_round3'
OLD=ROOT/'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/runs/e0_t1_s9108401'

def first_difference(x,y,clock):
    assert x.shape==y.shape,(x.shape,y.shape)
    different=np.any(x!=y,axis=tuple(range(1,x.ndim))) if x.ndim>1 else x!=y
    idx=np.flatnonzero(different)
    return dict(equal_all=not len(idx),first_difference_s=float(clock[idx[0]]) if len(idx) else None,
                maximum_absolute_difference=float(np.max(np.abs(x.astype(float)-y.astype(float)),initial=0)))

def main():
    with np.load(sorted((OLD/'chunks').glob('*.npz'))[0]) as d:old={k:d[k] for k in d.files}
    assert int(old['end_step'])==100000
    rows=[]
    for folder in sorted((OUT/'runs').glob('*mmoderate_s9108401')):
        paths=sorted((folder/'chunks').glob('*.npz'))
        if not paths:continue
        with np.load(paths[0]) as d:a={k:d[k] for k in d.files}
        assert int(a['start_step'])==0 and int(a['end_step'])==100000
        job=json.loads((OUT/'jobs'/(folder.name+'.json')).read_text())
        assert job['eta_m']==.005 and job['tau_M_s']==2 and job['seed']==9108401
        idx=np.searchsorted(old['slow_time_ms'],a['slow_time_ms']);assert np.array_equal(old['slow_time_ms'][idx],a['slow_time_ms'])
        li=np.searchsorted(old['lfp_time_ms'],a['lfp_time_ms']);assert np.array_equal(old['lfp_time_ms'][li],a['lfp_time_ms'])
        pairs={
            'all_population_spike_counts':(old['spikes_1ms'],a['spikes_1ms'],a['time_ms']/1000),
            'six_regional_spike_counts':(old['regions_1ms'],a['regions_1ms'],a['time_ms']/1000),
            'native_400_cell_field_counts':(old['field_1ms'].reshape(-1,5,400).sum(1),a['field_5ms'],a['field_time_ms']/1000),
            'fixed_neuron_raster':(old['raster'],a['raster'],np.arange(len(a['raster']))*.0001),
            'Z':(old['Z'][idx,:8],a['Z'][:,:8],a['slow_time_ms']/1000),
            'M':(old['M'][idx],a['M'],a['slow_time_ms']/1000),
            'raw_IE_II':(old['currents'][idx,:2],a['currents'][:,:2],a['slow_time_ms']/1000),
            'effective_II':(old['currents'][idx,2],a['currents'][:,2],a['slow_time_ms']/1000),
            'legacy_contact_proxy':(old['lfp_raw'][li],a['lfp_raw'],a['lfp_time_ms']/1000),
            'background_input_summaries':(old['inputs'],a['inputs'],np.arange(len(a['inputs']))*.1)}
        comparisons={k:first_difference(*v) for k,v in pairs.items()}
        pq=sorted((folder/'pool_chunks').glob('*.npz'))[0]
        with np.load(pq) as q:
            ix=np.flatnonzero(q['raw_global_current']>0)
            first=float(q['time_ms'][ix[0]]/1000) if len(ix) else None
        row=dict(name=folder.name,reference=str(OLD),window_s=[0,10],comparisons=comparisons,
            first_recorded_global_current_s=first,global_observation_resolution_s=.005,
            interpretation='Exact equality of recorded prefixes is tested. First 5ms-recorded positive feedback need not be its exact activation time; raster covers80 fixed neurons, global/regional/native-grid counts cover all cells. Existing native control is reused, not an independent replicate.')
        analysis.write(folder/'moderate_prefix_audit.json',row);rows.append(row)
    analysis.write(OUT/'moderate_prefix_audit.json',dict(updated_at=time.time(),rows=rows))
    print(json.dumps([dict(name=r['name'],global_first=r['first_recorded_global_current_s'],differences={k:v['first_difference_s'] for k,v in r['comparisons'].items()}) for r in rows]))

if __name__=='__main__':main()

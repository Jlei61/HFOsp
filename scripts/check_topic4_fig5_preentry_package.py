#!/usr/bin/env python3
"""Source, observer and snapshot checks for the bounded figure follow-up."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
import analyze_topic4_fig5_preentry_events as audit
f=audit.f;OUT=audit.OUT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    protocol=f.read(OUT/'protocol.json')
    for path,value in protocol['source_hashes'].items():assert sha(path)==value,path
    runs=[];signatures=[]
    for seed in [9108401,9108402]:
        name=f'eta0.0005_s{seed}';folder=OUT/'runs'/name;r=f.read(folder/'result.json')
        assert r['status']=='COMPLETE' and r['job']['eta_m']==.0005 and r['job']['tau_M_s']==1.
        assert r['identity']==protocol['identity'] and not r['M_reset']
        assert r['tracker']['restore_s'] is None and r['tracker']['release_s'] is None
        assert len(r['tracker']['entries'])==1
        a,r=audit.load_small(folder,keys=('spikes_1ms','regions_1ms','Z','slow_time_ms','M','inputs'))
        rate=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/320
        first=next((lo,hi) for lo,hi in f.spans(rate>=200) if hi-lo>=20)
        e=r['tracker']['entries'][0]
        assert np.isclose(first[0]*.01,e['onset_s']) and np.isclose(first[0]*.01+.2,e['confirmation_s'])
        prev=0;count=0
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as c:
                assert int(c['start_step'])==prev;prev=int(c['end_step']);count+=1
                assert np.array_equal(c['field_0p1ms'].reshape(-1,10,400).sum(1),c['field_1ms'])
                assert np.array_equal(c['population_0p1ms'].reshape(-1,10,2).sum(1),c['spikes_1ms'])
                assert np.array_equal(c['field_1ms'].sum(1),c['spikes_1ms'][:,0])
                assert np.array_equal(c['regions_1ms'][:,:3].sum(1),c['spikes_1ms'][:,0])
        signatures.append(hashlib.sha256(a['spikes_1ms'][:1000].tobytes()).hexdigest())
        runs.append(dict(name=name,status='PASS',first_onset_s=e['onset_s'],confirmation_s=e['confirmation_s'],
            duration_s=r['end_s'],continuous_chunks=count,raw_to_1ms_spatial_and_population_conservation=True,
            same_physical_topology=True,M_on=True,no_reset=True,first_second_spikes_sha256=signatures[-1]))
    assert len(set(signatures))==2,'Noise realizations must not be duplicate traces.'
    figures=[]
    for meta in sorted((OUT/'readable_fig5').glob('*/fig5_metadata.json')):
        d=f.read(meta);assert sha(d['producer'])==d['producer_sha256']
        assert d['raster']['row_order_unchanged'] and d['raster']['fixed_neurons']==80
        assert list(d['panel_mapping'])==list('ABCDEF')
        for s,link,m,tr in zip(d['snapshots'],d['state_time_links'],d['native_maps'],d['E1']['states']):
            assert s['number']==link['number']==m['number']==tr['number']
            assert np.isclose(s['time_s'],link['time_s']) and np.isclose(s['time_s'],tr['time_s'])
            assert np.isclose(np.mean(m['time_window_s']),s['time_s'])
            assert abs(link['raster_x_px']-link['ZM_x_px'])<1e-6
        a,r=audit.load_small(Path(d['source']),keys=('field_1ms','spikes_1ms','raster'))
        for m in d['native_maps']:
            lo,hi=[round(v*1000) for v in m['time_window_s']]
            assert np.allclose(a['field_1ms'][lo:hi].sum(0)/a['cell_e_counts']/.05,m['rate_Hz'])
        baseline=meta.parent.name.startswith('eta0.001_')
        if baseline:
            previous=f.read(audit.pilot.OUT.parent/'fig5_single_transition_20260914'/'raster_alignment_v2'/meta.parent.name/'fig5_metadata.json')
            for k in ['early_delta_power','groups','events']:assert previous['E2'][k]==d['E2'][k]
        figures.append(dict(name=meta.parent.name,status='PASS_DATA_CHECKS',snapshot_identity=True,
            early_power_semantics_unchanged=True,localization=d['selection']['locality_display_criterion_pass'],
            both_cores_active_in_snapshot=d['selection']['state2']['both_cores_over20Hz'],
            png_sha256=sha(meta.parent/'figures/fig5.png'),pdf_sha256=sha(meta.parent/'figures/fig5.pdf')))
    f.write(OUT/'artifact_qa.json',dict(status='PASS',runs=runs,figures=figures,
        original_physical_sources_unchanged=True,distinct_noise_realizations=True,new_runs_completed=2,new_runs_failed=0,
        tests='Native observation conservation, continuous coverage, exact high-state endpoint, source and snapshot identities.',
        human_review='PENDING'))
    print('PASS:',len(runs),'new runs;',len(figures),'figures')
if __name__=='__main__':main()

"""Check exact spike timing, native count consistency and completed deliverables."""
from pathlib import Path
import hashlib,json,re,subprocess
import numpy as np
from PIL import Image
from analyze import OUT,read,write

def main():
    validations=read(OUT/'native_validation.json')
    assert validations['completed']==16
    rows=[];identities=[]
    for folder in sorted((OUT/'per_run').glob('ee*')):
        r=read(folder/'result.json');identities.append(r['identity'])
        assert r['duration_ms']==20000 and r['runaway_ms'] is None and all(r['prefix_bitwise'].values())
        with np.load(folder/'trajectory.npz') as z:
            t=z['exact_spike_time_ms'];cells=z['exact_spike_cell']
            steps=np.rint(t/.1).astype(np.int64)
            assert np.all(np.diff(t)>=0) and np.max(abs(t-steps*.1))<1e-8
            exact=np.unique(np.column_stack(((steps-1)//20,cells)),axis=0)
            occupied=np.column_stack((np.rint(z['raster_time_ms']/2).astype(np.int64)-1,z['raster_cell']))
            assert np.array_equal(exact,occupied),r['name']
            force_recorded=None
            if r['arm']=='all_off_probe':
                sample=z['raster_sample_ids'];sample=sample[sample<len(z['core_index_E'])]
                expected=sample[z['core_index_E'][sample]==0]
                observed=cells[steps==120001]
                assert np.all(np.isin(expected,observed))
                force_recorded=len(expected)
            counts=z['spike_counts_2ms'];names=z['group_names'].tolist()
            assert int(counts[:,names.index('allE')].sum())==int(z['total_spikes_per_cell'][:32000].sum())
            assert int(counts[:,names.index('allI')].sum())==int(z['total_spikes_per_cell'][32000:].sum())
            rows.append(dict(run=r['name'],exact_spikes_recorded=len(t),occupied_bins_reconstructed=len(exact),
                exact_recording_matches_occupied_raster=True,forced_core_A_sample_cells_verified=force_recorded,
                full_E_I_counts_match_per_cell=True))
    assert len(rows)==16
    for key in ('vtheta_sha256','positions_E_sha256','core_index_sha256','ampa_topology_sha256','gaba_topology_sha256','gaba_values_sha256'):
        assert len({i[key] for i in identities})==1,key
    fig=read(OUT/'figure_manifest.json');assert len(fig['figures'])==26
    for f in fig['figures']:
        with Image.open(f['files']['png']) as im: im.load();assert min(im.size)>=900
        assert Path(f['files']['pdf']).stat().st_size>1000
    info=subprocess.check_output(['pdfinfo',str(OUT/'figures/core_burst_onset_v1_booklet.pdf')],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1));assert pages==26
    report=(OUT/'scientific_report.md').read_text();assert '\t' not in report and '\x0c' not in report
    missing=[]
    for doc in [OUT/'scientific_report.md',OUT/'README.md',OUT/'figures/README.md']:
        for url in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if not url.startswith(('http:','https:','#')) and not (doc.parent/url).exists():missing.append(url)
    assert not missing,missing
    payload=dict(status='PASS',formal_runs=16,figures=26,booklet_pages=pages,
        unchanged_geometry_threshold_and_inhibition_identity=True,recording_checks=rows,
        scientific_scope='Verification supports delivered measurements; it does not establish a noisy-population bifurcation.')
    write(OUT/'delivery_validation.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='recording_checks'}))

if __name__=='__main__':main()

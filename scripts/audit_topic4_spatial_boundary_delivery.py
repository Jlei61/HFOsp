#!/usr/bin/env python3
"""Check the completed bounded package, separately from scientific acceptance."""
from topic4_spatial_boundary_common import OUT, ROOT, read, write
from pathlib import Path
import numpy as np
import subprocess
import re
import hashlib


def main():
    statuses=['replay_status','native_batch_status','native_adaptive_status','high_duration_status',
        'rate_replay_status','rate_batch_status','input20_status','resolution_status',
        'mixed_transfer_status','mixed_timescale_status','mixed_boundary_input_status','mixed_boundary_status']
    for name in statuses:assert read(OUT/f'{name}.json')['status']=='COMPLETE',name
    native=list((OUT/'native').glob('*.json'));rate=list((OUT/'rate').glob('*.json'))
    assert len(native)==16 and len(rate)==10
    for p in native:
        r=read(p);assert r['status']=='COMPLETE' and r['frozen_Z_bitwise'] and r['count_conservation'],p
    arrays=[]
    for folder,frames in [('mixed_timescale',13680),('mixed_boundary',6000)]:
        paths=list((OUT/folder).glob('*.npz'));assert len(paths)==2
        for p in paths:
            a=np.load(p);f=a['fields_hz'];assert f.shape==(frames,2,100) and np.isfinite(f).all() and (f>=0).all()
            z=a['z'];assert np.isfinite(z).all() and z.min()>=0 and z.max()<=1
            assert np.isfinite(a['current']).all()
            arrays.append({'path':str(p.relative_to(OUT)),'frames':frames,'min_rate_hz':float(f.min()),'max_rate_hz':float(f.max())})
    # The corrected implementation is numerically checked, but its autonomous
    # matching gate must remain separate from successful execution.
    assert read(OUT/'mixed_timescale_quadrature_final_qa.json')['status']=='PASS'
    assert read(OUT/'mixed_boundary_input_status.json')['maximum_previous_input_difference']==0
    figure_paths=sorted((OUT/'figures').glob('*.pdf'));assert len(figure_paths)==12
    figures=[];readme=(OUT/'figures/README.md').read_text()
    for p in figure_paths:
        png=p.with_suffix('.png');assert png.exists() and png.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
        info=subprocess.run(['pdfinfo',str(p)],check=True,capture_output=True,text=True).stdout
        pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1]);assert pages==1
        assert p.stem in readme,p.stem
        figures.append({'name':p.name,'pages':pages,'PNG_exists':True})
    write(OUT/'pdf_structure_qa.json',{'status':'PASS','files':figures,
        'scope':'PDF readability and structure, paired PNG signatures and figure documentation; visual and human acceptance are separate.'})
    report=ROOT/'docs/archive/topic4/sustained_spatial_recruitment_boundary_2026-09-09.md'
    assert report.exists()
    producers=['validate_topic4_mixed_timescale_boundary.py','analyze_topic4_mixed_timescale_boundary.py',
        'run_topic4_mixed_timescale_validation.py','topic4_mixed_timescale_rate.py','run_topic4_spatial_boundary_rate.py',
        'audit_topic4_z_target_teacher_forcing.py','audit_topic4_mixed_input_transfer.py']
    source=[{'file':f'scripts/{name}','sha256':hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()} for name in producers]
    write(OUT/'delivery_qa.json',{'status':'PASS','stage_statuses':statuses,'native_runs':len(native),'original_rate_factorial_runs':len(rate),
        'new_arrays':arrays,'figures':figures,'sources':source,
        'scientific_acceptance':'Successful execution and file QA do not establish a matched autonomous reduction or native SNN bifurcation.'})


if __name__=='__main__':main()

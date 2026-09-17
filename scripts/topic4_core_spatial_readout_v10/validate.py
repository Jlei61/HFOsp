"""Scientific checks for the new readouts, state boundary, and figure package."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ast,hashlib,json
import numpy as np
from scipy.stats import rankdata
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/core_spatial_readout_v10_20260916'
read=lambda p:json.loads(Path(p).read_text())
def main():
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
    audit=read(OUT/'curve_audit.json');assert audit['status']=='PASS'
    assert audit['max_original_mean_change_hz']<1e-10
    assert audit['max_16384_to_32768_extrema_change_hz']<.01
    assert abs(audit['B_low_rate_projection_check']['difference_AB_hz'][1])<1e-12
    seq=read(OUT/'periodic_branches.json')
    for rows in seq:
        assert [r['continuation_order'] for r in rows]==list(range(len(rows)))
        assert len({r['branch_id'] for r in rows})==1
        for r in rows:
            assert Path(r['path']).exists()
            assert r['residual']<1e-8
            assert np.all(np.array(r['mean'])<=np.array(r['hi'])+1e-9)
            assert np.all(np.array(r['mean'])>=np.array(r['lo'])-1e-9)
    inherited=read(ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916/numerical_validation.json')
    assert inherited['status']=='PASS'
    comp=read(OUT/'pd1_comparison.json')
    assert comp['mother']['residual']<1e-9
    assert abs(comp['daughter']['T_full_ms']/comp['mother']['T_full_ms']-2)<5e-6
    assert comp['daughter']['J_exact']<comp['critical_J']<comp['mother']['J_exact']
    prov=read(OUT/'metric_provenance.json');detail=read(OUT/'native_details.json');summary=read(OUT/'native_summary.json')
    names=prov['contact_names'];identities=[];checks=[]
    for row in summary:
        key=row['native_key'];folder=OUT/'native'/key;z=np.load(folder/'trajectory.npz');res=read(folder/'result.json')
        physical=read(folder/'applied_physics.json');identities.append(physical['identity'])
        assert physical['threshold']['n_raised']==0
        blocks=physical['graph']['stage_audits']['weights']['blocks']
        for name,b in blocks.items():
            expected=row['J'] if name=='EE_same_core_scale' else 1.
            assert abs(b['weight_after']/b['weight_before']-expected)<1e-12
        assert res['duration_ms']==12000 and res['runaway_early_stop_ms'] is None
        assert hashlib.sha256((folder/'trajectory.npz').read_bytes()).hexdigest()==res['arrays_sha256']
        np.testing.assert_array_equal(z['sheet_activity_counts'].sum((1,2)),z['six_group_counts_2ms'][:,:3].sum(1))
        ids=detail[key]['valid_event_indices'];t=z['centroid_ms'][ids];m=np.isfinite(t)
        # Independently recompute raw ranks and probabilities from event arrays.
        r=np.full(t.shape,np.nan)
        for i in range(len(t)):r[i,m[i]]=(rankdata(t[i,m[i]])-1)/(m[i].sum()-1)
        q=detail[key]['summary'];target=prov['patient_target']
        np.testing.assert_allclose(np.nanmean(r,axis=0),np.array(q['mean_rank']))
        observed=[]
        for shaft in ('SCL','ICL'):
            ix=[i for i,n in enumerate(names) if n.startswith(shaft)]
            rank_error=np.mean(abs(np.nanmean(r,axis=0)[ix]-np.array(target['mean_rank'])[ix]))
            part_error=np.mean([abs(m[:,i].mean()-target['contacts'][names[i]]['participation']) for i in ix])
            delta=[]
            for i in ix:
                for j in ix:
                    if j<=i:continue
                    valid=m[:,i]&m[:,j];d=t[valid,j]-t[valid,i]
                    p=np.mean((d>0)+.5*(d==0));pair=names[i]+'→'+names[j]
                    assert abs(p-q['pairs'][pair]['order_probability'])<1e-12
                    assert len(d)==q['pairs'][pair]['n']
                    delta.append(abs(p-target['pairs'][pair]['order_probability']))
            observed.append([rank_error,np.mean(delta),part_error])
        np.testing.assert_allclose(np.mean(observed,axis=0),[row['rank_error'],row['within_shaft_order_error'],row['participation_error']],atol=1e-12,rtol=0)
        assert row['TA_count']+row['TB_count']+row['unassigned_count']==row['N_valid']
        checks.append(dict(key=key,count_conservation=True,independent_metric_recompute=True,
            N=row['N_valid'],native_replay=res['parity'],contact_count=len(names),pair_count=61))
    for name in identities[0]:
        if name!='ampa_values_sha256':assert len({r[name] for r in identities})==1,name
    mapping=read(OUT/'state_correspondence.json')
    assert len(mapping)==4 and mapping[2]['parameter_matched_native_key']==mapping[3]['parameter_matched_native_key']=='cd'
    assert all(r['spatial_state_validated'] is False for r in mapping)
    assert all(r['branch_conditioned_rank_error'] is None for r in mapping)
    sel=read(OUT/'snapshot_selection.json')
    for key in ('a','b','cd'):
        z=np.load(OUT/'native'/key/'trajectory.npz');want=[i for i in detail[key]['valid_event_indices'] if 4000<=np.nanmin(z['centroid_ms'][i])<6000]
        got=[e['event'] for e in sel['events'] if e['native_key']==key];assert got==want
        assert set(z['event_mode'][got])=={0,1}
        for e in [v for v in sel['events'] if v['native_key']==key]:
            assert all(abs(v['requested_offset_ms']-v['actual_offset_ms'])<=1+1e-8 for v in e['snapshots'])
    figures=read(OUT/'figure_manifest.json');qa=[]
    for r in figures:
        assert not r.get('text_outside_canvas',[]),(r['name'],r.get('text_outside_canvas'))
        ext=r.get('format','png');p=OUT/'figures'/f'{r["name"]}.{ext}'
        with Image.open(p) as im:
            im.load();assert list(im.size)==list(r['pixels'])
            if ext=='gif':assert im.n_frames==500
        if ext=='png':assert (OUT/'figures'/f'{r["name"]}.svg').exists()
        qa.append(dict(name=p.name,readable=True))
    (OUT/'validation.json').write_text(json.dumps(dict(status='PASS',scope='Numerical, provenance and figure checks; not native bifurcation equivalence',
        native_runs=checks,figures=qa,all_displayed_windows_include_both_frozen_labels=True,
        same_geometry_thresholds_and_inhibition=True,three_metric_values_recomputed_from_raw_events=True,
        reduced_inherited_validation='core_network_bifurcation_v7_20260916/numerical_validation.json',
        scientific_state_mapping='NOT_ESTABLISHED',human_visual_review='PENDING'),indent=2)+'\n')
    print('VALIDATION_PASS',len(qa),'figures',flush=True)

if __name__=='__main__':main()

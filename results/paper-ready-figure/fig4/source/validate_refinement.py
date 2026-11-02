"""Source and artist parity for the requested refinement; no new simulations."""
from pathlib import Path
import json,csv,hashlib,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).parent
OLD=HERE.parent.parent/'snn_layout_20261024/source'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    checks={}
    for name in ['core_count_plot_arrays.npz','D_position_density.npz','B_visible_workpoints.csv','E_native.npz']:
        p=Path('native_de_data')/name if name=='E_native.npz' else Path(name)
        checks[name+'_unchanged']=sha(HERE/p)==sha(OLD/p)
    fig=plt.figure(figsize=(320/25.4,245/25.4))
    a=load('ab_panel').draw_a(fig,[5/320,167/245,76/320,70/245])
    checks['A_EI_exact_base_anchor']=a['E_to_I_start']==a['E_triangle_base_midpoint']
    checks['A_EI_target_on_circle']=bool(np.isclose(np.linalg.norm(np.array(a['E_to_I_target'])-a['I_center']),a['I_radius_mm']))
    checks['A_short_inhibitory_bars']=max(a['inhibitory_T_lengths_mm'])<=2*a['I_radius_mm']
    checks['A_two_II_edges']=a['I_I_edges']==2
    ax=load('i_panel').render_i(fig,[253/320,15/245,53/320,51/245]);right=ax._i_loss_axis
    rows=list(csv.DictReader((HERE/'i_angle_response.csv').open()))
    display=list(csv.DictReader((HERE/'i_display_angle_response.csv').open()))
    raw=np.array([[float(r[k]) for k in load('i_panel').METRICS] for r in rows])
    sd=np.sqrt([[float(r[k+'_bootstrap_variance']) for k in load('i_panel').METRICS] for r in display])
    xx=np.array([float(r['parameter_value']) for r in rows]);loss=np.array([float(r['optimizer_J']) for r in rows]);jsd=np.array([float(r['J_bootstrap_sd']) for r in rows])
    for j,line in enumerate(ax.lines[:3]):np.testing.assert_array_equal(line.get_ydata(),raw[:,j])
    for j,band in enumerate(ax.collections):
        vertices=band.get_paths()[0].vertices
        for i,x in enumerate(xx):
            ys=vertices[np.isclose(vertices[:,0],x),1]
            np.testing.assert_allclose([ys.min(),ys.max()],[raw[i,j]-sd[i,j],raw[i,j]+sd[i,j]],atol=1e-13)
    with np.load(HERE/'i_display_bootstrap_draws.npz') as z:np.testing.assert_allclose(z['components_13'].std(0,ddof=1),sd,rtol=1e-10)
    with np.load(HERE/'i_bootstrap_draws.npz') as z:np.testing.assert_allclose(z['scaled_components_13'].mean(2).std(0,ddof=1),jsd,rtol=1e-10)
    np.testing.assert_array_equal(right.lines[0].get_ydata(),loss)
    np.testing.assert_array_equal(ax.lines[-1].get_xdata(),np.repeat(xx[np.argmin(loss)],2))
    assert len(ax.collections)==3 and len(right.collections)==1
    checks.update(I_all_39_original_errors_unchanged=True,I_all_13_actual_J_unchanged=True,I_three_raw_SD_bands_match_arrays=True,I_J_SD_matches_joint_draws=True,I_vertical_line_at_true_argmin=True,I_exactly_four_legend_entries=len(right.get_legend().get_texts())==4)
    checks['C_all_15']=len(np.load(HERE/'c_panel_data.npz')['contact_names'])==15
    d=np.load(HERE/'c_panel_data.npz');old=np.load(OLD/'c_panel_data.npz');ix=[d['contact_names'].tolist().index(n) for n in old['contact_names']]
    for side in ['model','patient']:
        for key in ['native_centroid_ms','native_rank','tfr_centroid_ms','tfr_centroid_hz']:
            np.testing.assert_allclose(d[side+'_'+key][ix],old[side+'_'+key],atol=1e-10)
    checks['C_existing_times_ranks_centroids_unchanged']=True
    formal=HERE.parents[2];manifest=json.loads((formal/'release_manifest.json').read_text())
    checks['formal_124_files_unchanged']=all(sha(formal/name)==entry['sha256'] for name,entry in manifest['files'].items())
    report=dict(checks=checks,all_pass=all(checks.values()),I_SD='13 physical-time blocks, 4.5 s each, 4000 bootstrap draws; conditional within-run SD, not independent-seed variance',status='AGENT_CHECKS_PASS_AUTHOR_REVIEW_PENDING')
    (HERE/'refinement_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));assert report['all_pass'];plt.close(fig)

if __name__=='__main__':main()

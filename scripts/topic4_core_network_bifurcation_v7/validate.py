"""Scientific and output checks for the explicitly completed v7 scope."""
from common import *
import numpy as np,hashlib

def main():
    native=read(OUT/'native_coordinates.json');assert sorted(r['number'] for r in native)==list(range(1,21))
    nr=[]
    for r in native:
        path=Path(r['path']);z=np.load(path);meta=read(path.with_name('result.json'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        assert r['window_s']==[2.,12.] and all(x is not None for x in r['means'])
        if r['number']<=4:continue
        assert meta['status']=='COMPLETE' and meta['actual_duration_ms']==12000 and meta['runaway_early_stop_ms'] is None
        assert meta['threshold_raised']==0 and meta['seed']==848101 and meta['topology']==2511
        names=z['group_names'].tolist();six=z['six_group_counts_2ms'];counts=z['spike_counts_2ms']
        assert np.array_equal(six[:,:3].sum(1),counts[:,names.index('allE')]);assert np.array_equal(six[:,3:].sum(1),counts[:,names.index('allI')])
        samples=z['raster_sample_ids'];region=z['region'];assert np.array_equal(np.bincount(region[samples],minlength=6),np.full(6,100))
        assert np.isin(z['exact_spike_cell'],samples).all();assert z['exact_spike_time_ms'].min()>=0 and z['exact_spike_time_ms'].max()<=12000.01
        nr.append(dict(number=r['number'],J=r['g'],duration_ms=meta['actual_duration_ms'],six_group_count_conservation=True,recorded_cells=600,threshold_raised=0))
    assert read(OUT/'native_prefix_validation.json')['status']=='PASS'
    children=read(OUT/'secondary_flip_validation.json');assert children['status']=='PASS'
    folds=read(OUT/'new_fold_validation.json');assert folds['status']=='PASS' and len(folds['folds'])==3
    low=read(OUT/'poincare/low_stable_grid/g1.12182298_N4096/rk4_orthogonal_dt0.0125.json');assert low['max_transverse']<1 and low['orbit_tangent_defect']<1e-3
    p4=[]
    for amp,folder in [('2e-05','surround_period4_refined'),('4e-05','surround_period4')]:
        q=read(OUT/f'poincare/{folder}/amp{amp}_N8192/rk4_orthogonal_dt0.025.json');assert q['max_transverse']<1 and q['orbit_tangent_defect']<1e-3;p4.append(q)
    pd=read(OUT/'poincare/flips/surround_2T_flip_N4096/rk4_orthogonal_dt0.0125.json');assert abs(pd['multipliers'][0][0]+1)<5e-4
    spectral=[]
    for index in (40,55):
        folder=next((OUT/'shifted_floquet').glob(f'point{index:03d}*'))
        a=read(folder/'alpha66_dt0.025.json');b=read(folder/'alpha66_dt0.0125.json');c=read(folder/'alpha40_dt0.025.json')
        err=abs(a['growth_per_s'][0]-b['growth_per_s'][0]);alphaerr=abs(a['growth_per_s'][0]-c['growth_per_s'][0]);assert err<1e-4 and alphaerr<1e-4
        assert a['growth_per_s'][0]>1 and max(a['relative_residual'])<1e-8
        spectral.append(dict(source=a['source'],growth_per_s=b['growth_per_s'][0],angle_rad=b['angle_rad'][0],step_difference_per_s=err,shift_difference_per_s=alphaerr,scope='Leading growth of this discretized orbit only; secondary multipliers not accepted'))
    h=read(OUT/'homoclinic_audit.json');assert h['grid']['J_difference']<1e-11 and h['grid']['mean_difference_hz']<1e-7
    condition=read(OUT/'condition_orbit_validation.json');assert len(condition)==17 and max(r['offgrid_hz'] for r in condition)<.001
    assert len(read(OUT/'native_figure_manifest.json'))==40 and len(read(OUT/'joint_critical_gallery.json'))==12
    rc=read(OUT/'reduced_condition_coordinates.json');assert len(rc)==24
    pairs=read(OUT/'native_paired_comparison.json');assert len(pairs)==6
    arcs=read(OUT/'arc_offgrid_validation.json')
    analytic=read(OUT/'analytic_critical_mode_validation.json');assert len(analytic)==12 and max(r['analytic_right_null_max'] for r in analytic)<1e-6
    gain=read(OUT/'analytic_gain_validation.json');assert gain['analytic_chain_rule_error']<1e-12
    tail=read(OUT/'long_tail_stability_validation.json') if (OUT/'long_tail_stability_validation.json').exists() else dict(status='NOT_ESTIMABLE_YET')
    row=dict(status='PASS',scope='Declared 20-condition native atlas, accepted local critical points and finite continuations; not global branch completeness or native bifurcation equivalence',
        native_conditions=20,new_native_runs=nr,native_pair_count=len(pairs),native_figure_count=40,reduced_waveform_count=24,critical_orbit_count=12,
        prefix=read(OUT/'native_prefix_validation.json'),cycle_folds=folds,secondary_flip=children,critical_flip_return=pd,
        stable_four_cycle_return=p4,stable_low_cycle_return=low,shifted_leading_spectrum=spectral,
        homoclinic_grid=h['grid'],analytic_critical_modes=analytic,analytic_gain_check=gain,long_tail_stability=tail,condition_orbit_max_offgrid_hz=max(r['offgrid_hz'] for r in condition),
        distant_continuation_max_offgrid_hz=max(r['offgrid_hz'] for r in arcs),
        scientific_limits=['Homoclinic limit is finite-period numerical evidence, not an infinite-time connection BVP.',
            'Recruited family global endpoint remains undetermined after 120 further arc points.',
            'Distant N2048 continuation is a diagnostic at <0.009 Hz off-grid defect; do not assign microcritical precision to all its points.',
            'Strongly unstable secondary multipliers are not precision accepted.',
            'Native cold-start single-seed parameter atlas is not a native bifurcation proof.'],human_visual_acceptance='PENDING')
    write('numerical_validation.json',row);print('VALIDATED 20 native conditions, 12 critical orbits, 24 reduced correspondences',flush=True)

if __name__=='__main__':main()

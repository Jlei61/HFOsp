"""Compare canonical finite-window readouts for the same0.5mm state at two steps."""
from native_spatial_refinement_audit import canonical, broad
from native_spatial_refinement import DEST, restriction_and_parent
from common import *
from scipy import sparse


def main():
    assert read(DEST/'time_refinement_run.json')['status']=='COMPLETE'
    assert read(DEST/'time_refinement_preparation.json')['status']=='PASS'
    s20=model(20);s40=model(40);Q,parent=restriction_and_parent(s20,s40)
    fine_count=np.bincount(s40.geo['group_cell'][s40.E],weights=s40.sizes[s40.E],minlength=1600)
    coarse_count=np.bincount(s20.geo['group_cell'][s20.E],weights=s20.sizes[s20.E],minlength=400)
    pixel_parent=np.full(1600,-1,dtype=int)
    for i in np.flatnonzero(fine_count):
        values=np.unique(s20.geo['group_cell'][parent[s40.geo['group_cell']==i]])
        assert len(values)==1;pixel_parent[i]=values[0]
    used=fine_count>0
    restrict=sparse.csr_matrix((fine_count[used]/coarse_count[pixel_parent[used]],
        (pixel_parent[used],np.flatnonzero(used))),shape=(400,1600))
    assert np.array_equal(np.bincount(pixel_parent[used],weights=fine_count[used],minlength=400),coarse_count)
    initial_large=np.load(DEST/'initial_g40.npz');initial_small=np.load(DEST/'initial_g40_dt0.025.npz')
    assert np.array_equal(initial_large['state'],initial_small['state'])
    i=(-np.arange(len(initial_large['history'])))%len(initial_large['history'])
    j=(-2*np.arange(len(initial_large['history'])))%len(initial_small['history'])
    assert np.array_equal(initial_large['history'][i],initial_small['history'][j])
    rows=[];Z=None
    for dt in [.05,.025]:
        folder=OUT/'runs'/f'native_g40_liftedZ_D0.2190000_dt{dt}'
        c=read(folder/'contract.json');z=np.load(folder/'trajectory.npz')
        assert c['duration_ms']==12000 and c['dt_ms']==dt
        assert c['Z']=='held' and c['M']=='dynamic'
        assert c['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
        expected=DEST/('initial_g40.npz' if dt==.05 else 'initial_g40_dt0.025.npz')
        assert Path(c['initial']).resolve()==expected.resolve()
        if Z is None:Z=z['Z_source']
        assert np.array_equal(Z,z['Z_source'])
        assert np.all(z['Z_every50ms']==Z) and np.array_equal(z['final_state'][11],Z)
        assert np.array_equal(z['time_ms'],np.arange(12000)+1)
        field=z['field_E_hz'].astype(float);counts=z['cell_counts']
        assert np.array_equal(counts,fine_count)
        common=(restrict@field.T).T
        path=DEST/(folder.name+'_common400.npz')
        np.savez_compressed(path,field_E_hz=common,cell_counts=coarse_count)
        rows.append(dict(dt_ms=dt,source=str(folder/'trajectory.npz'),
            canonical_native=canonical(folder),canonical_common400=canonical(path),
            broad_native_start_ms=broad(field,counts),broad_common400_start_ms=broad(common,coarse_count)))
    same=all(rows[0][key]['category']==rows[1][key]['category'] for key in ['canonical_native','canonical_common400'])
    high=all(r['canonical_native']['high_rate'] is None for r in rows)
    q=dict(status='QUALITATIVE_STEP_CHECK_PASS' if same and high else 'STEP_SENSITIVITY_UNRESOLVED',
        rows=rows,same_tail_category=same,both_no_original_high_entry=high,
        tail_mean_difference_hz=rows[1]['canonical_native']['tail']['mean_rate_hz']-rows[0]['canonical_native']['tail']['mean_rate_hz'],
        scope='Same finite-window category only; no trajectory, distribution, spatial-grid, asymptotic or bifurcation convergence claim.')
    write(DEST/'time_refinement_result.json',q);log('SPATIAL TIME AUDIT',q['status'],q['tail_mean_difference_hz'])


if __name__=='__main__':main()

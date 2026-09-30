"""Independent finite-window audit of the same-graph spatial refinement.

Compare at native resolution and after E-count-weighted restriction to the
common coarse pixels. Neither classification certifies a bifurcation.
"""
from native_spatial_refinement import DEST, INITIAL, restriction_and_parent
from common import *
from scipy import sparse
import argparse
import subprocess
import json


def canonical(path):
    return json.loads(subprocess.check_output(
        [sys.executable,str(HERE/'canonical_case_readout.py'),str(path)],text=True))


def broad(field,counts):
    cell=field.reshape(-1,10,field.shape[1]).mean(1)
    w=counts/counts.sum();rate=cell@w;area=(cell>=50)@w
    edges=np.diff(np.r_[0,((rate>=200)&(area>=.75)).astype(int),0])
    return next((int(a*10) for a,b in zip(np.flatnonzero(edges==1),
        np.flatnonzero(edges==-1)) if b-a>=20),None)


def main(partial=False):
    c=read(OUT/'native_spatial_refinement_contract.json')
    assert read(DEST/'preparation.json')['status']=='PREPARATION_PASS'
    assert read(DEST/'coarse_prefix_replay.json')['status']=='PASS'
    s20=model(20);s40=model(40);Q,parent=restriction_and_parent(s20,s40)
    source=np.load(INITIAL);lift=np.load(DEST/'initial_g40.npz')
    assert np.array_equal(lift['state'],source['state'][:,parent])
    assert np.array_equal(lift['history'],source['history'][:,parent])
    # Pixel correspondence is recovered from member groups, not assumed from
    # flattened image order. Empty fine pixels carry zero population weight.
    pixel_parent=np.full(1600,-1,dtype=int)
    for i in range(1600):
        groups=np.flatnonzero(s40.geo['group_cell']==i)
        values=np.unique(s20.geo['group_cell'][parent[groups]])
        if len(values):
            assert len(values)==1
            pixel_parent[i]=values[0]
    fine_count=np.bincount(s40.geo['group_cell'][s40.E],weights=s40.sizes[s40.E],minlength=1600)
    coarse_count=np.bincount(s20.geo['group_cell'][s20.E],weights=s20.sizes[s20.E],minlength=400)
    used=fine_count>0
    assert np.all(pixel_parent[used]>=0)
    recover=np.bincount(pixel_parent[used],weights=fine_count[used],minlength=400)
    assert np.array_equal(recover,coarse_count)
    restrict=sparse.csr_matrix((fine_count[used]/coarse_count[pixel_parent[used]],
        (pixel_parent[used],np.flatnonzero(used))),shape=(400,1600))
    arms=[dict(D=d,Z_detail='coarse') for d in c['D']]+c['new_arms']
    rows=[];pending=[]
    for arm in arms:
        d=arm['D'];detail=arm['Z_detail'];fine=detail!='coarse'
        name=f'native_g40_{detail}Z_D{d:.7f}_dt0.05' if fine else f'endpoint_D{d:.7f}_dt0.05'
        folder=OUT/'runs'/name
        if not (folder/'result.json').exists():
            pending.append(name);continue
        run=read(folder/'contract.json');z=np.load(folder/'trajectory.npz')
        expected=DEST/'initial_g40.npz' if fine else INITIAL
        assert Path(run['initial']).resolve()==expected.resolve()
        assert run['duration_ms']==12000 and run['dt_ms']==.05
        assert run['Z']=='held' and run['M']=='dynamic'
        assert run['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
        assert abs(run['D_initial']-d)<2e-14
        assert np.array_equal(z['time_ms'],np.arange(12000)+1)
        assert np.all(z['Z_every50ms']==z['Z_source'])
        assert np.array_equal(z['final_state'][11],z['Z_source'])
        field=z['field_E_hz'].astype(float);counts=z['cell_counts']
        assert np.array_equal(counts,fine_count if fine else coarse_count)
        m_change=float(np.max(abs(z['M_every50ms']-z['M_every50ms'][0])))
        assert m_change>1e-10,'M unexpectedly constant'
        native=canonical(folder)
        common_field=(restrict@field.T).T if fine else field
        global_difference=float(np.max(abs(common_field@(coarse_count/coarse_count.sum())-field@(counts/counts.sum()))))
        assert global_difference<1e-10
        dest=DEST/(name+'_common400.npz')
        np.savez_compressed(dest,field_E_hz=common_field,cell_counts=coarse_count)
        common=canonical(dest)
        rows.append(dict(label=name,D=d,Z_mean=1-d,grid=40 if fine else 20,
            Z_detail=detail,canonical_native_resolution=native,canonical_common400=common,
            broad_native_start_ms=broad(field,counts),broad_common400_start_ms=broad(common_field,coarse_count),
            common400_global_rate_preservation_error=global_difference,M_max_change=m_change,
            source=str(folder/'trajectory.npz'),common_source=str(dest)))
        log('SPATIAL AUDIT',name,native['category'],common['category'],native['tail']['mean_rate_hz'])
    if not partial:assert not pending,pending
    comparisons=[]
    for d in c['D']:
        match={r['Z_detail']:r for r in rows if r['D']==d}
        for key in ['lifted','native']:
            if key not in match:continue
            coarse=match['coarse'];fine=match[key]
            comparisons.append(dict(D=d,Z_detail=key,
                same_category_native_resolution=coarse['canonical_native_resolution']['category']==fine['canonical_native_resolution']['category'],
                same_category_common400=coarse['canonical_common400']['category']==fine['canonical_common400']['category'],
                coarse_mean_hz=coarse['canonical_common400']['tail']['mean_rate_hz'],
                fine_mean_hz=fine['canonical_common400']['tail']['mean_rate_hz']))
    payload=dict(status='PARTIAL' if pending else 'COMPLETE',rows=rows,pending=pending,comparisons=comparisons,
        model='Frozen v3, same original physical graph, Z held and M dynamic',
        statistical_unit='One deterministic complete-history continuation per grid/Z-field condition',
        readout='Canonical last 4 s, 10 ms nonoverlapping bins; additional common400-pixel restriction',
        scope='Spatial robustness of these finite-window states only. Not critical-D convergence, bifurcation type, asymptotic classification, or SNN correspondence.')
    write(DEST/('partial_audit.json' if pending else 'result.json'),payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    main(parser.parse_args().partial)

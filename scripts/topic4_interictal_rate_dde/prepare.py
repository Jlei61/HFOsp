"""Conservative coarsening of the already verified realized-graph projection."""
from common import *
from scipy import sparse
import argparse

def main(grid):
    parent=FIELD/'operators/g80_theta0.25';g=dict(np.load(parent/'geometry.npz'));prep=read(parent/'prepared.json')
    destination=BASE/f'operators/g{grid}';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'prepared.json').exists():return
    ratio=80//grid;assert grid*ratio==80
    cells=(g['group_cell']%80)//ratio+grid*((g['group_cell']//80)//ratio)
    key=np.column_stack([g['population'],cells,g['group_region'],np.floor((18-g['threshold_mv'])/.5+1e-8).astype(int)])
    unique,group=np.unique(key,axis=0,return_inverse=True);P=len(unique);size=np.bincount(group,weights=g['group_size'])
    native_group=group[g['cell_group']];original_theta=g['threshold_mv'][g['cell_group']]+g['threshold_error_mv']
    theta=np.bincount(native_group,weights=original_theta)/size
    contacts=np.stack([np.bincount(group,weights=g['contact_rate_weights'][:,k],minlength=P) for k in range(15)],axis=1)
    xy=np.stack([np.bincount(native_group,weights=g['original_positions'][:,k])/size for k in range(2)],axis=1)
    np.savez_compressed(destination/'geometry.npz',cell_group=native_group,group_size=size,population=unique[:,0].astype(np.uint8),
        group_cell=unique[:,1],group_region=unique[:,2],threshold_mv=theta,positions=xy,contact_rate_weights=contacts,
        original_positions=g['original_positions'],centers_mm=g['centers_mm'])
    records=[];rng=np.random.default_rng(190971)
    for kind in ['ampa','gaba']:
        matrix=sparse.load_npz(parent/f'delay_{kind}.npz').tocoo();oldP=len(g['group_size'])
        d=matrix.col//oldP;s=matrix.col%oldP
        weight=matrix.data*g['group_size'][matrix.row]/size[group[matrix.row]]
        reduced=sparse.coo_matrix((weight,(group[matrix.row],group[s]+d*P)),shape=(P,P*prep['max_delay_steps'])).tocsr()
        # Constant source population activity preserves every target mean and
        # every original delay; this checks projection, not dynamical fidelity.
        current=np.asarray(matrix.tocsr().sum(1)).ravel()
        expected=np.bincount(group,weights=current*g['group_size'])/size
        err=np.max(abs(np.asarray(reduced.sum(1)).ravel()-expected));assert err<1e-10
        sparse.save_npz(destination/f'delay_{kind}.npz',reduced)
        coo=reduced.tocoo();dc=sparse.coo_matrix((coo.data,(coo.row,coo.col%P)),shape=(P,P)).tocsr()
        sparse.save_npz(destination/f'dc_{kind}.npz',dc)
        records.append(dict(kind=kind,terms=reduced.nnz,incoming_sum_error=err))
    assert np.allclose(contacts.sum(0),1) and size.sum()==40000
    write(destination/'prepared.json',dict(**prep,rate_groups=P,rate_grid=grid,rate_theta_width_mv=.5,
        source=str(parent),projection_checks_new=records,
        definition='Space x E/I x core membership x empirical threshold bins; each unit has continuous rate/filter states, no sampled members',
        mathematical_target='Autonomous delay-rate vector field with analytic directional derivatives and equilibrium characteristic matrix'))
    print('PREPARED',grid,P,records,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=20);a=ap.parse_args();main(a.grid)


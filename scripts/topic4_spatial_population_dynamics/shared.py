"""Second-stage spatial population bridge; original SNN and observer frozen."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_spatial_reduction_validation'))
from common import np,json,runtime,read,write,safe,smooth_contacts,observations,V10
PRIOR=ROOT/'results/topic4_sef_hfo/spatial_reduction_validation_20260916'
OUT=ROOT/'results/topic4_sef_hfo/spatial_population_dynamics_20260916'
J=1.355

def make_partition(finer_background=False):
    old=np.load(PRIOR/'grid20/model.npz');z=np.load(PRIOR/'adaptive_readout.npz')
    region=np.load(V10/'native/a/trajectory.npz')['region'].astype(np.int64)
    positions=old['positions'];tiles=z['tiles']
    if finer_background:
        tiles=np.array([
            item for x,y,w in tiles for item in ([[x+dx,y+dy,1.] for dy in (0,1) for dx in (0,1)] if w==2 else [[x,y,w]])])
    lookup=np.empty((40,40),np.int64)
    for k,(x,y,w) in enumerate(tiles):
        i,j=np.round(np.array([x,y])*2).astype(int);n=round(w*2);lookup[j:j+n,i:i+n]=k
    ij=np.minimum((positions*2).astype(int),39);tile=lookup[ij[:,1],ij[:,0]]
    labels,group=np.unique(region*len(tiles)+tile,return_inverse=True)
    assert np.array_equal((labels//len(tiles))[group],region)
    if not finer_background:
        assert np.array_equal(labels//len(tiles),z['region'])
        assert np.array_equal(np.bincount(group),z['count'])
    return group,region,positions,tiles

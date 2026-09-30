"""Third-stage paths; original data and second-stage results stay read-only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_spatial_population_dynamics'))
from shared import np,json,runtime,read,write,safe,smooth_contacts,observations,V10,PRIOR,J
BASE=ROOT/'results/topic4_sef_hfo/spatial_population_dynamics_20260916'
OUT=ROOT/'results/topic4_sef_hfo/spatial_source_projection_20260916'

def operator_directory(partition='adaptive1'):return OUT if partition=='adaptive1' else OUT/'half_grid'

def model(partition='adaptive1'):return np.load(BASE/'model_adaptive1.npz' if partition=='adaptive1' else OUT/'half_grid/model.npz')

def make_half_model():
    folder=operator_directory('half');folder.mkdir(exist_ok=True)
    if (folder/'model.npz').exists():return
    old=model();data={k:old[k] for k in old.files};pos=old['positions'];region=old['region']
    tiles=np.array([[x+dx,y+dy,w/2] for x,y,w in old['tiles'] for dy in [0,w/2] for dx in [0,w/2]])
    lookup=np.full((80,80),-1,np.int64)
    for k,(x,y,w) in enumerate(tiles):
        i,j=np.rint(np.array([x,y])*4).astype(int);n=round(w*4);lookup[j:j+n,i:i+n]=k
    assert (lookup>=0).all();ij=np.minimum((pos*4).astype(int),79);tile=lookup[ij[:,1],ij[:,0]]
    labels,group=np.unique(region*len(tiles)+tile,return_inverse=True);count=np.bincount(group);order=np.argsort(group,kind='stable');ptr=np.r_[0,np.cumsum(count)]
    inverse=np.empty(len(order),int);inverse[old['order']]=np.arange(len(order))
    for key in ['vtheta','region_sorted','weights_sorted','field_sorted','ext_index','raster_index']:data[key]=old[key][inverse[order]]
    weights=data['weights_sorted'];original_weights=np.empty_like(weights);original_weights[order]=weights
    data.update(group=group,count=count,order=order,ptr=ptr,tiles=tiles,
        contact_weights=np.stack([np.bincount(group,weights=original_weights[:,j],minlength=len(count)) for j in range(15)]))
    parent=np.array([old['group'][order[ptr[a]]] for a in range(len(count))]);assert np.array_equal(parent[group],old['group'])
    data['parent_group']=parent;np.savez_compressed(folder/'model.npz',**data)
    write(folder/'partition.json',dict(groups=len(count),tiles=len(tiles),nested_parent=True,
        widths_mm=sorted(np.unique(tiles[:,2]).tolist()),source_refinement_only=True))

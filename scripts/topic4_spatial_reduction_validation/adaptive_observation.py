"""Geometry-only adaptive partition; validate its readout on actual SNN spikes.

This is a discretization diagnostic, not a new autonomous dynamics result.
"""
from common import *
from analyze import metric_distance
from scipy import sparse

def touches(lo,width,points,radii):
    nearest=np.maximum(lo,np.minimum(points,lo+width))
    return bool(np.any(np.linalg.norm(points-nearest,axis=1)<=radii))

def main():
    cfg=read(OUT/'model_config.json');m=np.load(OUT/'grid20/model.npz')
    xy=m['contact_xy'];centers=np.array(cfg['core_centers']);radii=np.array(cfg['core_radii'])
    # Contact kernel sigma .25 mm: refine every intersecting 3-sigma footprint.
    tiles=[];fine_to_tile=np.empty((40,40),int)
    for j in range(10):
        for i in range(10):
            lo=np.array([i,j],float)*2
            if not (touches(lo,2.,xy,.75) or touches(lo,2.,centers,radii)):
                cells=[(lo,2.)]
            else:
                cells=[]
                for jj in range(2):
                    for ii in range(2):
                        sub=lo+np.array([ii,jj])
                        if touches(sub,1.,xy,.75):
                            cells.extend((sub+np.array([iii,jjj])*.5,.5) for jjj in range(2) for iii in range(2))
                        else:cells.append((sub,1.))
            for corner,width in cells:
                ix,iy=np.round(corner*2).astype(int);size=round(width*2)
                fine_to_tile[iy:iy+size,ix:ix+size]=len(tiles);tiles.append([*corner,width])
    first=np.load(OUT/'native/848101/trajectory.npz');region=first['region_40'];cell=first['cell_40'];T=len(tiles)
    labels,inverse=np.unique(region*T+fine_to_tile.ravel()[cell],return_inverse=True)
    reg=labels//T;tile=labels%T;P=len(labels)
    op=sparse.coo_matrix((np.ones(len(inverse)),(np.arange(len(inverse)),inverse)),shape=(len(inverse),P)).tocsr()
    count=np.bincount(inverse[first['group_40']],minlength=P)
    assert np.array_equal(reg[inverse[first['group_40']]],region[first['group_40']])
    weights=first['contact_weights_40']@op
    assert np.allclose(weights.sum(1),1.)
    protocol=dict(status='DEFINED_BEFORE_ADAPTIVE_READOUT_SCORING',
        geometry_rule='2 mm outside; 1 mm boxes intersecting either core disk; 0.5 mm boxes intersecting a contact 3-sigma footprint',
        contact_sigma_mm=.25,contact_refinement_radius_mm=.75,
        physical_tiles=T,nonempty_populations=P,split='space x core identity x E/I',
        intended_use='candidate spatial discretization; no autonomous dynamics or propagation-field accuracy established')
    write(OUT/'adaptive_readout_protocol.json',protocol)
    rows=[]
    for seed in SEEDS:
        z=np.load(OUT/'native'/str(seed)/'trajectory.npz')
        counts=np.asarray(z['counts_40']@op)
        for j in range(6):assert np.array_equal(counts[:,reg==j].sum(1),z['six_counts'][:,j])
        env=smooth_contacts((counts/count)@weights.T)
        ob,ids,mu,q=observations(env);native=read(OUT/f'summaries/native_{seed}.json')
        err=metric_distance(q,native['summary'])
        rows.append(dict(seed=seed,N=len(ids),errors=err))
        write(OUT/f'summaries/adaptive_{seed}.json',dict(N=len(ids),summary=q,errors_to_same_seed=err))
        if seed==848101:
            np.savez_compressed(OUT/'adaptive_readout.npz',tiles=np.array(tiles),region=reg,tile=tile,
                count=count,contact_envelope=env,contact_weights=weights,contact_xy=xy)
    write(OUT/'adaptive_readout.json',dict(protocol=protocol,rows=rows,
        native_pair_max=read(OUT/'comparison.json')['native_pair_max'],
        limitations='Observed native spikes only. Not a dynamics model; contact agreement does not prove 2D propagation-field agreement.'))
    print(json.dumps(safe(rows),indent=2),flush=True)

if __name__=='__main__':main()

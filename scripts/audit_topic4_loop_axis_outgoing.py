#!/usr/bin/env python3
"""Quantify unmatched output degree and pathological-source allocation in axis controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from scipy import sparse
import run_topic4_loop_axis_native as axis
import topic4_historical_manual_z_common as substrate


def main():
    with np.load(substrate.OUT/'substrate.npz') as z:v=z['vtheta'];pos=z['positions_e']
    identity=axis.native.base.read(axis.native.OUT/'protocol.json')['identity']
    assert substrate.array_sha256(np.asarray(v,np.float32))==identity['vtheta_sha256']
    with np.load(axis.native.OUT/'geometry.npz') as z:
        centers=z['centers_mm'];assert np.array_equal(pos,z['positions_e'])
    lowered=v[:32000]<18;dist=np.linalg.norm(pos[:,None]-centers[None],axis=2)
    groups={'AllE':np.ones(32000,bool),'lowered_E':lowered,'CoreA_observer':dist[:,0]<1.75,
            'CoreB_observer':dist[:,1]<1.75,'OtherE':(dist>=1.75).all(1)}
    rows=[];baseline=None
    for condition in ['reference','rotated','isotropic']:
        outw=np.zeros(32000);outd=np.zeros(32000);ins=np.zeros(32000);inlow=np.zeros(32000);invth=np.zeros(32000)
        for path in sorted((axis.graph_folder(condition)/'ampa_by_delay').glob('*.npz')):
            m=sparse.load_npz(path).tocoo();mask=m.row<32000;rr,cc,w=m.row[mask],m.col[mask],m.data[mask]
            outw+=np.bincount(cc,weights=w,minlength=32000);outd+=np.bincount(cc,minlength=32000)
            ins+=np.bincount(rr,weights=w,minlength=32000)
            inlow+=np.bincount(rr,weights=w*lowered[cc],minlength=32000)
            invth+=np.bincount(rr,weights=w*v[cc],minlength=32000)
        row=dict(condition=condition,lowered_cells=int(lowered.sum()),
            outdegree_quantiles=np.quantile(outd,[0,.1,.5,.9,1]).tolist(),
            lowered_source_fraction_of_all_EE_weight=float(outw[lowered].sum()/outw.sum()),groups={})
        for name,mask in groups.items():
            row['groups'][name]=dict(cells=int(mask.sum()),mean_outgoing_strength=float(outw[mask].mean()),
                mean_incoming_fraction_from_lowered_sources=float((inlow/ins)[mask].mean()),
                incoming_weighted_mean_source_threshold_mV=float((invth/ins)[mask].mean()))
        if baseline is None:baseline=(outw.copy(),outd.copy(),inlow/ins)
        else:
            row.update(outstrength_correlation_with_original=float(np.corrcoef(outw,baseline[0])[0,1]),
                median_absolute_incoming_lowered_fraction_change=float(np.median(abs(inlow/ins-baseline[2]))),
                maximum_absolute_incoming_lowered_fraction_change=float(np.max(abs(inlow/ins-baseline[2]))))
        rows.append(row)
        np.savez_compressed(axis.ROOT/f'outgoing_audit_{condition}.npz',outgoing_weight=outw,outdegree=outd,
            incoming_lowered_fraction=inlow/ins,incoming_source_threshold=invth/ins,lowered_mask=lowered)
        print(condition,row['lowered_source_fraction_of_all_EE_weight'],row['groups']['lowered_E'],flush=True)
    axis.native.write(axis.ROOT/'outgoing_and_excitability_audit.json',dict(status='DESCRIPTIVE_CONTROL_LIMIT_AUDIT',
        rows=rows,threshold_identity=identity['vtheta_sha256'],
        scope='Source outdegree and pathological-source allocation were not fixed by target/source-observer-region/delay incoming weight multisets. This audit does not prove the cause of a dynamic difference.'))


if __name__=='__main__':main()

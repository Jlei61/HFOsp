#!/usr/bin/env python3
"""Matched angular partner reassignment, retaining incoming weight multisets."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from topic4_historical_manual_z_common import setup
from prepare_topic4_loop_axis_controls import geometry, write, OUT as PARENT
from src.topic4_multidimensional_parameters import sparse_digest

OUT=PARENT/'angular_reassignment'


def incoming(bins,ne):
    rr=[];cc=[];ww=[];dd=[]
    for delay,mat in enumerate(bins):
        coo=mat.tocoo();m=coo.row<ne
        rr.append(coo.row[m]);cc.append(coo.col[m]);ww.append(coo.data[m])
        dd.append(np.full(np.count_nonzero(m),delay,np.int16))
    r=np.concatenate(rr);c=np.concatenate(cc);w=np.concatenate(ww);d=np.concatenate(dd)
    order=np.argsort(r,kind='stable')
    counts=np.bincount(r,minlength=ne)
    return np.r_[0,np.cumsum(counts)],c[order],w[order],d[order]


def row_reassign(target,source,weight,delay,positions,regions,p,condition):
    d=positions-positions[target]
    distance=np.linalg.norm(d,axis=1)
    step=max(1,round(p.delay_dt/p.dt))
    candidate_delay=np.maximum(1,np.rint((p.tau0+distance/p.v_axon)/p.delay_dt).astype(int))*step
    assert np.array_equal(candidate_delay[source],delay), 'Original edge delay does not match physical distance'
    assert target not in source and len(np.unique(source))==len(source)
    if condition=='reference':
        return source.copy(),np.zeros(len(source))
    code=candidate_delay*3+regions
    wanted=delay.astype(int)*3+regions[source]
    eligible=np.flatnonzero((candidate_delay<=delay.max()) & (np.arange(len(positions))!=target))
    order=eligible[np.argsort(code[eligible],kind='stable')]
    codes=code[order]
    angle=np.arctan2(d[:,1],d[:,0])
    rng=np.random.default_rng(92924000+target)
    result=np.empty_like(source);angle_error=np.empty(len(source))
    for group in np.unique(wanted):
        slots=np.flatnonzero(wanted==group)
        pool=order[np.searchsorted(codes,group,'left'):np.searchsorted(codes,group,'right')]
        assert len(pool)>=len(slots)
        if condition=='rotated':
            desired=angle[source[slots]]+np.pi/2
        else:
            # Equal weighted angular coverage within each matched incoming stratum.
            # A fixed random order decouples old edge angle from its new target angle.
            perm=rng.permutation(len(slots));ws=weight[slots][perm]
            uniform=2*np.pi*(np.cumsum(ws)-.5*ws)/ws.sum()+rng.uniform(0,2*np.pi)
            desired=np.empty(len(slots));desired[perm]=uniform
        difference=desired[:,None]-angle[pool][None,:]
        radial=abs(distance[source[slots]][:,None]-distance[pool][None,:])/max(p.v_axon*p.delay_dt,1e-12)
        cost=(1-np.cos(difference))+.001*radial
        rows,cols=linear_sum_assignment(cost)
        assert np.array_equal(rows,np.arange(len(slots)))
        result[slots]=pool[cols]
        angle_error[slots]=np.arccos(np.clip(np.cos(desired-angle[pool[cols]]),-1.,1.))
    assert len(np.unique(result))==len(result) and target not in result
    assert np.array_equal(regions[result],regions[source])
    assert np.array_equal(candidate_delay[result],delay)
    return result,angle_error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['rotated','isotropic'],required=True)
    args=parser.parse_args();dest=OUT/args.condition;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'audit.json').exists(), 'Finished graph is immutable'
    start=time.time();s,tr,frozen,identity=setup(9108405)
    with np.load('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923/geometry.npz') as geo:
        centers=geo['centers_mm']
    pos=s.positions_e
    d=np.linalg.norm(pos[:,None]-centers[None],axis=2)
    region=np.full(s.n_e,2,np.int32);region[d[:,0]<1.75]=0;region[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1
    original=s.net['ampa_by_delay'];ptr,source,weight,delay=incoming(original,s.n_e)
    target=np.repeat(np.arange(s.n_e,dtype=np.int32),np.diff(ptr))
    # Same-angle transform must be the literal original graph, not a random sham.
    for i in [0,100,1000,10000,20000,31999]:
        a,b=ptr[i:i+2]
        check,_=row_reassign(i,source[a:b],weight[a:b],delay[a:b],pos,region,s.params,'reference')
        assert np.array_equal(check,source[a:b])
    columns=np.empty_like(source);err=np.empty(len(source),np.float32)
    blocks=dest/'target_blocks';blocks.mkdir(exist_ok=True)
    for lo in range(0,s.n_e,1000):
        hi=min(lo+1000,s.n_e);a,b=ptr[lo],ptr[hi]
        path=blocks/f'{lo:05d}_{hi:05d}.npz'
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['original_sources'],source[a:b])
                columns[a:b]=z['new_sources'];err[a:b]=z['angular_error_rad']
        else:
            for i in range(lo,hi):
                aa,bb=ptr[i:i+2]
                columns[aa:bb],err[aa:bb]=row_reassign(i,source[aa:bb],weight[aa:bb],delay[aa:bb],pos,region,s.params,args.condition)
            np.savez_compressed(path,original_sources=source[a:b],new_sources=columns[a:b],angular_error_rad=err[a:b])
        write(dest/'progress.json',dict(status='BUILDING',pid=os.getpid(),targets_complete=hi,total_targets=s.n_e,elapsed_s=time.time()-start))
        print(args.condition,hi,'/',s.n_e,round(time.time()-start,1),'s',flush=True)
    new=[];folder=dest/'ampa_by_delay';folder.mkdir(exist_ok=True)
    moment_errors=[]
    for lag,mat in enumerate(original):
        pick=delay==lag
        coo=mat.tocoo();keep=coo.row>=s.n_e
        replacement=sparse.coo_matrix((np.r_[weight[pick],coo.data[keep]],
                                      (np.r_[target[pick],coo.row[keep]],np.r_[columns[pick],coo.col[keep]])),shape=mat.shape).tocsc()
        assert replacement.nnz==mat.nnz
        for power in [1,2]:
            old=mat.copy();old.data **= power
            new_power=replacement.copy();new_power.data **= power
            error=np.max(abs(np.asarray(old.sum(axis=1))-np.asarray(new_power.sum(axis=1))),initial=0.)
            moment_errors.append(float(error))
        assert np.array_equal(mat.tocsr()[s.n_e:].data,replacement.tocsr()[s.n_e:].data)
        sparse.save_npz(folder/f'{lag:04d}.npz',replacement);new.append(replacement)
    assert max(moment_errors)<1e-8
    assert np.array_equal(region[source],region[columns])
    before=geometry(original,s.net['pos'],s.n_e);after=geometry(new,s.net['pos'],s.n_e)
    a0=before['central_targets']['angle_deg_mod180'];a1=after['central_targets']['angle_deg_mod180']
    difference=abs((a1-a0+90)%180-90)
    passed=(abs(difference-90)<15 and abs(after['central_targets']['axis_ratio']/before['central_targets']['axis_ratio']-1)<.2) if args.condition=='rotated' else after['central_targets']['axis_ratio']<1.15
    audit=dict(status='STATIC_CONTROL_PASS' if passed else 'STATIC_GEOMETRY_FAIL',condition=args.condition,
        method='Angular assignment of source partners within each target x source region x exact physical delay; incoming edge weights carried without change.',
        matched=['incomingdegree pertarget/sourceregion/delay','complete incoming weight multiset pertarget/sourceregion/delay','source/target positions','all thresholds','all nonEE edges','physical delay bins'],
        changed=['EE source partner identity','source outdegree and outgoing allocation'],
        baseline_zero_rotation_exact=True,source_identity=identity,ampa_values_sha256=sparse_digest(new),
        ampa_topology_sha256=sparse_digest(new,topology=True),
        original_partner_fraction=float(np.mean(columns==source)),weighted_angular_RMS_error_deg=float(np.rad2deg(np.sqrt(np.average(err.astype(float)**2,weights=weight)))),
        maximum_row_delay_moment_absolute_error=max(moment_errors),before=before,after=after,
        central_axis_change_deg=difference,geometry_gate='Rotation90±15deg and axis ratio within20%; isotropic central second-moment axis ratio<1.15, specified before generation.',
        seconds=time.time()-start,dynamics_started=False,not_patient_connectivity_estimate=True)
    write(dest/'audit.json',audit);write(dest/'progress.json',dict(status=audit['status'],targets_complete=s.n_e))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    main()

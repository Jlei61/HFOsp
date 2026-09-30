#!/usr/bin/env python3
"""Audit degree/delay/region/first-and-second-moment matched EE axis controls."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import pickle
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from topic4_historical_manual_z_common import setup
from src.topic4_multidimensional_parameters import sparse_digest
from src.topic4_rev20_dual_core_mechanism import _elliptical_radius, _invalidate_ampa_caches

OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls')
ANGLE = -29.682336421368863
ASPECT = 1.9359022695571184


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def moments(weights, score, keys):
    """Softmax score with per-stratum temperature set by the original weight CV.

    Each stratum is one target x source region x exact delay bin. The original
    sums and sums of squares are conserved. This is an orientation preference,
    not a promise that a constrained finite graph realizes a requested angle.
    """
    order = np.argsort(keys, kind='stable')
    key = keys[order]; w = weights[order]; s = score[order]
    start = np.r_[0, np.flatnonzero(np.diff(key))+1]
    lengths = np.diff(np.r_[start, len(w)])
    group = np.repeat(np.arange(len(start)), lengths)
    total = np.add.reduceat(w, start)
    target = np.add.reduceat(w*w, start)/(total*total)
    spread = np.maximum.reduceat(s, start)-np.minimum.reduceat(s, start)
    s -= np.maximum.reduceat(s, start)[group]
    lo = np.zeros(len(start)); hi = np.ones(len(start))
    def distribution(alpha):
        value = np.exp(s*alpha[group])
        sums = np.add.reduceat(value, start)
        squared = np.add.reduceat(value*value, start)/(sums*sums)
        return value, sums, squared
    for _ in range(40):
        _, _, q = distribution(hi)
        below = q < target-1e-14
        if not np.any(below):
            break
        hi[below] *= 2.
    # An exactly tied score may have an unattainable second moment. Preserve
    # that stratum verbatim instead of silently changing the noise amplitude.
    available = (spread > 1e-12) & (target > 1/lengths+1e-14) & (q >= target-1e-14)
    for _ in range(40):
        mid = (lo+hi)/2
        _, _, q = distribution(mid)
        low = q < target
        lo[low] = mid[low];hi[~low] = mid[~low]
    value, sums, _ = distribution((lo+hi)/2)
    new = value*(total/sums)[group]
    # Constant-weight strata have no freedom under both moment constraints.
    new[~available[group]] = w[~available[group]]
    assert np.all(new > 0)
    m1 = np.add.reduceat(new, start)
    m2 = np.add.reduceat(new*new, start)
    err1 = np.max(abs(m1-total)/np.maximum(total, 1e-100), initial=0.)
    old2 = np.add.reduceat(w*w, start)
    err2 = np.max(abs(m2-old2)/np.maximum(old2, 1e-100), initial=0.)
    if err1 > 1e-10 or err2 > 1e-7:
        raise AssertionError(('Moment matching failed', err1, err2))
    restored = np.empty_like(new);restored[order] = new
    return restored, dict(maximum_relative_first_moment_error=float(err1),
                          maximum_relative_second_moment_error=float(err2),
                          strata=len(start), adjustable_strata=int(available.sum()),
                          unattainable_score_strata=int(np.sum(q < target-1e-14)))


def geometry(bins, positions, ne):
    total = 0.; second = np.zeros((2, 2)); offset = np.zeros(2)
    center_total = 0.; center_second = np.zeros((2, 2))
    weight2 = 0.;delay_weight=0.;distance_weight=0.
    for delay, matrix in enumerate(bins):
        coo = matrix.tocoo();mask=coo.row<ne
        rr, cc, w = coo.row[mask], coo.col[mask], coo.data[mask]
        d = positions[cc]-positions[rr]
        total += w.sum();weight2 += (w*w).sum();offset += w@d
        second += (d*w[:, None]).T@d
        delay_weight += delay*w.sum()
        distance_weight += np.linalg.norm(d, axis=1)@w
        inside = np.all((positions[rr] >= 5) & (positions[rr] <= 15), axis=1)
        center_total += w[inside].sum()
        center_second += (d[inside]*w[inside, None]).T@d[inside]
    def tensor(tensor):
        values, vectors = np.linalg.eigh(tensor)
        return dict(angle_deg_mod180=float(np.rad2deg(np.arctan2(vectors[1,1],vectors[0,1]))%180),
                    axis_ratio=float(np.sqrt(values[1]/values[0])), second_moment=tensor.tolist())
    return dict(total_weight=float(total), total_squared_weight=float(weight2),
                mean_offset_mm=(offset/total).tolist(), whole=tensor(second/total),
                central_targets=tensor(center_second/center_total),
                weighted_mean_delay_steps=float(delay_weight/total),
                weighted_mean_distance_mm=float(distance_weight/total))


def transform(net, positions, regions, angle, aspect, length):
    ne = net['NE']; new=[];checks=[]
    for delay, matrix in enumerate(net['ampa_by_delay']):
        coo=matrix.tocoo(copy=True);mask=coo.row<ne
        rr, cc = coo.row[mask], coo.col[mask]
        if not len(rr):
            new.append(matrix);continue
        w=coo.data[mask]
        d=positions[cc]-positions[rr]
        desired=-_elliptical_radius(d,length_scale=length,angle_deg=angle,aspect_ratio=aspect)
        current=-_elliptical_radius(d,length_scale=length,angle_deg=ANGLE,aspect_ratio=ASPECT)
        score=np.log(w)+desired-current
        coo.data[mask], check=moments(w,score,rr*3+regions[cc])
        new.append(coo.tocsc());checks.append(check)
        if delay%20==0:
            print('delay',delay,'/',len(net['ampa_by_delay']),flush=True)
    assert sparse_digest(new,topology=True)==sparse_digest(net['ampa_by_delay'],topology=True)
    return new,checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['reference','rotated','isotropic'],required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/args.condition;dest.mkdir(exist_ok=True)
    assert not (dest/'audit.json').exists(), 'Existing graph/audit is immutable'
    start=time.time();s,tr,frozen,identity=setup(9108405)
    with np.load('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923/geometry.npz') as g:
        centers=g['centers_mm']
    d=np.linalg.norm(s.positions_e[:,None]-centers[None],axis=2)
    regions=np.full(s.n_e,2);regions[d[:,0]<1.75]=0;regions[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1
    original=s.net['ampa_by_delay'];before=geometry(original,s.net['pos'],s.n_e)
    angle,aspect=(ANGLE,ASPECT) if args.condition=='reference' else ((ANGLE+90,ASPECT) if args.condition=='rotated' else (ANGLE,1.))
    if args.condition=='reference':
        new,checks=original,[]
    else:
        new,checks=transform(s.net,s.net['pos'],regions,angle,aspect,s.params.l_EE)
    after=geometry(new,s.net['pos'],s.n_e)
    folder=dest/'ampa_by_delay';folder.mkdir(exist_ok=True)
    for delay,matrix in enumerate(new):
        sparse.save_npz(folder/f'{delay:04d}.npz',matrix)
    # Keep topology/weight provenance independent from any future dynamics.
    audit=dict(condition=args.condition,status='STATIC_CANDIDATE_PENDING_GEOMETRY_REVIEW',
        mechanism='EE direction preference with exact per-target x source-region x delay first and second weight moments.',
        source_identity=identity,requested_angle=angle,requested_aspect=aspect,
        topology_unchanged=True,delay_assignment_unchanged=True,source_region_incoming_degree_unchanged=True,
        inhibitory_and_E_to_I_unchanged=all(np.array_equal(a.tocsr()[s.n_e:].data,b.tocsr()[s.n_e:].data) for a,b in zip(original,new)),
        before=before,after=after,checks=checks,source_groups='Original A/B observer radius1.75mm and otherE; physicalthresholds unchanged.',
        length_scale_mm=s.params.l_EE,ampa_values_sha256=sparse_digest(new),ampa_topology_sha256=sparse_digest(new,topology=True),
        seconds=time.time()-start,not_assumed_successful_axis_manipulation=True,
        caveat='Requested kernel parameter is not the achieved finite weighted tensor; inspect whole/central tensor before any dynamics or direction claim.')
    write(dest/'audit.json',audit);print(json.dumps(audit['after']),flush=True)


if __name__=='__main__':
    main()

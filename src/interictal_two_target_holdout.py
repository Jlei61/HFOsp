"""Matched rank reconstruction and direction readouts for frozen templates."""
from __future__ import annotations
import numpy as np


def rank_error_suite(truth, mask, predictions):
    """All losses are event-level; each rank loss chooses its own closest template."""
    diff=predictions-truth[None,:,None,:]
    count=mask.sum(axis=1)
    mae=np.sum(np.abs(diff)*mask[None,:,None,:],axis=3)/count[None,:,None]
    rmse=np.sqrt(np.sum(diff**2*mask[None,:,None,:],axis=3)/count[None,:,None])
    chosen=np.argmin(mae,axis=2)
    # Accumulate pairs without constructing a full event x channel x channel tensor.
    penalties=np.zeros_like(mae); pairs=np.zeros(len(truth),int)
    for i in range(truth.shape[1]):
        for j in range(i+1,truth.shape[1]):
            sign=np.sign(truth[:,i]-truth[:,j])
            valid=mask[:,i]&mask[:,j]&(sign!=0)
            if not valid.any(): continue
            predicted_sign=np.sign(predictions[:,:,:,i]-predictions[:,:,:,j])
            penalties+=((predicted_sign*sign[None,:,None]<0)+.5*(predicted_sign==0))*valid[None,:,None]
            pairs+=valid
    if np.any(pairs==0): raise ValueError('no ordered channel pairs')
    return dict(rank_mae=mae.min(axis=2),rank_rmse=rmse.min(axis=2),
                pair_order_error=(penalties/pairs[None,:,None]).min(axis=2),assignments=chosen)


def direction_error_suite(directions,axes,assignments):
    """Do not select the axis with the test direction: assignments are rank-only."""
    directions=np.asarray(directions,float);axes=np.asarray(axes,float)
    if not np.isfinite(directions).all() or not np.isfinite(axes).all():
        raise ValueError('finite directions and axes required')
    if not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-7):
        raise ValueError('unit directions required')
    if not np.allclose(np.linalg.norm(axes,axis=2),1,atol=1e-7):
        raise ValueError('unit axes required')
    cosines=np.clip(np.einsum('ec,mkc->mek',directions,axes),-1,1)
    all_angles=np.degrees(np.arccos(cosines))
    selected=np.take_along_axis(cosines,assignments[...,None],axis=2)[...,0]
    return dict(angular_mae_deg=np.degrees(np.arccos(selected)),cosine_loss=1-selected,
                all_angles=all_angles)


def block_direction_null(all_angles,assignments,blocks,n_draws,seed):
    """Permute direction rows within each block; share each draw between methods."""
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(blocks==b) for b in np.unique(blocks)]
    null=np.empty((2,n_draws));n=len(blocks)
    index=np.arange(n)
    for draw in range(n_draws):
        perm=index.copy()
        for group in groups: perm[group]=rng.permutation(group)
        for m in range(2):
            null[m,draw]=all_angles[m,perm,assignments[m]].mean()
    return null,dict(singleton_events=sum(len(g) for g in groups if len(g)==1),n_blocks=len(groups))


def holm_adjust(p_values):
    p=np.asarray(p_values,float)
    order=np.argsort(p)
    adjusted=np.maximum.accumulate(np.minimum(1.,p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty_like(p);out[order]=adjusted
    return out

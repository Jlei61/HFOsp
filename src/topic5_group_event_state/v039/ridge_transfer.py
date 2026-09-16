"""Weighted closed-form ridge probes, preserving fitted parent predictions."""
import numpy as np


def fit_supported_response_columns(response,phases,anchors,minimum_anchors=3):
    """Keep a response component only when distinct FIT anchors measured it."""
    response=np.asarray(response);phases=np.asarray(phases);anchors=np.asarray(anchors)
    return np.array([len(np.unique(anchors[(phases=='FIT')&np.isfinite(response[:,j])]))>=minimum_anchors
                     for j in range(response.shape[1])],bool)
from .frozen_transfer import equal_anchor_mean


def fit_candidates(x,y,fit,anchor,*,pool_fit_by_anchor=False):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if y.ndim==1:y=y[:,None]
    xf=x[fit];yf=y[fit]
    unique,index=np.unique(anchor[fit],return_inverse=True)
    if pool_fit_by_anchor:
        # State features are constant within an anchor. Pooling its response
        # is algebraically exact for an equal-anchor squared-error objective.
        first=np.array([np.flatnonzero(index==i)[0] for i in range(len(unique))])
        if not np.array_equal(xf,xf[first][index]):raise ValueError('Cannot pool varying prefix features as an anchor state')
        xf=xf[first];yf=equal_anchor_mean(yf,anchor[fit]);weight=np.full(len(xf),1/len(xf))
    else:weight=1/(len(unique)*np.bincount(index)[index])
    center=(xf*weight[:,None]).sum(0);scale=np.sqrt(((xf-center)**2*weight[:,None]).sum(0));scale=np.where(scale>1e-6,scale,1.)
    xnorm=(x-center)/scale;xf=(xf-center)/scale;ymean=(yf*weight[:,None]).sum(0)
    a=xf*np.sqrt(weight[:,None]);b=(yf-ymean)*np.sqrt(weight[:,None]);models=[]
    # Both branches solve exactly the same ridge objective; use the smaller
    # matrix when a full fixed-history bank has more columns than FIT anchors.
    primal=a.shape[1]<=a.shape[0];gram=a.T@a if primal else a@a.T
    eigen,v=np.linalg.eigh(gram);eigen=np.maximum(eigen,0.)
    rhs=v.T@(a.T@b if primal else b)
    for alpha in (.01,.1,1.,10.):
        solution=v@(rhs/(eigen[:,None]+alpha));coef=solution if primal else a.T@solution
        prediction=xnorm@coef+ymean
        models.append(dict(alpha=alpha,center=center,scale=scale,intercept=ymean,coefficients=coef,prediction=prediction))
    return models,ymean


def select_probe(models,target,inner,anchor,parent=None,allow_zero=True):
    target=np.asarray(target,float)
    if target.ndim==1:target=target[:,None]
    parent=np.zeros_like(target) if parent is None else np.asarray(parent,float)
    candidates=[]
    for model in models:
        prediction=parent+model['prediction'];error=((prediction-target)**2).mean(-1)
        value=float(equal_anchor_mean(error[inner],anchor[inner]).mean())
        candidates.append((value,model['alpha'],prediction,model))
    if allow_zero:
        error=((parent-target)**2).mean(-1);value=float(equal_anchor_mean(error[inner],anchor[inner]).mean())
        candidates.append((value,0.,parent.copy(),dict(alpha=None,zero_residual=True)))
    value,_,prediction,model=min(candidates,key=lambda row:(row[0],row[1]))
    return dict(inner_loss=value,prediction=prediction,model=model)

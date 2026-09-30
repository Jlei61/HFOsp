"""Shared/private covariance using actual physical transmission delays.

The normalized continuous current-filter covariance depends on d_i-d_j in
milliseconds, not the numerical integration step. The Poisson closure itself
is unchanged and remains approximate for renewal/correlated population input.
"""
from common import np
from scipy import sparse


def covariance_kernel(lag_ms,rise,decay):
    lag=np.abs(np.asarray(lag_ms))
    return (decay*np.exp(-lag/decay)-rise*np.exp(-lag/rise))/(decay-rise)


def physical_split(s):
    lag=np.abs(s.delays[:,None]-s.delays[None,:]);out={};qa=[]
    for k,kind in enumerate(['ampa','gaba']):
        row,col,a=s.raw[k];rq,cq,q=s.raw[k+2]
        assert np.array_equal(row,rq) and np.array_equal(col,cq)
        K=covariance_kernel(lag,s.rise[k],s.decay[k])
        shared=np.asarray(a.multiply(a@K).sum(1)).ravel()/s.sizes[col]
        total=np.asarray(q.sum(1)).ravel();fraction=shared/total
        assert np.isfinite(fraction).all() and fraction.min()>=-1e-12 and fraction.max()<=1+1e-12
        roundoff=float(np.max(np.maximum(-fraction,fraction-1),initial=0.))
        fraction=np.clip(fraction,0,1)
        private=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
        target=np.repeat(np.arange(s.P),np.diff(private.indptr))
        keys=target*s.P+private.indices%s.P;basekeys=row*s.P+col
        where=np.searchsorted(basekeys,keys);assert np.array_equal(basekeys[where],keys)
        private.data*=1-fraction[where]
        remaining=np.bincount(where,weights=private.data,minlength=len(row))
        error=float(np.max(abs(remaining+shared-total)/np.maximum(total,1e-30)));assert error<1e-12
        out[kind]=private
        qa.append(dict(synapse=kind,pairwise_reconstruction_relative_error=error,
            removed_fraction_range=[float(fraction.min()),float(fraction.max())],roundoff_cleanup=roundoff,
            delay_min_ms=float(s.delays.min()),delay_max_ms=float(s.delays.max()),
            delay_spacing_ms=float(np.diff(s.delays).min()),integration_step_used=False))
    return out,qa

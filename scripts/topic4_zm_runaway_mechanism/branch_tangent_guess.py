"""A periodic-family rate tangent as an Arnoldi starting vector only.

The current 14-state perturbation is zero; the past-rate perturbation is the
Fourier family tangent at the correct physical history times. This is merely
an admissible trial vector, not a Floquet mode or a stability estimate.
"""
import numpy as np
from scipy.signal import resample


def history_guess(source,target,steps,depth,states=14):
    r=source['r'];p=r.shape[1];n=len(r)
    assert float(source['residual'])<2e-8 and float(target['residual'])<2e-8
    assert r.shape[1]==target['r'].shape[1]
    assert abs(float(source['T'])-float(target['T']))<1e-6
    assert abs(float(source['D'])-float(target['D']))<1e-7
    assert np.max(abs(source['Z']-target['Z']))<1e-6
    difference=float(np.linalg.norm(resample(r,len(target['r']),axis=0)-target['r'])/
                     np.linalg.norm(target['r']))
    assert difference<.001,('The tangent source must describe the same nearby-refined orbit',difference)
    tangent=source['tangent'];assert tangent.shape==(n*p+2,)
    # Stored BVP coordinates are r/RS with RS=.001 (/ms).
    dr=tangent[:n*p].reshape(n,p)*.001
    sampled=resample(dr,steps,axis=0)
    history=sampled[(-np.arange(1,depth+1))%steps]
    vector=np.r_[np.zeros(states*p),history.ravel()]
    assert np.isfinite(vector).all() and np.linalg.norm(vector)>0
    return vector,dict(source_N=n,target_N=len(target['r']),
        source_target_waveform_relative_error=difference,
        source_dD_dlogT=float(tangent[-1])*.001,
        state_perturbation='zero',
        history_perturbation='Fourier periodic-family rate tangent at -j*actual_dt',
        scope='Krylov starting vector only. No multiplier, stability, or branch connection is inherited.')


def sanity():
    n=129;steps=512;depth=31;p=2;t=np.arange(n)/n
    r=np.stack([.01+.002*np.sin(2*np.pi*t),.02+.003*np.cos(4*np.pi*t)],axis=1)
    tangent=np.r_[(np.stack([np.cos(2*np.pi*t),np.sin(4*np.pi*t)],axis=1)/.001).ravel(),1.,0.]
    q=dict(r=r,T=3.,D=.2,Z=np.ones(p)*.8,residual=1e-12,tangent=tangent)
    v,_=history_guess(q,q,steps,depth)
    lag=-np.arange(1,depth+1)/steps
    expected=np.stack([np.cos(2*np.pi*lag),np.sin(4*np.pi*lag)],axis=1)
    assert np.max(abs(v[14*p:].reshape(depth,p)-expected))<1e-12
    assert np.count_nonzero(v[:14*p])==0
    return float(np.max(abs(v[14*p:].reshape(depth,p)-expected)))


if __name__=='__main__':print('PASS',sanity())

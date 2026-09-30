"""Expose an already reconstructed Fourier orbit on its exact uniform grid.

There is no interpolation or new dynamics here. The callback lets the cached
variational integrator upload only a bounded block of the same orbit states.
"""
import numpy as np


class SpectralGridSampler:
    def __init__(self,states,period,derivative=None,derivative_indices=None):
        self.states=states
        self.n=len(states)-1
        self.period=float(period)
        self.derivative_indices=derivative_indices
        assert np.array_equal(states[-1],states[0])
        if derivative is not None:
            assert derivative.shape==((len(derivative_indices),*states.shape[1:]) if derivative_indices is not None else states.shape)
            self.derivative=derivative
            return
        self.derivative=np.empty_like(states)
        lam=2j*np.pi*np.fft.rfftfreq(self.n,d=self.period/self.n)
        # Separate state components to keep Fourier workspace bounded.
        for k in range(states.shape[1]):
            self.derivative[:-1,k]=np.fft.irfft(
                np.fft.rfft(states[:-1,k],axis=0)*lam[:,None],n=self.n,axis=0)
        self.derivative[-1]=self.derivative[0]

    def __call__(self,times,nu=0):
        x=np.asarray(times)*self.n/self.period
        idx=np.rint(x).astype(np.int64)
        assert np.max(abs(x-idx),initial=0)<1e-7, 'Sampler only accepts exact orbit-grid times'
        assert np.min(idx,initial=0)>=0 and np.max(idx,initial=0)<=self.n
        assert nu in [0,1]
        if nu and self.derivative_indices is not None:
            indices=np.asarray(self.derivative_indices)
            slot=np.searchsorted(indices,idx)
            assert np.all(slot<len(indices)) and np.array_equal(indices[slot],idx)
            return self.derivative[slot]
        return (self.derivative if nu else self.states)[idx]


if __name__=='__main__':
    n=257;T=3.7;t=np.arange(n+1)*T/n
    y=np.empty((n+1,2,3));dy=np.empty_like(y)
    for k in range(2):
        for j in range(3):
            omega=2*np.pi*(j+1)/T
            y[:,k,j]=np.sin(omega*t+k)
            dy[:,k,j]=omega*np.cos(omega*t+k)
    y[-1]=y[0];s=SpectralGridSampler(y,T)
    ids=np.r_[0,1,37,256,257];value_error=float(abs(s(t[ids])-y[ids]).max())
    derivative_error=float(abs(s(t,1)-dy).max())
    assert value_error==0 and derivative_error<1e-11
    print('PASS',value_error,derivative_error)

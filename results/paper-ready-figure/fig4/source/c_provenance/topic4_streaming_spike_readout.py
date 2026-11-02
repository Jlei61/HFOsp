"""Read-only online equivalent of the R1 contact envelope and 2-ms native movie.

The existing integration function is cloned with a private numpy allocation proxy.
Only its dense E-spike recorder allocation is replaced. No model source, RNG,
integration operation or recorded spike is modified. Linear spatial sampling and
temporal convolution commute; output equivalence is checked against dense R1.
"""
import types
import numpy as np
from src.topic4_node_dualmode import sheet_bin_indices


class StreamingSpikes:
    def __init__(self, nsteps, ne, positions, montage, dt, bin_ms=2., smooth_ms=5., kernel_width=.25):
        self.shape=(nsteps,ne);self.dt=dt;self.bin_ms=bin_ms;self.smooth_ms=smooth_ms
        self.bs=round(bin_ms/dt);self.seen=0;self.count=np.zeros(ne,np.uint16)
        self.bins,self.size=sheet_bin_indices(positions,bin_mm=1.,sheet_mm=20.)
        self.movie=np.zeros(((nsteps+self.bs-1)//self.bs,self.size,self.size),np.uint16)
        weights=[]
        for c in montage.contacts:
            d=np.linalg.norm(positions-c[None,:],axis=1)
            w=np.exp(-d*d/(2*kernel_width**2));weights.append(w/max(w.sum(),1e-12))
        self.weights=np.asarray(weights)
        self.raw=np.zeros((len(weights),nsteps//self.bs),float)

    def __len__(self):return self.shape[0]

    def __setitem__(self,t,spikes):
        if t!=self.seen:raise ValueError('recorder must advance one step at a time')
        self.count+=spikes;self.seen+=1
        if self.seen%self.bs==0:
            frame=self.seen//self.bs-1
            self.raw[:,frame]=self.weights@self.count
            self.movie[frame]=np.bincount(self.bins[self.count>0],minlength=self.size**2).reshape(self.size,self.size)
            self.count.fill(0)

    def __getitem__(self,key):
        if not isinstance(key,slice) or key.start is not None or key.step is not None or key.stop!=self.seen:
            raise ValueError('only the engine early-stop prefix slice is supported')
        self.shape=(key.stop,self.shape[1]);return self

    def native(self):
        nf=(self.seen+self.bs-1)//self.bs
        if self.seen%self.bs:
            self.movie[nf-1]=np.bincount(self.bins[self.count>0],minlength=self.size**2).reshape(self.size,self.size)
        return dict(activity_counts=self.movie[:nf],frame_ms=self.bin_ms,bin_mm=1.,sheet_mm=20.)

    def envelope(self):
        raw=self.raw[:,:self.seen//self.bs]
        sig=max(1e-6,self.smooth_ms/self.bin_ms);half=int(np.ceil(3*sig));x=np.arange(-half,half+1)
        kernel=np.exp(-x*x/(2*sig*sig));kernel/=kernel.sum()
        env=np.stack([np.convolve(row,kernel,mode='same') for row in raw])
        return env,self.bin_ms,env.mean(0)


class RecorderNumpy:
    def __init__(self,shape,positions,montage,dt):
        self.shape=shape;self.positions=positions;self.montage=montage;self.dt=dt;self.hits=0
    def __getattr__(self,name):return getattr(np,name)
    def zeros(self,shape,dtype=float,*args,**kwargs):
        if isinstance(shape,tuple) and shape==self.shape and np.dtype(dtype)==np.dtype(bool):
            self.hits+=1
            if self.hits!=1:raise RuntimeError('more than one matching recorder allocation')
            return StreamingSpikes(*shape,self.positions,self.montage,self.dt)
        return np.zeros(shape,dtype,*args,**kwargs)


def simulate_streaming(original,p,net,*args,positions,montage,**kwargs):
    proxy=RecorderNumpy((round(p.T/p.dt),net['NE']),positions,montage,p.dt)
    namespace=dict(original.__globals__);namespace['np']=proxy
    clone=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    clone.__kwdefaults__=original.__kwdefaults__
    result=clone(p,net,*args,**kwargs)
    if proxy.hits!=1:raise RuntimeError('dense E-spike recorder was not intercepted')
    return result

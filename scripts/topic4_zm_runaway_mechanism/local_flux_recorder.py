"""Aggregate local-LIF spike flux without altering any neuron update."""
from native_cycle_waveform_response import CODE
import numpy as np

CODE_FLUX=CODE.replace('const double* wave,unsigned int* counts,',
    'const double* wave,unsigned int* counts,unsigned int* stepcounts,unsigned int* eligiblecounts,')
_old='double cur=mu+ia-ig;bool fired=false;ref=max(0,ref-1);'
assert CODE_FLUX.count(_old)==1
CODE_FLUX=CODE_FLUX.replace(_old,'double cur=mu+ia-ig;bool eligible=(ref<=1);bool fired=false;ref=max(0,ref-1);')
_old='if(t>=0&&fired){int bin=min((int)(phase*B),B-1);counts[id*B+bin]++;}'
assert CODE_FLUX.count(_old)==1
CODE_FLUX=CODE_FLUX.replace(_old,_old+'''
 unsigned int hit=__ballot_sync(0xffffffff,fired);
 unsigned int free=__ballot_sync(0xffffffff,eligible);
 if((threadIdx.x&31)==0){
   long long index=(long long)g*(steps+burn)+t+burn;
   if(hit)atomicAdd(stepcounts+index,(unsigned int)__popc(hit));
   if(free)atomicAdd(eligiblecounts+index,(unsigned int)__popc(free));
 }
''')
_kernels={}


def capture(pars,wave,R,T,dt,burn,steps,seed,device=0,B=128):
    import cupy as cp
    cp.cuda.Device(device).use();assert R%128==0
    if device not in _kernels:_kernels[device]=cp.RawKernel(CODE_FLUX,'waveform',options=('--fmad=false',))
    pars=np.ascontiguousarray(pars,dtype='f8');wave=np.ascontiguousarray(wave,dtype='f8')
    P=len(pars);W=wave.shape[-1]
    assert wave.shape==(P,3,W)
    count=cp.zeros((P,R,B),dtype=cp.uint32);fire=cp.zeros((P,burn+steps),dtype=cp.uint32);avail=cp.zeros_like(fire)
    _kernels[device]((P*R//128,),(128,),(cp.asarray(pars),cp.asarray(wave),count,fire,avail,
        np.int32(P),np.int32(R),np.int32(W),np.int32(B),np.int32(steps),np.int32(burn),float(dt),float(T),np.uint64(seed)))
    return count.get(),fire.get(),avail.get()


def check_availability(fired,available,nref,R):
    cs=np.r_[np.int64(0),np.cumsum(fired,dtype=np.int64)];k=np.arange(len(fired))
    expected=R-cs[k]+cs[np.maximum(k-nref+1,0)]
    assert np.array_equal(expected,available)
    assert np.all(fired<=available) and np.all(available<=R)

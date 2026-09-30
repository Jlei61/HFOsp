"""Actual cached-RHS parity beyond the signed 32-bit double-index limit."""
from common import *
from cached_monodromy import CACHED
import argparse


def main(a):
    import cupy as cp
    cp.cuda.Device(a.device).use();P=935
    code=CACHED[CACHED.index('extern "C" __global__ void cached_rhs'):]
    rkcode=code.replace('(*offset+local)*8*P','(2*(*offset)+local)*8*P')
    assert '1LL*(2*(*offset)+local)*8*P' in rkcode
    fixed=cp.RawKernel(f'#define P {P}\n'+rkcode,'cached_rhs',options=('--fmad=false',))
    legacy=cp.RawKernel(f'#define P {P}\n'+rkcode.replace('1LL*',''),'cached_rhs',options=('--fmad=false',))
    rng=np.random.default_rng(20260919)
    payload=cp.asarray(rng.normal(size=(8,P)))
    Z=cp.full(P,.8);dy=cp.asarray(rng.normal(size=(14,P)));arr=cp.asarray(rng.normal(size=(4,P)))
    pars=cp.ones((20,P));consts=cp.ones(11)
    def apply(kernel,gains,tick,local):
        out=cp.empty_like(dy);rate=cp.empty(P);offset=cp.asarray([tick],dtype=cp.int32)
        kernel(((P+127)//128,),(128,),
               (gains,Z,offset,np.int32(local),dy,arr,pars,consts,out,rate))
        cp.cuda.get_current_stream().synchronize()
        return out.get(),rate.get()
    reference=apply(legacy,payload[None],0,0)
    small=apply(fixed,payload[None],0,0)
    assert all(np.array_equal(x,y) for x,y in zip(reference,small))
    tick=149999;local=2;sample=2*tick+local
    elements=sample*8*P
    assert elements>np.iinfo(np.int32).max
    large=cp.empty((sample+1,8,P));large[sample]=payload
    actual=apply(fixed,large,tick,local)
    assert all(np.array_equal(x,y) for x,y in zip(reference,actual))
    q=dict(status='PASS',P=P,steps=150000,stage_sample=sample,double_index=elements,
           old_signed_int32_limit=int(np.iinfo(np.int32).max),
           small_grid_bitwise_parity=True,actual_large_array_bitwise_parity=True,
           correction='64-bit pointer offsets only; unchanged variational equations',
           failed_job='rate_upper_dt0018_gpu1.log; first phase map had no accepted result')
    write(OUT/'large_gain_index_check.json',q);log('LARGE GAIN INDEX',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())

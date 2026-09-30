"""Same phase-input neuron kernel with explicit replica V/ref/M history."""
import numpy as np
import coherent_phase_sampler as original

OLD='const double* cfg,const int* extra,const double* wave'
NEW='const double* cfg,const int* extra,const double* initial,const double* wave'
STATE='double v=a[9],m=a[11];int ref=(int)a[10];'
REPLACE='double v=initial[(long long)id*3],m=initial[(long long)id*3+2];int ref=(int)initial[(long long)id*3+1];'
assert original.CODE.count(OLD)==original.CODE.count(STATE)==1
CODE=original.CODE.replace(OLD,NEW).replace(STATE,REPLACE)


def kernel(cp):return cp.RawKernel(CODE,'sample',options=('--fmad=false',))


def run(cp,fn,ie,ii,p,cfg,R,burn,extra,seed,first_cell,wave,phases,initial):
    N,B=ie.shape[0],len(p);Q=B*R;period=wave.shape[1]
    assert ie.shape==ii.shape==(N,Q) and wave.shape==(2,period,B) and initial.shape==(B,R,3)
    flags=cp.empty((N,Q),dtype='u1');stats=cp.empty((Q,16),dtype='f8')
    fn(((Q+127)//128,),(128,), (ie,ii,cp.asarray(p),cp.asarray(cfg),cp.asarray(extra,dtype='i4'),
        cp.asarray(initial),cp.asarray(wave),cp.asarray(phases,dtype='i4'),np.int32(period),flags,stats,
        np.int32(B),np.int32(R),np.int32(N),np.int32(burn),np.uint64(seed),np.int32(first_cell),np.int32(0)))
    return flags,stats.reshape(B,R,16)

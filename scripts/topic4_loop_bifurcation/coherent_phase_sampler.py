"""Original local neuron equations with an explicit periodic input mean."""
import numpy as np
import individual_spectral_sampler as original

OLD='const double* cfg,const int* extra,unsigned char* flags'
NEW='const double* cfg,const int* extra,const double* wave,const int* phases,int period,unsigned char* flags'
INPUT='double e=ie[(long long)row*Q+id],i=ii[(long long)row*Q+id];'
assert original.CODE.count(OLD)==original.CODE.count(INPUT)==1
CODE=original.CODE.replace(OLD,NEW).replace(INPUT,INPUT+'''
  int phi=(t+warm+phases[id])%period;
  e+=wave[phi*B+cell];i+=wave[(period+phi)*B+cell];
''')


def kernel(cp):
    return cp.RawKernel(CODE,'sample',options=('--fmad=false',))


def run(cp,fn,ie,ii,p,cfg,R,burn,extra,seed,first_cell,wave,phases):
    N,B=ie.shape[0],len(p);Q=B*R;period=wave.shape[1]
    assert ie.shape==ii.shape==(N,Q) and wave.shape==(2,period,B)
    flags=cp.empty((N,Q),dtype='u1');stats=cp.empty((Q,16),dtype='f8')
    fn(((Q+127)//128,),(128,), (ie,ii,cp.asarray(p),cp.asarray(cfg),cp.asarray(extra,dtype='i4'),
        cp.asarray(wave),cp.asarray(phases,dtype='i4'),np.int32(period),flags,stats,
        np.int32(B),np.int32(R),np.int32(N),np.int32(burn),np.uint64(seed),np.int32(first_cell),np.int32(0)))
    return flags,stats.reshape(B,R,16)

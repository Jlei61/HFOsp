"""Independent finite differences of the local nonlinear RHS along an actual orbit.
No stability result is inferred here. Frozen forward kernels are unmodified.
"""
from native_path import *
from orbit_reconstruction import orbit_states_and_derivative
from streaming_periodic import StreamPeriodic
from floquet_v3 import TANGENT
from dynamics_v3 import cuda_code,model_device_arrays,CUDA_DEVICE,CUDA_RESP
import argparse


def main(a):
    import cupy as cp
    cp.cuda.Device(a.device).use()
    s=model();attach_native_path(s);z=np.load(a.orbit)
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    o=StreamPeriodic(s,len(sol['r']),a.device);o.cache_mean_operators=False
    from scipy.fft import next_fast_len
    Y,dY,_=orbit_states_and_derivative(o,sol,next_fast_len(2*len(sol['r']),real=True),include_rate=False)
    del o;cp.get_default_memory_pool().free_all_blocks()
    indices=np.unique(np.linspace(0,len(Y)-2,a.samples,dtype=int))
    scale=np.maximum(np.std(Y[:-1],axis=0),1e-8)
    pars,consts,SE,SI,WE,WI=model_device_arrays(s,cp,dynamic_z=False)
    consts=cp.concatenate([consts,cp.asarray([1.])])
    mod=cp.RawModule(code=cuda_code(s),options=('--fmad=false',),name_expressions=['rhs'])
    rhs=mod.get_function('rhs')
    modt=cp.RawModule(code=f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+TANGENT,
        options=('--fmad=false',),name_expressions=['tangent_rhs'])
    tangent=modt.get_function('tangent_rhs')
    arr=cp.zeros((4,s.P));dummy=cp.zeros((1,s.P));f=cp.empty((14,s.P));r=cp.empty(s.P)
    fp=cp.empty_like(f);fm=cp.empty_like(f);rp=cp.empty_like(r);rm=cp.empty_like(r)
    rng=np.random.default_rng(9191206);rows=[]
    for mode in ['phase','random_scaled']:
        epsilons=[1e-3,1e-4,1e-5,1e-6]
        accum=np.zeros((len(epsilons),6));worst=[]
        for j in indices:
            y=cp.asarray(Y[j]);direction=dY[j].copy() if mode=='phase' else scale*rng.normal(size=scale.shape)
            direction[11]=0;dy=cp.asarray(direction)
            tangent(((s.P+127)//128,),(128,),(y,dy,arr,pars,consts,SE,SI,WE,WI,f,r))
            for k,h in enumerate(epsilons):
                for sign,out,rr in [(1,fp,rp),(-1,fm,rm)]:
                    rhs(((s.P+127)//128,),(128,),(y+sign*h*dy,arr,pars,consts,SE,SI,WE,WI,
                        dummy,np.int32(0),np.int32(0),out,rr))
                df=(fp-fm)/(2*h)-f;dr=(rp-rm)/(2*h)-r
                accum[k]+=np.array([float(cp.sum(df*df)),float(cp.sum(f*f)),
                    float(cp.sum(dr*dr)),float(cp.sum(r*r)),float(cp.max(abs(df))),float(cp.max(abs(dr)))])
        for h,q in zip(epsilons,accum):
            rows.append(dict(direction=mode,epsilon=h,rhs_relative_L2=float(np.sqrt(q[0]/q[1])),
                rate_relative_L2=float(np.sqrt(q[2]/q[3]))))
        log('LOCAL DIRECTIONAL CHECK',rows[-4:])
    write(OUT/'native_orbit_local_tangent_check.json',dict(status='DIAGNOSTIC_COMPLETE',source=a.orbit,
        samples=len(indices),rows=rows,scope='Local analytic derivative vs central finite differences; no Floquet acceptance'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--device',type=int,default=1)
    p.add_argument('--samples',type=int,default=257);main(p.parse_args())

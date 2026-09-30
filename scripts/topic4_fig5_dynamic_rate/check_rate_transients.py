"""Prescribed-current native LIF responses versus the two-state rate candidate."""
from common import *
from local_response_table import CODE
from rate_unit import trajectory,positive_rate
import cupy as cp
import time


def main():
    folder=OUT/'local_response/transient_validation_v1';folder.mkdir(parents=True,exist_ok=False)
    prep=read(GRID/'prepared.json');p=prep['params'];N=8192
    specs=[(0,15.,'pulse'),(0,18.,'pulse'),(1,18.,'pulse'),
           (0,15.,'multisine'),(0,18.,'multisine'),(1,18.,'multisine')]
    C=len(specs);theta=np.array([x[1] for x in specs]);pop=np.array([x[0] for x in specs])
    tau=np.where(pop==0,20.,10.);jump=tau/p['tau_r_AMPA']*np.where(pop==0,p['J_ext_E'],p['J_ext_I'])
    nr=np.where(pop==0,20,10).astype('int32');T=1000;times=np.arange(T*10)*.1
    u=np.zeros((len(times),C))
    for i,(_,_,kind) in enumerate(specs):
        if kind=='pulse':
            u[(times>=100)&(times<300),i]=20
            u[(times>=300)&(times<450),i]=-40
            u[(times>=500)&(times<650),i]=100
            u[(times>=650)&(times<800),i]=-20
        else:
            a=times/1000;u[:,i]=8+12*np.sin(2*np.pi*13*a)+8*np.sin(2*np.pi*37*a+.4)+5*np.sin(2*np.pi*5*a+1)
    write(folder/'config.json',dict(specs=specs,replicates=N,seed=231001,
        calibration='Constant-current transfer only; no transient fitting in this run',
        temporal_comparison='Native and rate outputs use identical nonoverlapping 10-ms bins',
        reference='Original independent colored-Poisson LIF units with prescribed common recurrent current',
        scope='Local response qualification, not spatial or network acceptance'))
    cp.cuda.Device(0).use();module=cp.RawModule(code=CODE,options=('--fmad=false','-I/usr/local/cuda/include'),name_expressions=['size','init','run'])
    size=cp.zeros(1,dtype=cp.int32);module.get_function('size')((1,),(1,),(size,))
    rng=cp.empty(C*N*int(size.get()[0]),dtype=cp.uint8);blocks=((C*N+127)//128,)
    module.get_function('init')(blocks,(128,),(rng,np.int32(C*N),np.uint64(231001)))
    arrays=[cp.zeros(C*N),cp.zeros(C*N),cp.full(C*N,11.),cp.zeros(C*N,dtype=cp.int32),
            cp.zeros(C*N,dtype=cp.uint32),cp.zeros(C*N,dtype=cp.uint32),cp.zeros(C*N),cp.zeros(C*N),cp.full(C*N,-1,dtype=cp.int32)]
    th=cp.asarray(theta);ct=cp.asarray(u[0]);ta=cp.asarray(tau);ju=cp.asarray(jump);re=cp.asarray(nr)
    ar=float(np.exp(-DT/p['tau_r_AMPA']));ad=float(np.exp(-DT/p['tau_d_AMPA']));lam=float(prep['nu_ext_per_ms']*DT)
    trace=cp.zeros((T,C),dtype=cp.uint32);start=time.time();fun=module.get_function('run')
    fun(blocks,(128,),(rng,*arrays,th,ct,ta,ju,re,np.int32(N),np.int32(C),np.int32(10000),np.int32(0),np.int32(0),ar,ad,lam,trace,np.int32(-1)))
    for k in range(len(times)):
        ct.set(u[k])
        fun(blocks,(128,),(rng,*arrays,th,ct,ta,ju,re,np.int32(N),np.int32(C),np.int32(1),np.int32(k),np.int32(1),ar,ad,lam,trace,np.int32(0)))
    native=cp.asnumpy(trace)/N*1000
    z=np.stack([trajectory(u[:,i],th0,po) for i,(po,th0,_) in enumerate(specs)],axis=1)
    rate=positive_rate(z[:,:,0]).reshape(T,10,C).mean(1)*1000
    n10=native.reshape(-1,10,C).mean(1);r10=rate.reshape(-1,10,C).mean(1)
    rows=[]
    for i,spec in enumerate(specs):
        rows.append(dict(population=spec[0],theta=spec[1],input=spec[2],
            native_mean_hz=float(native[:,i].mean()),rate_mean_hz=float(rate[:,i].mean()),
            rmse_10ms_hz=float(np.sqrt(np.mean((r10[:,i]-n10[:,i])**2))),
            normalized_rmse=float(np.linalg.norm(r10[:,i]-n10[:,i])/max(np.linalg.norm(n10[:,i]),1e-9)),
            raw_negative_rate_fraction=float(np.mean(z[:,i,0]<0)),
            raw_min_rate_hz=float(z[:,i,0].min()*1000)))
    np.savez_compressed(folder/'traces.npz',input_mv=u,native_rate_1ms=native,rate_1ms=rate,raw_state=z,specs=np.array(specs,dtype=str))
    write(folder/'result.json',dict(status='LOCAL_TRANSIENT_COMPARISON_COMPLETE',wall_s=time.time()-start,rows=rows,
        acceptance='Not assigned automatically; inspect waveform, negative-rate correction and low-activity return'))
    print(rows,flush=True)


if __name__=='__main__':main()

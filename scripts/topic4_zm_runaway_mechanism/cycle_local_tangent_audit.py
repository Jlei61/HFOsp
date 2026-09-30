"""Separate time-discretization error from a wrong local variational equation."""
from native_path import *
from streaming_periodic import StreamPeriodic
from floquet_v3 import orbit_states,TANGENT,CUDA_DEVICE,CUDA_RESP,model_device_arrays
import argparse,gc


def main(a):
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s);z=np.load(a.orbit)
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    o=StreamPeriodic(s,len(sol['r']),a.device);o.cache_mean_operators=False;cp=o.cp
    if a.physical_state:
        from local_cubic import LocalCubic
        sy=LocalCubic(z['cycle_times_ms'],z['state_cycle']);sr=LocalCubic(z['cycle_times_ms'],z['rate_cycle'])
        times=np.arange(a.samples)*sol['T']/a.samples
        Y=sy(times);r=sr(times);physical_derivative=sy(times,1)
    else:
        Y,r=orbit_states(o,sol,a.samples);Y=Y[:-1];r=r[:-1]
    del o;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    lam=2j*np.pi*np.fft.rfftfreq(a.samples,d=sol['T']/a.samples)
    dY=physical_derivative if a.physical_state else np.fft.irfft(np.fft.rfft(Y,axis=0)*lam[:,None,None],n=a.samples,axis=0)
    dr=np.fft.irfft(np.fft.rfft(r,axis=0)*lam[:,None],n=a.samples,axis=0)
    importance=np.linalg.norm(dr,axis=1)
    inds=np.unique(np.r_[np.linspace(0,a.samples-1,9,dtype=int),np.argsort(importance)[-12:]])
    pars,consts,SE,SI,WE,WI=model_device_arrays(s,cp,dynamic_z=False)
    consts=cp.r_[consts,cp.asarray([1.])]
    mod=cp.RawModule(code=f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+TANGENT,
                     options=('--fmad=false',),name_expressions=['tangent_rhs'])
    kernel=mod.get_function('tangent_rhs');out=cp.empty((14,s.P));rate=cp.empty(s.P);arr=cp.zeros((4,s.P))
    rows=[]
    for k in inds:
        y=Y[k];v=dY[k].copy();v[11]=0
        kernel(((s.P+127)//128,),(128,),
               (cp.asarray(y),cp.asarray(v),arr,pars,consts,SE,SI,WE,WI,out,rate))
        analytic=rate.get();fd_rows=[]
        for h in [1e-5,3e-6,1e-6]:
            fd=(s.output(y+h*v)-s.output(y-h*v))/(2*h)
            fd_rows.append(dict(h_ms=h,relative_error=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd),1e-30)),
                                max_error_hz_per_ms=float(abs(fd-analytic).max()*1000)))
        rows.append(dict(index=int(k),time_ms=k*sol['T']/a.samples,finite_difference=fd_rows,
            spectral_rate_derivative_error=float(np.linalg.norm(dr[k]-analytic)/max(np.linalg.norm(analytic),1e-30)),
            min_instantaneous_vE=float((y[8]+s.private_ve).min()),
            min_instantaneous_vI=float((y[11]**2*y[9]).min())))
    row=dict(orbit=a.orbit,samples=a.samples,rows=rows,
             max_best_fd_error=max(min(q['relative_error'] for q in r['finite_difference']) for r in rows))
    dest=OUT/'phase_audit';dest.mkdir(exist_ok=True)
    write(dest/f'{Path(a.orbit).stem}_local_tangent.json',row);log('LOCAL TANGENT',row['max_best_fd_error'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--family',choices=['rate','native'],default='native')
    p.add_argument('--physical-state',action='store_true');main(p.parse_args())

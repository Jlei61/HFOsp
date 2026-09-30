"""Original core OU and private Poisson sequence, independent of recurrence."""
from shared import *
import argparse,time

def generate(seed,duration=12000.):
    path=OUT/f'input_{seed}.npz'
    if path.exists():return path
    started=time.time();cfg=read(PRIOR/'model_config.json');p=cfg['params'];dt=p['dt'];signal=cfg['signal_per_ms']
    region=np.load(V10/'native/a/trajectory.npz')['region'];ne=np.sum(region<3);ni=len(region)-ne;core=region[np.isin(region,[0,1])]
    steps=round(duration/dt);rng=np.random.default_rng(seed)
    rng.choice(ne,size=min(80,ne),replace=False);rng.choice(ni,size=min(20,ni),replace=False)
    driver=runtime.CoreOUMixture(np.array([0,1]),signal,.95,0.,dt,p['tau_n'],p['sigma_n'],seed)
    nu=np.empty((steps,2));ext=np.empty((steps,len(core)),np.uint8)
    for k in range(steps):
        rng.standard_normal() # native legacy global OU still consumes this draw, with loading zero
        nu[k]=np.maximum(signal+driver.step(k*dt),0.)
        counts=rng.poisson(nu[k,core]*dt);assert counts.max()<256;ext[k]=counts
    native=PRIOR/'native'/str(seed)/'trajectory.npz'
    if native.exists():assert np.array_equal(nu.astype(np.float32),np.load(native)['nu_core'][:steps])
    np.savez_compressed(path,nu_core=nu,core_arrivals=ext)
    write(OUT/f'input_{seed}.json',dict(seed=seed,duration_ms=duration,status='COMPLETE',seconds=time.time()-started,
        parity='Native RNG namespace, initial raster draws, per-step global draw and actual core-only Poisson draw order; native saved OU exact at float32',
        recurrence_used=False))
    print('input',seed,'complete',time.time()-started,flush=True)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=848101);a=p.parse_args();generate(a.seed)

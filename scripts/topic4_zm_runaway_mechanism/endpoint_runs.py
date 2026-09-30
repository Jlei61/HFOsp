"""Direct DDE integration with endpoint-consistent rate history, from an actual periodic orbit."""
from native_path import *
import runner
from periodic_v3 import PeriodicV3
from floquet_v3 import orbit_states
import argparse,gc,os


class EndpointIntegrator(Integrator):
    history_scheme='instantaneous rate at the labelled endpoint'
    def step(self):
        super().step()
        assert not self.noise
        # State is already updated by Heun; evaluate output at that endpoint.
        self.eval_rhs(self.y,self.tick,self.f,self.rate2)
        self.cp.copyto(self.history[self.tick%self.depth],self.rate2)
        return self.history[self.tick%self.depth]


def from_orbit(s,path,dt,device):
    z=np.load(path);sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    s.set_D(sol['D'])
    if 'Z' in z:assert abs(s.Z-z['Z']).max()<1e-12,'Orbit and selected Z path disagree'
    from streaming_periodic import StreamPeriodic
    o=StreamPeriodic(s,len(sol['r']),device);o.cache_mean_operators=False
    Y,_=orbit_states(o,sol,len(sol['r']))
    depth=round(s.delays[-1]/dt)+1
    # Canonical ring: slot -j contains r(-j*dt), with exact Fourier interpolation.
    times=-np.arange(depth)*dt;coef=np.fft.rfft(sol['r'],axis=0)/len(sol['r'])
    coef[1:]*=2
    if len(sol['r'])%2==0:coef[-1]*=.5
    phase=np.exp(2j*np.pi*times[:,None]*np.arange(len(coef))[None,:]/sol['T'])
    history=np.empty((depth,s.P));history[(-np.arange(depth))%depth]=(phase@coef).real
    dest=OUT/'initial_states';dest.mkdir(parents=True,exist_ok=True);p=dest/f'{Path(path).stem}_dt{dt}.npz'
    tmp=p.with_name(f'{p.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,state=Y[0],history=history,tick=0,dt_ms=dt,Z=Y[0,11],source=str(path))
    tmp.replace(p)
    cp=o.cp;del o,Y;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return p


def main(a):
    runner.Integrator=EndpointIntegrator
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    if a.initial:
        initial=Path(a.initial);saved=np.load(initial)
        assert abs(float(saved['dt_ms'])-a.dt)<1e-12
        assert Path(str(saved['source'])).resolve()==Path(a.orbit).resolve()
    else:initial=from_orbit(s,a.orbit,a.dt,a.device)
    rows=[]
    for D in a.D:
        s.set_D(D)
        tag=('_rate' if a.family=='rate' else '')+('_dynamicZ' if a.dynamic_z else '')+('_fixedM' if a.fixed_m else '')
        label=f'endpoint_D{D:.7f}_dt{a.dt}'+tag
        rows.append(runner.run_condition(s,s.Z.copy(),label,a.duration,a.device,initial,dt=a.dt,
                                        dynamic_m=not a.fixed_m,dynamic_z=a.dynamic_z))
        record=f'endpoint_runs_{a.batch_label}.json' if a.batch_label else f'endpoint_runs_dt{a.dt}{tag}.json'
        write(OUT/record,dict(status='RUNNING',rows=rows,initial=str(initial)))
    write(OUT/record,dict(status='COMPLETE',rows=rows,initial=str(initial)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--D',type=float,nargs='+',required=True);p.add_argument('--duration',type=int,default=12000)
    p.add_argument('--device',type=int,default=0);p.add_argument('--fixed-m',action='store_true')
    p.add_argument('--dynamic-z',action='store_true')
    p.add_argument('--batch-label',default='')
    p.add_argument('--initial',help='Reuse a verified full orbit initial state and history')
    p.add_argument('--family',choices=['native','rate'],default='native');main(p.parse_args())

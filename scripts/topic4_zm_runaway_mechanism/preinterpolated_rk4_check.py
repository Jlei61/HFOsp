"""Check complete map parity and fine-step delayed-arrival arithmetic."""
from native_path import *
from streaming_periodic import StreamPeriodic
from orbit_reconstruction import orbit_states_and_derivative
from spectral_grid_sampler import SpectralGridSampler
from rk4_monodromy import RK4Monodromy
from preinterpolated_rk4 import PreinterpolatedRK4
import argparse,gc


def main(a):
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    o=StreamPeriodic(s,len(z['r']),a.device);o.cache_mean_operators=False;cp=o.cp
    n=int(np.ceil(sol['T']/.05))
    state,_,_=orbit_states_and_derivative(o,sol,2*n,derivative_indices=np.array([0]),include_rate=False)
    o.sample_state=SpectralGridSampler(state,sol['T'])
    old=RK4Monodromy(s,o,sol,dtmax=.05,device=a.device,host_gain_cache=True)
    x=np.random.default_rng(19191).normal(size=old.dim);x[11*s.P:12*s.P]=0.
    start=time.time();expected=old.matvec(x);old_seconds=time.time()-start
    new=PreinterpolatedRK4(s,o,sol,dtmax=.05,device=a.device,host_gain_cache=True)
    assert np.array_equal(old.host_gains,new.host_gains)
    start=time.time();actual=new.matvec(x);new_seconds=time.time()-start
    relative=float(np.linalg.norm(expected-actual)/np.linalg.norm(expected))
    assert np.array_equal(expected,actual),(relative,np.max(abs(expected-actual)))
    del old,x,expected,actual;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    # Separate arrivals at representative wrapping positions and the production
    # fine dt, without allocating a second full fine orbit/gain cache.
    checks=[];rng=np.random.default_rng(19192)
    for dt in [.05,.0015300925925925924,.0007650462962962962]:
        depth=int(np.ceil(s.delays[-1]/dt))+3
        hist=cp.asarray(rng.normal(size=(depth,s.P)));out=cp.empty((4,s.P))
        for tick,local in [(0,0),(depth-2,0),(3*depth+7,127)]:
            new.offset.fill(tick)
            for stage in [0.,.5,1.]:
                new.k['delayed_cubic']((s.P,),(128,),(*new.ops,hist,out,new.offset,
                    np.int32(local),stage,np.int32(depth),dt))
                new.interpolate(((new.delay_table.size+127)//128,),(128,),
                    (hist,new.delay_table,new.offset,np.int32(local),stage,np.int32(depth),dt,np.int32(new.delay_table.size)))
                new.table_arrivals((s.P,),(128,),(*new.ops,new.delay_table,new.arr))
                error=float(cp.max(abs(out-new.arr)))
                assert error==0,(dt,tick,stage,error)
                checks.append(dict(dt_ms=dt,tick=tick,local=local,stage=stage,max_error=error))
        del hist,out;cp.get_default_memory_pool().free_all_blocks()
    q=dict(status='PASS',complete_map_bitwise_equal=True,complete_map_relative_error=relative,
        complete_map_seconds=[old_seconds,new_seconds],local_arrival_checks=checks,
        scope='Same float64 cubic interpolation and CSR reduction order, identical RK4 stages, physical delays, frozen equations and all acceptance gates. Only reused arithmetic changes.')
    write(OUT/'preinterpolated_rk4_check.json',q);log('PREINTERPOLATED RK4',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())

"""Parity of streamed and full-orbit local-gradient construction."""
from native_path import *
from cached_monodromy import CachedMonodromy
from local_cubic import LocalCubic
from types import SimpleNamespace
import chunk_monodromy
import cupy as cp
import argparse,gc


def main(a):
    s=model();attach_rate_entry_path(s);z=np.load(OUT/'physical_cycles/rate_D14496_dt025.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    sy=LocalCubic(z['cycle_times_ms'],z['state_cycle'])
    def reconstruct(o,sol,n):return sy(np.arange(n+1)*sol['T']/n),None
    chunk_monodromy.orbit_states=reconstruct;cp.cuda.Device(a.device).use()
    o=SimpleNamespace(cp=cp,cache=None,cache_key=None)
    old=CachedMonodromy(s,o,sol,dtmax=.05,device=a.device)
    x=np.random.default_rng(19551).normal(size=old.dim);x[11*s.P:12*s.P]=0.
    expected=old.matvec(x);gain=old.gains.get();del old;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    o.sample_state=sy;new=CachedMonodromy(s,o,sol,dtmax=.05,device=a.device)
    actual=new.matvec(x);gerror=float(abs(new.gains.get()-gain).max())
    error=float(np.linalg.norm(expected-actual)/np.linalg.norm(expected))
    assert error<1e-12 and gerror<1e-10,(error,gerror)
    row=dict(status='PASS',matvec_relative_error=error,maximum_gain_difference=gerror,
             retained_orbit_nodes=len(new.orbit),period_nodes=new.n+1,dt_ms=new.dt,
             model_change=False,meaning='Only the schedule of identical local derivative evaluations changes')
    write(OUT/'streamed_cached_monodromy_check.json',row);log('STREAMED CACHE',row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())

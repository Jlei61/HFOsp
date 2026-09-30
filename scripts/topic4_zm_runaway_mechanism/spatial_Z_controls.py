"""Factorial spatial Z transplants on the rate network's observed entry segment.

The same periodic full initial state/history and dynamic M are used throughout.
Only E-group Z in the two cores or surround is replaced by the 7.8-s field.
The original Fig.5 readout is applied later by canonical_readouts.py.
"""
from native_path import *
import runner
from endpoint_runs import EndpointIntegrator
import argparse


def main(a):
    runner.Integrator=EndpointIntegrator;s=model();path=attach_rate_entry_path(s)
    before,after=path['fields'];core=s.E&(s.geo['group_region']<2);surround=s.E&~core
    initial=OUT/'initial_states/rate_seed_N1024_dt0.05.npz';z=np.load(initial)
    assert np.max(abs(z['state'][11]-before))<1e-12
    cases=[('both',after.copy()),('cores',np.where(core,after,before)),('surround',np.where(surround,after,before))]
    rows=[]
    for label,field in cases:
        q=runner.run_condition(s,field,'rate_spatial_Z_'+label,12000,a.device,initial,dt=.05)
        q.update(Z_A_B_surround=(np.array(s.regional_rates(field))/1000).tolist())
        rows.append(q);write(OUT/'spatial_Z_controls.json',dict(status='RUNNING',rows=rows))
    write(OUT/'spatial_Z_controls.json',dict(status='COMPLETE',rows=rows,
        baseline='endpoint_D0.1429804_dt0.05_rate',initial=str(initial),
        manipulation='7.7 to 7.8 s rate Z field, transplanted in named regions only; identical full fast history and dynamic M',
        interpretation='Finite-time necessity/sufficiency by region; not a claim that global mean Z alone specifies the state.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())

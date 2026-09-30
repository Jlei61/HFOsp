"""Matched Z interventions from a saved full endpoint-consistent DDE history."""
from endpoint_runs import EndpointIntegrator
from native_path import *
import runner,argparse


def main(a):
    runner.Integrator=EndpointIntegrator;s=model()
    (attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    source=np.load(a.source);initial=a.source
    if 'dt_ms' in source:dt=float(source['dt_ms'])
    else:
        contract=read(Path(a.source).parent/'contract.json');dt=float(contract['dt_ms'])
        assert contract['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
        state,history=checkpoint_initial(a.source)
        assert len(history)==round(s.delays[-1]/dt)+1
        folder=OUT/'initial_states';folder.mkdir(exist_ok=True)
        initial=str(folder/f'{Path(a.source).parent.name}_restart_dt{dt}.npz')
        np.savez_compressed(initial,state=state,history=history,tick=0,dt_ms=dt,
                            source=a.source,dt_source=str(Path(a.source).parent/'contract.json'))
    rows=[]
    for D in a.D:
        s.set_D(D);name=f'{a.label}_D{D:.7f}_dt{dt}'
        q=runner.run_condition(s,s.Z.copy(),name,a.duration,a.device,initial,dt=dt,
                               dynamic_z=a.dynamic_z,dynamic_m=not a.fixed_m)
        rows.append(q);write(OUT/f'{a.label}.json',dict(status='RUNNING',source=a.source,rows=rows))
    write(OUT/f'{a.label}.json',dict(status='COMPLETE',source=a.source,rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--D',type=float,nargs='+',required=True)
    p.add_argument('--label',required=True);p.add_argument('--duration',type=int,default=12000)
    p.add_argument('--device',type=int,default=0);p.add_argument('--dynamic-z',action='store_true')
    p.add_argument('--family',choices=['native','rate'],default='native')
    p.add_argument('--fixed-m',action='store_true');main(p.parse_args())

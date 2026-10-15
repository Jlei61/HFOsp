"""Bounded CPU experiment dispatch from the network built in this notebook.

Forked workers inherit only this new graph; each candidate generates fresh
spikes. Outputs are written as soon as jobs complete. No precomputed model
results are loaded. Linux fork avoids serializing a 40-million-edge graph.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from modeling.computational import simulate,score_simulation,optimize_cores,angle_response,save_run
from workflows.common import write_json
_CONTEXT=None


def _worker(job):
    sub,contacts,observer,objective,out=_CONTEXT
    cfg=sub['config'];kind=job['kind']
    if kind=='search':
        value=optimize_cores(sub,contacts,observer,objective,k=job['k'],budget=cfg['optimization_budget'],restart=job['restart'])
        write_json(out/(job['name']+'.json'),value['rows'])
    elif kind=='angle':
        value=angle_response(cfg,cfg['centers_mm'],contacts,observer,objective,angles=job['angles'])
        write_json(out/(job['name']+'.json'),value)
    else:
        options=dict(outward=job.get('outward',1.25),seed=job.get('seed',cfg['training_seed']))
        if kind=='zm':options.update(zm=cfg['zm'],record_current=True,duration_ms=18000.)
        value=simulate(sub,cfg['centers_mm'],contacts,**options)
        if kind!='zm':score_simulation(value,observer,objective,cfg['burnin_ms'])
        save_run(out/(job['name']+'.npz'),value)
        write_json(out/(job['name']+'.json'),{k:v for k,v in value.items() if not isinstance(v,np.ndarray) and k!='slow'})
    print('Completed new simulation job:',job['name'],flush=True)
    return job['name'],value


def run_experiments(substrate,contacts,observer,objective,output):
    global _CONTEXT
    output.mkdir(parents=True,exist_ok=True);_CONTEXT=(substrate,contacts,observer,objective,output)
    cfg=substrate['config']
    jobs=[dict(kind='comparison',name='outward_100',outward=1.0),
          dict(kind='zm',name='zm_enabled'),
          dict(kind='confirmation',name='new_noise_confirmation',seed=928401)]
    jobs += [dict(kind='search',name=f'core{k}_restart{r}',k=k,restart=r) for k in [1,2,3,4] for r in range(cfg['restarts'])]
    jobs += [dict(kind='angle',name=f'angle_{angle}',angles=[angle]) for angle in cfg['angle_values'] if angle!=0]
    write_json(output/'jobs.json',jobs)
    result={}
    try:
        with ProcessPoolExecutor(max_workers=cfg['max_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
            future={pool.submit(_worker,job):job['name'] for job in jobs}
            for completed in as_completed(future):
                name,value=completed.result();result[name]=value
                write_json(output/'progress.json',dict(completed=list(result),total=len(jobs)))
    finally:_CONTEXT=None
    return result

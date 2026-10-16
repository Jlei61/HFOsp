"""Equal-budget training batches; simulations never read stored paper results."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from modeling import computational as snn
from modeling.search import initial_vectors,propose
from workflows.common import write_json

_CONTEXT=None


def _candidate(job):
    config,contacts,observer,objective,output=_CONTEXT
    k,restart,epoch,stage,vector,role=job
    substrate=snn.build_substrate(config,angle_offset=float(vector[-1]))
    run=snn.simulate(substrate,np.asarray(vector[:2*k]).reshape(k,2),contacts,
                     outward=float(vector[-2]),seed=config['training_seed'])
    score=snn.score_simulation(run,observer,objective,config['burnin_ms'])
    row=dict(k=k,core_count=k,restart=restart,epoch=epoch,stage=stage,vector=list(vector),
             proposal=role,centroid_ms=run['primary_centroids_ms'].tolist(),**score)
    stem=output/f'core{k}_restart{restart}_epoch{epoch}'
    write_json(stem.with_suffix('.json'),row)
    # Save newly computed trajectories so the selected run can be examined later.
    snn.save_run(stem.with_suffix('.npz'),run)
    print(f"K={k} r={restart} epoch={epoch} {role}: N={score['N']} J={score['J']}",flush=True)
    return row


def search_core_counts(substrate,contacts,observer,objective,output,*,core_counts=(1,2,3,4),budget=10,restarts=2):
    global _CONTEXT
    output.mkdir(parents=True,exist_ok=True);config=substrate['config']
    _CONTEXT=config,contacts,observer,objective,output
    arms={(k,r):[] for k in core_counts for r in range(restarts)}
    initial=min(budget,config['initial_conditions']);completed=0;stage=0
    try:
        while completed<budget:
            size=initial if stage==0 else min(config['batch_size'],budget-completed)
            jobs=[]
            for (k,r),rows in arms.items():
                proposals=[(v,'random_space_filling') for v in initial_vectors(k,r,size,config)] if stage==0 else propose(k,r,stage,rows,size,config)
                jobs.extend((k,r,completed+i+1,stage,v,role) for i,(v,role) in enumerate(proposals))
            # Each batch freezes its training table before any new result is observed.
            with ProcessPoolExecutor(max_workers=config['max_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
                for future in as_completed([pool.submit(_candidate,j) for j in jobs]):
                    row=future.result();arms[row['k'],row['restart']].append(row)
            completed+=size;stage+=1
            for (k,r),rows in arms.items():
                rows.sort(key=lambda x:x['epoch']);write_json(output/f'core{k}_restart{r}.json',rows)
    finally:_CONTEXT=None
    return [dict(rows=rows,best=min((r for r in rows if r['J'] is not None),key=lambda r:r['J'],default=None)) for rows in arms.values()]


def simulate_selected(config,searches,contacts,observer,objective,*,k=2):
    candidates=[r for s in searches for r in s['rows'] if r['k']==k and r['J'] is not None]
    if not candidates:raise ValueError('No scorable two-core configuration')
    best=min(candidates,key=lambda r:(r['J'],r['restart'],r['epoch']))
    v=best['vector'];sub=snn.build_substrate(config,angle_offset=v[-1])
    run=snn.simulate(sub,np.asarray(v[:2*k]).reshape(k,2),contacts,outward=v[-2],seed=config['training_seed'])
    snn.score_simulation(run,observer,objective,config['burnin_ms'])
    np.testing.assert_allclose(run['score']['J'],best['J'],rtol=1e-10)
    return sub,run,best


def angle_response(config,centers,contacts,observer,objective,output):
    global _CONTEXT
    output.mkdir(parents=True,exist_ok=True);_CONTEXT=config,contacts,observer,objective,output
    jobs=[(2,0,i+1,0,[*np.asarray(centers).ravel(),1.25,angle],'angle_scan') for i,angle in enumerate(config['angle_values'])]
    rows=[]
    try:
        with ProcessPoolExecutor(max_workers=config['max_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
            for future in as_completed([pool.submit(_candidate,j) for j in jobs]):
                row=future.result();rows.append(dict(angle=row['vector'][-1],**row))
    finally:_CONTEXT=None
    return sorted(rows,key=lambda r:r['angle'])

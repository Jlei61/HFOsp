"""Outcome-dependent batched BO with durable proposals and paired confirmation."""
from pathlib import Path
import argparse,copy,fcntl,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_three_observable_bo as run
from scripts import analyze_topic4_three_observable_bo as analysis
OUT=run.OUT;rt=run.rt

def training_data():
    rows=analysis.records();p=rt.read(OUT/'plan.json');ids=list(p['initial_ids'])
    for f in sorted((OUT/'proposals').glob('adaptive_*.json')):ids.extend(rt.read(f)['ids'])
    result=[]
    for cid in ids:
        rr=[r for r in rows if r['candidate']==cid and r['topology']==2511 and r['noise'] in p['seeds'] and r['stage'] in ['initial','adaptive']]
        if len(rr)!=2:continue
        rr.sort(key=lambda r:r['noise']);ok=all(r['J'] is not None for r in rr)
        result.append(dict(candidate=cid,x=run.vector(rt.read(OUT/'candidates'/f'{cid}.json')).tolist(),scorable=ok,
            J=float(np.mean([r['J'] for r in rr])) if ok else None,
            individual_J=[r['J'] for r in rr],components=np.mean([r['components'] for r in rr],axis=0).tolist() if ok else None,
            score_status=[r['status'] for r in rr]))
    return result

def fit_model(rows,seed,device):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.kernels import ScaleKernel,MaternKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    torch.set_num_threads(4);lo,hi=run.bounds();good=[r for r in rows if r['scorable']]
    X=torch.tensor((np.array([r['x'] for r in good])-lo)/(hi-lo),dtype=torch.float64,device=device)
    Y=torch.tensor([[-r['J']] for r in good],dtype=torch.float64,device=device)
    v=np.array([np.var(r['individual_J'],ddof=1)/2 for r in good]);v=.5*v+.5*np.median(v)
    v=np.maximum(v,max(float(Y.var().item()),1e-12)*1e-6);Yvar=torch.tensor(v[:,None],device=device,dtype=torch.float64)
    error=None
    for restart in range(3):
        try:
            torch.manual_seed(seed+restart)
            kernel=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=6)).to(device=device,dtype=torch.float64)
            # Different deterministic initial hyperparameters on each retry.
            kernel.base_kernel.lengthscale=torch.exp(torch.rand(6,device=device,dtype=torch.float64)*np.log(20)+np.log(.1))
            kernel.outputscale=torch.exp(torch.rand((),device=device,dtype=torch.float64)*np.log(4)+np.log(.5))
            model=SingleTaskGP(X,Y,train_Yvar=Yvar,covar_module=kernel,outcome_transform=Standardize(m=1)).to(device)
            mll=ExactMarginalLogLikelihood(model.likelihood,model)
            fit_gpytorch_mll(mll,optimizer_kwargs={'options':{'maxiter':150}})
            if not torch.isfinite(model.posterior(X).mean).all():raise RuntimeError('nonfinite posterior')
            return model,X,Y,Yvar,good
        except Exception as exc:error=repr(exc)
    raise RuntimeError(f'GP fitting failed after three fixed restarts: {error}')

def propose(batch,device='cuda:0'):
    import torch
    from scipy.special import ndtr
    from scipy.stats import qmc
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    dest=OUT/'proposals'/f'adaptive_{batch:02d}.json'
    if dest.exists():
        saved=rt.read(dest)
        rt.write(OUT/'proposals/trust_state.json',saved['trust_after'])
        return saved
    frozen=rt.read(OUT/'analysis/objective_frozen.json');assert frozen['adaptive_dispatch_allowed']
    rows=training_data();assert sum(r['scorable'] for r in rows)>=8
    model,X,Y,Yvar,good=fit_model(rows,2026091403+batch*100,device)
    lo,hi=run.bounds();p=rt.read(OUT/'plan.json');ref=rt.read(OUT/'candidates'/f"{p['reference_id']}.json")
    state_path=OUT/'proposals/trust_state.json'
    state=rt.read(state_path) if state_path.exists() else dict(center=((run.vector(ref)-lo)/(hi-lo)).tolist(),halfwidth=.25,success=0,failure=0)
    state_before=copy.deepcopy(state)
    with torch.no_grad():
        posterior=model.posterior(X);best=int(posterior.mean[:,0].argmax());point=X[best]
        center=torch.tensor(state['center'],dtype=torch.float64,device=device)
        joint=model.posterior(torch.stack([point,center]));mu=joint.mean[:,0];cov=joint.mvn.covariance_matrix
        variance=float((cov[0,0]+cov[1,1]-2*cov[0,1]).clamp_min(1e-16).item())
        probability=float(ndtr(float((mu[0]-mu[1]).item())/np.sqrt(variance)))
    improved=probability>.8
    if improved:state['center']=point.cpu().tolist()
    if batch>1:
        if improved:state['success']+=1;state['failure']=0
        else:state['failure']+=1;state['success']=0
        if state['success']>=2:state['halfwidth']=min(.5,state['halfwidth']*1.5);state['success']=0
        if state['failure']>=2:state['halfwidth']=max(.125,state['halfwidth']/2);state['failure']=0
    rng=qmc.Sobol(6,scramble=True,seed=2026091402+batch)
    raw=rng.random_base2(13);c=np.array(state['center']);h=state['halfwidth']
    local=np.maximum(0,c-h)+raw[:4096]*(np.minimum(1,c+h)-np.maximum(0,c-h));global_pool=raw[4096:]
    allX=(np.array([r['x'] for r in rows])-lo)/(hi-lo)
    def unique(pool,extra=()):
        compare=np.vstack([allX,np.array(extra).reshape(-1,6)]) if len(extra) else allX
        return pool[np.min(np.linalg.norm(pool[:,None]-compare[None,:],axis=2),axis=1)>1e-7]
    local=unique(local);global_pool=unique(global_pool)
    feasible=None;flags=np.array([r['scorable'] for r in rows],int)
    if len(np.unique(flags))==2:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        feasible=GaussianProcessClassifier(1.*RBF(np.ones(6)),random_state=2026091402+batch,max_iter_predict=100).fit(allX,flags)
    def probability_feasible(pool):return np.ones(len(pool)) if feasible is None else np.maximum(feasible.predict_proba(pool)[:,1],.05)
    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128]),seed=2026091403+batch)
    acq=qLogNoisyExpectedImprovement(model,X_baseline=X,sampler=sampler,prune_baseline=True)
    selected=[];details=[]
    for k in range(3):
        pool=unique(local,selected);values=[]
        if selected:acq.set_X_pending(torch.tensor(np.array(selected),dtype=torch.float64,device=device))
        with torch.no_grad():
            for start in range(0,len(pool),64):
                xx=torch.tensor(pool[start:start+64,None,:],dtype=torch.float64,device=device)
                values.append(acq(xx).cpu().numpy())
        values=np.concatenate(values)+np.log(probability_feasible(pool));index=int(np.argmax(values));selected.append(pool[index])
        details.append(dict(kind='qLogNEI',value=float(values[index])))
    pool=unique(global_pool,selected);std=[]
    with torch.no_grad():
        for start in range(0,len(pool),128):
            xx=torch.tensor(pool[start:start+128],dtype=torch.float64,device=device)
            std.append(model.posterior(xx).variance.sqrt()[:,0].cpu().numpy())
    std=np.concatenate(std)*probability_feasible(pool);index=int(std.argmax());selected.append(pool[index]);details.append(dict(kind='uncertainty',value=float(std[index])))
    vectors=lo+np.array(selected)*(hi-lo);ids=[]
    with torch.no_grad():prediction=model.posterior(torch.tensor(np.array(selected),dtype=torch.float64,device=device))
    for k,x in enumerate(vectors):
        cid=f'g2_b{batch:02d}_p{k+1:02d}';c=run.candidate_at(ref,x,cid,'adaptive_joint');c['stage']='adaptive';run.validate_candidate(c)
        rt.write(OUT/'candidates'/f'{cid}.json',c);ids.append(cid)
    checkpoint=OUT/'proposals'/f'gp_batch{batch:02d}.pt'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},X=X.cpu(),Y=Y.cpu(),Yvar=Yvar.cpu()),checkpoint)
    payload=dict(batch=batch,ids=ids,vectors=vectors.tolist(),details=details,training_data=rows,
        predicted_J=(-prediction.mean[:,0]).cpu().tolist(),predicted_sd=prediction.variance.sqrt()[:,0].cpu().tolist(),
        trust_before=state_before,trust_after=state,center_improvement_probability=probability,
        checkpoint=str(checkpoint),checkpoint_sha256=rt.sha(checkpoint),objective_sha256=frozen['objective_sha256'],
        proposal_source_sha256=rt.sha(__file__),time=time.time())
    rt.write(dest,payload);rt.write(state_path,state);return payload

def wait_scores(stage,ids,topologies,noises):
    while True:
        ready={(r['candidate'],r['stage'],r['topology'],r['noise']) for r in analysis.records()}
        expected={(c,stage,t,n) for c in ids for t in topologies for n in noises}
        if expected<=ready:return
        if (OUT/'analysis/observer_failure.json').exists():raise RuntimeError('analysis observer failed')
        time.sleep(10)

def nomination():
    path=OUT/'nomination.json'
    if path.exists():return rt.read(path)
    good=[r for r in training_data() if r['scorable']];p=rt.read(OUT/'plan.json')
    best=min(good,key=lambda r:r['J']);pareto=[]
    for r in good:
        x=np.array(r['components'])
        if not any(np.all(np.array(s['components'])<=x) and np.any(np.array(s['components'])<x) for s in good):pareto.append(r)
    second=min(pareto,key=lambda r:r['components'][1]);ids=list(dict.fromkeys([p['reference_id'],best['candidate'],second['candidate']]))
    payload=dict(ids=ids,best=best,second=second,all_training=good,rule='minimum J plus timing minimum on three-component Pareto set',time=time.time())
    rt.write(path,payload);return payload

def build_confirmation_graphs():
    from scripts import run_topic4_multiseed_response_curves as legacy
    legacy.OUT=OUT;manifest=rt.read(OUT/'confirmation/frozen_networks.json');p=rt.read(OUT/'plan.json')
    for seed in p['confirmation_seeds']['topology']:
        if str(seed) not in manifest['replication_networks']:
            manifest['replication_networks'][str(seed)]=legacy.build_network(seed)
            rt.write(OUT/'confirmation/frozen_networks.json',manifest)

def controller():
    p=rt.read(OUT/'plan.json')
    while not (OUT/'initial_simulation_complete.json').exists() or not (OUT/'analysis/objective_frozen.json').exists():
        s=rt.read(OUT/'status.json')
        if s.get('failures'):raise RuntimeError('initial physical queue has failed')
        if (OUT/'analysis/calibration_failure.json').exists():raise RuntimeError('G0 calibration failed')
        rt.write(OUT/'optimizer_status.json',dict(status='WAITING_FOR_INITIAL_AND_FROZEN_OBJECTIVE',updated_unix=time.time()));time.sleep(15)
    assert rt.read(OUT/'analysis/objective_frozen.json')['adaptive_dispatch_allowed'],'objective diagnostics failed'
    assert rt.read(OUT/'implementation_tests.json')['passed'],'implementation tests not passed'
    wait_scores('initial',p['initial_ids'],[2511],p['seeds'])
    if sum(r['scorable'] for r in training_data())<8:raise RuntimeError('fewer than eight scorable initial conditions')
    bad_batches=0
    for batch in range(1,5):
        proposal=propose(batch)
        rt.write(OUT/'optimizer_status.json',dict(status='ADAPTIVE_BATCH',batch=batch,ids=proposal['ids'],updated_unix=time.time()))
        run.run_queue('adaptive',proposal['ids']);wait_scores('adaptive',proposal['ids'],[2511],p['seeds'])
        by_id={r['candidate']:r for r in training_data()}
        rows=[by_id[cid] for cid in proposal['ids']]
        observed=[r['J'] for r in rows];rt.write(OUT/'proposals'/f'feedback_{batch:02d}.json',dict(predicted=proposal['predicted_J'],observed=observed,rows=rows))
        bad_batches=bad_batches+1 if sum(not r['scorable'] for r in rows)>=2 else 0
        if bad_batches>=2:raise RuntimeError('two adaptive batches with >= half conditions unscorable; stop for review')
    nominated=nomination();build_confirmation_graphs();topos=p['confirmation_seeds']['topology'];noises=p['confirmation_seeds']['dynamics']
    rt.write(OUT/'optimizer_status.json',dict(status='CONFIRMING_NOMINATED_CONDITIONS',ids=nominated['ids'],updated_unix=time.time()))
    run.run_queue('confirmation',nominated['ids'],topos,noises);wait_scores('confirmation',nominated['ids'],topos,noises)
    analysis.report()
    # G4 requires an actual native-field scientific review, not a file-decoding proxy.
    rt.write(OUT/'g3_ready_for_native_review.json',dict(status='G3_COMPLETE_NATIVE_REVIEW_REQUIRED',nomination=nominated['ids'],confirmation_units=4*len(nominated['ids']),conditional_G4_budget=40,time=time.time()))
    rt.write(OUT/'status.json',dict(status='G3_COMPLETE_NATIVE_REVIEW_REQUIRED',new_round_budget_expanded=False,updated_unix=time.time()))
    rt.write(OUT/'optimizer_status.json',dict(status='G3_COMPLETE_NATIVE_REVIEW_REQUIRED',updated_unix=time.time()))
    from scripts.continue_topic4_three_observable_response import evaluate,response
    evaluate()
    if (OUT/'analysis/g3_agent_native_review.json').exists():response()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controller','propose','build-graphs']);p.add_argument('--batch',type=int,default=1);a=p.parse_args()
    if a.action=='propose':print(json.dumps(propose(a.batch)));return
    if a.action=='build-graphs':build_confirmation_graphs();return
    with (OUT/'optimizer_controller.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:controller()
        except Exception as exc:
            rt.write(OUT/'optimizer_failure.json',dict(error=repr(exc),time=time.time()));raise

if __name__=='__main__':main()

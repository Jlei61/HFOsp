"""Bounded scientific/optimizer checks; no physical simulation or proposal use."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
from scripts import control_topic4_three_observable_bo as c
from src.topic4_three_observable_objective import signed_statistic,GROUPS

def main():
    checks={};obj=a.load_objective();ev,_,_=a.patient();rng=np.random.default_rng(512014)
    x=ev.fit[:24];labels=obj.labels(x);phi=obj.features(x,'cpu')
    for g,v in phi.items():
        psi=np.column_stack([v/np.sqrt(2)]+[.5*(labels==k)[:,None]*np.column_stack([np.ones(len(v)),v])/obj.proportions[k] for k in [0,1]])
        target=obj.targets[g]['psi'];s=signed_statistic(psi.sum(0),np.sum(psi*psi),len(psi),target)
        gram=psi@psi.T;explicit=(gram.sum()-np.trace(gram))/(len(psi)*(len(psi)-1))-2*np.mean(psi,0)@target+target@target
        np.testing.assert_allclose(s['D_off'],explicit,atol=1e-12)
        score=obj.score(x,labels)['groups'][g];np.testing.assert_allclose(score['D_off'],explicit,atol=1e-12)
        np.testing.assert_allclose(16/len(x)*(s['D16']-s['D_off']),s['B'],atol=1e-12)
    checks['off_diagonal_explicit_pairs_and_D16_identity']=True
    negative=signed_statistic(np.array([0.]),2.,2,np.array([0.]))
    assert negative['D_off']==-1.;checks['negative_scores_preserved']=True
    shifted=x+rng.uniform(-1e4,1e4,(len(x),1));raw=obj.raw_features(x);raw2=obj.raw_features(shifted)
    for key in raw:np.testing.assert_allclose(raw[key],raw2[key],atol=1e-10)
    checks['per_event_time_origin_invariance']=True
    assert obj.score(np.empty((0,15)))['status']=='INSUFFICIENT_EVENTS'
    assert obj.score(x[:15])['J'] is None
    assert obj.score(x,np.zeros(len(x),int))['J'] is not None
    checks['empty_small_and_absent_mode_semantics']=True
    raw=obj.raw_features(x);assert np.isfinite(np.concatenate(list(raw.values()),axis=1)).all()
    assert len(obj.pairs[0])==6 and len(obj.pairs[1])==55
    checks['missingness_and_pair_dimensions']=True
    # A biased finite-m statistic prefers a shifted Bernoulli proportion; the
    # off-diagonal expectation has its minimum at the target p, including p=.02.
    for p in [.02,.1,.2]:
        from scipy.stats import binom
        q=np.unique(np.r_[np.linspace(0,1,1001),p]);n=16
        loss=np.array([signed_statistic(np.array([k]),float(k),n,np.array([p]))['D_off'] for k in range(n+1)])
        risk=binom.pmf(np.arange(n+1)[None,:],n,q[:,None])@loss
        np.testing.assert_allclose(risk,(q-p)**2,atol=1e-13)
        assert abs(q[np.argmin(risk)]-p)<1e-12
    checks['mode_frequency_population_target']=True
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    lo,hi=c.run.bounds();z=rng.uniform(0,1,(12,6));scores=((z-.3)**2).sum(1)
    rows=[dict(candidate=f'TEST_ONLY_{i}',x=(lo+zi*(hi-lo)).tolist(),scorable=True,J=float(y),individual_J=[float(y-.03),float(y+.03)]) for i,(zi,y) in enumerate(zip(z,scores))]
    model,X,Y,V,good=c.fit_model(rows,2026091499,'cuda:0')
    acq=qLogNoisyExpectedImprovement(model,X_baseline=X,sampler=SobolQMCNormalSampler(torch.Size([32]),seed=25),prune_baseline=True)
    pool=torch.tensor(rng.uniform(0,1,(16,1,6)),dtype=torch.float64,device='cuda:0')
    with torch.no_grad():
        first=acq(pool);acq.set_X_pending(pool[int(first.argmax())]);second=acq(pool)
    assert first.shape==second.shape==(16,) and torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.allclose(first,second)
    checks['GPU_GP_qLogNEI_and_pending_batch_execution']=True
    a.rt.write(a.OUT/'implementation_tests.json',dict(passed=all(checks.values()),checks=checks,physical_simulations=0,
        test_source_sha256=a.rt.sha(__file__),objective_sha256=a.rt.sha(a.A/'training_objective.pkl'),time=time.time()))
    print(checks)

if __name__=='__main__':main()

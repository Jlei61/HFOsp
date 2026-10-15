"""Conditional, budgeted G4 after measured G3 improvement and real native review.

`evaluate` prepares a reviewable decision; `run` requires a signed agent review
artifact containing inspected media and scientific reasoning. It does not ask
for a second user authorization or replace native review with file checks.
"""
from pathlib import Path
import argparse,fcntl,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import control_topic4_three_observable_bo as ctl
run=ctl.run;rt=run.rt;OUT=run.OUT;analysis=ctl.analysis

def evaluate():
    plan=rt.read(OUT/'plan.json');nom=rt.read(OUT/'nomination.json');rows=analysis.records()
    lookup={(r['candidate'],r['topology'],r['noise']):r for r in rows if r['stage']=='confirmation'}
    ev,names,_=analysis.patient();from scripts.analyze_topic4_core_connectivity_search import measures
    target=measures(ev.fit[ev.fit_labels==0],ev.fit[ev.fit_labels==0],np.array(names))['SCL_minus_ICL_lag_median_ms']
    results=[]
    for cid in nom['ids']:
        if cid==plan['reference_id']:continue
        pairs=[];readable=True
        for topo in plan['confirmation_seeds']['topology']:
            for noise in plan['confirmation_seeds']['dynamics']:
                a=lookup.get((plan['reference_id'],topo,noise));b=lookup.get((cid,topo,noise))
                if a is None or b is None or a['J'] is None or b['J'] is None:readable=False;continue
                la=a['raw']['TB'].get('SCL_minus_ICL_lag_median_ms');lb=b['raw']['TB'].get('SCL_minus_ICL_lag_median_ms')
                if la is None or lb is None:readable=False;continue
                pairs.append(dict(topology=topo,noise=noise,delta_J=b['J']-a['J'],delta_components=(np.array(b['components'])-a['components']).tolist(),delta_TB_lag_error=abs(lb-target)-abs(la-target),J=b['J'],TB_N=b['mode_counts']['TB']))
        criteria={}
        if readable and len(pairs)==4:
            criteria=dict(J_better_at_least_3=sum(p['delta_J']<0 for p in pairs)>=3,
                timing_mean_better=np.mean([p['delta_components'][1] for p in pairs])<0,
                TB_lag_error_better_at_least_3=sum(p['delta_TB_lag_error']<0 for p in pairs)>=3)
            for topo in plan['confirmation_seeds']['topology']:
                delta=np.mean([p['delta_components'] for p in pairs if p['topology']==topo],axis=0)
                criteria[f'topology_{topo}_rank_and_participation_not_worse']=bool(delta[0]<=0 and delta[2]<=0)
        results.append(dict(candidate=cid,estimable=readable and len(pairs)==4,numerical_trigger=bool(criteria and all(criteria.values())),criteria={k:bool(v) for k,v in criteria.items()},pairs=pairs))
    payload=dict(patient_TB_lag_median_ms=target,candidates=results,native_review_required=True,
        interpretation='Budget allocation rule; not a significance test or full recovery acceptance',time=time.time())
    rt.write(OUT/'g4_numerical_decision.json',payload);return payload

def proposals(center):
    path=OUT/'proposals/response.json'
    if path.exists():return rt.read(path)
    x0=run.vector(center);lo,hi=run.bounds();steps=[.375,.375,.375,.375,.0625,5.];ids=[];meta=[]
    for axis,h in enumerate(steps):
        delta=[-h,h]
        if x0[axis]-h<lo[axis]:delta=[h,2*h]
        if x0[axis]+h>hi[axis]:delta=[-h,-2*h]
        for j,d in enumerate(delta):
            x=x0.copy();x[axis]+=d;cid=f'g4_axis{axis}_probe{j}'
            c=run.candidate_at(center,x,cid,'response_single_axis',axis);c['stage']='response';c['comparison']=center['id'];run.validate_candidate(c);rt.write(OUT/'candidates'/f'{cid}.json',c)
            ids.append(cid);meta.append(dict(candidate=cid,axis=axis,delta=d,one_sided=delta!=[-h,h]))
    payload=dict(center=center['id'],ids=ids,meta=meta,budget=24,time=time.time());rt.write(path,payload);return payload

def response():
    assert (OUT/'g3_ready_for_native_review.json').exists()
    decision=evaluate();review=rt.read(OUT/'analysis/g3_agent_native_review.json')
    assert review['review_type']=='ACTUAL_AGENT_VISUAL_AND_SCIENTIFIC_REVIEW'
    assert review['inspected_files'] and review['scientific_reasoning']
    for item in review['inspected_files']:assert rt.sha(item['path'])==item['sha256']
    eligible=[r for r in decision['candidates'] if r['numerical_trigger'] and review['candidates'].get(r['candidate'],{}).get('no_new_contradictory_propagation') is True]
    if not eligible:
        rt.write(OUT/'optimization_complete.json',dict(status='ROUND_COMPLETE_NO_G4_TRIGGER',decision=decision,review=review,time=time.time()));return
    selected=min(eligible,key=lambda r:np.mean([p['J'] for p in r['pairs']]))['candidate'];center=rt.read(OUT/'candidates'/f'{selected}.json');p=rt.read(OUT/'plan.json')
    proposal=proposals(center);run.run_queue('response',proposal['ids']);ctl.wait_scores('response',proposal['ids'],[2511],p['seeds'])
    rows=analysis.records();lookup={(r['candidate'],r['topology'],r['noise']):r for r in rows}
    candidates=[]
    for axis in range(6):
        strength=0.;sources=[]
        for m in proposal['meta']:
            if m['axis']!=axis:continue
            diff=[]
            for noise in p['seeds']:
                ref=lookup[(selected,2511,noise)];q=lookup[(m['candidate'],2511,noise)]
                if q['components'] is None:continue
                diff.append(np.array(q['components'])-ref['components'])
            if len(diff)!=2:continue
            d=np.array(diff);same=(d[0]*d[1])>0;values=np.where(same,np.abs(d.mean(0)),0)
            strength=max(strength,float(values.max()));sources.append(dict(candidate=m['candidate'],deltas=d.tolist()))
        if strength>0:candidates.append(dict(axis=axis,strength=strength,sources=sources))
    axes=sorted(candidates,key=lambda r:r['strength'],reverse=True)[:2]
    ids=[m['candidate'] for m in proposal['meta'] if m['axis'] in [a['axis'] for a in axes]]
    rt.write(OUT/'proposals/response_confirmation.json',dict(ids=ids,axes=axes,rule='largest mean absolute standardized component response with same sign in both training noises',new_runs=len(ids)*4))
    if ids:
        run.run_queue('response_confirmation',ids,p['confirmation_seeds']['topology'],p['confirmation_seeds']['dynamics'])
        ctl.wait_scores('response_confirmation',ids,p['confirmation_seeds']['topology'],p['confirmation_seeds']['dynamics'])
    analysis.report()
    rt.write(OUT/'optimization_complete.json',dict(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW',response_center=selected,new_response_runs=24+len(ids)*4,time=time.time()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['evaluate','run']);args=parser.parse_args()
    if args.action=='evaluate':print(evaluate());return
    with (OUT/'response_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:response()
        except run.DispatchWindowClosed:
            rt.write(OUT/'status.json',dict(status='WINDOW_END_CONDITIONAL_RESPONSE_PARTIAL',time=time.time()))
            analysis.report()

if __name__=='__main__':main()

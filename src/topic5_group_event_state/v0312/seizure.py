"""Case-wise frozen association on published information, not seizure forecasting."""
import numpy as np
import torch
from . import data as D
from .engine import infer_asof,query_noise,sample_posterior
from .train import tensor_hash


def controls_mask(times,seizures,pre_hours=2.,post_hours=1.):
    forbidden=[(float(r['onset_epoch'])-pre_hours*3600,float(r['offset_epoch'])+post_hours*3600) for r in seizures]
    # Query is a point; explicitly include both interval boundaries.
    return ~np.array([any(a<=t<=b for a,b in forbidden) for t in times],bool)


def available_recent(prep,query,minutes=30,role='descriptive'):
    pk=prep.payload['packets'];time=pk['end'][query]
    legal=D.input_mask(prep.split,role)&(pk['release']<=time)&(pk['end']<=time)&(pk['start']>=time-minutes*60)
    legal&=np.arange(prep.n_packets)>=prep.split['episode_start'][query]
    exposure=float(pk['exposure'][legal].sum());count=int((pk['event_hi'][legal]-pk['event_lo'][legal]).sum())
    return dict(exposure_seconds=exposure,count=count,rate_per_hour=count/exposure*3600 if exposure>0 else None,
                coverage=exposure/(minutes*60),packet_digest=D.digest(np.flatnonzero(legal)))


@torch.no_grad()
def association(model,prep,cfg,quick=False,case_override=None,include_placebo=True,reference=None):
    pk=prep.payload['packets'];qs=np.flatnonzero(prep.split['valid_packet'])[::30 if not quick else 120]
    qs=qs[pk['end'][qs]<=prep.split['support_end']]
    cases=prep.split['seizures'] if case_override is None else case_override
    if reference is None:
        from .frozen import fit_reference
        reference=fit_reference(model,prep,cfg)
    safe=controls_mask(pk['end'][qs],cases+prep.split['seizures'])
    rows=[]
    for a in range(0,len(qs),cfg.eval_chunk):
        state=infer_asof(model,prep,qs[a:a+cfg.eval_chunk],'descriptive',cfg.history_hours,producer_hash=tensor_hash(model.state_dict()))
        phase=prep.clock_end[torch.as_tensor(state.query_packet,device=prep.device)]
        noise=query_noise(state,cfg.eval_seed,1,cfg.eval_paths,prep.device,state.m.dtype)[0]
        z=sample_posterior(state.m,state.P,cfg.eval_paths,noise=noise)
        out=model.readout(z,phase.unsqueeze(0).expand(cfg.eval_paths,-1,-1))
        log_rate=torch.logsumexp(out['log_rate'].clamp(-20,15),0)-np.log(cfg.eval_paths)
        # Recipient clock and MC draws are identical: isolate the state-dependent
        # readout contribution from the clock already present in g.
        rz=sample_posterior(reference['m0'].expand_as(state.m),reference['P0'].expand_as(state.P),cfg.eval_paths,noise=noise)
        rout=model.readout(rz,phase.unsqueeze(0).expand(cfg.eval_paths,-1,-1))
        reset_log_rate=torch.logsumexp(rout['log_rate'].clamp(-20,15),0)-np.log(cfg.eval_paths)
        for j,q in enumerate(state.query_packet):
            recent=available_recent(prep,int(q));wide=available_recent(prep,int(q),120)
            rows.append(dict(query=int(q),time=float(state.query_time[j]),log_rate=float(log_rate[j]),
                reset_log_rate=float(reset_log_rate[j]),state_log_rate_increment=float(log_rate[j]-reset_log_rate[j]),
                variance=float(state.P[j].diagonal().mean()),has_observation=bool(np.isfinite(state.release_time[j])),
                age=float(state.information_age_minutes[j]),clock=int(D.clock_stratum([state.query_time[j]])[0]),
                recent=recent,coverage_2h=wide['coverage'],producer_hash=state.producer_hash,
                input_digest=state.input_digest[j],query_metadata=state.subset([j]).metadata()))
    summaries=[]
    for seizure,window,lo,hi in [(s,w,a,b) for s in cases for w,a,b in (('primary',-7200,-1800),('secondary',-1800,-300))]:
        onset=float(seizure['onset_epoch']);pre=[i for i,r in enumerate(rows) if onset+lo<=r['time']<onset+hi]
        pairs=[];increment=[];used=set();used_inc=set()
        for i in pre:
            r=rows[i]
            if not r['has_observation']:continue
            pool=[j for j,s in enumerate(rows) if safe[j] and j not in used and s['has_observation'] and s['clock']==r['clock']
                  and abs(s['age']-r['age'])<=10 and abs(s['coverage_2h']-r['coverage_2h'])<=.2
                  and abs(s['recent']['coverage']-r['recent']['coverage'])<=.2]
            # Match recorded support and clock first; recent rate is a separate estimand.
            if pool:
                j=min(pool,key=lambda j:abs(rows[j]['time']-r['time']));used.add(j)
                pairs.append(dict(case_query=r['query'],control_query=rows[j]['query'],
                    log_rate_difference=r['log_rate']-rows[j]['log_rate'],
                    state_log_rate_increment_difference=r['state_log_rate_increment']-rows[j]['state_log_rate_increment'],
                    variance_difference=r['variance']-rows[j]['variance']))
            rr=r['recent']['rate_per_hour']
            if rr is not None:
                inc=[j for j,s in enumerate(rows) if safe[j] and j not in used_inc and s['has_observation'] and s['clock']==r['clock']
                    and abs(s['age']-r['age'])<=10 and abs(s['coverage_2h']-r['coverage_2h'])<=.2
                    and s['recent']['rate_per_hour'] is not None and abs(s['recent']['coverage']-r['recent']['coverage'])<=.2
                    and abs(np.log1p(s['recent']['rate_per_hour'])-np.log1p(rr))<=.5]
                if inc:
                    j=min(inc,key=lambda j:abs(rows[j]['time']-r['time']));used_inc.add(j)
                    increment.append(dict(case_query=r['query'],control_query=rows[j]['query'],
                        log_rate_difference=r['log_rate']-rows[j]['log_rate'],
                        state_log_rate_increment_difference=r['state_log_rate_increment']-rows[j]['state_log_rate_increment'],
                        variance_difference=r['variance']-rows[j]['variance']))
        summaries.append(dict(onset_epoch=onset,seizure_id=seizure.get('seizure_id'),window=window,relative_seconds=[lo,hi],pre_queries=len(pre),pairs=pairs,incremental_pairs=increment,
            status='DESCRIPTIVE' if pairs else 'NOT_ESTIMABLE',
            after_producer_fit=onset>prep.split['fit_end']))
    result=dict(status='COMPLETE',endpoint='S-A case-wise association',rows=rows,cases=summaries,
        limitations='Raw log rate includes clock. State increment is relative to the FIT state distribution at the identical recipient clock, not a causal effect. No seizure-driven producer selection or pooled query p-value. Reused controls are not independent seizures.',
        S_B=dict(status='NOT_REQUESTED',reason='Clinical-onset spatial consumer is invoked separately after the frozen contact readout.'),
        S_C=dict(status='NOT_ESTIMABLE',reason='No complete prospective seizure-risk denominator registered for this window.'))

    if include_placebo and case_override is None:
        pool=[r for r in rows if r['has_observation'] and controls_mask(np.array([r['time']]),cases,pre_hours=4,post_hours=2)[0]]
        rng=np.random.default_rng(cfg.eval_seed+71);fake=[]
        for i in rng.permutation(len(pool)):
            t=pool[i]['time']
            if t-prep.split['support_start']<7200 or any(abs(t-r['onset_epoch'])<14400 for r in fake):continue
            fake.append(dict(onset_epoch=t,offset_epoch=t+10,seizure_id=f'placebo{len(fake)}'))
            if len(fake)>=min(3,len(cases)):break
        result['placebo']=association(model,prep,cfg,quick=quick,case_override=fake,include_placebo=False,reference=reference) if fake else dict(status='NOT_ESTIMABLE')
        result['placebo_note']='Fixed-seed pseudo-onsets outside real seizure neighbourhoods. Descriptive falsification, not a calibrated null distribution.'
    return result

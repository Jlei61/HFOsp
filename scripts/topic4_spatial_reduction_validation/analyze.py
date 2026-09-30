"""Direct SNN/reduction comparison with unchanged contact and event observers."""
from common import *
import csv,itertools,warnings
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
NAMES=read(V10/'native/a/observation_contract.json')['contact_names']

def metric_distance(a,b):
    v=metrics(a,b,NAMES)
    return np.array([v[0],v[1],v[3]],float)

def dynamics(six,sizes):
    r=six/sizes/.002;smooth=gaussian_filter1d(r,2.5,axis=0);out={}
    for j,name in enumerate(['AE','BE','SE','AI','BI','SI']):
        v=smooth[1000:,j];peaks,_=find_peaks(v,prominence=10.,distance=40,height=20.)
        intervals=np.diff(peaks)*2
        out[name]=dict(mean_hz=float(r[1000:,j].mean()),peak95_hz=float(np.quantile(v,.95)),
            low_rate_fraction=float(np.mean(v<5.)),q05_hz=float(np.quantile(v,.05)),
            n_rate_peaks=len(peaks),median_peak_interval_ms=float(np.median(intervals)) if len(intervals) else None,
            peak_interval_CV=float(intervals.std(ddof=1)/intervals.mean()) if len(intervals)>1 else None)
    return out

def coverage(q):
    return dict(contacts=sum(x is not None and np.isfinite(x) for x in q['mean_rank']),
        pairs={shaft:sum(v['shaft']==shaft and v['order_probability'] is not None for v in q['pairs'].values()) for shaft in ('SCL','ICL')})

def summarize(name,z,envkey,sizes):
    ob,ids,mu,q=observations(z[envkey]);out=dict(name=name,N=len(ids),summary=q,coverage=coverage(q),
        valid_event_ids=ids.tolist(),centroid_ms=mu[ids],dynamics=dynamics(z['six_counts'],sizes))
    write(OUT/'observations'/f'{name}.json',ob)
    write(OUT/'summaries'/f'{name}.json',out)
    return out

def main():
    sizes=np.bincount(np.load(V10/'native/a/trajectory.npz')['region'],minlength=6)
    native={};oracle={};rate={};arrays={}
    for seed in SEEDS:
        z=np.load(OUT/'native'/str(seed)/'trajectory.npz');arrays[seed]=z
        native[seed]=summarize(f'native_{seed}',z,'contact_envelope',sizes)
        for grid in (10,20,40):oracle[grid,seed]=summarize(f'projection{grid}_{seed}',z,f'contact_envelope_{grid}',sizes)
    pair=[]
    for a,b in itertools.combinations(SEEDS,2):
        pair.append(dict(a=a,b=b,errors=metric_distance(native[a]['summary'],native[b]['summary'])))
    spread=np.max([r['errors'] for r in pair],axis=0)
    rows=[]
    for grid in (10,20,40):
        for seed in SEEDS:
            q=oracle[grid,seed];err=metric_distance(q['summary'],native[seed]['summary'])
            rows.append(dict(kind='observed_spikes_projected',grid=grid,seed=seed,N=q['N'],rank_error=err[0],
                order_error=err[1],participation_error=err[2],interpretation='spatial information loss only'))
    for grid in (10,20):
        path=OUT/'rate'/f'grid{grid}_seed848101.npz'
        if not path.exists():continue
        z=np.load(path);assert np.array_equal(z['nu_core'],arrays[848101]['nu_core'])
        q=summarize(f'rate{grid}_848101',z,'contact_envelope',sizes);rate[grid]=q
        err=np.stack([metric_distance(q['summary'],native[seed]['summary']) for seed in SEEDS])
        cov=q['coverage'];complete=cov['contacts']==15 and cov['pairs']=={'SCL':6,'ICL':55}
        outside=np.all(err>spread[None,:],axis=0)
        diag={}
        for pop in ('AE','BE','SE'):
            diag[pop]={}
            for metric in ('mean_hz','low_rate_fraction','median_peak_interval_ms','peak_interval_CV'):
                values=np.array([native[s]['dynamics'][pop][metric] if native[s]['dynamics'][pop][metric] is not None else np.nan for s in SEEDS])
                value=q['dynamics'][pop][metric]
                diag[pop][metric]=dict(native_values=values,rate=value)
        q.update(errors_to_native=err,native_pair_max=spread,all_references_exceed_native_range=outside,
            first_screen='FAIL_PROPAGATION' if outside.any() else ('INSUFFICIENT_EVENTS_OR_COVERAGE' if not complete or q['N']<16 else 'NO_CLEAR_PROPAGATION_FAILURE_NOT_YET_VALIDATED'),dynamic_comparison=diag)
        if q['N']==0:q['first_screen']='FAIL_NO_VALID_EVENTS'
        write(OUT/'summaries'/f'rate{grid}_848101.json',q)
        rows.append(dict(kind='closed_loop_rate',grid=grid,seed=848101,N=q['N'],rank_error=err[0,0],
            order_error=err[0,1],participation_error=err[0,2],interpretation=q['first_screen']))
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(safe(rows))
    results=dict(native_pairs=pair,native_pair_max=spread,rows=rows,
        native={s:dict(N=q['N'],dynamics=q['dynamics']) for s,q in native.items()},
        rate={g:dict(N=q['N'],dynamics=q['dynamics'],first_screen=q['first_screen'],errors_to_native=q['errors_to_native']) for g,q in rate.items()},
        bifurcation_allowed=False,reason='correspondence gate; no native/reduced matched onset and perturbation validation',
        unit='three noise seeds on one frozen topology; event samples are within-run observations',
        input_parity='float32 per-step core afferent rates exactly equal in seed848101')
    write(OUT/'comparison.json',results)
    print(json.dumps(safe(results),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

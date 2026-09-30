"""Independently aggregate fresh counts and evaluate real state-space filters."""
from common import *
import hashlib

DEST = OUT/'response_oscillatory_capacity'


def main():
    c = read(OUT/'response_oscillatory_capacity_contract.json'); settings=c['fresh_assay']
    folder=DEST/'fresh'; metadata=read(folder/'metadata.json')
    assert metadata['status']=='COMPLETE'
    locked=read(DEST/'fit_result.json'); result=read(folder/'result.json')
    digest=hashlib.sha256((DEST/'fit_result.json').read_bytes()).hexdigest()
    assert digest==metadata['fit_sha256']==result['fit_sha256']
    z=np.load(folder/'checkpoint.npz'); assert z['done'].all()
    obs=np.load(folder/'observations.npy', mmap_mode='r'); pars=z['parameters']
    lookup={}; maxmean=0.; maxsem=0.; maxpred=0.
    for i, m in enumerate(metadata['meta']):
        p=pars[i]; amp=p[4]*(1 if m['channel']==0 else p[2 if m['channel']==1 else 3])
        data=obs[i, :, :2]*1000/(settings['duration_ms']*amp)
        avg=data.mean(0); sem=np.sqrt(data.var(0).sum()/settings['replicates'])
        old=result['measurements'][i]
        maxmean=max(maxmean, float(np.max(abs(avg-old['measured']))))
        maxsem=max(maxsem, abs(float(sem)-old['sem']))
        lookup[(m['point'], m['channel'], m['frequency_hz'])]=(complex(*avg), float(sem))
    assert maxmean<1e-10 and maxsem<1e-10
    s=model(); counts={}; errors={}; by_point={}; noise=[]
    for r in result['rows']:
        point=locked['rows'][r['point']]; ch=r['channel']; f=r['frequency_hz']
        w=2j*np.pi*f/1000; target, sem=lookup[(r['point'], ch, f)]
        dc, dcsem=lookup[(r['point'], ch, 0.)]
        assert point['eligible']==r['eligible']
        for method in point['variants']:
            p=method['full_fit']; states=[]
            # First-order h'=(q-h)/tau, output q-h.
            for tau in p['real_times_ms']:
                h=(1/tau)/(w+1/tau)
                states.append(1-h)
            if p['kind']=='real3_pair':
                a,b=p['decay_per_ms'],p['oscillation_per_ms']
                A=np.array([[-a,b],[-b,-a]])
                state=np.linalg.solve(w*np.eye(2)-A, np.array([a,b]))
                states.extend([1-state[0],state[1]])
            prediction=(point['static_DC']+point['gain_factor']*np.dot(states,p['coefficients']))/(1+w*s.tau[ch-1]/2)
            old=complex(*r['methods'][p['kind']]['predicted'])
            maxpred=max(maxpred,float(abs(prediction-old)))
            err=abs(prediction-target)/max(abs(dc.real),1e-12)
            assert np.isclose(err,r['methods'][p['kind']]['error'],rtol=1e-8,atol=1e-8)
            failed=err>(.1 if f<=25 else .15)
            assert bool(failed)==r['methods'][p['kind']]['failed']
            if r['eligible']:
                band='inside' if f<=80 else 'above'
                key=f'{p["kind"]}_{band}'
                pair=counts.setdefault(key,[0,0]); pair[0]+=int(failed); pair[1]+=1
                errors.setdefault(key,[]).append(float(err))
                by_point.setdefault(f'{r["point"]}_{p["kind"]}_{band}',[]).append(bool(failed))
        if r['eligible']:noise.append(dict(point=r['point'],frequency_hz=f,
             AC_SEM_over_DC=float(sem/max(abs(dc.real),1e-12)), DC_SNR=float(abs(dc.real)/max(dcsem,1e-12))))
    assert maxpred<1e-8
    summary={name:dict(failures=pair[0],n=pair[1],median_error=float(np.median(errors[name]))) for name,pair in counts.items()}
    no_failed_points={}
    for key, failures in by_point.items():
        label='_'.join(key.split('_')[1:]); tally=no_failed_points.setdefault(label,[0,0]);tally[0]+=not any(failures);tally[1]+=1
    write(folder/'independent_audit.json',dict(status='RAW_COUNTS_AND_STATE_SPACE_PREDICTIONS_PASS',
         conditions=len(pars),prediction_lock_unchanged=True,max_count_mean_error=maxmean,
         max_SEM_error=maxsem,max_state_space_prediction_error=maxpred,summary=summary,
         workpoints_with_all_frequencies_passing=no_failed_points,precision=noise,
         scope='Independent implementation/provenance audit, not a response-accuracy or replacement-network pass.'))
    log('OSCILLATORY INDEPENDENT AUDIT',summary,no_failed_points)


if __name__=='__main__':main()

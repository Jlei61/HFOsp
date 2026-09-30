"""Read-only independent count, separation and score reconstruction."""
from common import OUT,np,read,write
from pathlib import Path
import hashlib

DEST=OUT/'refractory_rate_response'

def audit():
    rows=read(DEST/'profiles.json')['rows'];train={r['id'] for r in rows if r['split']=='train'};valid={r['id'] for r in rows if r['split']=='validation'}
    assert len(train)==256 and len(valid)==64 and not train&valid
    assert read(DEST/'acquisition.json')['status']=='COMPLETE'
    for pop in 'EI':
        ids=read(DEST/f'training_arrays/{pop}_provenance.json')['profiles']
        fresh=[int(x.rsplit('/',1)[1]) for x in ids if x.startswith('refractory_rate_response/')]
        assert set(fresh).issubset(train)
        assert not read(DEST/f'training_arrays/{pop}_provenance.json')['validation_targets_opened']
    for row in rows:
        z=np.load(DEST/f'local_data/profile{row["id"]:03d}.npz');f=z['spike_counts'].astype(np.int64);a=z['available_counts'].astype(np.int64)
        burn=int(z['burn_steps']);nref=int(z['nref_steps']);R=int(z['replicates']);s=np.r_[0,np.cumsum(f)];k=np.arange(len(f))
        assert np.array_equal(a,R-s[k]+s[np.maximum(0,k-nref+1)])
        assert np.all((f>=0)&(f<=a)) and np.all(a<=R)
        counts=z['phase_counts'];assert counts.sum(dtype=np.uint64)==f[burn:].sum()
        expected=counts.astype(float).mean(0)/z['exposure_ms']*1000
        assert np.max(abs(expected-z['rate_hz']))<1e-9
    weights=read(DEST/'fit/locked_weights.json')
    for name,h in weights['files'].items():assert hashlib.sha256((DEST/'fit'/name).read_bytes()).hexdigest()==h
    for name,h in weights['source_hashes'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    locked=read(DEST/'validation/predictions_locked.json');result=read(DEST/'validation/result.json');maxdifference=0.
    for row in result['rows']:
        p=DEST/f'validation/{row["label"]}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==locked['prediction_hashes'][p.name]
        pred=np.load(p);w=pred['exposure_ms']
        if row['kind'] in ['fresh','reused']:
            root=DEST if row['kind']=='fresh' else OUT/'nonlinear_rate_response'
            ref=np.load(root/f'local_data/profile{row["index"]:03d}.npz')['rate_hz']
        else:ref=np.load(OUT/row['kind']/'response.npz')['measured_hz'][row['index']]
        norm=max(np.sqrt(sum(ref*ref)),np.sqrt(len(ref)))
        L2=np.sqrt(sum((pred['predicted_hz']-ref)**2))/norm
        true_mean=sum(ref*w)/sum(w);mean=abs(sum(pred['predicted_hz']*w)/sum(w)-true_mean)/max(true_mean,1.)
        numerical=np.sqrt(sum((pred['predicted_hz']-pred['fine_step_prediction_hz'])**2))/norm
        maxdifference=max(maxdifference,abs(L2-row['waveform_L2']),abs(mean-row['mean_error']),abs(numerical-row['time_step_error']))
        assert bool(L2<=.15 and mean<=.1)==row['passed']
    assert maxdifference<1e-12
    for kind,count in result['groups'].items():
        group=[r for r in result['rows'] if r['kind']==kind]
        assert len(group)==count['count'] and sum(r['passed'] for r in group)==count['passed']
    answer=dict(status='PASS',additional_profiles_count_audited=320,predictions_scored=152,score_max_absolute_difference=maxdifference,
        training_validation_disjoint=True,weights_and_source_unchanged_since_lock=True,all_integer_refractory_identities=True,
        scientific_status=result['status'],model_promoted=False,scope='Audit pass verifies a failed candidate result; it is not model acceptance.')
    write(DEST/'independent_audit.json',answer);print(answer)

if __name__=='__main__':audit()

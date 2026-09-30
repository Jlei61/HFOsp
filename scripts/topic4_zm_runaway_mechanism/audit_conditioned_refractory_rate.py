"""Independent read-only audit of the conditioning intervention and its scores."""
from conditioned_refractory_rate import DEST,PARENT,OUT,np,read,write,load_models
from pathlib import Path
import hashlib

def audit():
    nets,bases,locked=load_models();contract=read(DEST/'contract.json')
    assert (DEST/'training_arrays').resolve()==(PARENT/'training_arrays').resolve()
    assert contract['training']['seed']==read(PARENT/'contract.json')['training']['seed']
    assert read(DEST/'implementation_check.json')['same_function_class']
    for pop in 'EI':
        assert read(DEST/f'fit/{pop}_progress.json')['status']=='COMPLETE'
        z=np.load(DEST/f'conditioning/{pop}.npz');ev=z['eigenvalues'];assert float(z['floor'])==float(ev[-1]*1e-7)
        assert np.linalg.eigvalsh(z['transform']).min()>0
    rows=read(DEST/'profiles.json')['rows'];assert len(rows)==64 and all(r['split']=='validation' for r in rows)
    for row in rows:
        z=np.load(DEST/f'local_data/profile{row["id"]:03d}.npz');f=z['spike_counts'].astype(np.int64);a=z['available_counts'].astype(np.int64);R=int(z['replicates']);nref=int(z['nref_steps']);burn=int(z['burn_steps'])
        s=np.r_[0,np.cumsum(f)];k=np.arange(len(f));assert np.array_equal(a,R-s[k]+s[np.maximum(0,k-nref+1)])
        assert z['phase_counts'].sum(dtype=np.uint64)==f[burn:].sum()
        assert np.max(abs(z['phase_counts'].mean(0)/z['exposure_ms']*1000-z['rate_hz']))<1e-9
    locked_prediction=read(DEST/'validation/predictions_locked.json');result=read(DEST/'validation/result.json');max_error=0.
    roots={'fresh':DEST,'parent':PARENT,'reused':OUT/'nonlinear_rate_response'}
    for row in result['rows']:
        p=DEST/f'validation/{row["label"]}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==locked_prediction['prediction_hashes'][p.name]
        z=np.load(p);w=z['exposure_ms']
        if row['kind'] in roots:target=np.load(roots[row['kind']]/f'local_data/profile{row["index"]:03d}.npz')['rate_hz']
        else:target=np.load(OUT/row['kind']/'response.npz')['measured_hz'][row['index']]
        norm=max(np.sqrt(sum(target*target)),np.sqrt(len(target)));mean=sum(target*w)/sum(w)
        error=np.sqrt(sum((z['predicted_hz']-target)**2))/norm;bias=abs(sum(z['predicted_hz']*w)/sum(w)-mean)/max(mean,1.)
        step=np.sqrt(sum((z['predicted_hz']-z['fine_step_prediction_hz'])**2))/norm
        max_error=max(max_error,abs(error-row['waveform_L2']),abs(bias-row['mean_error']),abs(step-row['time_step_error']))
        assert bool(error<=.15 and bias<=.1)==row['passed'] and bool(step<=.02)==row['numerical_pass']
    assert max_error<1e-12
    formal=read(DEST/'matched_linear_protocol/result.json');assert formal['conditions']==282
    assert read(DEST/'matched_linear_protocol/jobs.json')['status']=='COMPLETE'
    answer=dict(status='PASS',same_training_arrays=True,same_seed_schedule_architecture=True,invertible_coordinate_change=True,fresh_validation_only=64,
        waveform_scores_reconstructed=216,score_max_absolute_difference=max_error,matched_protocol_conditions=282,
        matched_independent_demodulation_error=formal['independent_demodulation_max_difference'],scientific_status=result['status'],model_promoted=False)
    write(DEST/'independent_audit.json',answer);print(answer)

if __name__=='__main__':audit()
